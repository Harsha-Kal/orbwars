"""
Orbit Wars - ML-Compatible Advanced Agent
==========================================
This agent is designed to work with learn_strategy.py.
It uses markers to allow the ML script to automatically update parameters.
"""

import math
import os
import json
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# <<LEARNED_PARAMS_START>>
LEARNED_PARAMS = {
    "prod_weight": 8.195,
    "ship_cost_weight": 1.045,
    "dist_weight": 0.375,
    "early_end_turn": 89,
    "late_start_turn": 319,
    "defend_threshold": 17,
    "min_hold_base": 0,
    "min_hold_threat": 7,
    "attack_buffer_ratio": 0.468,
    "min_attack_avail": 3,
    "min_expand_avail": 6,
    "consolidate_avail": 102,
    "consolidate_frac": 0.666,
    "aggressive_ship_ratio": 3.222,
    "defensive_ship_ratio": 0.604,
    "prod_target_early": 1.292,
}
# <<LEARNED_PARAMS_END>>

SUN_X, SUN_Y = 50.0, 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0


# ── geometry ──────────────────────────────────────────────────────────────────

def dist(ax, ay, bx, by): return math.hypot(bx - ax, by - ay)
def dp(a, b): return math.hypot(b.x - a.x, b.y - a.y)
def fleet_speed(n):
    n = max(1, int(n))
    if n == 1: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5
def travel_turns(d_val, ships): return d_val / fleet_speed(ships)

def hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx*dx + dy*dy
    if a < 1e-9: return math.hypot(fx, fy) < SUN_R
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - SUN_R*SUN_R
    disc = b*b - 4*a*c
    if disc < 0: return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)

def safe_angle(sx, sy, tx, ty):
    direct = math.atan2(ty - sy, tx - sx)
    if not hits_sun(sx, sy, tx, ty): return direct
    dist_st = dist(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0]:
        ang = direct + off
        ex = sx + dist_st * math.cos(ang)
        ey = sy + dist_st * math.sin(ang)
        if not hits_sun(sx, sy, ex, ey): return ang
    return direct

def predict_pos(p, ang_vel, turns):
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= 50.0: return p.x, p.y
    cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
    new_ang = cur_ang + ang_vel * turns
    return SUN_X + orb_r * math.cos(new_ang), SUN_Y + orb_r * math.sin(new_ang)

def aim_at(src, tgt, ang_vel, ships):
    tx, ty = tgt.x, tgt.y
    for _ in range(8):
        d_est = dist(src.x, src.y, tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    return safe_angle(src.x, src.y, tx, ty)

def fleet_impact(planet, all_fleets, me):
    fi = ei = 0
    for fl in all_fleets:
        ea = math.atan2(planet.y - fl.y, planet.x - fl.x)
        if abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea))) < 0.35:
            if fl.owner == me: fi += fl.ships
            else: ei += fl.ships
    return fi, ei


# ── main agent ────────────────────────────────────────────────────────────────

def agent(obs):
    g = lambda key, default: (obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default))
    me, ang_vel = int(g("player", 0)), float(g("angular_velocity", 0.033))
    step = int(g("step", 0))
    planets = [Planet(*p) for p in g("planets", [])]
    fleets = [Fleet(*f) for f in g("fleets", [])]
    
    my_p = [p for p in planets if p.owner == me]
    neutral_p = [p for p in planets if p.owner == -1]
    enemy_p = [p for p in planets if p.owner not in (-1, me)]

    # Dynamic phase logic
    is_early = step < LEARNED_PARAMS["early_end_turn"]
    is_late  = step > LEARNED_PARAMS["late_start_turn"]

    committed, moves = {}, []
    
    def get_min_hold(p):
        _, ei = fleet_impact(p, fleets, me)
        if ei > 0: return LEARNED_PARAMS["min_hold_threat"]
        return LEARNED_PARAMS["min_hold_base"]

    def avail(p): 
        return max(0, p.ships - committed.get(p.id, 0) - get_min_hold(p))

    def send_to(src, tgt, n):
        n = int(n)
        if n <= 0: return
        ang = aim_at(src, tgt, ang_vel, n)
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])

    # Tracking targeted
    targeted = {}
    for fl in fleets:
        if fl.owner == me:
            for p in planets:
                if p.owner == me: continue
                ea = math.atan2(p.y - fl.y, p.x - fl.x)
                if abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea))) < 0.35:
                    targeted[p.id] = targeted.get(p.id, 0) + fl.ships

    # 1. Defense
    for p in my_p:
        fi, ei = fleet_impact(p, fleets, me)
        net = p.ships + fi - ei
        if net < LEARNED_PARAMS["defend_threshold"]:
            needed = LEARNED_PARAMS["defend_threshold"] * 2 - net
            for d in sorted(my_p, key=lambda d: dp(d, p)):
                if d.id == p.id: continue
                av = avail(d)
                if av > 0:
                    to_send = min(av, needed)
                    send_to(d, p, to_send)
                    needed -= to_send
                    if needed <= 0: break

    # 2. Expansion / Attack Scoring
    targets = []
    for p in neutral_p + enemy_p:
        dist_to = min(dp(p, m) for m in my_p) if my_p else 50
        # Score = (Production * Weight) - (Garrison * Weight) - (Distance * Weight)
        score = (p.production * LEARNED_PARAMS["prod_weight"]) \
                - (p.ships * LEARNED_PARAMS["ship_cost_weight"]) \
                - (dist_to * LEARNED_PARAMS["dist_weight"])
        
        # Phase bonuses
        if is_early and p.owner == -1: 
            score *= LEARNED_PARAMS["prod_target_early"]
        
        targets.append((p, score))
    
    targets.sort(key=lambda x: x[1], reverse=True)

    for tgt, score in targets:
        needed = tgt.ships + 1
        if tgt.owner != -1:
            # Enemy growth
            t_est = travel_turns(min(dp(tgt, m) for m in my_p), 20)
            needed += int(tgt.production * t_est)
            # Apply aggressive/defensive ratios
            ratio = LEARNED_PARAMS["aggressive_ship_ratio"] if is_late else LEARNED_PARAMS["defensive_ship_ratio"]
            needed = int(needed * ratio)
        
        needed = max(1, needed - targeted.get(tgt.id, 0))
        if needed <= 0: continue

        # Coordinate
        can_reach = sorted([d for d in my_p if avail(d) > 0], key=lambda d: dp(d, tgt))
        if sum(avail(d) for d in can_reach) >= needed:
            for d in can_reach:
                to_send = min(avail(d), needed)
                send_to(d, tgt, to_send)
                needed -= to_send
                if needed <= 0: break

    # 3. Consolidation
    if enemy_p and not is_early:
        front = min(my_p, key=lambda p: min(dp(p, e) for e in enemy_p))
        for p in my_p:
            if p.id == front.id: continue
            av = avail(p)
            if av >= LEARNED_PARAMS["consolidate_avail"]:
                send_to(p, front, int(av * LEARNED_PARAMS["consolidate_frac"]))

    return moves
