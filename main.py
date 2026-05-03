"""
Orbit Wars - Competitive Agent v4 (Evolvable)
=============================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

This version uses a global PARAMS dictionary so that evolve.py can tune it.
"""

import math
import os
import json
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# These are the default values from v4
PARAMS = {
    'MIN_HOLD': 1,
    'REINFORCE_NET_LIMIT': 3,
    'REINFORCE_MIN_SEND': 5,
    'REINFORCE_TARGET_NET': 10,
    'NEUTRAL_VALUE_PROD_MULT': 10,
    'ATTACK_DIST_WEIGHT': 0.2,
    'ATTACK_PROD_WEIGHT': 3.0,
    'ATTACK_BUFFER': 5,
    'CONSOLIDATE_MIN_SURPLUS': 20,
    'CONSOLIDATE_RATIO': 0.5,
    'ATTACK_THRESHOLD_AV': 5,
}

# Load evolved parameters if they exist
_params_path = os.path.join(os.path.dirname(__file__), "best_params.json")
if os.path.exists(_params_path):
    try:
        with open(_params_path, "r") as _f:
            _loaded = json.load(_f)
            PARAMS.update(_loaded)
    except: pass

SUN_X, SUN_Y = 50.0, 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0


# ── geometry ──────────────────────────────────────────────────────────────────

def dist(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)

def dp(a, b):   # dist between two Planet-like objects
    return math.hypot(b.x - a.x, b.y - a.y)

def fleet_speed(n):
    n = max(1, int(n))
    if n == 1: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5

def travel_turns(d_val, ships): 
    return d_val / fleet_speed(ships)

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
    if not hits_sun(sx, sy, tx, ty):
        return direct
    dist_st = dist(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0]:
        ang = direct + off
        ex = sx + dist_st * math.cos(ang)
        ey = sy + dist_st * math.sin(ang)
        if not hits_sun(sx, sy, ex, ey):
            return ang
    return direct


# ── orbit prediction ──────────────────────────────────────────────────────────

def predict_pos(p, ang_vel, turns):
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= 50.0:
        return p.x, p.y     # static
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


# ── fleet analysis ────────────────────────────────────────────────────────────

def fleet_net(planet, all_fleets, me):
    fi = ei = 0
    for fl in all_fleets:
        ea   = math.atan2(planet.y - fl.y, planet.x - fl.x)
        diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
        if diff < 0.35:
            if fl.owner == me:
                fi += fl.ships
            else:
                ei += fl.ships
    return fi, ei


# ── main agent ────────────────────────────────────────────────────────────────

def agent(obs):
    g = lambda key, default: (obs.get(key, default) if isinstance(obs, dict)
                              else getattr(obs, key, default))

    me         = int(g("player", 0))
    ang_vel    = float(g("angular_velocity", 0.033))
    comet_ids  = set(g("comet_planet_ids", []))

    planets    = [Planet(*p) for p in g("planets", [])]
    fleets     = [Fleet(*f)  for f in g("fleets",  [])]

    my_p      = [p for p in planets if p.owner == me]
    neutral_p = [p for p in planets if p.owner == -1]
    enemy_p   = [p for p in planets if p.owner not in (-1, me)]
    
    committed = {}
    moves     = []

    def avail(p):
        return max(0, p.ships - committed.get(p.id, 0) - PARAMS['MIN_HOLD'])

    def send_to(src, tgt, n):
        n = int(n)
        if n <= 0: return
        ang = aim_at(src, tgt, ang_vel, n)
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])

    # Tracking targeted planets
    targeted = set()
    for fl in fleets:
        if fl.owner == me:
            for p in planets:
                if p.owner == me: continue
                ea = math.atan2(p.y - fl.y, p.x - fl.x)
                if abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea))) < 0.35:
                    targeted.add(p.id)

    # ── Phase 1: Reinforcement ────────────────────────────────────────────────
    threat = {p.id: fleet_net(p, fleets, me) for p in my_p}
    for p in sorted(my_p, key=lambda p: threat[p.id][1] - threat[p.id][0] - p.ships, reverse=True):
        fi, ei = threat[p.id]
        net = p.ships + fi - ei
        if net <= PARAMS['REINFORCE_NET_LIMIT']:
            needed = max(PARAMS['REINFORCE_MIN_SEND'], PARAMS['REINFORCE_TARGET_NET'] - net)
            donors = sorted([q for q in my_p if q.id != p.id and avail(q) >= needed], key=lambda q: dp(q, p))
            if donors:
                send_to(donors[0], p, min(needed, avail(donors[0])))

    # ── Phase 2: Expansion (Neutrals) ─────────────────────────────────────────
    free_neutrals = sorted([n for n in neutral_p if n.id not in targeted], 
                           key=lambda n: -(n.production * PARAMS['NEUTRAL_VALUE_PROD_MULT'] - n.ships))
    for src in sorted(my_p, key=lambda p: -avail(p)):
        for tgt in list(free_neutrals):
            av = avail(src)
            cost = tgt.ships + 1
            if cost <= av:
                send_to(src, tgt, cost)
                targeted.add(tgt.id)
                free_neutrals.remove(tgt)
            if avail(src) < 2: break

    # ── Phase 3: Attacks (Enemy) ──────────────────────────────────────────────
    free_enemies = [e for e in enemy_p if e.id not in targeted]
    for src in sorted(my_p, key=lambda p: -avail(p)):
        av = avail(src)
        if av < PARAMS['ATTACK_THRESHOLD_AV'] or not free_enemies: continue
        
        tgt = min(free_enemies, key=lambda e: e.ships + dp(src, e) * PARAMS['ATTACK_DIST_WEIGHT'] - e.production * PARAMS['ATTACK_PROD_WEIGHT'])
        
        t_est = travel_turns(dp(src, tgt), av)
        need = tgt.ships + int(tgt.production * t_est) + 1
        
        if av >= need:
            send_to(src, tgt, min(av, need + PARAMS['ATTACK_BUFFER']))
            targeted.add(tgt.id)
            free_enemies.remove(tgt)

    # ── Phase 4: Consolidate ──────────────────────────────────────────────────
    if enemy_p and len(my_p) > 1:
        front = min(my_p, key=lambda p: min(dp(p, e) for e in enemy_p))
        for src in my_p:
            if src.id != front.id:
                av = avail(src)
                if av >= PARAMS['CONSOLIDATE_MIN_SURPLUS']:
                    send_to(src, front, int(av * PARAMS['CONSOLIDATE_RATIO']))

    return moves
