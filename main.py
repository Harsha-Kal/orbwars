"""
Orbit Wars - ML-Compatible Advanced Agent
==========================================
This agent is designed to work with learn_strategy.py.
It uses markers to allow the ML script to automatically update parameters.
"""

import math
import pickle
from pathlib import Path
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# <<LEARNED_PARAMS_START>>
LEARNED_PARAMS = {
    "prod_weight": 15.0,
    "ship_cost_weight": 1.0,
    "dist_weight": 0.15,
    "early_end_turn": 99,
    "late_start_turn": 349,
    "defend_threshold": 13,
    "min_hold_base": 1,
    "min_hold_threat": 6,
    "attack_buffer_ratio": 0.25,
    "min_attack_avail": 4,
    "min_expand_avail": 2,
    "consolidate_avail": 193,
    "consolidate_frac": 0.5,
    "aggressive_ship_ratio": 3.58,
    "defensive_ship_ratio": 0.88,
    "prod_target_early": 1.56,
}
# <<LEARNED_PARAMS_END>>


# Manual fallback values. Edit LEARNED_PARAMS above when you want to override the
# local strategy without regenerating strategy_data.pkl.
PARAM_DEFAULTS = {
    "reserve_fraction": 0.30,
    "active_attack_fraction": 0.70,
    "near_idle_dist_weight": 0.85,
    "idle_first_max_distance": 100.0,
    "recapture_buffer": 2,
}
MANUAL_PARAMS = dict(PARAM_DEFAULTS)
MANUAL_PARAMS.update(LEARNED_PARAMS)
USE_STRATEGY_DATA = True


def load_strategy_params():
    params = dict(MANUAL_PARAMS)
    ml = {}
    if not USE_STRATEGY_DATA:
        return params, ml

    pkl_path = Path(__file__).parent / "strategy_data.pkl" if "__file__" in globals() else Path("strategy_data.pkl")
    try:
        if pkl_path.exists():
            with open(pkl_path, "rb") as fp:
                bundle = pickle.load(fp)
            ml = {
                "win_rf": bundle.get("win_rf"),
                "win_gbc": bundle.get("win_gbc"),
                "scaler": bundle.get("scaler"),
                "params": bundle.get("params", {}),
            }
            if ml["params"]:
                params.update(ml["params"])
    except Exception:
        pass
    return params, ml


LEARNED_PARAMS, _ML = load_strategy_params()

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

def is_idle_planet(p):
    return dist(p.x, p.y, SUN_X, SUN_Y) + p.radius >= 50.0


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
    if not my_p:
        return []

    # Dynamic phase logic
    is_early = step < LEARNED_PARAMS["early_end_turn"]
    is_late  = step > LEARNED_PARAMS["late_start_turn"]

    committed, moves = {}, []
    
    def get_min_hold(p):
        _, ei = fleet_impact(p, fleets, me)
        reserve = int(math.ceil(p.ships * LEARNED_PARAMS.get("reserve_fraction", 0.30)))
        if ei > 0:
            return max(reserve, LEARNED_PARAMS["min_hold_threat"])
        return max(reserve, LEARNED_PARAMS["min_hold_base"])

    def avail(p): 
        return max(0, p.ships - committed.get(p.id, 0) - get_min_hold(p))

    def send_to(src, tgt, n):
        n = int(n)
        if n <= 0: return
        ang = aim_at(src, tgt, ang_vel, n)
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])
        if tgt.owner != me:
            targeted[tgt.id] = targeted.get(tgt.id, 0) + n

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
    recapture_needed = {}
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
        if net <= 0:
            recapture_needed[p.id] = int(-net + p.production + LEARNED_PARAMS.get("recapture_buffer", 2))

    # If an enemy attack looks strong enough to flip one of our planets, queue
    # enough support from nearby planets so it can be held or immediately retaken.
    for p in sorted(my_p, key=lambda p: recapture_needed.get(p.id, 0), reverse=True):
        needed = recapture_needed.get(p.id, 0)
        if needed <= 0:
            continue
        for d in sorted(my_p, key=lambda d: dp(d, p)):
            if d.id == p.id:
                continue
            to_send = min(avail(d), needed)
            if to_send > 0:
                send_to(d, p, to_send)
                needed -= to_send
                if needed <= 0:
                    break

    # 2. Prefer nearby idle neutrals before the broader expansion/attack pass.
    idle_neutrals = sorted(
        [p for p in neutral_p if is_idle_planet(p)],
        key=lambda p: min(dp(p, m) for m in my_p),
    )
    max_idle_dist = LEARNED_PARAMS.get("idle_first_max_distance", 100.0)
    for tgt in idle_neutrals:
        dist_to = min(dp(tgt, m) for m in my_p)
        if dist_to > max_idle_dist:
            continue

        needed = max(1, tgt.ships + 1 - targeted.get(tgt.id, 0))
        can_reach = sorted([d for d in my_p if avail(d) > 0], key=lambda d: dp(d, tgt))
        if sum(avail(d) for d in can_reach) < needed:
            continue

        for d in can_reach:
            to_send = min(avail(d), needed)
            send_to(d, tgt, to_send)
            needed -= to_send
            if needed <= 0:
                break

    # 3. Expansion / Attack Scoring
    targets = []
    for p in neutral_p + enemy_p:
        dist_to = min(dp(p, m) for m in my_p) if my_p else 50
        # Score = (Production * Weight) - (Garrison * Weight) - (Distance * Weight)
        score = (p.production * LEARNED_PARAMS["prod_weight"]) \
                - (p.ships * LEARNED_PARAMS["ship_cost_weight"]) \
                - (dist_to * LEARNED_PARAMS["dist_weight"])
        
        if p.owner == -1 and is_idle_planet(p):
            score += max(0.0, 100.0 - dist_to) * LEARNED_PARAMS.get("near_idle_dist_weight", 0.85)
        if is_early and p.owner == -1:
            score *= LEARNED_PARAMS["prod_target_early"]
        
        targets.append((p, score))
    
    targets.sort(key=lambda x: (
        x[0].owner == -1 and is_idle_planet(x[0]),
        -min(dp(x[0], m) for m in my_p),
        x[1],
    ), reverse=True)

    attack_source_limit = max(1, int(math.ceil(len(my_p) * LEARNED_PARAMS.get("active_attack_fraction", 0.30))))
    active_sources = sorted([p for p in my_p if avail(p) > 0], key=lambda p: avail(p), reverse=True)[:attack_source_limit]
    attack_source_ids = {p.id for p in active_sources}

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

        # Coordinate attacks, but only from the active slice of planets this turn.
        can_reach = sorted([d for d in my_p if d.id in attack_source_ids and avail(d) > 0], key=lambda d: dp(d, tgt))
        if sum(avail(d) for d in can_reach) >= needed:
            for d in can_reach:
                to_send = min(avail(d), needed)
                send_to(d, tgt, to_send)
                needed -= to_send
                if needed <= 0: break

    # 4. Consolidation
    if enemy_p and not is_early:
        front = min(my_p, key=lambda p: min(dp(p, e) for e in enemy_p))
        for p in my_p:
            if p.id == front.id: continue
            av = avail(p)
            if av >= LEARNED_PARAMS["consolidate_avail"]:
                send_to(p, front, int(av * LEARNED_PARAMS["consolidate_frac"]))

    return moves
