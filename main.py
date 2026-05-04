"""
Orbit Wars – ML-Adaptive Agent v3
===================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

Run  python learn_strategy.py  to update LEARNED_PARAMS automatically.
"""

import math
import os
import pickle
import numpy as np
from pathlib import Path

# Try to import from kaggle environment; fall back to mock classes if unavailable
try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
except ImportError:
    class Planet:
        def __init__(self, id, owner, x, y, radius, ships, production):
            self.id, self.owner, self.x, self.y = id, owner, x, y
            self.radius, self.ships, self.production = radius, ships, production
    class Fleet:
        def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
            self.id, self.owner, self.x, self.y = id, owner, x, y
            self.angle, self.from_planet_id, self.ships = angle, from_planet_id, ships

# ─────────────────────────────────────────────────────────────────────────────
# <<LEARNED_PARAMS_START>>
LEARNED_PARAMS = {
    "prod_weight": 11.512,
    "ship_cost_weight": 0.92,
    "dist_weight": 0.481,
    "early_end_turn": 94,
    "late_start_turn": 357,
    "defend_threshold": 17,
    "min_hold_base": 5,
    "min_hold_threat": 10,
    "attack_buffer_ratio": 0.042,
    "min_attack_avail": 5,
    "min_expand_avail": 4,
    "consolidate_avail": 22,
    "consolidate_frac": 0.699,
    "aggressive_ship_ratio": 2.912,
    "defensive_ship_ratio": 1.355,
    "prod_target_early": 1.312,
}
# <<LEARNED_PARAMS_END>>

# ─────────────────────────────────────────────────────────────────────────────
# Optional: load trained sklearn models
# Try multiple paths for robustness on Kaggle
_PKL_PATHS = [
    Path(__file__).parent / "strategy_data.pkl",
    Path("strategy_data.pkl"),
    Path("/kaggle/working/strategy_data.pkl")
]
_ML  = {}
for _p in _PKL_PATHS:
    try:
        if _p.exists():
            with open(_p, "rb") as _fp:
                # We use a try-except here because unpickling might trigger 
                # ModuleNotFoundError if sklearn is not installed.
                _bundle = pickle.load(_fp)
                _ML = {
                    "win_rf":  _bundle.get("win_rf"),
                    "win_gbc": _bundle.get("win_gbc"),
                    "scaler":  _bundle.get("scaler"),
                    "params":  _bundle.get("params", {}),
                }
                if _ML.get("params"):
                    LEARNED_PARAMS.update(_ML["params"])
            break # Success
    except (Exception, ImportError, ModuleNotFoundError):
        continue # Try next path or fall back

# ─────────────────────────────────────────────────────────────────────────────
SUN_X, SUN_Y = 50.0, 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0
_prev_ships: dict = {}

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
    t1, t2 = (-b - sq) / (2*a), (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)

def safe_angle(sx, sy, tx, ty):
    direct = math.atan2(ty - sy, tx - sx)
    if not hits_sun(sx, sy, tx, ty): return direct
    d = dist(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]:
        ang = direct + off
        ex, ey = sx + d * math.cos(ang), sy + d * math.sin(ang)
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
    for _ in range(10):
        d_est = dist(src.x, src.y, tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    return safe_angle(src.x, src.y, tx, ty)

def fleet_net(planet, all_fleets, me):
    fi = ei = 0
    for fl in all_fleets:
        ea   = math.atan2(planet.y - fl.y, planet.x - fl.x)
        diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
        if diff < 0.30:
            if fl.owner == me: fi += fl.ships
            else: ei += fl.ships
    return fi, ei

def compute_threat(my_planets, fleets, me):
    threat = {}
    for p in my_planets:
        fi, ei = fleet_net(p, fleets, me)
        threat[p.id] = p.ships + fi - ei
    return threat

EARLY, MID, LATE = "EARLY", "MID", "LATE"
AGGRESSIVE, BALANCED, DEFENSIVE = "AGGRESSIVE", "BALANCED", "DEFENSIVE"

def game_phase(step):
    if step < LEARNED_PARAMS["early_end_turn"]: return EARLY
    if step >= LEARNED_PARAMS["late_start_turn"]: return LATE
    return MID

def strategic_mode(my_total, enemy_total):
    ratio = my_total / max(1, enemy_total)
    agg = LEARNED_PARAMS["aggressive_ship_ratio"]
    dfn = LEARNED_PARAMS["defensive_ship_ratio"]
    if ratio >= agg: return AGGRESSIVE
    if ratio <= dfn: return DEFENSIVE
    return BALANCED

def comet_turns_remaining(comet_id, obs_comets, obs_step):
    if not obs_comets: return 999
    try:
        for group in obs_comets:
            ids = list(group.get("planet_ids", []) if isinstance(group, dict) else getattr(group, "planet_ids", []))
            if comet_id not in ids: continue
            paths = (group.get("paths") if isinstance(group, dict) else getattr(group, "paths", None))
            idx   = (group.get("path_index", 0) if isinstance(group, dict) else getattr(group, "path_index", 0))
            if paths: return max(0, len(paths[ids.index(comet_id)]) - int(idx))
    except: pass
    return 999

def comet_is_capturable(comet, src, ships, ang_vel, comet_id, obs_comets, obs_step):
    travel = travel_turns(dp(src, comet), ships)
    remaining = comet_turns_remaining(comet_id, obs_comets, obs_step)
    return remaining > travel + 10

def should_expand(my_total, enemy_total, my_prod, enemy_prod, phase, mode, neutral_count):
    prod_ratio = my_prod / max(1, enemy_prod)
    if phase == EARLY: return neutral_count > 0
    if mode == DEFENSIVE: return prod_ratio < LEARNED_PARAMS["prod_target_early"] and neutral_count > 0
    return neutral_count > 0 and prod_ratio < 2.0

def agent(obs):
    global _prev_ships
    g = lambda key, default: (obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default))
    me = int(g("player", 0)); ang_vel = float(g("angular_velocity", 0.033))
    comet_ids = set(g("comet_planet_ids", [])); obs_step = int(g("step", 0)); obs_comets = g("comets", [])
    planets = [Planet(*p) for p in g("planets", [])]; fleets = [Fleet(*f) for f in g("fleets", [])]
    my_p = [p for p in planets if p.owner == me]; neutral_p = [p for p in planets if p.owner == -1]; enemy_p = [p for p in planets if p.owner not in (-1, me)]
    if not my_p: return []
    my_fl = [fl for fl in fleets if fl.owner == me]; en_fl = [fl for fl in fleets if fl.owner != me]
    my_total = sum(p.ships for p in my_p) + sum(fl.ships for fl in my_fl); en_total = sum(p.ships for p in enemy_p) + sum(fl.ships for fl in en_fl)
    my_prod = sum(p.production for p in my_p); en_prod = sum(p.production for p in enemy_p)
    phase = game_phase(obs_step); mode = strategic_mode(my_total, en_total)
    threat = compute_threat(my_p, fleets, me)
    def_thr = LEARNED_PARAMS["defend_threshold"]
    any_threat = any(v < def_thr for v in threat.values())
    min_hold = LEARNED_PARAMS["min_hold_threat"] if any_threat else LEARNED_PARAMS["min_hold_base"]
    committed, moves = {}, []
    def avail(p): return max(0, p.ships - committed.get(p.id, 0) - max(min_hold, int(p.ships * 0.30)))
    def send_to(src, tgt, n):
        n = int(n)
        if n <= 0: return
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, aim_at(src, tgt, ang_vel, n), n])

    # Phase 0: Emergency
    for p in my_p:
        prev = _prev_ships.get(p.id)
        if prev and (prev - p.ships) / prev > 0.60:
            donors = sorted([q for q in my_p if q.id != p.id], key=lambda q: dp(q, p))
            for donor in donors:
                av = avail(donor)
                if av > 0: send_to(donor, p, av)
    # Phase 1: Reinforce
    for p in sorted(my_p, key=lambda p: threat[p.id]):
        if threat[p.id] < def_thr:
            needed = def_thr - threat[p.id] + 2
            donors = sorted([q for q in my_p if q.id != p.id and avail(q) >= needed and threat[q.id] >= def_thr], key=lambda q: dp(q, p))
            if donors: send_to(donors[0], p, min(needed, avail(donors[0])))
    # Phase 2: Expand
    targeted = set()
    for fl in my_fl:
        for p in neutral_p + enemy_p:
            if fleet_net(p, [fl], me)[0] > 0: targeted.add(p.id)
    if should_expand(my_total, en_total, my_prod, en_prod, phase, mode, len(neutral_p)):
        for src in sorted(my_p, key=lambda p: -avail(p)):
            av = avail(src)
            if av < LEARNED_PARAMS["min_expand_avail"]: continue
            candidates = sorted([n for n in neutral_p if n.id not in targeted], key=lambda n: (1 if n.id not in comet_ids else 2, dp(src, n)))
            for tgt in candidates:
                if tgt.id in comet_ids and not comet_is_capturable(tgt, src, tgt.ships+1, ang_vel, tgt.id, obs_comets, obs_step): continue
                cost = tgt.ships + 1 + fleet_net(tgt, fleets, me)[1]
                if cost <= av:
                    send_to(src, tgt, cost)
                    targeted.add(tgt.id); break
    # Phase 3: Attack
    attack_mult = {"AGGRESSIVE": 1.0, "BALANCED": 1.0, "DEFENSIVE": 1.5}[mode]
    if enemy_p:
        cx, cy = sum(p.x for p in my_p)/len(my_p), sum(p.y for p in my_p)/len(my_p)
        sorted_enemies = sorted(enemy_p, key=lambda e: dist(e.x, e.y, cx, cy))
        coord_tgt = sorted_enemies[obs_step % len(sorted_enemies)]
        for src in sorted(my_p, key=lambda p: -avail(p)):
            av = avail(src)
            if av >= LEARNED_PARAMS["min_attack_avail"]:
                need = coord_tgt.ships + 1
                to_send = min(av, need + max(3, int(need * LEARNED_PARAMS["attack_buffer_ratio"] * attack_mult)))
                send_to(src, coord_tgt, to_send)
    # Phase 4: Consolidate
    if phase != EARLY and enemy_p and len(my_p) > 1:
        front = min(my_p, key=lambda p: min(dp(p, e) for e in enemy_p) + min(dp(p, e) for e in enemy_p)*0.5)
        c_thr = LEARNED_PARAMS["consolidate_avail"] // (2 if mode == AGGRESSIVE else 1)
        f_dist = min(dp(front, e) for e in enemy_p)
        for src in my_p:
            if src.id != front.id and fleet_net(src, fleets, me)[0] == 0 and min(dp(src, e) for e in enemy_p) > f_dist * 1.2:
                av = avail(src)
                if av >= c_thr and threat[src.id] >= def_thr: send_to(src, front, int(av * LEARNED_PARAMS["consolidate_frac"]))
    owned_ids = {p.id for p in my_p}
    for p in my_p: _prev_ships[p.id] = p.ships
    for pid in list(_prev_ships.keys()):
        if pid not in owned_ids: del _prev_ships[pid]
    return moves
