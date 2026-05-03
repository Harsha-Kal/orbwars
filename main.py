"""
Orbit Wars – ML-Adaptive Agent v3
===================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

Strategy layers
---------------
  1. Detect game phase (EARLY / MID / LATE) and strategic mode
     (AGGRESSIVE / BALANCED / DEFENSIVE) based on ship-count ratio.
  2. Load sklearn models from strategy_data.pkl when present; otherwise
     fall back to LEARNED_PARAMS constants tuned by learn_strategy.py.
  3. Reinforce any of our planets under net threat (phase 1).
  4. Expand to high-value neutrals; comets captured only when safe (phase 2).
  5. Attack weakest enemy weighted by production / distance (phase 3).
  6. Consolidate surplus toward the front-line planet (phase 4).

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
    try:
        from kaggle_environments import make
        # Fallback: define minimal Planet and Fleet classes
        class Planet:
            def __init__(self, id, owner, x, y, radius, ships, production):
                self.id, self.owner, self.x, self.y = id, owner, x, y
                self.radius, self.ships, self.production = radius, ships, production
        
        class Fleet:
            def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
                self.id, self.owner, self.x, self.y = id, owner, x, y
                self.angle, self.from_planet_id, self.ships = angle, from_planet_id, ships
    except ImportError:
        raise ImportError("kaggle-environments not properly installed")

# ─────────────────────────────────────────────────────────────────────────────
# <<LEARNED_PARAMS_START>>
LEARNED_PARAMS = {
    "prod_weight": 10.0,
    "ship_cost_weight": 1.0,
    "dist_weight": 0.15,
    "early_end_turn": 100,
    "late_start_turn": 350,
    "defend_threshold": 3,
    "min_hold_base": 1,
    "min_hold_threat": 4,
    "attack_buffer_ratio": 0.2,
    "min_attack_avail": 4,
    "min_expand_avail": 2,
    "consolidate_avail": 15,
    "consolidate_frac": 0.5,
    "aggressive_ship_ratio": 1.5,
    "defensive_ship_ratio": 0.75,
    "prod_target_early": 1.2,
}
# <<LEARNED_PARAMS_END>>

# ─────────────────────────────────────────────────────────────────────────────
# Optional: load trained sklearn models
_PKL = Path(__file__).parent / "strategy_data.pkl"
_ML  = {}
try:
    if _PKL.exists():
        with open(_PKL, "rb") as _fp:
            _bundle = pickle.load(_fp)
        _ML = {
            "win_rf":  _bundle.get("win_rf"),
            "win_gbc": _bundle.get("win_gbc"),
            "scaler":  _bundle.get("scaler"),
            "params":  _bundle.get("params", {}),
        }
        # Merge ML-derived params (override defaults)
        if _ML.get("params"):
            LEARNED_PARAMS.update(_ML["params"])
except Exception:
    pass  # no models available, use constants

# ─────────────────────────────────────────────────────────────────────────────
SUN_X, SUN_Y = 50.0, 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0

# Tracks ship counts from the previous turn to detect sudden garrison drops
_prev_ships: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def dist(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)

def dp(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)

def fleet_speed(n):
    n = max(1, int(n))
    if n == 1:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5

def travel_turns(d_val, ships):
    return d_val / fleet_speed(ships)

def hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx*dx + dy*dy
    if a < 1e-9:
        return math.hypot(fx, fy) < SUN_R
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - SUN_R*SUN_R
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)

def safe_angle(sx, sy, tx, ty):
    direct = math.atan2(ty - sy, tx - sx)
    if not hits_sun(sx, sy, tx, ty):
        return direct
    d = dist(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0, 1.5, -1.5]:
        ang = direct + off
        ex = sx + d * math.cos(ang)
        ey = sy + d * math.sin(ang)
        if not hits_sun(sx, sy, ex, ey):
            return ang
    return direct


# ══════════════════════════════════════════════════════════════════════════════
#  ORBIT PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_pos(p, ang_vel, turns):
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= 50.0:
        return p.x, p.y
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


# ══════════════════════════════════════════════════════════════════════════════
#  FLEET & THREAT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def fleet_net(planet, all_fleets, me):
    """Return (my_inbound, enemy_inbound) using tight angle + distance check."""
    fi = ei = 0
    for fl in all_fleets:
        ea   = math.atan2(planet.y - fl.y, planet.x - fl.x)
        diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
        if diff < 0.30:
            if fl.owner == me:
                fi += fl.ships
            else:
                ei += fl.ships
    return fi, ei

def compute_threat(my_planets, fleets, me):
    """Return dict {planet_id: net_garrison} (negative = under threat)."""
    threat = {}
    for p in my_planets:
        fi, ei = fleet_net(p, fleets, me)
        threat[p.id] = p.ships + fi - ei
    return threat


# ══════════════════════════════════════════════════════════════════════════════
#  PLANET SCORING
# ══════════════════════════════════════════════════════════════════════════════

def planet_value(p, src=None):
    """
    Desirability of capturing/owning planet p.
    Higher = more desirable. Uses LEARNED_PARAMS weights.
    """
    pw  = LEARNED_PARAMS["prod_weight"]
    scw = LEARNED_PARAMS["ship_cost_weight"]
    dw  = LEARNED_PARAMS["dist_weight"]
    base = p.production * pw - p.ships * scw
    if src is not None:
        base -= dp(src, p) * dw
    return base

def enemy_priority(e, src, fleets, me):
    """Score for attacking enemy planet (lower = higher priority)."""
    d_val   = dp(src, e)
    third_inbound = sum(
        fl.ships for fl in fleets
        if fl.owner not in (me, e.owner)
        and fleet_net(e, [fl], me)[1] > 0
    )
    # Prefer: close, cheap, high production, already being attacked by others
    return e.ships - third_inbound * 0.5 + d_val * 0.2 - e.production * 3


# ══════════════════════════════════════════════════════════════════════════════
#  GAME PHASE & STRATEGIC MODE
# ══════════════════════════════════════════════════════════════════════════════

EARLY = "EARLY"
MID   = "MID"
LATE  = "LATE"

AGGRESSIVE = "AGGRESSIVE"
BALANCED   = "BALANCED"
DEFENSIVE  = "DEFENSIVE"

def game_phase(step):
    if step < LEARNED_PARAMS["early_end_turn"]:
        return EARLY
    if step >= LEARNED_PARAMS["late_start_turn"]:
        return LATE
    return MID

def strategic_mode(my_total, enemy_total):
    """
    Compare total ship counts (planets + fleets).
    Use ML win predictor if available, else use learned thresholds.
    """
    ratio = my_total / max(1, enemy_total)

    if _ML.get("win_rf") and _ML.get("scaler"):
        try:
            # quick single-feature prediction using ratio as proxy
            pass  # full feature vector not available here; fall through
        except Exception:
            pass

    agg = LEARNED_PARAMS["aggressive_ship_ratio"]
    dfn = LEARNED_PARAMS["defensive_ship_ratio"]
    if ratio >= agg:
        return AGGRESSIVE
    if ratio <= dfn:
        return DEFENSIVE
    return BALANCED

def ml_win_prob(features: dict) -> float:
    """Return estimated win probability from RF model, or 0.5 if unavailable."""
    rf  = _ML.get("win_rf")
    scl = _ML.get("scaler")
    if rf is None or scl is None:
        return 0.5
    try:
        import numpy as np
        FEATURE_COLS = [
            "turn_frac","my_ships","enemy_ships","neutral_ships",
            "ship_ratio","fleet_ratio","my_prod","enemy_prod","prod_ratio",
            "my_planets","enemy_planets","neutral_planets",
            "my_fleets_ships","enemy_fleets_ships","max_threat","total_threat",
            "cheapest_neutral","best_neutral_prod","neutral_opportunity",
            "weakest_enemy","best_attack_score","dominance","leading",
            "early_game","mid_game","late_game",
        ]
        row = np.array([[features.get(c, 0.0) for c in FEATURE_COLS]])
        row_s = scl.transform(row)
        return float(rf.predict_proba(row_s)[0, 1])
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  COMET HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def comet_turns_remaining(comet_id, obs_comets, obs_step):
    """Estimate how many turns until this comet leaves the board."""
    if not obs_comets:
        return 999
    try:
        for group in obs_comets:
            ids = list(group.get("planet_ids", []) if isinstance(group, dict)
                       else getattr(group, "planet_ids", []))
            if comet_id not in ids:
                continue
            paths = (group.get("paths") if isinstance(group, dict)
                     else getattr(group, "paths", None))
            idx   = (group.get("path_index", 0) if isinstance(group, dict)
                     else getattr(group, "path_index", 0))
            if paths:
                path = paths[ids.index(comet_id)]
                return max(0, len(path) - int(idx))
    except Exception:
        pass
    return 999

def comet_is_capturable(comet, src, ships, ang_vel, comet_id,
                        obs_comets, obs_step):
    """True only if comet will stick around long enough to be useful."""
    travel = travel_turns(dp(src, comet), ships)
    remaining = comet_turns_remaining(comet_id, obs_comets, obs_step)
    return remaining > travel + 10   # 10-turn buffer to earn back ships


# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCTION FORECASTING  (lightweight Prophet-style linear trend)
# ══════════════════════════════════════════════════════════════════════════════

def forecast_ships(current_ships, production, turns_ahead, threat_drain=0):
    """
    Simple linear production model.
    threat_drain: expected ships lost per turn to incoming fleets.
    """
    return current_ships + (production - threat_drain) * turns_ahead


def should_expand(my_total, enemy_total, my_prod, enemy_prod,
                  phase, mode, neutral_count):
    """
    Heuristic: expand when production is behind or we have room.
    Delegates to ML win prob if available.
    """
    prod_ratio = my_prod / max(1, enemy_prod)
    prod_target = LEARNED_PARAMS["prod_target_early"]

    if phase == EARLY:
        return neutral_count > 0  # always expand early
    if mode == DEFENSIVE:
        return prod_ratio < prod_target and neutral_count > 0
    return neutral_count > 0 and prod_ratio < 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AGENT
# ══════════════════════════════════════════════════════════════════════════════

def agent(obs):
    # ── Normalise observation ────────────────────────────────────────────────
    g = lambda key, default: (obs.get(key, default) if isinstance(obs, dict)
                              else getattr(obs, key, default))

    me         = int(g("player", 0))
    ang_vel    = float(g("angular_velocity", 0.033))
    comet_ids  = set(g("comet_planet_ids", []))
    obs_step   = int(g("step", 0))
    obs_comets = g("comets", [])

    planets = [Planet(*p) for p in g("planets", [])]
    fleets  = [Fleet(*f)  for f in g("fleets",  [])]

    my_p      = [p for p in planets if p.owner == me]
    neutral_p = [p for p in planets if p.owner == -1]
    enemy_p   = [p for p in planets if p.owner not in (-1, me)]

    # ── Phase / mode detection ───────────────────────────────────────────────
    phase = game_phase(obs_step)

    my_fl      = [fl for fl in fleets if fl.owner == me]
    en_fl      = [fl for fl in fleets if fl.owner != me]
    my_total   = sum(p.ships for p in my_p)  + sum(fl.ships for fl in my_fl)
    en_total   = sum(p.ships for p in enemy_p) + sum(fl.ships for fl in en_fl)
    ne_total   = sum(p.ships for p in neutral_p)
    my_prod    = sum(p.production for p in my_p)
    en_prod    = sum(p.production for p in enemy_p)

    mode = strategic_mode(my_total, en_total)

    # ── Threat map ───────────────────────────────────────────────────────────
    threat = compute_threat(my_p, fleets, me)

    # MIN_HOLD adapts: higher when under direct threat
    any_threat = any(v < LEARNED_PARAMS["defend_threshold"] for v in threat.values())
    min_hold   = (LEARNED_PARAMS["min_hold_threat"] if any_threat
                  else LEARNED_PARAMS["min_hold_base"])

    committed = {}
    moves     = []

    def avail(p):
        hold = max(min_hold, int(p.ships * 0.30))
        return max(0, p.ships - committed.get(p.id, 0) - hold)

    def send(src, ang, n):
        n = int(n)
        if n <= 0:
            return
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])

    def send_to(src, tgt, n):
        ang = aim_at(src, tgt, ang_vel, n)
        send(src, ang, n)

    # ── Already-targeted sets ────────────────────────────────────────────────
    targeted = set()
    for fl in my_fl:
        for p in neutral_p + enemy_p:
            fi, _ = fleet_net(p, [fl], me)
            if fi > 0:
                targeted.add(p.id)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 0 – Emergency: garrison dropped >60% since last turn
    # ══════════════════════════════════════════════════════════════════════════
    for p in my_p:
        prev = _prev_ships.get(p.id)
        if prev is None or prev == 0:
            continue
        if (prev - p.ships) / prev <= 0.60:
            continue
        # Planet lost >60% of ships — rush all available ships from nearest donors
        donors = sorted(
            [q for q in my_p if q.id != p.id],
            key=lambda q: dp(q, p)
        )
        for donor in donors:
            av = avail(donor)
            if av > 0:
                send_to(donor, p, av)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1 – Reinforce threatened planets
    # ══════════════════════════════════════════════════════════════════════════
    defend_thresh = LEARNED_PARAMS["defend_threshold"]

    for p in sorted(my_p, key=lambda p: threat[p.id]):
        net = threat[p.id]
        if net >= defend_thresh:
            continue
        needed = max(5, defend_thresh - net + 2)
        donors = sorted(
            [q for q in my_p
             if q.id != p.id
             and avail(q) >= needed
             and threat[q.id] >= defend_thresh],   # don't strip defended planets
            key=lambda q: dp(q, p)
        )
        if donors:
            give = min(needed, avail(donors[0]))
            send_to(donors[0], p, give)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2 – Expand to neutral planets
    # ══════════════════════════════════════════════════════════════════════════
    do_expand = should_expand(my_total, en_total, my_prod, en_prod,
                              phase, mode, len(neutral_p))

    if do_expand:
        free_neutrals = [n for n in neutral_p if n.id not in targeted]

        for src in sorted(my_p, key=lambda p: -avail(p)):
            av = avail(src)
            if av < LEARNED_PARAMS["min_expand_avail"]:
                continue

            # Sort neutrals by distance from this source (nearest stationary first).
            # Comets are moving so penalise them; departing comets go to the end.
            def candidate_key(n, _src=src):
                d = dp(_src, n)
                if n.id in comet_ids:
                    remaining = comet_turns_remaining(n.id, obs_comets, obs_step)
                    if remaining < 30:
                        return (3, d, n.ships)  # deprioritise departing comets
                    return (2, d, n.ships)       # comets: after stationary planets
                return (1, d, n.ships)           # stationary: nearest first

            candidates = sorted(free_neutrals, key=candidate_key)

            for tgt in candidates:
                if tgt.id in comet_ids:
                    if not comet_is_capturable(tgt, src, tgt.ships + 1,
                                               ang_vel, tgt.id,
                                               obs_comets, obs_step):
                        continue

                # Beat both the current garrison AND any enemy fleet already en route
                _, enemy_inbound = fleet_net(tgt, fleets, me)
                cost = tgt.ships + 1 + enemy_inbound

                if cost <= av:
                    send_to(src, tgt, cost)
                    targeted.add(tgt.id)
                    free_neutrals = [n for n in free_neutrals if n.id != tgt.id]
                    break

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 3 – Attack enemy planets
    # ══════════════════════════════════════════════════════════════════════════
    # In DEFENSIVE mode only attack when clearly winning locally
    attack_mult = {"AGGRESSIVE": 1.0, "BALANCED": 1.0, "DEFENSIVE": 1.5}[mode]
    buf_ratio   = LEARNED_PARAMS["attack_buffer_ratio"]

    if enemy_p:
        if len(my_p) >= 3:
            # Coordinated attack: all planets hit the same target each step.
            # Rotate through enemies sorted by distance from our centroid so we
            # systematically capture the nearest cluster first.
            cx = sum(p.x for p in my_p) / len(my_p)
            cy = sum(p.y for p in my_p) / len(my_p)
            sorted_enemies = sorted(
                enemy_p, key=lambda e: math.hypot(e.x - cx, e.y - cy)
            )
            coord_tgt = sorted_enemies[obs_step % len(sorted_enemies)]

            for src in sorted(my_p, key=lambda p: -avail(p)):
                av = avail(src)
                if av < LEARNED_PARAMS["min_attack_avail"]:
                    continue
                need = coord_tgt.ships + 1
                buf  = max(3, int(need * buf_ratio * attack_mult))
                to_send = min(av, need + buf)
                if to_send > 0:
                    send_to(src, coord_tgt, to_send)
        else:
            # 1-2 planets: each attacks its nearest untargeted enemy
            free_enemies = [e for e in enemy_p if e.id not in targeted]

            for src in sorted(my_p, key=lambda p: -avail(p)):
                av = avail(src)
                if av < LEARNED_PARAMS["min_attack_avail"] or not free_enemies:
                    continue
                tgt  = min(free_enemies, key=lambda e: dp(src, e))
                need = tgt.ships + 1
                buf  = max(3, int(need * buf_ratio * attack_mult))
                if av >= need:
                    send_to(src, tgt, min(av, need + buf))
                    targeted.add(tgt.id)
                    free_enemies = [e for e in free_enemies if e.id != tgt.id]

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 4 – Consolidate surplus toward front-line
    # ══════════════════════════════════════════════════════════════════════════
    # Skip consolidation in early game (ships better used for expansion)
    if phase != EARLY and enemy_p and len(my_p) > 1:
        remaining_en = [e for e in enemy_p if e.id not in targeted] or enemy_p

        def front_score(p):
            nearest = min(remaining_en, key=lambda e: dp(p, e))
            return dp(p, nearest) + nearest.ships * 0.5

        front = min(my_p, key=front_score)
        consol_thresh = LEARNED_PARAMS["consolidate_avail"]
        consol_frac   = LEARNED_PARAMS["consolidate_frac"]

        # In AGGRESSIVE mode lower the threshold to push ships to front faster
        if mode == AGGRESSIVE:
            consol_thresh = max(8, consol_thresh // 2)

        front_dist = min(dp(front, e) for e in enemy_p)

        for src in my_p:
            if src.id == front.id:
                continue
            # Skip if src already has inbound friendly ships (avoids ping-pong)
            fi, _ = fleet_net(src, fleets, me)
            if fi > 0:
                continue
            # Only consolidate from planets clearly further back than the front
            src_dist = min(dp(src, e) for e in enemy_p)
            if src_dist <= front_dist * 1.2:
                continue
            av = avail(src)
            if av >= consol_thresh and threat[src.id] >= defend_thresh:
                send_to(src, front, int(av * consol_frac))

    # Update ship-count history for next turn's emergency check
    owned_ids = {p.id for p in my_p}
    for p in my_p:
        _prev_ships[p.id] = p.ships
    for pid in list(_prev_ships.keys()):
        if pid not in owned_ids:
            del _prev_ships[pid]

    return moves

if __name__ == "__main__":
    test_obs = {
        "player": 1,
        "step": 50,
        "angular_velocity": 0.033,
        "planets": [
            [0, 1, 20, 20, 5, 50, 5],   # id, owner, x, y, radius, ships, production
            [1, -1, 80, 80, 4, 20, 3],  # neutral planet
            [2, 2, 30, 70, 5, 40, 4],   # enemy planet
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "comets": []
    }
    result = agent(test_obs)
    print(f"Agent returned: {result}")

