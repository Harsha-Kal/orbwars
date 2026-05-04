"""
Orbit Wars - Competitive Agent v10
====================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

Design: starter-agent rhythm + orbit prediction + enemy targeting.

Benchmarks: starter-clone (same as Python starter but with orbit prediction)
achieves 17/20 vs the Python starter. This builds on that with:
  * Enemy targeting when neutrals are captured / we have garrison advantage.
  * Incoming tracking: each target is claimed by at most one planet per turn,
    so ships are distributed across multiple targets (no wasted pile-ons).
  * Orbiting neutrals: aim_at() converges on predicted intercept point —
    captures planets the starter completely ignores.
  * Smarter target selection: nearest idle first (fast compound), then orbiting,
    then enemies (highest production x remaining / garrison).

Core rhythm (inspired by Python starter):
  - Only send when ships // 2 >= MIN_FLEET.
  - Send ships // 2 per planet per turn (SEND_FRAC = 0.5).
  - committed dict tracks actual spend so Phase 1 + Phase 2 don't over-commit.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

SUN_X, SUN_Y   = 50.0, 50.0
SUN_R          = 10.0
MAX_SPEED      = 6.0
ROTATION_LIMIT = 50.0
TOTAL_STEPS    = 500

SEND_FRAC  = 0.50   # send half our available ships each time we act
MIN_FLEET  = 15     # never send fewer than this many ships in a fleet
MAX_ARC    = 12.0   # skip orbiting target if it arcs this far during travel
HEADING_T  = 0.40   # fleet heading detection tolerance (rad)
DEFEND_NET = 5      # emergency reinforce if net projected ships < this


# ── geometry ──────────────────────────────────────────────────────────────────

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
    a = dx * dx + dy * dy
    if a < 1e-9:
        return math.hypot(fx, fy) < SUN_R
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_R * SUN_R
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)

def safe_angle(sx, sy, tx, ty):
    direct = math.atan2(ty - sy, tx - sx)
    if not hits_sun(sx, sy, tx, ty):
        return direct
    d = dist(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0]:
        ang = direct + off
        ex, ey = sx + d * math.cos(ang), sy + d * math.sin(ang)
        if not hits_sun(sx, sy, ex, ey):
            return ang
    return direct


# ── orbit prediction ──────────────────────────────────────────────────────────

def predict_pos(p, ang_vel, turns):
    """Predict where planet p will be after `turns` steps."""
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= ROTATION_LIMIT:
        return p.x, p.y
    cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
    new_ang = cur_ang + ang_vel * turns
    return SUN_X + orb_r * math.cos(new_ang), SUN_Y + orb_r * math.sin(new_ang)

def aim_at(src, tgt, ang_vel, ships):
    """Iteratively converge on the intercept angle for a moving target."""
    tx, ty = tgt.x, tgt.y
    for _ in range(10):
        d_est = dist(src.x, src.y, tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    return safe_angle(src.x, src.y, tx, ty)

def fleet_heading_to(fl, p, ang_vel):
    """True if fleet is heading toward p's predicted future position."""
    d = dist(fl.x, fl.y, p.x, p.y)
    t_est = travel_turns(d, fl.ships)
    pred_x, pred_y = predict_pos(p, ang_vel, t_est)
    ea = math.atan2(pred_y - fl.y, pred_x - fl.x)
    diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
    return diff < HEADING_T

def is_idle(p):
    """True when planet does not orbit the sun."""
    return dist(p.x, p.y, SUN_X, SUN_Y) + p.radius >= ROTATION_LIMIT

def arc_travel(p, ang_vel, sx, sy):
    """How many distance-units does an orbiting planet move during fleet travel?"""
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= ROTATION_LIMIT:
        return 0.0
    d = dist(sx, sy, p.x, p.y)
    return abs(ang_vel) * orb_r * travel_turns(d, 10)


# ── main agent ────────────────────────────────────────────────────────────────

def agent(obs):
    g = lambda k, d: (obs.get(k, d) if isinstance(obs, dict) else getattr(obs, k, d))
    me      = int(g("player", 0))
    ang_vel = float(g("angular_velocity", 0.033))
    step    = int(g("step", 0))

    planets = [Planet(*p) for p in g("planets", [])]
    fleets  = [Fleet(*f)  for f in g("fleets",  [])]

    my_p      = [p for p in planets if p.owner == me]
    neutral_p = [p for p in planets if p.owner == -1]
    enemy_p   = [p for p in planets if p.owner not in (-1, me)]

    if not my_p:
        return []

    my_fleets    = [fl for fl in fleets if fl.owner == me]
    enemy_fleets = [fl for fl in fleets if fl.owner != me]

    moves     = []
    committed = {}   # planet_id → ships committed this turn
    incoming  = {}   # ships already heading to each non-owned planet

    def avail(p):
        """Available ships after garrison hold and already-committed this turn."""
        return max(0, p.ships - committed.get(p.id, 0))

    def fire(src, tgt, n):
        n = int(n)
        if n <= 0:
            return
        ang = aim_at(src, tgt, ang_vel, n)
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])
        incoming[tgt.id] = incoming.get(tgt.id, 0) + n

    # Build incoming: existing in-flight fleets to non-owned planets
    for fl in my_fleets:
        for p in neutral_p + enemy_p:
            if fleet_heading_to(fl, p, ang_vel):
                incoming[p.id] = incoming.get(p.id, 0) + fl.ships

    # ── Phase 1: Emergency defense ─────────────────────────────────────────────
    for p in my_p:
        fi = sum(fl.ships for fl in my_fleets    if fleet_heading_to(fl, p, ang_vel))
        ei = sum(fl.ships for fl in enemy_fleets if fleet_heading_to(fl, p, ang_vel))
        net = p.ships + fi - ei
        if net < DEFEND_NET:
            needed = DEFEND_NET * 2 - net
            for d in sorted(my_p, key=lambda q: dp(q, p)):
                if d.id == p.id:
                    continue
                av = avail(d)
                if av > 0:
                    to_send = min(av, needed)
                    fire(d, p, to_send)
                    needed -= to_send
                    if needed <= 0:
                        break

    # ── Phase 2: Per-planet offensive ─────────────────────────────────────────
    # Each planet independently fires at the best available target.
    # Because `incoming` is updated after each fire(), richer planets claim
    # targets first and later planets naturally spread to different targets.

    rem_steps = max(1, TOTAL_STEPS - step)

    idle_neu = sorted([n for n in neutral_p if is_idle(n)],
                      key=lambda n: min(dp(n, m) for m in my_p))

    for src in sorted(my_p, key=lambda p: -avail(p)):
        av = avail(src)
        fleet_size = int(av * SEND_FRAC)
        if fleet_size < MIN_FLEET:
            # Allow small fleets very early (only 1–2 owned planets)
            if fleet_size < 5 or len(my_p) >= 3:
                continue
            fleet_size = av  # spend all available if small but early

        chosen = None

        # ── 2a: Nearest idle neutral that isn't fully covered ─────────────────
        for tgt in idle_neu:
            already = incoming.get(tgt.id, 0)
            if already >= tgt.ships + 1:  # already fully covered by in-flight
                continue
            if arc_travel(tgt, ang_vel, src.x, src.y) > MAX_ARC:
                continue
            chosen = tgt
            break

        # ── 2b: Nearest orbiting neutral (skip fast movers) ───────────────────
        if chosen is None:
            for tgt in sorted(neutral_p, key=lambda n: dp(src, n)):
                if is_idle(tgt):
                    continue
                if arc_travel(tgt, ang_vel, src.x, src.y) > MAX_ARC:
                    continue
                already = incoming.get(tgt.id, 0)
                if already >= tgt.ships + 1:
                    continue
                chosen = tgt
                break

        # ── 2c: Best enemy planet ─────────────────────────────────────────────
        if chosen is None and enemy_p:
            def en_val(e):
                d       = dp(src, e)
                t       = travel_turns(d, fleet_size)
                g_est   = e.ships + int(e.production * t)
                already = incoming.get(e.id, 0)
                if already >= g_est * 3:
                    return -1.0
                rem_g = max(1, g_est - already)
                return (e.production * rem_steps / (rem_g + 5)
                        / (1 + 0.02 * t) * 30 / (30 + d))
            best_e = max(enemy_p, key=en_val)
            if en_val(best_e) > 0:
                chosen = best_e

        # ── 2d: Force nearest enemy if nothing else ───────────────────────────
        if chosen is None and enemy_p:
            chosen = min(enemy_p, key=lambda e: dp(src, e))

        if chosen is None:
            continue

        # Cap: don't send more than 2× what's still needed
        d     = dp(src, chosen)
        t     = travel_turns(d, fleet_size)
        g_est = chosen.ships + (int(chosen.production * t) if chosen.owner != -1 else 0)
        still_needed = max(fleet_size, g_est + 1) - incoming.get(chosen.id, 0)
        send_n = int(min(fleet_size, max(fleet_size, still_needed)))

        fire(src, chosen, send_n)

    return moves
