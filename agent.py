"""
Orbit Wars Agent - Competition Submission
=========================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  [id, owner, x, y, radius, ships, production]
Fleet:   [id, owner, x, y, angle, from_planet_id, ships]
Obs attributes: .player, .planets, .fleets, .angular_velocity, .step

Strategy:
  1. Reinforce any threatened own planet.
  2. Capture nearest static neutrals (pool ships from multiple planets).
  3. Attack enemy planets with remaining forces in parallel.
  4. Skip orbiting planets that move too fast to hit reliably.
  5. Consolidate surplus toward the front.
"""

import math

SUN_X, SUN_Y   = 50.0, 50.0
SUN_R          = 10.0
MAX_SPEED      = 6.0
ROTATION_LIMIT = 50.0

RESERVE_FRAC   = 0.15
MIN_HOLD       = 1
DEFEND_NET     = 5
MAX_ARC_TRAVEL = 12.0
HEADING_THRESH = 0.40


def dist2d(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)

def dp(a, b):
    return math.hypot(b["x"] - a["x"], b["y"] - a["y"])

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
    d = dist2d(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0]:
        ang = direct + off
        ex, ey = sx + d * math.cos(ang), sy + d * math.sin(ang)
        if not hits_sun(sx, sy, ex, ey):
            return ang
    return direct

def predict_pos(p, ang_vel, turns):
    orb_r = dist2d(p["x"], p["y"], SUN_X, SUN_Y)
    if orb_r + p["radius"] >= ROTATION_LIMIT:
        return p["x"], p["y"]
    cur_ang = math.atan2(p["y"] - SUN_Y, p["x"] - SUN_X)
    new_ang = cur_ang + ang_vel * turns
    return SUN_X + orb_r * math.cos(new_ang), SUN_Y + orb_r * math.sin(new_ang)

def aim_at(src, tgt, ang_vel, ships):
    tx, ty = tgt["x"], tgt["y"]
    for _ in range(10):
        d_est = dist2d(src["x"], src["y"], tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    return safe_angle(src["x"], src["y"], tx, ty)

def fleet_heading_to(fl, p, ang_vel):
    d = dist2d(fl["x"], fl["y"], p["x"], p["y"])
    t_est = travel_turns(d, fl["ships"])
    pred_x, pred_y = predict_pos(p, ang_vel, t_est)
    ea = math.atan2(pred_y - fl["y"], pred_x - fl["x"])
    diff = abs(math.atan2(math.sin(fl["angle"] - ea), math.cos(fl["angle"] - ea)))
    return diff < HEADING_THRESH

def is_idle(p):
    return dist2d(p["x"], p["y"], SUN_X, SUN_Y) + p["radius"] >= ROTATION_LIMIT

def arc_during_travel(p, ang_vel, src_x, src_y):
    orb_r = dist2d(p["x"], p["y"], SUN_X, SUN_Y)
    if orb_r + p["radius"] >= ROTATION_LIMIT:
        return 0.0
    d = dist2d(src_x, src_y, p["x"], p["y"])
    t_est = travel_turns(d, 10)
    return abs(ang_vel) * orb_r * t_est

def parse_planets(raw):
    return [{"id": int(p[0]), "owner": int(p[1]), "x": float(p[2]), "y": float(p[3]),
             "radius": float(p[4]), "ships": int(p[5]), "prod": int(p[6])} for p in raw]

def parse_fleets(raw):
    return [{"id": int(f[0]), "owner": int(f[1]), "x": float(f[2]), "y": float(f[3]),
             "angle": float(f[4]), "from": int(f[5]), "ships": int(f[6])} for f in raw]


def agent(obs, cfg=None):
    me      = int(obs.player)
    ang_vel = float(getattr(obs, "angular_velocity", 0.033))

    planets = parse_planets(obs.planets)
    fleets  = parse_fleets(obs.fleets)

    my_p      = [p for p in planets if p["owner"] == me]
    neutral_p = [p for p in planets if p["owner"] == -1]
    enemy_p   = [p for p in planets if p["owner"] not in (-1, me)]

    if not my_p:
        return []

    my_fleets    = [fl for fl in fleets if fl["owner"] == me]
    enemy_fleets = [fl for fl in fleets if fl["owner"] != me]

    committed = {}
    moves     = []

    def avail(p):
        hold = max(MIN_HOLD, int(p["ships"] * RESERVE_FRAC))
        return max(0, p["ships"] - committed.get(p["id"], 0) - hold)

    def send_to(src, tgt, n):
        n = int(n)
        if n <= 0:
            return
        ang = aim_at(src, tgt, ang_vel, n)
        committed[src["id"]] = committed.get(src["id"], 0) + n
        moves.append([src["id"], ang, n])
        targeted[tgt["id"]] = targeted.get(tgt["id"], 0) + n

    # Ships already en route — use predicted positions for orbiting planets
    targeted = {}
    for fl in my_fleets:
        for p in neutral_p + enemy_p:
            if fleet_heading_to(fl, p, ang_vel):
                targeted[p["id"]] = targeted.get(p["id"], 0) + fl["ships"]

    # ── Phase 1: Reinforce threatened own planets ──────────────────────────────
    for p in sorted(my_p, key=lambda p: p["ships"]):
        fi = sum(fl["ships"] for fl in my_fleets    if fleet_heading_to(fl, p, ang_vel))
        ei = sum(fl["ships"] for fl in enemy_fleets if fleet_heading_to(fl, p, ang_vel))
        net = p["ships"] + fi - ei
        if net < DEFEND_NET:
            needed = DEFEND_NET * 2 - net
            for d in sorted(my_p, key=lambda q: dp(q, p)):
                if d["id"] == p["id"]:
                    continue
                av = avail(d)
                if av > 0:
                    to_send = min(av, needed)
                    send_to(d, p, to_send)
                    needed -= to_send
                    if needed <= 0:
                        break

    # ── Phase 2: Capture nearest static neutrals ──────────────────────────────
    idle_neutrals = sorted(
        [p for p in neutral_p if is_idle(p)],
        key=lambda p: min(dp(p, m) for m in my_p)
    )
    for tgt in idle_neutrals:
        needed = max(0, tgt["ships"] + 1 - targeted.get(tgt["id"], 0))
        if needed <= 0:
            continue
        for src in sorted(my_p, key=lambda s: dp(s, tgt)):
            to_send = min(avail(src), needed)
            if to_send > 0:
                send_to(src, tgt, to_send)
                needed -= to_send
            if needed <= 0:
                break

    # ── Phase 3: Attack enemy planets ─────────────────────────────────────────
    if enemy_p:
        my_total = sum(p["ships"] for p in my_p)
        en_total = sum(p["ships"] for p in enemy_p)

        def enemy_score(e):
            d = min(dp(e, m) for m in my_p)
            t = travel_turns(d, 20)
            return (e["ships"] + int(e["prod"] * t)) - e["prod"] * 8 + d * 0.3

        for tgt in sorted(enemy_p, key=enemy_score):
            already = targeted.get(tgt["id"], 0)
            d       = min(dp(tgt, m) for m in my_p)
            t       = travel_turns(d, 20)
            needed  = max(0, tgt["ships"] + int(tgt["prod"] * t) + 3 - already)
            if needed <= 0:
                continue

            sources     = sorted(my_p, key=lambda s: dp(s, tgt))
            total_avail = sum(avail(s) for s in sources)
            if total_avail <= 0:
                continue

            can_attack = (total_avail >= needed) or (my_total >= en_total * 1.5)
            if not can_attack:
                continue

            send_cap = min(needed, total_avail)
            for src in sources:
                to_send = min(avail(src), send_cap)
                if to_send > 0:
                    send_to(src, tgt, to_send)
                    send_cap -= to_send
                if send_cap <= 0:
                    break

    # ── Phase 4: Capture nearby orbiting neutrals (skip fast movers) ──────────
    orbiting = sorted(
        [p for p in neutral_p if not is_idle(p)],
        key=lambda p: min(dp(p, m) for m in my_p)
    )
    for tgt in orbiting:
        nearest_src = min(my_p, key=lambda s: dp(s, tgt))
        if arc_during_travel(tgt, ang_vel, nearest_src["x"], nearest_src["y"]) > MAX_ARC_TRAVEL:
            continue

        needed = max(0, tgt["ships"] + 1 - targeted.get(tgt["id"], 0))
        if needed <= 0:
            continue

        sources = sorted(my_p, key=lambda s: dp(s, tgt))
        if sum(avail(s) for s in sources) < needed:
            continue

        for src in sources:
            to_send = min(avail(src), needed)
            if to_send > 0:
                send_to(src, tgt, to_send)
                needed -= to_send
            if needed <= 0:
                break

    # ── Phase 5: Consolidate surplus toward front ─────────────────────────────
    if enemy_p and len(my_p) > 1:
        front = min(my_p, key=lambda p: min(dp(p, e) for e in enemy_p))
        for src in my_p:
            if src["id"] == front["id"]:
                continue
            av = avail(src)
            if av >= 25:
                send_to(src, front, av // 2)

    return moves
