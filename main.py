"""
Orbit Wars – Competitive Agent v11
====================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

v11 improvements over v10:
  * planet_size(): classify planets as small / medium / large by radius.
  * Quick-capture pass (Phase 1b): planets needing ≤ QUICK_SHIPS ships and
    reachable in ≤ QUICK_STEPS turns are blitzed every step until captured.
  * Rotating-planet timing: closest_approach_timing() scans the next 60 steps
    to find the minimum-distance window; fleets wait until that window opens.
  * Multi-planet rotating coordination (Phase 3): when a rotating target needs
    more ships than any single planet can provide, nearby planets pool ships.
  * Bridgehead strategy: neutrals on the direct path to a far enemy receive a
    value bonus, building stepping-stone chains toward distant opponents.
  * 4-player awareness: when ≥ 3 factions exist, idle neutrals are saturated
    first; side opponents (~90°) are scored higher than polar ones (~180°).
  * Bug fix: send_n cap in Phase 2 now correctly limits to still_needed.
  * Tuned constants throughout.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

SUN_X, SUN_Y   = 50.0, 50.0
SUN_R          = 10.0
MAX_SPEED      = 6.0
ROTATION_LIMIT = 50.0
TOTAL_STEPS    = 500

SEND_FRAC  = 0.55   # send 55 % of available ships each turn (was 0.50)
MIN_FLEET  = 10     # main offensive minimum fleet size (was 15)
MAX_ARC    = 14.0   # skip orbiting target if arc exceeds this (was 12.0)
HEADING_T  = 0.40
DEFEND_NET = 8      # reinforce if net projected ships < this (was 5)
INCOMING_T = 0.85

OPENING_STEPS    = 90
OPENING_RESERVE  = 1
OPENING_MIN_SEND = 4    # was 5
OPENING_MAX_ARC  = 20.0  # was 18.0

EXPAND_STEPS          = 260   # was 220
EXPAND_RESERVE        = 1     # was 2
EXPAND_MIN_SEND       = 2     # was 3
EXPAND_MAX_PER_PLANET = 4     # was 3
ROTATING_SAVE_RESERVE = 1
EVAC_RESERVE          = 0
ROTATING_NEAR_FACTOR  = 0.75
ROTATING_NEAR_MARGIN  = 8.0
FAST_ROTATING_TURNS   = 20.0  # was 18.0
SAME_TIME_MARGIN      = 4.0

# Quick-capture: blitz cheap nearby planets every step until captured
QUICK_SHIPS       = 20
QUICK_STEPS       = 5
QUICK_MAX_PER_SRC = 3     # max quick-capture targets fired per source per turn

# Finisher / micro-wave
FINISHER_SHIPS = 4     # target with ≤ this remaining need gets a finisher fleet
WAVE_FRAC      = 0.28  # send fraction for secondary micro-wave pass

# Rotating-planet timing: fire when fleet meets tgt near its closest pass
CLOSEST_APPROACH_MARGIN = 6.0   # steps of headroom before closest point
HIT_MARGIN              = 3.0   # acceptable aimed-point miss beyond planet radius

# Bridgehead: neutrals on the path to a far enemy get a value bonus
BRIDGE_ANGLE_THRESH = 0.35   # radians; neutral within this cone = on-path
BRIDGE_BONUS        = 40.0   # additive value boost for bridge neutrals

# Planet size thresholds (radius)
SMALL_RADIUS = 3.0
LARGE_RADIUS = 6.0

# 4-player tuning
SIDE_ENEMY_BONUS    = 1.35   # multiplier for ~90° side opponents
POLAR_ENEMY_PENALTY = 0.70   # multiplier for ~180° polar opponents

# Early-game aggression
FIRST_CAPTURE_DEADLINE = 33   # force-capture 2nd planet by this step
EARLY_STEPS            = 55   # lighter reserves and looser rotating timing before this step
EARLY_WAIT_THRESH      = 8.0  # max closest-approach wait when step < EARLY_STEPS

# Phase-based reserve tuning
HIGH_PROD_THRESHOLD  = 4     # production >= this → core planet
HIGH_PROD_EXTRA      = 8     # extra ships held on core planets (mid/late only)

# Frontline / backline
FRONTLINE_DIST        = 25.0  # distance to nearest enemy below which = frontline
FRONTLINE_SEND_FRAC   = 0.35  # send fraction for frontline planets (vs SEND_FRAC)
FRONTLINE_RESERVE_MULT = 1.5  # reserve multiplier for frontline planets

# Source safety thresholds by move type
SAFETY_EXPAND_THRESH = 0.40   # looser: expanding to a neutral
SAFETY_ATTACK_THRESH = 0.65   # tighter: attacking an enemy planet

# Endgame
ENDGAME_STEPS   = 400   # all-in mode: use avail() not safe_avail(), allow_unsafe everywhere
LATE_GAME_STEPS = 350   # prefer enemy planets over neutrals from this step on
FINAL_STEPS     = 460   # drain pass: send all idle ships to any reachable target

# Target selection
ETA_SCORE_WEIGHT     = 0.8    # penalty per turn of capture ETA in idle planet scoring
OPPORTUNISTIC_SHIPS  = 15     # attack enemy planets with ≤ this many ships before expanding
BRIDGE_MIN_PLANETS = 2      # enable bridgehead bonus only once we own this many planets
COMET_ARC_LIMIT    = 10.0   # skip low-production rotating planets with arc above this
EXPAND_WAR_MAX     = 1      # max expansion launches per planet when under_attack

# Opponent style detection
SWARM_STYLE_FLEETS     = 3     # enemy fleet count at or above this → swarm classifier
SWARM_STYLE_SIZE       = 20    # avg enemy fleet size at or below this → swarm classifier
AGGRESSIVE_EXPAND_RATE = 0.07  # enemy_planets / step above this → aggressive classifier
TURTLE_SHIP_THRESH     = 40    # avg ships per enemy planet above this → turtle classifier

# Defense reactivity
DEFENSE_ETA_HORIZON   = 30    # only count inbound fleets arriving within this many turns
SWARM_FLEET_COUNT     = 2     # ≥ this many enemy fleets inbound → swarm threat
SWARM_RESERVE_MULT    = 1.4   # global reserve multiplier under swarm attack
WAR_MODE_DIST         = 20.0  # nearest enemy planet within this distance → war mode
WAR_MODE_RESERVE_MULT = 1.25  # reserve multiplier in war mode


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
    for off in [0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 1.0, -1.0, 1.3, -1.3]:
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

def aim_valid(src, tgt, ang_vel, ships):
    """
    Return (angle, hit_ok).  15 iterations for better convergence.
    hit_ok is False when the aimed point misses a rotating target by > HIT_MARGIN.
    Always True for idle targets (they don't move).
    """
    tx, ty = tgt.x, tgt.y
    d_est = dp(src, tgt)
    for _ in range(15):
        d_est = dist(src.x, src.y, tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    angle = safe_angle(src.x, src.y, tx, ty)
    if is_idle(tgt):
        return angle, True
    aimed_x = src.x + d_est * math.cos(angle)
    aimed_y = src.y + d_est * math.sin(angle)
    miss = dist(aimed_x, aimed_y, tx, ty)
    return angle, miss <= tgt.radius + HIT_MARGIN

def aim_at(src, tgt, ang_vel, ships):
    """Iteratively converge on the intercept angle for a moving target."""
    angle, _ = aim_valid(src, tgt, ang_vel, ships)
    return angle

def fleet_heading_to(fl, p, ang_vel):
    """True if fleet is heading toward p's predicted future position."""
    d = dist(fl.x, fl.y, p.x, p.y)
    t_est = travel_turns(d, fl.ships)
    pred_x, pred_y = predict_pos(p, ang_vel, t_est)
    ea = math.atan2(pred_y - fl.y, pred_x - fl.x)
    diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
    return diff < HEADING_T

def fleet_target(fl, targets, ang_vel):
    """Best current target match for an in-flight fleet."""
    best = None
    best_diff = INCOMING_T
    for p in targets:
        d = dist(fl.x, fl.y, p.x, p.y)
        t_est = travel_turns(d, fl.ships)
        pred_x, pred_y = predict_pos(p, ang_vel, t_est)
        ea = math.atan2(pred_y - fl.y, pred_x - fl.x)
        diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
        if diff < best_diff:
            best = p
            best_diff = diff
    return best

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

def neutral_value(src, tgt, ang_vel, av):
    """Opening value: production matters more than raw closeness."""
    d = dp(src, tgt)
    arc = arc_travel(tgt, ang_vel, src.x, src.y)
    if arc > OPENING_MAX_ARC:
        return -1.0
    need = tgt.ships + 1
    batches = max(1.0, need / max(OPENING_MIN_SEND, av))
    finish_bonus = 2.4 if need <= av else 1.0
    return (tgt.production * tgt.production * 120.0 * finish_bonus
            / (tgt.ships + 8.0 + 0.55 * d + 0.7 * arc)
            / (batches * batches))

def target_need(tgt, incoming):
    return tgt.ships + 1 - incoming.get(tgt.id, 0)

def idle_capture_value(src, tgt):
    return tgt.production * 100.0 - tgt.ships * 4.0 - dp(src, tgt)

def rotating_capture_value(src, tgt, ang_vel):
    arc = arc_travel(tgt, ang_vel, src.x, src.y)
    if arc > OPENING_MAX_ARC:
        return -1e9
    return tgt.production * 80.0 - tgt.ships * 3.0 - dp(src, tgt) - arc * 2.0

def rotating_front_score(src, tgt, ang_vel):
    arc = arc_travel(tgt, ang_vel, src.x, src.y)
    if arc > OPENING_MAX_ARC:
        return -1e9
    return tgt.production * tgt.production * 90.0 / (tgt.ships + 5.0 + 0.35 * dp(src, tgt) + 0.45 * arc)

def incoming_counts(p, my_fleets, enemy_fleets, ang_vel):
    fi = sum(fl.ships for fl in my_fleets if fleet_heading_to(fl, p, ang_vel))
    ei = sum(fl.ships for fl in enemy_fleets if fleet_heading_to(fl, p, ang_vel))
    return fi, ei

def capture_eta(src, tgt, ships):
    if ships <= 0:
        return 1e9
    return travel_turns(dp(src, tgt), ships)

def enemy_need(src, tgt, ships, incoming):
    t = travel_turns(dp(src, tgt), max(1, ships))
    return tgt.ships + int(tgt.production * t) + 1 - incoming.get(tgt.id, 0)


# ── planet meta ───────────────────────────────────────────────────────────────

def planet_size(p):
    """Classify planet as 'small', 'medium', or 'large' by radius."""
    if p.radius <= SMALL_RADIUS:
        return 'small'
    if p.radius <= LARGE_RADIUS:
        return 'medium'
    return 'large'


# ── rotating-planet timing ────────────────────────────────────────────────────

def closest_approach_timing(src, tgt, ang_vel, ships):
    """
    Return turns to wait before firing so the fleet meets tgt near its
    closest orbital pass to src.  Scans the next 60 steps with predict_pos
    to find the minimum-distance moment, then returns
    max(0, closest_step - travel_time - CLOSEST_APPROACH_MARGIN).
    Returns 0 for idle planets or when firing now is already near-optimal.
    """
    if is_idle(tgt) or abs(ang_vel) < 1e-9:
        return 0.0
    best_w, best_d = 0, 1e9
    for w in range(61):
        tx, ty = predict_pos(tgt, ang_vel, w)
        d_w = dist(src.x, src.y, tx, ty)
        if d_w < best_d:
            best_d = d_w
            best_w = w
    travel_t = travel_turns(best_d, max(1, int(ships)))
    wait = best_w - travel_t - CLOSEST_APPROACH_MARGIN
    return max(0.0, wait)


# ── bridgehead helper ─────────────────────────────────────────────────────────

def is_bridge(src_p, neutral, enemy_p):
    """True when neutral lies within BRIDGE_ANGLE_THRESH of the direct path src_p→enemy_p."""
    direct_ang = math.atan2(enemy_p.y - src_p.y, enemy_p.x - src_p.x)
    neut_ang   = math.atan2(neutral.y  - src_p.y, neutral.x  - src_p.x)
    diff = abs(math.atan2(math.sin(neut_ang - direct_ang), math.cos(neut_ang - direct_ang)))
    return diff < BRIDGE_ANGLE_THRESH and dp(src_p, neutral) < dp(src_p, enemy_p)


# ── main agent ────────────────────────────────────────────────────────────────

def agent(obs):
    g = lambda k, d: (obs.get(k, d) if isinstance(obs, dict) else getattr(obs, k, d))
    me   = int(g("player", 0))
    step = int(g("step", 0))
    # Try snake_case first, then camelCase, then a sane default
    _av = g("angular_velocity", None)
    if _av is None:
        _av = g("angularVelocity", 0.033)
    ang_vel = float(_av)

    planets = [Planet(*p) for p in g("planets", [])]
    fleets  = [Fleet(*f)  for f in g("fleets",  [])]

    my_p      = [p for p in planets if p.owner == me]
    neutral_p = [p for p in planets if p.owner == -1]
    enemy_p   = [p for p in planets if p.owner not in (-1, me)]

    if not my_p:
        return []

    my_fleets    = [fl for fl in fleets if fl.owner == me]
    enemy_fleets = [fl for fl in fleets if fl.owner != me]

    factions    = set(p.owner for p in planets if p.owner >= 0)
    four_player = len(factions) >= 3

    moves     = []
    committed = {}   # planet_id → ships committed this turn
    incoming  = {}   # ships already heading to each non-owned planet

    def avail(p):
        return max(0, p.ships - committed.get(p.id, 0))

    _inbound_enemy_fleet_count = sum(
        1 for fl in enemy_fleets
        for p2 in my_p if fleet_heading_to(fl, p2, ang_vel)
    )
    under_swarm  = _inbound_enemy_fleet_count >= SWARM_FLEET_COUNT
    under_attack = (
        _inbound_enemy_fleet_count > 0
        or (bool(enemy_p) and bool(my_p)
            and min(dp(p2, e) for p2 in my_p for e in enemy_p) < WAR_MODE_DIST)
    )

    # ── Opponent style detection ───────────────────────────────────────────────
    _opp_fleet_n   = len(enemy_fleets)
    _opp_avg_size  = sum(fl.ships for fl in enemy_fleets) / max(1, _opp_fleet_n)
    _opp_avg_ships = sum(e.ships  for e in enemy_p)       / max(1, len(enemy_p))
    opp_is_swarm = (
        _opp_fleet_n >= SWARM_STYLE_FLEETS and _opp_avg_size <= SWARM_STYLE_SIZE
    )
    opp_is_aggressive = bool(enemy_p) and (
        len(enemy_p) / max(1, step) > AGGRESSIVE_EXPAND_RATE
        or (under_attack and step < EXPAND_STEPS // 2)
    )
    opp_is_turtle = bool(enemy_p) and (
        _opp_avg_ships >= TURTLE_SHIP_THRESH
        and _opp_fleet_n < max(1, len(enemy_p))
    )

    def enemy_pressure_on(p, horizon=25):
        pressure = 0
        for e in enemy_p:
            d = dp(e, p)
            possible_send = max(0, e.ships - 5)
            eta = travel_turns(d, max(1, possible_send))
            if eta <= horizon:
                pressure += possible_send * (1.0 - eta / horizon)
        return pressure

    def reserve_for(p):
        if step < EARLY_STEPS:
            base = 2 + p.production
        elif step < EXPAND_STEPS:
            base = 4 + p.production
            if p.production >= HIGH_PROD_THRESHOLD:
                base += HIGH_PROD_EXTRA
        else:
            base = 5 + p.production * 2
            if p.production >= HIGH_PROD_THRESHOLD:
                base += HIGH_PROD_EXTRA
        if enemy_p and min(dp(p, e) for e in enemy_p) < FRONTLINE_DIST:
            base = int(base * FRONTLINE_RESERVE_MULT)
        if under_swarm:
            base = int(base * SWARM_RESERVE_MULT)
        elif under_attack:
            base = int(base * WAR_MODE_RESERVE_MULT)
        return int(base + 0.45 * enemy_pressure_on(p))

    def safe_avail(p):
        return max(0, p.ships - committed.get(p.id, 0) - reserve_for(p))

    def incoming_enemy_pressure_on(p):
        return sum(fl.ships for fl in enemy_fleets if fleet_heading_to(fl, p, ang_vel))

    def eta_incoming_counts(p):
        """Like incoming_counts but only fleets arriving within DEFENSE_ETA_HORIZON turns."""
        fi = sum(
            fl.ships for fl in my_fleets
            if fleet_heading_to(fl, p, ang_vel)
            and travel_turns(dist(fl.x, fl.y, p.x, p.y), fl.ships) <= DEFENSE_ETA_HORIZON
        )
        ei = sum(
            fl.ships for fl in enemy_fleets
            if fleet_heading_to(fl, p, ang_vel)
            and travel_turns(dist(fl.x, fl.y, p.x, p.y), fl.ships) <= DEFENSE_ETA_HORIZON
        )
        return fi, ei

    def source_safe_after_send(src, send_n, safety_thresh=0.55):
        remaining = src.ships - committed.get(src.id, 0) - send_n
        nearby_enemy_power = incoming_enemy_pressure_on(src)
        for fl in enemy_fleets:
            if not fleet_heading_to(fl, src, ang_vel):
                d = dist(fl.x, fl.y, src.x, src.y)
                if d < 20:
                    eta = travel_turns(d, fl.ships)
                    nearby_enemy_power += fl.ships * max(0.0, 1.0 - eta / 20) * 0.5
        for e in enemy_p:
            possible = max(0, e.ships - 5)
            if possible <= 0:
                continue
            eta = travel_turns(dp(e, src), possible)
            if eta <= 20:
                nearby_enemy_power += possible * (1.0 - eta / 20)
        future_defense = remaining + src.production * 8
        return future_defense >= nearby_enemy_power * safety_thresh

    def fire(src, tgt, n, allow_unsafe=False, safety_thresh=0.55):
        n = int(n)
        if n <= 0:
            return False
        if not allow_unsafe and not source_safe_after_send(src, n, safety_thresh):
            return False
        ang, hit_ok = aim_valid(src, tgt, ang_vel, n)
        if not hit_ok:
            return False  # shot would miss the rotating target
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])
        incoming[tgt.id] = incoming.get(tgt.id, 0) + n
        return True

    # Build incoming: existing in-flight fleets to non-owned planets
    for fl in my_fleets:
        tgt = fleet_target(fl, neutral_p + enemy_p, ang_vel)
        if tgt is not None:
            incoming[tgt.id] = incoming.get(tgt.id, 0) + fl.ships

    # Precompute bridge status once: neutral ids that lie on the path to an enemy
    bridge_ids = set()
    if enemy_p and len(my_p) >= BRIDGE_MIN_PLANETS:
        for m in my_p:
            for n in neutral_p:
                for e in enemy_p:
                    if is_bridge(m, n, e):
                        bridge_ids.add(n.id)
                        break

    threatened = {}

    # ── Phase 1: Emergency defense / evacuation ───────────────────────────────
    for p in my_p:
        fi, ei = eta_incoming_counts(p)
        net = p.ships + fi - ei
        threatened[p.id] = ei > fi and net < DEFEND_NET

        if ei > fi and net <= 0:
            av = avail(p) - EVAC_RESERVE
            if av > 0:
                if not is_idle(p):
                    rotating_neu = [n for n in neutral_p if not is_idle(n) and target_need(n, incoming) > 0]
                    idle_neu_ev  = [n for n in neutral_p if is_idle(n)       and target_need(n, incoming) > 0]
                    friendly_ev  = [q for q in my_p if q.id != p.id]
                    evac_tgt = (
                        min(rotating_neu,  key=lambda n: dp(p, n)) if rotating_neu  else
                        min(idle_neu_ev,   key=lambda n: dp(p, n)) if idle_neu_ev   else
                        min(friendly_ev,   key=lambda q: dp(p, q)) if friendly_ev   else None
                    )
                    if evac_tgt is not None:
                        fire(p, evac_tgt, av, allow_unsafe=True)
                    continue
                else:
                    bigger_idle = [
                        q for q in my_p
                        if q.id != p.id and is_idle(q) and q.production > p.production
                    ]
                    if bigger_idle:
                        fire(p, min(bigger_idle, key=lambda q: dp(p, q)), av, allow_unsafe=True)
                    else:
                        fallback = [q for q in my_p if q.id != p.id]
                        if fallback:
                            fire(p, min(fallback, key=lambda q: dp(p, q)), av, allow_unsafe=True)

            if is_idle(p):
                needed = max(1, ei - p.ships - fi + 1)
                for d in sorted([q for q in my_p if q.id != p.id], key=lambda q: dp(q, p)):
                    av = avail(d)
                    if av <= 0:
                        continue
                    to_send = min(av, needed)
                    fire(d, p, to_send, allow_unsafe=True)
                    needed -= to_send
                    if needed <= 0:
                        break
            continue

        if ei > fi and net < DEFEND_NET:
            needed = DEFEND_NET * 2 - net
            for d in sorted(my_p, key=lambda q: dp(q, p)):
                if d.id == p.id:
                    continue
                av = avail(d)
                if av > 0:
                    to_send = min(av, needed)
                    fire(d, p, to_send, allow_unsafe=True)
                    needed -= to_send
                    if needed <= 0:
                        break

    # ── Phase 1b: Quick capture ───────────────────────────────────────────────
    # Blitz any neutral needing ≤ QUICK_SHIPS ships reachable in ≤ QUICK_STEPS
    # turns.  Called every step so ships keep flowing until the planet falls.
    for src in sorted(my_p, key=lambda p: -safe_avail(p)):
        av = safe_avail(src)
        if av <= 0:
            continue
        quick_hits = []
        for tgt in neutral_p:
            need = target_need(tgt, incoming)
            if need <= 0 or need > QUICK_SHIPS:
                continue
            eta = capture_eta(src, tgt, max(1, need))
            if eta <= QUICK_STEPS:
                quick_hits.append((eta, need, tgt))
        quick_hits.sort()
        q_count = 0
        for eta, need, tgt in quick_hits:
            if q_count >= QUICK_MAX_PER_SRC:
                break
            av = safe_avail(src)
            send_n = min(av, need)
            if send_n > 0 and fire(src, tgt, send_n, safety_thresh=SAFETY_EXPAND_THRESH):
                q_count += 1

    # ── Phase 1c: Finisher pass ───────────────────────────────────────────────
    # For any planet already receiving committed ships but still needing ≤
    # finisher_limit more, find the nearest source and send a tiny finisher.
    finisher_limit = FINISHER_SHIPS * 2 if opp_is_swarm else FINISHER_SHIPS
    for tgt in sorted(neutral_p + enemy_p,
                      key=lambda t: target_need(t, incoming)):
        need = target_need(tgt, incoming)
        if need <= 0 or need > finisher_limit:
            continue
        if incoming.get(tgt.id, 0) == 0:
            continue  # only top-up attacks already in flight
        for src in sorted(my_p, key=lambda p: dp(p, tgt)):
            av = avail(src) - 1
            send_n = min(av, need)
            if send_n > 0 and fire(src, tgt, send_n, allow_unsafe=True):
                need -= send_n
            if need <= 0:
                break

    # ── Single-planet opener ──────────────────────────────────────────────────
    if step < OPENING_STEPS and len(my_p) == 1 and neutral_p:
        src = my_p[0]
        # Deadline: must own a 2nd planet by FIRST_CAPTURE_DEADLINE — force fire
        if step >= FIRST_CAPTURE_DEADLINE:
            av_dl = avail(src) - 1
            in_prog = sorted(
                [n for n in neutral_p if incoming.get(n.id, 0) > 0 and target_need(n, incoming) > 0],
                key=lambda n: target_need(n, incoming)
            )
            dl_tgt = in_prog[0] if in_prog else min(neutral_p, key=lambda n: dp(src, n))
            dl_need = target_need(dl_tgt, incoming)
            if dl_need > 0 and av_dl > 0:
                fire(src, dl_tgt, min(av_dl, dl_need), allow_unsafe=True)
        av = safe_avail(src)
        if av >= OPENING_MIN_SEND:
            idle_targets     = [n for n in neutral_p if     is_idle(n) and target_need(n, incoming) > 0]
            rotating_targets = [n for n in neutral_p if not is_idle(n) and target_need(n, incoming) > 0]
            nearest_idle = min(idle_targets,     key=lambda n: dp(src, n), default=None)
            nearest_rot  = min(rotating_targets, key=lambda n: dp(src, n), default=None)

            if is_idle(src):
                chosen = None
                if nearest_idle is not None and nearest_rot is not None:
                    idle_d = dp(src, nearest_idle)
                    rot_d  = dp(src, nearest_rot)
                    if rot_d + ROTATING_NEAR_MARGIN < idle_d or rot_d < idle_d * ROTATING_NEAR_FACTOR:
                        chosen = nearest_rot
                    else:
                        chosen = nearest_idle
                else:
                    chosen = nearest_idle or nearest_rot

                if chosen is not None:
                    need   = target_need(chosen, incoming)
                    send_n = min(av, need)
                    if send_n >= EXPAND_MIN_SEND or send_n >= need:
                        fire(src, chosen, send_n)
            else:
                rot_need  = target_need(nearest_rot,  incoming) if nearest_rot  is not None else 0
                idle_need = target_need(nearest_idle, incoming) if nearest_idle is not None else 0
                rot_eta   = capture_eta(src, nearest_rot,  rot_need)  if nearest_rot  is not None and rot_need  <= av else 1e9
                idle_eta  = capture_eta(src, nearest_idle, idle_need) if nearest_idle is not None and idle_need <= av else 1e9

                if nearest_rot is not None and rot_eta <= FAST_ROTATING_TURNS:
                    if nearest_idle is not None and idle_eta <= rot_eta + SAME_TIME_MARGIN:
                        fire(src, nearest_idle, min(av, idle_need))
                        av = safe_avail(src)
                        if av > 0:
                            need   = target_need(nearest_rot, incoming)
                            send_n = min(av, need)
                            if send_n >= OPENING_MIN_SEND or send_n >= need:
                                fire(src, nearest_rot, send_n)
                    else:
                        fire(src, nearest_rot, min(av, rot_need))
                elif nearest_idle is not None:
                    need   = target_need(nearest_idle, incoming)
                    send_n = min(av, need)
                    if send_n >= OPENING_MIN_SEND or send_n >= need:
                        fire(src, nearest_idle, send_n)

    # ── Early economy roles ───────────────────────────────────────────────────
    if step < EXPAND_STEPS and neutral_p:
        if opp_is_swarm:
            effective_max = 0                   # vs swarm: stop expanding, defend and counter
        elif opp_is_aggressive or under_attack:
            effective_max = EXPAND_WAR_MAX      # vs aggressive: one launch max
        else:
            effective_max = EXPAND_MAX_PER_PLANET
        for src in sorted(my_p, key=lambda p: (is_idle(p), p.production, avail(p)), reverse=True):
            launched = 0
            while launched < effective_max:
                if is_idle(src):
                    av = safe_avail(src)
                    if av < EXPAND_MIN_SEND:
                        break

                    idle_targets     = []
                    rotating_targets = []
                    enemy_targets    = []
                    for tgt in neutral_p:
                        need = target_need(tgt, incoming)
                        if need <= 0:
                            continue
                        bridge = BRIDGE_BONUS if tgt.id in bridge_ids else 0.0
                        if is_idle(tgt):
                            eta = capture_eta(src, tgt, max(1, need))
                            prod_w = 150.0 if step < EARLY_STEPS else 100.0
                            _icv = tgt.production * prod_w - tgt.ships * 4.0 - eta * ETA_SCORE_WEIGHT
                            idle_targets.append((eta, -(_icv + bridge), tgt.id, tgt, need))
                        else:
                            arc = arc_travel(tgt, ang_vel, src.x, src.y)
                            if arc > COMET_ARC_LIMIT and tgt.production < HIGH_PROD_THRESHOLD:
                                continue
                            score = rotating_capture_value(src, tgt, ang_vel)
                            if score > -1e8:
                                front_score = rotating_front_score(src, tgt, ang_vel) + bridge
                                rotating_targets.append((-score, -front_score, dp(src, tgt), tgt.id, tgt, need))
                    for tgt in enemy_p:
                        need = enemy_need(src, tgt, av, incoming)
                        if need <= 0:
                            continue
                        d = dp(src, tgt)
                        score = tgt.production * 65.0 / (need + 8.0) / (1.0 + 0.025 * d)
                        enemy_targets.append((-score, d, tgt.id, tgt, need))

                    should_take_rotating_first = False
                    if idle_targets and rotating_targets:
                        nearest_idle_d = min(item[0] for item in idle_targets)
                        nearest_rot_d  = min(item[2] for item in rotating_targets)
                        should_take_rotating_first = (
                            nearest_rot_d + ROTATING_NEAR_MARGIN < nearest_idle_d
                            or nearest_rot_d < nearest_idle_d * ROTATING_NEAR_FACTOR
                        )

                    chosen = None
                    need   = 0

                    rot_wait_thresh = EARLY_WAIT_THRESH if step < OPENING_STEPS else 1.0
                    if should_take_rotating_first:
                        # Pick best rotating that is within its timing window
                        for item in sorted(rotating_targets):
                            _, _, _, _, cand, cand_need = item
                            wait = closest_approach_timing(src, cand, ang_vel, min(av, cand_need))
                            if wait <= rot_wait_thresh:
                                chosen, need = cand, cand_need
                                break
                        # Fall back to idle if no rotating target is ready
                        if chosen is None and idle_targets:
                            _, _, _, chosen, need = min(idle_targets)
                    elif idle_targets:
                        _, _, _, chosen, need = min(idle_targets)
                    elif rotating_targets:
                        for item in sorted(rotating_targets):
                            _, _, _, _, cand, cand_need = item
                            wait = closest_approach_timing(src, cand, ang_vel, min(av, cand_need))
                            if wait <= rot_wait_thresh:
                                chosen, need = cand, cand_need
                                break
                    elif enemy_targets:
                        _, _, _, chosen, need = min(enemy_targets)

                    if chosen is None:
                        fb = [n for n in neutral_p if target_need(n, incoming) > 0]
                        if fb:
                            chosen = min(fb, key=lambda n: dp(src, n))
                            need = target_need(chosen, incoming)
                        else:
                            break

                    send_n = min(av, need)
                    if send_n < EXPAND_MIN_SEND and send_n < need:
                        break
                    if fire(src, chosen, send_n):
                        launched += 1
                    else:
                        break
                else:
                    av = safe_avail(src)
                    if av < OPENING_MIN_SEND:
                        break

                    claimed_idle  = []
                    high_idle     = []
                    high_rotating = []
                    high_enemy    = []
                    for tgt in neutral_p:
                        need = target_need(tgt, incoming)
                        if need <= 0:
                            continue
                        bridge = BRIDGE_BONUS if tgt.id in bridge_ids else 0.0
                        if is_idle(tgt):
                            eta = capture_eta(src, tgt, max(1, need))
                            prod_w = 150.0 if step < EARLY_STEPS else 100.0
                            _icv = tgt.production * prod_w - tgt.ships * 4.0 - eta * ETA_SCORE_WEIGHT
                            item = (eta, -(_icv + bridge), tgt.id, tgt, need)
                            if incoming.get(tgt.id, 0) > 0:
                                claimed_idle.append(item)
                            high_idle.append((-(_icv + bridge), eta, tgt.id, tgt, need))
                        else:
                            arc = arc_travel(tgt, ang_vel, src.x, src.y)
                            if arc > COMET_ARC_LIMIT and tgt.production < HIGH_PROD_THRESHOLD:
                                continue
                            score = rotating_front_score(src, tgt, ang_vel) + bridge
                            if score > -1e8:
                                high_rotating.append((-score, dp(src, tgt), tgt.id, tgt, need))
                    for tgt in enemy_p:
                        need = enemy_need(src, tgt, av, incoming)
                        if need <= 0:
                            continue
                        d = dp(src, tgt)
                        score = tgt.production * 70.0 / (need + 8.0) / (1.0 + 0.025 * d)
                        high_enemy.append((-score, d, tgt.id, tgt, need))

                    chosen = None
                    need   = 0
                    if claimed_idle:
                        _, _, _, chosen, need = min(claimed_idle)
                    elif launched % 2 == 0 and high_idle:
                        _, _, _, chosen, need = min(high_idle)
                    elif high_rotating:
                        rot_wait_thresh = EARLY_WAIT_THRESH if step < OPENING_STEPS else 1.0
                        for item in sorted(high_rotating):
                            _, _, _, cand, cand_need = item
                            wait = closest_approach_timing(src, cand, ang_vel, min(av, cand_need))
                            if wait <= rot_wait_thresh:
                                chosen, need = cand, cand_need
                                break
                        if chosen is None and high_idle:
                            _, _, _, chosen, need = min(high_idle)
                    elif high_idle:
                        _, _, _, chosen, need = min(high_idle)
                    elif high_enemy:
                        _, _, _, chosen, need = min(high_enemy)

                    if chosen is None:
                        fb = [n for n in neutral_p if target_need(n, incoming) > 0]
                        if fb:
                            chosen = min(fb, key=lambda n: dp(src, n))
                            need = target_need(chosen, incoming)
                        else:
                            break
                    send_n = min(av, need)
                    if send_n < OPENING_MIN_SEND and send_n < need:
                        break
                    if fire(src, chosen, send_n):
                        launched += 1
                    else:
                        break

    # ── Phase 2: Per-planet offensive ─────────────────────────────────────────
    rem_steps = max(1, TOTAL_STEPS - step)
    endgame   = step >= ENDGAME_STEPS
    late_game = (
        step >= LATE_GAME_STEPS
        or (opp_is_turtle     and step >= LATE_GAME_STEPS // 2)
        or (opp_is_aggressive and step >= LATE_GAME_STEPS * 2 // 3)
    )

    # Idle neutrals sorted by fastest-capture ETA from any owned planet (bridge targets bumped up)
    idle_neu = sorted(
        [n for n in neutral_p if is_idle(n)],
        key=lambda n: (
            min(capture_eta(m, n, max(1, target_need(n, incoming))) for m in my_p)
            - (BRIDGE_BONUS * 0.4 if n.id in bridge_ids and len(my_p) >= BRIDGE_MIN_PLANETS else 0.0)
        )
    )

    for src in sorted(my_p, key=lambda p: -(avail(p) if endgame else safe_avail(p))):
        av = avail(src) if endgame else safe_avail(src)
        if endgame:
            fleet_size = av
            if fleet_size < 2:
                continue
        else:
            send_frac = FRONTLINE_SEND_FRAC if (enemy_p and min(dp(src, e) for e in enemy_p) < FRONTLINE_DIST) else SEND_FRAC
            fleet_size = int(av * send_frac)
            if fleet_size < MIN_FLEET:
                if fleet_size < 5 or len(my_p) >= 3:
                    continue
                fleet_size = av

        chosen = None

        # ── 2_opp: Kill any exposed enemy planet before expanding ─────────────
        for e in sorted(enemy_p, key=lambda x: x.ships):
            if e.ships > OPPORTUNISTIC_SHIPS:
                break
            if incoming.get(e.id, 0) >= e.ships + 1:
                continue
            need_n = enemy_need(src, e, fleet_size, incoming)
            if 0 < need_n <= fleet_size:
                chosen = e
                break

        # ── 2a: Nearest idle neutral not fully covered ────────────────────────
        if chosen is None and not (late_game and enemy_p):
            for tgt in idle_neu:
                already = incoming.get(tgt.id, 0)
                if already >= tgt.ships + 1:
                    continue
                if arc_travel(tgt, ang_vel, src.x, src.y) > MAX_ARC:
                    continue
                if capture_eta(src, tgt, fleet_size) >= rem_steps:
                    continue
                chosen = tgt
                break

        # ── 2b: Nearest orbiting neutral (closest-approach gated) ─────────────
        if chosen is None and not (late_game and enemy_p):
            for tgt in sorted(neutral_p, key=lambda n: dp(src, n)):
                if is_idle(tgt):
                    continue
                if arc_travel(tgt, ang_vel, src.x, src.y) > MAX_ARC:
                    continue
                already = incoming.get(tgt.id, 0)
                if already >= tgt.ships + 1:
                    continue
                if capture_eta(src, tgt, fleet_size) >= rem_steps:
                    continue
                wait = closest_approach_timing(src, tgt, ang_vel, fleet_size)
                if wait > 1.0:
                    continue  # not at closest approach yet; try next candidate
                chosen = tgt
                break

        # ── 2c: Best enemy planet (4-player-aware) ────────────────────────────
        if chosen is None and enemy_p:
            # In 4-player, keep expanding neutrals before attacking enemies;
            # but once under_attack, switch to war mode immediately.
            if not (four_player and len(neutral_p) > len(my_p) and not under_attack):
                def en_val(e):
                    d       = dp(src, e)
                    t       = travel_turns(d, fleet_size)
                    if t >= rem_steps - 5:
                        return -1.0  # won't arrive before game ends
                    g_est   = e.ships + int(e.production * t)
                    already = incoming.get(e.id, 0)
                    if already >= g_est + 1:
                        return -1.0
                    rem_g = max(1, g_est - already)
                    base  = (e.production * rem_steps / (rem_g + 5)
                             / (1 + 0.02 * t) * 30 / (30 + d))
                    if four_player:
                        src_ang  = math.atan2(src.y - SUN_Y, src.x - SUN_X)
                        e_ang    = math.atan2(e.y   - SUN_Y, e.x   - SUN_X)
                        ang_diff = abs(math.atan2(math.sin(e_ang - src_ang),
                                                  math.cos(e_ang - src_ang)))
                        if ang_diff < math.pi * 0.67:   # side opponent
                            base *= SIDE_ENEMY_BONUS
                        else:                            # polar opponent
                            base *= POLAR_ENEMY_PENALTY
                    return base
                best_e = max(enemy_p, key=en_val)
                if en_val(best_e) > 0:
                    chosen = best_e

        # ── 2d: Force nearest reachable enemy if nothing else ────────────────
        if chosen is None and enemy_p:
            reachable_e = [e for e in enemy_p if capture_eta(src, e, fleet_size) < rem_steps]
            if reachable_e:
                chosen = min(reachable_e, key=lambda e: dp(src, e))

        if chosen is None:
            continue

        d            = dp(src, chosen)
        t            = travel_turns(d, fleet_size)
        g_est        = chosen.ships + (int(chosen.production * t) if chosen.owner != -1 else 0)
        still_needed = (g_est + 1) - incoming.get(chosen.id, 0)   # fixed: was always fleet_size
        send_n       = int(min(fleet_size, max(1, still_needed)))

        s_thresh = SAFETY_ATTACK_THRESH if chosen.owner != -1 else SAFETY_EXPAND_THRESH
        fire(src, chosen, send_n, allow_unsafe=(endgame or rem_steps < 50), safety_thresh=s_thresh)

    # ── Phase 2e: Micro second-wave ───────────────────────────────────────────
    # After the main wave, fire small second fleets at uncovered targets.
    for src in sorted(my_p, key=lambda p: -safe_avail(p)):
        av = safe_avail(src)
        fleet2 = int(av * WAVE_FRAC)
        if fleet2 < 2:
            continue
        micro_tgt = None
        for n in sorted(neutral_p, key=lambda n: capture_eta(src, n, max(1, fleet2))):
            if target_need(n, incoming) > 0:
                micro_tgt = n
                break
        if micro_tgt is None and enemy_p:
            candidates = [e for e in enemy_p if enemy_need(src, e, fleet2, incoming) > 0]
            if candidates:
                micro_tgt = min(candidates, key=lambda e: dp(src, e))
        if micro_tgt is None:
            continue
        need2 = target_need(micro_tgt, incoming)
        send2 = int(min(fleet2, max(1, need2)))
        s2 = SAFETY_ATTACK_THRESH if micro_tgt.owner != -1 else SAFETY_EXPAND_THRESH
        fire(src, micro_tgt, send2, safety_thresh=s2)

    # ── Phase 3: Multi-planet rotating coordination ───────────────────────────
    # When a rotating neutral target still has need > 0 after all per-planet
    # phases, pool ships from multiple nearby planets that are in their timing
    # window to cover it in a single coordinated wave.
    for tgt in sorted([n for n in neutral_p if not is_idle(n)],
                      key=lambda t: -t.production):
        need = target_need(tgt, incoming)
        if need <= 0:
            continue
        contributors = []
        for src in sorted(my_p, key=lambda p: dp(p, tgt)):
            if threatened.get(src.id, False):
                continue
            av = safe_avail(src)
            if av < EXPAND_MIN_SEND:
                continue
            wait = closest_approach_timing(src, tgt, ang_vel, max(1, av))
            if wait <= 1.0:
                contributors.append((src, av))
        if not contributors:
            continue
        total = sum(av for _, av in contributors)
        if total < need:
            continue
        remaining = need
        for src, av in sorted(contributors, key=lambda x: -x[1]):
            if remaining <= 0:
                break
            send_n = min(av, remaining)
            if send_n >= EXPAND_MIN_SEND:
                fire(src, tgt, send_n, allow_unsafe=True)
                remaining -= send_n

    # ── Phase 4: Coordinated enemy assault ───────────────────────────────────
    # Pool ships from multiple planets to take the most valuable exposed enemy.
    # Only runs when Phase 2 left the target under-committed.
    if enemy_p and my_p:
        def assault_value(e):
            nearest = min(my_p, key=lambda p2: dp(p2, e))
            still = enemy_need(nearest, e, max(1, safe_avail(nearest)), incoming)
            if still <= 0:
                return -1.0
            if sum(safe_avail(p2) for p2 in my_p) < still:
                return -1.0
            return e.production * rem_steps / (still + 5.0)

        atgt = max(enemy_p, key=assault_value)
        if assault_value(atgt) > 0:
            nearest = min(my_p, key=lambda p2: dp(p2, atgt))
            a_need = max(1, enemy_need(nearest, atgt, max(1, safe_avail(nearest)), incoming))
            remaining = a_need
            p4_min = 2 if opp_is_turtle else MIN_FLEET
            for src in sorted(my_p, key=lambda p2: dp(p2, atgt)):
                if remaining <= 0:
                    break
                if threatened.get(src.id, False):
                    continue
                av = safe_avail(src)
                if av < p4_min:
                    continue
                send_n = min(av, remaining)
                if fire(src, atgt, send_n, safety_thresh=SAFETY_ATTACK_THRESH):
                    remaining -= send_n

    # ── Endgame drain ─────────────────────────────────────────────────────────
    # Near game end, send all idle ships to every reachable target.
    if step >= FINAL_STEPS:
        drain_targets = sorted(
            [t for t in neutral_p + enemy_p if target_need(t, incoming) > 0],
            key=lambda t: min(capture_eta(p2, t, max(1, avail(p2))) for p2 in my_p)
        )
        for src in sorted(my_p, key=lambda p: -avail(p)):
            av = avail(src) - 1
            if av <= 0:
                continue
            for tgt in drain_targets:
                if capture_eta(src, tgt, max(1, av)) >= rem_steps:
                    continue
                need = target_need(tgt, incoming)
                if need <= 0:
                    continue
                send_n = min(av, need)
                if send_n > 0 and fire(src, tgt, send_n, allow_unsafe=True):
                    av -= send_n
                if av <= 0:
                    break

    return moves
