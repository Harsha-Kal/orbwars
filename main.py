"""
Orbit Wars – Competitive Agent v12
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


# ── Strategy phase ────────────────────────────────────────────────────────────

def get_phase(my_p, enemy_p, enemy_fleets, step, ang_vel):
    """Classify current state into one of six strategic phases."""
    my_prod    = sum(p.production for p in my_p)
    enemy_prod = sum(p.production for p in enemy_p)
    factions   = set(p.owner for p in enemy_p)

    if len(my_p) < 3 or step < 50:
        return 'OPENING'
    if len(enemy_p) <= 2:
        return 'ENDGAME'
    if my_p and (enemy_p or enemy_fleets):
        in_contact = (
            any(fleet_heading_to(fl, m, ang_vel) for fl in enemy_fleets for m in my_p)
            or (enemy_p and min(dp(m, e) for m in my_p for e in enemy_p) < 30)
        )
        if in_contact:
            return 'CONTACT'
    if enemy_prod > my_prod * 1.3:
        return 'BEHIND'
    # COLLAPSE: clearly ahead in production or planet count
    best_ep = max((sum(p.production for p in enemy_p if p.owner == o)
                   for o in factions), default=0)
    best_ec = max((sum(1 for p in enemy_p if p.owner == o)
                   for o in factions), default=0)
    if my_prod > best_ep * 1.35 or len(my_p) > best_ec + 3:
        return 'COLLAPSE'
    if len(factions) >= 2:
        fp = {}
        for e in enemy_p:
            fp[e.owner] = fp.get(e.owner, 0) + e.production
        if max(fp.values(), default=0) > my_prod * 1.25:
            return 'ANTI_LEADER'
    return 'EXPANSION'


def planet_value_score(src, tgt, need, eta, already, enemy_p, rem_steps, phase,
                       finish_bonus=0.0, chain_bonus=0.0, fresh_enemy=False):
    """
    Unified target score: 4*prod + 3*prod/ETA + 2*prod/need + bonuses - risks.
    Higher = more attractive target.
    """
    if need <= 0:
        return -1e9
    eta  = max(0.1, eta)
    prod = tgt.production
    score  = 4.0 * prod
    score += 3.0 * prod / eta
    score += 2.0 * prod / max(1.0, need)
    score += prod * min(rem_steps, 100) * 0.01  # future production value
    score += finish_bonus
    score += chain_bonus
    if fresh_enemy:
        score += 25.0
    score -= dp(src, tgt) * 0.15
    if enemy_p:
        retake_d = min(dist(e.x, e.y, tgt.x, tgt.y) for e in enemy_p)
        score -= max(0.0, 30.0 - retake_d) * 0.5
    if already > need * 1.5:
        score -= 20.0
    if phase == 'BEHIND':
        score += prod * 2.0
    elif phase == 'OPENING':
        score -= dp(src, tgt) * 0.3
    return score


def predicted_angle(src, tgt, ang_vel, ships):
    """Intercept angle to a possibly-moving target (wraps aim_valid)."""
    angle, _ = aim_valid(src, tgt, ang_vel, ships)
    return angle


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

    # ── Game phase ────────────────────────────────────────────────────────────
    phase       = get_phase(my_p, enemy_p, enemy_fleets, step, ang_vel)
    behind      = phase == 'BEHIND'
    in_opening  = phase == 'OPENING'
    in_contact  = phase == 'CONTACT'
    anti_ldr    = phase == 'ANTI_LEADER'
    in_collapse = phase == 'COLLAPSE'

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
        elif under_attack or in_contact:
            base = int(base * WAR_MODE_RESERVE_MULT)
        if behind:
            base = int(base * 1.15)
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
    # BEHIND: disable tiny spam; SWARM: widen net to catch more targets
    finisher_limit = 0 if behind else (FINISHER_SHIPS * 2 if opp_is_swarm else FINISHER_SHIPS)
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
            def opening_ok(n):
                if n.ships > 35 and n.production <= 2:
                    return False
                if enemy_p and min(dp(e, n) for e in enemy_p) < dp(src, n) * 0.85:
                    return False  # enemy closer → they'll beat us there
                return True
            idle_targets     = [n for n in neutral_p if     is_idle(n) and target_need(n, incoming) > 0 and opening_ok(n)]
            rotating_targets = [n for n in neutral_p if not is_idle(n) and target_need(n, incoming) > 0 and opening_ok(n)]
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
        if opp_is_swarm or in_contact:
            effective_max = 0                           # stop expanding under pressure
        elif opp_is_aggressive or under_attack or behind:
            effective_max = EXPAND_WAR_MAX              # consolidate when at risk
        elif in_opening:
            effective_max = min(2, EXPAND_MAX_PER_PLANET)  # max 2 targets in opening
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

    # ── Macro wave engine ─────────────────────────────────────────────────────
    rem_steps  = max(1, TOTAL_STEPS - step)
    endgame    = step >= ENDGAME_STEPS  # kept for drain pass
    late_game  = (in_collapse
                  or step >= LATE_GAME_STEPS
                  or (opp_is_turtle     and step >= LATE_GAME_STEPS // 2)
                  or (opp_is_aggressive and step >= LATE_GAME_STEPS * 2 // 3)
                  or (in_contact        and step >= LATE_GAME_STEPS * 2 // 3))

    # ── Per-target scoring and shared helpers ──────────────────────────────────

    def reserve_needed_fn(p):
        base = reserve_for(p)
        if p.ships < p.production * 6 and p.production > 0:
            base = max(base, 6)   # recently captured — keep minimum cushion
        if p.production >= HIGH_PROD_THRESHOLD:
            base = max(base, 10)
        if enemy_p and min(dp(p, e) for e in enemy_p) < 20:
            base = max(base, 12)  # enemy-facing
        return base

    def target_score_fn(tgt):
        """Unified score per target (higher = more attractive)."""
        already = incoming.get(tgt.id, 0)
        if tgt.owner == -1:
            need = target_need(tgt, incoming)
        else:
            ref = min(my_p, key=lambda p: dp(p, tgt)) if my_p else None
            if ref is None:
                return -1e9
            need = enemy_need(ref, tgt, max(1, safe_avail(ref)), incoming)
        if need <= 0:
            return -1e9
        eta = min(capture_eta(p, tgt, max(1, need)) for p in my_p) if my_p else 1e9
        if eta >= rem_steps - 5:
            return -1e9
        prod  = tgt.production
        score = 5.0 * prod + 4.0 * prod / max(1.0, eta) + 3.0 * prod / max(1.0, need)
        score += prod * min(rem_steps, 100) * 0.01
        # finish bonus
        if tgt.ships <= 5:
            score += 35.0
        elif tgt.ships <= 12:
            score += 15.0
        # chain bonus: nearby unclaimed neutrals reachable after capture
        score += sum(1 for n in neutral_p if n.id != tgt.id and dp(tgt, n) < 22) * 6.0
        # launchpad bonus
        if prod >= 3:
            score += 12.0
        # fresh enemy capture
        if tgt.owner not in (-1, me) and tgt.ships <= 10:
            score += 20.0
        # 4-player side/polar modifier
        if four_player and my_p and tgt.owner not in (-1, me):
            src_ang = math.atan2(my_p[0].y - SUN_Y, my_p[0].x - SUN_X)
            t_ang   = math.atan2(tgt.y - SUN_Y, tgt.x - SUN_X)
            adiff   = abs(math.atan2(math.sin(t_ang - src_ang), math.cos(t_ang - src_ang)))
            score  *= SIDE_ENEMY_BONUS if adiff < math.pi * 0.67 else POLAR_ENEMY_PENALTY
        # enemy retake risk
        if enemy_p:
            min_ed = min(dist(e.x, e.y, tgt.x, tgt.y) for e in enemy_p)
            score -= max(0.0, 28.0 - min_ed) * 0.7
        score -= eta * 0.25
        if already > need * 1.5:
            score -= 30.0
        if in_collapse and tgt.owner not in (-1, me):
            score += prod * 3.0
        elif behind:
            score += prod * 2.5
        return score

    def can_hold(tgt):
        """True when we can reasonably defend tgt after capture."""
        if not enemy_p:
            return True
        return min(dist(e.x, e.y, tgt.x, tgt.y) for e in enemy_p) > 18 or tgt.production >= 3

    def send_wave(tgt, need, allow_u=False, max_src=4, min_send=EXPAND_MIN_SEND):
        """Pool ships from up to max_src nearest unthreatened sources."""
        s = SAFETY_ATTACK_THRESH if tgt.owner != -1 else SAFETY_EXPAND_THRESH
        sources = sorted([p for p in my_p if not threatened.get(p.id, False)],
                         key=lambda p: dp(p, tgt))[:max_src]
        pooled = 0
        for src in sources:
            to_send = min(safe_avail(src) if not allow_u else avail(src) - 1,
                         need - pooled)
            if to_send >= (1 if allow_u else min_send):
                if fire(src, tgt, to_send, allow_unsafe=allow_u, safety_thresh=s):
                    pooled += to_send
            if pooled >= need:
                break
        return pooled

    # ── Macro decision functions ───────────────────────────────────────────────

    def choose_expand_wave():
        """Priority 4: decisive grouped wave to highest-scoring reachable target."""
        if behind and not in_contact:
            return
        prefer_enemy = late_game or in_collapse
        scored = sorted(
            [(target_score_fn(t), t) for t in
             (enemy_p if prefer_enemy else neutral_p + enemy_p)
             if target_need(t, incoming) > 0],
            reverse=True
        )
        for score, tgt in scored:
            if score <= 0:
                break
            if tgt.owner == -1:
                need = target_need(tgt, incoming)
            else:
                ref  = min(my_p, key=lambda p: dp(p, tgt))
                need = enemy_need(ref, tgt, max(1, safe_avail(ref)), incoming)
            if need <= 0:
                continue
            if not can_hold(tgt):
                continue
            # Require 80 % pooling coverage before committing
            pool_cap = sum(safe_avail(p) for p in my_p if not threatened.get(p.id, False))
            if pool_cap < need * 0.8:
                continue
            if send_wave(tgt, need) > 0:
                return

    def choose_launchpad_chain():
        """Priority 5: use border planets as relay to chain neutral expansion."""
        if not neutral_p or behind:
            return
        frontier = sorted(
            [p for p in my_p
             if safe_avail(p) >= reserve_needed_fn(p) + 3
             and (not enemy_p or min(dp(p, e) for e in enemy_p) < 40)],
            key=lambda p: -safe_avail(p)
        )
        for src in frontier[:2]:
            cands = sorted(
                [n for n in neutral_p
                 if target_need(n, incoming) > 0
                 and capture_eta(src, n, max(1, target_need(n, incoming))) < rem_steps - 5],
                key=lambda n: dp(src, n)
            )
            for tgt in cands[:1]:
                need = target_need(tgt, incoming)
                av   = safe_avail(src)
                if av >= need:  # only decisive sends
                    fire(src, tgt, need, safety_thresh=SAFETY_EXPAND_THRESH)
                    break

    def choose_enemy_attack_wave():
        """Priority 6: multi-source assault on best enemy planet."""
        if not enemy_p:
            return
        best = max(enemy_p, key=target_score_fn)
        if target_score_fn(best) <= 0:
            return
        ref      = min(my_p, key=lambda p: dp(p, best))
        need     = max(1, enemy_need(ref, best, max(1, safe_avail(ref)), incoming))
        min_chip = 2 if opp_is_turtle else EXPAND_MIN_SEND  # turtle: smaller chips allowed
        if sum(safe_avail(p) for p in my_p if not threatened.get(p.id, False)) < need:
            return
        send_wave(best, need, max_src=6 if opp_is_turtle else 4, min_send=min_chip)

    def choose_collapse_move():
        """Priority 7 (COLLAPSE): harvest all enemy production with large waves."""
        if not in_collapse or not enemy_p:
            return
        for tgt in sorted(enemy_p, key=lambda e: -e.production):
            ref  = min(my_p, key=lambda p: dp(p, tgt))
            need = max(1, enemy_need(ref, tgt, max(1, avail(ref) - 2), incoming))
            if need <= 0:
                continue
            remaining = need
            for src in sorted(my_p, key=lambda p: dp(p, tgt)):
                if remaining <= 0:
                    break
                to_send = min(avail(src) - 2, remaining)
                if to_send >= 3 and fire(src, tgt, to_send, allow_unsafe=True):
                    remaining -= to_send

    def choose_consolidate():
        """Priority 8 (BEHIND): pull ships toward strongest core planet."""
        if not behind:
            return
        core = max(my_p, key=lambda p: p.production, default=None)
        if core is None:
            return
        for src in sorted([p for p in my_p if p.id != core.id],
                          key=lambda p: -safe_avail(p))[:2]:
            if dp(src, core) < 10:
                continue
            send_n = safe_avail(src) // 2
            if send_n >= 5:
                fire(src, core, send_n)

    def choose_rotating_coordination():
        """Rotating-planet coordination: pool ships for planets in timing window."""
        for tgt in sorted([n for n in neutral_p if not is_idle(n)],
                          key=lambda t: -t.production):
            need = target_need(tgt, incoming)
            if need <= 0:
                continue
            contributors = []
            for src in sorted(my_p, key=lambda p: dp(p, tgt)):
                if threatened.get(src.id, False):
                    continue
                av   = safe_avail(src)
                wait = closest_approach_timing(src, tgt, ang_vel, max(1, av))
                if av >= EXPAND_MIN_SEND and wait <= 1.0:
                    contributors.append((src, av))
            total = sum(a for _, a in contributors)
            if total < need:
                continue
            remaining = need
            for src, av in sorted(contributors, key=lambda x: -x[1]):
                if remaining <= 0:
                    break
                send_n = min(av, remaining)
                if send_n >= EXPAND_MIN_SEND and fire(src, tgt, send_n, allow_unsafe=True):
                    remaining -= send_n

    def choose_comet_chase():
        """Priority 10: chase fast-moving planets only when safe and high-value."""
        if behind or in_contact:
            return
        for src in sorted(my_p, key=lambda p: -safe_avail(p)):
            av = safe_avail(src)
            if av < 5:
                continue
            comets = [
                n for n in neutral_p
                if not is_idle(n)
                and target_need(n, incoming) > 0
                and arc_travel(n, ang_vel, src.x, src.y) <= COMET_ARC_LIMIT
                and capture_eta(src, n, max(1, target_need(n, incoming))) < 15
                and n.production >= HIGH_PROD_THRESHOLD
            ]
            for tgt in sorted(comets, key=lambda n: -n.production)[:1]:
                need   = target_need(tgt, incoming)
                send_n = min(av, need)
                if send_n >= need:
                    fire(src, tgt, send_n, safety_thresh=SAFETY_EXPAND_THRESH)
                    break

    # Run the macro wave priority sequence
    choose_rotating_coordination()   # always pool rotating planets first
    choose_expand_wave()             # decisive grouped wave to best target
    choose_launchpad_chain()         # chain from frontier planets
    choose_enemy_attack_wave()       # dedicated enemy assault pass
    choose_collapse_move()           # all-in when clearly ahead
    choose_consolidate()             # pull together when behind
    choose_comet_chase()             # opportunistic fast-moving planets last

    # Endgame all-in: use avail() instead of safe_avail for reachable targets
    if endgame:
        for src in sorted(my_p, key=lambda p: -avail(p)):
            av = avail(src)
            fleet_size = av
            if fleet_size < 2:
                continue
            for tgt in sorted(neutral_p + enemy_p,
                               key=lambda t: capture_eta(src, t, max(1, fleet_size))):
                if capture_eta(src, tgt, max(1, fleet_size)) >= rem_steps:
                    continue
                if tgt.owner == -1:
                    need = target_need(tgt, incoming)
                else:
                    need = enemy_need(src, tgt, fleet_size, incoming)
                if need <= 0:
                    continue
                send_n = int(min(fleet_size, max(1, need)))
                s = SAFETY_ATTACK_THRESH if tgt.owner != -1 else SAFETY_EXPAND_THRESH
                fire(src, tgt, send_n, allow_unsafe=True, safety_thresh=s)
                break

    # ── Priority functions (helpers for the structured decision loop) ─────────

    def projected_state(tgt, eta):
        """Estimate tgt.ships at fleet arrival; positive = still defending."""
        e_in  = sum(fl.ships for fl in enemy_fleets  if fleet_heading_to(fl, tgt, ang_vel))
        f_in  = incoming.get(tgt.id, 0)
        prod  = int(tgt.production * eta) if tgt.owner != -1 else 0
        return max(0, tgt.ships + prod + e_in - f_in)

    def choose_reinforce():
        """Priority 2: top up thin border/new planets from safe backline sources."""
        for tgt in sorted(my_p, key=lambda p: p.ships):
            if tgt.ships >= tgt.production * 10:
                continue
            if not enemy_p:
                continue
            if min(dp(tgt, e) for e in enemy_p) > 40:
                continue  # not close enough to the front to matter
            want = tgt.production * 10 - tgt.ships
            for src in sorted([q for q in my_p if q.id != tgt.id],
                               key=lambda q: -safe_avail(q)):
                av = safe_avail(src)
                send_n = min(av, want)
                if send_n >= 3:
                    fire(src, tgt, send_n)
                    break

    def choose_finish_lock():
        """Priority 3: commit to damaged targets (low ships + incoming already sent)."""
        finish_cap = 0 if behind else 18
        # Score candidates: smaller projected state + bigger finish bonus = higher priority
        candidates = []
        for tgt in neutral_p + enemy_p:
            if tgt.ships > finish_cap:
                continue
            if incoming.get(tgt.id, 0) == 0:
                continue
            need = target_need(tgt, incoming)
            if need <= 0:
                continue
            src_ref = min(my_p, key=lambda p: dp(p, tgt), default=None)
            if src_ref is None:
                continue
            eta = capture_eta(src_ref, tgt, max(1, need))
            proj = projected_state(tgt, eta)        # ships remaining at arrival
            fb   = 40.0 if proj <= 5 else 20.0     # finish bonus: used for sorting
            candidates.append((-fb, proj, tgt, need))
        for _, proj, tgt, need in sorted(candidates):
            for src in sorted(my_p, key=lambda p: dp(p, tgt)):
                if threatened.get(src.id, False):
                    continue
                av = safe_avail(src)
                send_n = min(av, need)
                if send_n > 0:
                    fire(src, tgt, send_n, allow_unsafe=True)
                    break

    def choose_punish():
        """Priority 6: exploit freshly captured or low-ship enemy planets."""
        punish_cap = 14
        for tgt in sorted(enemy_p, key=lambda e: e.ships):
            if tgt.ships > punish_cap:
                break
            already = incoming.get(tgt.id, 0)
            if already >= tgt.ships + 1:
                continue
            for src in sorted(my_p, key=lambda p: dp(p, tgt)):
                if threatened.get(src.id, False):
                    continue
                av = safe_avail(src)
                need_n = enemy_need(src, tgt, av, incoming)
                if 0 < need_n <= av:
                    fire(src, tgt, min(av, need_n), safety_thresh=SAFETY_ATTACK_THRESH)
                    break

    def choose_anti_leader():
        """Priority 7: in 4-player, attack the production leader's weakest planets."""
        if not anti_ldr:
            return
        fp = {}
        for e in enemy_p:
            fp[e.owner] = fp.get(e.owner, 0) + e.production
        if not fp:
            return
        leader = max(fp, key=fp.get)
        targets = sorted([e for e in enemy_p if e.owner == leader], key=lambda e: e.ships)
        for tgt in targets[:2]:
            already = incoming.get(tgt.id, 0)
            for src in sorted(my_p, key=lambda p: dp(p, tgt)):
                if threatened.get(src.id, False):
                    continue
                av = safe_avail(src)
                need_n = enemy_need(src, tgt, av, incoming)
                if need_n <= 0 or already >= need_n:
                    continue
                send_n = min(av, need_n)
                if send_n >= need_n // 2:
                    fire(src, tgt, send_n, safety_thresh=SAFETY_ATTACK_THRESH)
                    break

    # Run structured priority loop (phases 1-5 already ran; these fill remaining gaps)
    choose_reinforce()
    choose_finish_lock()
    choose_punish()
    choose_anti_leader()

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
