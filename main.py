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
OPENING_MIN_SEND = 6    # keep opener decisive; tiny sends only finish targets
OPENING_MAX_ARC  = 20.0  # was 18.0

EXPAND_STEPS          = 260   # was 220
EXPAND_RESERVE        = 1     # was 2
EXPAND_MIN_SEND       = 5     # avoid micro-spam in expansion loops
EXPAND_MAX_PER_PLANET = 2     # fewer, more decisive launches per source
ROTATING_SAVE_RESERVE = 1
EVAC_RESERVE          = 0
ROTATING_NEAR_FACTOR  = 0.75
ROTATING_NEAR_MARGIN  = 8.0
FAST_ROTATING_TURNS   = 20.0  # was 18.0
SAME_TIME_MARGIN      = 4.0

# Quick-capture: blitz cheap nearby planets every step until captured
QUICK_SHIPS       = 20
QUICK_STEPS       = 5
QUICK_MAX_PER_SRC = 1     # max quick-capture targets fired per source per turn

# Finisher / micro-wave
FINISHER_SHIPS = 3     # target with ≤ this remaining need gets a finisher fleet
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
    if len(my_p) < 3 or step < 50:
        return 'OPENING'
    if len(enemy_p) <= 2:
        return 'ENDGAME'
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


# ── tempo / wave controller ──────────────────────────────────────────────────

def agent(obs):
    g = lambda k, d: (obs.get(k, d) if isinstance(obs, dict) else getattr(obs, k, d))
    me   = int(g("player", 0))
    step = int(g("step", 0))
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
    factions     = set(p.owner for p in planets if p.owner >= 0)
    four_player  = len(factions) >= 3
    rem_steps    = max(1, TOTAL_STEPS - step)

    if step <= 1 or not hasattr(agent, "_last_meaningful"):
        agent._last_meaningful = {}
    last_meaningful = agent._last_meaningful.get(me, step)

    moves     = []
    committed = {}
    incoming  = {}
    target_enemy_incoming = {}
    tempo_move = {"offensive_ships": 0, "wave_attempted": False}

    def avail(p):
        return max(0, p.ships - committed.get(p.id, 0))

    def eta_to_planet(fl, p):
        return travel_turns(dist(fl.x, fl.y, p.x, p.y), fl.ships)

    def eta_incoming_counts(p, horizon=DEFENSE_ETA_HORIZON):
        fi = sum(
            fl.ships for fl in my_fleets
            if fleet_heading_to(fl, p, ang_vel) and eta_to_planet(fl, p) <= horizon
        )
        ei = sum(
            fl.ships for fl in enemy_fleets
            if fleet_heading_to(fl, p, ang_vel) and eta_to_planet(fl, p) <= horizon
        )
        return fi, ei

    for fl in my_fleets:
        tgt = fleet_target(fl, neutral_p + enemy_p, ang_vel)
        if tgt is not None:
            incoming[tgt.id] = incoming.get(tgt.id, 0) + fl.ships

    for fl in enemy_fleets:
        tgt = fleet_target(fl, neutral_p + enemy_p, ang_vel)
        if tgt is not None:
            target_enemy_incoming[tgt.id] = target_enemy_incoming.get(tgt.id, 0) + fl.ships

    bridge_ids = set()
    if enemy_p and len(my_p) >= BRIDGE_MIN_PLANETS:
        for m in my_p:
            for n in neutral_p:
                for e in enemy_p:
                    if is_bridge(m, n, e):
                        bridge_ids.add(n.id)
                        break

    def real_incoming_threat(p):
        fi, ei = eta_incoming_counts(p)
        net = p.ships + fi - ei
        deficit = max(0, DEFEND_NET - net) if ei > fi else 0
        return {"friendly": fi, "enemy": ei, "net": net, "deficit": deficit}

    threat = {p.id: real_incoming_threat(p) for p in my_p}
    threatened = {p.id: threat[p.id]["deficit"] > 0 for p in my_p}

    def is_frontline(p):
        return bool(enemy_p) and min(dp(p, e) for e in enemy_p) < FRONTLINE_DIST

    def reserve_for(p):
        if step < EARLY_STEPS:
            base = min(3, 1 + p.production)
        elif step < LATE_GAME_STEPS:
            base = 4 + p.production
        else:
            base = 6 + p.production

        raw = base
        if threatened.get(p.id, False):
            raw += threat[p.id]["deficit"] + 3
        elif is_frontline(p):
            raw += max(2, int(p.production * 0.75))

        cap = max(20, int(p.ships * (0.45 if step < EARLY_STEPS else 0.55)))
        return int(min(raw, cap))

    def calculate_surplus(p):
        return max(0, p.ships - committed.get(p.id, 0) - reserve_for(p))

    def total_surplus():
        return sum(calculate_surplus(p) for p in my_p)

    def target_need(tgt, incoming_map):
        enemy_claim = target_enemy_incoming.get(tgt.id, 0)
        return tgt.ships + 1 + enemy_claim - incoming_map.get(tgt.id, 0)

    def need_for(src, tgt, ships_hint=None):
        if tgt.owner == -1:
            return target_need(tgt, incoming)
        ships = max(1, int(ships_hint if ships_hint is not None else calculate_surplus(src)))
        return enemy_need(src, tgt, ships, incoming) + target_enemy_incoming.get(tgt.id, 0)

    my_prod = sum(p.production for p in my_p)
    my_ships = sum(p.ships for p in my_p) + sum(fl.ships for fl in my_fleets)

    enemy_stats = {}
    for p in enemy_p:
        stat = enemy_stats.setdefault(p.owner, {"prod": 0, "ships": 0, "planets": 0})
        stat["prod"] += p.production
        stat["ships"] += p.ships
        stat["planets"] += 1
    for fl in enemy_fleets:
        stat = enemy_stats.setdefault(fl.owner, {"prod": 0, "ships": 0, "planets": 0})
        stat["ships"] += fl.ships

    def power_score(stat):
        return stat["prod"] * 18 + stat["ships"] + stat["planets"] * 12

    leader = None
    leader_score = 0
    if enemy_stats:
        leader = max(enemy_stats, key=lambda o: power_score(enemy_stats[o]))
        leader_score = power_score(enemy_stats[leader])
    my_score = my_prod * 18 + my_ships + len(my_p) * 12
    leader_ahead = leader is not None and leader_score > my_score * 1.22

    enemy_prod_by_owner = {}
    for e in enemy_p:
        enemy_prod_by_owner[e.owner] = enemy_prod_by_owner.get(e.owner, 0) + e.production
    best_enemy_prod = max(enemy_prod_by_owner.values(), default=0)
    behind = best_enemy_prod > my_prod * 1.30
    no_good_neutrals = not any(n.production >= 2 and target_need(n, incoming) <= 45 for n in neutral_p)
    ahead = bool(enemy_p) and (
        my_prod > best_enemy_prod * 1.20
        or len(my_p) >= len(enemy_p) + 3
        or (leader_score > 0 and my_score > leader_score * 1.20)
    )
    collapse_mode = ahead and (
        (step > 220 and no_good_neutrals)
        or (my_ships > 350 and step - last_meaningful >= 8)
        or my_prod > best_enemy_prod * 1.30
        or len(my_p) >= len(enemy_p) + 3
    )

    def target_score_fn(tgt, src=None, leader_focus=False):
        ref = src or min(my_p, key=lambda p: dp(p, tgt), default=None)
        if ref is None:
            return -1e9
        av = max(1, calculate_surplus(ref))
        need = need_for(ref, tgt, av)
        if need <= 0:
            return -1e9
        eta = capture_eta(ref, tgt, max(1, min(need, av)))
        if eta >= rem_steps - 5:
            return -1e9
        phase_hint = "BEHIND" if behind else ("OPENING" if step < EARLY_STEPS else "EXPANSION")
        finish_bonus = 45.0 if incoming.get(tgt.id, 0) > 0 else 0.0
        chain_bonus = sum(1 for n in neutral_p if n.id != tgt.id and dp(tgt, n) < 22) * 5.0
        if tgt.id in bridge_ids:
            chain_bonus += BRIDGE_BONUS
        fresh_enemy = tgt.owner not in (-1, me) and tgt.ships <= OPPORTUNISTIC_SHIPS
        score = planet_value_score(
            ref, tgt, need, eta, incoming.get(tgt.id, 0), enemy_p, rem_steps, phase_hint,
            finish_bonus=finish_bonus, chain_bonus=chain_bonus, fresh_enemy=fresh_enemy
        )
        if tgt.owner == -1:
            score += tgt.production * 8.0 - tgt.ships * 0.7
            if incoming.get(tgt.id, 0) > 0:
                score += 30.0
        else:
            score += max(0, 28 - tgt.ships) * 1.8
            if tgt.owner == leader:
                score += 25.0 if leader_focus else 10.0
                score += tgt.production * 3.0
        if not is_idle(tgt):
            arc = arc_travel(tgt, ang_vel, ref.x, ref.y)
            if step >= EARLY_STEPS and arc > MAX_ARC and tgt.production < HIGH_PROD_THRESHOLD:
                return -1e9
            score -= arc * (0.18 if step < EARLY_STEPS else 0.35)
        if four_player and tgt.owner not in (-1, me):
            src_ang = math.atan2(ref.y - SUN_Y, ref.x - SUN_X)
            t_ang = math.atan2(tgt.y - SUN_Y, tgt.x - SUN_X)
            adiff = abs(math.atan2(math.sin(t_ang - src_ang), math.cos(t_ang - src_ang)))
            score *= SIDE_ENEMY_BONUS if adiff < math.pi * 0.67 else POLAR_ENEMY_PENALTY
        return score

    def fire(src, tgt, n, allow_partial=False):
        n = int(max(0, n))
        if n <= 0:
            return False
        if not allow_partial and n > avail(src):
            return False
        n = min(n, avail(src))
        ang, hit_ok = aim_valid(src, tgt, ang_vel, n)
        if not hit_ok:
            return False
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])
        if tgt.owner != me:
            incoming[tgt.id] = incoming.get(tgt.id, 0) + n
            tempo_move["offensive_ships"] += n
        return True

    def source_list_for(tgt, max_src=6):
        srcs = [p for p in my_p if not threatened.get(p.id, False)]
        return sorted(srcs, key=lambda p: (dp(p, tgt), -calculate_surplus(p)))[:max_src]

    def build_grouped_wave(tgt, need, max_src=6, allow_partial=False,
                           budget_fraction=1.0, min_send=EXPAND_MIN_SEND):
        sources = source_list_for(tgt, max_src=max_src)
        pool = sum(calculate_surplus(p) for p in sources)
        if pool <= 0:
            return 0
        target_amount = need if need <= pool else (int(pool * budget_fraction) if allow_partial else 0)
        if target_amount <= 0:
            return 0
        if allow_partial and target_amount < min_send:
            target_amount = min(pool, max(min_send, target_amount))

        sent = 0
        for src in sources:
            if sent >= target_amount:
                break
            av = calculate_surplus(src)
            if av <= 0:
                continue
            send_n = min(av, target_amount - sent)
            if send_n < min_send and sent + send_n < target_amount:
                continue
            if fire(src, tgt, send_n, allow_partial=allow_partial):
                sent += send_n
        if sent > 0 and tgt.owner != me:
            tempo_move["wave_attempted"] = True
        return sent

    def choose_opening_target(src):
        cands = []
        for n in neutral_p:
            need = target_need(n, incoming)
            if need <= 0:
                continue
            av = max(1, calculate_surplus(src))
            score = neutral_value(src, n, ang_vel, av)
            score += n.production * 20.0 - need * 1.5 - dp(src, n)
            cands.append((score, -need, -dp(src, n), n))
        return max(cands, default=(None, None, None, None))[3]

    def choose_expansion_target(src):
        cands = []
        for tgt in neutral_p:
            need = target_need(tgt, incoming)
            if need <= 0:
                continue
            score = target_score_fn(tgt, src)
            score += 35.0 if incoming.get(tgt.id, 0) > 0 else 0.0
            cands.append((score, tgt))
        for tgt in enemy_p:
            need = need_for(src, tgt)
            if need <= 0:
                continue
            if tgt.ships <= OPPORTUNISTIC_SHIPS or is_frontline(src) or step > 150:
                cands.append((target_score_fn(tgt, src), tgt))
        return max(cands, default=(None, None), key=lambda x: x[0])[1]

    def choose_leader_target():
        if leader is None:
            return None
        cands = []
        for tgt in [e for e in enemy_p if e.owner == leader]:
            ref = min(my_p, key=lambda p: dp(p, tgt), default=None)
            if ref is None:
                continue
            score = target_score_fn(tgt, ref, leader_focus=True)
            score += max(0, 35 - tgt.ships) * 2.0 + tgt.production * 8.0
            cands.append((score, tgt))
        return max(cands, default=(None, None), key=lambda x: x[0])[1]

    def best_collapse_target():
        pool = max(1, total_surplus())
        cands = []
        for tgt in enemy_p + ([] if step > LATE_GAME_STEPS else neutral_p):
            ref = min(my_p, key=lambda p: dp(p, tgt), default=None)
            if ref is None:
                continue
            need = need_for(ref, tgt, pool)
            if need <= 0:
                continue
            owner_bonus = 45.0 if tgt.owner != -1 else 0.0
            score = target_score_fn(tgt, ref, leader_focus=(tgt.owner == leader))
            score += owner_bonus - need * 0.25
            cands.append((score, tgt, need))
        return max(cands, default=(None, None, None), key=lambda x: x[0])

    def choose_finish_lock():
        cands = []
        for tgt in neutral_p + enemy_p:
            if incoming.get(tgt.id, 0) <= 0:
                continue
            ref = min(my_p, key=lambda p: dp(p, tgt), default=None)
            if ref is None:
                continue
            need = need_for(ref, tgt)
            if need <= 0:
                continue
            eta = capture_eta(ref, tgt, max(1, need))
            projected = max(0, tgt.ships + (int(tgt.production * eta) if tgt.owner != -1 else 0)
                            + target_enemy_incoming.get(tgt.id, 0) - incoming.get(tgt.id, 0))
            cands.append((projected, -target_score_fn(tgt, ref), tgt, need))
        for _, _, tgt, need in sorted(cands):
            if build_grouped_wave(tgt, need, max_src=4, allow_partial=False, min_send=1) > 0:
                return True
        return False

    def emergency_defense():
        acted = False
        for tgt in sorted(my_p, key=lambda p: -threat[p.id]["deficit"]):
            needed = threat[tgt.id]["deficit"]
            if needed <= 0:
                continue
            for src in sorted([p for p in my_p if p.id != tgt.id], key=lambda p: dp(p, tgt)):
                send_n = min(calculate_surplus(src), needed)
                if send_n <= 0:
                    continue
                if fire(src, tgt, send_n, allow_partial=True):
                    needed -= send_n
                    acted = True
                if needed <= 0:
                    break
        return acted

    def opening_phase():
        if not neutral_p or step >= OPENING_STEPS:
            return False
        acted = False

        def opening_partial_ok(tgt, need, send_n):
            return (
                send_n >= need
                or send_n >= 8
                or send_n >= need * 0.70
                or incoming.get(tgt.id, 0) > 0
            )

        if len(my_p) == 1:
            src = my_p[0]
            tgt = choose_opening_target(src)
            if tgt is None:
                return False
            need = max(1, target_need(tgt, incoming) + (2 if step < 25 else 1))
            av = max(0, src.ships - committed.get(src.id, 0) - (1 if step < 25 else 2))
            send_n = min(av, need)
            if (
                send_n > 0
                and (step < 25 or calculate_surplus(src) >= OPENING_MIN_SEND)
                and opening_partial_ok(tgt, need, send_n)
            ):
                acted = fire(src, tgt, send_n, allow_partial=True) or acted
        else:
            for src in sorted(my_p, key=lambda p: -calculate_surplus(p))[:3]:
                av = calculate_surplus(src)
                if av < OPENING_MIN_SEND:
                    continue
                tgt = choose_opening_target(src)
                if tgt is None:
                    continue
                need = target_need(tgt, incoming) + 1
                if need <= 0:
                    continue
                send_n = min(av, need)
                if opening_partial_ok(tgt, need, send_n):
                    acted = fire(src, tgt, send_n, allow_partial=True) or acted
        return acted

    def normal_expansion():
        acted = False
        for src in sorted(my_p, key=lambda p: -calculate_surplus(p)):
            launches = 0
            while launches < EXPAND_MAX_PER_PLANET:
                av = calculate_surplus(src)
                if av < EXPAND_MIN_SEND:
                    break
                tgt = choose_expansion_target(src)
                if tgt is None:
                    break
                need = need_for(src, tgt, av) + (1 if tgt.owner == -1 else 0)
                if need <= 0:
                    break
                send_n = min(av, need)
                if send_n < EXPAND_MIN_SEND and send_n < need:
                    break
                if fire(src, tgt, send_n):
                    acted = True
                    launches += 1
                else:
                    break
        return acted

    def launchpad_chain():
        if not neutral_p:
            return False
        acted = False
        frontier = sorted(
            [p for p in my_p if calculate_surplus(p) >= EXPAND_MIN_SEND
             and (is_frontline(p) or any(dp(p, n) < 25 for n in neutral_p))],
            key=lambda p: -calculate_surplus(p)
        )
        for src in frontier[:3]:
            cands = [n for n in neutral_p if target_need(n, incoming) > 0]
            if not cands:
                continue
            tgt = max(cands, key=lambda n: target_score_fn(n, src) - dp(src, n) * 0.25)
            need = target_need(tgt, incoming)
            send_n = min(calculate_surplus(src), need + 1)
            if send_n >= min(EXPAND_MIN_SEND, need):
                acted = fire(src, tgt, send_n) or acted
        return acted

    def weak_enemy_attacks():
        acted = False
        for tgt in sorted(enemy_p, key=lambda e: (e.ships, -e.production))[:3]:
            ref = min(my_p, key=lambda p: dp(p, tgt), default=None)
            if ref is None or target_score_fn(tgt, ref) <= 0:
                continue
            need = need_for(ref, tgt)
            if need <= 0:
                continue
            pool = total_surplus()
            if pool <= 0:
                return acted
            if need <= pool or tgt.ships <= OPPORTUNISTIC_SHIPS:
                acted = build_grouped_wave(
                    tgt, need, max_src=5, allow_partial=tgt.ships <= OPPORTUNISTIC_SHIPS,
                    budget_fraction=0.45, min_send=3
                ) > 0 or acted
        return acted

    def anti_leader_overlay():
        if not (four_player and leader_ahead):
            return False
        tgt = choose_leader_target()
        if tgt is None:
            return False
        ref = min(my_p, key=lambda p: dp(p, tgt), default=None)
        if ref is None:
            return False
        pool = total_surplus()
        if pool < 10:
            return False
        need = need_for(ref, tgt, pool)
        budget = max(1, int(pool * 0.32))
        return build_grouped_wave(
            tgt, min(need, budget), max_src=6, allow_partial=True,
            budget_fraction=0.32, min_send=3
        ) > 0

    def force_wave_if_inactive():
        idle_turns = step - last_meaningful
        if step < 120 or step > 420 or idle_turns < 9 or my_ships <= 250:
            return False
        pool = total_surplus()
        if pool < 25:
            return False
        targets = enemy_p + neutral_p
        if not targets:
            return False
        tgt = max(targets, key=lambda t: target_score_fn(t, None, leader_focus=(t.owner == leader)))
        ref = min(my_p, key=lambda p: dp(p, tgt), default=None)
        if ref is None:
            return False
        need = need_for(ref, tgt, pool)
        send_goal = max(int(pool * 0.25), min(need, int(pool * 0.35)))
        return build_grouped_wave(
            tgt, send_goal, max_src=6, allow_partial=True,
            budget_fraction=0.35, min_send=5
        ) > 0

    def collapse_attack():
        if not collapse_mode:
            return False
        _, tgt, need = best_collapse_target()
        if tgt is None or need is None:
            return False
        return build_grouped_wave(
            tgt, need, max_src=8, allow_partial=True,
            budget_fraction=0.55, min_send=5
        ) > 0

    def final_drain():
        if step < FINAL_STEPS:
            return False
        acted = False
        targets = [t for t in neutral_p + enemy_p if target_need(t, incoming) > 0]
        for src in sorted(my_p, key=lambda p: -avail(p)):
            av = avail(src) - 1
            if av <= 0:
                continue
            for tgt in sorted(targets, key=lambda t: capture_eta(src, t, max(1, av))):
                if capture_eta(src, tgt, max(1, av)) >= rem_steps:
                    continue
                need = need_for(src, tgt, av)
                if need <= 0:
                    continue
                if fire(src, tgt, min(av, need), allow_partial=True):
                    acted = True
                    break
        return acted

    meaningful = False
    meaningful = emergency_defense() or meaningful
    meaningful = choose_finish_lock() or meaningful
    meaningful = opening_phase() or meaningful
    meaningful = normal_expansion() or meaningful
    meaningful = launchpad_chain() or meaningful
    meaningful = weak_enemy_attacks() or meaningful
    meaningful = anti_leader_overlay() or meaningful
    meaningful = force_wave_if_inactive() or meaningful
    meaningful = collapse_attack() or meaningful
    meaningful = final_drain() or meaningful

    if tempo_move["offensive_ships"] >= 15 or tempo_move["wave_attempted"]:
        agent._last_meaningful[me] = step

    return moves
