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
INCOMING_T = 0.85   # looser target lock for fleets already in flight

OPENING_STEPS    = 90    # rush the first meaningful neutral before settling in
OPENING_RESERVE  = 1     # keep only a token home garrison during the opening
OPENING_MIN_SEND = 5     # smallest useful opening chip/finish fleet
OPENING_MAX_ARC  = 18.0  # opening can tolerate a little more target motion

EXPAND_STEPS    = 220    # keep spreading after the first capture
EXPAND_RESERVE  = 2      # avoid draining new colonies completely
EXPAND_MIN_SEND = 3      # cheap neutral finishers are worth sending early
EXPAND_MAX_PER_PLANET = 3 # fan out instead of one launch then idling
ROTATING_SAVE_RESERVE = 1 # rotating planets save until they can finish an idle
EVAC_RESERVE = 0          # threatened planets should move everything useful
POWER_IDLE_COUNT = 2      # once we own this many idle planets...
POWER_ROT_COUNT = 2       # ...and this many rotating planets, widen attacks
POWER_RESERVE = 1
POWER_MAX_PER_PLANET = 4
ROTATING_NEAR_FACTOR = 0.75 # take rotating first if idle target is much farther
ROTATING_NEAR_MARGIN = 8.0
FAST_ROTATING_TURNS = 18.0  # rotating opener is worth it if capture is this quick
SAME_TIME_MARGIN = 4.0      # "same time" capture window for idle vs rotating
IDLE_LOWEST_NEAR_MARGIN = 6.0 # lowest idle can count as nearest within this gap
LOCKED_TARGET_RADIUS = 35.0 # finish nearby invested targets before starting new ones
LOCKED_MAX_NEED_RATIO = 1.25 # abandon if target now needs far more than available
FRONTIER_BONUS = 35.0      # prefer bridge idle planets toward opponent idle bases
ROTATING_RAID_BONUS = 45.0 # rotating fronts near opponent planets are valuable
DEN_RADIUS = 28.0          # rotating planets inside this range can defend home


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

def frontier_idle_bonus(src, tgt, enemy_p):
    enemy_idle = [e for e in enemy_p if is_idle(e)]
    if not enemy_idle:
        return 0.0
    nearest_enemy = min(enemy_idle, key=lambda e: dp(tgt, e))
    src_to_enemy = dp(src, nearest_enemy)
    tgt_to_enemy = dp(tgt, nearest_enemy)
    if tgt_to_enemy >= src_to_enemy:
        return 0.0
    progress = src_to_enemy - tgt_to_enemy
    bridge = max(0.0, 1.0 - abs(dp(src, tgt) - tgt_to_enemy) / max(1.0, src_to_enemy))
    return FRONTIER_BONUS * progress / max(1.0, src_to_enemy) + 12.0 * bridge

def rotating_raid_bonus(src, tgt, enemy_p):
    if not enemy_p:
        return 0.0
    nearest_enemy = min(enemy_p, key=lambda e: dp(tgt, e))
    src_to_enemy = dp(src, nearest_enemy)
    tgt_to_enemy = dp(tgt, nearest_enemy)
    progress = max(0.0, src_to_enemy - tgt_to_enemy)
    close_pressure = max(0.0, 1.0 - tgt_to_enemy / max(1.0, src_to_enemy))
    return ROTATING_RAID_BONUS * progress / max(1.0, src_to_enemy) + 18.0 * close_pressure

def den_defense_target(src, my_p, my_fleets, enemy_fleets, ang_vel):
    idle_owned = [p for p in my_p if is_idle(p)]
    if not idle_owned:
        return None
    den = min(idle_owned, key=lambda p: dp(src, p))
    if dp(src, den) > DEN_RADIUS:
        return None
    threatened = []
    for p in idle_owned:
        fi, ei = incoming_counts(p, my_fleets, enemy_fleets, ang_vel)
        net = p.ships + fi - ei
        if ei > fi and net < DEFEND_NET:
            threatened.append((net, dp(src, p), p))
    if not threatened:
        return None
    _, _, target = min(threatened)
    return target

def support_rotating_value(src, rot, idle, ang_vel):
    arc = arc_travel(rot, ang_vel, src.x, src.y)
    if arc > OPENING_MAX_ARC:
        return 1e9
    idle_to_src_x, idle_to_src_y = src.x - idle.x, src.y - idle.y
    idle_to_rot_x, idle_to_rot_y = rot.x - idle.x, rot.y - idle.y
    denom = max(1e-6, math.hypot(idle_to_src_x, idle_to_src_y) * math.hypot(idle_to_rot_x, idle_to_rot_y))
    cos_from_idle = (idle_to_src_x * idle_to_rot_x + idle_to_src_y * idle_to_rot_y) / denom
    opposite_bonus = 10.0 if cos_from_idle < -0.25 else 0.0
    return rot.ships * 3.0 + dp(src, rot) + 0.7 * dp(rot, idle) + 1.5 * arc - opposite_bonus

def locked_target(src, targets, incoming, av):
    candidates = []
    for tgt in targets:
        if incoming.get(tgt.id, 0) <= 0:
            continue
        if tgt.owner == -1:
            need = target_need(tgt, incoming)
        else:
            need = enemy_need(src, tgt, max(1, av), incoming)
        if need <= 0:
            continue
        d = dp(src, tgt)
        if d <= LOCKED_TARGET_RADIUS and need <= max(1, int(av * LOCKED_MAX_NEED_RATIO)):
            candidates.append((d, tgt.ships, tgt.id, tgt, need))
    if not candidates:
        return None, 0
    _, _, _, tgt, need = min(candidates)
    return tgt, need


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

    # Build incoming: existing in-flight fleets to non-owned planets. Assign
    # each fleet to one best target so orbiting targets do not get "forgotten"
    # when their position shifts between turns.
    for fl in my_fleets:
        tgt = fleet_target(fl, neutral_p + enemy_p, ang_vel)
        if tgt is not None:
            incoming[tgt.id] = incoming.get(tgt.id, 0) + fl.ships

    threatened = {}

    # ── Phase 1: Emergency defense / evacuation ───────────────────────────────
    for p in my_p:
        fi, ei = incoming_counts(p, my_fleets, enemy_fleets, ang_vel)
        net = p.ships + fi - ei
        threatened[p.id] = ei > fi and net < DEFEND_NET

        if ei > fi and net <= 0:
            av = avail(p) - EVAC_RESERVE
            if av > 0:
                escape_targets = [
                    q for q in my_p
                    if q.id != p.id and p.ships + fi + av < ei
                ]
                if escape_targets:
                    fire(p, min(escape_targets, key=lambda q: dp(p, q)), av)
                    continue
                if not is_idle(p):
                    rotating_neutrals = [
                        n for n in neutral_p
                        if not is_idle(n) and target_need(n, incoming) <= av
                    ]
                    if rotating_neutrals:
                        tgt = min(rotating_neutrals, key=lambda n: dp(p, n))
                        fire(p, tgt, av)
                        continue
                else:
                    bigger_idle = [
                        q for q in my_p
                        if q.id != p.id and is_idle(q) and q.production > p.production
                    ]
                    if bigger_idle:
                        fire(p, min(bigger_idle, key=lambda q: dp(p, q)), av)
                    else:
                        fallback = [q for q in my_p if q.id != p.id]
                        if fallback:
                            fire(p, min(fallback, key=lambda q: dp(p, q)), av)

            if is_idle(p):
                needed = max(1, ei - p.ships - fi + 1)
                for d in sorted([q for q in my_p if q.id != p.id], key=lambda q: dp(q, p)):
                    av = avail(d)
                    if av <= 0:
                        continue
                    to_send = min(av, needed)
                    fire(d, p, to_send)
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
                    fire(d, p, to_send)
                    needed -= to_send
                    if needed <= 0:
                        break

    # ── Single-planet opener ──────────────────────────────────────────────────
    # First move is about speed: take the fastest useful planet, then let the
    # normal owned-planet role logic fan out across idle, rotating, and enemy.
    if step < OPENING_STEPS and len(my_p) == 1 and neutral_p:
        src = my_p[0]
        av = avail(src) - OPENING_RESERVE
        if av >= OPENING_MIN_SEND:
            idle_targets = [n for n in neutral_p if is_idle(n) and target_need(n, incoming) > 0]
            rotating_targets = [n for n in neutral_p if not is_idle(n) and target_need(n, incoming) > 0]
            nearest_idle = min(idle_targets, key=lambda n: dp(src, n), default=None)
            nearest_rot = min(rotating_targets, key=lambda n: dp(src, n), default=None)

            if is_idle(src):
                chosen = None
                if nearest_idle is not None and nearest_rot is not None:
                    idle_d = dp(src, nearest_idle)
                    rot_d = dp(src, nearest_rot)
                    if rot_d + ROTATING_NEAR_MARGIN < idle_d or rot_d < idle_d * ROTATING_NEAR_FACTOR:
                        chosen = nearest_rot
                    else:
                        chosen = nearest_idle
                else:
                    chosen = nearest_idle or nearest_rot

                if chosen is not None:
                    need = target_need(chosen, incoming)
                    send_n = min(av, need)
                    if send_n >= EXPAND_MIN_SEND or send_n >= need:
                        fire(src, chosen, send_n)
            else:
                lowest_idle = min(idle_targets, key=lambda n: (n.ships, dp(src, n)), default=None)
                idle_choice = lowest_idle or nearest_idle
                idle_is_lowest_and_near = False
                if idle_choice is not None and nearest_idle is not None:
                    idle_is_lowest_and_near = (
                        idle_choice.id == nearest_idle.id
                        or dp(src, idle_choice) <= dp(src, nearest_idle) + IDLE_LOWEST_NEAR_MARGIN
                    )

                if idle_choice is not None and idle_is_lowest_and_near:
                    need = target_need(idle_choice, incoming)
                    send_n = min(av, need)
                    if send_n >= OPENING_MIN_SEND or send_n >= need:
                        fire(src, idle_choice, send_n)
                else:
                    support_rots = rotating_targets
                    if idle_choice is not None:
                        support_rots = sorted(
                            rotating_targets,
                            key=lambda n: support_rotating_value(src, n, idle_choice, ang_vel)
                        )
                    support_rot = support_rots[0] if support_rots else None

                    if support_rot is not None:
                        need = target_need(support_rot, incoming)
                        send_n = min(av, need)
                        if send_n >= OPENING_MIN_SEND or send_n >= need:
                            fire(src, support_rot, send_n)
                    elif idle_choice is not None:
                        need = target_need(idle_choice, incoming)
                        send_n = min(av, need)
                        if send_n >= OPENING_MIN_SEND or send_n >= need:
                            fire(src, idle_choice, send_n)

    # ── Early economy roles ───────────────────────────────────────────────────
    # Rotating planets are unreliable launch bases early: save until they can
    # finish an idle planet in one launch. Idle planets are the spreaders: take
    # nearest idle planets first, then coordinate on valuable rotating targets.
    if step < EXPAND_STEPS and neutral_p:
        for src in sorted(my_p, key=lambda p: (is_idle(p), p.production, avail(p)), reverse=True):
            launched = 0
            while launched < EXPAND_MAX_PER_PLANET:
                if is_idle(src):
                    av = avail(src) - EXPAND_RESERVE
                    if av < EXPAND_MIN_SEND:
                        break

                    locked, locked_need = locked_target(src, neutral_p + enemy_p, incoming, av)
                    if locked is not None:
                        send_n = min(av, locked_need)
                        if send_n >= EXPAND_MIN_SEND or send_n >= locked_need:
                            fire(src, locked, send_n)
                            launched += 1
                            continue

                    idle_targets = []
                    rotating_targets = []
                    enemy_targets = []
                    for tgt in neutral_p:
                        need = target_need(tgt, incoming)
                        if need <= 0:
                            continue
                        if is_idle(tgt):
                            value = idle_capture_value(src, tgt) + frontier_idle_bonus(src, tgt, enemy_p)
                            idle_targets.append((dp(src, tgt), -value, tgt.id, tgt, need))
                        else:
                            score = rotating_capture_value(src, tgt, ang_vel) + rotating_raid_bonus(src, tgt, enemy_p)
                            if score > -1e8:
                                front_score = rotating_front_score(src, tgt, ang_vel) + rotating_raid_bonus(src, tgt, enemy_p)
                                rotating_targets.append((-score, -front_score, dp(src, tgt), tgt.id, tgt, need))
                    for tgt in enemy_p:
                        need = enemy_need(src, tgt, av, incoming)
                        if need <= 0:
                            continue
                        d = dp(src, tgt)
                        score = tgt.production * 65.0 / (need + 8.0) / (1.0 + 0.025 * d)
                        enemy_targets.append((-score, d, tgt.id, tgt, need))

                    chosen = None
                    need = 0
                    should_take_rotating_first = False
                    if idle_targets and rotating_targets:
                        nearest_idle_d = min(item[0] for item in idle_targets)
                        nearest_rot_d = min(item[2] for item in rotating_targets)
                        should_take_rotating_first = (
                            nearest_rot_d + ROTATING_NEAR_MARGIN < nearest_idle_d
                            or nearest_rot_d < nearest_idle_d * ROTATING_NEAR_FACTOR
                        )

                    if should_take_rotating_first:
                        _, _, _, _, chosen, need = min(rotating_targets)
                    elif idle_targets:
                        _, _, _, chosen, need = min(idle_targets)
                    elif rotating_targets:
                        _, _, _, _, chosen, need = min(rotating_targets)
                    elif enemy_targets:
                        _, _, _, chosen, need = min(enemy_targets)

                    if chosen is None:
                        break

                    send_n = min(av, need)
                    if send_n < EXPAND_MIN_SEND and send_n < need:
                        break
                    fire(src, chosen, send_n)
                    launched += 1
                else:
                    av = avail(src) - ROTATING_SAVE_RESERVE
                    if av < OPENING_MIN_SEND:
                        break

                    den_target = den_defense_target(src, my_p, my_fleets, enemy_fleets, ang_vel)
                    if den_target is not None:
                        fire(src, den_target, max(0, avail(src)))
                        launched += 1
                        continue

                    claimed_idle = []
                    high_idle = []
                    high_rotating = []
                    high_enemy = []
                    for tgt in neutral_p:
                        need = target_need(tgt, incoming)
                        if need <= 0:
                            continue
                        if is_idle(tgt):
                            value = idle_capture_value(src, tgt) + frontier_idle_bonus(src, tgt, enemy_p)
                            item = (dp(src, tgt), -value, tgt.id, tgt, need)
                            if incoming.get(tgt.id, 0) > 0:
                                claimed_idle.append(item)
                            high_idle.append((-value, dp(src, tgt), tgt.id, tgt, need))
                        else:
                            score = rotating_front_score(src, tgt, ang_vel) + rotating_raid_bonus(src, tgt, enemy_p)
                            if score > -1e8:
                                high_rotating.append((-score, dp(src, tgt), tgt.id, tgt, need))
                    for tgt in enemy_p:
                        need = enemy_need(src, tgt, av, incoming)
                        if need <= 0:
                            continue
                        d = dp(src, tgt)
                        raid = rotating_raid_bonus(src, tgt, enemy_p) if not is_idle(src) else 0.0
                        score = (tgt.production * 70.0 + raid) / (need + 8.0) / (1.0 + 0.025 * d)
                        high_enemy.append((-score, d, tgt.id, tgt, need))

                    chosen = None
                    need = 0
                    if claimed_idle:
                        _, _, _, chosen, need = min(claimed_idle)
                    elif launched % 2 == 0 and high_idle:
                        _, _, _, chosen, need = min(high_idle)
                    elif high_rotating:
                        _, _, _, chosen, need = min(high_rotating)
                    elif high_idle:
                        _, _, _, chosen, need = min(high_idle)
                    elif high_enemy:
                        _, _, _, chosen, need = min(high_enemy)

                    if chosen is None:
                        break
                    send_n = min(av, need)
                    if send_n < OPENING_MIN_SEND and send_n < need:
                        break
                    fire(src, chosen, send_n)
                    launched += 1

    # ── Power state: enough bases, widen into new idles and opponents ─────────
    my_idle_count = sum(1 for p in my_p if is_idle(p))
    my_rot_count = len(my_p) - my_idle_count
    if my_idle_count >= POWER_IDLE_COUNT and my_rot_count >= POWER_ROT_COUNT:
        power_planets = sorted(my_p, key=lambda p: (p.production, avail(p)), reverse=True)
        expand_slots = max(1, len(power_planets) // 2)
        expand_ids = {p.id for p in power_planets[:expand_slots]}

        for src in power_planets:
            role = "expand" if src.id in expand_ids else "attack"
            launched = 0
            while launched < POWER_MAX_PER_PLANET:
                av = avail(src) - POWER_RESERVE
                if av < EXPAND_MIN_SEND:
                    break

                target = None
                need = 0

                if not is_idle(src):
                    den_target = den_defense_target(src, my_p, my_fleets, enemy_fleets, ang_vel)
                    if den_target is not None:
                        fire(src, den_target, max(0, avail(src)))
                        launched += 1
                        continue

                locked, locked_need = locked_target(src, neutral_p + enemy_p, incoming, av)
                if locked is not None:
                    target = locked
                    need = locked_need

                idle_candidates = []
                rotating_candidates = []
                if target is None:
                    for tgt in neutral_p:
                        n = target_need(tgt, incoming)
                        if n <= 0:
                            continue
                        d = dp(src, tgt)
                        if is_idle(tgt):
                            score = (tgt.production * 90.0 + frontier_idle_bonus(src, tgt, enemy_p)) / (n + 4.0) / (1.0 + 0.025 * d)
                            idle_candidates.append((-score, d, tgt.id, tgt, n))
                        else:
                            score = (tgt.production * 75.0 + rotating_raid_bonus(src, tgt, enemy_p)) / (n + 5.0) / (1.0 + 0.025 * d)
                            rotating_candidates.append((-score, d, tgt.id, tgt, n))

                enemy_candidates = []
                if target is None:
                    for tgt in enemy_p:
                        n = enemy_need(src, tgt, av, incoming)
                        if n <= 0:
                            continue
                        d = dp(src, tgt)
                        score = tgt.production * 95.0 / (n + 8.0) / (1.0 + 0.02 * d)
                        enemy_candidates.append((-score, d, tgt.id, tgt, n))

                    if role == "expand" and idle_candidates:
                        _, _, _, target, need = min(idle_candidates)
                    elif role == "expand" and rotating_candidates:
                        _, _, _, target, need = min(rotating_candidates)
                    elif role == "attack" and enemy_candidates:
                        _, _, _, target, need = min(enemy_candidates)
                    elif role == "attack" and rotating_candidates:
                        _, _, _, target, need = min(rotating_candidates)
                    elif role == "expand" and enemy_candidates:
                        _, _, _, target, need = min(enemy_candidates)
                    elif role == "attack" and idle_candidates:
                        _, _, _, target, need = min(idle_candidates)

                if target is None:
                    break

                send_n = min(av, need)
                if send_n < EXPAND_MIN_SEND and send_n < need:
                    break
                fire(src, target, send_n)
                launched += 1

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

        locked, locked_need = locked_target(src, neutral_p + enemy_p, incoming, av)
        if locked is not None:
            send_n = min(av, locked_need)
            if send_n > 0:
                fire(src, locked, send_n)
                continue

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
