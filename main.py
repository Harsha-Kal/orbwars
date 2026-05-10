"""
Orbit Wars – Competitive Agent
================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

Decision order each turn (agent()):
  1. Emergency defense (immediate threat response)
  2. Cheap recent-loss recapture / counterattack
  3. Finish-zero / save-under-attack / fall-recapture / doomed evacuation
  4. Local high-production neutral capture / neutral races
  5. Missed-neutral force (stuck detector)
  6. Forced opening tempo (first 2-3 planets, bypass safety layers)
  7. Midgame control
  8. Mission coordinator — breach-kill, snipe, expansion, sync attacks,
     anti-leader, collapse, final drain
  9. Fallback tempo (last resort)

Key systems:
  - MissionLedger: central coordination, prevents trickle attacks
  - WorldModel: per-turn state, ownership simulation, source safety checks
  - Forced opening tempo: FORCED_OPENING_STEP / FORCED_OPENING_PLANETS gates
  - No comet targeting (enforced in valid_fleet_launch and every selector)
"""

import math
import time
import heapq
from dataclasses import dataclass
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# ── game / geometry ───────────────────────────────────────────────────────────
SUN_X, SUN_Y   = 50.0, 50.0
SUN_R          = 10.0
MAX_SPEED      = 6.0
ROTATION_LIMIT = 50.0
TOTAL_STEPS    = 500
HIT_MARGIN     = 3.0    # acceptable aimed-point miss beyond planet radius
INCOMING_T     = 0.85   # angular tolerance for fleet-target matching

# ── timing / phases ───────────────────────────────────────────────────────────
EARLY_STEPS      = 55    # lighter reserves before this step
LATE_GAME_STEPS  = 350   # prefer enemy planets over neutrals
FINAL_STEPS      = 460   # drain pass: send all idle ships to any reachable target

# ── opening / expansion ───────────────────────────────────────────────────────
FORCED_OPENING_STEP     = 80    # step below which forced opening tempo is active
FORCED_OPENING_PLANETS  = 3     # owned-planet count below which forced opening is active
OPENING_STUCK_STEP      = 25    # if still 1 planet at this step: force-capture
NEAREST_LOCK_DIST       = 28.0
NEAREST_LOCK_ETA        = 18.0
NEAREST_LOCK_MAX_SOURCES       = 3
PROACTIVE_EXPANSION_MAX_SOURCES = 4
STALL_FORCE_TURNS  = 6
STALL_FORCE_SHIPS  = 180
MAX_WAVE_WAIT      = 3
MIN_WAVE_FRACTION  = 0.90

# ── defense ───────────────────────────────────────────────────────────────────
DEFEND_NET           = 8     # reinforce if net projected ships < this
DEFENSE_ETA_HORIZON  = 30    # only count inbound fleets arriving within this many turns
FRONTLINE_DIST       = 25.0  # distance to nearest enemy below which = frontline
FRONTLINE_FAR_ATTACK_DIST = 42.0
HOSTILE_MARGIN_BASE  = 5
HOSTILE_MARGIN_CAP   = 16

# ── fleet ratio caps ──────────────────────────────────────────────────────────
FLEET_RATIO_SOFT = 0.60   # above this: block new non-critical launches
FLEET_RATIO_HARD = 0.70   # above this: only defense/reinforce/finish allowed

# ── finish-zero capture ───────────────────────────────────────────────────────
FINISH_ZERO_MAX_SHIPS = 2
FINISH_ZERO_MAX_ETA   = 16.0
FINISH_ZERO_NEAR_DIST = 34.0

# ── save / fall-turn recapture ────────────────────────────────────────────────
FALL_RECAPTURE_LOOKAHEAD = 10
FALL_RECAPTURE_HORIZON   = 28
DOOMED_EVAC_HORIZON      = 24
DOOMED_EVAC_MIN_SHIPS    = 8

# ── snipe ─────────────────────────────────────────────────────────────────────
SNIPE_LOOKAHEAD = 22
SNIPE_ETA_SLACK = 2

# ── mission coordinator ───────────────────────────────────────────────────────
LOCAL_HUB_SHIPS  = 60      # planet with >= this many ships acts as a command hub
LOCAL_HUB_RADIUS = 40.0    # command hub targets within this distance first
ETA_SYNC_WINDOW  = 8       # max turn spread for a sync attack to be valid
SOURCE_COOLDOWN_TURNS = 8

# ── high-value neutral production race ───────────────────────────────────────
LOCAL_PRODUCTION_MIN_PROD = 4
LOCAL_PRODUCTION_PREMIER_PROD = 5
LOCAL_PRODUCTION_MAX_SHIPS = 35
LOCAL_PRODUCTION_MAX_DIST = 46.0
LOCAL_PRODUCTION_MAX_ETA = 22.0
LOCAL_PRODUCTION_HUB_SHIPS = 70
LOCAL_PRODUCTION_RACE_MARGIN = 3.0

# ── cheap recent-loss response ───────────────────────────────────────────────
CHEAP_RECAPTURE_LOCAL_DIST = 35.0
CHEAP_RECAPTURE_HOLD_MARGIN = 7
CHEAP_RECAPTURE_TOTAL_FLEET_CAP = 0.25
CHEAP_RECAPTURE_LOCAL_SURPLUS_CAP = 0.60
CHEAP_RECAPTURE_PRIORITY  = 118.0
COUNTERATTACK_PRIORITY    = 116.0

# ── mission priority constants ─────────────────────────────────────────────────
PRIORITY_SAVE_ATTACK_BASE    = 108.0
PRIORITY_FINISH_ZERO_BASE    = 101.0
PRIORITY_HV_RACE_BASE        = 128.0
PRIORITY_HV_CAPTURE_BASE     = 114.0
PRIORITY_HV_PREMIER_STEP     = 42.0   # prod>=5 bonus after step 50
PRIORITY_HV_PREMIER_EARLY    = 28.0   # prod>=5 bonus before step 50
PRIORITY_BREACH_KILL_BASE    = 90.0
PRIORITY_BREACH_KILL_CLOSE   = 15.0   # extra bonus when <=5 enemy planets
PRIORITY_MG_INFECT_BASE      = 126.0
PRIORITY_MG_BREACH_BASE      = 88.0
PRIORITY_MG_CONTEST_BASE     = 78.0
PRIORITY_SYNC_ATTACK_BASE    = 62.0
PRIORITY_LOCAL_STRIKE_BASE   = 72.0
PRIORITY_ANTI_LEADER_BASE    = 55.0
PRIORITY_COLLAPSE_BASE       = 52.0
PRIORITY_PROTECT_LEAD_BASE   = 76.0
PRIORITY_FINAL_DRAIN_BASE    = 40.0
PRIORITY_ENDGAME_CONSOL_BASE = 44.0
PRIORITY_CHAIN_PLAN_BASE     = 88.0
PRIORITY_OPENING_360_BASE    = 85.0

# ── rotational expansion / opportunistic strikes ───────────────────────────────
OCCUPIABLE_MAX_DIST           = 42.0  # cluster distance limit for nearest-occupiable scan
OCCUPIABLE_HOLD_MARGIN        = 5     # extra ships above need for hold plan estimate
WEAKNESS_DROP_THRESHOLD       = 12    # ship drop (vs expected) that marks an enemy as weak
PRIORITY_NEAREST_OCCUPIABLE   = 95.0  # capture nearest neutral / weak enemy
PRIORITY_EXPAND_FROM_HUB      = 83.0  # hub → next nearby planet
PRIORITY_OPPORTUNISTIC_STRIKE = 68.0  # attack recently weakened enemy planet
PRIORITY_HUB_REINFORCE_BASE   = 92.0  # reinforce vulnerable rotational hub

# ── MAIN19_TEMPO_ARBITER ───────────────────────────────────────────────────────
ARBITER_NEAREST_MAX_DIST  = 32.0  # max source→target distance for arbiter nearest check
ARBITER_NEAREST_MAX_ETA   = 18.0  # max ETA for arbiter nearest check
ARBITER_HOLD_MARGIN       = 5     # extra ships above need required for hold check
ARBITER_HV_PROD_OVERRIDE  = 4     # HV neutral must have this prod to override nearest

# ── rotational hubs ────────────────────────────────────────────────────────────
ROTATIONAL_HUB_TTL        = 30    # turns a newly captured planet stays marked as hub
ROTATIONAL_HUB_REINFORCE_THRESH = 8  # min ships below reserve before hub reinforce fires

# ── planet radius roles ───────────────────────────────────────────────────────
SMALL_RADIUS  = 1.2
LARGE_RADIUS  = 2.3

# ── start-type-aware opening ──────────────────────────────────────────────────
LARGE_START_PROD_THRESH   = 4     # prod >= this → LARGE_START (primary)
LARGE_START_RADIUS_THRESH = 2.4   # radius >= this → LARGE_START (tie-breaker; was 6.0)
SMALL_START_PROD_THRESH   = 1     # prod <= this → SMALL_START (primary)
SMALL_START_RADIUS_THRESH = 1.1   # radius <= this → SMALL_START (tie-breaker; was 3.5)
LARGE_START_STALL_STEP_1  = 15    # LARGE_START: force capture if still 1 planet here
LARGE_START_STALL_STEP_3  = 35    # LARGE_START: force sweep if < 3 planets here
SMALL_START_STALL_STEP    = 20    # SMALL_START: force escape if still 1 planet here

# ── breach-kill ───────────────────────────────────────────────────────────────
BREACH_KILL_DIST    = 45.0
BREACH_KILL_STEP_MIN = 70
BREACH_ETA_SYNC     = 10

# ── endgame ───────────────────────────────────────────────────────────────────
PROTECT_LEAD_STEP      = 420
PROTECT_LEAD_REMAINING = 85
TURTLE_SHIP_THRESH     = 40   # avg ships per enemy planet above this → turtle

# ── search / simulation ───────────────────────────────────────────────────────
SOFT_DEADLINE    = 0.82
SIM_HORIZON      = 100
MAX_GROUP_SOURCES = 7

# ── strategic state evaluation ───────────────────────────────────────────────
STATE_PLANET_CONTROL_WEIGHT = 42.0
STATE_PRODUCTION_WEIGHT     = 34.0
STATE_STATIONED_SHIP_WEIGHT = 0.85
STATE_USEFUL_FLEET_WEIGHT   = 0.30
STATE_IDLE_FLEET_PENALTY    = 0.55
STATE_FLEET_RATIO_PENALTY   = 120.0

# ── SEARCH_ATTACK_PLANNER ─────────────────────────────────────────────────────
SEARCH_MAX_CANDIDATES  = 10     # top proposals evaluated in the beam pass
SEARCH_SELECT_LIMIT    = 2      # max offensive missions committed per turn
SEARCH_MIN_SCORE       = -35.0  # reject proposals whose search score < this
SEARCH_TIME_BUDGET     = 0.10   # seconds budget for the full search pass
SEARCH_RELAY_HORIZON   = 20.0   # max ETA for relay capture to count toward relay value

# ── fleet packet discipline ───────────────────────────────────────────────────
MIN_SEND_SHIPS   = 10   # no fleet below this size; prevents panic micro-fleets
SEND_GRANULARITY = 5    # every launch size must be a multiple of this value

# ── map control phases (planet % based, step number is secondary) ─────────────
PHASE_OPENING_PCT  = 0.12   # my control below this → OPENING_EXPANSION
PHASE_SWEEP_PCT    = 0.28   # my control below this → LOCAL_SWEEP
PHASE_EXPAND_PCT   = 0.45   # my control below this → EXPANSION_CONTROL
# above 0.45 → CONTACT or COLLAPSE based on enemy proximity

# ── bridge planet detection ───────────────────────────────────────────────────
BRIDGE_RELAY_DIST   = 50.0  # max distance from bridge to next useful capture target
BRIDGE_MIN_SHORTCUT = 0.15  # bridge must shorten direct route by at least this fraction

# ── enemy-attack scoring nudges ──────────────────────────────────────────────
ENEMY_GATE_NEUTRAL_PCT     = 0.35  # neutral density where enemy attacks get a soft score penalty
ENEMY_GATE_MAX_MY_PCT      = 0.28  # my control below which neutrals are softly preferred
ENEMY_GATE_WEAK_SHIPS      = 12    # <= this ships → very weak enemy
ENEMY_GATE_WEAK_LOCAL      = 15    # <= this ships → "weak" for LOCAL_SWEEP phase rule

# ── small planet bridge logic ─────────────────────────────────────────────────
SMALL_BRIDGE_THRESHOLD    = 20.0  # small_bridge_score above this → worth capturing
SMALL_BRIDGE_CAPTURE_DIST = 32.0  # only capture small bridge planets within this cluster_dist
SMALL_STORAGE_CAPTURE_DIST = 20.0 # pure-storage small planets captured only when this close

# ── always-on capture opportunity engine ──────────────────────────────────────
CAPTURE_OPP_MAX_DIST      = 55.0  # max cluster_distance to scan for opportunities
CAPTURE_OPP_MAX_ETA       = 30.0  # max ETA for an opportunity candidate
CAPTURE_OPP_MIN_SCORE     = -60.0 # discard opportunities below this score
CAPTURE_OPP_MAX_PROPOSALS = 5     # max proposals returned per turn
CAPTURE_OPP_DRAINED_DROP  = 10    # ship drop (vs expected) to count as recently drained
# Soft-priority penalties; lower absolute value = gentler deduction
CAPTURE_OPP_4P_EARLY_PEN  = 18.0  # 4-player + early + non-local enemy
CAPTURE_OPP_NEUTRAL_PEN   = 15.0  # many neutrals remain + enemy + not urgent

# ── 4-player ──────────────────────────────────────────────────────────────────
FOUR_P_ATTACK_STEP = 80
FOUR_P_EXPAND_STEP = 100
FOUR_P_CORNER_STRATEGY_STEP_MAX = 220
FOUR_P_CORNER_LOCAL_DIST = 48.0
FOUR_P_ADJACENT_BRIDGE_DIST = 58.0

# ── midgame control ───────────────────────────────────────────────────────────
MIDGAME_START_STEP          = 55     # first step where MIDGAME_CONTROL is active
MIDGAME_END_STEP            = 220    # after this, late-game logic takes over
MIDGAME_FLEET_SOFT          = 0.55   # tighter than global: block attacks above this
MIDGAME_FLEET_HARD          = 0.65   # only critical missions above this
MIDGAME_FLEET_PANIC         = 0.75   # emergency-only territory
MIDGAME_FRONT_RADIUS        = 42.0   # one-front clustering radius
MIDGAME_STABILITY_THRESHOLD = 0.35   # below this: pause all offensive missions
MIDGAME_CONTEST_MAX_DIST    = 52.0   # max cluster_distance for a neutral to be contested
MIDGAME_MAJOR_MISSION_LIMIT = 2      # anti-scatter: at most 1-2 meaningful missions
MIDGAME_MIN_AVG_SHIPS       = 10.0   # reject busy-looking low-impact waves
MIDGAME_MIN_WAVE_SHIPS      = 18     # enemy attacks must have real punch
MIDGAME_ATTACK_SOURCE_MAX   = 4      # grouped attacks should be compact and nearby

# ── bowwow-style launchpad chain ──────────────────────────────────────────────
LAUNCHPAD_PROD_MIN       = 4
LAUNCHPAD_SURPLUS_MIN    = 24
LAUNCHPAD_RADIUS         = 46.0
LAUNCHPAD_CHAIN_ETA      = 22.0
LAUNCHPAD_RECENT_TTL     = 25
LAUNCHPAD_RECAPTURE_TTL  = 45
CHAIN_PROD4_BONUS        = 90.0
CHAIN_PROD5_BONUS        = 150.0

DEBUG = False


def round_up_to_granularity(amount, granularity=SEND_GRANULARITY):
    if amount <= 0:
        return 0
    return int(math.ceil(float(amount) / granularity) * granularity)


def round_down_to_granularity(amount, granularity=SEND_GRANULARITY):
    if amount <= 0:
        return 0
    return int(math.floor(float(amount) / granularity) * granularity)


def normalize_send_amount(need):
    """
    Round `need` up to the nearest valid fleet-packet size.
    All offensive launches must be >= MIN_SEND_SHIPS (10) and a multiple of
    SEND_GRANULARITY (5): valid sizes are 10, 15, 20, 25, 30, ...
    Returns 0 if need <= 0.

    Callers must still verify the source can spare the normalized amount;
    if it cannot, skip the source rather than sending a smaller invalid amount.
    """
    if need <= 0:
        return 0
    need = max(MIN_SEND_SHIPS, int(need))
    return int(math.ceil(need / SEND_GRANULARITY) * SEND_GRANULARITY)


# ── persistent global state ───────────────────────────────────────────────────
_wave_reservation = {
    "target_id": None,
    "source_ids": [],
    "started_step": -1,
    "launch_by_step": -1,
    "required_ships": 0,
    "reason": ""
}

_prev_owners: dict = {}           # planet_id -> owner at start of previous turn
_prev_ships: dict = {}            # planet_id -> ship count at start of previous turn
_rotational_hubs: dict = {}           # player -> {planet_id -> step when marked as hub}
_primary_launchpads: dict = {}        # player -> {planet_id -> step when marked as launchpad}
_start_type_cache: dict = {}          # player -> "LARGE" | "SMALL" | "MEDIUM"


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


def radius_class(p):
    if p.radius <= SMALL_RADIUS:
        return "SMALL"
    if p.radius >= LARGE_RADIUS:
        return "LARGE"
    return "MEDIUM"


def is_static_planet(p):
    return is_idle(p)


def is_storage_planet(p):
    return radius_class(p) == "SMALL"


def _corner_index_for_planet(p):
    """Return a stable 0..3 corner index using the sun/map center as origin."""
    right = p.x >= SUN_X
    lower = p.y >= SUN_Y
    if right and lower:
        return 0
    if not right and lower:
        return 1
    if not right and not lower:
        return 2
    return 3


def _my_start_corner_index(world):
    start = next((p for p in world.initial_planets.values() if p.owner == world.player), None)
    if start is None:
        start = world.my_planets[0] if world.my_planets else None
    return _corner_index_for_planet(start) if start is not None else 0


def classify_corner_zone(world, planet):
    start_idx = _my_start_corner_index(world)
    idx = _corner_index_for_planet(planet)
    diff = (idx - start_idx) % 4
    if diff == 0:
        zone = "my_start_corner"
    elif diff == 1:
        zone = "clockwise_adjacent_corner"
    elif diff == 3:
        zone = "counterclockwise_adjacent_corner"
    else:
        zone = "opposite_corner"
    seen = getattr(world, "_corner_debug_seen", set())
    key = (planet.id, zone)
    if key not in seen:
        world.add_debug(f"CORNER_ZONE_CLASSIFIED p{planet.id} zone={zone}")
        seen.add(key)
        world._corner_debug_seen = seen
    return zone


def _is_corner_control_target(world, target):
    if target is None or not world.is_four_player or world.is_comet(target):
        return False
    zone = classify_corner_zone(world, target)
    if zone == "my_start_corner" and radius_class(target) in ("MEDIUM", "LARGE"):
        return True
    if zone in ("clockwise_adjacent_corner", "counterclockwise_adjacent_corner"):
        return radius_class(target) in ("MEDIUM", "LARGE")
    if zone == "opposite_corner":
        return (
            radius_class(target) in ("MEDIUM", "LARGE")
            and (
                _adjacent_corner_secured(world, "clockwise_adjacent_corner")
                and _adjacent_corner_secured(world, "counterclockwise_adjacent_corner")
            )
        )
    return False


# ── neutral race classifier ────────────────────────────────────────────────────

def neutral_race_status(world, target):
    """Classify a neutral as SAFE / CONTESTED / ENEMY_FAVORED based on ETA delta."""
    my_eta, enemy_eta = world.reaction_times(target)
    delta = my_eta - enemy_eta  # positive = we're slower
    if delta <= -3.0:
        return "SAFE", my_eta, enemy_eta
    elif delta <= 4.0:
        return "CONTESTED", my_eta, enemy_eta
    else:
        return "ENEMY_FAVORED", my_eta, enemy_eta

def is_safe_neutral(world, target):
    status, _, _ = neutral_race_status(world, target)
    return status == "SAFE"

def is_contested_neutral(world, target):
    status, _, _ = neutral_race_status(world, target)
    return status == "CONTESTED"

def is_enemy_favored_neutral(world, target):
    status, _, _ = neutral_race_status(world, target)
    return status == "ENEMY_FAVORED"

def enemy_earliest_capture_turn(world, target):
    """Earliest turn an enemy fleet could realistically take this planet."""
    inbound = [
        eta for eta, owner, ships in world.arrivals_by_target.get(target.id, [])
        if owner != world.player and ships > 0
    ]
    if inbound:
        return min(inbound)
    return min(
        (world.eta(e, target, max(1, int(e.ships) // 2)) for e in world.enemy_planets),
        default=999.0,
    )

def validate_grouped_launch(world, tgt, planned):
    """
    Verify a grouped attack will flip ownership.
    ETA spread: <=3 for neutrals, <=6 for enemy planets.
    Returns (ok, reason).
    """
    if not planned:
        return False, "no sources"
    total = sum(s for _, s, _, _ in planned)
    eta_vals = [e for _, _, _, e in planned]
    spread = max(eta_vals) - min(eta_vals) if len(eta_vals) >= 2 else 0.0
    max_eta = max(eta_vals)
    if tgt.owner == -1 and spread > 3.0:
        return False, f"neutral spread={spread:.1f}>3"
    if tgt.owner not in (-1, world.player) and spread > 6.0:
        return False, f"enemy spread={spread:.1f}>6"
    eval_turn = max(1, int(math.ceil(max_eta)))
    extra = tuple(
        (max(1, int(math.ceil(e))), world.player, int(s))
        for _, s, _, e in planned if int(s) > 0
    )
    owner_after, _ = world.projected_state(tgt.id, eval_turn, extra_arrivals=extra)
    if owner_after != world.player:
        return False, f"won't flip at t={eval_turn}"
    return True, ""


class StrategyMode:
    OPENING_TEMPO = "OPENING_TEMPO"
    EXPAND_CHAIN = "EXPAND_CHAIN"
    CONTEST_HUBS = "CONTEST_HUBS"
    RECOVER_AND_HOLD = "RECOVER_AND_HOLD"
    SAFE_EXPANSION = "SAFE_EXPANSION"
    CONTEST_NEUTRALS = "CONTEST_NEUTRALS"
    ANTI_LEADER = "ANTI_LEADER"
    BEHIND_STEAL = "BEHIND_STEAL"
    TURTLE_BREAKER = "TURTLE_BREAKER"
    COLLAPSE = "COLLAPSE"
    FORCE_WAVE = "FORCE_WAVE"
    FINAL_DRAIN = "FINAL_DRAIN"
    FOUR_PLAYER_EXPAND_FIRST = "FOUR_PLAYER_EXPAND_FIRST"


class MidgameState:
    STABLE_EXPAND        = "STABLE_EXPAND"
    CONTEST_NEUTRALS     = "CONTEST_NEUTRALS"
    FRONTLINE_STABILIZE  = "FRONTLINE_STABILIZE"
    FOCUSED_BREACH       = "FOCUSED_BREACH"
    RECOVER_AND_HOLD     = "RECOVER_AND_HOLD"


class ControlPhase:
    """Strategic phase driven by map-control percentage, not step number."""
    OPENING_EXPANSION = "OPENING_EXPANSION"   # my_pct < 12% or <= 3 planets
    LOCAL_SWEEP       = "LOCAL_SWEEP"         # 12%–28%, many neutrals remain
    EXPANSION_CONTROL = "EXPANSION_CONTROL"   # 28%–45%, cluster stable
    CONTACT           = "CONTACT"             # frontier reached or enemy near
    COLLAPSE          = "COLLAPSE"            # dominant control, finish opponent


@dataclass
class RoutePlan:
    target_id: int
    first_target_id: int
    score: float
    cost: float
    route: list


@dataclass
class NearestCandidate:
    score: float
    target_id: int
    source_id: int
    need: int
    eta: float
    distance: float
    enemy_eta: float
    grouped_pool: int
    route: list
    can_lock: bool
    reason: str


@dataclass
class MissionProposal:
    kind: str
    target_id: int
    priority: float
    required_ships: int
    planned_sources: list   # [(src_id, ships, angle, eta), ...]
    eta_min: float
    eta_max: float
    reason: str

    @property
    def score(self):
        return self.priority

    @property
    def sources(self):
        return [src_id for src_id, _, _, _ in self.planned_sources]

    @property
    def arrival_turn(self):
        return int(math.ceil(self.eta_max))

    @property
    def converts_ownership(self):
        return self.kind in OFFENSIVE_MISSIONS


@dataclass
class MissionLedgerEntry:
    mission_id: int
    mission_type: str
    target_id: int
    source_ids: list
    ships_committed: int
    required_ships: int
    launch_step: int
    expected_arrival_steps: list
    status: str
    reason: str


MISSION_TYPE_ALIASES = {
    "DEFEND": "DEFEND_HOLD",
    "REINFORCE": "REINFORCE_CAPTURE",
    "HOLD_CAPTURE": "REINFORCE_CAPTURE",
    "CAPTURE_HIGH_PROD_NEUTRAL": "LOCAL_PRODUCTION_CAPTURE",
    "REINFORCE_FRONTIER_HUB": "REINFORCE_CAPTURE",
    "RECAPTURE_LOST_PLANET": "RECAPTURE_LOST",
    "CONTEST_ENEMY_PRODUCTION": "SYNC_ATTACK",
    "RESCUE_BEFORE_FALL": "SAVE_UNDER_ATTACK",
    "CONTEST_NEUTRAL": "HIGH_VALUE_NEUTRAL_RACE",
    "COLLAPSE_FINISH": "COLLAPSE",
    "ENDGAME_CONSOLIDATE": "DEFEND_HOLD",
    "FINISH_ENEMY_HOME": "BREACH_KILL",
    "FINISH_CAPTURE": "FINISH_ZERO_CAPTURE",
    "LOCAL_STRIKE": "SYNC_ATTACK",
    "LOCAL_PRODUCTION_CAPTURE": "LOCAL_PRODUCTION_CAPTURE",
    "HIGH_VALUE_NEUTRAL_RACE": "HIGH_VALUE_NEUTRAL_RACE",
}

MISSION_TYPES = {
    "DEFEND_HOLD",
    "SAVE_UNDER_ATTACK",
    "RECAPTURE_LOST",
    "FINISH_ZERO_CAPTURE",
    "CAPTURE_NEUTRAL",
    "LOCAL_PRODUCTION_CAPTURE",
    "HIGH_VALUE_NEUTRAL_RACE",
    "REINFORCE_CAPTURE",
    "SYNC_ATTACK",
    "SNIPE_NEUTRAL",
    "DOOMED_EVACUATION",
    "BREACH_KILL",
    "COLLAPSE",
    "FINAL_DRAIN",
}

CRITICAL_MISSIONS = {
    "DEFEND_HOLD",
    "SAVE_UNDER_ATTACK",
    "RECAPTURE_LOST",
    "FINISH_ZERO_CAPTURE",
    "DOOMED_EVACUATION",
    "FINAL_DRAIN",
}

REINFORCEMENT_MISSIONS = {
    "DEFEND_HOLD",
    "SAVE_UNDER_ATTACK",
    "REINFORCE_CAPTURE",
}

OFFENSIVE_MISSIONS = {
    "RECAPTURE_LOST",
    "FINISH_ZERO_CAPTURE",
    "CAPTURE_NEUTRAL",
    "LOCAL_PRODUCTION_CAPTURE",
    "HIGH_VALUE_NEUTRAL_RACE",
    "SYNC_ATTACK",
    "SNIPE_NEUTRAL",
    "BREACH_KILL",
    "COLLAPSE",
    "FINAL_DRAIN",
}

_mission_ledger: dict = {}          # player -> mission_id -> MissionLedgerEntry
_mission_seq: dict = {}             # player -> next id
_recently_reinforced: dict = {}     # player -> planet_id -> last step
_doomed_owned_targets: dict = {}    # player -> planet_id -> last doomed step


def canonical_mission_type(kind):
    mission_type = MISSION_TYPE_ALIASES.get(kind, kind)
    return mission_type if mission_type in MISSION_TYPES else "SYNC_ATTACK"


SMALL_PACKET_MISSIONS = {
    "DEFEND_HOLD",
    "SAVE_UNDER_ATTACK",
    "RECAPTURE_LOST",
    "FINISH_ZERO_CAPTURE",
    "DOOMED_EVACUATION",
    "REINFORCE_CAPTURE",
    "FINAL_DRAIN",
}


def mission_allows_small_packet(mission_type):
    return canonical_mission_type(mission_type) in SMALL_PACKET_MISSIONS


def valid_packet_size(mission_type, ships):
    ships = int(ships)
    if ships <= 0:
        return False
    if mission_allows_small_packet(mission_type):
        return True
    return ships >= MIN_SEND_SHIPS and ships % SEND_GRANULARITY == 0


class MissionLedger:
    """Persistent mission registry that gates launches through mission validity."""

    def __init__(self, world):
        self.world = world
        _mission_ledger.setdefault(world.player, {})
        _mission_seq.setdefault(world.player, 1)
        self.entries = _mission_ledger[world.player]
        self.update_active_missions()

    def get(self, mission_id):
        if mission_id is None:
            return None
        return self.entries.get(mission_id)

    def update_active_missions(self):
        world = self.world
        for entry in list(self.entries.values()):
            if entry.status in ("completed", "invalidated") and world.step - entry.launch_step > 20:
                del self.entries[entry.mission_id]
                continue
            if world.step - entry.launch_step > 90:
                del self.entries[entry.mission_id]
                continue
            if entry.status not in ("planned", "active"):
                continue
            tgt = world.planet_by_id.get(entry.target_id)
            if tgt is None or world.is_comet(tgt):
                self.invalidate(entry.mission_id, "mission invalidated: target gone")
                continue
            if world.step - entry.launch_step > 45:
                self.invalidate(entry.mission_id, "mission invalidated: stale")
                continue
            if tgt.owner == world.player and entry.mission_type in OFFENSIVE_MISSIONS:
                threat = world.real_incoming_threat(tgt)
                if threat["deficit"] <= 0:
                    entry.status = "completed"
                    world.add_debug(f"MISSION_COMPLETE {entry.mission_type} id={entry.mission_id} target=p{tgt.id}")
            if tgt.owner != world.player and entry.mission_type in REINFORCEMENT_MISSIONS:
                self.invalidate(entry.mission_id, "mission invalidated: target flipped")
            if tgt.owner != world.player and entry.mission_type in OFFENSIVE_MISSIONS:
                src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
                if src is not None:
                    need = world.required_ships_to_capture(tgt, src)
                    if world.incoming_to_targets.get(tgt.id, 0) >= need:
                        entry.status = "completed"
                        world.add_debug(
                            f"MISSION_COMPLETE {entry.mission_type} id={entry.mission_id} target=p{tgt.id} reason=already handled by incoming"
                        )
                    elif need > max(25, world.my_total_ships * 1.4):
                        self.invalidate(entry.mission_id, "mission invalidated: target too heavily defended")

    def create(self, mission_type, target_id, source_ids, required_ships, expected_arrival_steps, reason):
        mission_type = canonical_mission_type(mission_type)
        for entry in self.entries.values():
            if (
                entry.status in ("planned", "active")
                and entry.mission_type == mission_type
                and entry.target_id == target_id
                and self.world.step - entry.launch_step <= 10
            ):
                entry.source_ids = sorted(set(entry.source_ids) | set(source_ids))
                entry.required_ships = max(entry.required_ships, int(required_ships))
                entry.expected_arrival_steps.extend(int(math.ceil(e)) for e in expected_arrival_steps)
                entry.reason = reason or entry.reason
                self.world.add_debug(
                    f"MISSION_REUSE {mission_type} id={entry.mission_id} target=p{target_id} reason={reason}"
                )
                return entry.mission_id
        mission_id = _mission_seq[self.world.player]
        _mission_seq[self.world.player] = mission_id + 1
        entry = MissionLedgerEntry(
            mission_id=mission_id,
            mission_type=mission_type,
            target_id=target_id,
            source_ids=list(source_ids),
            ships_committed=0,
            required_ships=int(required_ships),
            launch_step=self.world.step,
            expected_arrival_steps=[int(math.ceil(e)) for e in expected_arrival_steps],
            status="planned",
            reason=reason,
        )
        self.entries[mission_id] = entry
        self.world.add_debug(
            f"MISSION_SELECT {mission_type} id={mission_id} target=p{target_id} "
            f"sources={list(source_ids)} required={int(required_ships)} "
            f"eta_spread={(max(expected_arrival_steps) - min(expected_arrival_steps)) if expected_arrival_steps else 0:.1f} "
            f"reason={reason}"
        )
        return mission_id

    def create_from_proposal(self, prop):
        return self.create(
            prop.kind,
            prop.target_id,
            [src_id for src_id, _, _, _ in prop.planned_sources],
            prop.required_ships,
            [eta for _, _, _, eta in prop.planned_sources],
            prop.reason,
        )

    def record_launch(self, mission_id, source_id, ships, eta):
        if mission_id is None:
            return
        entry = self.entries.get(mission_id)
        if entry is None:
            return
        entry.status = "active"
        entry.ships_committed += int(ships)
        if source_id not in entry.source_ids:
            entry.source_ids.append(source_id)
        entry.expected_arrival_steps.append(int(math.ceil(eta)))
        self.world.add_debug(
            f"MISSION_LAUNCH {entry.mission_type} id={entry.mission_id} target=p{entry.target_id} "
            f"src=p{source_id} ships={int(ships)} eta={eta:.1f} committed={entry.ships_committed}/{entry.required_ships}"
        )

    def invalidate(self, mission_id, reason):
        entry = self.entries.get(mission_id)
        if entry is None or entry.status == "invalidated":
            return
        entry.status = "invalidated"
        self.world.add_debug(
            f"MISSION_INVALIDATED {entry.mission_type} id={entry.mission_id} target=p{entry.target_id} reason={reason}"
        )


def _read(obs, key, default=None):
    return obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default)


class WorldModel:
    """Per-turn state, cached aiming, simple timelines, reserves, and reactions."""

    def __init__(self, obs):
        self.player = int(_read(obs, "player", 0))
        self.step = int(_read(obs, "step", 0))
        av = _read(obs, "angular_velocity", None)
        if av is None:
            av = _read(obs, "angularVelocity", 0.033)
        self.ang_vel = float(av)
        self.planets = [Planet(*p) for p in (_read(obs, "planets", []) or [])]
        self.fleets = [Fleet(*f) for f in (_read(obs, "fleets", []) or [])]
        self.initial_planets = {
            Planet(*p).id: Planet(*p)
            for p in (_read(obs, "initial_planets", []) or _read(obs, "planets", []) or [])
        }
        self.comets = _read(obs, "comets", []) or []
        self.comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
        self.planet_by_id = {p.id: p for p in self.planets}

        self.normal_planets = [p for p in self.planets if p.id not in self.comet_ids]
        self.my_planets = [p for p in self.normal_planets if p.owner == self.player]
        self.neutral_planets = [p for p in self.normal_planets if p.owner == -1]
        self.enemy_planets = [p for p in self.normal_planets if p.owner not in (-1, self.player)]
        self.my_fleets = [f for f in self.fleets if f.owner == self.player]
        self.enemy_fleets = [f for f in self.fleets if f.owner != self.player]
        self.remaining = max(1, TOTAL_STEPS - self.step)

        self.my_total_ships = sum(int(p.ships) for p in self.my_planets) + sum(int(f.ships) for f in self.my_fleets)
        self.enemy_total_ships = sum(int(p.ships) for p in self.enemy_planets) + sum(int(f.ships) for f in self.enemy_fleets)
        self.my_prod = sum(int(p.production) for p in self.my_planets)
        self.enemy_prod = sum(int(p.production) for p in self.enemy_planets)
        self.ships_by_owner = {}
        self.prod_by_owner = {}
        self.planets_by_owner = {}
        for p in self.normal_planets:
            if p.owner == -1:
                continue
            self.ships_by_owner[p.owner] = self.ships_by_owner.get(p.owner, 0) + int(p.ships)
            self.prod_by_owner[p.owner] = self.prod_by_owner.get(p.owner, 0) + int(p.production)
            self.planets_by_owner[p.owner] = self.planets_by_owner.get(p.owner, 0) + 1
        for f in self.fleets:
            self.ships_by_owner[f.owner] = self.ships_by_owner.get(f.owner, 0) + int(f.ships)
        self.enemy_prod_by_owner = {}
        self.enemy_ships_by_owner = {}
        self.enemy_planets_by_owner = {}
        for p in self.enemy_planets:
            self.enemy_prod_by_owner[p.owner] = self.enemy_prod_by_owner.get(p.owner, 0) + int(p.production)
            self.enemy_ships_by_owner[p.owner] = self.enemy_ships_by_owner.get(p.owner, 0) + int(p.ships)
            self.enemy_planets_by_owner[p.owner] = self.enemy_planets_by_owner.get(p.owner, 0) + 1
        for f in self.enemy_fleets:
            self.enemy_ships_by_owner[f.owner] = self.enemy_ships_by_owner.get(f.owner, 0) + int(f.ships)

        self.arrivals_by_target = {}
        self.incoming_to_targets = {}
        self.enemy_incoming_to_targets = {}
        self.debug_events = []
        self._build_arrivals()

        self.leader = None
        self.leader_score = 0
        for owner in set(self.enemy_prod_by_owner) | set(self.enemy_ships_by_owner):
            score = self.strategic_state_score(owner)
            if score > self.leader_score:
                self.leader = owner
                self.leader_score = score
        self.my_score = self.strategic_state_score(self.player)

        self.shot_cache = {}
        self.reaction_cache = {}
        self.timeline_cache = {}
        self.keep_needed_map = {}
        self.attack_budget_map = {}
        self.reaction_time_map = {}
        self.committed = {}
        self.offensive_ships = 0
        self.wave_attempted = False
        self.features = {}
        self.recently_reinforced_planets = dict(_recently_reinforced.get(self.player, {}))
        self.doomed_owned_targets = set(
            pid for pid, step in _doomed_owned_targets.get(self.player, {}).items()
            if self.step - step <= 18
        )
        self.mission_ledger = MissionLedger(self)
        self._compute_features()
        self._detect_game_type()
        self._build_policy_maps()

    def _detect_game_type(self):
        initial_enemy_owners = set(
            p.owner for p in self.initial_planets.values()
            if p.owner not in (-1, self.player)
        )
        current_enemy_owners = (
            set(p.owner for p in self.enemy_planets)
            | set(f.owner for f in self.enemy_fleets)
        )
        all_enemy_owners = initial_enemy_owners | current_enemy_owners
        if len(all_enemy_owners) >= 2:
            self.game_type = "FOUR_PLAYER"
        else:
            self.game_type = "TWO_PLAYER"
        self.is_four_player = self.game_type == "FOUR_PLAYER"
        self.is_two_player = self.game_type == "TWO_PLAYER"

    def _build_arrivals(self):
        for fl in self.fleets:
            tgt = fleet_target(fl, self.planets, self.ang_vel)
            if tgt is None:
                continue
            eta = travel_turns(dist(fl.x, fl.y, tgt.x, tgt.y), max(1, fl.ships))
            self.arrivals_by_target.setdefault(tgt.id, []).append((eta, fl.owner, int(fl.ships)))
            if fl.owner == self.player and tgt.owner != self.player:
                self.incoming_to_targets[tgt.id] = self.incoming_to_targets.get(tgt.id, 0) + int(fl.ships)
            elif fl.owner != self.player:
                self.enemy_incoming_to_targets[tgt.id] = self.enemy_incoming_to_targets.get(tgt.id, 0) + int(fl.ships)

    def _fleet_has_control_job(self, fl, tgt, eta):
        if tgt is None or self.is_comet(tgt) or eta > 45:
            return False
        arrivals = self.arrivals_by_target.get(tgt.id, [])
        owner_incoming = sum(ships for a_eta, owner, ships in arrivals if owner == fl.owner and a_eta <= eta + 6)
        enemy_incoming = sum(ships for a_eta, owner, ships in arrivals if owner != fl.owner and a_eta <= eta + 6)
        if tgt.owner == fl.owner:
            return enemy_incoming > 0 and owner_incoming >= enemy_incoming
        projected_need = int(tgt.ships) + 1
        if tgt.owner != -1:
            projected_need += int(tgt.production) * max(1, int(math.ceil(eta)))
        return owner_incoming >= projected_need

    def flying_ship_breakdown(self, owner):
        fleets = [f for f in self.fleets if f.owner == owner]
        useful_flying = 0
        idle_flying = 0
        for fl in fleets:
            tgt = fleet_target(fl, self.planets, self.ang_vel)
            if tgt is None:
                idle_flying += int(fl.ships)
                continue
            eta = travel_turns(dist(fl.x, fl.y, tgt.x, tgt.y), max(1, int(fl.ships)))
            if self._fleet_has_control_job(fl, tgt, eta):
                useful_flying += int(fl.ships)
            else:
                idle_flying += int(fl.ships)
        return useful_flying, idle_flying

    def strategic_state_score(self, owner):
        owned = [p for p in self.normal_planets if p.owner == owner]
        owned_prod = sum(int(p.production) for p in owned)
        stationed_ships = sum(int(p.ships) for p in owned)
        useful_flying, idle_flying = self.flying_ship_breakdown(owner)
        total_force = max(1, stationed_ships + useful_flying + idle_flying)
        fleet_ratio = (useful_flying + idle_flying) / total_force
        high_fleet_penalty = max(0.0, fleet_ratio - FLEET_RATIO_SOFT) * STATE_FLEET_RATIO_PENALTY
        if owner == self.player:
            self.add_debug(
                f"PLANET_VALUE_OVER_FLYING_SHIPS planets={len(owned)} prod={owned_prod} "
                f"stationed={stationed_ships} useful_flying={useful_flying} idle_flying={idle_flying}"
            )
            self.add_debug(f"PRODUCTION_VALUE_PRIORITY prod={owned_prod} weight={STATE_PRODUCTION_WEIGHT}")
            if useful_flying or idle_flying:
                self.add_debug(
                    f"FLYING_SHIP_DISCOUNT_APPLIED useful={useful_flying} idle={idle_flying}"
                )
            if idle_flying:
                self.add_debug(f"SCATTERED_FLEET_PENALTY ships={idle_flying}")
            if high_fleet_penalty > 0:
                self.add_debug(
                    f"HIGH_FLEET_RATIO_STATE_PENALTY ratio={fleet_ratio:.2f} penalty={high_fleet_penalty:.1f}"
                )
        return (
            len(owned) * STATE_PLANET_CONTROL_WEIGHT
            + owned_prod * STATE_PRODUCTION_WEIGHT
            + stationed_ships * STATE_STATIONED_SHIP_WEIGHT
            + useful_flying * STATE_USEFUL_FLEET_WEIGHT
            - idle_flying * STATE_IDLE_FLEET_PENALTY
            - high_fleet_penalty
        )

    def _build_policy_maps(self):
        """Marco/824-style per-turn policy state: reserves, attack budgets, and reaction ETAs."""
        for p in self.my_planets:
            tl = self.simulate_planet_timeline(p, min(SIM_HORIZON, DEFENSE_ETA_HORIZON))
            keep = max(self.reserve_for(p), tl.get("keep_needed", 0))
            cap = max(20, int(p.ships * 0.55))
            keep = min(int(p.ships), min(int(keep), cap))
            self.keep_needed_map[p.id] = keep
            self.attack_budget_map[p.id] = max(0, int(p.ships) - keep)
        for tgt in self.normal_planets:
            if tgt.owner == self.player:
                continue
            self.reaction_time_map[tgt.id] = self.reaction_times(tgt)

    def _compute_features(self):
        nearest_enemy = min((dp(m, e) for m in self.my_planets for e in self.enemy_planets), default=999.0)
        high_neutrals = sum(1 for n in self.neutral_planets if n.production >= 4)
        enemy_avg_ships = sum(p.ships for p in self.enemy_planets) / max(1, len(self.enemy_planets))
        incoming_threat_count = sum(1 for p in self.my_planets if self.real_incoming_threat(p)["deficit"] > 0)
        self.features = {
            "prod_ratio": self.my_prod / max(1, self.enemy_prod),
            "ship_ratio": self.my_total_ships / max(1, self.enemy_total_ships),
            "leader_ahead": self.leader is not None and self.leader_score > self.my_score * 1.22,
            "neutral_count": len(self.neutral_planets),
            "high_neutral_count": high_neutrals,
            "nearest_enemy": nearest_enemy,
            "enemy_avg_ships": enemy_avg_ships,
            "incoming_threat_count": incoming_threat_count,
            "ahead": self.my_score > max(1, self.leader_score) * 1.18 or self.my_prod > max(1, self.enemy_prod) * 1.25,
            "behind": self.my_score * 1.25 < max(1, self.leader_score) or self.my_prod * 1.3 < max(1, self.enemy_prod),
            "late": self.step > 380 or self.remaining < 120,
            "final": self.step >= FINAL_STEPS or self.remaining < 45,
        }

    def aim(self, src, tgt, ships):
        key = (src.id, tgt.id, int(ships), int(self.step))
        if key not in self.shot_cache:
            self.shot_cache[key] = aim_valid(src, tgt, self.ang_vel, int(ships))
        return self.shot_cache[key]

    def eta(self, src, tgt, ships):
        return travel_turns(dp(src, tgt), max(1, int(ships)))

    def real_incoming_threat(self, p, horizon=DEFENSE_ETA_HORIZON):
        friendly = 0
        enemy = 0
        for eta, owner, ships in self.arrivals_by_target.get(p.id, []):
            if eta > horizon:
                continue
            if owner == self.player:
                friendly += ships
            else:
                enemy += ships
        net = int(p.ships) + friendly - enemy
        return {"friendly": friendly, "enemy": enemy, "net": net, "deficit": max(0, DEFEND_NET - net) if enemy > friendly else 0}

    def nearest_enemy_distance(self, p):
        return min((dp(p, e) for e in self.enemy_planets), default=999.0)

    def is_frontline(self, p):
        return self.nearest_enemy_distance(p) < FRONTLINE_DIST

    def is_backline(self, p):
        return not self.is_frontline(p)

    def is_comet(self, p):
        return p is not None and p.id in self.comet_ids

    def valid_target(self, p):
        return p is not None and p.id not in self.comet_ids and p.owner != self.player

    def enemy_pressure_near(self, p, radius=30.0):
        pressure = 0.0
        for enemy in self.enemy_planets:
            d = max(1.0, dp(p, enemy))
            if d <= radius:
                pressure += max(0, int(enemy.ships) - 3) * (radius - d + 1.0) / radius
        for fl in self.enemy_fleets:
            d = max(1.0, dist(p.x, p.y, fl.x, fl.y))
            if d <= radius:
                pressure += int(fl.ships) * 0.5
        return pressure

    def cluster_distance(self, p, count=3):
        distances = sorted(dp(p, m) for m in self.my_planets)
        if not distances:
            return 999.0
        top = distances[:count]
        return sum(top) / len(top)

    def reserve_for(self, p):
        if hasattr(self, "keep_needed_map") and p.id in self.keep_needed_map:
            cap = max(20, int(p.ships * 0.55))
            keep = int(self.keep_needed_map[p.id])
            if is_storage_planet(p):
                keep = max(keep, int(p.ships * 0.65), 6 + int(p.production) * 2)
                self.add_debug(f"SMALL_STORAGE_RESERVE_HELD p{p.id} reserve={keep}")
                return keep
            return min(keep, cap)
        if self.step < EARLY_STEPS:
            base = min(3, 1 + int(p.production))
        elif self.step < LATE_GAME_STEPS:
            base = 4 + int(p.production)
        else:
            base = 6 + int(p.production)
        threat = self.real_incoming_threat(p)
        raw = base
        if threat["deficit"] > 0:
            raw += threat["deficit"] + 3
        elif self.nearest_enemy_distance(p) < FRONTLINE_DIST:
            raw += max(2, int(p.production))
        if is_storage_planet(p):
            raw += max(4, int(p.production) * 2)
            raw = max(raw, int(p.ships * 0.60))
            self.add_debug(f"SMALL_STORAGE_RESERVE_HELD p{p.id} reserve={raw}")
        cap = max(20, int(p.ships * (0.45 if self.step < EARLY_STEPS else 0.55)))
        if is_storage_planet(p):
            cap = max(cap, int(p.ships * 0.80))
        return min(int(raw), cap)

    def surplus(self, p):
        if hasattr(self, "attack_budget_map") and p.id in self.attack_budget_map:
            return max(0, self.attack_budget_map[p.id] - self.committed.get(p.id, 0))
        return max(0, int(p.ships) - self.committed.get(p.id, 0) - self.reserve_for(p))

    def target_need_now(self, tgt):
        return int(tgt.ships) + 1 + self.enemy_incoming_to_targets.get(tgt.id, 0) - self.incoming_to_targets.get(tgt.id, 0)

    def simulate_planet_timeline(self, planet, horizon=SIM_HORIZON, planned=()):
        key = (planet.id, horizon, tuple(sorted(planned)))
        if key in self.timeline_cache:
            return self.timeline_cache[key]
        owner = planet.owner
        ships = int(planet.ships)
        arrivals = list(self.arrivals_by_target.get(planet.id, []))
        arrivals.extend(planned)
        buckets = {}
        for eta, who, count in arrivals:
            turn = max(1, min(horizon, int(math.ceil(eta))))
            buckets.setdefault(turn, []).append((who, int(count)))

        min_owned = ships if owner == self.player else 0
        fall_turn = None
        owner_at = {}
        ships_at = {}
        for t in range(1, horizon + 1):
            if owner != -1:
                ships += int(planet.production)
            if t in buckets:
                forces = {}
                forces[owner] = forces.get(owner, 0) + max(0, ships)
                for who, count in buckets[t]:
                    forces[who] = forces.get(who, 0) + count
                ranked = sorted(forces.items(), key=lambda item: item[1], reverse=True)
                if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
                    owner, ships = -1, 0
                else:
                    owner, ships = ranked[0][0], ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)
            owner_at[t] = owner
            ships_at[t] = ships
            if owner == self.player:
                min_owned = min(min_owned, ships)
            elif planet.owner == self.player and fall_turn is None:
                fall_turn = t
        result = {
            "owner_at": owner_at,
            "ships_at": ships_at,
            "fall_turn": fall_turn,
            "min_owned": min_owned,
            "keep_needed": max(0, -min_owned + DEFEND_NET),
            "holds": fall_turn is None,
        }
        self.timeline_cache[key] = result
        return result

    def ships_needed_to_capture(self, src, tgt, ships_hint=None, planned=()):
        hint = max(1, int(ships_hint if ships_hint is not None else self.surplus(src)))
        eta = self.eta(src, tgt, hint)
        if tgt.owner == -1:
            need = self.target_need_now(tgt)
            if need <= 0:
                return 0
        else:
            need = int(tgt.ships + tgt.production * eta) + 1 + self.enemy_incoming_to_targets.get(tgt.id, 0) - self.incoming_to_targets.get(tgt.id, 0)
            if need <= 0:
                return 0
        if tgt.owner != -1:
            need += min(HOSTILE_MARGIN_CAP, HOSTILE_MARGIN_BASE + int(tgt.production))
        elif is_idle(tgt):
            need += 2
        return max(0, int(need))

    def required_ships_to_capture(self, tgt, src=None):
        src = src or min(self.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            return int(tgt.ships) + 1
        return self.ships_needed_to_capture(src, tgt)

    def can_hold_after_capture(self, tgt, eta, sent, final_all_in=False):
        if final_all_in or tgt.owner not in (-1, self.player):
            return True if final_all_in else self.nearest_enemy_distance(tgt) > 16 or tgt.production >= 3
        planned = ((eta, self.player, sent),)
        tl = self.simulate_planet_timeline(tgt, min(SIM_HORIZON, int(eta) + 35), planned=planned)
        return tl["owner_at"].get(min(SIM_HORIZON, int(eta) + 25), tgt.owner) == self.player

    def reaction_times(self, tgt):
        if tgt.id in self.reaction_cache:
            return self.reaction_cache[tgt.id]
        my_t = min((self.eta(p, tgt, max(1, min(int(p.ships), max(1, self.target_need_now(tgt))))) for p in self.my_planets), default=999.0)
        enemy_t = min((self.eta(e, tgt, max(1, int(e.ships))) for e in self.enemy_planets), default=999.0)
        self.reaction_cache[tgt.id] = (my_t, enemy_t)
        return my_t, enemy_t

    def add_debug(self, message):
        if DEBUG:
            self.debug_events.append(message)

    def is_recently_reinforced(self, p):
        last = self.recently_reinforced_planets.get(p.id)
        return last is not None and self.step - last <= SOURCE_COOLDOWN_TURNS

    def source_is_safe_for(self, src, tgt, mission_type, ships, mission_reason=""):
        if mission_type == "FINAL_DRAIN":
            return True, ""
        # Opening tempo fast-path: bypass all safety layers for CAPTURE_NEUTRAL during the
        # first 2-3 planet acquisitions.  Only block on a real incoming enemy fleet.
        if (mission_type == "CAPTURE_NEUTRAL"
                and self.step < FORCED_OPENING_STEP
                and len(self.my_planets) < FORCED_OPENING_PLANETS):
            threat = self.real_incoming_threat(src)
            if threat["deficit"] > 0:
                return False, "source unsafe: under attack"
            opening_reserve = 1 if len(self.my_planets) <= 1 else 2
            remaining = int(src.ships) - self.committed.get(src.id, 0) - int(ships)
            if remaining < opening_reserve:
                return False, f"opening reserve blocked {remaining}<{opening_reserve}"
            return True, ""
        evac_fall_turn = None
        if mission_type == "DOOMED_EVACUATION":
            evac_tl = self.simulate_planet_timeline(src, DOOMED_EVAC_HORIZON)
            evac_fall_turn = evac_tl["fall_turn"]
        evacuating = (
            (mission_type == "DOOMED_EVACUATION" and evac_fall_turn is not None)
            or "evac" in mission_reason
        )
        critical = mission_type in CRITICAL_MISSIONS
        threat = self.real_incoming_threat(src)
        if is_storage_planet(src) and mission_type in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK"):
            self.add_debug(f"SMALL_STORAGE_RELEASE_DEFENSE src=p{src.id} target=p{getattr(tgt, 'id', '?')}")
        if (
            is_storage_planet(src)
            and mission_type == "REINFORCE_CAPTURE"
            and tgt is not None
            and radius_class(tgt) == "LARGE"
        ):
            self.add_debug(f"SMALL_STORAGE_RELEASE_HOLD_LAUNCHPAD src=p{src.id} target=p{tgt.id}")
        if threat["deficit"] > 0 and mission_type not in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK") and not evacuating:
            return False, "source unsafe: under attack"
        remaining = int(src.ships) - self.committed.get(src.id, 0) - int(ships)
        reserve = self.reserve_for(src)
        if remaining < reserve and not critical and not evacuating:
            return False, f"source unsafe: below reserve {remaining}<{reserve}"
        if self.is_recently_reinforced(src) and mission_type in OFFENSIVE_MISSIONS and not evacuating:
            return False, "source unsafe: recently reinforced cooldown"
        if is_storage_planet(src) and mission_type in OFFENSIVE_MISSIONS and not evacuating:
            allowed_storage_release = mission_type in (
                "RECAPTURE_LOST", "FINISH_ZERO_CAPTURE", "FINAL_DRAIN"
            )
            target_role = radius_class(tgt) if tgt is not None else "SMALL"
            launchpad_escape = (
                mission_type in ("CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE")
                and tgt is not None
                and target_role in ("MEDIUM", "LARGE")
                and dp(src, tgt) <= CAPTURE_OPP_MAX_DIST
            )
            # Allow small storage to fund a bridge-value small planet
            small_bridge_escape = (
                mission_type == "CAPTURE_NEUTRAL"
                and tgt is not None
                and target_role == "SMALL"
                and small_bridge_score(self, tgt) >= SMALL_BRIDGE_THRESHOLD
                and dp(src, tgt) <= SMALL_BRIDGE_CAPTURE_DIST
            )
            local_support = tgt is not None and dp(src, tgt) <= CHEAP_RECAPTURE_LOCAL_DIST
            if allowed_storage_release:
                if mission_type == "RECAPTURE_LOST":
                    self.add_debug(f"SMALL_STORAGE_RELEASE_RECAPTURE src=p{src.id} target=p{getattr(tgt, 'id', '?')}")
            elif launchpad_escape or small_bridge_escape:
                pass
            elif local_support and mission_type in ("CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE"):
                pass
            else:
                self.add_debug(
                    f"SMALL_STORAGE_SKIP_FAR_ATTACK src=p{src.id} target=p{getattr(tgt, 'id', '?')} mission={mission_type}"
                )
                return False, "source unsafe: small storage reserve"
        if (
            self.is_four_player
            and is_static_planet(src)
            and radius_class(src) == "LARGE"
            and mission_type in ("SYNC_ATTACK", "BREACH_KILL", "COLLAPSE")
            and tgt is not None
            and not _is_corner_control_target(self, tgt)
            and remaining < reserve + max(12, int(src.production) * 4)
        ):
            return False, "source unsafe: static launchpad reserve"
        prev_owner = _prev_owners.get(src.id)
        prev_ships = _prev_ships.get(src.id)
        if prev_owner == self.player and prev_ships is not None and not critical and not evacuating:
            recent_loss = int(prev_ships) + int(src.production) - int(src.ships)
            if recent_loss >= max(5, int(src.production) * 2):
                return False, "source unsafe: recently lost ships"
        if self.is_frontline(src) and mission_type not in CRITICAL_MISSIONS and not evacuating:
            if tgt is not None and dp(src, tgt) > FRONTLINE_FAR_ATTACK_DIST:
                return False, "source unsafe: frontline far mission"
            if remaining < reserve + max(3, int(src.production)):
                return False, "source unsafe: frontline reserve"
        if (
            int(src.ships) >= LOCAL_PRODUCTION_HUB_SHIPS
            and mission_type in ("SYNC_ATTACK", "COLLAPSE", "BREACH_KILL")
            and tgt is not None
            and dp(src, tgt) > LOCAL_HUB_RADIUS
        ):
            local_hv = nearest_high_value_neutral_for_source(self, src)
            if local_hv is not None:
                return False, f"source unsafe: hub has local high-value neutral p{local_hv.id}"
        return True, ""

    def target_owner_at_arrival(self, tgt, eta, planned=()):
        horizon = max(1, min(SIM_HORIZON, int(math.ceil(eta)) + 2))
        tl = self.simulate_planet_timeline(tgt, horizon, planned=planned)
        turn = max(1, min(horizon, int(math.ceil(eta))))
        return tl["owner_at"].get(turn, tgt.owner), tl["ships_at"].get(turn, int(tgt.ships)), tl

    def projected_state(self, target_id, eval_turn, extra_arrivals=()):
        target = self.planet_by_id.get(target_id)
        if target is None:
            return None, 0
        turn = max(1, min(SIM_HORIZON, int(math.ceil(eval_turn))))
        normalized = tuple(
            (max(1, int(math.ceil(eta))), owner, int(ships))
            for eta, owner, ships in extra_arrivals
            if int(ships) > 0 and max(1, int(math.ceil(eta))) <= turn
        )
        tl = self.simulate_planet_timeline(target, turn, planned=normalized)
        return tl["owner_at"].get(turn, target.owner), tl["ships_at"].get(turn, int(target.ships))

    def min_ships_to_own_by(self, target_id, eval_turn, attacker_owner, arrival_turn=None, extra_arrivals=(), upper_bound=None):
        eval_turn = max(1, min(SIM_HORIZON, int(math.ceil(eval_turn))))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            return int(upper_bound or self.my_total_ships) + 1
        owner_before, ships_before = self.projected_state(target_id, eval_turn, extra_arrivals=extra_arrivals)
        if owner_before == attacker_owner:
            return 0

        def owns_with(count):
            owner_after, _ = self.projected_state(
                target_id,
                eval_turn,
                extra_arrivals=tuple(extra_arrivals) + ((arrival_turn, attacker_owner, int(count)),),
            )
            return owner_after == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_with(hi):
                return hi + 1
        else:
            hi = max(1, int(ships_before) + 1)
            cap = max(32, int(self.my_total_ships + self.my_prod * max(2, eval_turn + 2) + 32))
            while hi <= cap and not owns_with(hi):
                hi *= 2
            if hi > cap:
                hi = cap
                if not owns_with(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_with(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def mark_doomed(self, tgt):
        _doomed_owned_targets.setdefault(self.player, {})[tgt.id] = self.step
        self.doomed_owned_targets.add(tgt.id)

    def valid_fleet_launch(self, src, tgt, ships, mission_type, mission_entry=None, planned_sources=None, mission_reason=""):
        if tgt is None or src is None or ships <= 0:
            return False, "mission invalidated: missing source/target"
        if self.is_comet(tgt):
            return False, "comet blocked"
        eta = self.eta(src, tgt, ships)
        if eta > self.remaining - 1 and mission_type != "FINAL_DRAIN":
            return False, "mission invalidated: no arrival time"

        source_ok, source_reason = self.source_is_safe_for(src, tgt, mission_type, ships, mission_reason=mission_reason)
        if not source_ok:
            return False, source_reason

        if mission_type == "DOOMED_EVACUATION":
            src_tl = self.simulate_planet_timeline(src, DOOMED_EVAC_HORIZON)
            fall_turn = src_tl["fall_turn"]
            if fall_turn is None:
                return False, "mission invalidated: evacuation source not doomed"
            if eta >= fall_turn:
                return False, "mission invalidated: evacuation arrives after fall"
            if tgt.owner == self.player:
                tgt_tl = self.simulate_planet_timeline(tgt, DOOMED_EVAC_HORIZON)
                if tgt_tl["fall_turn"] is not None:
                    return False, "mission invalidated: evacuation target unsafe"
                owner_at, _, _ = self.target_owner_at_arrival(tgt, eta, planned=((eta, self.player, ships),))
                if owner_at != self.player:
                    return False, "mission invalidated: evacuation target will not hold"
            else:
                total = sum(s for _, s, _, _ in (planned_sources or [])) or ships
                need = self.ships_needed_to_capture(src, tgt, total)
                if need <= 0 or total < need:
                    return False, "trickle blocked"
                if not self.can_hold_after_capture(tgt, eta, total):
                    return False, "mission invalidated: evacuation capture cannot hold"
            return True, ""

        planned_arrivals = []
        if planned_sources:
            for src_id, planned_ships, _, planned_eta in planned_sources:
                if src_id == src.id:
                    planned_arrivals.append((planned_eta, self.player, planned_ships))

        if mission_type in REINFORCEMENT_MISSIONS:
            if tgt.owner != self.player:
                return False, "target predicted to flip before arrival"
            owner_at, _, tl = self.target_owner_at_arrival(tgt, eta, planned=planned_arrivals)
            if owner_at != self.player:
                if mission_type == "SAVE_UNDER_ATTACK":
                    all_planned = [(e, self.player, s) for _, s, _, e in (planned_sources or [])]
                    full_tl = self.simulate_planet_timeline(tgt, min(SIM_HORIZON, int(math.ceil(max(eta, 1))) + 8), planned=all_planned)
                    if full_tl["fall_turn"] is None:
                        return True, ""
                self.mark_doomed(tgt)
                return False, "target predicted to flip before arrival"
            if mission_type not in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK") and "evac" not in mission_reason:
                threat = self.real_incoming_threat(tgt)
                if threat["deficit"] <= 0 and tl["min_owned"] >= DEFEND_NET:
                    return False, "target already sufficiently defended"
            return True, ""

        if mission_type in OFFENSIVE_MISSIONS:
            if tgt.owner == self.player and mission_type != "RECAPTURE_LOST":
                return False, "mission invalidated: target already mine"
            owner_at, _, _ = self.target_owner_at_arrival(tgt, eta, planned=planned_arrivals)
            if mission_type == "RECAPTURE_LOST" and tgt.owner == self.player and owner_at == self.player:
                return False, "mission invalidated: recapture before fall"
            if mission_type in ("CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE") and owner_at not in (-1, self.player):
                return False, "target predicted enemy before arrival"
            if mission_type == "HIGH_VALUE_NEUTRAL_RACE" and owner_at not in (-1, self.player):
                all_planned = [
                    (planned_eta, self.player, planned_ships)
                    for _, planned_ships, _, planned_eta in (planned_sources or [])
                    if planned_ships > 0
                ] or [(eta, self.player, ships)]
                eval_turn = max(1, int(math.ceil(max(e for e, _, _ in all_planned))))
                owner_after, _ = self.projected_state(tgt.id, eval_turn, extra_arrivals=tuple(all_planned))
                if owner_after != self.player:
                    return False, "target predicted enemy before arrival"
            if mission_type == "SNIPE_NEUTRAL":
                if tgt.owner != -1:
                    return False, "mission invalidated: snipe target not neutral"
                if owner_at not in (-1, self.player):
                    return False, "target predicted enemy before arrival"
            if mission_type in ("SYNC_ATTACK", "COLLAPSE", "BREACH_KILL") and tgt.owner not in (-1, self.player):
                total = sum(s for _, s, _, _ in (planned_sources or [])) or ships
                need = max(1, self.required_ships_to_capture(tgt, src))
                if total < need * MIN_WAVE_FRACTION:
                    return False, "trickle blocked"
                if planned_sources:
                    etas = [e for _, _, _, e in planned_sources]
                    if max(etas) - min(etas) > max(ETA_SYNC_WINDOW, BREACH_ETA_SYNC):
                        return False, "trickle blocked: eta spread"
            if self.incoming_to_targets.get(tgt.id, 0) >= self.required_ships_to_capture(tgt, src):
                return False, "target already doomed"
        return True, ""

    def commit(self, src, tgt, ships, moves, mission_type=None, mission_id=None, planned_sources=None):
        available = int(src.ships) - self.committed.get(src.id, 0)
        ships = int(ships)
        if ships <= 0:
            return False
        if mission_type is None:
            if tgt.owner == self.player:
                mission_type = "REINFORCE_CAPTURE"
            elif tgt.owner == -1:
                mission_type = "CAPTURE_NEUTRAL"
            else:
                mission_type = "SYNC_ATTACK"
        mission_type = canonical_mission_type(mission_type)

        if ships > available:
            self.add_debug(
                f"COMMIT_REJECT_INVALID_PACKET mission={mission_type} src=p{src.id} "
                f"ships={ships} available={available} reason=unavailable"
            )
            return False
        if not valid_packet_size(mission_type, ships):
            self.add_debug(
                f"COMMIT_REJECT_INVALID_PACKET mission={mission_type} src=p{src.id} "
                f"ships={ships} reason=packet_size"
            )
            return False
        self.add_debug(f"COMMIT_VALID_PACKET mission={mission_type} src=p{src.id} ships={ships}")

        ok_launch, reason = self.valid_fleet_launch(
            src, tgt, ships, mission_type, mission_entry=self.mission_ledger.get(mission_id),
            planned_sources=planned_sources,
            mission_reason=(self.mission_ledger.get(mission_id).reason if self.mission_ledger.get(mission_id) else ""),
        )
        if not ok_launch:
            self.add_debug(
                f"MISSION_BLOCK {mission_type} target=p{getattr(tgt, 'id', '?')} src=p{getattr(src, 'id', '?')} ships={ships} reason={reason}"
            )
            if mission_id is not None:
                self.mission_ledger.invalidate(mission_id, reason)
            return False
        angle, ok = self.aim(src, tgt, ships)
        if not ok:
            self.add_debug(f"MISSION_BLOCK {mission_type} target=p{tgt.id} src=p{src.id} ships={ships} reason=aim invalid")
            return False
        if mission_id is None:
            planned = planned_sources or [(src.id, ships, angle, self.eta(src, tgt, ships))]
            mission_id = self.mission_ledger.create(
                mission_type,
                tgt.id,
                [src_id for src_id, _, _, _ in planned],
                sum(s for _, s, _, _ in planned),
                [eta for _, _, _, eta in planned],
                "auto ledger assignment",
            )
        moves.append([src.id, angle, ships])
        self.committed[src.id] = self.committed.get(src.id, 0) + ships
        if tgt.owner != self.player:
            self.incoming_to_targets[tgt.id] = self.incoming_to_targets.get(tgt.id, 0) + ships
            self.offensive_ships += ships
        else:
            _recently_reinforced.setdefault(self.player, {})[tgt.id] = self.step
            self.recently_reinforced_planets[tgt.id] = self.step
        self.mission_ledger.record_launch(mission_id, src.id, ships, eta=self.eta(src, tgt, ships))
        return True


def is_local_enemy_opportunity(world, target):
    """True for close/frontier/weak enemy planets worth considering despite expansion pressure."""
    if target is None or target.owner in (-1, world.player) or world.is_comet(target):
        return False
    cluster_d = world.cluster_distance(target)
    prev = _prev_ships.get(target.id)
    recently_drained = (
        prev is not None
        and (int(prev) + int(target.production) - int(target.ships)) >= CAPTURE_OPP_DRAINED_DROP
    )
    return (
        cluster_d <= MIDGAME_FRONT_RADIUS
        or int(target.ships) <= ENEMY_GATE_WEAK_LOCAL
        or recently_drained
        or int(target.production) >= LOCAL_PRODUCTION_MIN_PROD and cluster_d <= CAPTURE_OPP_MAX_DIST * 0.75
    )


def should_allow_enemy_attack(world, target, mission_type, reason=""):
    """
    Central enemy-attack policy gate.  Returns True when attacking target (which
    must be enemy-owned) is strategically appropriate this turn.

    Call this for every enemy-planet attack before generating or committing a
    proposal.  No enemy attack should bypass it.

    Rules applied in order:
      1. Always block comets.
      2. Always allow: recently-mine, actively-threatening, RECAPTURE_LOST,
         FINISH_ZERO_CAPTURE, enemy has <= 3 planets (collapse possible).
      3. Hard cap: fleet ratio > FLEET_RATIO_HARD → block.
      4. Not holdable AND far from cluster → block.

    Phase, 4-player context, and neutral density are scoring inputs, not
    standalone vetoes.
    """
    # Sanity: gate should only be called for enemy-owned targets
    if target.owner in (-1, world.player):
        return True
    if world.is_comet(target):
        world.add_debug(f"ENEMY_ATTACK_BLOCK comet p{target.id}")
        return False

    # ── Always-allow shortcuts ────────────────────────────────────────────────
    if mission_type in ("RECAPTURE_LOST", "FINISH_ZERO_CAPTURE"):
        world.add_debug(f"ENEMY_ATTACK_ALLOWED reason={mission_type} target=p{target.id}")
        return True

    if _prev_owners.get(target.id) == world.player:
        world.add_debug(f"ENEMY_ATTACK_ALLOWED reason=recently_mine target=p{target.id}")
        return True

    if any(
        dp(my_p, target) <= CHEAP_RECAPTURE_LOCAL_DIST
        and world.real_incoming_threat(my_p)["deficit"] > 0
        for my_p in world.my_planets
    ):
        world.add_debug(f"ENEMY_ATTACK_ALLOWED reason=threatening_mine target=p{target.id}")
        return True

    if len(world.enemy_planets) <= 3:
        world.add_debug(
            f"ENEMY_ATTACK_ALLOWED reason=collapse_possible enemies={len(world.enemy_planets)} "
            f"target=p{target.id}"
        )
        return True

    # ── True safety stops only ────────────────────────────────────────────────
    # Phase, 4-player context, and neutral density are handled by scoring.
    fleet_ratio = compute_fleet_ratio(world)
    src         = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    cluster_d   = world.cluster_distance(target)
    pool        = sum(world.surplus(p) for p in world.my_planets)
    need        = world.ships_needed_to_capture(src, target, pool) if src else pool + 1
    eta         = world.eta(src, target, max(1, need)) if src else 999.0
    can_capture = src is not None and 0 < need <= pool
    can_hold    = world.can_hold_after_capture(target, eta, need) if can_capture else False

    if fleet_ratio > FLEET_RATIO_HARD:
        world.add_debug(
            f"ENEMY_ATTACK_BLOCK_FLEET_RATIO target=p{target.id} ratio={fleet_ratio:.2f}"
        )
        return False

    # Only block "not holdable" when the target is far from cluster; nearby
    # captures can still be useful frontier/control conversions.
    if can_capture and not can_hold and cluster_d > MIDGAME_FRONT_RADIUS:
        world.add_debug(
            f"ENEMY_ATTACK_BLOCK_NOT_HOLDABLE target=p{target.id} "
            f"eta={eta:.1f} cluster_d={cluster_d:.1f}"
        )
        return False

    world.add_debug(
        f"ENEMY_ATTACK_ALLOWED reason={reason or 'standard'} target=p{target.id}"
    )
    return True


def update_ownership_memory(world):
    """Snapshot current planet state for next turn's threat/source-safety checks."""
    global _prev_owners, _prev_ships
    _prev_owners = {p.id: p.owner for p in world.normal_planets}
    _prev_ships = {p.id: int(p.ships) for p in world.normal_planets}


def very_recent_losses(world):
    """Planets that flipped from me to enemy since the previous decision tick."""
    world.add_debug("RECENT_LOSS_SCAN")
    if not _prev_owners:
        return []
    losses = []
    for p in world.enemy_planets:
        if world.is_comet(p):
            continue
        if _prev_owners.get(p.id) == world.player:
            losses.append((p, 0))
    losses.sort(key=lambda item: (world.cluster_distance(item[0]), -int(item[0].production)))
    return losses


def build_cheap_recapture_plan(world, lost):
    """Return a decisive local recapture proposal, or None with a debug marker."""
    world.add_debug(f"CHEAP_RECAPTURE_CHECK p{lost.id}")
    local_sources = sorted(
        [
            src for src in world.my_planets
            if dp(src, lost) <= CHEAP_RECAPTURE_LOCAL_DIST
            and world.real_incoming_threat(src)["deficit"] <= 0
            and world.simulate_planet_timeline(src, 18)["fall_turn"] is None
            and world.surplus(src) >= MIN_SEND_SHIPS
        ],
        key=lambda s: (dp(s, lost), -world.surplus(s)),
    )[:3]
    if not local_sources:
        world.add_debug(f"CHEAP_RECAPTURE_REJECT_NO_LOCAL_SURPLUS p{lost.id}")
        return None

    local_surplus = sum(world.surplus(s) for s in local_sources)
    primary = local_sources[0]
    hold_need = int(lost.ships) + int(lost.production) * 4 + CHEAP_RECAPTURE_HOLD_MARGIN
    need = max(world.ships_needed_to_capture(primary, lost, local_surplus), hold_need)
    max_total = min(
        int(world.my_total_ships * CHEAP_RECAPTURE_TOTAL_FLEET_CAP),
        int(local_surplus * CHEAP_RECAPTURE_LOCAL_SURPLUS_CAP),
    )
    if need <= 0 or need > max_total:
        world.add_debug(
            f"CHEAP_RECAPTURE_REJECT_TOO_EXPENSIVE p{lost.id} need={need} "
            f"cap={max_total} total={world.my_total_ships} local={local_surplus}"
        )
        return None
    if local_surplus < need:
        world.add_debug(
            f"CHEAP_RECAPTURE_REJECT_NO_LOCAL_SURPLUS p{lost.id} pool={local_surplus} need={need}"
        )
        return None

    plan, reason = build_grouped_funding_plan(
        world,
        lost,
        need,
        local_sources,
        "RECAPTURE_LOST",
        max_sources=3,
        eta_spread_limit=6.0,
        allow_small_packets=True,
        require_hold=True,
    )
    if plan is None:
        world.add_debug(f"CHEAP_RECAPTURE_REJECT_NO_LOCAL_SURPLUS p{lost.id} reason={reason}")
        return None
    planned, total, eta_min, eta_max = plan
    if total > max_total:
        world.add_debug(f"CHEAP_RECAPTURE_REJECT_TOO_EXPENSIVE p{lost.id} send={total} cap={max_total}")
        return None

    marker = "CHEAP_RECAPTURE_SELECTED"
    if len(planned) > 1:
        marker += " EARLY_PROD_GROUPED_CAPTURE"
    world.add_debug(
        f"{marker} p{lost.id} send={total} need={need} srcs={[s for s, _, _, _ in planned]}"
    )
    return MissionProposal(
        kind="RECAPTURE_LOST",
        target_id=lost.id,
        priority=CHEAP_RECAPTURE_PRIORITY + int(lost.production) * 5,
        required_ships=total,
        planned_sources=planned,
        eta_min=eta_min,
        eta_max=eta_max,
        reason=f"cheap_recapture p{lost.id} need={need} hold={CHEAP_RECAPTURE_HOLD_MARGIN}",
    )


def generate_counterattack_after_loss_missions(world, recent_losses, deadline):
    """Punish a drained or weak enemy planet instead of tunneling on a lost planet."""
    if not recent_losses or time.perf_counter() > deadline:
        return []
    world.add_debug("COUNTERATTACK_AFTER_LOSS_SCAN")

    loss_anchor = recent_losses[0][0]
    drained_ids = {p.id for p, _score in detect_enemy_weakness(world)}
    candidates = []
    for tgt in world.enemy_planets:
        if time.perf_counter() > deadline:
            break
        if world.is_comet(tgt):
            continue
        cluster_d = world.cluster_distance(tgt)
        if cluster_d > CHEAP_RECAPTURE_LOCAL_DIST + 18:
            continue
        prev = _prev_ships.get(tgt.id)
        drained = False
        if prev is not None and (int(prev) + int(tgt.production) - int(tgt.ships)) >= CAPTURE_OPP_DRAINED_DROP:
            drained = True
            world.add_debug(f"COUNTERATTACK_DRAINED_SOURCE_FOUND p{tgt.id}")
        if tgt.id in drained_ids:
            drained = True
        if not drained and int(tgt.ships) > max(16, int(tgt.production) * 5):
            continue

        sources = [
            src for src in world.my_planets
            if dp(src, tgt) <= CHEAP_RECAPTURE_LOCAL_DIST + 10
            and world.surplus(src) >= MIN_SEND_SHIPS
            and world.real_incoming_threat(src)["deficit"] <= 0
        ]
        if not sources:
            continue
        prop = build_capture_plan(
            world, tgt, "SYNC_ATTACK", sources,
            max_sources=3, eta_spread_limit=6.0,
        )
        if prop is None:
            continue
        total = sum(s for _, s, _, _ in prop.planned_sources)
        if total > world.my_total_ships * CHEAP_RECAPTURE_TOTAL_FLEET_CAP:
            continue
        nearest_loss_d = dp(tgt, loss_anchor)
        score = (
            int(tgt.production) * 22.0
            - int(tgt.ships) * 1.3
            - world.cluster_distance(tgt) * 1.4
            - total * 0.8
            + (35.0 if drained else 0.0)
            + max(0.0, CHEAP_RECAPTURE_LOCAL_DIST - nearest_loss_d)
        )
        prop.priority = COUNTERATTACK_PRIORITY + score * 0.1
        prop.reason = (
            f"counterattack_after_loss p{tgt.id} score={score:.1f} "
            f"drained={drained} send={total}"
        )
        candidates.append((score, prop))

    if not candidates:
        return []
    candidates.sort(key=lambda item: -item[0])
    best = candidates[0][1]
    world.add_debug(
        f"COUNTERATTACK_WEAK_ENEMY_SELECTED p{best.target_id} "
        f"send={sum(s for _, s, _, _ in best.planned_sources)}"
    )
    return [best]


def try_cheap_recapture_or_counterattack(world, moves, fleet_ratio, deadline):
    """One-turn recent-loss response with no locks, containment, or TTL strategy."""
    before_moves = len(moves)
    world.add_debug("BACKYARD_LOGIC_REMOVED")
    recent_losses = very_recent_losses(world)
    if not recent_losses or time.perf_counter() > deadline:
        world.add_debug("NO_SOURCE_LOCK_WITHOUT_SELECTED_MISSION")
        return False

    recapture_props = []
    for lost, _age in recent_losses[:2]:
        prop = build_cheap_recapture_plan(world, lost)
        if prop is not None:
            recapture_props.append(prop)

    counter_props = generate_counterattack_after_loss_missions(world, recent_losses, deadline)
    if counter_props and not recapture_props:
        world.add_debug("COUNTERATTACK_OVER_RECAPTURE")
        coordinate_missions(world, counter_props, moves, fleet_ratio, deadline)
        return len(moves) > before_moves

    if recapture_props:
        if counter_props:
            rec_send = min(sum(s for _, s, _, _ in p.planned_sources) for p in recapture_props)
            ctr_send = sum(s for _, s, _, _ in counter_props[0].planned_sources)
            ctr_tgt = world.planet_by_id.get(counter_props[0].target_id)
            rec_tgt = world.planet_by_id.get(recapture_props[0].target_id)
            if ctr_tgt is not None and rec_tgt is not None and (
                ctr_send < rec_send * 0.80 or int(ctr_tgt.production) > int(rec_tgt.production)
            ):
                world.add_debug("CHEAP_RECAPTURE_SKIP_COUNTERATTACK COUNTERATTACK_OVER_RECAPTURE")
                coordinate_missions(world, counter_props, moves, fleet_ratio, deadline)
                return len(moves) > before_moves
        coordinate_missions(world, recapture_props[:1], moves, fleet_ratio, deadline)
        return len(moves) > before_moves

    world.add_debug("CHEAP_RECAPTURE_SKIP_COUNTERATTACK")
    if counter_props:
        world.add_debug("COUNTERATTACK_OVER_RECAPTURE")
        coordinate_missions(world, counter_props, moves, fleet_ratio, deadline)
        return len(moves) > before_moves

    world.add_debug("NO_SOURCE_LOCK_WITHOUT_SELECTED_MISSION")
    return False


def choose_strategy_mode(world, idle_turns):
    f = world.features
    if f["final"]:
        return StrategyMode.FINAL_DRAIN
    if (world.is_four_player and world.step < FOUR_P_EXPAND_STEP
            and world.neutral_planets and f["incoming_threat_count"] == 0):
        return StrategyMode.FOUR_PLAYER_EXPAND_FIRST
    if world.step < 55 or len(world.my_planets) < 3:
        return StrategyMode.OPENING_TEMPO
    if idle_turns >= 9 and world.my_total_ships > 250:
        return StrategyMode.FORCE_WAVE
    if f["leader_ahead"]:
        return StrategyMode.ANTI_LEADER
    if f["behind"]:
        return StrategyMode.BEHIND_STEAL
    if f["ahead"] and (world.step > 220 or not world.neutral_planets):
        return StrategyMode.COLLAPSE
    if world.enemy_planets and sum(p.ships for p in world.enemy_planets) / max(1, len(world.enemy_planets)) > TURTLE_SHIP_THRESH:
        return StrategyMode.TURTLE_BREAKER
    if f["high_neutral_count"] > 0:
        return StrategyMode.SAFE_EXPANSION
    return StrategyMode.CONTEST_NEUTRALS


def compute_control_pct(world):
    """Return (my_pct, enemy_pct, neutral_pct) as fractions of all non-comet planets."""
    total = max(1, len(world.normal_planets))
    return (
        len(world.my_planets)      / total,
        len(world.enemy_planets)   / total,
        len(world.neutral_planets) / total,
    )


def classify_strategic_phase(world):
    """
    Classify the current strategic phase from map-control percentage.
    Step number is used only as a secondary tiebreaker so the bot keeps
    expanding whenever the map still has unclaimed territory, regardless of turn.
    """
    my_pct, enemy_pct, neutral_pct = compute_control_pct(world)

    if my_pct < PHASE_OPENING_PCT or len(world.my_planets) <= 3:
        return ControlPhase.OPENING_EXPANSION

    if my_pct < PHASE_SWEEP_PCT:
        return ControlPhase.LOCAL_SWEEP

    near_enemy = any(
        min((dp(m, e) for m in world.my_planets), default=999.0) < FRONTLINE_DIST
        for e in world.enemy_planets
        if not world.is_comet(e)
    )

    if my_pct < PHASE_EXPAND_PCT:
        return ControlPhase.CONTACT if near_enemy else ControlPhase.EXPANSION_CONTROL

    # Dominant control → check if opponent is nearly finished
    if (enemy_pct < 0.15
            or (world.enemy_prod > 0 and world.enemy_prod < world.my_prod * 0.5)
            or not world.neutral_planets):
        return ControlPhase.COLLAPSE

    return ControlPhase.CONTACT


def launchpad_target_score(world, src, tgt, mode=None):
    """Strategic score: production anchors, chain value, recapture, and collapse conversion."""
    if src is None or tgt is None or world.is_comet(tgt):
        return -1e9
    need = world.ships_needed_to_capture(src, tgt, max(1, world.surplus(src)))
    if need <= 0:
        need = max(1, int(tgt.ships) + 1)
    eta = world.eta(src, tgt, need)
    d = dp(src, tgt)
    prod = int(tgt.production)

    production_value_bonus = prod * 62.0
    if prod >= 4:
        production_value_bonus += CHAIN_PROD4_BONUS + 65.0
    if prod >= 5:
        production_value_bonus += CHAIN_PROD5_BONUS + 110.0

    chain_bonus = 0.0
    nearby_next = [
        n for n in world.neutral_planets + world.enemy_planets
        if n.id != tgt.id
        and not world.is_comet(n)
        and dp(tgt, n) <= LAUNCHPAD_RADIUS
        and int(n.production) >= 3
    ]
    if nearby_next:
        chain_bonus += min(90.0, sum(int(n.production) * 8.0 for n in nearby_next[:4]))
    if prod >= LAUNCHPAD_PROD_MIN or int(tgt.ships) + prod * 3 >= LAUNCHPAD_SURPLUS_MIN:
        chain_bonus += 35.0
    strategic_position_bonus = max(0.0, LAUNCHPAD_RADIUS - world.cluster_distance(tgt)) * 2.0
    if world.enemy_planets:
        nearest_enemy = min(dp(tgt, e) for e in world.enemy_planets)
        if 18.0 <= nearest_enemy <= 48.0:
            strategic_position_bonus += 35.0
        # Frontier bonus: prod-3+ planet between our cluster and enemy
        my_d = min((dp(tgt, m) for m in world.my_planets), default=999.0)
        if prod >= 3 and my_d <= 44.0 and nearest_enemy <= 52.0:
            strategic_position_bonus += 95.0

    recapture_bonus = 0.0

    enemy_core_bonus = 0.0
    if tgt.owner not in (-1, world.player):
        enemy_core_bonus += prod * 45.0
        if mode in (StrategyMode.COLLAPSE,):
            enemy_core_bonus += 85.0
        if len(world.enemy_planets) <= 6:
            enemy_core_bonus += 45.0

    enemy_t = min((world.eta(e, tgt, max(1, int(e.ships))) for e in world.enemy_planets), default=999.0)
    overextension_penalty = 0.0
    if world.is_four_player and world.step < FOUR_P_ATTACK_STEP and tgt.owner not in (-1, world.player):
        overextension_penalty += 140.0
    if world.cluster_distance(tgt) > MIDGAME_CONTEST_MAX_DIST:
        overextension_penalty += (world.cluster_distance(tgt) - MIDGAME_CONTEST_MAX_DIST) * 3.0
    if tgt.owner == -1 and enemy_t < eta - 3.0:
        overextension_penalty += 80.0

    travel_penalty = eta * 5.0 + d * 1.2
    ship_cost_penalty = need * (1.15 if prod >= 4 else 1.55)
    return (
        production_value_bonus + chain_bonus + strategic_position_bonus
        + recapture_bonus + enemy_core_bonus
        - travel_penalty - ship_cost_penalty - overextension_penalty
    )


def score_target(world, src, tgt, mode):
    need = world.ships_needed_to_capture(src, tgt)
    if need <= 0:
        return -1e9
    eta = world.eta(src, tgt, min(max(1, world.surplus(src)), need))
    if eta > world.remaining - 3:
        return -1e9
    my_t, enemy_t = world.reaction_times(tgt)
    future_turns = max(1, world.remaining - eta)
    value = tgt.production * future_turns
    value += max(0, 45 - tgt.ships) * (1.0 if tgt.owner != -1 else 0.25)
    if is_idle(tgt):
        value *= 1.18
    if tgt.owner == -1 and mode in (StrategyMode.OPENING_TEMPO, StrategyMode.SAFE_EXPANSION, StrategyMode.BEHIND_STEAL, StrategyMode.FOUR_PLAYER_EXPAND_FIRST):
        value *= 1.25
    if tgt.owner not in (-1, world.player):
        value *= 1.65
        if tgt.owner == world.leader:
            value *= 1.25
    if mode == StrategyMode.CONTEST_NEUTRALS and tgt.owner == -1 and abs(my_t - enemy_t) < 5:
        value *= 1.35
    if mode == StrategyMode.TURTLE_BREAKER and tgt.owner not in (-1, world.player):
        value *= 1.35
    if mode in (StrategyMode.COLLAPSE, StrategyMode.FORCE_WAVE) and tgt.owner not in (-1, world.player):
        value *= 1.5
    if mode == StrategyMode.BEHIND_STEAL and need > max(20, world.my_total_ships * 0.18):
        value *= 0.55
    risk = max(0, enemy_t - my_t) * -0.1 if tgt.owner == -1 else max(0, 12 - enemy_t) * 2.0
    bridge = route_bridge_value(world, src, tgt)
    return value + bridge - need * 1.4 - eta * 3.0 - risk


def route_bridge_value(world, src, tgt):
    if not world.enemy_planets:
        return 0.0
    enemy_anchor = min(world.enemy_planets, key=lambda e: dp(tgt, e))
    closer = max(0.0, dp(src, enemy_anchor) - dp(tgt, enemy_anchor))
    return closer * 0.8 + (25.0 if tgt.production >= 5 else 0.0)


def route_edge_cost(world, src, tgt):
    """Non-negative travel/capture risk cost for Dijkstra/A* style routing."""
    if src.id == tgt.id:
        return 0.0
    need = world.ships_needed_to_capture(src, tgt, max(1, int(tgt.ships) + 1))
    eta = world.eta(src, tgt, need)
    _, enemy_eta = world.reaction_times(tgt)
    sun_penalty = 14.0 if hits_sun(src.x, src.y, tgt.x, tgt.y) else 0.0
    contest_penalty = max(0.0, eta - enemy_eta + 1.5) * 12.0 if tgt.owner == -1 else 0.0
    pressure_penalty = world.enemy_pressure_near(tgt) * 0.25
    overextension = max(0.0, 18.0 - world.nearest_enemy_distance(tgt)) * 0.8
    reward = int(tgt.production) * 3.0 + route_bridge_value(world, src, tgt) * 0.25
    return max(0.25, eta * 3.5 + need * 0.45 + sun_penalty + contest_penalty + pressure_penalty + overextension - reward)


def multi_source_dijkstra_from_owned(world, deadline, max_depth=3):
    """Map each reachable neutral to the cheapest hypothetical route from my cluster."""
    routes = {}
    if not world.my_planets or not world.neutral_planets:
        return routes

    queue = []
    best = {}
    for src in world.my_planets:
        heapq.heappush(queue, (0.0, src.id, src.id, ()))
        best[(src.id, ())] = 0.0

    neutral_ids = {p.id for p in world.neutral_planets}
    expansion_nodes = world.neutral_planets
    while queue and time.perf_counter() < deadline:
        cost, node_id, source_id, route = heapq.heappop(queue)
        node = world.planet_by_id.get(node_id)
        if node is None:
            continue
        if node_id in neutral_ids and route:
            first_hop = route[0]
            current = routes.get(node_id)
            score = -cost
            if current is None or cost < current.cost:
                routes[node_id] = RoutePlan(node_id, first_hop, score, cost, list(route))
        if len(route) >= max_depth:
            continue
        for nxt in expansion_nodes:
            if nxt.id == node_id or nxt.id in route:
                continue
            edge = route_edge_cost(world, node, nxt)
            new_cost = cost + edge
            new_route = route + (nxt.id,)
            key = (nxt.id, new_route)
            if new_cost + 1e-6 >= best.get(key, 1e18):
                continue
            best[key] = new_cost
            heapq.heappush(queue, (new_cost, nxt.id, source_id, new_route))
    return routes


def estimate_grouped_sources(world, tgt, need, max_sources=NEAREST_LOCK_MAX_SOURCES):
    """Pick nearby safe sources and the amount each can contribute to a neutral capture."""
    buffer = 1 if world.step < EARLY_STEPS else min(4, max(1, int(tgt.production)))
    goal = max(1, int(need + buffer))
    sources = sorted(
        [
            p for p in world.my_planets
            if world.real_incoming_threat(p)["deficit"] <= 0 and world.surplus(p) > 0
        ],
        key=lambda p: (
            world.eta(p, tgt, max(1, min(world.surplus(p), goal))),
            0 if world.is_backline(p) else 1,
            -world.surplus(p),
        ),
    )
    selected = []
    total = 0
    for src in sources[:max_sources]:
        if total >= goal:
            break
        available = world.surplus(src)
        send = min(available, goal - total)
        if send < 3 and total + send < need:
            continue
        angle, ok = world.aim(src, tgt, send)
        if not ok:
            continue
        selected.append((src, int(send), angle))
        total += int(send)
    return selected, total, goal


def nearest_neutral_candidates(world, deadline):
    """Nearest-first expansion scoring. Distance dominates early; production breaks ties."""
    if not world.neutral_planets or not world.my_planets:
        return []
    route_map = multi_source_dijkstra_from_owned(world, deadline)
    candidates = []
    for tgt in world.neutral_planets:
        if time.perf_counter() > deadline:
            break
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        if world.step < 140 and not validate_initial_target_choice(world, src, tgt):
            continue
        need = world.ships_needed_to_capture(src, tgt)
        if need <= 0:
            continue
        selected, grouped_pool, _ = estimate_grouped_sources(world, tgt, need, max_sources=PROACTIVE_EXPANSION_MAX_SOURCES)
        if not selected:
            continue
        closest_eta = world.eta(src, tgt, max(1, min(max(need, 1), max(1, int(src.ships)))))
        closest_dist = dp(src, tgt)
        my_eta, enemy_eta = world.reaction_times(tgt)
        enemy_faster = enemy_eta + 1.5 < closest_eta
        route = route_map.get(tgt.id)
        route_cost = route.cost if route is not None else closest_eta * 3.5 + need * 0.45
        cluster_dist = world.cluster_distance(tgt)
        cluster_bonus = max(0.0, 34.0 - cluster_dist) * 2.5
        close_bonus = max(0.0, NEAREST_LOCK_DIST - closest_dist) * (5.0 if world.step < 140 else 3.0)
        bridge_bonus = route_bridge_value(world, src, tgt)
        weak_bonus = max(0.0, 24.0 - need) * 2.0
        static_bonus = 18.0 if is_idle(tgt) else 0.0
        contest_risk = max(0.0, closest_eta - enemy_eta + 1.0) if enemy_eta < 999 else 0.0
        missing_penalty = 120.0 if grouped_pool < need else 0.0
        early_distance_weight = 16.0 if world.step < 140 or len(world.my_planets) < 4 else 9.0
        score = (
            int(tgt.production) * 40.0
            - closest_eta * early_distance_weight
            - closest_dist * 1.35
            - need * 1.75
            - contest_risk * 24.0
            - route_cost * 0.35
            - missing_penalty
            + bridge_bonus
            + cluster_bonus
            + close_bonus
            + weak_bonus
            + static_bonus
        )
        can_lock = (
            grouped_pool >= need
            and not enemy_faster
            and closest_dist <= NEAREST_LOCK_DIST
            and closest_eta <= NEAREST_LOCK_ETA
        )
        if grouped_pool >= need and not enemy_faster and int(tgt.production) >= 4 and closest_dist <= NEAREST_LOCK_DIST + 6:
            can_lock = True
        if can_lock:
            score += 110.0
        reason = (
            f"p{tgt.id} score={score:.1f} d={closest_dist:.1f} eta={closest_eta:.1f} "
            f"need={need} pool={grouped_pool} prod={int(tgt.production)} enemy_eta={enemy_eta:.1f}"
        )
        candidates.append(
            NearestCandidate(
                score=score,
                target_id=tgt.id,
                source_id=src.id,
                need=need,
                eta=closest_eta,
                distance=closest_dist,
                enemy_eta=enemy_eta,
                grouped_pool=grouped_pool,
                route=route.route if route is not None else [tgt.id],
                can_lock=can_lock,
                reason=reason,
            )
        )
    candidates.sort(key=lambda c: (-c.can_lock, -c.score, c.distance, -world.planet_by_id[c.target_id].production))
    return candidates


def plan_grouped_capture(world, tgt, moves, need=None, max_sources=NEAREST_LOCK_MAX_SOURCES, force=False):
    """Capture a close neutral with one or a few nearby planets without draining reserves."""
    if tgt.owner != -1:
        return False
    src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
    if src is None:
        return False
    need = int(need if need is not None else world.ships_needed_to_capture(src, tgt))
    if need <= 0:
        return False
    candidate_sources = [
        p for p in world.my_planets
        if world.real_incoming_threat(p)["deficit"] <= 0 and world.surplus(p) > 0
    ]
    plan, reason = build_grouped_funding_plan(
        world,
        tgt,
        need,
        candidate_sources,
        "CAPTURE_NEUTRAL",
        max_sources=max_sources,
        eta_spread_limit=3.0,
        require_hold=not force,
    )
    if plan is None:
        world.add_debug(f"grouped_capture VALIDATE_REJECT p{tgt.id} reason={reason}")
        return False
    planned, goal, _eta_min, _eta_max = plan
    mission_id = world.mission_ledger.create(
        "CAPTURE_NEUTRAL",
        tgt.id,
        [src_id for src_id, _, _, _ in planned],
        goal,
        [eta for _, _, _, eta in planned],
        f"grouped_capture p{tgt.id} need={need} goal={goal}",
    )
    sent = 0
    for src_id, n, _angle, _eta in planned:
        if sent >= goal:
            break
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        if world.commit(src, tgt, n, moves, mission_type="CAPTURE_NEUTRAL", mission_id=mission_id, planned_sources=planned):
            sent += n
    if sent >= need:
        world.wave_attempted = True
        world.add_debug(f"grouped_capture target=p{tgt.id} need={need} sent={sent} sources={[s for s, _, _, _ in planned]}")
        return True
    return False


def nearest_expansion_plan(world, moves, mode, deadline, force=False):
    """Hard proactive neutral capture pass before combat and leader pressure."""
    if not world.neutral_planets or world.remaining < 35:
        return False
    candidates = nearest_neutral_candidates(world, deadline)
    if not candidates:
        world.add_debug("nearest_expansion no candidates")
        return False
    world.add_debug(
        f"phase={mode} owned={len(world.my_planets)} prod={world.my_prod} "
        f"top_neutrals={[c.reason for c in candidates[:5]]}"
    )
    locked = [c for c in candidates if c.can_lock]
    if locked:
        chosen = locked[0]
        trigger = "NEAREST_LOCK"
    elif force:
        chosen = candidates[0]
        trigger = "FORCE_NEAREST"
    elif mode in (StrategyMode.OPENING_TEMPO, StrategyMode.SAFE_EXPANSION, StrategyMode.CONTEST_NEUTRALS, StrategyMode.BEHIND_STEAL, StrategyMode.ANTI_LEADER, StrategyMode.FOUR_PLAYER_EXPAND_FIRST):
        chosen = candidates[0] if candidates[0].score > -60.0 and candidates[0].grouped_pool >= candidates[0].need else None
        trigger = "PROACTIVE_EXPANSION"
    else:
        chosen = candidates[0] if candidates[0].can_lock else None
        trigger = "LOCK_ONLY"
    if chosen is None:
        world.add_debug(f"nearest_expansion skipped best={candidates[0].reason}")
        return False
    target = world.planet_by_id.get(chosen.target_id)
    if target is None:
        return False
    acted = plan_grouped_capture(
        world,
        target,
        moves,
        need=chosen.need,
        max_sources=PROACTIVE_EXPANSION_MAX_SOURCES if force else NEAREST_LOCK_MAX_SOURCES,
        force=force or chosen.can_lock,
    )
    world.add_debug(f"{trigger} target=p{target.id} acted={acted} route={chosen.route} reason={chosen.reason}")
    return acted


def build_grouped_wave(world, tgt, need, moves, max_sources=MAX_GROUP_SOURCES, allow_partial=False, sync=False):
    sources = sorted(
        [p for p in world.my_planets if world.real_incoming_threat(p)["deficit"] <= 0 and world.surplus(p) > 0],
        key=lambda p: (world.eta(p, tgt, max(1, min(world.surplus(p), need))), -world.surplus(p))
    )[:max_sources]
    pool = sum(world.surplus(p) for p in sources)
    if pool <= 0:
        return 0
    # For offensive (enemy-owned) targets: require full pool; reserve if close but not ready
    if tgt.owner not in (-1, world.player) and not allow_partial and pool < need:
        if not wait_for_wave_if_better(world, tgt, sources, need):
            if should_wait_for_better_wave(world, tgt):
                reserve_wave(world, tgt, sources)
            else:
                world.add_debug(f"LOW_VALUE_NOW_SKIP target=p{tgt.id} reason=cant_reach_need pool={pool} need={need}")
        world.add_debug(f"TRICKLE_BLOCK target=p{tgt.id} ships={pool} need={need}")
        return 0
    mission_type = "FINAL_DRAIN" if allow_partial else ("CAPTURE_NEUTRAL" if tgt.owner == -1 else "SYNC_ATTACK")
    goal = need if pool >= need else (int(pool * 0.32) if allow_partial else 0)
    if goal < min(8, need):
        return 0
    if sync and len(sources) >= 2:
        etas = [(world.eta(p, tgt, max(1, min(world.surplus(p), goal))), p) for p in sources]
        median_eta = sorted(e for e, _ in etas)[len(etas) // 2]
        sources = [p for eta, p in etas if abs(eta - median_eta) <= 4] or sources[:3]
    plan, reason = build_grouped_funding_plan(
        world,
        tgt,
        goal,
        sources,
        mission_type,
        max_sources=max_sources,
        eta_spread_limit=3.0 if tgt.owner == -1 else 6.0,
        allow_small_packets=allow_partial,
        require_hold=not allow_partial,
    )
    if plan is None:
        world.add_debug(f"GROUPED_WAVE_REJECT target=p{tgt.id} reason={reason}")
        return 0
    planned, goal, _eta_min, _eta_max = plan
    mission_id = world.mission_ledger.create(
        mission_type,
        tgt.id,
        [src_id for src_id, _, _, _ in planned],
        goal,
        [eta for _, _, _, eta in planned],
        f"grouped_wave target=p{tgt.id} need={need} goal={goal}",
    )
    sent = 0
    for src_id, n, _, _ in planned:
        if sent >= goal:
            break
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        if world.commit(src, tgt, n, moves, mission_type=mission_type, mission_id=mission_id, planned_sources=planned):
            sent += n
    if sent > 0 and tgt.owner != world.player:
        world.wave_attempted = True
    return sent


def emergency_defense(world, moves):
    acted = False
    for tgt in sorted(world.my_planets, key=lambda p: -world.real_incoming_threat(p)["deficit"]):
        deficit = world.real_incoming_threat(tgt)["deficit"]
        if deficit <= 0:
            continue
        # Value-aware: cap defense of low-prod planets when prod-3+ neutrals exist
        if int(tgt.production) <= 1 and world.step < 90 and len(world.my_planets) > 1:
            if any(not world.is_comet(n) and int(n.production) >= 3 for n in world.neutral_planets):
                deficit = min(deficit, 4)
        planned = []
        remaining = deficit
        for src in sorted([p for p in world.my_planets if p.id != tgt.id], key=lambda p: dp(p, tgt)):
            n = min(world.surplus(src), remaining)
            if n <= 0:
                continue
            planned.append((src.id, n, 0.0, world.eta(src, tgt, n)))
            remaining -= n
            if remaining <= 0:
                break
        mission_id = world.mission_ledger.create(
            "DEFEND_HOLD",
            tgt.id,
            [src_id for src_id, _, _, _ in planned],
            deficit,
            [eta for _, _, _, eta in planned],
            f"emergency defend p{tgt.id} deficit={deficit}",
        ) if planned else None
        for src in sorted([p for p in world.my_planets if p.id != tgt.id], key=lambda p: dp(p, tgt)):
            n = min(world.surplus(src), deficit)
            if n <= 0:
                continue
            if world.commit(src, tgt, n, moves, mission_type="DEFEND_HOLD", mission_id=mission_id, planned_sources=planned):
                deficit -= n
                acted = True
            if deficit <= 0:
                break
    return acted


def force_action_if_stalling(world, moves, idle_turns, deadline):
    """Prefer forced nearest expansion before falling back to enemy pressure."""
    if idle_turns < STALL_FORCE_TURNS or world.my_total_ships < STALL_FORCE_SHIPS:
        return False
    if world.neutral_planets and nearest_expansion_plan(world, moves, StrategyMode.FORCE_WAVE, deadline, force=True):
        return True
    if idle_turns < 8 or world.my_total_ships <= 230:
        return False
    if (
        world.is_four_player
        and world.step < FOUR_P_ATTACK_STEP
        and world.neutral_planets
        and not any(is_local_enemy_opportunity(world, e) or _is_corner_control_target(world, e) for e in world.enemy_planets)
    ):
        return False
    targets = sorted(
        world.enemy_planets,
        key=lambda p: (
            min(dp(m, p) for m in world.my_planets),
            -int(p.production),
            int(p.ships),
        ),
    )
    for tgt in targets[:4]:
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        need = world.ships_needed_to_capture(src, tgt, world.my_total_ships)
        if build_grouped_wave(world, tgt, need, moves, max_sources=6, allow_partial=False, sync=True) > 0:
            world.add_debug(f"stall_force_enemy target=p{tgt.id} need={need}")
            return True
    return False


def fallback_tempo(world, moves):
    if moves:
        return False
    targets = world.neutral_planets or world.enemy_planets
    if not targets:
        return False
    acted = False
    for src in sorted(world.my_planets, key=lambda p: -world.surplus(p)):
        surplus = world.surplus(src)
        if surplus <= 0:
            continue
        if world.step < 140:
            valid = [t for t in targets if world.valid_target(t)
                     and validate_initial_target_choice(world, src, t)]
            tgt = (max(valid, key=lambda t: early_target_score(world, src, t))
                   if valid else min(targets, key=lambda t: dp(src, t)))
        else:
            tgt = min(targets, key=lambda t: dp(src, t))
        need = world.ships_needed_to_capture(src, tgt, surplus)
        mission_type = "CAPTURE_NEUTRAL" if tgt.owner == -1 else "SYNC_ATTACK"
        send = normalize_send_amount(need)
        if 0 < send <= surplus and world.commit(src, tgt, send, moves, mission_type=mission_type):
            acted = True
    return acted


# ── Mission Coordinator ───────────────────────────────────────────────────────

def compute_fleet_ratio(world):
    planet_ships = sum(int(p.ships) for p in world.my_planets)
    fleet_ships = sum(int(f.ships) for f in world.my_fleets)
    return fleet_ships / max(1, planet_ships + fleet_ships)


def generate_finish_capture_missions(world):
    proposals = []
    seen_targets = set()
    targets = [p for p in world.neutral_planets + world.enemy_planets
               if world.incoming_to_targets.get(p.id, 0) > 0]
    for tgt in sorted(targets, key=lambda p: world.target_need_now(p)):
        if world.is_comet(tgt):
            continue
        need = world.target_need_now(tgt)
        if need <= 0 or need > 20:
            continue
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        send = min(world.surplus(src), need)
        if send <= 0:
            continue
        angle, ok = world.aim(src, tgt, send)
        if not ok:
            continue
        eta = world.eta(src, tgt, send)
        proposals.append(MissionProposal(
            kind="FINISH_CAPTURE",
            target_id=tgt.id,
            priority=95.0 + tgt.production * 5,
            required_ships=need,
            planned_sources=[(src.id, send, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"finish p{tgt.id} need={need} in_transit={world.incoming_to_targets.get(tgt.id, 0)}",
        ))
        seen_targets.add(tgt.id)

    weak_targets = [
        tgt for tgt in world.neutral_planets + world.enemy_planets
        if tgt.id not in seen_targets
        and not world.is_comet(tgt)
        and int(tgt.ships) <= FINISH_ZERO_MAX_SHIPS
        and tgt.owner != world.player
    ]
    for tgt in sorted(
        weak_targets,
        key=lambda p: (
            min((dp(src, p) for src in world.my_planets), default=999.0),
            -int(p.production),
            int(p.ships),
        ),
    )[:6]:
        nearest = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if nearest is None:
            continue
        nearest_dist = dp(nearest, tgt)
        if nearest_dist > FINISH_ZERO_NEAR_DIST and world.cluster_distance(tgt) > FINISH_ZERO_NEAR_DIST + 8:
            continue
        candidates = sorted(
            [
                src for src in world.my_planets
                if world.surplus(src) >= 3
                and world.real_incoming_threat(src)["deficit"] <= 0
            ],
            key=lambda src: world.eta(src, tgt, max(3, min(world.surplus(src), 8))),
        )
        for src in candidates[:5]:
            rough_send = max(3, min(world.surplus(src), max(3, int(tgt.ships) + 3)))
            eta = world.eta(src, tgt, rough_send)
            if eta > FINISH_ZERO_MAX_ETA:
                continue
            need = world.min_ships_to_own_by(
                tgt.id,
                max(1, int(math.ceil(eta))),
                world.player,
                arrival_turn=max(1, int(math.ceil(eta))),
                upper_bound=world.surplus(src),
            )
            send = max(3, need)
            if send <= 0 or send > world.surplus(src):
                continue
            eta = world.eta(src, tgt, send)
            if eta > FINISH_ZERO_MAX_ETA:
                continue
            source_ok, reason = world.source_is_safe_for(src, tgt, "FINISH_ZERO_CAPTURE", send)
            if not source_ok:
                world.add_debug(f"FINISH_ZERO_SKIP p{tgt.id} src=p{src.id} reason={reason}")
                continue
            angle, ok = world.aim(src, tgt, send)
            if not ok:
                continue
            priority = PRIORITY_FINISH_ZERO_BASE + int(tgt.production) * 2.5 + max(0.0, FINISH_ZERO_NEAR_DIST - nearest_dist) * 0.25
            proposals.append(MissionProposal(
                kind="FINISH_ZERO_CAPTURE",
                target_id=tgt.id,
                priority=priority,
                required_ships=send,
                planned_sources=[(src.id, send, angle, eta)],
                eta_min=eta,
                eta_max=eta,
                reason=(
                    f"finish-zero forced p{tgt.id} ships={int(tgt.ships)} "
                    f"prod={int(tgt.production)} d={nearest_dist:.1f} eta={eta:.1f}"
                ),
            ))
            world.add_debug(
                f"FINISH_ZERO_FORCED target=p{tgt.id} src=p{src.id} ships={send} eta={eta:.1f}"
            )
            break
    return proposals


def generate_save_under_attack_missions(world):
    """Either save an owned planet with a real grouped hold wave, or mark it doomed."""
    proposals = []
    for tgt in sorted(world.my_planets, key=lambda p: -world.real_incoming_threat(p)["enemy"]):
        threat = world.real_incoming_threat(tgt, horizon=DEFENSE_ETA_HORIZON)
        tl = world.simulate_planet_timeline(tgt, DEFENSE_ETA_HORIZON)
        if threat["enemy"] <= threat["friendly"] and tl["fall_turn"] is None:
            continue
        if tl["fall_turn"] is None and threat["deficit"] <= 0:
            continue

        # Value-aware: skip saving prod<=1 planets when prod-3+ neutral is capturable
        if int(tgt.production) <= 1 and world.step < 90 and len(world.my_planets) > 1:
            if any(not world.is_comet(n) and int(n.production) >= 3 for n in world.neutral_planets):
                world.add_debug(f"SAVE_SKIP_LOW_PROD p{tgt.id} prod={tgt.production}")
                continue

        need = max(threat["deficit"], tl["keep_needed"], DEFEND_NET - tl["min_owned"]) + 2
        if need <= 0:
            continue

        candidates = sorted(
            [
                src for src in world.my_planets
                if src.id != tgt.id
                and world.surplus(src) > 0
                and not world.is_recently_reinforced(src)
                and world.real_incoming_threat(src)["deficit"] <= 0
            ],
            key=lambda s: (world.eta(s, tgt, max(1, min(world.surplus(s), need))), dp(s, tgt)),
        )[:MAX_GROUP_SOURCES]

        planned = []
        remaining = need
        latest_enemy_eta = max(
            [eta for eta, owner, _ in world.arrivals_by_target.get(tgt.id, []) if owner != world.player and eta <= DEFENSE_ETA_HORIZON]
            or [DEFENSE_ETA_HORIZON]
        )
        for src in candidates:
            if remaining <= 0:
                break
            av = world.surplus(src)
            if av <= 0:
                continue
            send = min(av, remaining)
            eta = world.eta(src, tgt, send)
            if eta > latest_enemy_eta + 2:
                continue
            angle, ok = world.aim(src, tgt, send)
            if not ok:
                continue
            source_ok, _ = world.source_is_safe_for(src, tgt, "SAVE_UNDER_ATTACK", send)
            if not source_ok:
                continue
            planned.append((src.id, send, angle, eta))
            remaining -= send

        if not planned:
            world.mark_doomed(tgt)
            world.add_debug(f"SKIP SAVE_UNDER_ATTACK p{tgt.id} step={world.step} reason=no_planned_sources need={need}")
            continue

        full_tl = world.simulate_planet_timeline(
            tgt,
            DEFENSE_ETA_HORIZON,
            planned=[(eta, world.player, ships) for _, ships, _, eta in planned],
        )
        if full_tl["fall_turn"] is not None:
            world.mark_doomed(tgt)
            world.add_debug(
                f"SKIP SAVE_UNDER_ATTACK p{tgt.id} step={world.step} reason=still_falls "
                f"need={need} planned={sum(s for _, s, _, _ in planned)} fall={full_tl['fall_turn']}"
            )
            continue

        eta_vals = [e for _, _, _, e in planned]
        priority_val = PRIORITY_SAVE_ATTACK_BASE + int(tgt.production) * 4 + threat["enemy"] * 0.2
        proposals.append(MissionProposal(
            kind="SAVE_UNDER_ATTACK",
            target_id=tgt.id,
            priority=priority_val,
            required_ships=need,
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"save p{tgt.id} enemy={threat['enemy']} need={need} min_after={full_tl['min_owned']}",
        ))
        world.add_debug(
            f"SELECT SAVE_UNDER_ATTACK p{tgt.id} step={world.step} "
            f"srcs={[s for s, _, _, _ in planned]} ships={sum(s for _, s, _, _ in planned)} "
            f"eta={max(eta_vals):.1f} priority={priority_val:.1f} "
            f"reason=save enemy={threat['enemy']} need={need}"
        )
    return proposals


def _save_possible_before_fall(world, target, fall_turn):
    if fall_turn is None or fall_turn <= 0:
        return True
    nearby = [
        src for src in world.my_planets
        if src.id != target.id
        and world.surplus(src) > 0
        and world.real_incoming_threat(src)["deficit"] <= 0
    ]
    possible = 0
    for src in nearby:
        send = world.surplus(src)
        if world.eta(src, target, send) <= fall_turn:
            possible += send
    need = world.min_ships_to_own_by(
        target.id,
        fall_turn + 3,
        world.player,
        arrival_turn=max(1, fall_turn),
        upper_bound=possible,
    )
    return possible >= need


def generate_fall_turn_recapture_missions(world):
    """Schedule recapture waves to arrive just after an unsavable owned planet falls."""
    proposals = []
    for target in world.my_planets:
        tl = world.simulate_planet_timeline(target, FALL_RECAPTURE_HORIZON)
        fall_turn = tl["fall_turn"]
        if fall_turn is None or fall_turn > FALL_RECAPTURE_HORIZON:
            continue
        if _save_possible_before_fall(world, target, fall_turn):
            continue
        world.mark_doomed(target)
        sources = sorted(
            [
                src for src in world.my_planets
                if src.id != target.id
                and world.surplus(src) > 0
                and world.real_incoming_threat(src)["deficit"] <= 0
            ],
            key=lambda s: dp(s, target),
        )[:MAX_GROUP_SOURCES]
        if not sources:
            continue
        pool = sum(world.surplus(src) for src in sources)
        best_src = min(sources, key=lambda s: dp(s, target))
        rough_eta = world.eta(best_src, target, max(1, min(pool, max(1, int(target.ships) + 1))))
        if rough_eta <= fall_turn or rough_eta > fall_turn + FALL_RECAPTURE_LOOKAHEAD:
            world.add_debug(
                f"RECAPTURE_LOST_SKIP p{target.id} reason=outside useful window fall={fall_turn} eta={rough_eta:.1f}"
            )
            continue
        need = world.min_ships_to_own_by(
            target.id,
            math.ceil(rough_eta),
            world.player,
            arrival_turn=math.ceil(rough_eta),
            upper_bound=pool,
        )
        if need <= 0 or pool < need:
            continue
        planned = []
        remaining = need
        for src in sources:
            if remaining <= 0:
                break
            av = world.surplus(src)
            if av <= 0:
                continue
            send = min(av, remaining)
            eta = world.eta(src, target, send)
            if eta <= fall_turn or eta > fall_turn + FALL_RECAPTURE_LOOKAHEAD:
                continue
            angle, ok = world.aim(src, target, send)
            if not ok:
                continue
            planned.append((src.id, send, angle, eta))
            remaining -= send
        if not planned or sum(s for _, s, _, _ in planned) < need:
            continue
        eta_vals = [eta for _, _, _, eta in planned]
        proposals.append(MissionProposal(
            kind="RECAPTURE_LOST",
            target_id=target.id,
            priority=104.0 + int(target.production) * 4,
            required_ships=need,
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"fall_turn_recapture p{target.id} fall={fall_turn} need={need}",
        ))
    return proposals


def generate_snipe_missions(world):
    """Steal neutrals just after enemy fleets soften them, only near our cluster."""
    proposals = []
    if not world.neutral_planets or not world.my_planets:
        return proposals
    for target in sorted(world.neutral_planets, key=lambda n: (world.cluster_distance(n), -int(n.production))):
        if world.is_comet(target) or world.cluster_distance(target) > 34.0:
            continue
        enemy_etas = sorted(
            int(math.ceil(eta))
            for eta, owner, ships in world.arrivals_by_target.get(target.id, [])
            if owner != world.player and ships > 0 and eta <= SNIPE_LOOKAHEAD
        )
        if not enemy_etas:
            continue
        for enemy_eta in enemy_etas[:2]:
            eval_turn = enemy_eta + 1
            owner_after_enemy, ships_after_enemy = world.projected_state(target.id, eval_turn)
            if owner_after_enemy == world.player:
                continue
            if ships_after_enemy >= int(target.ships) and owner_after_enemy == -1:
                continue
            candidates = sorted(
                [src for src in world.my_planets if world.surplus(src) > 0],
                key=lambda s: world.eta(s, target, max(1, min(world.surplus(s), int(target.ships) + 1))),
            )
            for src in candidates[:4]:
                av = world.surplus(src)
                if av <= 0:
                    continue
                need = world.min_ships_to_own_by(
                    target.id,
                    eval_turn,
                    world.player,
                    arrival_turn=eval_turn,
                    upper_bound=av,
                )
                if need <= 0 or need > av:
                    continue
                eta = world.eta(src, target, need)
                if eta < enemy_eta or eta > enemy_eta + SNIPE_ETA_SLACK:
                    continue
                if not world.can_hold_after_capture(target, eval_turn, need):
                    continue
                angle, ok = world.aim(src, target, need)
                if not ok:
                    continue
                proposals.append(MissionProposal(
                    kind="SNIPE_NEUTRAL",
                    target_id=target.id,
                    priority=78.0 + int(target.production) * 5 - need,
                    required_ships=need,
                    planned_sources=[(src.id, need, angle, eta)],
                    eta_min=eta,
                    eta_max=eta,
                    reason=f"snipe p{target.id} enemy_eta={enemy_eta} need={need} after_ships={ships_after_enemy}",
                ))
                break
    return proposals[:3]


def generate_doomed_evacuation_missions(world):
    """Move excess ships off planets that are projected to fall and cannot be saved."""
    proposals = []
    for src in world.my_planets:
        tl = world.simulate_planet_timeline(src, DOOMED_EVAC_HORIZON)
        fall_turn = tl["fall_turn"]
        if fall_turn is None or fall_turn > DOOMED_EVAC_HORIZON:
            continue
        if _save_possible_before_fall(world, src, fall_turn):
            continue
        spare = max(0, world.surplus(src))
        if spare < DOOMED_EVAC_MIN_SHIPS:
            continue
        world.mark_doomed(src)

        capture_targets = sorted(
            [
                tgt for tgt in world.neutral_planets + world.enemy_planets
                if not world.is_comet(tgt)
                and world.eta(src, tgt, spare) < fall_turn
                and dp(src, tgt) <= LOCAL_HUB_RADIUS + 12
            ],
            key=lambda t: (world.ships_needed_to_capture(src, t, spare), dp(src, t)),
        )
        chosen = None
        send = 0
        for tgt in capture_targets[:5]:
            need = world.ships_needed_to_capture(src, tgt, spare)
            eta = world.eta(src, tgt, need) if need > 0 else 999.0
            if 0 < need <= spare and eta < fall_turn and world.can_hold_after_capture(tgt, eta, need):
                chosen = tgt
                send = need
                break
        if chosen is None:
            safe_allies = [
                ally for ally in world.my_planets
                if ally.id != src.id
                and world.simulate_planet_timeline(ally, DOOMED_EVAC_HORIZON)["fall_turn"] is None
                and world.eta(src, ally, spare) < fall_turn
                and dp(src, ally) <= LOCAL_HUB_RADIUS + 8
            ]
            if not safe_allies:
                world.add_debug(f"DOOMED_EVAC_SKIP src=p{src.id} reason=no safe target fall={fall_turn}")
                continue
            chosen = min(safe_allies, key=lambda a: dp(src, a))
            send = min(spare, max(DOOMED_EVAC_MIN_SHIPS, int(spare * 0.65)))
        angle, ok = world.aim(src, chosen, send)
        if not ok:
            continue
        eta = world.eta(src, chosen, send)
        proposals.append(MissionProposal(
            kind="DOOMED_EVACUATION",
            target_id=chosen.id,
            priority=82.0 + max(0, DOOMED_EVAC_HORIZON - fall_turn),
            required_ships=send,
            planned_sources=[(src.id, send, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"doomed evacuation src=p{src.id} fall={fall_turn} to=p{chosen.id}",
        ))
        world.add_debug(
            f"DOOMED_EVACUATION src=p{src.id} target=p{chosen.id} ships={send} eta={eta:.1f} fall={fall_turn}"
        )
    return proposals


def generate_expansion_missions(world, deadline):
    proposals = []
    if not world.neutral_planets or world.remaining < 35:
        return proposals
    # When enemy is nearly eliminated and we have forward presence, kill not expand
    if len(world.enemy_planets) <= 5 and is_breach_kill_mode(world) and world.step > BREACH_KILL_STEP_MIN:
        world.add_debug(
            f"EXPANSION_SKIP reason=breach_kill_priority enemies={len(world.enemy_planets)}"
        )
        return proposals
    candidates = nearest_neutral_candidates(world, deadline)
    for c in candidates[:4]:
        tgt = world.planet_by_id.get(c.target_id)
        if tgt is None:
            continue
        if world.step < 140:
            src_nearby = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
            if src_nearby and not validate_initial_target_choice(world, src_nearby, tgt):
                continue
        selected, total, _ = estimate_grouped_sources(world, tgt, c.need)
        if not selected or total < c.need:
            continue
        planned = [(s.id, send, angle, world.eta(s, tgt, send)) for s, send, angle in selected]
        eta_vals = [e for _, _, _, e in planned]
        proposals.append(MissionProposal(
            kind="CAPTURE_NEUTRAL",
            target_id=tgt.id,
            priority=c.score + (60.0 if c.can_lock else 0.0),
            required_ships=c.need,
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=c.reason,
        ))
    return proposals


def generate_local_strike_missions(world):
    """Command hubs (>= LOCAL_HUB_SHIPS) attack/capture the best neighbor first."""
    proposals = []
    for src in world.my_planets:
        if int(src.ships) < LOCAL_HUB_SHIPS:
            continue
        surplus = world.surplus(src)
        if surplus < 20:
            continue
        nearby = []
        for tgt in world.enemy_planets + world.neutral_planets:
            d = dp(src, tgt)
            if d > LOCAL_HUB_RADIUS:
                continue
            if tgt.owner not in (-1, world.player) and not should_allow_enemy_attack(
                world, tgt, "SYNC_ATTACK", "local_strike"
            ):
                continue
            need = world.ships_needed_to_capture(src, tgt)
            if need <= 0 or need > surplus:
                continue
            angle, ok = world.aim(src, tgt, need)
            if not ok:
                continue
            eta = world.eta(src, tgt, need)
            score = tgt.production * 15.0 - d * 0.8 - need * 0.3
            score += launchpad_target_score(world, src, tgt, StrategyMode.CONTEST_HUBS) * 0.12
            if tgt.owner not in (-1, world.player):
                score += 30.0
                if tgt.owner == world.leader:
                    score += 20.0
            if is_idle(tgt):
                score += 10.0
            nearby.append((score, tgt, need, angle, eta))
        if not nearby:
            continue
        nearby.sort(key=lambda x: -x[0])
        score_val, tgt, need, angle, eta = nearby[0]
        proposals.append(MissionProposal(
            kind="LOCAL_STRIKE",
            target_id=tgt.id,
            priority=PRIORITY_LOCAL_STRIKE_BASE + score_val * 0.4,
            required_ships=need,
            planned_sources=[(src.id, need, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"hub p{src.id}({int(src.ships)}) -> p{tgt.id} d={dp(src, tgt):.1f} need={need}",
        ))
    return proposals


def generate_sync_attack_missions(world, mode, deadline):
    """Coordinated multi-source attacks where all fleets arrive within ETA_SYNC_WINDOW turns."""
    proposals = []
    if not world.enemy_planets:
        return proposals
    if mode in (StrategyMode.ANTI_LEADER, StrategyMode.TURTLE_BREAKER):
        targets = ([e for e in world.enemy_planets if e.owner == world.leader]
                   or world.enemy_planets)[:4]
    elif mode in (StrategyMode.COLLAPSE, StrategyMode.FINAL_DRAIN, StrategyMode.FORCE_WAVE):
        targets = sorted(world.enemy_planets, key=lambda e: -e.production)[:4]
    else:
        targets = sorted(world.enemy_planets, key=lambda e: e.ships)[:4]

    for tgt in targets:
        if time.perf_counter() > deadline:
            break
        if not should_allow_enemy_attack(world, tgt, "SYNC_ATTACK", "sync_attack"):
            continue
        pool_srcs = sorted(
            [
                p for p in world.my_planets
                if world.surplus(p) >= 5
                and world.real_incoming_threat(p)["deficit"] <= 0
            ],
            key=lambda p: dp(p, tgt),
        )[:5]
        if not pool_srcs:
            continue
        pool = sum(world.surplus(p) for p in pool_srcs)
        need = world.ships_needed_to_capture(min(pool_srcs, key=lambda p: dp(p, tgt)), tgt, pool)
        enemy_help = sum(
            ships for eta, owner, ships in world.arrivals_by_target.get(tgt.id, [])
            if owner != world.player and eta <= 18
        )
        need = max(need, int(tgt.ships) + int(tgt.production) * 8 + enemy_help + max(8, int(tgt.production) * 4))
        if need <= 0 or pool < need:
            world.add_debug(f"SYNC_ATTACK_SKIP p{tgt.id} pool={pool} need={need}")
            continue
        source_etas = []
        for src in pool_srcs:
            av = world.surplus(src)
            if av < 5:
                continue
            send = min(av, max(5, need))
            angle, ok = world.aim(src, tgt, send)
            if not ok:
                continue
            eta = world.eta(src, tgt, send)
            source_etas.append((eta, src, av, angle))
        if len(source_etas) < 2:
            world.add_debug(f"SYNC_ATTACK_SKIP p{tgt.id} reason=need 2 sources")
            continue
        source_etas.sort()
        best_synced = []
        for i, (anchor_eta, _anchor_src, _anchor_av, _anchor_angle) in enumerate(source_etas):
            window = [
                (eta, src, av, angle) for eta, src, av, angle in source_etas
                if abs(eta - anchor_eta) <= 2.0
            ][:3]
            if sum(av for _, _, av, _ in window) >= need and len(window) >= 2:
                best_synced = window
                break
        if not best_synced:
            world.add_debug(f"SYNC_ATTACK_SKIP p{tgt.id} reason=no tight swarm need={need}")
            continue
        planned = []
        remaining = need
        for eta, src, av, _angle in best_synced:
            if remaining <= 0:
                break
            send = min(av, remaining)
            angle, ok = world.aim(src, tgt, send)
            if not ok:
                continue
            planned.append((src.id, send, angle, world.eta(src, tgt, send)))
            remaining -= send
        if not planned or sum(s for _, s, _, _ in planned) < need:
            continue
        eta_vals = [e for _, _, _, e in planned]
        if max(eta_vals) - min(eta_vals) > 2.0:
            world.add_debug(f"SYNC_ATTACK_SKIP p{tgt.id} reason=eta spread {max(eta_vals)-min(eta_vals):.1f}")
            continue
        ok_grp, grp_reason = validate_grouped_launch(world, tgt, planned)
        if not ok_grp:
            world.add_debug(f"SYNC_ATTACK_SKIP p{tgt.id} reason=validate_grouped_launch: {grp_reason}")
            continue
        if not world.can_hold_after_capture(tgt, max(eta_vals), sum(s for _, s, _, _ in planned)):
            world.add_debug(f"SYNC_ATTACK_SKIP p{tgt.id} reason=cannot hold")
            continue
        base_score = score_target(world, best_synced[0][1], tgt, mode)
        proposals.append(MissionProposal(
            kind="SYNC_ATTACK",
            target_id=tgt.id,
            priority=PRIORITY_SYNC_ATTACK_BASE + base_score * 0.25,
            required_ships=need,
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"sync p{tgt.id} need={need} sources={len(planned)} spread={max(eta_vals) - min(eta_vals):.1f}",
        ))
    return proposals


def generate_anti_leader_missions(world):
    proposals = []
    if world.leader is None or world.leader_score <= world.my_score * 1.22:
        return proposals
    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < 15:
        return proposals
    targets = [p for p in world.enemy_planets if p.owner == world.leader]
    if not targets:
        return proposals
    tgt = max(targets, key=lambda p: p.production * 12 + max(0, 45 - p.ships)
              - min(dp(m, p) for m in world.my_planets))
    if not should_allow_enemy_attack(world, tgt, "SYNC_ATTACK", "anti_leader"):
        return proposals
    candidate_sources = [
        p for p in world.my_planets
        if world.surplus(p) > 0
        and world.real_incoming_threat(p)["deficit"] <= 0
    ]
    prop = build_capture_plan(world, tgt, "SYNC_ATTACK", candidate_sources)
    if prop is None:
        return proposals
    prop.priority = PRIORITY_ANTI_LEADER_BASE + tgt.production * 3
    prop.reason = f"anti_leader p{tgt.id} prod={tgt.production} leader={world.leader}"
    proposals.append(prop)
    return proposals


def generate_collapse_missions(world, mode):
    proposals = []
    if mode not in (StrategyMode.COLLAPSE, StrategyMode.FORCE_WAVE):
        return proposals
    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < 20:
        return proposals
    targets = world.enemy_planets + (world.neutral_planets if world.step <= LATE_GAME_STEPS else [])
    candidate_sources = [
        p for p in world.my_planets
        if world.surplus(p) > 0
        and world.real_incoming_threat(p)["deficit"] <= 0
    ]
    for tgt in sorted(targets, key=lambda t: -(t.production * 10 - t.ships))[:4]:
        if tgt.owner not in (-1, world.player) and not should_allow_enemy_attack(
            world, tgt, "COLLAPSE", "collapse"
        ):
            continue
        prop = build_capture_plan(world, tgt, "COLLAPSE", candidate_sources)
        if prop is None:
            continue
        prop.priority = PRIORITY_COLLAPSE_BASE + tgt.production * 4
        prop.reason = f"collapse p{tgt.id} prod={tgt.production} need={prop.required_ships}"
        proposals.append(prop)
    return proposals


def is_protect_lead_mode(world):
    if world.step < PROTECT_LEAD_STEP and world.remaining > PROTECT_LEAD_REMAINING:
        return False
    if not world.enemy_planets:
        return True
    score_clear = world.my_score > max(1, world.leader_score) * 1.18
    prod_clear = world.my_prod >= max(1, world.enemy_prod) * 1.15
    ship_safe = world.my_total_ships >= max(1, world.enemy_total_ships) * 0.95
    return score_clear and prod_clear and ship_safe


def generate_protect_lead_missions(world):
    """Late winning posture: consolidate production/core planets instead of blind draining."""
    proposals = []
    if not is_protect_lead_mode(world):
        return proposals
    world.add_debug(
        f"PROTECT_LEAD_ACTIVE step={world.step} score={world.my_score}/{world.leader_score} prod={world.my_prod}/{world.enemy_prod}"
    )
    targets = sorted(
        [
            p for p in world.my_planets
            if int(p.production) >= 3
            and world.simulate_planet_timeline(p, min(40, world.remaining))["fall_turn"] is None
            and int(p.ships) < max(18, int(p.production) * 6)
        ],
        key=lambda p: (-int(p.production), int(p.ships)),
    )[:3]
    for tgt in targets:
        desired = max(18, int(tgt.production) * 6)
        need = max(0, desired - int(tgt.ships))
        if need <= 0:
            continue
        sources = sorted(
            [
                src for src in world.my_planets
                if src.id != tgt.id
                and world.surplus(src) > 0
                and world.real_incoming_threat(src)["deficit"] <= 0
                and world.simulate_planet_timeline(src, min(30, world.remaining))["fall_turn"] is None
            ],
            key=lambda s: (dp(s, tgt), -world.surplus(s)),
        )
        planned = []
        remaining = need
        for src in sources[:3]:
            if remaining <= 0:
                break
            send = min(world.surplus(src), remaining)
            if send < 4 and remaining > send:
                continue
            angle, ok = world.aim(src, tgt, send)
            if not ok:
                continue
            eta = world.eta(src, tgt, send)
            if eta > world.remaining - 2:
                continue
            planned.append((src.id, send, angle, eta))
            remaining -= send
        if not planned:
            continue
        eta_vals = [eta for _, _, _, eta in planned]
        proposals.append(MissionProposal(
            kind="DEFEND_HOLD",
            target_id=tgt.id,
            priority=PRIORITY_PROTECT_LEAD_BASE + int(tgt.production) * 3,
            required_ships=sum(s for _, s, _, _ in planned),
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"protect-lead reinforce p{tgt.id} desired={desired}",
        ))
    return proposals


ENDGAME_CONSOL_REMAINING  = 70   # steps remaining below which consolidation activates
ENDGAME_CONSOL_MIN_XFER   = 8    # minimum ships to bother transferring


def generate_endgame_consolidation_missions(world):
    """
    Pre-drain: consolidate surplus from weak backline planets into the best hub.
    Active only when remaining < 70 and past LATE_GAME_STEPS.
    """
    proposals = []
    if world.remaining >= ENDGAME_CONSOL_REMAINING:
        return proposals
    if world.step < LATE_GAME_STEPS:
        return proposals
    if len(world.my_planets) < 2:
        return proposals

    cx = sum(p.x for p in world.my_planets) / len(world.my_planets)
    cy = sum(p.y for p in world.my_planets) / len(world.my_planets)

    safe_hubs = [
        p for p in world.my_planets
        if world.real_incoming_threat(p)["deficit"] <= 0
        and world.simulate_planet_timeline(p, min(30, world.remaining))["fall_turn"] is None
        and world.nearest_enemy_distance(p) > FRONTLINE_DIST
    ]
    if not safe_hubs:
        return proposals

    def _hub_val(p):
        centrality = 1.0 / max(1.0, dist(p.x, p.y, cx, cy))
        return int(p.production) * 15.0 + int(p.ships) * 0.4 + centrality * 20.0

    hub = max(safe_hubs, key=_hub_val)

    for src in sorted(
        world.my_planets,
        key=lambda p: (world.nearest_enemy_distance(p), -world.surplus(p)),
        reverse=True,
    ):
        if len(proposals) >= 3:
            break
        if src.id == hub.id:
            continue
        if world.nearest_enemy_distance(src) < FRONTLINE_DIST:
            continue
        if world.real_incoming_threat(src)["deficit"] > 0:
            continue
        spare = world.surplus(src)
        send = int(spare * 0.75)
        if send < ENDGAME_CONSOL_MIN_XFER:
            continue
        eta = world.eta(src, hub, send)
        if eta > world.remaining - 1:
            continue
        angle, ok = world.aim(src, hub, send)
        if not ok:
            continue
        proposals.append(MissionProposal(
            kind="DEFEND_HOLD",
            target_id=hub.id,
            priority=PRIORITY_ENDGAME_CONSOL_BASE + int(hub.production),
            required_ships=send,
            planned_sources=[(src.id, send, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"endgame_consol src=p{src.id}->hub=p{hub.id} send={send}",
        ))
    return proposals


def generate_final_drain_missions(world):
    proposals = []
    if world.step < FINAL_STEPS and world.remaining > 45:
        return proposals
    if is_protect_lead_mode(world):
        world.add_debug("FINAL_DRAIN_SKIP reason=protect-lead mode active")
        return proposals
    for src in sorted(world.my_planets, key=lambda p: -int(p.ships)):
        spare = max(0, int(src.ships) - world.committed.get(src.id, 0) - 1)
        if spare <= 0:
            continue
        for tgt in sorted(world.enemy_planets + world.neutral_planets,
                          key=lambda t: world.eta(src, t, spare)):
            if world.eta(src, tgt, spare) > world.remaining - 1:
                continue
            need = world.ships_needed_to_capture(src, tgt, spare)
            if 0 < need <= spare:
                angle, ok = world.aim(src, tgt, need)
                if ok:
                    eta = world.eta(src, tgt, need)
                    proposals.append(MissionProposal(
                        kind="FINAL_DRAIN",
                        target_id=tgt.id,
                        priority=PRIORITY_FINAL_DRAIN_BASE + tgt.production,
                        required_ships=need,
                        planned_sources=[(src.id, need, angle, eta)],
                        eta_min=eta,
                        eta_max=eta,
                        reason=f"drain src=p{src.id} tgt=p{tgt.id}",
                    ))
                    break
    return proposals


def midgame_mission_quality_ok(world, prop, tgt, fleet_ratio, midgame_front):
    """Midgame anti-scatter filter: only launch missions that can convert or stabilize."""
    if prop.kind in CRITICAL_MISSIONS or prop.kind in (
        "DEFEND_HOLD", "SAVE_UNDER_ATTACK", "RECAPTURE_LOST",
        "FINISH_ZERO_CAPTURE", "REINFORCE_CAPTURE", "DOOMED_EVACUATION"
    ):
        return True, ""

    total_send = sum(s for _, s, _, _ in prop.planned_sources)
    action_count = max(1, len(prop.planned_sources))
    avg_send = total_send / action_count

    if fleet_ratio > MIDGAME_FLEET_PANIC:
        if tgt.owner == -1 and int(tgt.ships) <= FINISH_ZERO_MAX_SHIPS:
            return True, ""
        return False, f"midgame panic fleet ratio {fleet_ratio:.2f}"

    if avg_send < MIDGAME_MIN_AVG_SHIPS:
        return False, f"anti-scatter avg ships {avg_send:.1f}"

    if tgt.owner not in (-1, world.player):
        if midgame_front is not None and dp(tgt, midgame_front) > MIDGAME_FRONT_RADIUS:
            return False, f"outside active front d={dp(tgt, midgame_front):.1f}"
        if total_send < max(MIDGAME_MIN_WAVE_SHIPS, int(prop.required_ships * MIN_WAVE_FRACTION)):
            return False, f"weak attack wave send={total_send} need={prop.required_ships}"
        if len(prop.planned_sources) > MIDGAME_ATTACK_SOURCE_MAX:
            return False, f"too many scattered sources n={len(prop.planned_sources)}"
        if len(prop.planned_sources) < 2 and total_send < MIDGAME_MIN_WAVE_SHIPS * 2:
            return False, "single-source enemy poke blocked"

    if prop.kind in ("SYNC_ATTACK", "BREACH_KILL", "COLLAPSE") and fleet_ratio > MIDGAME_FLEET_HARD:
        return False, f"blocked speculative attack fleet={fleet_ratio:.2f}"
    if prop.kind in ("COLLAPSE", "FINAL_DRAIN") and tgt.owner not in (-1, world.player) and world.my_prod < world.enemy_prod * 1.3:
        return False, "collapse not ready"

    return True, ""


def normalize_proposal(world, prop):
    """
    Mission-level packet validation. The planner is responsible for funding the
    grouped mission; this function only rejects invalid packet shapes and
    recomputes ETAs without per-source over-rounding.
    """
    tgt = world.planet_by_id.get(prop.target_id)
    if tgt is None:
        return False
    world.add_debug(f"MISSION_LEVEL_NORMALIZE {prop.kind} p{prop.target_id}")
    world.add_debug("PER_SOURCE_OVERROUND_REMOVED")

    grouped_normalize_missions = {
        "CAPTURE_NEUTRAL",
        "LOCAL_PRODUCTION_CAPTURE",
        "HIGH_VALUE_NEUTRAL_RACE",
        "SYNC_ATTACK",
        "BREACH_KILL",
        "COLLAPSE",
        "SNIPE_NEUTRAL",
    }
    if prop.kind in grouped_normalize_missions and tgt.owner != world.player:
        planned_ids = [src_id for src_id, _, _, _ in prop.planned_sources]
        preferred = [
            world.planet_by_id[src_id]
            for src_id in planned_ids
            if src_id in world.planet_by_id and world.planet_by_id[src_id].owner == world.player
        ]
        preferred_ids = {p.id for p in preferred}
        nearby = sorted(
            [
                p for p in world.my_planets
                if p.id not in preferred_ids
                and world.real_incoming_threat(p)["deficit"] <= 0
                and world.surplus(p) > 0
            ],
            key=lambda p: dp(p, tgt),
        )
        required = max(int(prop.required_ships), sum(int(s) for _, s, _, _ in prop.planned_sources))
        plan, reason = build_grouped_funding_plan(
            world,
            tgt,
            required,
            preferred + nearby,
            prop.kind,
            max_sources=max(MAX_GROUP_SOURCES, len(preferred) + 2),
            eta_spread_limit=3.0 if tgt.owner == -1 else 6.0,
            require_hold=prop.kind != "SNIPE_NEUTRAL",
        )
        if plan is None:
            world.add_debug(
                f"GROUPED_FUNDING_REJECT_UNDERFUNDED {prop.kind} p{prop.target_id} reason={reason}"
            )
            return False
        planned, total, eta_min, eta_max = plan
        prop.planned_sources = planned
        prop.required_ships = total
        prop.eta_min = eta_min
        prop.eta_max = eta_max
        return True

    new_sources = []
    for src_id, ships, angle, _eta in prop.planned_sources:
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        ships = int(ships)
        available = int(src.ships) - world.committed.get(src_id, 0)
        if ships > available or not valid_packet_size(prop.kind, ships):
            world.add_debug(
                f"COMMIT_REJECT_INVALID_PACKET mission={prop.kind} src=p{src_id} "
                f"ships={ships} available={available} reason=proposal_packet"
            )
            return False
        new_eta = world.eta(src, tgt, ships)
        new_sources.append((src_id, ships, angle, new_eta))
    if not new_sources:
        return False
    prop.planned_sources = new_sources
    prop.required_ships  = sum(s for _, s, _, _ in new_sources)
    etas = [e for _, _, _, e in new_sources]
    prop.eta_min = min(etas)
    prop.eta_max = max(etas)
    tgt = world.planet_by_id.get(prop.target_id)
    if (
        tgt is not None
        and prop.kind in OFFENSIVE_MISSIONS
        and prop.kind != "FINAL_DRAIN"
        and tgt.owner != world.player
    ):
        ok_grp, grp_reason = validate_grouped_launch(world, tgt, prop.planned_sources)
        if not ok_grp:
            world.add_debug(
                f"PARTIAL_PACKET_REJECT_NO_CONVERSION {prop.kind} p{prop.target_id} reason={grp_reason}"
            )
            return False
    return True


def coordinate_missions(world, proposals, moves, fleet_ratio, deadline, midgame_active=False, midgame_front=None):
    """Select best non-conflicting missions. Enforce fleet ratio cap. Prevent scatter."""
    proposals.sort(key=lambda p: -p.priority)

    spent: dict = {}         # src_id -> ships committed this coordinator pass
    used_targets: set = set()
    defense_count = 0
    expansion_count = 0
    attack_count = 0
    breach_count = 0
    recapture_count = 0
    midgame_major_count = 0
    if midgame_active:
        midgame_major_count = min(
            MIDGAME_MAJOR_MISSION_LIMIT,
            sum(1 for move in moves if len(move) >= 3 and int(move[2]) >= MIDGAME_MIN_AVG_SHIPS),
        )

    for prop in proposals:
        prop.kind = canonical_mission_type(prop.kind)
        if time.perf_counter() > deadline:
            break
        if prop.target_id in used_targets:
            continue

        # Normalize planned_sources before ledger / validation so the ledger,
        # valid_fleet_launch, and commit all see the same normalized packet sizes.
        if not normalize_proposal(world, prop):
            world.add_debug(
                f"NORMALIZE_PROPOSAL_REJECT {prop.kind} p{prop.target_id} "
                f"reason=no valid normalized sources"
            )
            continue

        tgt = world.planet_by_id.get(prop.target_id)
        immediate_value_capture = (
            tgt is not None
            and tgt.owner == -1
            and prop.kind in ("LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE", "CAPTURE_NEUTRAL")
            and (int(tgt.production) >= LAUNCHPAD_PROD_MIN or prop.eta_max <= 8)
        )
        # Fleet ratio guards — after 0.60 stop normal launches; high-value sure captures/recaptures may still convert.
        if fleet_ratio > FLEET_RATIO_HARD and prop.kind not in CRITICAL_MISSIONS | {"SAVE_UNDER_ATTACK", "HIGH_VALUE_NEUTRAL_RACE"} and not immediate_value_capture:
            world.add_debug(f"COORD_SKIP blocked: fleet ratio too high ratio={fleet_ratio:.2f} mission={prop.kind}")
            continue
        if fleet_ratio > FLEET_RATIO_SOFT and prop.kind not in CRITICAL_MISSIONS | {"SAVE_UNDER_ATTACK", "LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE"} and not immediate_value_capture:
            world.add_debug(f"COORD_SKIP blocked: fleet ratio too high ratio={fleet_ratio:.2f} mission={prop.kind}")
            continue

        # Per-kind limits: prevent scatter
        if prop.kind in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK") and defense_count >= len(world.my_planets):
            continue
        if prop.kind in (
            "CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE",
            "FINISH_ZERO_CAPTURE", "SNIPE_NEUTRAL"
        ) and expansion_count >= 2:
            continue
        if prop.kind == "BREACH_KILL" and breach_count >= 1:
            continue
        if prop.kind in ("LOCAL_STRIKE", "SYNC_ATTACK") and attack_count >= 1:
            continue
        if prop.kind == "COLLAPSE" and attack_count >= 2:
            continue
        if prop.kind == "RECAPTURE_LOST" and recapture_count >= 2:
            continue

        # Source availability check
        can_commit = True
        for src_id, ships, _, _ in prop.planned_sources:
            src = world.planet_by_id.get(src_id)
            if src is None:
                can_commit = False
                break
            available = int(src.ships) - world.committed.get(src_id, 0) - spent.get(src_id, 0)
            if ships > available:
                can_commit = False
                break
        if not can_commit:
            continue

        # Reject tiny sends unless finishing or defending
        total_send = sum(s for _, s, _, _ in prop.planned_sources)
        if total_send < 5 and prop.kind not in (
            "FINISH_ZERO_CAPTURE", "DEFEND_HOLD", "SAVE_UNDER_ATTACK",
            "DOOMED_EVACUATION", "RECAPTURE_LOST", "REINFORCE_CAPTURE",
        ):
            continue

        if tgt is None:
            continue

        if midgame_active:
            ok, reason = midgame_mission_quality_ok(world, prop, tgt, fleet_ratio, midgame_front)
            if not ok:
                world.add_debug(f"MIDGAME_QUALITY_BLOCK {prop.kind} p{prop.target_id} reason={reason}")
                continue
            is_major = (
                prop.kind in OFFENSIVE_MISSIONS
                and prop.kind not in ("FINISH_ZERO_CAPTURE", "SNIPE_NEUTRAL")
                and sum(s for _, s, _, _ in prop.planned_sources) >= MIDGAME_MIN_WAVE_SHIPS
            )
            if is_major and midgame_major_count >= MIDGAME_MAJOR_MISSION_LIMIT:
                world.add_debug(f"MIDGAME_SCATTER_BLOCK {prop.kind} p{prop.target_id} reason=major mission limit")
                continue

        mission_valid = True
        for src_id, ships, _, _ in prop.planned_sources:
            src = world.planet_by_id.get(src_id)
            if src is None:
                mission_valid = False
                break
            ok, reason = world.valid_fleet_launch(
                src, tgt, ships, prop.kind, planned_sources=prop.planned_sources, mission_reason=prop.reason
            )
            if not ok:
                world.add_debug(
                    f"COORD_SKIP {prop.kind} target=p{prop.target_id} reason={reason} "
                    f"sources={[s for s, _, _, _ in prop.planned_sources]}"
                )
                mission_valid = False
                break
        if not mission_valid:
            continue

        # Block speculative offensive moves from wave-reserved sources
        if _wave_reservation["target_id"] is not None and prop.kind in (
            "SYNC_ATTACK", "COLLAPSE"
        ) and prop.target_id != _wave_reservation["target_id"]:
            rsv_set = set(_wave_reservation["source_ids"])
            if any(src_id in rsv_set for src_id, _, _, _ in prop.planned_sources):
                world.add_debug(
                    f"WAVE_RESERVE_BLOCK {prop.kind} p{prop.target_id} "
                    f"blocked for wave on p{_wave_reservation['target_id']}"
                )
                continue

        mission_id = world.mission_ledger.create_from_proposal(prop)
        committed_any = False
        for src_id, ships, _, _ in prop.planned_sources:
            src = world.planet_by_id.get(src_id)
            if src is None:
                continue
            available = int(src.ships) - world.committed.get(src_id, 0) - spent.get(src_id, 0)
            send = min(ships, available)
            if send <= 0:
                continue
            if world.commit(src, tgt, send, moves, mission_type=prop.kind, mission_id=mission_id, planned_sources=prop.planned_sources):
                spent[src_id] = spent.get(src_id, 0) + send
                committed_any = True

        if committed_any:
            world.wave_attempted = True
            used_targets.add(prop.target_id)
            world.add_debug(f"COORD {prop.kind} p{prop.target_id} send={total_send} {prop.reason}")
            if prop.kind in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK"):
                defense_count += 1
            elif prop.kind == "RECAPTURE_LOST":
                recapture_count += 1
            elif prop.kind in (
                "CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE",
                "FINISH_ZERO_CAPTURE", "SNIPE_NEUTRAL"
            ):
                expansion_count += 1
            elif prop.kind == "BREACH_KILL":
                breach_count += 1
            elif prop.kind in ("SYNC_ATTACK", "COLLAPSE", "FINAL_DRAIN"):
                attack_count += 1
            if midgame_active and (
                prop.kind in OFFENSIVE_MISSIONS
                and prop.kind not in ("FINISH_ZERO_CAPTURE", "SNIPE_NEUTRAL")
                and total_send >= MIDGAME_MIN_WAVE_SHIPS
            ):
                midgame_major_count += 1


def is_breach_kill_mode(world):
    """True when we have a forward planet inside enemy territory and can press the kill."""
    if not world.enemy_planets or not world.my_planets:
        return False
    if (
        world.is_four_player
        and world.step < FOUR_P_ATTACK_STEP
        and world.neutral_planets
        and not any(is_local_enemy_opportunity(world, e) or _is_corner_control_target(world, e) for e in world.enemy_planets)
    ):
        return False
    has_forward = any(
        min(dp(mp, ep) for ep in world.enemy_planets) <= BREACH_KILL_DIST
        for mp in world.my_planets
    )
    if not has_forward:
        return False
    return (
        world.step > BREACH_KILL_STEP_MIN
        or len(world.neutral_planets) < 8
        or len(world.my_planets) >= len(world.enemy_planets)
        or world.my_prod >= world.enemy_prod * 0.9
    )


def generate_breach_kill_missions(world):
    """Convert forward presence into synchronized enemy planet captures."""
    if not is_breach_kill_mode(world):
        return []

    kill_close = len(world.enemy_planets) <= 5
    proposals = []
    forward = [
        mp for mp in world.my_planets
        if any(dp(mp, ep) <= BREACH_KILL_DIST for ep in world.enemy_planets)
    ]
    if not forward:
        return []

    all_sources = [p for p in world.my_planets if world.surplus(p) >= 5]

    seen_targets: set = set()
    for fwd in sorted(forward, key=lambda p: -world.surplus(p)):
        nearby_enemies = sorted(
            [(dp(fwd, ep), ep) for ep in world.enemy_planets
             if dp(fwd, ep) <= BREACH_KILL_DIST + 10],
            key=lambda x: x[0],
        )
        for d_to_enemy, tgt in nearby_enemies:
            if tgt.id in seen_targets:
                continue
            if not should_allow_enemy_attack(world, tgt, "BREACH_KILL", "breach_kill"):
                continue
            pool_srcs = sorted(
                [s for s in all_sources if dp(s, tgt) <= BREACH_KILL_DIST + 10],
                key=lambda s: dp(s, tgt),
            )[:6]
            if not pool_srcs:
                continue
            pool = sum(world.surplus(s) for s in pool_srcs)
            need = world.ships_needed_to_capture(
                min(pool_srcs, key=lambda s: dp(s, tgt)), tgt, pool
            )
            if need <= 0 or pool < need:
                continue

            source_etas = []
            for src in pool_srcs:
                av = world.surplus(src)
                if av < 5:
                    continue
                send = min(av, max(5, int(av * 0.7)))
                angle, ok = world.aim(src, tgt, send)
                if not ok:
                    continue
                eta = world.eta(src, tgt, send)
                source_etas.append((eta, src, send, angle))

            if not source_etas:
                continue
            source_etas.sort()
            anchor_eta = source_etas[-1][0]
            synced = [
                (eta, src, send, angle) for eta, src, send, angle in source_etas
                if anchor_eta - eta <= BREACH_ETA_SYNC
            ]
            if sum(s for _, _, s, _ in synced) < need:
                continue

            planned_check = [(src.id, send, angle, eta) for eta, src, send, angle in synced]
            ok_grp, grp_reason = validate_grouped_launch(world, tgt, planned_check)
            if not ok_grp:
                world.add_debug(f"SKIP BREACH_KILL p{tgt.id} step={world.step} reason=validate_grouped_launch:{grp_reason}")
                continue

            score = (
                int(tgt.production) * 120.0
                + max(0, 50 - int(tgt.ships)) * 2.0
                + max(0.0, BREACH_KILL_DIST - d_to_enemy) * 2.0
                - need * 0.5
            )
            if tgt.owner == world.leader:
                score += 60.0

            planned = [(src.id, send, angle, eta) for eta, src, send, angle in synced]
            breach_base_priority = PRIORITY_BREACH_KILL_BASE + (PRIORITY_BREACH_KILL_CLOSE if kill_close else 0.0)
            bk_priority = breach_base_priority + score * 0.1
            proposals.append(MissionProposal(
                kind="BREACH_KILL",
                target_id=tgt.id,
                priority=bk_priority,
                required_ships=need,
                planned_sources=planned,
                eta_min=source_etas[0][0],
                eta_max=anchor_eta,
                reason=(
                    f"breach p{tgt.id} prod={tgt.production} ships={int(tgt.ships)} "
                    f"d={d_to_enemy:.1f} fwd=p{fwd.id} need={need} "
                    f"sources={len(synced)} spread={anchor_eta - source_etas[0][0]:.1f}"
                ),
            ))
            world.add_debug(
                f"SELECT BREACH_KILL p{tgt.id} step={world.step} "
                f"srcs={[s for s, _, _, _ in planned]} ships={sum(s for _, s, _, _ in planned)} "
                f"eta={anchor_eta:.1f} priority={bk_priority:.1f} "
                f"reason=breach fwd=p{fwd.id} spread={anchor_eta - source_etas[0][0]:.1f}"
            )
            seen_targets.add(tgt.id)
            break  # one target per forward planet

    return proposals


_missed_skipped: dict = {}


def early_surplus(world, p):
    """Light reserve for early game: avoids over-holding during expansion."""
    if world.step < 80:
        reserve = 2
    elif world.step < 140:
        reserve = 3
    else:
        reserve = world.reserve_for(p)
    return max(0, int(p.ships) - world.committed.get(p.id, 0) - reserve)


def should_wait_for_better_wave(world, target):
    """True if waiting 2-3 turns will produce enough ships to capture target."""
    if target is None or target.id in world.comet_ids:
        return False
    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if src is None:
        return False
    need = world.ships_needed_to_capture(src, target, world.my_total_ships)
    if need <= 0:
        return False
    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool >= need * MIN_WAVE_FRACTION:
        return False
    for wait in (2, 3):
        projected = pool + sum(int(p.production) * wait for p in world.my_planets)
        if projected >= need:
            world.add_debug(
                f"WAVE_WAIT target=p{target.id} now={pool} projected_{wait}={projected} need={need}"
            )
            return True
    return False


def reserve_wave(world, target, sources):
    """Record a wave reservation so other missions don't consume these sources."""
    global _wave_reservation
    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    need = world.ships_needed_to_capture(src, target, world.my_total_ships) if src else 0
    _wave_reservation.update({
        "target_id": target.id,
        "source_ids": [s.id for s in sources],
        "started_step": world.step,
        "launch_by_step": world.step + MAX_WAVE_WAIT,
        "required_ships": need,
        "reason": f"p{target.id} need={need}",
    })
    world.add_debug(
        f"WAVE_RESERVE target=p{target.id} sources={[s.id for s in sources]} "
        f"launch_by={world.step + MAX_WAVE_WAIT}"
    )


def wait_for_wave_if_better(world, target, sources, required_ships):
    """
    Reserve sources and wait 1-3 turns if target is valuable and pool is close.
    Returns True if we reserve and should skip launching now.
    Never waits for prod < 4 neutrals or when already have enough ships.
    """
    prod = int(target.production)
    is_valuable = prod >= 4 or target.owner not in (-1, world.player)
    if not is_valuable:
        return False
    pool = sum(world.surplus(s) for s in sources)
    if pool >= required_ships:
        return False
    gap = required_ships - pool
    for wait in range(1, MAX_WAVE_WAIT + 1):
        gain = sum(int(s.production) * wait for s in sources)
        if gain >= gap:
            reserve_wave(world, target, sources)
            world.add_debug(
                f"WAIT_FOR_WAVE target=p{target.id} prod={prod} pool={pool} "
                f"need={required_ships} wait={wait}"
            )
            return True
    return False


def generate_organized_wave_mission(world, target):
    """Build a full-strength MissionProposal for a wave-reserved target."""
    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if src is None:
        return None
    pool = sum(world.surplus(p) for p in world.my_planets)
    need = world.ships_needed_to_capture(src, target, pool)
    if need <= 0 or pool < need:
        return None
    sources = sorted(
        [p for p in world.my_planets
         if world.real_incoming_threat(p)["deficit"] <= 0 and world.surplus(p) >= 5],
        key=lambda p: dp(p, target),
    )[:MAX_GROUP_SOURCES]
    planned = []
    remaining_need = need
    for s in sources:
        if remaining_need <= 0:
            break
        av = world.surplus(s)
        if av < 3:
            continue
        send = min(av, remaining_need)
        angle, ok = world.aim(s, target, send)
        if not ok:
            continue
        eta = world.eta(s, target, send)
        planned.append((s.id, send, angle, eta))
        remaining_need -= send
    if not planned or sum(sh for _, sh, _, _ in planned) < need:
        return None
    eta_vals = [e for _, _, _, e in planned]
    score = int(target.production) * 120.0 + max(0, 50 - int(target.ships)) * 2.0 - need * 0.5
    if target.owner == world.leader:
        score += 60.0
    world.add_debug(
        f"WAVE_LAUNCH target=p{target.id} ships={sum(sh for _, sh, _, _ in planned)} "
        f"sources={[s for s, _, _, _ in planned]}"
    )
    return MissionProposal(
        kind="BREACH_KILL",
        target_id=target.id,
        priority=90.0 + score * 0.1,
        required_ships=need,
        planned_sources=planned,
        eta_min=min(eta_vals),
        eta_max=max(eta_vals),
        reason=f"organized_wave p{target.id} need={need}",
    )


# ── early target intelligence ─────────────────────────────────────────────────

def _connects_good_neutral(world, tgt, radius=28.0, min_prod=2):
    return any(
        n.id != tgt.id and dp(tgt, n) <= radius and int(n.production) >= min_prod
        for n in world.neutral_planets
    )


def _support_source_count(world, tgt, radius=42.0):
    return sum(1 for m in world.my_planets if dp(m, tgt) <= radius and world.surplus(m) > 0)


# ── start-type-aware opening ──────────────────────────────────────────────────

def classify_start_type(world):
    """Classify my starting planet as LARGE / SMALL / MEDIUM based on radius role."""
    for p in world.initial_planets.values():
        if p.owner == world.player:
            return radius_class(p)
    return "MEDIUM"


def get_start_type(world):
    """Cached start-type. Logs classification once per game."""
    if _start_type_cache.get(world.player) is None:
        st = classify_start_type(world)
        _start_type_cache[world.player] = st
        world.add_debug(f"START_RADIUS_CLASSIFIED_{st}")
        world.add_debug(f"OPENING_CLASS_{st}_START prod={world.my_planets[0].production if world.my_planets else '?'} step={world.step}")
        start_planet = next((p for p in world.initial_planets.values() if p.owner == world.player), None)
        current = world.planet_by_id.get(start_planet.id) if start_planet is not None else None
        if current is not None and current.owner == world.player:
            if st == "LARGE":
                _primary_launchpads.setdefault(world.player, {})[current.id] = world.step
                world.add_debug(f"PRIMARY_LAUNCHPAD_MARKED p{current.id} radius={current.radius:.1f}")
                if not is_static_planet(current):
                    world.add_debug(f"ROTATING_LAUNCHPAD_MARKED p{current.id}")
            elif st == "MEDIUM":
                world.add_debug(f"MEDIUM_BRIDGE_MARKED p{current.id} radius={current.radius:.1f}")
            else:
                world.add_debug(f"SMALL_STORAGE_MARKED p{current.id} radius={current.radius:.1f}")
    return _start_type_cache[world.player]


def mark_launchpad_after_capture(world, p):
    store = _primary_launchpads.setdefault(world.player, {})
    role = radius_class(p)
    if role == "LARGE":
        if p.id not in store:
            store[p.id] = world.step
            world.add_debug(f"PRIMARY_LAUNCHPAD_MARKED p{p.id} radius={p.radius:.1f}")
            if not is_static_planet(p):
                world.add_debug(f"ROTATING_LAUNCHPAD_MARKED p{p.id}")
        return True
    if role == "MEDIUM":
        world.add_debug(f"MEDIUM_BRIDGE_MARKED p{p.id} radius={p.radius:.1f}")
        start_type = _start_type_cache.get(world.player)
        large_owned = any(radius_class(m) == "LARGE" for m in world.my_planets)
        if start_type == "SMALL" and not large_owned and p.id not in store:
            store[p.id] = world.step
            world.add_debug(f"PRIMARY_LAUNCHPAD_MARKED p{p.id} radius={p.radius:.1f}")
        return True
    world.add_debug(f"SMALL_STORAGE_MARKED p{p.id} radius={p.radius:.1f}")
    return False


def owned_launchpads(world):
    store = _primary_launchpads.setdefault(world.player, {})
    pads = []
    for p in world.my_planets:
        if radius_class(p) == "LARGE" or p.id in store:
            pads.append(p)
    return pads


def is_launchpad_candidate(world, p, start_type):
    if p is None or p.owner == world.player or world.is_comet(p):
        return False
    role = radius_class(p)
    if start_type == "SMALL":
        return role in ("MEDIUM", "LARGE")
    if start_type == "MEDIUM":
        return role == "LARGE"
    return role in ("MEDIUM", "LARGE")


def launchpad_role_score(world, p, start_type):
    role = radius_class(p)
    d = world.cluster_distance(p)
    src = min(world.my_planets, key=lambda m: dp(m, p), default=None)
    need = world.ships_needed_to_capture(src, p, world.my_total_ships) if src else int(p.ships) + 1
    score = -d * 3.0 - need * 1.4 + int(p.production) * 18.0
    if role == "LARGE":
        score += 260.0
    elif role == "MEDIUM":
        score += 130.0
    else:
        # Small planet: replace blanket -140 with bridge-value-aware scoring.
        # If the planet has meaningful bridge/connector value, it should be
        # captured. Otherwise penalise but not as harshly as before.
        bv = small_bridge_score(world, p)
        if bv >= SMALL_BRIDGE_THRESHOLD:
            score += bv * 0.8          # bridge value partially offsets small penalty
        else:
            score -= 80.0              # cheaper penalty; still below medium/large
    if is_static_planet(p):
        score += 120.0
        world.add_debug(f"STATIC_LAUNCHPAD_BONUS p{p.id} role={role}")
    if start_type == "SMALL" and role == "LARGE":
        score += 70.0
    if start_type == "MEDIUM" and role == "LARGE":
        score += 100.0
    if start_type == "LARGE" and role == "LARGE":
        score += 90.0
    return score


def find_nearest_launchpad_target(world, start_type):
    world.add_debug(f"LAUNCHPAD_TARGET_SCAN start={start_type}")
    candidates = []
    if not world.my_planets:
        return None
    for tgt in world.neutral_planets:
        if not is_launchpad_candidate(world, tgt, start_type):
            continue
        src = min(world.my_planets, key=lambda m: dp(m, tgt), default=None)
        if src is None:
            continue
        pool = sum(world.surplus(p) for p in world.my_planets if world.real_incoming_threat(p)["deficit"] <= 0)
        need = world.ships_needed_to_capture(src, tgt, pool)
        if need <= 0 or pool < normalize_send_amount(need):
            continue
        eta = world.eta(src, tgt, max(1, need))
        if eta > 60:
            continue
        role = radius_class(tgt)
        score = launchpad_role_score(world, tgt, start_type)
        candidates.append((score, dp(src, tgt), eta, need, tgt))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0]


def _corner_control_planets(world, owner=None, zones=None):
    zones = set(zones or (
        "my_start_corner",
        "clockwise_adjacent_corner",
        "counterclockwise_adjacent_corner",
        "opposite_corner",
    ))
    planets = world.normal_planets
    if owner is not None:
        planets = [p for p in planets if p.owner == owner]
    return [
        p for p in planets
        if classify_corner_zone(world, p) in zones
        and radius_class(p) in ("MEDIUM", "LARGE")
        and not world.is_comet(p)
    ]


def _owned_stable_corner_launchpads(world, zones):
    return [
        p for p in _corner_control_planets(world, owner=world.player, zones=zones)
        if is_static_planet(p)
    ]


def _my_corner_secured(world):
    stable = _owned_stable_corner_launchpads(world, ("my_start_corner",))
    if stable:
        return True
    owned_medium_large = _corner_control_planets(world, owner=world.player, zones=("my_start_corner",))
    return len(owned_medium_large) >= 2 and any(radius_class(p) == "LARGE" for p in owned_medium_large)


def _adjacent_corner_secured(world, zone):
    stable = _owned_stable_corner_launchpads(world, (zone,))
    return bool(stable) or sum(1 for p in _corner_control_planets(world, owner=world.player, zones=(zone,))) >= 2


def _has_useful_rotating_bridge(world):
    if not world.my_planets:
        return False
    adjacent_zones = ("clockwise_adjacent_corner", "counterclockwise_adjacent_corner")
    launchpads = owned_launchpads(world) or [
        p for p in world.my_planets if radius_class(p) in ("MEDIUM", "LARGE")
    ]
    for p in world.my_planets:
        if is_static_planet(p) or radius_class(p) not in ("MEDIUM", "LARGE"):
            continue
        zone = classify_corner_zone(world, p)
        near_pad = min((dp(p, hub) for hub in launchpads if hub.id != p.id), default=0.0)
        if zone == "my_start_corner" and near_pad <= FOUR_P_CORNER_LOCAL_DIST:
            return True
        if zone in adjacent_zones and near_pad <= FOUR_P_ADJACENT_BRIDGE_DIST:
            return True
    return False


def _corner_bridge_score(world, target):
    role = radius_class(target)
    if role == "SMALL":
        return -40.0
    launchpads = owned_launchpads(world) or [
        p for p in world.my_planets if radius_class(p) in ("MEDIUM", "LARGE")
    ]
    nearest_pad_d = min((dp(target, hub) for hub in launchpads), default=999.0)
    adjacent_static = [
        p for p in world.normal_planets
        if p.id != target.id
        and classify_corner_zone(world, p) in ("clockwise_adjacent_corner", "counterclockwise_adjacent_corner")
        and radius_class(p) in ("MEDIUM", "LARGE")
        and is_static_planet(p)
        and p.owner != world.player
        and not world.is_comet(p)
    ]
    shortcut_value = 0.0
    for hub in launchpads:
        for anchor in adjacent_static:
            shortcut_value = max(shortcut_value, dp(hub, anchor) - dp(target, anchor))
    adjacency_value = 45.0 if classify_corner_zone(world, target) in (
        "clockwise_adjacent_corner", "counterclockwise_adjacent_corner"
    ) else 18.0
    launchpad_connection_value = max(0.0, FOUR_P_ADJACENT_BRIDGE_DIST - nearest_pad_d) * 1.3
    bridge_score = max(0.0, shortcut_value) * 2.2 + adjacency_value + launchpad_connection_value
    if role == "MEDIUM":
        bridge_score += 55.0
    return bridge_score


def static_corner_value(world, target):
    if target is None or world.is_comet(target):
        return -1e9
    zone = classify_corner_zone(world, target)
    role = radius_class(target)
    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if src is None:
        return -1e9
    d = dp(src, target)
    pool = sum(
        world.surplus(p) for p in world.my_planets
        if world.real_incoming_threat(p)["deficit"] <= 0 and dp(p, target) <= FOUR_P_ADJACENT_BRIDGE_DIST + 12
    )
    need = world.ships_needed_to_capture(src, target, max(pool, 1))
    eta = world.eta(src, target, max(1, need))
    score = int(target.production) * 42.0 - d * 2.3 - eta * 4.0 - need * 1.1

    if is_static_planet(target):
        score += 360.0
        if role == "LARGE":
            score += 280.0
            world.add_debug(f"STATIC_HIGH_RADIUS_PRIORITY p{target.id} zone={zone}")
        elif role == "MEDIUM":
            score += 150.0
        else:
            bv = small_bridge_score(world, target)
            score += bv * 0.7 - 60.0  # static small: bridge value rescues it
    elif role == "LARGE":
        score += 120.0
    elif role == "MEDIUM":
        score += 75.0
    else:
        bv = small_bridge_score(world, target)
        if bv >= SMALL_BRIDGE_THRESHOLD:
            score += bv * 0.6 - 20.0
        else:
            score -= 120.0

    if zone == "my_start_corner":
        score += 170.0 if not _my_corner_secured(world) else 45.0
    elif zone in ("clockwise_adjacent_corner", "counterclockwise_adjacent_corner"):
        score += 120.0 if _my_corner_secured(world) else -45.0
    else:
        score += 90.0 if (
            _adjacent_corner_secured(world, "clockwise_adjacent_corner")
            and _adjacent_corner_secured(world, "counterclockwise_adjacent_corner")
        ) else -180.0

    score += _corner_bridge_score(world, target)
    enemy_t = min((world.eta(e, target, max(1, int(e.ships))) for e in world.enemy_planets), default=999.0)
    if enemy_t < eta + 5:
        score += 90.0
    if pool < round_up_to_granularity(max(need, MIN_SEND_SHIPS)):
        score -= 220.0
    if target.owner not in (-1, world.player):
        if world.step < FOUR_P_ATTACK_STEP and not is_local_enemy_opportunity(world, target):
            score -= 240.0
        if int(target.ships) <= ENEMY_GATE_WEAK_LOCAL:
            score += 85.0
    return score


def _corner_candidate_sources(world, target):
    return [
        p for p in world.my_planets
        if world.real_incoming_threat(p)["deficit"] <= 0
        and world.surplus(p) >= MIN_SEND_SHIPS
        and (
            dp(p, target) <= FOUR_P_ADJACENT_BRIDGE_DIST + 12
            or radius_class(p) == "LARGE"
            or p.id in _primary_launchpads.get(world.player, {})
        )
    ]


def _corner_target_allowed(world, target):
    if target.owner == world.player or world.is_comet(target):
        return False
    # Small planets are allowed when they have meaningful bridge value for
    # connecting corners or launchpads; otherwise still excluded from the
    # primary corner strategy to keep focus on medium/large targets.
    if radius_class(target) == "SMALL":
        bv = small_bridge_score(world, target)
        if bv < SMALL_BRIDGE_THRESHOLD:
            return False
    if target.owner == -1:
        return True
    if world.step >= FOUR_P_ATTACK_STEP:
        return should_allow_enemy_attack(world, target, "SYNC_ATTACK", "4p_corner")
    close = world.cluster_distance(target) <= MIDGAME_FRONT_RADIUS
    weak = int(target.ships) <= ENEMY_GATE_WEAK_LOCAL
    blocks_corner = (
        classify_corner_zone(world, target) in ("my_start_corner", "clockwise_adjacent_corner", "counterclockwise_adjacent_corner")
        and is_static_planet(target)
    )
    return (close and weak) or blocks_corner


def _make_4p_corner_proposal(world, candidates, stage):
    if not candidates:
        return None
    candidates.sort(key=lambda p: (-static_corner_value(world, p), world.cluster_distance(p), int(p.ships)))
    for target in candidates[:6]:
        sources = _corner_candidate_sources(world, target)
        if not sources:
            continue
        mission_type = "LOCAL_PRODUCTION_CAPTURE" if target.owner == -1 else "SYNC_ATTACK"
        prop = build_capture_plan(
            world,
            target,
            mission_type,
            sources,
            max_sources=4,
            eta_spread_limit=4.0 if target.owner == -1 else 6.0,
        )
        if prop is None:
            if not world.can_hold_after_capture(target, 20, max(MIN_SEND_SHIPS, int(target.ships) + 1)):
                world.add_debug(f"STATIC_CORNER_CAPTURE_REJECT_NOT_HOLDABLE p{target.id}")
            continue
        value = static_corner_value(world, target)
        prop.priority = 132.0 + value * 0.12
        prop.reason = (
            f"4p_corner stage={stage} zone={classify_corner_zone(world, target)} "
            f"role={radius_class(target)} static={is_static_planet(target)} value={value:.1f}"
        )
        if len(prop.planned_sources) > 1:
            world.add_debug(f"STATIC_CORNER_GROUPED_FUNDING_USED p{target.id} sources={prop.sources}")
        if stage in ("my_corner_static", "my_corner_bridge"):
            world.add_debug(f"MY_CORNER_LAUNCHPAD_SELECTED p{target.id} stage={stage}")
        elif stage == "rotating_bridge":
            world.add_debug(f"ROTATING_BRIDGE_SELECTED p{target.id}")
        elif stage == "adjacent_bridge":
            world.add_debug(f"MEDIUM_BRIDGE_TO_ADJACENT_CORNER p{target.id}")
        elif stage == "adjacent_static":
            world.add_debug(f"ADJACENT_CORNER_LAUNCHPAD_SELECTED p{target.id}")
        return prop
    return None


def find_4p_corner_expansion_target(world):
    if not world.is_four_player or world.step > FOUR_P_CORNER_STRATEGY_STEP_MAX:
        return None
    if not world.my_planets:
        return None
    world.add_debug("FOUR_PLAYER_CORNER_STRATEGY_ACTIVE")

    cw_secured = _adjacent_corner_secured(world, "clockwise_adjacent_corner")
    ccw_secured = _adjacent_corner_secured(world, "counterclockwise_adjacent_corner")
    my_secured = _my_corner_secured(world)
    rotating_bridge = _has_useful_rotating_bridge(world)

    if cw_secured and ccw_secured:
        world.add_debug("BOTH_ADJACENT_CORNERS_SECURED")
        world.add_debug("FINAL_CORNER_CONTEST_MODE")

    stage = "my_corner_static"
    candidates = []
    if not my_secured:
        world.add_debug("MY_CORNER_STATIC_TARGET_SCAN")
        candidates = [
            p for p in world.neutral_planets + world.enemy_planets
            if _corner_target_allowed(world, p)
            and classify_corner_zone(world, p) == "my_start_corner"
            and is_static_planet(p)
            and radius_class(p) in ("MEDIUM", "LARGE")
        ]
        if not candidates:
            stage = "my_corner_bridge"
            candidates = [
                p for p in world.neutral_planets
                if _corner_target_allowed(world, p)
                and classify_corner_zone(world, p) == "my_start_corner"
                and radius_class(p) in ("MEDIUM", "LARGE")
            ]
    elif not rotating_bridge:
        stage = "rotating_bridge"
        candidates = [
            p for p in world.neutral_planets
            if _corner_target_allowed(world, p)
            and not is_static_planet(p)
            and radius_class(p) in ("MEDIUM", "LARGE")
            and classify_corner_zone(world, p) in ("my_start_corner", "clockwise_adjacent_corner", "counterclockwise_adjacent_corner")
            and world.cluster_distance(p) <= FOUR_P_ADJACENT_BRIDGE_DIST
        ]
    elif not (cw_secured and ccw_secured):
        stage = "adjacent_static"
        world.add_debug("ADJACENT_CORNER_STATIC_SCAN")
        wanted_zones = []
        if not cw_secured:
            wanted_zones.append("clockwise_adjacent_corner")
        if not ccw_secured:
            wanted_zones.append("counterclockwise_adjacent_corner")
        candidates = [
            p for p in world.neutral_planets + world.enemy_planets
            if _corner_target_allowed(world, p)
            and classify_corner_zone(world, p) in wanted_zones
            and is_static_planet(p)
            and radius_class(p) in ("MEDIUM", "LARGE")
        ]
        if not candidates:
            stage = "adjacent_bridge"
            candidates = [
                p for p in world.neutral_planets
                if _corner_target_allowed(world, p)
                and classify_corner_zone(world, p) in wanted_zones
                and radius_class(p) == "MEDIUM"
                and world.cluster_distance(p) <= FOUR_P_ADJACENT_BRIDGE_DIST + 8
            ]
    else:
        stage = "final_corner"
        candidates = [
            p for p in world.neutral_planets + world.enemy_planets
            if _corner_target_allowed(world, p)
            and classify_corner_zone(world, p) == "opposite_corner"
            and radius_class(p) in ("MEDIUM", "LARGE")
        ]

    prop = _make_4p_corner_proposal(world, candidates, stage)
    if prop is not None:
        return prop

    if stage == "my_corner_static":
        fallback = [
            p for p in world.neutral_planets
            if _corner_target_allowed(world, p)
            and classify_corner_zone(world, p) == "my_start_corner"
            and radius_class(p) in ("MEDIUM", "LARGE")
        ]
        return _make_4p_corner_proposal(world, fallback, "my_corner_bridge")

    if stage == "adjacent_static":
        wanted_zones = [
            zone for zone, secured in (
                ("clockwise_adjacent_corner", cw_secured),
                ("counterclockwise_adjacent_corner", ccw_secured),
            )
            if not secured
        ]
        fallback = [
            p for p in world.neutral_planets
            if _corner_target_allowed(world, p)
            and classify_corner_zone(world, p) in wanted_zones
            and radius_class(p) == "MEDIUM"
            and world.cluster_distance(p) <= FOUR_P_ADJACENT_BRIDGE_DIST + 8
        ]
        return _make_4p_corner_proposal(world, fallback, "adjacent_bridge")

    return None


def small_bridge_score(world, p):
    """
    Calculate how valuable a small-radius planet is as a bridge, connector,
    stepping-stone, or territory node.

    Returns a float ≥ 0.  A score above SMALL_BRIDGE_THRESHOLD makes the
    planet worth capturing even though its radius is below SMALL_RADIUS.

    Factors rewarded:
      - shortens route from my cluster to a static/large launchpad,
      - connects two owned/target launchpads,
      - bridges my corner to an adjacent corner (4-player),
      - creates a safe route toward enemy territory,
      - close and cheap to capture,
      - can be held safely.

    Emits SMALL_BRIDGE_VALUE_FOUND when score > 0.
    """
    if radius_class(p) != "SMALL":
        return 0.0
    if p.owner == world.player or world.is_comet(p):
        return 0.0

    score = 0.0

    cluster_d = world.cluster_distance(p)
    src = min(world.my_planets, key=lambda m: dp(m, p), default=None)
    if src is None:
        return 0.0
    pool = sum(world.surplus(m) for m in world.my_planets)
    need = world.ships_needed_to_capture(src, p, pool) if pool > 0 else 999
    eta  = world.eta(src, p, max(1, need)) if need > 0 else 999.0

    # ── Cheap & close: any small planet near my cluster is a territory bonus ──
    if cluster_d <= 22.0 and need <= 15:
        score += 30.0

    # ── Shortens route to a medium/large/static planet ───────────────────────
    cx = sum(m.x for m in world.my_planets) / max(1, len(world.my_planets))
    cy = sum(m.y for m in world.my_planets) / max(1, len(world.my_planets))
    for target in world.normal_planets:
        if target.id == p.id or target.owner == world.player or world.is_comet(target):
            continue
        if radius_class(target) not in ("MEDIUM", "LARGE") and not is_static_planet(target):
            continue
        direct = dist(cx, cy, target.x, target.y)
        via    = dist(cx, cy, p.x, p.y) + dp(p, target)
        if via < direct * (1.0 + BRIDGE_MIN_SHORTCUT) and dp(p, target) <= BRIDGE_RELAY_DIST:
            shortcut_gain = direct - via + dp(p, target) * 0.5
            score += max(10.0, shortcut_gain * 1.5)
            world.add_debug(f"SMALL_BRIDGE_VALUE_FOUND p{p.id} -> target=p{target.id} gain={shortcut_gain:.1f}")

    # ── Connects two owned launchpads ─────────────────────────────────────────
    pads = owned_launchpads(world)
    if len(pads) >= 2:
        for i, pad_a in enumerate(pads):
            for pad_b in pads[i + 1:]:
                via_p = dp(pad_a, p) + dp(p, pad_b)
                direct_ab = dp(pad_a, pad_b)
                if via_p < direct_ab * 1.25:
                    score += 20.0

    # ── 4-player: bridge between corners ─────────────────────────────────────
    if world.is_four_player:
        zone = classify_corner_zone(world, p)
        if zone in ("clockwise_adjacent_corner", "counterclockwise_adjacent_corner"):
            score += 18.0
            if not _adjacent_corner_secured(world, zone):
                score += 22.0  # stepping stone toward unsecured adjacent corner

    # ── Safe route toward enemy territory ────────────────────────────────────
    if world.enemy_planets:
        nearest_enemy_d = min(dp(p, e) for e in world.enemy_planets)
        # Sits between my cluster and the enemy
        toward_enemy = dist(cx, cy, p.x, p.y) < min(dist(cx, cy, e.x, e.y) for e in world.enemy_planets)
        if toward_enemy and nearest_enemy_d < 35.0:
            score += 15.0

    # ── Holdability ──────────────────────────────────────────────────────────
    can_hold = world.can_hold_after_capture(p, eta, need) if 0 < need <= pool else False
    if not can_hold:
        score -= 20.0

    return max(0.0, score)


def find_best_launchpad_target(world):
    """Single-result wrapper: return the best capturable launchpad for the current start type."""
    st = get_start_type(world) if world.my_planets else "MEDIUM"
    return find_nearest_launchpad_target(world, st)


def find_2p_pressure_route_target(world):
    """
    2-player game launchpad-then-pressure strategy.

    Priority ladder:
      1. Unowned static LARGE launchpads in my half of the map.
      2. Unowned LARGE or static MEDIUM planets anywhere reachable.
      3. Weak nearby enemy planets once a launchpad is secured.
      4. Medium bridge neutrals that shorten route to enemy territory.

    Returns a MissionProposal or None.
    """
    if not world.my_planets:
        return None
    world.add_debug("TWO_PLAYER_LAUNCHPAD_PRESSURE_ACTIVE")

    pool = sum(world.surplus(p) for p in world.my_planets
               if world.real_incoming_threat(p)["deficit"] <= 0)
    if pool < MIN_SEND_SHIPS:
        return None

    large_owned = any(radius_class(p) == "LARGE" for p in world.my_planets)
    launchpad_secured = bool(owned_launchpads(world))

    candidates = []
    for tgt in world.normal_planets:
        if tgt.owner == world.player or world.is_comet(tgt):
            continue
        rc = radius_class(tgt)
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        need = world.ships_needed_to_capture(src, tgt, pool)
        if need <= 0 or pool < need:
            continue
        cluster_d = world.cluster_distance(tgt)
        if cluster_d > CAPTURE_OPP_MAX_DIST:
            continue
        eta = world.eta(src, tgt, need)
        if eta > CAPTURE_OPP_MAX_ETA:
            continue
        can_hold = world.can_hold_after_capture(tgt, eta, need)

        # Score using launchpad_role_score as the base
        start_type = get_start_type(world)
        score = launchpad_role_score(world, tgt, start_type)

        is_enemy = tgt.owner not in (-1, world.player)

        # 2P-specific adjustments
        if not launchpad_secured:
            # Before we have a launchpad, heavily discount enemy attacks
            if is_enemy:
                score -= 80.0
        else:
            # After launchpad is secured, add pressure bonus for weak enemies
            if is_enemy and int(tgt.ships) <= ENEMY_GATE_WEAK_LOCAL:
                score += 55.0
                world.add_debug(f"TWO_PLAYER_PRESSURE_ROUTE_SELECTED p{tgt.id} ships={int(tgt.ships)}")
            # Medium bridge toward enemy territory
            if rc == "MEDIUM" and not is_enemy and world.enemy_planets:
                nearest_enemy_d = min(dp(tgt, e) for e in world.enemy_planets)
                if nearest_enemy_d < 30.0 and cluster_d < 38.0:
                    score += 30.0

        if not can_hold and cluster_d > MIDGAME_FRONT_RADIUS:
            score -= 60.0

        score -= eta * 4.0 - need * 1.0

        if score < CAPTURE_OPP_MIN_SCORE:
            continue
        candidates.append((score, tgt, src, need, eta))

    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])

    for score, target, primary_src, need, eta in candidates[:5]:
        if world.is_comet(target):
            continue
        mtype = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        srcs = sorted(
            [p for p in world.my_planets
             if world.surplus(p) >= MIN_SEND_SHIPS
             and world.real_incoming_threat(p)["deficit"] <= 0
             and p.id not in world.backyard_locked_sources
             and dp(p, target) <= CAPTURE_OPP_MAX_DIST + 8],
            key=lambda p: (dp(p, target), -world.surplus(p)),
        )[:MAX_GROUP_SOURCES]
        if not srcs:
            continue
        prop = build_capture_plan(
            world, target, mtype, srcs,
            max_sources=min(4, len(srcs)),
            eta_spread_limit=3.0 if target.owner == -1 else 6.0,
        )
        if prop is None:
            continue
        lp_score = launchpad_role_score(world, target, get_start_type(world))
        prop.priority = 133.0 + lp_score * 0.12 + score * 0.08
        prop.reason = (
            f"2p_launchpad_pressure p{target.id} "
            f"rc={radius_class(target)} static={is_static_planet(target)} "
            f"secured={launchpad_secured} score={score:.1f}"
        )
        if is_static_planet(target) and radius_class(target) == "LARGE":
            world.add_debug(f"STATIC_HIGH_RADIUS_PRIORITY p{target.id} 2p")
            world.add_debug(f"PRIMARY_LAUNCHPAD_MARKED p{target.id} radius={target.radius:.1f}")
        elif radius_class(target) == "LARGE":
            world.add_debug(f"ROTATING_LAUNCHPAD_MARKED p{target.id}")
        elif radius_class(target) == "MEDIUM":
            world.add_debug(f"MEDIUM_BRIDGE_MARKED p{target.id}")
        if len(prop.planned_sources) > 1:
            world.add_debug(
                f"STATIC_LAUNCHPAD_GROUPED_FUNDING_USED p{target.id} "
                f"sources={len(prop.planned_sources)} ships={prop.required_ships}"
            )
        return prop
    return None


def generate_launchpad_strategy_missions(world, fleet_ratio, deadline):
    """
    Unified stable-launchpad strategy proposal generator.

    In 4-player games  → wraps find_4p_corner_expansion_target.
    In 2-player games  → wraps find_2p_pressure_route_target.

    Returns 0–2 high-priority MissionProposals for the search pool and the
    direct coordinator (runs in BOTH the pre-search and search-active paths).

    Debug markers:
        STABLE_LAUNCHPAD_STRATEGY_ACTIVE
        TWO_PLAYER_LAUNCHPAD_PRESSURE_ACTIVE  (via find_2p_pressure_route_target)
        FOUR_PLAYER_CORNER_STRATEGY_ACTIVE    (via find_4p_corner_expansion_target)
        LAUNCHPAD_CAPTURE_REJECT_NOT_HOLDABLE
    """
    proposals = []
    if not world.my_planets or fleet_ratio > FLEET_RATIO_SOFT:
        return proposals

    world.add_debug(
        f"STABLE_LAUNCHPAD_STRATEGY_ACTIVE step={world.step} "
        f"is_4p={world.is_four_player} fleet={fleet_ratio:.2f}"
    )

    if world.is_four_player:
        prop = find_4p_corner_expansion_target(world)
        if prop is not None:
            proposals.append(prop)
    else:
        prop = find_2p_pressure_route_target(world)
        if prop is not None:
            proposals.append(prop)

    # Regardless of game type, also try a radius-role launchpad capture if no
    # proposal was generated yet (covers edge cases where the specialised path
    # found nothing).
    if not proposals and time.perf_counter() < deadline:
        found = find_best_launchpad_target(world)
        if found is not None:
            score, _d, _eta, need, tgt = found
            mtype = "CAPTURE_NEUTRAL" if tgt.owner == -1 else "SYNC_ATTACK"
            srcs = sorted(
                [p for p in world.my_planets
                 if world.surplus(p) >= MIN_SEND_SHIPS
                 and world.real_incoming_threat(p)["deficit"] <= 0
                 and p.id not in world.backyard_locked_sources],
                key=lambda p: (dp(p, tgt), -world.surplus(p)),
            )[:MAX_GROUP_SOURCES]
            if srcs:
                prop = build_capture_plan(world, tgt, mtype, srcs, max_sources=4,
                                         eta_spread_limit=3.0 if tgt.owner == -1 else 6.0)
                if prop is None:
                    if not world.can_hold_after_capture(tgt, 20, max(MIN_SEND_SHIPS, int(tgt.ships) + 1)):
                        world.add_debug(f"LAUNCHPAD_CAPTURE_REJECT_NOT_HOLDABLE p{tgt.id}")
                else:
                    prop.priority = 130.0 + score * 0.10
                    prop.reason = (
                        f"launchpad_fallback p{tgt.id} rc={radius_class(tgt)} "
                        f"static={is_static_planet(tgt)} score={score:.1f}"
                    )
                    proposals.append(prop)

    return proposals


def start_aware_opening_score_adjustment(world, src, tgt, start_type):
    """Score delta applied on top of early_target_score during opening (step < 80)."""
    d = dp(src, tgt)
    prod = int(tgt.production)
    need = world.target_need_now(tgt)
    if need <= 0:
        need = int(tgt.ships) + 1
    eta = world.eta(src, tgt, max(1, need))
    role = radius_class(tgt)

    if start_type == "LARGE":
        # Build a launchpad network: large first, medium second, small as filler.
        delta = max(0.0, 30.0 - d) * 8.0 + max(0.0, 12.0 - need) * 3.5 + max(0.0, 14.0 - eta) * 4.0
        if role == "LARGE":
            delta += 180.0
        elif role == "MEDIUM":
            delta += 90.0
        else:
            delta -= 45.0
        if prod <= 1:
            delta += 20.0
        delta += prod * 10.0
        return delta

    elif start_type == "SMALL":
        # Escape to medium/large launchpad; avoid small unless it is a cheap stepping stone.
        if role == "LARGE":
            return 260.0 + prod * 18.0
        if role == "MEDIUM":
            return 150.0 + prod * 14.0
        if d <= 20.0 and need <= 8:
            return 10.0  # cheap close stepping stone is acceptable
        return -40.0     # additional penalty for prod-1 far targets

    else:  # MEDIUM
        if role == "LARGE":
            return 220.0 + (120.0 if is_static_planet(tgt) else 0.0)
        if role == "MEDIUM" and d <= 32.0:
            return 45.0
        return 0.0


def _force_nearest_opening_capture(world, moves, label="LARGE_START_NEAREST_SELECTED"):
    """Force-capture the nearest safe neutral, bypassing validate_initial_target_choice."""
    if not world.my_planets or not world.neutral_planets:
        return False
    src = min(world.my_planets, key=lambda p: -world.surplus(p), default=None)
    if src is None:
        return False
    available = int(src.ships) - world.committed.get(src.id, 0) - 1
    candidates = sorted(
        [t for t in world.neutral_planets if not world.is_comet(t)],
        key=lambda t: dp(src, t),
    )
    for tgt in candidates[:8]:
        d = dp(src, tgt)
        enemy_d = min((dp(e, tgt) for e in world.enemy_planets), default=999.0)
        if enemy_d < d - 5.0:
            continue
        need = world.target_need_now(tgt)
        send = normalize_send_amount(need)
        if need <= 0 or send > available:
            continue
        eta = world.eta(src, tgt, send)
        if eta > 55:
            continue
        if world.commit(src, tgt, send, moves, mission_type="CAPTURE_NEUTRAL"):
            world.add_debug(
                f"{label} p{tgt.id} prod={int(tgt.production)} d={d:.1f} eta={eta:.1f} need={need} send={send}"
            )
            world.wave_attempted = True
            return True
    return False


def _force_best_escape_capture(world, moves):
    """SMALL_START: force-capture nearest high-prod planet or cheapest stepping stone."""
    if not world.my_planets or not world.neutral_planets:
        return False
    src = world.my_planets[0]
    available = int(src.ships) - world.committed.get(src.id, 0) - 1
    candidates = []
    for tgt in world.neutral_planets:
        if world.is_comet(tgt):
            continue
        d = dp(src, tgt)
        enemy_d = min((dp(e, tgt) for e in world.enemy_planets), default=999.0)
        if enemy_d < d - 5.0:
            continue
        need = world.target_need_now(tgt)
        send = normalize_send_amount(need)
        if need <= 0 or send > available:
            continue
        eta = world.eta(src, tgt, need)
        if eta > 55:
            continue
        score = int(tgt.production) * 100.0 - d * 1.5 - need * 2.0
        candidates.append((score, tgt, need, eta, d))
    if not candidates:
        return False
    candidates.sort(key=lambda x: -x[0])
    _, tgt, need, eta, d = candidates[0]
    send = normalize_send_amount(need)
    if send > available:
        return False
    label = "SMALL_START_ESCAPE_TARGET" if int(tgt.production) >= 2 else "SMALL_START_STEPPING_STONE"
    if world.commit(src, tgt, send, moves, mission_type="CAPTURE_NEUTRAL"):
        world.add_debug(f"{label} p{tgt.id} prod={int(tgt.production)} d={d:.1f} eta={eta:.1f}")
        world.wave_attempted = True
        return True
    return False


def run_planet_role_opening(world, moves):
    """Radius-role opening: secure the right launchpad before generic expansion."""
    if world.step > 90 or not world.my_planets or not world.neutral_planets:
        return False
    start_type = get_start_type(world)
    large_owned = any(radius_class(p) == "LARGE" for p in world.my_planets)
    launchpad_owned = bool(owned_launchpads(world))

    if start_type == "SMALL" and not launchpad_owned:
        marker = "SMALL_START_ESCAPE_TO_LAUNCHPAD"
    elif start_type == "MEDIUM" and not large_owned:
        marker = "MEDIUM_START_FIND_LARGE_LAUNCHPAD"
    elif start_type == "LARGE":
        marker = "LARGE_START_SWEEP_LARGE_MEDIUM"
    else:
        return False

    found = find_nearest_launchpad_target(world, start_type)
    if found is None:
        return False
    score, _d, _eta, need, tgt = found
    candidate_sources = sorted(
        [
            p for p in world.my_planets
            if world.real_incoming_threat(p)["deficit"] <= 0
            and world.surplus(p) > 0
        ],
        key=lambda p: (
            0 if radius_class(p) == "LARGE" else 1 if radius_class(p) == "MEDIUM" else 4,
            dp(p, tgt),
            -world.surplus(p),
        ),
    )
    plan, reason = build_grouped_funding_plan(
        world,
        tgt,
        need,
        candidate_sources,
        "CAPTURE_NEUTRAL",
        max_sources=4,
        eta_spread_limit=3.0,
        require_hold=True,
    )
    if plan is None:
        world.add_debug(f"LAUNCHPAD_CAPTURE_SKIP p{tgt.id} reason={reason}")
        return False
    planned, total, eta_min, eta_max = plan
    mission_id = world.mission_ledger.create(
        "CAPTURE_NEUTRAL",
        tgt.id,
        [src_id for src_id, _, _, _ in planned],
        total,
        [eta for _, _, _, eta in planned],
        f"planet_role_launchpad p{tgt.id} role={radius_class(tgt)} score={score:.1f}",
    )
    sent = 0
    for src_id, ships, _angle, _eta in planned:
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        if world.commit(src, tgt, ships, moves, mission_type="CAPTURE_NEUTRAL", mission_id=mission_id, planned_sources=planned):
            sent += ships
    if sent >= total:
        world.wave_attempted = True
        world.add_debug(
            f"{marker} p{tgt.id} role={radius_class(tgt)} sent={sent} "
            f"eta={eta_min:.1f}-{eta_max:.1f}"
        )
        world.add_debug(f"LAUNCHPAD_CAPTURE_SELECTED p{tgt.id} score={score:.1f} role={radius_class(tgt)}")
        return True
    return False


def early_target_score(world, src, tgt):
    """Composite early-game score for a (src→tgt) pair. Higher is better."""
    if tgt.id in world.comet_ids:
        return -100000.0

    d = dp(src, tgt)
    need = world.target_need_now(tgt)
    if need <= 0:
        need = int(tgt.ships) + 1
    eta = world.eta(src, tgt, max(1, need))
    prod = int(tgt.production)
    role = radius_class(tgt)

    my_nearest = d
    enemy_nearest = min((dp(e, tgt) for e in world.enemy_planets), default=999.0)

    score = prod * 85.0
    if role == "LARGE":
        score += 95.0
    elif role == "MEDIUM":
        score += 45.0
    else:
        # Small planet: apply bridge-value-aware adjustment.
        bv = small_bridge_score(world, tgt)
        if bv >= SMALL_BRIDGE_THRESHOLD:
            score += bv * 0.5           # bridge value partially compensates
            world.add_debug(f"SMALL_BRIDGE_VALUE_FOUND p{tgt.id} bv={bv:.1f} in early_target_score")
        elif d <= 18.0 and need <= 10:
            score += 5.0                # very cheap close small → minor bonus
        else:
            score -= 25.0
    if prod >= 5:
        score += 70.0
    elif prod >= 4:
        score += 40.0
    elif prod >= 3:
        score += 20.0
    elif prod <= 1:
        score -= 50.0
    score += max(0.0, 35.0 - d) * 4.0

    lane_margin = enemy_nearest - my_nearest
    if lane_margin > 5.0:
        score += 30.0
    elif lane_margin > 0.0:
        score += 15.0

    my_cluster = world.cluster_distance(tgt, count=2)
    if my_cluster < enemy_nearest:
        score += 25.0

    if _connects_good_neutral(world, tgt):
        score += 55.0

    if _support_source_count(world, tgt) >= 2:
        score += 20.0

    # Blend in the launchpad score so opening choices favor prod/cost/chain value
    # rather than merely nearest reachable rocks.
    score += launchpad_target_score(world, src, tgt, StrategyMode.OPENING_TEMPO) * 0.35

    score -= need * (0.95 if prod >= 4 else 1.4)
    score -= eta * 5.0
    score -= d * 1.5

    if prod <= 1 and d > 25.0:
        score -= 80.0

    if enemy_nearest < my_nearest - 2.0:
        score -= 80.0

    if enemy_nearest < my_nearest - 10.0:
        score -= 100.0

    # Hard rule: never prefer prod-1 bridge theory when prod-3+ is capturable nearby
    if prod <= 1 and world.step < 80:
        has_better = any(
            n.id != tgt.id
            and int(n.production) >= 3
            and world.eta(src, n, max(1, int(n.ships) + 1)) <= 25.0
            and min((dp(e, n) for e in world.enemy_planets), default=999.0) >= dp(src, n) - 4.0
            for n in world.neutral_planets
        )
        if has_better:
            score -= 250.0

    # Race-status bonus/penalty
    status, _, _ = neutral_race_status(world, tgt)
    if status == "SAFE":
        score += 35.0 + max(0.0, prod - 2) * 8.0
    elif status == "CONTESTED":
        score += 10.0 if prod >= 4 else 0.0
    elif status == "ENEMY_FAVORED":
        score -= 60.0 if prod < 4 else 20.0

    # Start-type-aware adjustment (opening only)
    if world.step < 80 and world.my_planets:
        score += start_aware_opening_score_adjustment(world, src, tgt, get_start_type(world))

    return score


def validate_initial_target_choice(world, src, tgt):
    """Return False to hard-reject an early target (before step 60)."""
    if tgt.id in world.comet_ids:
        world.add_debug(f"VALIDATE_REJECT p{tgt.id} reason=comet")
        return False
    d = dp(src, tgt)
    prod = int(tgt.production)
    enemy_nearest = min((dp(e, tgt) for e in world.enemy_planets), default=999.0)
    if enemy_nearest < d - 5.0:
        world.add_debug(f"VALIDATE_REJECT p{tgt.id} reason=enemy_side d={d:.1f} enemy_d={enemy_nearest:.1f}")
        return False
    if prod <= 1 and d > 30.0:
        better_exists = any(
            n.id != tgt.id and dp(src, n) <= d and int(n.production) >= 2
            for n in world.neutral_planets
        )
        if better_exists:
            world.add_debug(f"VALIDATE_REJECT p{tgt.id} reason=far_low_prod d={d:.1f} prod={prod}")
            return False
    if prod <= 1 and world.step < 80:
        better = any(
            n.id != tgt.id
            and int(n.production) >= 3
            and world.eta(src, n, max(1, int(n.ships) + 1)) <= 25.0
            and min((dp(e, n) for e in world.enemy_planets), default=999.0) >= dp(src, n) - 4.0
            for n in world.neutral_planets
        )
        if better:
            world.add_debug(f"VALIDATE_REJECT p{tgt.id} reason=prod1_skip_prod3_available")
            return False
    # LARGE_START: accept all nearby non-comet non-enemy-side planets (local sweep)
    if world.step < 80 and d <= 35.0 and world.my_planets:
        if get_start_type(world) == "LARGE":
            return True  # bypass remaining prod/race filters
    # Skip enemy-favored low-production neutrals early game
    if world.step < 100 and prod < 4:
        status, _, _ = neutral_race_status(world, tgt)
        if status == "ENEMY_FAVORED":
            world.add_debug(f"VALIDATE_REJECT p{tgt.id} reason=enemy_favored_low_prod prod={prod}")
            return False
    return True


def is_sunk_cost_target(world, tgt):
    """Return True if we've committed ships to tgt but should cut losses and move on."""
    if world.step >= 60:
        return False
    friendly_incoming = world.incoming_to_targets.get(tgt.id, 0)
    if friendly_incoming <= 0:
        return False
    need = world.target_need_now(tgt)
    if need <= 5:
        return False
    src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
    if src is None:
        return False
    current_score = early_target_score(world, src, tgt)
    for alt in world.neutral_planets:
        if alt.id == tgt.id:
            continue
        alt_src = min(world.my_planets, key=lambda p: dp(p, alt), default=None)
        if alt_src is None:
            continue
        if early_target_score(world, alt_src, alt) > current_score + 50.0 and int(tgt.production) <= 1:
            world.add_debug(f"SUNK_COST p{tgt.id} incoming={friendly_incoming} need={need} better=p{alt.id}")
            return True
    return False


def choose_best_opening_target(world):
    """
    Score all reachable neutral candidates with early_target_score and return
    (src, tgt, need, angle) for the best strategic opening target, or None.
    """
    if not world.neutral_planets or not world.my_planets:
        return None

    scored = []
    for src in world.my_planets:
        opening_reserve = 1 if world.step < 40 else 2
        available = int(src.ships) - world.committed.get(src.id, 0) - opening_reserve
        if available <= 0:
            continue
        for tgt in world.neutral_planets:
            need = int(tgt.ships) + 1 - world.incoming_to_targets.get(tgt.id, 0)
            if need <= 0:
                continue
            if world.step < 80:
                need += 1
            if need > available:
                continue
            eta = world.eta(src, tgt, need)
            if eta > 45:
                continue
            angle, ok = world.aim(src, tgt, need)
            if not ok:
                continue
            s = early_target_score(world, src, tgt)
            scored.append((s, dp(src, tgt), need, src, tgt, angle))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    for s, d, need, src, tgt, _ in scored[:5]:
        prod = int(tgt.production)
        enemy_nearest = min((dp(e, tgt) for e in world.enemy_planets), default=999.0)
        lane = "my" if enemy_nearest > d + 5 else ("enemy" if enemy_nearest < d - 5 else "central")
        support = _support_source_count(world, tgt)
        world.add_debug(
            f"OPENING_TARGET_EVAL p{tgt.id} score={s:.1f} dist={d:.1f} prod={prod} need={need} lane={lane} support={support}"
        )

    for s, d, need, src, tgt, angle in scored:
        if is_sunk_cost_target(world, tgt):
            world.add_debug(f"OPENING_TARGET_REJECT p{tgt.id} reason=sunk_cost")
            continue
        for s2, _, _, _, tgt2, _ in scored:
            if tgt2.id != tgt.id:
                world.add_debug(f"OPENING_TARGET_REJECT p{tgt2.id} reason=lower_score score={s2:.1f}")
                break
        world.add_debug(f"OPENING_TARGET_SELECT p{tgt.id} reason=best_early_value score={s:.1f}")
        return src, tgt, need, angle

    return None


def should_retarget_opening(world, current_target_id):
    """
    Return (True, new_tgt_id) if a massively better opening target exists.
    Gated to before step 40; never retargets when in-flight ships already cover the target.
    """
    if world.step >= 40 or not world.neutral_planets or not world.my_planets:
        return False, None
    current_tgt = world.planet_by_id.get(current_target_id)
    if current_tgt is None:
        return False, None
    # If ships already in flight are sufficient, never abandon this target.
    if world.target_need_now(current_tgt) <= 0:
        return False, None
    src = min(world.my_planets, key=lambda p: dp(p, current_tgt), default=None)
    if src is None:
        return False, None
    current_score = early_target_score(world, src, current_tgt)
    best_alt = None
    best_alt_score = current_score + 80.0   # was 40 — much harder to retarget now
    for tgt in world.neutral_planets:
        if tgt.id == current_target_id:
            continue
        alt_src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if alt_src is None:
            continue
        s = early_target_score(world, alt_src, tgt)
        if s > best_alt_score:
            best_alt_score = s
            best_alt = tgt
    if best_alt is not None:
        world.add_debug(
            f"RETARGET_OPENING old=p{current_target_id} new=p{best_alt.id} "
            f"reason=better_local_value score_diff={best_alt_score - current_score:.1f}"
        )
        return True, best_alt.id
    return False, None


def opening_chain_plan(world, deadline):
    """
    Depth-2 beam search: source -> first_neutral -> second_neutral chains.
    Only when step < 60 and <=3 planets owned and no urgent threat.
    Returns best MissionProposal (targeting first_neutral) or None.
    Uses if it beats best single-target score by >= 15%.
    """
    if world.step >= 60 or len(world.my_planets) > 3:
        return None
    if world.features.get("incoming_threat_count", 0) > 0:
        return None
    if not world.neutral_planets or not world.my_planets:
        return None

    remaining = max(1, TOTAL_STEPS - world.step)
    best_chain_score = -1e9
    best_chain = None
    best_normal_score = -1e9

    for src in world.my_planets:
        av = early_surplus(world, src)
        if av < 2:
            continue
        for first in world.neutral_planets:
            if world.is_comet(first):
                continue
            if not validate_initial_target_choice(world, src, first):
                continue
            need1 = world.target_need_now(first)
            if need1 <= 0 or need1 > av:
                continue
            eta1 = world.eta(src, first, need1)
            if eta1 > 40:
                continue
            prod1 = int(first.production)
            status1, _, _ = neutral_race_status(world, first)
            if status1 == "ENEMY_FAVORED" and prod1 < 4:
                continue
            ns = early_target_score(world, src, first)
            if ns > best_normal_score:
                best_normal_score = ns

            for second in world.neutral_planets:
                if time.perf_counter() > deadline:
                    break
                if second.id == first.id or world.is_comet(second):
                    continue
                need2 = world.target_need_now(second)
                if need2 <= 0:
                    continue
                eta2 = travel_turns(dp(first, second), max(1, need2))
                if eta2 > 20:
                    continue
                prod2 = int(second.production)
                status2, _, _ = neutral_race_status(world, second)
                if status2 == "ENEMY_FAVORED" and prod2 < 3:
                    continue
                future1 = max(1, remaining - int(math.ceil(eta1)))
                future2 = max(1, remaining - int(math.ceil(eta1 + eta2)))
                chain_score = (
                    prod1 * future1
                    + prod2 * future2 * 0.75
                    - need1 * 1.2
                    - need2 * 0.9
                    - (eta1 + eta2) * 3.5
                )
                if status1 == "SAFE":
                    chain_score += 30.0
                if status2 == "SAFE":
                    chain_score += 20.0
                if prod1 >= 3:
                    chain_score += 25.0
                if prod2 >= 3:
                    chain_score += 20.0
                if prod1 <= 1:
                    chain_score -= 50.0
                if chain_score > best_chain_score:
                    best_chain_score = chain_score
                    best_chain = (src, first, need1, second)
            if time.perf_counter() > deadline:
                break

    if best_chain is None:
        return None
    src, first, need1, second = best_chain
    threshold = best_normal_score * 1.15 if best_normal_score > 0 else best_normal_score + 20.0
    if best_chain_score < threshold:
        world.add_debug(
            f"CHAIN_PLAN_SKIP chain={best_chain_score:.1f} normal={best_normal_score:.1f}"
        )
        return None
    angle, ok = world.aim(src, first, need1)
    if not ok:
        return None
    eta = world.eta(src, first, need1)
    world.add_debug(
        f"CHAIN_PLAN_SELECT src=p{src.id} first=p{first.id}(prod={int(first.production)}) "
        f"second=p{second.id}(prod={int(second.production)}) "
        f"chain={best_chain_score:.1f} normal={best_normal_score:.1f}"
    )
    return MissionProposal(
        kind="CAPTURE_NEUTRAL",
        target_id=first.id,
        priority=PRIORITY_CHAIN_PLAN_BASE + int(first.production) * 5,
        required_ships=need1,
        planned_sources=[(src.id, need1, angle, eta)],
        eta_min=eta,
        eta_max=eta,
        reason=f"chain_plan first=p{first.id}->p{second.id} score={best_chain_score:.1f}",
    )


def first_capture_360(world, moves):
    """Steps 0-70 or <2 planets: strategic 360-degree opening capture.
    Uses early_target_score to pick the best local tempo/value/position target."""
    if world.step > 70 and len(world.my_planets) >= 2:
        return False
    if not world.neutral_planets:
        return False

    # Anti-stall: force a capture before normal scoring if start type demands it
    if world.my_planets:
        st = get_start_type(world)
        if st == "LARGE" and world.step >= LARGE_START_STALL_STEP_1 and len(world.my_planets) == 1:
            world.add_debug(f"OPENING_FORCE_LOCAL_SWEEP step={world.step} planets=1")
            if _force_nearest_opening_capture(world, moves):
                return True
        elif st == "SMALL" and world.step >= SMALL_START_STALL_STEP and len(world.my_planets) == 1:
            world.add_debug(f"OPENING_FORCE_BIG_PLANET_ESCAPE step={world.step} planets=1")
            if _force_best_escape_capture(world, moves):
                return True

    result = choose_best_opening_target(world)
    if result is None:
        return False

    src, tgt, need, angle = result
    send = normalize_send_amount(need)
    angle, ok = world.aim(src, tgt, send)
    if not ok:
        return False
    eta_val = world.eta(src, tgt, send)
    planned = [(src.id, send, angle, eta_val)]
    mission_id = world.mission_ledger.create(
        "CAPTURE_NEUTRAL",
        tgt.id,
        [src.id],
        send,
        [eta_val],
        f"first_capture_360 p{tgt.id} need={need}",
    )
    if not world.commit(src, tgt, send, moves, mission_type="CAPTURE_NEUTRAL", mission_id=mission_id, planned_sources=planned):
        return False
    world.wave_attempted = True
    d = dp(src, tgt)
    world.add_debug(
        f"FIRST_CAPTURE_360 src={src.id} tgt={tgt.id} d={d:.1f} eta={eta_val:.1f} need={need} prod={tgt.production}"
    )
    return True


def early_nearest_expansion_360(world, moves):
    """Steps 0-140 or <4 planets: strategic early expansion.
    Candidates are ranked by early_target_score so the bot picks lane/value/support
    over raw proximity. Single-source if possible; grouped from 2-3 nearby sources."""
    if world.step >= 140 and len(world.my_planets) >= 4:
        return False
    if not world.neutral_planets:
        return False

    # General early-prod force: step >= 30 with <3 planets (any start type)
    if world.my_planets and world.step >= 30 and len(world.my_planets) < 3:
        world.add_debug(f"EARLY_PROD_FORCE_CAPTURE step={world.step} planets={len(world.my_planets)}")
        if _force_nearest_opening_capture(world, moves, label="EARLY_PROD_FORCE_CAPTURE"):
            return True

    # LARGE_START anti-stall: if stuck with <3 planets past step 35, force local sweep
    if world.my_planets:
        st = get_start_type(world)
        if st == "LARGE" and world.step >= LARGE_START_STALL_STEP_3 and len(world.my_planets) < 3:
            world.add_debug(f"OPENING_FORCE_LOCAL_SWEEP step={world.step} planets={len(world.my_planets)}")
            if _force_nearest_opening_capture(world, moves, label="LARGE_START_LOCAL_SWEEP"):
                return True

    candidates = []
    for src in world.my_planets:
        av = early_surplus(world, src)
        if av < 5:
            continue
        for tgt in world.neutral_planets:
            if not validate_initial_target_choice(world, src, tgt):
                continue
            need = world.target_need_now(tgt)
            if need <= 0:
                continue
            d = dp(src, tgt)
            eta = world.eta(src, tgt, max(1, need))
            if eta > 60:
                continue
            angle, ok = world.aim(src, tgt, max(1, min(need, av)))
            if not ok:
                continue
            score = early_target_score(world, src, tgt)
            candidates.append((score, eta, d, need, src, tgt, angle))

    if not candidates:
        return False

    # Best strategic score first; ETA and distance break ties
    candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    seen: set = set()
    for score, eta_val, d, need, src, tgt, _ in candidates:
        if tgt.id in seen:
            continue
        seen.add(tgt.id)
        if is_sunk_cost_target(world, tgt):
            world.add_debug(f"EARLY_EXPANSION_360_SKIP p{tgt.id} reason=sunk_cost")
            continue
        av = early_surplus(world, src)
        send = normalize_send_amount(need)
        if av >= send:
            if world.commit(src, tgt, send, moves, mission_type="CAPTURE_NEUTRAL"):
                world.add_debug(
                    f"EARLY_EXPANSION_360 src={src.id} tgt={tgt.id} score={score:.1f} d={d:.1f} eta={eta_val:.1f} need={need} send={send}"
                )
                return True
        else:
            pool_srcs = sorted(
                [p for p in world.my_planets if world.real_incoming_threat(p)["deficit"] <= 0],
                key=lambda p: dp(p, tgt),
            )[:3]
            if sum(early_surplus(world, p) for p in pool_srcs) >= send:
                plan, reason = build_grouped_funding_plan(
                    world,
                    tgt,
                    need,
                    pool_srcs,
                    "CAPTURE_NEUTRAL",
                    max_sources=3,
                    eta_spread_limit=3.0,
                    require_hold=False,
                )
                if plan is None:
                    world.add_debug(f"EARLY_EXPANSION_360_GROUPED_SKIP tgt={tgt.id} reason={reason}")
                    continue
                planned, goal, _eta_min, _eta_max = plan
                mission_id = world.mission_ledger.create(
                    "CAPTURE_NEUTRAL",
                    tgt.id,
                    [src_id for src_id, _, _, _ in planned],
                    goal,
                    [eta for _, _, _, eta in planned],
                    f"early_expansion_360 p{tgt.id} need={need}",
                )
                sent = 0
                for src_id, sn, _, _ in planned:
                    psrc = world.planet_by_id.get(src_id)
                    if psrc is not None and world.commit(psrc, tgt, sn, moves, mission_type="CAPTURE_NEUTRAL", mission_id=mission_id, planned_sources=planned):
                        sent += sn
                if sent >= goal:
                    world.wave_attempted = True
                    world.add_debug(
                        f"EARLY_EXPANSION_360_GROUPED tgt={tgt.id} score={score:.1f} d={d:.1f} need={need} sent={sent}"
                    )
                    return True
    return False


def forced_opening_capture(world, moves):
    """Stuck detector: force-capture nearest neutral when stuck on 1 planet past OPENING_STUCK_STEP."""
    if len(world.my_planets) != 1 or world.step < OPENING_STUCK_STEP:
        return False
    if not world.neutral_planets:
        return False
    world.add_debug(f"OPENING_STUCK step={world.step} trying force-capture")
    src = world.my_planets[0]
    candidates = []
    for tgt in world.neutral_planets:
        if world.is_comet(tgt):
            continue
        d = dp(src, tgt)
        enemy_nearest = min((dp(e, tgt) for e in world.enemy_planets), default=999.0)
        if enemy_nearest < d - 5:
            continue
        need = world.target_need_now(tgt)
        if need <= 0:
            continue
        available = max(0, int(src.ships) - world.committed.get(src.id, 0) - 1)
        if available < need:
            continue
        eta = world.eta(src, tgt, need)
        if eta > 55:
            continue
        angle, ok = world.aim(src, tgt, need)
        if not ok:
            continue
        score = early_target_score(world, src, tgt)
        candidates.append((score, d, need, tgt, angle, eta))
    if not candidates:
        world.add_debug(f"OPENING_STUCK step={world.step} no candidate found")
        return False
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    score, d, need, tgt, angle, eta = candidates[0]
    send = normalize_send_amount(need)
    angle, ok = world.aim(src, tgt, send)
    if not ok:
        return False
    eta = world.eta(src, tgt, send)
    planned = [(src.id, send, angle, eta)]
    mission_id = world.mission_ledger.create(
        "CAPTURE_NEUTRAL", tgt.id, [src.id], send, [eta],
        f"stuck_opening p{tgt.id} need={need}",
    )
    if world.commit(src, tgt, send, moves, mission_type="CAPTURE_NEUTRAL",
                    mission_id=mission_id, planned_sources=planned):
        world.wave_attempted = True
        world.add_debug(
            f"OPENING_STUCK_FORCED tgt=p{tgt.id} need={need} d={d:.1f} eta={eta:.1f} score={score:.1f}"
        )
        return True
    return False


def _nearest_capturable_neutral(world):
    """Return (tgt, src, d, eta, need, pool, enemy_eta) for the closest capturable neutral, or None."""
    best_c = None
    best_key = None
    for tgt in world.neutral_planets:
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        need = world.target_need_now(tgt)
        if need <= 0:
            continue
        pool = sum(world.surplus(p) for p in world.my_planets)
        if pool < need:
            continue
        d = dp(src, tgt)
        eta = world.eta(src, tgt, max(1, need))
        _, enemy_eta = world.reaction_times(tgt)
        key = (eta, d, need)
        if best_key is None or key < best_key:
            best_key = key
            best_c = (tgt, src, d, eta, need, pool, enemy_eta)
    return best_c


def force_repeated_missed_neutral(world, moves):
    """Before beam search: if the same close neutral has been skipped 3+ turns, force-capture it."""
    global _missed_skipped
    if not world.neutral_planets or not world.my_planets:
        return False
    best_c = _nearest_capturable_neutral(world)
    if best_c is None:
        return False
    tgt, _, _, _, need, _, _ = best_c
    if world.is_comet(tgt):
        return False
    count = _missed_skipped.get(tgt.id, 0)
    if count < 3 or world.step >= 200:
        return False
    if world.cluster_distance(tgt) > FINISH_ZERO_NEAR_DIST + 10:
        return False
    pool_srcs = sorted(
        [
            p for p in world.my_planets
            if world.real_incoming_threat(p)["deficit"] <= 0
            and world.surplus(p) > 0
        ],
        key=lambda p: dp(p, tgt),
    )[:3]
    if sum(world.surplus(p) for p in pool_srcs) < need:
        return False
    plan, reason = build_grouped_funding_plan(
        world,
        tgt,
        need,
        pool_srcs,
        "CAPTURE_NEUTRAL",
        max_sources=3,
        eta_spread_limit=3.0,
    )
    if plan is None:
        world.add_debug(f"MISSED_NEUTRAL_FORCE_BLOCK target=p{tgt.id} reason={reason}")
        return False
    planned, goal, _eta_min, _eta_max = plan
    mission_id = world.mission_ledger.create(
        "CAPTURE_NEUTRAL",
        tgt.id,
        [src_id for src_id, _, _, _ in planned],
        goal,
        [eta for _, _, _, eta in planned],
        f"missed-neutral forced p{tgt.id} skipped={count}",
    )
    sent = 0
    for src_id, sn, _, _ in planned:
        if sent >= goal:
            break
        psrc = world.planet_by_id.get(src_id)
        if psrc is None:
            continue
        if world.commit(psrc, tgt, sn, moves, mission_type="CAPTURE_NEUTRAL", mission_id=mission_id, planned_sources=planned):
            sent += sn
    if sent >= goal:
        _missed_skipped[tgt.id] = 0
        world.add_debug(f"MISSED_NEUTRAL_FORCED target=p{tgt.id} skipped={count} sent={sent}")
        return True
    return False


def missed_opportunity_detector(world, chosen_moves):
    """Debug tool: logs nearby neutrals being skipped each turn. Updates _missed_skipped."""
    global _missed_skipped
    if not world.neutral_planets or not world.my_planets:
        return

    chosen_tgt_ids: set = set()
    for move in chosen_moves:
        src_id, angle, ships = move[0], move[1], move[2]
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        best_match, best_diff = None, 0.3
        for tgt in world.neutral_planets + world.enemy_planets:
            aim_angle, _ = world.aim(src, tgt, max(1, int(ships)))
            diff = abs(math.atan2(math.sin(angle - aim_angle), math.cos(angle - aim_angle)))
            if diff < best_diff:
                best_diff = diff
                best_match = tgt
        if best_match:
            chosen_tgt_ids.add(best_match.id)

    best_c = _nearest_capturable_neutral(world)
    if best_c is None:
        _missed_skipped.clear()
        return

    tgt, src, d, eta, need, pool, enemy_eta = best_c
    if tgt.id not in chosen_tgt_ids:
        _missed_skipped[tgt.id] = _missed_skipped.get(tgt.id, 0) + 1
        count = _missed_skipped[tgt.id]
        world.add_debug(
            f"MISSED_OPP step={world.step} nearest=p{tgt.id} src=p{src.id} "
            f"d={d:.1f} eta={eta:.1f} need={need} pool={pool} enemy_eta={enemy_eta:.1f} "
            f"skipped={count}" + (" FORCE_NEXT" if count >= 3 else "")
        )
    else:
        _missed_skipped.pop(tgt.id, None)


def local_threatened_planets(world):
    threatened = []
    for p in world.my_planets:
        if (
            world.real_incoming_threat(p)["deficit"] > 0
            or world.simulate_planet_timeline(p, DEFENSE_ETA_HORIZON)["fall_turn"] is not None
        ):
            threatened.append(p)
    return threatened


def lock_sources_near_local_threats(world, threatened):
    return


def nearest_high_value_neutral_for_source(world, src):
    """Return a nearby high-production neutral that this source should consider before far attacks."""
    candidates = [
        n for n in world.neutral_planets
        if not world.is_comet(n)
        and int(n.production) >= LOCAL_PRODUCTION_MIN_PROD
        and int(n.ships) <= LOCAL_PRODUCTION_MAX_SHIPS
        and dp(src, n) <= LOCAL_PRODUCTION_MAX_DIST
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda n: (dp(src, n), -int(n.production), int(n.ships)))


def _high_value_neutral_skip(world, tgt, src, reason, eta=None, need=None, enemy_eta=None):
    src_label = f"p{src.id}" if src is not None else "none"
    src_ships = int(src.ships) if src is not None else 0
    d = dp(src, tgt) if src is not None else 999.0
    eta_text = f"{eta:.1f}" if eta is not None else "?"
    need_text = str(need) if need is not None else "?"
    enemy_text = f"{enemy_eta:.1f}" if enemy_eta is not None else "?"
    world.add_debug(
        f"SKIP HIGH_VALUE_NEUTRAL p{tgt.id} step={world.step} ships={int(tgt.ships)} prod={int(tgt.production)} "
        f"src={src_label} src_ships={src_ships} d={d:.1f} eta={eta_text} need={need_text} "
        f"enemy_eta={enemy_text} reason={reason}"
    )


def generate_high_value_neutral_missions(world, deadline):
    """Midgame production-first layer: capture nearby 4-5 production neutrals before offense."""
    proposals = []
    if not world.neutral_planets or not world.my_planets or world.step < 45:
        return proposals

    valuable = [
        n for n in world.neutral_planets
        if not world.is_comet(n)
        and int(n.production) >= LOCAL_PRODUCTION_MIN_PROD
        and int(n.ships) <= LOCAL_PRODUCTION_MAX_SHIPS
    ]
    valuable.sort(key=lambda n: (-int(n.production), world.cluster_distance(n), int(n.ships)))

    for tgt in valuable:
        if time.perf_counter() > deadline:
            break
        nearest_src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if nearest_src is None:
            _high_value_neutral_skip(world, tgt, None, "no source")
            continue
        nearest_dist = dp(nearest_src, tgt)
        if nearest_dist > LOCAL_PRODUCTION_MAX_DIST and world.cluster_distance(tgt) > MIDGAME_CONTEST_MAX_DIST:
            _high_value_neutral_skip(world, tgt, nearest_src, "too far")
            continue

        rough_need = max(1, int(tgt.ships) + 1)
        my_eta, enemy_eta = world.reaction_times(tgt)
        enemy_inbound = world.enemy_incoming_to_targets.get(tgt.id, 0) > 0
        race = enemy_inbound or enemy_eta <= my_eta + LOCAL_PRODUCTION_RACE_MARGIN

        mission_type = "HIGH_VALUE_NEUTRAL_RACE" if race else "LOCAL_PRODUCTION_CAPTURE"
        base_priority = PRIORITY_HV_RACE_BASE if mission_type == "HIGH_VALUE_NEUTRAL_RACE" else PRIORITY_HV_CAPTURE_BASE
        if int(tgt.production) >= LOCAL_PRODUCTION_PREMIER_PROD:
            base_priority += PRIORITY_HV_PREMIER_STEP if world.step > 50 else PRIORITY_HV_PREMIER_EARLY

        safe_sources = []
        for src in sorted(world.my_planets, key=lambda p: (dp(p, tgt), 0 if int(p.ships) >= LOCAL_PRODUCTION_HUB_SHIPS else 1)):
            av = world.surplus(src)
            if av <= 0:
                continue
            test_send = min(av, max(rough_need + 2, 8))
            eta = world.eta(src, tgt, test_send)
            if eta > LOCAL_PRODUCTION_MAX_ETA and not race:
                continue
            ok, reason = world.source_is_safe_for(src, tgt, mission_type, test_send)
            if not ok:
                _high_value_neutral_skip(world, tgt, src, reason, eta=eta, need=rough_need, enemy_eta=enemy_eta)
                continue
            safe_sources.append(src)

        if not safe_sources:
            _high_value_neutral_skip(world, tgt, nearest_src, "no safe source", eta=my_eta, need=rough_need, enemy_eta=enemy_eta)
            continue

        primary = safe_sources[0]
        single_need = world.ships_needed_to_capture(primary, tgt, world.surplus(primary))
        single_need = max(single_need, int(tgt.ships) + 2)
        if race:
            eta_guess = world.eta(primary, tgt, single_need)
            eval_turn = max(1, int(math.ceil(eta_guess)))
            projected_need = world.min_ships_to_own_by(
                tgt.id,
                eval_turn,
                world.player,
                arrival_turn=eval_turn,
                upper_bound=world.surplus(primary),
            )
            single_need = max(single_need, projected_need)
        if single_need <= world.surplus(primary):
            eta = world.eta(primary, tgt, single_need)
            if eta > LOCAL_PRODUCTION_MAX_ETA and not race:
                _high_value_neutral_skip(world, tgt, primary, "single eta too late", eta=eta, need=single_need, enemy_eta=enemy_eta)
                continue
            if race and enemy_inbound and eta > enemy_eta + LOCAL_PRODUCTION_RACE_MARGIN + 2.0:
                _high_value_neutral_skip(world, tgt, primary, "too late after enemy impact", eta=eta, need=single_need, enemy_eta=enemy_eta)
                continue
            if race and eta > enemy_eta + LOCAL_PRODUCTION_RACE_MARGIN and not enemy_inbound:
                _high_value_neutral_skip(world, tgt, primary, "cannot beat enemy race", eta=eta, need=single_need, enemy_eta=enemy_eta)
                continue
            if not world.can_hold_after_capture(tgt, eta, single_need):
                _high_value_neutral_skip(world, tgt, primary, "cannot hold after capture", eta=eta, need=single_need, enemy_eta=enemy_eta)
                continue
            angle, ok = world.aim(primary, tgt, single_need)
            if not ok:
                _high_value_neutral_skip(world, tgt, primary, "aim invalid", eta=eta, need=single_need, enemy_eta=enemy_eta)
                continue
            planned = [(primary.id, single_need, angle, eta)]
            ok_grp, grp_reason = validate_grouped_launch(world, tgt, planned)
            if not ok_grp:
                _high_value_neutral_skip(world, tgt, primary, f"validate_grouped_launch: {grp_reason}", eta=eta, need=single_need, enemy_eta=enemy_eta)
                continue
            proposals.append(MissionProposal(
                kind=mission_type,
                target_id=tgt.id,
                priority=base_priority + max(0.0, LOCAL_PRODUCTION_MAX_DIST - nearest_dist) * 0.35,
                required_ships=single_need,
                planned_sources=planned,
                eta_min=eta,
                eta_max=eta,
                reason=(
                    f"{mission_type.lower()} p{tgt.id} prod={int(tgt.production)} ships={int(tgt.ships)} "
                    f"src=p{primary.id} eta={eta:.1f} enemy_eta={enemy_eta:.1f}"
                ),
            ))
            world.add_debug(
                f"{mission_type}_SELECT target=p{tgt.id} src=p{primary.id} ships={single_need} eta={eta:.1f} enemy_eta={enemy_eta:.1f}"
            )
            continue

        selected = []
        sent = 0
        group_pool = sum(world.surplus(s) for s in safe_sources[:MAX_GROUP_SOURCES])
        group_need = world.ships_needed_to_capture(primary, tgt, group_pool)
        group_need = max(group_need, int(tgt.ships) + 2)
        if race:
            eta_guess = min(world.eta(s, tgt, max(1, min(world.surplus(s), group_need))) for s in safe_sources[:MAX_GROUP_SOURCES])
            eval_turn = max(1, int(math.ceil(eta_guess)))
            projected_need = world.min_ships_to_own_by(
                tgt.id,
                eval_turn,
                world.player,
                arrival_turn=eval_turn,
                upper_bound=group_pool,
            )
            group_need = max(group_need, projected_need)
        for src in safe_sources[:MAX_GROUP_SOURCES]:
            if sent >= group_need:
                break
            sn = min(world.surplus(src), group_need - sent)
            if sn <= 0:
                continue
            angle, ok = world.aim(src, tgt, sn)
            if not ok:
                continue
            selected.append((src.id, sn, angle, world.eta(src, tgt, sn)))
            sent += sn
        if sent < group_need:
            _high_value_neutral_skip(world, tgt, primary, "insufficient grouped safe surplus", need=group_need, enemy_eta=enemy_eta)
            continue
        eta_vals = [eta for _, _, _, eta in selected]
        if max(eta_vals) > LOCAL_PRODUCTION_MAX_ETA and not race:
            _high_value_neutral_skip(world, tgt, primary, "group eta too late", eta=max(eta_vals), need=group_need, enemy_eta=enemy_eta)
            continue
        if race and enemy_inbound and max(eta_vals) > enemy_eta + LOCAL_PRODUCTION_RACE_MARGIN + 2.0:
            _high_value_neutral_skip(world, tgt, primary, "group too late after enemy impact", eta=max(eta_vals), need=group_need, enemy_eta=enemy_eta)
            continue
        if max(eta_vals) - min(eta_vals) > ETA_SYNC_WINDOW:
            _high_value_neutral_skip(world, tgt, primary, "group eta spread", eta=max(eta_vals), need=group_need, enemy_eta=enemy_eta)
            continue
        if not world.can_hold_after_capture(tgt, max(eta_vals), sent):
            _high_value_neutral_skip(world, tgt, primary, "group cannot hold", eta=max(eta_vals), need=group_need, enemy_eta=enemy_eta)
            continue
        ok_grp, grp_reason = validate_grouped_launch(world, tgt, selected)
        if not ok_grp:
            _high_value_neutral_skip(world, tgt, primary, f"validate_grouped_launch: {grp_reason}", eta=max(eta_vals), need=group_need, enemy_eta=enemy_eta)
            continue
        proposals.append(MissionProposal(
            kind=mission_type,
            target_id=tgt.id,
            priority=base_priority + max(0.0, LOCAL_PRODUCTION_MAX_DIST - nearest_dist) * 0.30,
            required_ships=group_need,
            planned_sources=selected,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=(
                f"{mission_type.lower()} grouped p{tgt.id} prod={int(tgt.production)} ships={int(tgt.ships)} "
                f"sources={len(selected)} enemy_eta={enemy_eta:.1f}"
            ),
        ))
        world.add_debug(
            f"{mission_type}_SELECT target=p{tgt.id} grouped sources={[s for s, _, _, _ in selected]} ships={sent} enemy_eta={enemy_eta:.1f}"
        )

    return proposals[:3]


def recently_captured_hub_ids(world, ttl=LAUNCHPAD_RECENT_TTL):
    prev_owners = _prev_owners if isinstance(_prev_owners, dict) else {}
    captured = set()
    for p in world.my_planets:
        prev_owner = prev_owners.get(p.id, world.player)
        if prev_owner != world.player and (int(p.production) >= 3 or world.nearest_enemy_distance(p) <= FRONTLINE_DIST + 12):
            captured.add(p.id)
    for entry in world.mission_ledger.entries.values():
        if (
            entry.status == "completed"
            and entry.mission_type in OFFENSIVE_MISSIONS
            and world.step - entry.launch_step <= ttl
        ):
            captured.add(entry.target_id)
    return captured


def command_hub_score(world, p, recent_ids):
    score = int(p.production) * 24.0 + world.surplus(p) * 0.7
    role = radius_class(p)
    if role == "LARGE":
        score += 120.0
    elif role == "MEDIUM":
        score += 35.0
    else:
        score -= 80.0
    if is_static_planet(p) and role in ("LARGE", "MEDIUM"):
        score += 80.0
    if int(p.production) >= LAUNCHPAD_PROD_MIN:
        score += 60.0
    if world.surplus(p) >= LAUNCHPAD_SURPLUS_MIN:
        score += 35.0
    enemy_d = world.nearest_enemy_distance(p)
    if enemy_d <= FRONTLINE_DIST + 18:
        score += 45.0
    if p.id in recent_ids:
        score += 75.0
    return score


def command_hubs(world):
    recent_ids = recently_captured_hub_ids(world)
    hubs = []
    for p in world.my_planets:
        role = radius_class(p)
        if (
            role == "LARGE"
            or (role == "MEDIUM" and world.surplus(p) >= MIN_SEND_SHIPS)
            or int(p.production) >= LAUNCHPAD_PROD_MIN
            or world.surplus(p) >= LAUNCHPAD_SURPLUS_MIN
            or world.nearest_enemy_distance(p) <= FRONTLINE_DIST + 18
            or p.id in recent_ids
        ):
            hubs.append((command_hub_score(world, p, recent_ids), p))
    hubs.sort(key=lambda item: -item[0])
    return [p for _score, p in hubs]


def capture_required_total(world, target, primary, pool, mission_type):
    capture_need = world.ships_needed_to_capture(primary, target, pool)
    if capture_need <= 0:
        return 0
    enemy_incoming_buffer = sum(
        ships for eta, owner, ships in world.arrivals_by_target.get(target.id, [])
        if owner != world.player and eta <= 18
    )
    if target.owner not in (-1, world.player):
        hold_margin = max(8, int(target.production) * 5)
    elif mission_type in ("LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE"):
        hold_margin = max(3, int(target.production) * 2)
    else:
        hold_margin = 2 if is_idle(target) else 0
    return max(1, int(capture_need + hold_margin + enemy_incoming_buffer))


def build_grouped_funding_plan(
    world,
    target,
    required_total,
    candidate_sources,
    mission_type,
    *,
    max_sources=MAX_GROUP_SOURCES,
    eta_spread_limit=6.0,
    allow_small_packets=False,
    require_hold=True,
):
    world.add_debug(
        f"GROUPED_FUNDING_START target=p{target.id} mission={mission_type} required={required_total}"
    )
    if target is None or world.is_comet(target) or required_total <= 0:
        return None, "invalid target/need"

    mission_type = canonical_mission_type(mission_type)
    small_packets = allow_small_packets or mission_allows_small_packet(mission_type)
    target_total = round_up_to_granularity(required_total)
    if not small_packets:
        target_total = max(MIN_SEND_SHIPS, target_total)
    world.add_debug(
        f"GROUPED_FUNDING_TARGET_TOTAL target=p{target.id} required={required_total} total={target_total}"
    )

    seen = set()
    sources = []
    for src in candidate_sources:
        if src is None or src.id in seen:
            continue
        seen.add(src.id)
        if src.owner != world.player:
            continue
        safe_avail = max(0, min(world.surplus(src), int(src.ships) - world.committed.get(src.id, 0)))
        if safe_avail <= 0:
            continue
        role = radius_class(src)
        local_support = dp(src, target) <= CHEAP_RECAPTURE_LOCAL_DIST
        critical_support = mission_type in (
            "DEFEND_HOLD", "SAVE_UNDER_ATTACK", "RECAPTURE_LOST", "REINFORCE_CAPTURE"
        )
        if role == "LARGE":
            role_priority = 0
        elif role == "MEDIUM":
            role_priority = 1
        elif critical_support and local_support:
            role_priority = 0
        else:
            role_priority = 5
        local_bonus = 0 if dp(src, target) <= LOCAL_HUB_RADIUS else 1
        eta_guess = world.eta(src, target, max(1, min(safe_avail, target_total)))
        sources.append((role_priority, eta_guess, local_bonus, dp(src, target), -safe_avail, src, safe_avail))
    sources.sort()
    sources = sources[:max_sources]

    planned = []
    total = 0
    for _role_priority, _eta_guess, _local_bonus, _dist, _neg_avail, src, safe_avail in sources:
        if total >= target_total:
            break
        remaining = max(0, target_total - total)
        if small_packets:
            candidate_amounts = [min(safe_avail, remaining)]
            if safe_avail > remaining:
                candidate_amounts.append(safe_avail)
        else:
            max_send = round_down_to_granularity(safe_avail)
            if max_send < MIN_SEND_SHIPS:
                world.add_debug(
                    f"GROUPED_FUNDING_SOURCE_SKIP_UNSAFE target=p{target.id} src=p{src.id} "
                    f"reason=below_packet_min avail={safe_avail}"
                )
                continue
            preferred = min(max_send, remaining)
            if preferred < MIN_SEND_SHIPS:
                preferred = min(max_send, MIN_SEND_SHIPS)
            preferred = round_down_to_granularity(preferred)
            if preferred < MIN_SEND_SHIPS:
                preferred = min(max_send, MIN_SEND_SHIPS)
            candidate_amounts = list(range(int(preferred), MIN_SEND_SHIPS - 1, -SEND_GRANULARITY))
            if max_send not in candidate_amounts:
                candidate_amounts.append(max_send)

        picked = None
        for send in candidate_amounts:
            send = int(send)
            if send <= 0 or send > safe_avail:
                continue
            if not small_packets:
                send = round_down_to_granularity(send)
                if send < MIN_SEND_SHIPS:
                    continue
            if not valid_packet_size(mission_type, send):
                continue
            ok, reason = world.source_is_safe_for(src, target, mission_type, send)
            if not ok:
                world.add_debug(
                    f"GROUPED_FUNDING_SOURCE_SKIP_UNSAFE target=p{target.id} src=p{src.id} "
                    f"send={send} reason={reason}"
                )
                continue
            angle, aim_ok = world.aim(src, target, send)
            if not aim_ok:
                world.add_debug(
                    f"GROUPED_FUNDING_SOURCE_SKIP_UNSAFE target=p{target.id} src=p{src.id} "
                    f"send={send} reason=aim"
                )
                continue
            picked = (src.id, send, angle, world.eta(src, target, send))
            break
        if picked is None:
            continue

        planned.append(picked)
        total += picked[1]
        if picked[1] < safe_avail:
            world.add_debug(
                f"GROUPED_FUNDING_SOURCE_PARTIAL_USED target=p{target.id} src=p{src.id} "
                f"send={picked[1]} avail={safe_avail}"
            )
        world.add_debug(
            f"GROUPED_FUNDING_SOURCE_ADD target=p{target.id} src=p{src.id} "
            f"send={picked[1]} total={total}/{target_total}"
        )

    if not planned:
        return None, "no valid planned sources"

    eta_vals = [eta for _, _, _, eta in planned]
    spread = max(eta_vals) - min(eta_vals) if len(eta_vals) >= 2 else 0.0
    if spread > eta_spread_limit:
        return None, f"eta_spread {spread:.1f}>{eta_spread_limit:.1f}"

    if total < target_total:
        ok_partial, partial_reason = validate_grouped_launch(world, target, planned)
        if not ok_partial:
            world.add_debug(
                f"PARTIAL_PACKET_REJECT_NO_CONVERSION target=p{target.id} total={total} "
                f"target_total={target_total} reason={partial_reason}"
            )
            world.add_debug(
                f"GROUPED_FUNDING_REJECT_UNDERFUNDED target=p{target.id} total={total} target_total={target_total}"
            )
            return None, f"underfunded total={total} target_total={target_total}"
    else:
        ok_full, full_reason = validate_grouped_launch(world, target, planned)
        if not ok_full:
            return None, f"validate_grouped_launch: {full_reason}"

    if require_hold and not world.can_hold_after_capture(target, max(eta_vals), total):
        return None, "cannot hold"

    planet_ships = sum(int(p.ships) for p in world.my_planets)
    fleet_ships = sum(int(f.ships) for f in world.my_fleets)
    post_ratio = (fleet_ships + total) / max(1, planet_ships + fleet_ships)
    if post_ratio > FLEET_RATIO_HARD and mission_type not in CRITICAL_MISSIONS | {"HIGH_VALUE_NEUTRAL_RACE", "FINAL_DRAIN"}:
        return None, f"fleet_ratio_after={post_ratio:.2f}"

    if total != MIN_SEND_SHIPS:
        world.add_debug(f"SEND_REQUIRED_NOT_DEFAULT_10 target=p{target.id} total={total}")
    world.add_debug(
        f"GROUPED_FUNDING_COMPLETE target=p{target.id} total={total} target_total={target_total} "
        f"sources={[src_id for src_id, _, _, _ in planned]}"
    )
    return (planned, total, min(eta_vals), max(eta_vals)), ""


def build_grouped_capture_plan(world, target, mission_type, max_sources=MIDGAME_ATTACK_SOURCE_MAX):
    def _source_role_key(s):
        role = radius_class(s)
        role_rank = 0 if role == "LARGE" else 1 if role == "MEDIUM" else 4
        return (role_rank, dp(s, target), -world.surplus(s))

    sources = sorted(
        [
            s for s in world.my_planets
            if world.surplus(s) > 0
            and world.real_incoming_threat(s)["deficit"] <= 0
        ],
        key=_source_role_key,
    )[:max_sources]
    if not sources:
        return None, "no safe sources"
    pool = sum(world.surplus(s) for s in sources)
    primary = sources[0]
    need = capture_required_total(world, target, primary, pool, mission_type)
    if need <= 0:
        return None, f"pool={pool} need={need}"
    return build_grouped_funding_plan(
        world,
        target,
        need,
        sources,
        mission_type,
        max_sources=max_sources,
        eta_spread_limit=3.0 if target.owner == -1 else 6.0,
    )


def build_capture_plan(world, target, mission_type, candidate_sources, max_sources=MAX_GROUP_SOURCES, eta_spread_limit=6.0):
    """
    Unified grouped capture planner. Returns MissionProposal (priority=0, caller sets it) or None.
    Picks sources from candidate_sources, validates ETA spread and ownership flip.
    eta_spread_limit: 3.0 for neutrals, 6.0 for enemy planets.
    """
    if not candidate_sources or world.is_comet(target):
        return None
    sources = sorted(
        candidate_sources,
        key=lambda s: (
            0 if radius_class(s) == "LARGE" else 1 if radius_class(s) == "MEDIUM" else 4,
            dp(s, target),
            -world.surplus(s),
        ),
    )[:max_sources]
    if not sources:
        return None
    pool = sum(world.surplus(s) for s in sources)
    primary = sources[0]
    need = capture_required_total(world, target, primary, pool, mission_type)
    if need <= 0:
        world.add_debug(
            f"SKIP {mission_type} p{target.id} step={world.step} "
            f"reason=insufficient_pool pool={pool} need={need}"
        )
        return None
    plan, reason = build_grouped_funding_plan(
        world,
        target,
        need,
        sources,
        mission_type,
        max_sources=max_sources,
        eta_spread_limit=eta_spread_limit,
    )
    if plan is None:
        world.add_debug(
            f"SKIP {mission_type} p{target.id} step={world.step} reason={reason}"
        )
        return None
    planned, total, eta_min, eta_max = plan
    world.add_debug(
        f"PLAN {mission_type} p{target.id} step={world.step} "
        f"srcs={[s for s, _, _, _ in planned]} ships={total} "
        f"eta_min={eta_min:.1f} eta_max={eta_max:.1f} need={need}"
    )
    return MissionProposal(
        kind=mission_type,
        target_id=target.id,
        priority=0.0,  # caller sets priority
        required_ships=total,
        planned_sources=planned,
        eta_min=eta_min,
        eta_max=eta_max,
        reason=f"build_capture_plan type={mission_type} need={need} srcs={len(planned)}",
    )


def detect_enemy_weakness(world):
    """
    Return (planet, weakness_score) pairs for enemy planets whose ship count dropped
    versus expected (they launched fleets away and are now vulnerable).
    """
    if not _prev_ships:
        return []
    weak = []
    for p in world.enemy_planets:
        if world.is_comet(p):
            continue
        prev = _prev_ships.get(p.id)
        if prev is None:
            continue
        expected = prev + int(p.production)
        drop = expected - int(p.ships)   # positive = they spent ships
        if drop < WEAKNESS_DROP_THRESHOLD:
            continue
        dist_to_nearest = min((dp(p, m) for m in world.my_planets), default=999.0)
        score = drop * 2.0 + int(p.production) * 8.0 - dist_to_nearest * 0.5
        weak.append((p, score))
    weak.sort(key=lambda x: -x[1])
    return weak


def score_occupiable_planet(world, tgt, src):
    """Score a planet for nearest-occupiable priority. Higher is better."""
    d = dp(src, tgt)
    prod = int(tgt.production)
    ships = int(tgt.ships)
    need = world.ships_needed_to_capture(src, tgt, world.surplus(src))
    if need <= 0:
        return -1e9
    cluster_d = world.cluster_distance(tgt)
    my_eta, _ = world.reaction_times(tgt)
    status, _, _ = neutral_race_status(world, tgt)
    score = (
        prod * 50.0
        + max(0.0, 30.0 - d) * 5.0
        - cluster_d * 1.0
        - need * 1.2
        - my_eta * 4.0
        - world.enemy_pressure_near(tgt, radius=25.0) * 0.3
    )
    if tgt.owner == -1:
        score += 20.0
    else:
        score += prod * 15.0 + (30.0 if ships <= 12 else 0.0)
    if status == "SAFE":
        score += 25.0
    elif status == "ENEMY_FAVORED":
        score -= 40.0
    return score


def generate_nearest_occupiable_expansion_missions(world, deadline):
    """
    NEAREST_OCCUPIABLE_AND_ROTATIONAL_EXPANSION layer.
    Scans neutral and nearby weak enemy planets; captures the best candidates
    using grouped ship waves from the nearest owned sources.

    Anti-panic-mode gate: returns nothing when fleet_ratio > FLEET_RATIO_SOFT.
    Each mission includes a hold margin so the captured planet stays alive.
    """
    proposals = []
    if not world.my_planets:
        return proposals
    fleet_ratio = compute_fleet_ratio(world)
    if fleet_ratio > FLEET_RATIO_SOFT:
        world.add_debug(
            f"SKIP NEAREST_OCCUPIABLE step={world.step} "
            f"reason=fleet_ratio_too_high ratio={fleet_ratio:.2f}"
        )
        return proposals

    targets = []
    for tgt in world.normal_planets:
        if tgt.owner == world.player or world.is_comet(tgt):
            continue
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        if world.cluster_distance(tgt) > OCCUPIABLE_MAX_DIST:
            continue
        pool = sum(world.surplus(p) for p in world.my_planets)
        need = world.ships_needed_to_capture(src, tgt, pool)
        if need <= 0:
            continue
        # Weak enemy filter: skip heavily fortified low-prod enemy planets
        if tgt.owner not in (-1, world.player) and int(tgt.ships) > 25 and int(tgt.production) < 3:
            continue
        # Central enemy-attack gate: applies to all enemy targets in this layer
        if tgt.owner not in (-1, world.player) and not should_allow_enemy_attack(
            world, tgt, "SYNC_ATTACK", "nearest_occupiable"
        ):
            continue
        eta = world.eta(src, tgt, max(1, need))
        if not world.can_hold_after_capture(tgt, eta, need + OCCUPIABLE_HOLD_MARGIN):
            continue
        score = score_occupiable_planet(world, tgt, src)
        targets.append((tgt, src, score))

    targets.sort(key=lambda x: -x[2])

    seen: set = set()
    for tgt, src, score in targets[:6]:
        if time.perf_counter() > deadline:
            break
        if tgt.id in seen:
            continue
        if world.incoming_to_targets.get(tgt.id, 0) >= world.required_ships_to_capture(tgt, src):
            world.add_debug(
                f"SKIP NEAREST_OCCUPIABLE p{tgt.id} step={world.step} reason=already_en_route"
            )
            continue
        candidate_sources = [
            p for p in world.my_planets
            if world.surplus(p) >= 4
            and world.real_incoming_threat(p)["deficit"] <= 0
            and dp(p, tgt) <= OCCUPIABLE_MAX_DIST + 8
        ]
        if not candidate_sources:
            continue
        mission_type = "CAPTURE_NEUTRAL" if tgt.owner == -1 else "SYNC_ATTACK"
        prop = build_capture_plan(
            world, tgt, mission_type, candidate_sources,
            max_sources=4,
            eta_spread_limit=3.0 if tgt.owner == -1 else 6.0,
        )
        if prop is None:
            continue
        prop.priority = (
            PRIORITY_NEAREST_OCCUPIABLE
            + score * 0.05
            + int(tgt.production) * 3.0
        )
        prop.reason = (
            f"nearest_occupiable p{tgt.id} prod={int(tgt.production)} "
            f"ships={int(tgt.ships)} score={score:.1f} d={dp(src,tgt):.1f}"
        )
        world.add_debug(
            f"SELECT NEAREST_OCCUPIABLE p{tgt.id} step={world.step} "
            f"srcs={[s for s,_,_,_ in prop.planned_sources]} ships={prop.required_ships} "
            f"priority={prop.priority:.1f} score={score:.1f}"
        )
        proposals.append(prop)
        seen.add(tgt.id)

    return proposals[:3]


# ── rotational hub tracking ───────────────────────────────────────────────────

def update_rotational_hubs(world):
    """Mark newly captured strategic planets as rotational hubs; expire stale ones."""
    global _rotational_hubs, _primary_launchpads
    store = _rotational_hubs.setdefault(world.player, {})
    launchpads = _primary_launchpads.setdefault(world.player, {})
    if _prev_owners:
        for p in world.my_planets:
            if _prev_owners.get(p.id) != world.player:
                mark_launchpad_after_capture(world, p)
            if _prev_owners.get(p.id) != world.player and p.id not in store:
                if (int(p.production) >= 2
                        or world.nearest_enemy_distance(p) <= FRONTLINE_DIST + 15
                        or world.cluster_distance(p) <= 30.0):
                    store[p.id] = world.step
                    world.add_debug(
                        f"ROTATIONAL_HUB_MARK p{p.id} prod={int(p.production)} "
                        f"step={world.step} enemy_d={world.nearest_enemy_distance(p):.1f}"
                    )
    for pid in list(store.keys()):
        p = world.planet_by_id.get(pid)
        if p is None or p.owner != world.player or world.step - store[pid] > ROTATIONAL_HUB_TTL:
            del store[pid]
    for pid in list(launchpads.keys()):
        p = world.planet_by_id.get(pid)
        if p is None or p.owner != world.player:
            del launchpads[pid]

    # Thin hubs are noted for reinforcement scoring, but never lock sources.
    for pid in store:
        p = world.planet_by_id.get(pid)
        if p is not None and p.owner == world.player:
            if int(p.ships) < world.reserve_for(p) + ROTATIONAL_HUB_REINFORCE_THRESH:
                world.add_debug(f"ROTATIONAL_HUB_THIN p{pid} ships={int(p.ships)}")


def get_active_rotational_hubs(world):
    return [
        world.planet_by_id[pid]
        for pid in _rotational_hubs.get(world.player, {})
        if pid in world.planet_by_id and world.planet_by_id[pid].owner == world.player
    ]


def generate_rotational_hub_reinforce_missions(world):
    """Reinforce vulnerable rotational hubs that are near the frontline."""
    proposals = []
    for hub in get_active_rotational_hubs(world):
        if world.real_incoming_threat(hub)["deficit"] > 0:
            continue  # emergency defense handles this
        reserve = world.reserve_for(hub)
        if int(hub.ships) >= reserve + ROTATIONAL_HUB_REINFORCE_THRESH:
            continue
        if world.nearest_enemy_distance(hub) > FRONTLINE_DIST + 22:
            continue
        need = reserve + ROTATIONAL_HUB_REINFORCE_THRESH - int(hub.ships)
        sources = sorted(
            [s for s in world.my_planets
             if s.id != hub.id
             and world.surplus(s) >= need
             and world.real_incoming_threat(s)["deficit"] <= 0
             and dp(s, hub) <= 42.0],
            key=lambda s: dp(s, hub),
        )
        if not sources:
            continue
        src = sources[0]
        send = min(world.surplus(src), need)
        angle, ok = world.aim(src, hub, send)
        if not ok:
            continue
        eta = world.eta(src, hub, send)
        if eta > world.remaining - 2:
            continue
        world.add_debug(
            f"ROTATIONAL_HUB_REINFORCE p{hub.id} src=p{src.id} send={send} "
            f"ships={int(hub.ships)} reserve={reserve} eta={eta:.1f}"
        )
        proposals.append(MissionProposal(
            kind="REINFORCE_CAPTURE",
            target_id=hub.id,
            priority=PRIORITY_HUB_REINFORCE_BASE + int(hub.production) * 3,
            required_ships=send,
            planned_sources=[(src.id, send, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"hub_reinforce p{hub.id} ships={int(hub.ships)} reserve={reserve}",
        ))
    return proposals[:2]


# ── MAIN19_TEMPO_ARBITER ───────────────────────────────────────────────────────

def _find_best_nearest_for_arbiter(world):
    """
    Find the single best nearby planet for the MAIN19_TEMPO_ARBITER.
    Returns (planet, source, need, score, race_status) or None.
    Criteria: dist<=32, ETA<=18, holdable, not already covered.
    """
    best = None
    best_score = -1e9
    for tgt in world.normal_planets:
        if tgt.owner == world.player or world.is_comet(tgt):
            continue
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        d = dp(src, tgt)
        if d > ARBITER_NEAREST_MAX_DIST:
            continue
        pool = sum(world.surplus(p) for p in world.my_planets
                   if dp(p, tgt) <= ARBITER_NEAREST_MAX_DIST + 8)
        need = world.ships_needed_to_capture(src, tgt, pool)
        if need <= 0 or pool < need:
            continue
        if world.incoming_to_targets.get(tgt.id, 0) >= need:
            continue
        eta = world.eta(src, tgt, max(1, need))
        if eta > ARBITER_NEAREST_MAX_ETA:
            continue
        if not world.can_hold_after_capture(tgt, eta, need + ARBITER_HOLD_MARGIN):
            continue
        prod = int(tgt.production)
        status, _, _ = neutral_race_status(world, tgt)
        if status == "ENEMY_FAVORED" and prod < 3:
            continue
        score = (prod * 60.0
                 + max(0.0, 32.0 - d) * 6.0
                 - need * 1.5
                 - eta * 5.0
                 + (20.0 if tgt.owner == -1 else 0.0)
                 + (30.0 if status == "SAFE" else -20.0 if status == "ENEMY_FAVORED" else 0.0))
        if score > best_score:
            best_score = score
            best = (tgt, src, need, score, status)
    return best


def run_tempo_arbiter(world, fleet_ratio, deadline):
    """Prioritise nearest-occupiable capture; yield to HV neutral if clearly better."""
    if fleet_ratio > FLEET_RATIO_SOFT:
        return []
    if not world.my_planets:
        return []

    nearest = _find_best_nearest_for_arbiter(world)
    if nearest is None:
        return []

    tgt, src, need, score, status = nearest
    prod = int(tgt.production)
    nearest_eta = world.eta(src, tgt, need)

    # Check if a HV neutral is clearly better (only matters when nearest is low-prod)
    if prod < 3:
        hv_candidates = sorted(
            [n for n in world.neutral_planets
             if not world.is_comet(n)
             and int(n.production) >= ARBITER_HV_PROD_OVERRIDE
             and int(n.ships) <= LOCAL_PRODUCTION_MAX_SHIPS
             and world.cluster_distance(n) <= MIDGAME_CONTEST_MAX_DIST],
            key=lambda n: world.cluster_distance(n),
        )
        for hv in hv_candidates[:2]:
            if time.perf_counter() > deadline:
                break
            hv_src = min(world.my_planets, key=lambda p: dp(p, hv), default=None)
            if hv_src is None:
                continue
            hv_need = world.ships_needed_to_capture(hv_src, hv, world.surplus(hv_src))
            if hv_need <= 0 or hv_need > world.surplus(hv_src):
                continue
            hv_eta = world.eta(hv_src, hv, hv_need)
            if hv_eta > nearest_eta + 8.0:
                continue
            if not world.can_hold_after_capture(hv, hv_eta, hv_need):
                continue
            world.add_debug(
                f"ARBITER_SELECT_HV_OVER_NEAREST hv=p{hv.id}(prod={int(hv.production)}) "
                f"nearest=p{tgt.id}(prod={prod}) hv_eta={hv_eta:.1f} near_eta={nearest_eta:.1f}"
            )
            return []  # yield to HV neutral layer

        world.add_debug(
            f"ARBITER_SKIP_HV_FOR_NEAREST nearest=p{tgt.id}(prod={prod}) "
            f"score={score:.1f} no_better_hv_found"
        )

    # Nearest wins — build grouped capture plan
    mission_type = "CAPTURE_NEUTRAL" if tgt.owner == -1 else "SYNC_ATTACK"
    candidate_sources = [
        p for p in world.my_planets
        if world.surplus(p) >= 3
        and world.real_incoming_threat(p)["deficit"] <= 0
        and dp(p, tgt) <= ARBITER_NEAREST_MAX_DIST + 8
    ]
    prop = build_capture_plan(
        world, tgt, mission_type, candidate_sources,
        max_sources=3,
        eta_spread_limit=3.0 if tgt.owner == -1 else 6.0,
    )
    if prop is None:
        # Fallback: single source
        ok_safe, _ = world.source_is_safe_for(src, tgt, mission_type, need)
        if not ok_safe:
            return []
        angle, ok = world.aim(src, tgt, need)
        if not ok:
            return []
        eta = world.eta(src, tgt, need)
        prop = MissionProposal(
            kind=mission_type,
            target_id=tgt.id,
            priority=PRIORITY_NEAREST_OCCUPIABLE + score * 0.05,
            required_ships=need,
            planned_sources=[(src.id, need, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"arbiter_nearest p{tgt.id} prod={prod} d={dp(src,tgt):.1f}",
        )
    else:
        prop.priority = PRIORITY_NEAREST_OCCUPIABLE + score * 0.05
        prop.reason = f"arbiter_nearest p{tgt.id} prod={prod} score={score:.1f}"

    world.add_debug(
        f"ARBITER_SELECT_NEAREST p{tgt.id} prod={prod} ships={int(tgt.ships)} "
        f"src=p{src.id} need={need} eta={nearest_eta:.1f} score={score:.1f} status={status}"
    )
    return [prop]


def _early_prod_on_track(world):
    """
    Returns (on_track, urgency_high, prod_target, planet_target).
    Compares current production/planet count to the early expansion target curve.
    """
    s = world.step
    prod_target   = 15 if s <= 25 else 30 if s <= 40 else 55 if s <= 60 else 80 if s <= 75 else 100
    planet_target = 2  if s <= 20 else 3  if s <= 30 else 5  if s <= 50 else 7
    on_track     = world.my_prod >= prod_target or len(world.my_planets) >= planet_target
    urgency_high = (world.my_prod < prod_target * 0.65) or (len(world.my_planets) < planet_target - 1)
    return on_track, urgency_high, prod_target, planet_target


def generate_early_production_rush_missions(world, deadline):
    """
    EARLY_PRODUCTION_RUSH_OPENING: runs steps 0–80.
    Captures nearby neutrals aggressively to maximise production before midgame.
    Generates grouped capture proposals; blocks enemy offense when behind target curve.
    """
    if world.step > 80 or world.features.get("final"):
        return []
    if not world.neutral_planets or not world.my_planets:
        return []

    fleet_ratio = compute_fleet_ratio(world)
    if fleet_ratio > FLEET_RATIO_SOFT:
        return []

    on_track, urgency_high, prod_target, planet_target = _early_prod_on_track(world)
    if on_track and not urgency_high:
        world.add_debug(
            f"EARLY_PROD_TARGET_ON_TRACK prod={world.my_prod}/{prod_target} "
            f"planets={len(world.my_planets)}/{planet_target}"
        )
        return []

    world.add_debug(
        f"EARLY_PROD_TARGET_BEHIND prod={world.my_prod}/{prod_target} "
        f"planets={len(world.my_planets)}/{planet_target} urgency={'high' if urgency_high else 'normal'}"
    )

    proposals = []
    rush_dist = 42.0 if urgency_high else 35.0

    targets = sorted(
        world.neutral_planets,
        key=lambda p: min(dp(m, p) for m in world.my_planets),
    )

    for tgt in targets:
        if time.perf_counter() > deadline:
            break
        if world.is_comet(tgt):
            continue
        d = min(dp(m, tgt) for m in world.my_planets)
        if d > rush_dist:
            continue
        if world.incoming_to_targets.get(tgt.id, 0) >= world.required_ships_to_capture(tgt):
            continue
        candidate_sources = [
            p for p in world.my_planets
            if world.surplus(p) >= 3
            and world.real_incoming_threat(p)["deficit"] <= 0
            and dp(p, tgt) <= rush_dist + 8
        ]
        if not candidate_sources:
            continue
        prop = build_capture_plan(
            world, tgt, "CAPTURE_NEUTRAL", candidate_sources,
            max_sources=3, eta_spread_limit=3.0,
        )
        if prop is None:
            continue
        prod_bonus = int(tgt.production) * 25.0
        dist_bonus = max(0.0, 38.0 - d) * 3.0
        prop.priority = 97.0 + prod_bonus + dist_bonus
        label = "EARLY_PROD_GROUPED_CAPTURE" if len(prop.planned_sources) > 1 else "EARLY_PROD_LOCAL_SWEEP"
        prop.reason = f"prod_rush p{tgt.id} prod={int(tgt.production)} d={d:.1f}"
        world.add_debug(
            f"{label} p{tgt.id} prod={int(tgt.production)} d={d:.1f} "
            f"srcs={[s for s,_,_,_ in prop.planned_sources]} ships={prop.required_ships}"
        )
        proposals.append(prop)

    return proposals[:4]


def generate_opportunistic_strike_missions(world, fleet_ratio, deadline, arbiter_fired=False):
    """
    Attacks enemy planets that recently launched fleets and are now thin.
    Only fires when: close enough, fleet_ratio safe, grouped attack flips ownership.
    All targets routed through should_allow_enemy_attack().
    """
    proposals = []
    if fleet_ratio > FLEET_RATIO_SOFT:
        return proposals
    if MIDGAME_START_STEP <= world.step < MIDGAME_END_STEP and fleet_ratio > MIDGAME_FLEET_SOFT:
        world.add_debug(f"NO_PANIC_BLOCK OPPORTUNISTIC_STRIKE fleet_ratio={fleet_ratio:.2f} midgame")
        return proposals
    if not world.enemy_planets or not world.my_planets:
        return proposals

    for tgt, w_score in detect_enemy_weakness(world)[:4]:
        if time.perf_counter() > deadline:
            break
        if not should_allow_enemy_attack(world, tgt, "SYNC_ATTACK", "opportunistic"):
            continue
        nearest_src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if nearest_src is None or dp(nearest_src, tgt) > BREACH_KILL_DIST + 8:
            world.add_debug(
                f"SKIP OPPORTUNISTIC_STRIKE p{tgt.id} step={world.step} "
                f"reason=too_far d={dp(nearest_src, tgt) if nearest_src else 999:.1f}"
            )
            continue
        candidate_sources = [
            p for p in world.my_planets
            if world.surplus(p) >= 5
            and dp(p, tgt) <= BREACH_KILL_DIST + 8
            and world.real_incoming_threat(p)["deficit"] <= 0
        ]
        if not candidate_sources:
            continue
        pool = sum(world.surplus(s) for s in candidate_sources)
        need = world.required_ships_to_capture(tgt, nearest_src)
        if need <= 0 or pool < need:
            world.add_debug(
                f"SKIP OPPORTUNISTIC_STRIKE p{tgt.id} step={world.step} "
                f"reason=pool={pool} need={need}"
            )
            continue
        prop = build_capture_plan(
            world, tgt, "SYNC_ATTACK", candidate_sources,
            max_sources=5, eta_spread_limit=6.0,
        )
        if prop is None:
            continue
        prop.priority = PRIORITY_OPPORTUNISTIC_STRIKE + w_score * 0.1 + int(tgt.production) * 4.0
        prop.reason = (
            f"opportunistic_strike p{tgt.id} weakness={w_score:.1f} "
            f"ships={int(tgt.ships)} prod={int(tgt.production)}"
        )
        world.add_debug(
            f"SELECT OPPORTUNISTIC_STRIKE p{tgt.id} step={world.step} "
            f"srcs={[s for s,_,_,_ in prop.planned_sources]} ships={prop.required_ships} "
            f"priority={prop.priority:.1f} weakness={w_score:.1f}"
        )
        proposals.append(prop)

    return proposals[:2]


def generate_launchpad_chain_missions(world, mode, deadline):
    """Use high-production/forward hubs to chain into nearby anchors before random offense."""
    proposals = []
    if not world.my_planets:
        return proposals
    hubs = command_hubs(world)[:5]
    if not hubs:
        return proposals

    for hub in hubs:
        if time.perf_counter() > deadline:
            break
        if world.real_incoming_threat(hub)["deficit"] > 0:
            continue
        candidates = []
        for tgt in world.neutral_planets + world.enemy_planets:
            if world.is_comet(tgt):
                continue
            if dp(hub, tgt) > LAUNCHPAD_RADIUS:
                continue
            if tgt.owner not in (-1, world.player) and not should_allow_enemy_attack(
                world, tgt, "SYNC_ATTACK", "launchpad_chain"
            ):
                continue
            if tgt.owner not in (-1, world.player) and mode not in (
                StrategyMode.CONTEST_HUBS, StrategyMode.RECOVER_AND_HOLD,
                StrategyMode.COLLAPSE, StrategyMode.FORCE_WAVE, StrategyMode.ANTI_LEADER,
            ):
                continue
            score = launchpad_target_score(world, hub, tgt, mode)
            candidates.append((score, tgt))
        if not candidates:
            continue
        candidates.sort(key=lambda item: -item[0])
        for score, tgt in candidates[:3]:
            mission_type = "LOCAL_PRODUCTION_CAPTURE" if tgt.owner == -1 else "SYNC_ATTACK"
            av = world.surplus(hub)
            need = world.ships_needed_to_capture(hub, tgt, av)
            if tgt.owner == -1 and av >= need and world.eta(hub, tgt, need) <= LAUNCHPAD_CHAIN_ETA:
                ok, reason = world.source_is_safe_for(hub, tgt, mission_type, need)
                if not ok:
                    world.add_debug(
                        f"LAUNCHPAD_CHAIN_SKIP hub=p{hub.id} target=p{tgt.id} need={need} avail={av} reason={reason}"
                    )
                    continue
                angle, aim_ok = world.aim(hub, tgt, need)
                if not aim_ok:
                    continue
                eta = world.eta(hub, tgt, need)
                priority = 108.0 + score * 0.18
                if int(tgt.production) >= 5:
                    priority += 35.0
                proposals.append(MissionProposal(
                    kind=mission_type,
                    target_id=tgt.id,
                    priority=priority,
                    required_ships=need,
                    planned_sources=[(hub.id, need, angle, eta)],
                    eta_min=eta,
                    eta_max=eta,
                    reason=f"launchpad_chain hub=p{hub.id} target=p{tgt.id} score={score:.1f}",
                ))
                break
            plan, reason = build_grouped_capture_plan(world, tgt, mission_type)
            if plan is None:
                world.add_debug(
                    f"LAUNCHPAD_CHAIN_SKIP hub=p{hub.id} target=p{tgt.id} need={need} avail={av} reason={reason}"
                )
                continue
            planned, group_need, eta_min, eta_max = plan
            proposals.append(MissionProposal(
                kind=mission_type,
                target_id=tgt.id,
                priority=104.0 + score * 0.16,
                required_ships=group_need,
                planned_sources=planned,
                eta_min=eta_min,
                eta_max=eta_max,
                reason=f"launchpad_group hub=p{hub.id} target=p{tgt.id} score={score:.1f}",
            ))
            break
    return proposals[:4]


# ── midgame control ───────────────────────────────────────────────────────────

def midgame_is_unstable(world, fleet_ratio):
    if world.step <= 50 or len(world.my_planets) < 4 or len(world.enemy_planets) < 4:
        return False
    prod_close = world.enemy_prod >= world.my_prod * 0.85
    local_enemy = any(
        min((dp(e, m) for m in world.my_planets), default=999.0) <= MIDGAME_FRONT_RADIUS
        for e in world.enemy_planets
        if not world.is_comet(e)
    )
    return (
        prod_close
        or fleet_ratio > 0.50
        or local_enemy
    )


def classify_midgame_state(world, fleet_ratio):
    """Classify the current midgame situation into one of 5 control states."""
    falling = sum(
        1 for p in world.my_planets
        if world.simulate_planet_timeline(p, 20)["fall_turn"] is not None
    )

    # RECOVER_AND_HOLD: over-committed or hemorrhaging planets
    if fleet_ratio > MIDGAME_FLEET_SOFT or falling >= 2:
        return MidgameState.RECOVER_AND_HOLD

    # FRONTLINE_STABILIZE: enemy close and we're losing ground
    enemy_near = any(world.nearest_enemy_distance(p) < FRONTLINE_DIST for p in world.my_planets)
    if enemy_near and falling >= 1:
        return MidgameState.FRONTLINE_STABILIZE

    # FOCUSED_BREACH: neutrals mostly gone and we have forward presence
    if len(world.neutral_planets) < 3 and is_breach_kill_mode(world) and fleet_ratio < MIDGAME_FLEET_SOFT - 0.05:
        return MidgameState.FOCUSED_BREACH

    # CONTEST_NEUTRALS: opponent racing for neutrals or catching up on production
    if world.neutral_planets:
        contested = sum(
            1 for n in world.neutral_planets
            if not world.is_comet(n) and world.enemy_incoming_to_targets.get(n.id, 0) > 0
        )
        prod_close = world.enemy_prod >= world.my_prod * 0.85
        if contested > 0 or prod_close:
            return MidgameState.CONTEST_NEUTRALS

    if world.neutral_planets and fleet_ratio < 0.40:
        return MidgameState.STABLE_EXPAND

    return MidgameState.CONTEST_NEUTRALS


def compute_cluster_stability(world, fleet_ratio):
    """Return a 0.0–1.0 score. Below MIDGAME_STABILITY_THRESHOLD → block offense."""
    if not world.my_planets:
        return 0.0

    thin = sum(1 for p in world.my_planets if int(p.ships) < world.reserve_for(p) + 5)
    thin_penalty = thin / len(world.my_planets) * 0.30

    falling = sum(
        1 for p in world.my_planets
        if world.simulate_planet_timeline(p, 20)["fall_turn"] is not None
    )
    fall_penalty = min(0.50, falling / max(1, len(world.my_planets)) * 0.50)

    ratio_penalty = max(0.0, (fleet_ratio - 0.35) * 1.50)

    cx = sum(p.x for p in world.my_planets) / len(world.my_planets)
    cy = sum(p.y for p in world.my_planets) / len(world.my_planets)
    nearby_enemy = sum(
        int(e.ships) for e in world.enemy_planets
        if dist(e.x, e.y, cx, cy) < MIDGAME_FRONT_RADIUS
    )
    pressure_penalty = min(0.25, nearby_enemy / max(1, world.enemy_total_ships) * 0.40)

    return max(0.0, min(1.0,
        1.0 - thin_penalty - fall_penalty - ratio_penalty - pressure_penalty
    ))


def select_active_front(world):
    """
    Pick one primary attack zone for this midgame turn.
    Returns (front_planet_or_None, description_str).
    Priority: contested neutral > nearest enemy cluster > weak enemy production.
    """
    if not world.my_planets:
        return None, "none"

    # Contested neutral with enemy inbound
    contested = [
        n for n in world.neutral_planets
        if not world.is_comet(n)
        and world.enemy_incoming_to_targets.get(n.id, 0) > 0
        and world.cluster_distance(n) <= MIDGAME_FRONT_RADIUS + 8
    ]
    if contested:
        best = min(contested, key=lambda n: world.cluster_distance(n))
        return best, f"contested_p{best.id}"

    # Nearest enemy planet
    near_enemy = min(
        (e for e in world.enemy_planets if not world.is_comet(e)),
        key=lambda e: min(dp(e, m) for m in world.my_planets),
        default=None,
    )
    if near_enemy:
        return near_enemy, f"nearest_enemy_p{near_enemy.id}"

    # Weakest enemy production cluster as a last offensive front.
    weak_prod = min(
        (e for e in world.enemy_planets if not world.is_comet(e) and int(e.production) >= 2),
        key=lambda e: (int(e.ships) / max(1, int(e.production)), min(dp(e, m) for m in world.my_planets)),
        default=None,
    )
    if weak_prod:
        return weak_prod, f"weak_prod_p{weak_prod.id}"

    return None, "none"


def generate_midgame_neutral_contest_missions(world, deadline):
    """Race or snipe nearby neutrals before opponent captures them."""
    proposals = []
    if not world.neutral_planets or not world.my_planets:
        return proposals

    for tgt in sorted(
        world.neutral_planets,
        key=lambda n: (world.cluster_distance(n), -int(n.production), int(n.ships)),
    ):
        if time.perf_counter() > deadline:
            break
        if world.is_comet(tgt):
            continue
        cluster_d = world.cluster_distance(tgt)
        if cluster_d > MIDGAME_CONTEST_MAX_DIST:
            continue
        if int(tgt.production) <= 1 and cluster_d > 32.0:
            continue

        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue

        need = world.ships_needed_to_capture(src, tgt)
        if need <= 0:
            continue

        my_eta, enemy_eta = world.reaction_times(tgt)
        enemy_inbound = world.enemy_incoming_to_targets.get(tgt.id, 0)
        is_contested = enemy_inbound > 0
        can_race = my_eta <= enemy_eta + 3.0

        # Skip enemy-favored low-prod neutrals during midgame
        race_status, _, _ = neutral_race_status(world, tgt)
        if race_status == "ENEMY_FAVORED" and int(tgt.production) < 4 and not is_contested:
            continue

        if not can_race and not is_contested:
            continue

        # For snipe: verify enemy won't own it before we arrive
        if is_contested and my_eta > enemy_eta:
            post_owner, _ = world.projected_state(tgt.id, max(1, int(my_eta)))
            if post_owner not in (-1, world.player):
                continue

        selected, pool, _ = estimate_grouped_sources(world, tgt, need, max_sources=PROACTIVE_EXPANSION_MAX_SOURCES)
        if not selected or pool < need:
            continue

        planned = [(s.id, send, angle, world.eta(s, tgt, send)) for s, send, angle in selected]
        eta_vals = [e for _, _, _, e in planned]

        ok_grp, grp_reason = validate_grouped_launch(world, tgt, planned)
        if not ok_grp:
            world.add_debug(f"MG_CONTEST_SKIP p{tgt.id} reason=validate_grouped_launch: {grp_reason}")
            continue

        priority = PRIORITY_MG_CONTEST_BASE + int(tgt.production) * 7.0
        if is_contested:
            priority += 18.0
        if my_eta < enemy_eta - 1:
            priority += 12.0

        proposals.append(MissionProposal(
            kind="CAPTURE_NEUTRAL",
            target_id=tgt.id,
            priority=priority,
            required_ships=need,
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"mg_contest p{tgt.id} prod={int(tgt.production)} contested={is_contested} my={my_eta:.1f} e={enemy_eta:.1f}",
        ))

    return proposals[:3]


def generate_capture_and_hold_missions(world):
    """Reinforce freshly captured forward planets that are thin and near enemy."""
    proposals = []
    if not world.my_planets or not _prev_owners:
        return proposals

    for tgt in world.my_planets:
        if world.is_comet(tgt):
            continue
        if _prev_owners.get(tgt.id, world.player) == world.player:
            continue  # not a fresh capture this turn
        if world.nearest_enemy_distance(tgt) > FRONTLINE_DIST + 18:
            continue  # far from enemy, no urgency

        current = int(tgt.ships)
        reserve = world.reserve_for(tgt)
        if current >= reserve + 10:
            continue  # already stocked

        reinforce = max(6, reserve + 12 - current)
        sources = sorted(
            [
                src for src in world.my_planets
                if src.id != tgt.id
                and world.surplus(src) >= reinforce
                and world.real_incoming_threat(src)["deficit"] <= 0
            ],
            key=lambda s: dp(s, tgt),
        )
        if not sources:
            continue

        src = sources[0]
        angle, ok = world.aim(src, tgt, reinforce)
        if not ok:
            continue
        eta = world.eta(src, tgt, reinforce)

        proposals.append(MissionProposal(
            kind="REINFORCE_CAPTURE",
            target_id=tgt.id,
            priority=86.0 + int(tgt.production) * 3,
            required_ships=reinforce,
            planned_sources=[(src.id, reinforce, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"cap_hold p{tgt.id} ships={current} enemy_d={world.nearest_enemy_distance(tgt):.1f}",
        ))

    return proposals


def generate_midgame_focused_breach_missions(world, front_planet, deadline):
    """Synchronized grouped attack on the single active-front enemy planet."""
    proposals = []
    if front_planet is None or world.is_comet(front_planet):
        return proposals
    if front_planet.owner in (-1, world.player):
        return proposals
    if not should_allow_enemy_attack(world, front_planet, "SYNC_ATTACK", "midgame_breach"):
        return proposals

    tgt = front_planet
    candidate_sources = sorted(
        [p for p in world.my_planets if world.surplus(p) >= 5],
        key=lambda p: dp(p, tgt),
    )[:MIDGAME_ATTACK_SOURCE_MAX]
    if not candidate_sources:
        return proposals

    prop = build_capture_plan(
        world, tgt, "SYNC_ATTACK", candidate_sources,
        max_sources=MIDGAME_ATTACK_SOURCE_MAX,
        eta_spread_limit=float(BREACH_ETA_SYNC + 4),
    )
    if prop is None:
        return proposals
    prop.priority = PRIORITY_MG_BREACH_BASE + int(tgt.production) * 4
    prop.reason = f"mg_breach p{tgt.id} need={prop.required_ships}"
    proposals.append(prop)
    return proposals


def generate_midgame_control_missions(world, mg_state, front_planet, fleet_ratio, deadline):
    """Dispatch midgame proposals based on classified state and active front."""
    proposals = []

    # Capture-and-hold is always applied when relevant
    if time.perf_counter() < deadline:
        proposals += generate_capture_and_hold_missions(world)

    if mg_state == MidgameState.RECOVER_AND_HOLD:
        world.add_debug(f"MIDGAME={mg_state} pause_offense fleet={fleet_ratio:.2f}")
        return proposals  # urgent passes already handled above

    if mg_state == MidgameState.FRONTLINE_STABILIZE:
        world.add_debug(f"MIDGAME={mg_state} hold_front")
        if time.perf_counter() < deadline:
            proposals += generate_midgame_neutral_contest_missions(world, deadline)
        return proposals

    if mg_state in (MidgameState.STABLE_EXPAND, MidgameState.CONTEST_NEUTRALS):
        if time.perf_counter() < deadline:
            proposals += generate_midgame_neutral_contest_missions(world, deadline)

    if mg_state == MidgameState.FOCUSED_BREACH:
        if front_planet is not None and front_planet.owner not in (-1, world.player):
            if time.perf_counter() < deadline:
                proposals += generate_midgame_focused_breach_missions(world, front_planet, deadline)

    return proposals


# ── always-on capture opportunity engine ─────────────────────────────────────

def should_allow_capture_opportunity(world, target, mission_type):
    """
    Permissive capture gate for the always-on opportunity engine.

    Hard-blocks ONLY when a capture is genuinely impossible or unsafe:
      - comet target
      - fleet ratio at hard cap
      - no safe surplus anywhere
      - cannot capture with available pool
      - absurdly far from cluster
      - not holdable AND far from cluster AND no urgent reason

    Phase, step count, 4-player context, and neutral density are soft scoring
    inputs handled by capture_opportunity_score().

    Returns (allowed: bool, reason: str).
    """
    if world.is_comet(target):
        world.add_debug(f"CAPTURE_REJECT_COMET p{target.id}")
        return False, "comet"
    if target.owner == world.player:
        return False, "already_mine"

    fleet_ratio = compute_fleet_ratio(world)
    if fleet_ratio > FLEET_RATIO_HARD:
        world.add_debug(f"CAPTURE_REJECT_FLEET_RATIO target=p{target.id} ratio={fleet_ratio:.2f}")
        return False, f"fleet_ratio={fleet_ratio:.2f}"

    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if src is None:
        return False, "no_source"

    safe_pool = sum(
        world.surplus(p) for p in world.my_planets
        if world.real_incoming_threat(p)["deficit"] <= 0
    )
    if safe_pool < MIN_SEND_SHIPS:
        world.add_debug(f"CAPTURE_REJECT_NO_SAFE_SOURCE target=p{target.id} pool={safe_pool}")
        return False, "no_safe_surplus"

    pool = sum(world.surplus(p) for p in world.my_planets)
    need = world.ships_needed_to_capture(src, target, pool)
    if need <= 0 or pool < need:
        return False, f"uncapturable pool={pool} need={need}"

    cluster_d = world.cluster_distance(target)
    if cluster_d > CAPTURE_OPP_MAX_DIST:
        world.add_debug(f"CAPTURE_REJECT_TOO_FAR target=p{target.id} cluster_d={cluster_d:.1f}")
        return False, f"too_far cluster_d={cluster_d:.1f}"

    eta = world.eta(src, target, need)
    if target.owner not in (-1, world.player):
        can_hold = world.can_hold_after_capture(target, eta, need)
        if not can_hold and cluster_d > MIDGAME_FRONT_RADIUS:
            world.add_debug(
                f"CAPTURE_REJECT_NOT_HOLDABLE target=p{target.id} "
                f"cluster_d={cluster_d:.1f} eta={eta:.1f}"
            )
            return False, "not_holdable_far"

    return True, "ok"


def capture_opportunity_score(world, target, src, need, eta, fleet_ratio):
    """
    Score a capture opportunity. Higher = more attractive.

    Phase, 4-player context, and neutral density act as soft deductions.
    Enemy planets receive bonuses for weakness, drain, proximity, and strategic value.
    """
    prod      = int(target.production)
    ships     = int(target.ships)
    cluster_d = world.cluster_distance(target)
    is_enemy  = target.owner not in (-1, world.player)
    arrival   = max(1, int(math.ceil(eta)))
    future    = max(1, world.remaining - arrival)

    s = 0.0

    # ── Core value ────────────────────────────────────────────────────────────
    s += 55.0                                 # +1 owned planet / control
    s += prod * 34.0                          # production gain
    s += prod * future * 0.52                 # future production value
    s -= need * 1.65                          # ship investment cost
    s -= eta  * 6.0                           # time cost
    s -= cluster_d * 0.5                      # distance from cluster
    world.add_debug(
        f"PLANET_VALUE_OVER_FLYING_SHIPS capture_score target=p{target.id} prod={prod} need={need}"
    )
    world.add_debug(f"PRODUCTION_VALUE_PRIORITY target=p{target.id} prod={prod}")

    # ── Enemy-specific bonuses ────────────────────────────────────────────────
    if is_enemy:
        s += prod * 30.0                      # weakens opponent production

        if ships <= ENEMY_GATE_WEAK_LOCAL:    # nearby weak target
            s += 28.0
            world.add_debug(f"ENEMY_CAPTURE_NEAR_WEAK target=p{target.id} ships={ships}")

        prev = _prev_ships.get(target.id)     # recently drained
        if prev is not None:
            drop = (prev + prod) - ships
            if drop >= CAPTURE_OPP_DRAINED_DROP:
                s += 35.0
                world.add_debug(f"ENEMY_CAPTURE_RECENTLY_DRAINED target=p{target.id} drop={drop}")

        if cluster_d <= MIDGAME_FRONT_RADIUS:
            s += 22.0
            world.add_debug(f"ENEMY_CAPTURE_FRONTIER target=p{target.id} d={cluster_d:.1f}")

        if any(                               # actively threatening my cluster
            dp(m, target) <= CHEAP_RECAPTURE_LOCAL_DIST
            and world.real_incoming_threat(m)["deficit"] > 0
            for m in world.my_planets
        ):
            s += 40.0

        if len(world.enemy_planets) <= 5:     # collapsing opponent
            s += 25.0 + max(0, 5 - len(world.enemy_planets)) * 8.0

        if target.owner == world.leader:      # leader planet
            s += 15.0

    # ── Holdability ───────────────────────────────────────────────────────────
    can_hold = world.can_hold_after_capture(target, eta, need)
    s += 35.0 if can_hold else -70.0

    # ── Bridge / route-fill ───────────────────────────────────────────────────
    if is_bridge_planet(world, target):
        s += 22.0

    # ── Small-planet bridge / territory scoring ───────────────────────────────
    if radius_class(target) == "SMALL":
        bv = small_bridge_score(world, target)
        if bv >= SMALL_BRIDGE_THRESHOLD:
            s += bv * 0.55
            world.add_debug(f"SMALL_BRIDGE_VALUE_FOUND p{target.id} bv={bv:.1f}")
        elif cluster_d <= SMALL_STORAGE_CAPTURE_DIST:
            # Pure storage value: useful for territory continuity when very close
            s += 12.0
        else:
            # Distant small planet with no bridge value: soft penalty
            s -= 20.0

    # ── Launchpad value: static/large-radius targets get a strong bonus ───────
    if is_launchpad_candidate(world, target, get_start_type(world) if world.my_planets else "MEDIUM"):
        s += launchpad_role_score(world, target, get_start_type(world) if world.my_planets else "MEDIUM") * 0.20

    # ── Anti-waiting: bonus when bot owns few planets ─────────────────────────
    if len(world.my_planets) < 4:
        s += 20.0                             # aggressively expand with few planets

    # ── Soft deductions (priority reduction, never veto) ─────────────────────
    my_pct, _, neutral_pct = compute_control_pct(world)

    # 4-player early game: slight de-priority for non-local enemy attacks
    if (world.is_four_player and world.step < FOUR_P_ATTACK_STEP
            and is_enemy and not is_local_enemy_opportunity(world, target)):
        s -= CAPTURE_OPP_4P_EARLY_PEN

    # Many neutrals remain: slight de-priority for enemy vs neutral preference
    if (neutral_pct > ENEMY_GATE_NEUTRAL_PCT
            and my_pct < ENEMY_GATE_MAX_MY_PCT
            and is_enemy):
        s -= CAPTURE_OPP_NEUTRAL_PEN

    # Fleet ratio approaching soft cap
    ratio_over = max(0.0, fleet_ratio - FLEET_RATIO_SOFT * 0.75)
    if ratio_over > 0:
        world.add_debug(
            f"HIGH_FLEET_RATIO_STATE_PENALTY capture target=p{target.id} ratio={fleet_ratio:.2f}"
        )
    s -= ratio_over * 110.0
    useful_flying, idle_flying = world.flying_ship_breakdown(world.player)
    if idle_flying:
        penalty = min(55.0, idle_flying * 0.25)
        s -= penalty
        world.add_debug(f"SCATTERED_FLEET_PENALTY capture idle={idle_flying} penalty={penalty:.1f}")
    world.add_debug(f"FLYING_SHIP_DISCOUNT_APPLIED capture target=p{target.id} need={need}")

    return s


def find_capture_opportunities(world, fleet_ratio, deadline):
    """
    Always-on capture opportunity engine.  Runs every turn after urgent actions.
    Evaluates BOTH neutral and enemy planets as capture targets. Fleet ratio
    pressure filters speculative options but still permits immediate conversion.

    Returns a list of MissionProposal objects sorted by capture_opportunity_score.
    Uses should_allow_capture_opportunity() for gating (permissive, safety-only).
    """
    proposals = []
    if not world.my_planets:
        return proposals

    world.add_debug(
        f"CAPTURE_OPPORTUNITY_SCAN step={world.step} "
        f"neutrals={len(world.neutral_planets)} enemies={len(world.enemy_planets)} "
        f"my_planets={len(world.my_planets)} fleet_ratio={fleet_ratio:.2f}"
    )
    world.add_debug(
        f"SMALL_BRIDGE_SCAN step={world.step} "
        f"small_planets={sum(1 for p in world.normal_planets if radius_class(p) == 'SMALL' and p.owner != world.player and not world.is_comet(p))}"
    )

    pool = sum(world.surplus(p) for p in world.my_planets)
    all_targets = [
        t for t in world.normal_planets
        if t.owner != world.player and not world.is_comet(t)
    ]

    scored: list = []
    for target in all_targets:
        if time.perf_counter() > deadline:
            break

        # Skip if already sufficiently covered by incoming fleets
        src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
        if src is None:
            continue
        if world.incoming_to_targets.get(target.id, 0) >= world.required_ships_to_capture(target, src):
            continue

        allowed, gate_reason = should_allow_capture_opportunity(
            world, target,
            "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK",
        )
        if not allowed:
            continue

        need = world.ships_needed_to_capture(src, target, pool)
        if need <= 0 or pool < need:
            continue
        eta = world.eta(src, target, need)
        if eta > CAPTURE_OPP_MAX_ETA:
            continue
        if fleet_ratio > FLEET_RATIO_SOFT:
            cluster_d = world.cluster_distance(target)
            can_hold = world.can_hold_after_capture(target, eta, need)
            quick_flip = eta <= min(CAPTURE_OPP_MAX_ETA, 18.0)
            high_value_neutral = (
                target.owner == -1
                and int(target.production) >= LOCAL_PRODUCTION_MIN_PROD
                and quick_flip
                and can_hold
            )
            weak_near_enemy = (
                target.owner not in (-1, world.player)
                and is_local_enemy_opportunity(world, target)
                and quick_flip
                and can_hold
            )
            local_neutral_fill = (
                target.owner == -1
                and cluster_d <= MIDGAME_FRONT_RADIUS
                and quick_flip
                and can_hold
            )
            if not (high_value_neutral or weak_near_enemy or local_neutral_fill):
                world.add_debug(
                    f"CAPTURE_OPPORTUNITY_RATIO_FILTER target=p{target.id} "
                    f"ratio={fleet_ratio:.2f} eta={eta:.1f} d={cluster_d:.1f}"
                )
                continue

        score = capture_opportunity_score(world, target, src, need, eta, fleet_ratio)
        if score < CAPTURE_OPP_MIN_SCORE:
            world.add_debug(
                f"CAPTURE_OPPORTUNITY_DEFER target=p{target.id} score={score:.1f}"
            )
            continue

        scored.append((score, target, src, need, eta))

    scored.sort(key=lambda x: -x[0])

    seen: set = set()
    for score, target, primary_src, need, eta in scored:
        if time.perf_counter() > deadline:
            break
        if target.id in seen or len(proposals) >= CAPTURE_OPP_MAX_PROPOSALS:
            break

        is_enemy  = target.owner not in (-1, world.player)
        mtype     = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"

        candidate_srcs = sorted(
            [p for p in world.my_planets
             if world.surplus(p) >= MIN_SEND_SHIPS
             and world.real_incoming_threat(p)["deficit"] <= 0
             and dp(p, target) <= CAPTURE_OPP_MAX_DIST + 8],
            key=lambda p: (dp(p, target), -world.surplus(p)),
        )[:MAX_GROUP_SOURCES]

        if not candidate_srcs:
            world.add_debug(f"CAPTURE_REJECT_NO_SAFE_SOURCE target=p{target.id}")
            continue

        prop = build_capture_plan(
            world, target, mtype, candidate_srcs,
            max_sources=min(4, len(candidate_srcs)),
            eta_spread_limit=3.0 if target.owner == -1 else 6.0,
        )
        if prop is None:
            continue

        prop.priority = 78.0 + score * 0.28

        # Emit context-specific debug markers
        if radius_class(target) == "SMALL":
            bv = small_bridge_score(world, target)
            if bv >= SMALL_BRIDGE_THRESHOLD:
                world.add_debug(f"SMALL_BRIDGE_CAPTURE_SELECTED p{target.id} bv={bv:.1f} score={score:.1f}")
                if world.is_four_player or (world.enemy_planets and world.cluster_distance(target) < 30.0):
                    world.add_debug(f"SMALL_PLANET_CAPTURED_AS_CONNECTOR p{target.id}")
            elif world.cluster_distance(target) <= SMALL_STORAGE_CAPTURE_DIST:
                world.add_debug(f"SMALL_STORAGE_CAPTURE_SELECTED p{target.id} cluster_d={world.cluster_distance(target):.1f}")
            else:
                world.add_debug(f"SMALL_PLANET_REJECT_NO_BRIDGE_VALUE p{target.id} bv={bv:.1f}")

        if is_enemy:
            neutral_pct = len(world.neutral_planets) / max(1, len(world.normal_planets))
            if world.is_four_player and world.step < FOUR_P_ATTACK_STEP:
                world.add_debug(
                    f"ENEMY_CAPTURE_ALLOWED_EARLY target=p{target.id} "
                    f"score={score:.1f} ships={int(target.ships)}"
                )
            if neutral_pct > ENEMY_GATE_NEUTRAL_PCT:
                world.add_debug(
                    f"ENEMY_CAPTURE_OVERRIDE_NEUTRAL_BLOCK target=p{target.id} "
                    f"neutral_pct={neutral_pct:.2f} score={score:.1f}"
                )

        world.add_debug(
            f"CAPTURE_OPPORTUNITY_ALLOWED target=p{target.id} "
            f"{'enemy' if is_enemy else 'neutral'} score={score:.1f} "
            f"need={prop.required_ships} eta={prop.eta_max:.1f} "
            f"d={world.cluster_distance(target):.1f}"
        )
        world.add_debug(f"CAPTURE_SELECTED target=p{target.id} score={score:.1f} pri={prop.priority:.1f}")

        proposals.append(prop)
        seen.add(target.id)

    return proposals


# ── bridge planet helpers ─────────────────────────────────────────────────────

def is_bridge_planet(world, planet):
    """
    True if `planet` (owned by us) lies usefully between the cluster center and
    a future expansion target, shortening the effective route by >= BRIDGE_MIN_SHORTCUT.

    A bridge gives us a closer launch point so fleet packets arrive faster.
    Used to:
      • give bonus score in the search planner for bridge captures,
      • decide whether to keep feeding a small/low-production planet,
      • avoid penalising reinforcement of small planets that genuinely help routing.
    """
    if not world.my_planets or not (world.neutral_planets + world.enemy_planets):
        return False
    cx = sum(p.x for p in world.my_planets) / len(world.my_planets)
    cy = sum(p.y for p in world.my_planets) / len(world.my_planets)
    for tgt in world.neutral_planets + world.enemy_planets:
        if world.is_comet(tgt) or dp(planet, tgt) > BRIDGE_RELAY_DIST:
            continue
        direct = dist(cx, cy, tgt.x, tgt.y)
        via    = dist(cx, cy, planet.x, planet.y) + dp(planet, tgt)
        if via < direct * (1.0 + BRIDGE_MIN_SHORTCUT):
            return True
    return False


def nearest_occupiable_from_bridge(world, bridge):
    """
    Return nearby capturable planets sorted by (distance, cost) from this bridge.
    Used to decide whether the bridge still has work to do (i.e., should_feed_bridge).
    """
    candidates = []
    for tgt in world.neutral_planets + world.enemy_planets:
        if world.is_comet(tgt):
            continue
        d = dp(bridge, tgt)
        if d > BRIDGE_RELAY_DIST:
            continue
        need = world.ships_needed_to_capture(bridge, tgt, int(bridge.ships))
        if need <= 0:
            continue
        candidates.append((d, need, tgt))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return [tgt for _, _, tgt in candidates]


def should_feed_bridge(world, planet):
    """
    True when a bridge planet should receive more ships before it can execute
    the next relay capture.  Only fires when:
      - planet is confirmed as a bridge,
      - it lacks sufficient surplus to capture the nearest reachable target
        after normalization.
    """
    if not is_bridge_planet(world, planet):
        return False
    reserve      = world.reserve_for(planet)
    next_targets = nearest_occupiable_from_bridge(world, planet)
    if not next_targets:
        return False
    need = world.ships_needed_to_capture(planet, next_targets[0], int(planet.ships))
    norm_need = normalize_send_amount(need)
    return int(planet.ships) < reserve + norm_need + 5


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH_ATTACK_PLANNER helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sim_proposal(world, prop):
    """
    Lightweight simulation of a MissionProposal's outcome.
    Uses cached simulate_planet_timeline / projected_state — fast in practice.
    Returns a stat dict, or None if the proposal is invalid.
    """
    tgt = world.planet_by_id.get(prop.target_id)
    if tgt is None or world.is_comet(tgt):
        return None
    total_sent = sum(s for _, s, _, _ in prop.planned_sources)
    if total_sent <= 0:
        return None
    arrival = int(math.ceil(prop.eta_max))

    # ── ownership after arrival ───────────────────────────────────────────────
    extra = tuple(
        (max(1, int(math.ceil(eta))), world.player, int(ships))
        for _, ships, _, eta in prop.planned_sources if int(ships) > 0
    )
    owner_after, ships_after = world.projected_state(
        prop.target_id, arrival + 1, extra_arrivals=extra
    )
    captured = (owner_after == world.player)

    # ── holdability ───────────────────────────────────────────────────────────
    holds = False
    if captured:
        plan_tl = tuple(
            (max(1, int(math.ceil(eta))), world.player, int(ships))
            for _, ships, _, eta in prop.planned_sources
        )
        horizon = min(SIM_HORIZON, arrival + 30)
        tl = world.simulate_planet_timeline(tgt, horizon, planned=plan_tl)
        holds = tl["fall_turn"] is None

    # ── fleet ratio after launch ──────────────────────────────────────────────
    cur_planet = sum(int(p.ships) for p in world.my_planets)
    cur_fleet  = sum(int(f.ships) for f in world.my_fleets)
    post_fleet_ratio = (cur_fleet + total_sent) / max(
        1, cur_planet - total_sent + cur_fleet + total_sent
    )

    # ── source hollowing ──────────────────────────────────────────────────────
    hollow_count = 0
    for src_id, ships, _, _ in prop.planned_sources:
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        after_commit = int(src.ships) - world.committed.get(src_id, 0) - ships
        reserve = world.reserve_for(src)
        if after_commit < reserve or after_commit < max(4, int(src.ships) * 0.18):
            hollow_count += 1

    # ── enemy response ETA ────────────────────────────────────────────────────
    enemy_back = 999.0
    if captured and world.enemy_planets:
        enemy_back = min(
            (world.eta(e, tgt, max(1, int(e.ships) // 2)) for e in world.enemy_planets),
            default=999.0,
        )

    # ── relay value: chain opportunities from the captured planet ─────────────
    relay_val = 0.0
    if captured and holds and ships_after > 0:
        for n in world.neutral_planets + world.enemy_planets:
            if n.id == prop.target_id or world.is_comet(n):
                continue
            d_relay = dp(tgt, n)
            if d_relay > LAUNCHPAD_RADIUS:
                continue
            relay_eta = travel_turns(d_relay, max(1, ships_after // 2))
            if relay_eta <= SEARCH_RELAY_HORIZON:
                relay_val += int(n.production) * 12.0 / max(1.0, relay_eta)

    cluster_d = world.cluster_distance(tgt)
    planet_control_gain = 1 if captured and tgt.owner != world.player else 0
    useful_flying, idle_flying = world.flying_ship_breakdown(world.player)
    return {
        "captured":        captured,
        "holds":           holds,
        "post_fleet_ratio": post_fleet_ratio,
        "hollow_count":    hollow_count,
        "enemy_back":      enemy_back,
        "prod_gain":       int(tgt.production) if captured else 0,
        "enemy_prod_lost": int(tgt.production) if (
            captured and tgt.owner not in (-1, world.player)
        ) else 0,
        "cluster_d":       cluster_d,
        "isolated":        cluster_d > OCCUPIABLE_MAX_DIST,
        "relay_val":       relay_val,
        "total_sent":      total_sent,
        "ships_after":     ships_after if captured else 0,
        "arrival":         arrival,
        "enemy_owns":      tgt.owner not in (-1, world.player),
        "planet_control_gain": planet_control_gain,
        "current_useful_flying": useful_flying,
        "current_idle_flying": idle_flying,
    }


def _score_search_proposal(world, prop, sim):
    """
    Score a proposal from its simulated outcome. Higher is better.

    Rewards: production gained, stable capture, enemy prod reduced, connected
    cluster growth, relay/route-fill value, bridge capture, nearest-occupiable
    bonus, strong grouped attack.

    Penalties: fleet overextension, hollow sources, isolated far captures,
    failed captures, fragile near-enemy planets, rapid enemy retake, scattered
    tiny attacks, draining useful sources, reinforcing non-bridge low-prod planets,
    fleets that do not convert to capture/defense/bridge value.
    """
    if sim is None:
        return -9999.0
    tgt = world.planet_by_id.get(prop.target_id)
    if tgt is None:
        return -9999.0

    s = 0.0
    future = max(1, world.remaining - sim["arrival"])
    ctrl_phase = classify_strategic_phase(world)

    # ── rewards ───────────────────────────────────────────────────────────────
    if sim["captured"] and sim["holds"]:
        planet_bonus = sim["planet_control_gain"] * 70.0
        prod_bonus = sim["prod_gain"] * future * 0.75
        s += planet_bonus                              # owned territory over airborne ships
        s += prod_bonus                                # production value over game
        s += 60.0 + int(tgt.production) * 24.0         # stably captured bonus
        s += sim["enemy_prod_lost"] * 46.0             # enemy production taken
        world.add_debug(
            f"SEARCH_SCORE_PLANET_CONTROL_BONUS {prop.kind}->p{prop.target_id} "
            f"planet={planet_bonus:.1f} prod={prod_bonus:.1f}"
        )

        # Connected cluster growth (closer to cluster = more useful territory)
        if not sim["isolated"]:
            s += max(0.0, OCCUPIABLE_MAX_DIST - sim["cluster_d"]) * 2.2
            # Count nearby owned planets as "cluster strength" bonus
            nearby_owned = sum(
                1 for m in world.my_planets if dp(m, tgt) <= LAUNCHPAD_RADIUS
            )
            s += nearby_owned * 6.0

        s += sim["relay_val"] * 0.38                  # relay / route-fill value

        # Bridge bonus: this capture shortens the route to future targets
        if is_bridge_planet(world, tgt):
            s += 32.0

        # Nearest-occupiable bonus: reward capturing the very closest available planet
        nearest_occ_d = min(
            (min(dp(m, n) for m in world.my_planets)
             for n in world.neutral_planets + world.enemy_planets
             if not world.is_comet(n)),
            default=999.0,
        )
        if sim["cluster_d"] <= nearest_occ_d * 1.25:
            s += 18.0

        # Phase-aware expansion bonus: extra reward when map still has space to grow
        if ctrl_phase in (ControlPhase.OPENING_EXPANSION, ControlPhase.LOCAL_SWEEP,
                          ControlPhase.EXPANSION_CONTROL):
            s += 12.0

        # Strong grouped attack (multiple coordinated sources)
        if len(prop.planned_sources) >= 2:
            s += 10.0

    elif sim["captured"]:
        s -= 35.0                                     # captured but can't hold

    # ── penalties ─────────────────────────────────────────────────────────────
    if not sim["captured"]:
        s -= 160.0                                    # failed/trickle attack
        if prop.kind not in CRITICAL_MISSIONS:
            world.add_debug(f"OWNERSHIP_CONVERSION_REQUIRED {prop.kind}->p{prop.target_id}")
            world.add_debug(f"MISSION_REJECT_NO_CONTROL_GAIN {prop.kind}->p{prop.target_id}")

    ratio_over = max(0.0, sim["post_fleet_ratio"] - FLEET_RATIO_SOFT)
    if ratio_over > 0:
        world.add_debug(
            f"HIGH_FLEET_RATIO_STATE_PENALTY {prop.kind}->p{prop.target_id} "
            f"ratio={sim['post_fleet_ratio']:.2f}"
        )
    s -= ratio_over * 210.0                           # fleet overextension
    s -= sim["hollow_count"] * 35.0                   # drained source planets
    if sim["current_idle_flying"] > 0:
        idle_pen = min(90.0, sim["current_idle_flying"] * 0.35)
        s -= idle_pen
        world.add_debug(
            f"SCATTERED_FLEET_PENALTY search idle={sim['current_idle_flying']} penalty={idle_pen:.1f}"
        )
    if sim["total_sent"] > 0:
        discount = sim["total_sent"] * (0.35 if sim["captured"] and sim["holds"] else 1.10)
        s -= discount
        world.add_debug(
            f"FLYING_SHIP_DISCOUNT_APPLIED {prop.kind}->p{prop.target_id} "
            f"sent={sim['total_sent']} discount={discount:.1f}"
        )

    if sim["isolated"]:
        s -= 45.0 + max(0.0, sim["cluster_d"] - OCCUPIABLE_MAX_DIST) * 0.9

    if sim["captured"]:
        enemy_d = world.nearest_enemy_distance(tgt)
        if enemy_d < 22.0 and sim["ships_after"] < int(tgt.production) * 3 + 6:
            s -= 30.0                                 # fragile near-enemy capture
        if sim["enemy_back"] < sim["arrival"] + 8:
            s -= 28.0                                 # enemy can retake very quickly

    # Scattered tiny attack (especially against enemy planets)
    if sim["total_sent"] < MIN_SEND_SHIPS * 1.5 and sim["enemy_owns"]:
        s -= 50.0

    # Penalize reinforcing low-production planets that are not useful bridges
    if prop.kind in ("REINFORCE_CAPTURE",) and tgt.owner == world.player:
        if int(tgt.production) <= 1 and not is_bridge_planet(world, tgt):
            s -= 30.0

    # During expansion phases, penalize far captures when closer ones exist
    if ctrl_phase in (ControlPhase.OPENING_EXPANSION, ControlPhase.LOCAL_SWEEP):
        if sim["cluster_d"] > OCCUPIABLE_MAX_DIST * 0.75:
            s -= 18.0

    # Penalize fleets whose source contributions are non-normalized (post-filter sanity)
    for _, ships, _, _ in prop.planned_sources:
        if 0 < ships < MIN_SEND_SHIPS:
            s -= 20.0  # proposal includes a sub-minimum contribution

    return s


def search_attack_planner(world, proposals, fleet_ratio, deadline):
    """
    Beam-search style offensive planner.

    1. Sort candidate proposals by existing priority.
    2. Simulate each (up to SEARCH_MAX_CANDIDATES) using _sim_proposal.
    3. Score each simulation with _score_search_proposal.
    4. Select the best SEARCH_SELECT_LIMIT proposals (reject negative-score ones).
    5. Return (selected, blocked); blocked=True suppresses all other offensive
       fallback for this turn.

    Debug labels emitted:
        SEARCH_START, SEARCH_CANDIDATES, SEARCH_EVAL,
        SEARCH_SELECT, SEARCH_REJECT_NEGATIVE, SEARCH_REJECT,
        SEARCH_BLOCK_PANIC, SEARCH_TIMEOUT_FALLBACK
    """
    if not proposals:
        return [], False

    # Panic gate: fleet ratio already too high
    if fleet_ratio > FLEET_RATIO_HARD:
        world.add_debug(f"SEARCH_BLOCK_PANIC fleet_ratio={fleet_ratio:.2f}")
        return [], True

    search_end = min(deadline, time.perf_counter() + SEARCH_TIME_BUDGET)

    candidates = sorted(proposals, key=lambda p: -p.priority)[:SEARCH_MAX_CANDIDATES]

    world.add_debug(
        f"SEARCH_START candidates={len(candidates)} fleet={fleet_ratio:.2f} step={world.step}"
    )
    world.add_debug(
        "SEARCH_CANDIDATES " + " | ".join(
            f"{p.kind[:6]}->p{p.target_id}(p={p.priority:.0f})" for p in candidates[:6]
        )
    )

    scored: list = []
    for prop in candidates:
        if time.perf_counter() > search_end:
            world.add_debug(
                f"SEARCH_TIMEOUT_FALLBACK evaluated={len(scored)}/{len(candidates)}"
            )
            break
        sim = _sim_proposal(world, prop)
        if sim is None:
            continue
        score = _score_search_proposal(world, prop, sim)
        world.add_debug(
            f"SEARCH_EVAL {prop.kind}->p{prop.target_id} "
            f"score={score:.1f} cap={sim['captured']} holds={sim['holds']} "
            f"prod={sim['prod_gain']} ratio={sim['post_fleet_ratio']:.2f} "
            f"hollow={sim['hollow_count']} relay={sim['relay_val']:.1f}"
        )
        scored.append((score, prop))

    if not scored:
        return [], False

    scored.sort(key=lambda x: -x[0])

    selected: list = []
    for score, prop in scored:
        if score < SEARCH_MIN_SCORE:
            world.add_debug(
                f"SEARCH_REJECT_NEGATIVE {prop.kind}->p{prop.target_id} score={score:.1f}"
            )
            continue
        if len(selected) >= SEARCH_SELECT_LIMIT:
            break
        selected.append(prop)

    selected_ids = {p.target_id for p in selected}
    for score, prop in scored:
        if prop.target_id in selected_ids:
            world.add_debug(
                f"SEARCH_SELECT {prop.kind}->p{prop.target_id} "
                f"score={score:.1f} pri={prop.priority:.1f} reason={prop.reason[:60]}"
            )
        elif score >= SEARCH_MIN_SCORE:
            world.add_debug(
                f"SEARCH_REJECT {prop.kind}->p{prop.target_id} "
                f"score={score:.1f} reason=not_top_{SEARCH_SELECT_LIMIT}"
            )

    # Any candidates scored → block other offensive fallback this turn
    blocked = len(scored) > 0
    return selected, blocked


# ── contact recovery ─────────────────────────────────────────────────────────

def agent(obs, config=None):
    start = time.perf_counter()
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    deadline = start + min(SOFT_DEADLINE, max(0.55, act_timeout * 0.82))
    world = WorldModel(obs)
    if not world.my_planets:
        update_ownership_memory(world)
        return []

    if not hasattr(agent, "_last_meaningful") or world.step <= 1:
        agent._last_meaningful = {}
        _wave_reservation["target_id"] = None
        _prev_owners.clear()
        _prev_ships.clear()
        _rotational_hubs[world.player] = {}
        _primary_launchpads[world.player] = {}
        _start_type_cache[world.player] = None
        _recently_reinforced[world.player] = {}
        _doomed_owned_targets[world.player] = {}
    last_meaningful = agent._last_meaningful.get(world.player, world.step)
    idle_turns = world.step - last_meaningful
    mode = choose_strategy_mode(world, idle_turns)

    def _finish(mark_meaningful=True):
        if mark_meaningful and (world.offensive_ships >= 15 or world.wave_attempted):
            agent._last_meaningful[world.player] = world.step
        if world.step == 80:
            world.add_debug(
                f"EARLY_PROD_80_SUMMARY prod={world.my_prod} enemy_prod={world.enemy_prod} "
                f"planets={len(world.my_planets)} ships={world.my_total_ships} "
                f"fleet_ratio={compute_fleet_ratio(world):.2f}"
            )
        if DEBUG:
            for event in world.debug_events:
                print(event)
        update_ownership_memory(world)
        return moves

    moves = []

    # ── 1. Emergency defense ──────────────────────────── commits to moves directly
    emergency_defense(world, moves)

    fleet_ratio = compute_fleet_ratio(world)
    ratio_blocks_normal = world.step >= EARLY_STEPS and fleet_ratio > FLEET_RATIO_SOFT
    in_forced_opening = len(world.my_planets) < FORCED_OPENING_PLANETS and world.step < FORCED_OPENING_STEP
    mg_active = (
        MIDGAME_START_STEP <= world.step < MIDGAME_END_STEP
        and len(world.my_planets) >= 3
        and not in_forced_opening
        and not world.features["final"]
    )
    # search_active: True in midgame (not forced-opening, not final).
    # Offensive missions are routed through search_attack_planner instead of the
    # plain coordinator; chain/occupiable expansion steps are also moved into the
    # search pool so the planner can evaluate them holistically.
    search_active = (
        not in_forced_opening
        and world.step >= MIDGAME_START_STEP
        and not world.features["final"]
    )
    update_rotational_hubs(world)

    world.add_debug(f"fleet_ratio={fleet_ratio:.2f} mode={mode} idle={idle_turns}")

    # Opening retarget: only before step 40, and only when in-flight ships are insufficient.
    if 0 < world.step < 40:
        for tgt_id, incoming in list(world.incoming_to_targets.items()):
            if incoming > 0:
                retarget, _ = should_retarget_opening(world, tgt_id)
                if retarget:
                    world.incoming_to_targets[tgt_id] = max(incoming, 999)

    # ── 2. Cheap recent-loss response / counterattack ─── proposals → coordinate_missions
    if time.perf_counter() < deadline:
        try_cheap_recapture_or_counterattack(world, moves, fleet_ratio, deadline)

    # ── 2a. 4-player stable-corner launchpad strategy ─── proposals → coordinate_missions
    if not moves and time.perf_counter() < deadline:
        corner_prop = find_4p_corner_expansion_target(world)
        if corner_prop is not None:
            coordinate_missions(world, [corner_prop], moves, fleet_ratio, deadline)

    # ── 2b. Planet-size role opening ───────────────────── commits grouped launchpad waves
    if not moves and time.perf_counter() < deadline:
        run_planet_role_opening(world, moves)

    # ── 3. Urgent: save / fall-recapture / evacuation ─── proposals → coordinate_missions
    protect_lead = is_protect_lead_mode(world)
    if protect_lead:
        world.add_debug("PROTECT_LEAD active")

    urgent: list = []
    if time.perf_counter() < deadline:
        urgent += generate_finish_capture_missions(world)
    if time.perf_counter() < deadline:
        urgent += generate_save_under_attack_missions(world)
    if time.perf_counter() < deadline:
        urgent += generate_doomed_evacuation_missions(world)
    if time.perf_counter() < deadline:
        urgent += generate_protect_lead_missions(world)
    if time.perf_counter() < deadline:
        urgent += generate_rotational_hub_reinforce_missions(world)
    if urgent and time.perf_counter() < deadline:
        coordinate_missions(world, urgent, moves, fleet_ratio, deadline)

    # ── 3a. EARLY_PRODUCTION_RUSH_OPENING (step 0–80) ──── proposals → coordinate_missions
    if world.step <= 80 and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
        rush_props = generate_early_production_rush_missions(world, deadline)
        if rush_props:
            world.add_debug(f"EARLY_PROD_RUSH_START step={world.step} missions={len(rush_props)}")
            coordinate_missions(world, rush_props, moves, fleet_ratio, deadline)

    # ── 3b. MAIN19_TEMPO_ARBITER: prioritise nearest-occupiable over HV/chain/strike ──
    _moves_before_arbiter = len(moves)
    if fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
        arbiter_props = run_tempo_arbiter(world, fleet_ratio, deadline)
        if arbiter_props:
            coordinate_missions(world, arbiter_props, moves, fleet_ratio, deadline)
    arbiter_fired = len(moves) > _moves_before_arbiter
    arbiter_turn_lock = arbiter_fired and world.offensive_ships >= 12
    if arbiter_turn_lock:
        world.add_debug(f"ARBITER_TURN_LOCK active offensive_ships={world.offensive_ships}")

    # ── 3.3. Stable launchpad strategy (pre-search path) ─────────────────────
    # High-priority proposals for static/large-radius planets. Runs in both
    # pre-search (here) and midgame (search pool, above).  The step 2a/2b path
    # already handles opening; this covers mid-game step < 55 and late-opening.
    if not search_active and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
        lp_props = generate_launchpad_strategy_missions(world, fleet_ratio, deadline)
        if lp_props and time.perf_counter() < deadline:
            coordinate_missions(world, lp_props, moves, fleet_ratio, deadline)

    # ── 3.5. Always-on capture opportunity engine (pre-search path only) ──────
    # In midgame (search_active=True) the engine feeds into the search pool below.
    # Here it runs unconditionally during opening / 4-player / any early step so
    # the bot never sits idle while capturable planets exist.
    if not search_active and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
        opp_props = find_capture_opportunities(world, fleet_ratio, deadline)
        if opp_props and time.perf_counter() < deadline:
            coordinate_missions(world, opp_props, moves, fleet_ratio, deadline)

    # ── 4. Production tempo: high-value neutrals ─────────proposals → coordinate_missions
    high_value_neutral_block = False
    hv_props = []
    if not arbiter_fired and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
        hv_props = generate_high_value_neutral_missions(world, deadline)
        high_value_neutral_block = bool(hv_props)
        if hv_props:
            coordinate_missions(world, hv_props, moves, fleet_ratio, deadline)

    # no_offense: True when HV neutral or arbiter locked this turn's offensive budget.
    # Also updated to True by search_attack_planner if fleet ratio panics.
    no_offense = high_value_neutral_block or arbiter_turn_lock

    # Chain missions run here only during early game; in midgame they enter the search pool.
    if not search_active and not arbiter_turn_lock and time.perf_counter() < deadline:
        chain_props = generate_launchpad_chain_missions(world, mode, deadline)
        if chain_props:
            coordinate_missions(
                world, chain_props, moves, fleet_ratio, deadline,
                midgame_active=mg_active, midgame_front=None,
            )
    elif arbiter_turn_lock:
        world.add_debug("SKIP_CHAIN_AFTER_ARBITER")

    # ── 4c. Nearest occupiable expansion (neutral + weak enemy near cluster) ──
    # In midgame, occupiable expansion enters the search pool instead of firing here.
    if not search_active and not arbiter_turn_lock and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
        occ_props = generate_nearest_occupiable_expansion_missions(world, deadline)
        if occ_props:
            coordinate_missions(world, occ_props, moves, fleet_ratio, deadline)
    elif arbiter_turn_lock:
        world.add_debug("SKIP_OCCUPIABLE_AFTER_ARBITER")

    # ── 4. Missed-neutral force (before opening, while ratio is low) ─────────
    if (
        not moves
        and not high_value_neutral_block
        and fleet_ratio <= FLEET_RATIO_SOFT
        and time.perf_counter() < deadline
    ):
        force_repeated_missed_neutral(world, moves)

    # ── 4b. Opening chain plan (depth-2, step<60) ────────proposals → coordinate_missions
    if (
        not moves
        and not high_value_neutral_block
        and world.step < 60
        and len(world.my_planets) <= 3
        and time.perf_counter() < deadline
    ):
        chain_prop = opening_chain_plan(world, deadline)
        if chain_prop is not None:
            coordinate_missions(world, [chain_prop], moves, fleet_ratio, deadline)

    # ── 5. Forced opening tempo ───────────────────────── commits directly or via coordinator
    # Bypasses normal ratio gates when opening tempo is still recoverable.
    opening_ok = (not ratio_blocks_normal) or in_forced_opening
    if not moves and not high_value_neutral_block and opening_ok and time.perf_counter() < deadline:
        if first_capture_360(world, moves):
            if world.step < 60 or len(world.my_planets) < 2:
                return _finish()

    if not moves and not high_value_neutral_block and opening_ok and time.perf_counter() < deadline:
        if early_nearest_expansion_360(world, moves):
            if world.step < 100 or len(world.my_planets) < 3:
                return _finish()

    if not moves and not high_value_neutral_block and in_forced_opening and world.step >= OPENING_STUCK_STEP and time.perf_counter() < deadline:
        if forced_opening_capture(world, moves):
            return _finish()

    # ── 6. Midgame state pre-computation ──────────────────────────────────────
    active_front_planet = None
    mg_state            = None
    cluster_stab        = 0.5

    if mg_active and time.perf_counter() < deadline:
        mg_state     = classify_midgame_state(world, fleet_ratio)
        cluster_stab = compute_cluster_stability(world, fleet_ratio)
        active_front_planet, front_desc = select_active_front(world)
        world.add_debug(
            f"MIDGAME mode={mg_state} stab={cluster_stab:.2f} front={front_desc} "
            f"fleet={fleet_ratio:.2f}"
        )

    # ── 6.5 / 7. SEARCH_ATTACK_PLANNER (midgame and beyond) ──────────────────
    #
    # When search_active (step >= 55, not forced-opening, not final):
    #   • Defensive midgame capture-hold can commit directly.
    #   • ALL offensive proposals are collected, beam-evaluated, and the best
    #     1–2 are forwarded to coordinate_missions.  All other offensive fallback
    #     is blocked for this turn.
    #
    # When NOT search_active (early game / final drain):
    #   • Original midgame control + coordinator run unchanged.
    #
    # search_offense_fired: set True when the search commits >= 10 offensive ships.
    # Used to suppress generic fallback tempo so we don't add scattered attacks
    # on top of a well-chosen search mission.
    search_offense_fired = False
    if search_active and time.perf_counter() < deadline:
        mg_blocked = mg_active and fleet_ratio > MIDGAME_FLEET_HARD

        # ── Defensive midgame: bypass search, commit directly ─────────────────
        if mg_active and mg_state is not None and not mg_blocked:
            if (not high_value_neutral_block and not arbiter_turn_lock
                    and cluster_stab >= MIDGAME_STABILITY_THRESHOLD):
                def_mg: list = []
                if time.perf_counter() < deadline:
                    def_mg += generate_capture_and_hold_missions(world)
                if def_mg and time.perf_counter() < deadline:
                    coordinate_missions(
                        world, def_mg, moves, fleet_ratio, deadline,
                        midgame_active=mg_active, midgame_front=active_front_planet,
                    )

        # ── Collect offensive proposals for beam-search ───────────────────────
        search_pool: list = []

        # Stable launchpad strategy: static/large-radius planets get highest priority
        # in the search pool so the planner always considers the best launchpad target.
        if not no_offense and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
            search_pool += generate_launchpad_strategy_missions(world, fleet_ratio, deadline)

        # Always-on capture opportunity engine feeds into the search pool.
        # This ensures the engine runs at all steps >= 55, with the search planner
        # ranking its candidates alongside the existing generators.
        if not no_offense and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline:
            search_pool += find_capture_opportunities(world, fleet_ratio, deadline)

        if not mg_blocked and cluster_stab >= MIDGAME_STABILITY_THRESHOLD:
            if (mg_active and mg_state is not None
                    and not high_value_neutral_block and not arbiter_turn_lock):
                if mg_state in (MidgameState.STABLE_EXPAND, MidgameState.CONTEST_NEUTRALS):
                    if time.perf_counter() < deadline:
                        search_pool += generate_midgame_neutral_contest_missions(world, deadline)
                if (mg_state == MidgameState.FOCUSED_BREACH
                        and active_front_planet is not None):
                    if time.perf_counter() < deadline:
                        search_pool += generate_midgame_focused_breach_missions(
                            world, active_front_planet, deadline
                        )

        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            search_pool += generate_breach_kill_missions(world)
        if not no_offense and time.perf_counter() < deadline:
            search_pool += generate_expansion_missions(world, deadline)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            search_pool += generate_local_strike_missions(world)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            search_pool += generate_sync_attack_missions(world, mode, deadline)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            search_pool += generate_anti_leader_missions(world)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            search_pool += generate_collapse_missions(world, mode)
        if (not no_offense and not protect_lead and not arbiter_turn_lock
                and time.perf_counter() < deadline):
            search_pool += generate_opportunistic_strike_missions(
                world, fleet_ratio, deadline, arbiter_fired
            )
        # Launchpad chain + nearest-occupiable moved from step 4 into the search pool
        if (not no_offense and not protect_lead and not arbiter_turn_lock
                and time.perf_counter() < deadline):
            search_pool += generate_launchpad_chain_missions(world, mode, deadline)
        if (not no_offense and fleet_ratio <= FLEET_RATIO_SOFT
                and time.perf_counter() < deadline):
            search_pool += generate_nearest_occupiable_expansion_missions(world, deadline)

        # Wave reservation: inject if pool is ready or overdue
        if _wave_reservation["target_id"] is not None and time.perf_counter() < deadline:
            rsv_tgt = world.planet_by_id.get(_wave_reservation["target_id"])
            if rsv_tgt is None or rsv_tgt.owner == world.player:
                _wave_reservation["target_id"] = None
            else:
                rsv_pool = sum(
                    world.surplus(p) for p in world.my_planets
                    if p.id in set(_wave_reservation["source_ids"])
                )
                force_launch = world.step >= _wave_reservation["launch_by_step"]
                if rsv_pool >= _wave_reservation["required_ships"] or force_launch:
                    wave_prop = generate_organized_wave_mission(world, rsv_tgt)
                    if wave_prop is not None:
                        search_pool.insert(0, wave_prop)
                    _wave_reservation["target_id"] = None

        # ── Central enemy-attack gate applied to entire search pool ───────────
        # Any proposal targeting an enemy planet that wasn't already filtered by
        # its generator is caught here as a final backstop.
        filtered_pool = []
        for prop in search_pool:
            tgt_p = world.planet_by_id.get(prop.target_id)
            if (tgt_p is not None
                    and tgt_p.owner not in (-1, world.player)
                    and not world.is_comet(tgt_p)
                    and not should_allow_enemy_attack(
                        world, tgt_p, prop.kind, "search_pool_backstop"
                    )):
                world.add_debug(
                    f"SEARCH_ENEMY_GATE_BLOCK {prop.kind} p{prop.target_id}"
                )
            else:
                filtered_pool.append(prop)
        search_pool = filtered_pool

        # ── Run beam-search planner ───────────────────────────────────────────
        if search_pool and time.perf_counter() < deadline:
            selected_offensive, search_blocked = search_attack_planner(
                world, search_pool, fleet_ratio, deadline
            )
            if search_blocked:
                no_offense = True

            # Track whether search committed any real offensive ships
            if selected_offensive and any(
                p.kind in OFFENSIVE_MISSIONS for p in selected_offensive
            ):
                search_offense_fired = True

            # Apply midgame fleet-ratio + one-front filters to selected proposals
            if mg_active and selected_offensive:
                if fleet_ratio > MIDGAME_FLEET_HARD:
                    selected_offensive = [
                        p for p in selected_offensive if p.kind in CRITICAL_MISSIONS
                    ]
                elif fleet_ratio > MIDGAME_FLEET_SOFT:
                    selected_offensive = [
                        p for p in selected_offensive
                        if p.kind in CRITICAL_MISSIONS | {
                            "CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE",
                            "HIGH_VALUE_NEUTRAL_RACE",
                        }
                    ]
                if active_front_planet is not None:
                    always_free = CRITICAL_MISSIONS | {
                        "FINISH_ZERO_CAPTURE", "REINFORCE_CAPTURE",
                        "CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE",
                        "HIGH_VALUE_NEUTRAL_RACE",
                    }
                    keep = []
                    for prop in selected_offensive:
                        if prop.kind in always_free:
                            keep.append(prop)
                            continue
                        tgt_c = world.planet_by_id.get(prop.target_id)
                        if (tgt_c is not None
                                and dp(tgt_c, active_front_planet) <= MIDGAME_FRONT_RADIUS):
                            keep.append(prop)
                        else:
                            world.add_debug(
                                f"SEARCH_MG_FRONT_FILTER {prop.kind} p{prop.target_id}"
                            )
                    selected_offensive = keep

            if selected_offensive and time.perf_counter() < deadline:
                coordinate_missions(
                    world, selected_offensive, moves, fleet_ratio, deadline,
                    midgame_active=mg_active, midgame_front=active_front_planet,
                )

        # Snipe: time-sensitive (specific timing window), always runs after search
        if not no_offense and time.perf_counter() < deadline:
            snipe_props = generate_snipe_missions(world)
            if snipe_props and time.perf_counter() < deadline:
                coordinate_missions(world, snipe_props, moves, fleet_ratio, deadline)

    else:
        # ── original midgame + coordinator (pre-55, post-220, or final) ───────

        if mg_active and time.perf_counter() < deadline:
            mg_blocked = fleet_ratio > MIDGAME_FLEET_HARD
            if mg_blocked:
                world.add_debug(f"MIDGAME_RATIO_GUARD fleet={fleet_ratio:.2f}")
            if (not high_value_neutral_block and not arbiter_turn_lock
                    and cluster_stab >= MIDGAME_STABILITY_THRESHOLD and not mg_blocked):
                mg_props = generate_midgame_control_missions(
                    world, mg_state, active_front_planet, fleet_ratio, deadline
                )
                if mg_props and time.perf_counter() < deadline:
                    coordinate_missions(
                        world, mg_props, moves, fleet_ratio, deadline,
                        midgame_active=True, midgame_front=active_front_planet,
                    )
            elif high_value_neutral_block:
                world.add_debug("MIDGAME_OFFENSE_BLOCKED reason=high-value neutral pending")
            elif mg_blocked:
                world.add_debug(f"MIDGAME_RATIO_GUARD fleet={fleet_ratio:.2f}")
            else:
                world.add_debug(
                    f"MIDGAME_OFFENSE_BLOCKED stab={cluster_stab:.2f} fleet={fleet_ratio:.2f}"
                )

        # ── original mission coordinator ──────────────────────────────────────
        proposals: list = []
        if high_value_neutral_block:
            world.add_debug("NORMAL_OFFENSE_BLOCKED reason=high-value neutral exists")
        if arbiter_turn_lock:
            world.add_debug(
                f"SKIP_COORDINATOR_AFTER_ARBITER offensive_ships={world.offensive_ships}"
            )
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            proposals += generate_breach_kill_missions(world)
        if not no_offense and time.perf_counter() < deadline:
            proposals += generate_snipe_missions(world)
        if not no_offense and time.perf_counter() < deadline:
            proposals += generate_expansion_missions(world, deadline)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            proposals += generate_local_strike_missions(world)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            proposals += generate_sync_attack_missions(world, mode, deadline)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            proposals += generate_anti_leader_missions(world)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            proposals += generate_collapse_missions(world, mode)
        if not no_offense and not protect_lead and time.perf_counter() < deadline:
            proposals += generate_opportunistic_strike_missions(
                world, fleet_ratio, deadline, arbiter_fired=arbiter_fired
            )
        elif arbiter_turn_lock and not high_value_neutral_block:
            world.add_debug(
                f"SKIP_STRIKE_AFTER_ARBITER offensive_ships={world.offensive_ships}"
            )

        # Wave reservation
        if _wave_reservation["target_id"] is not None and time.perf_counter() < deadline:
            rsv_tgt = world.planet_by_id.get(_wave_reservation["target_id"])
            if rsv_tgt is None or rsv_tgt.owner == world.player:
                _wave_reservation["target_id"] = None
            else:
                rsv_pool = sum(
                    world.surplus(p) for p in world.my_planets
                    if p.id in set(_wave_reservation["source_ids"])
                )
                force_launch = world.step >= _wave_reservation["launch_by_step"]
                if rsv_pool >= _wave_reservation["required_ships"] or force_launch:
                    wave_prop = generate_organized_wave_mission(world, rsv_tgt)
                    if wave_prop is not None:
                        proposals.insert(0, wave_prop)
                    _wave_reservation["target_id"] = None

        # Central enemy-attack gate backstop (non-search coordinator path)
        proposals = [
            p for p in proposals
            if (world.planet_by_id.get(p.target_id) is None
                or world.planet_by_id[p.target_id].owner in (-1, world.player)
                or world.is_comet(world.planet_by_id[p.target_id])
                or should_allow_enemy_attack(
                    world, world.planet_by_id[p.target_id], p.kind,
                    "coordinator_backstop"
                ))
        ]

        filter_front = active_front_planet
        if mg_active:
            if fleet_ratio > MIDGAME_FLEET_HARD:
                proposals = [p for p in proposals if p.kind in CRITICAL_MISSIONS]
            elif fleet_ratio > MIDGAME_FLEET_SOFT:
                proposals = [
                    p for p in proposals
                    if p.kind in CRITICAL_MISSIONS or p.kind in (
                        "CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE"
                    )
                ]
            if filter_front is not None:
                always_free = CRITICAL_MISSIONS | {
                    "FINISH_ZERO_CAPTURE", "REINFORCE_CAPTURE",
                    "CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE",
                }
                keep = []
                for prop in proposals:
                    if prop.kind in always_free:
                        keep.append(prop)
                        continue
                    tgt = world.planet_by_id.get(prop.target_id)
                    if tgt is None:
                        continue
                    d = dp(tgt, filter_front)
                    if d <= MIDGAME_FRONT_RADIUS:
                        keep.append(prop)
                    else:
                        world.add_debug(
                            f"MG_FRONT_FILTER {prop.kind} p{prop.target_id} d={d:.1f}"
                        )
                proposals = keep

        if not moves and proposals and time.perf_counter() < deadline:
            coordinate_missions(
                world, proposals, moves, fleet_ratio, deadline,
                midgame_active=mg_active, midgame_front=filter_front,
            )

    # ── 8. Endgame consolidation + final drain ─────────── always run ──────────
    if world.remaining < ENDGAME_CONSOL_REMAINING and time.perf_counter() < deadline:
        eg_props = generate_endgame_consolidation_missions(world)
        if eg_props and time.perf_counter() < deadline:
            coordinate_missions(world, eg_props, moves, fleet_ratio, deadline)
    if time.perf_counter() < deadline:
        drain_props = generate_final_drain_missions(world)
        if drain_props and time.perf_counter() < deadline:
            coordinate_missions(world, drain_props, moves, fleet_ratio, deadline)

    # ── 9. Fallback tempo ──────────────────────────────── commits directly ────
    # Skip generic fallback when the search already fired a real offensive mission
    # this turn — prevents tacking small uncoordinated attacks onto a chosen mission.
    if (not moves and not search_offense_fired
            and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline):
        force_action_if_stalling(world, moves, idle_turns, deadline)

    if (not moves and not search_offense_fired
            and fleet_ratio <= FLEET_RATIO_SOFT and time.perf_counter() < deadline):
        fallback_tempo(world, moves)

    missed_opportunity_detector(world, moves)
    return _finish()
