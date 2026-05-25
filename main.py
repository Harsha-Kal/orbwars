"""
Orbit Wars – Competitive Agent
================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

Decision order each turn (agent()):
  1. Packet-safe emergency defense
  2. Small-start escape / lost-planet chain retrigger
  3. Route-aware launchpad-chain plan
  4. Main33 opening tempo / nearest arbiter / HV neutral / local attacks
  5. Expansion obligation / campaign while neutrals or planet deficit remain
  6. Idle-army pressure / surplus conversion
  7. Production bank / rally to staging
  8. Verified rolling capture chain
  9. Chain-aware fallback only
  10. Final drain

Key systems:
  - MissionLedger: central coordination, prevents trickle attacks
  - WorldModel: per-turn state, ownership simulation, source safety checks
  - Launchpad-chain opening: start-type-aware route scoring, no safety bypass
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
FORCED_OPENING_STEP     = 80    # launchpad-chain opening scoring window; no safety bypass
FORCED_OPENING_PLANETS  = 3     # deprecated old-pipeline threshold kept for compatibility
OPENING_STUCK_STEP      = 6     # if still 1 planet at this step: force-capture
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
DECOY_FLEET_THRESHOLD = 8    # enemy fleets below this ship count are ignored for strategic panic
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
ENEMY_ATTACK_SYNC_WINDOW = 2
CONTESTED_NEUTRAL_SYNC_WINDOW = 3
NORMAL_NEUTRAL_SYNC_WINDOW = 4
STAGING_SYNC_WINDOW = 3
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
PRIORITY_SMALL_START_ESCAPE  = 285.0
PRIORITY_PARALLEL_OPENING_SWEEP = 270.0
PRIORITY_EARLY_NEAREST_SWEEP = 235.0
PRIORITY_MULTI_AXIS_EXPAND   = 168.0

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
SMALL_START_STALL_STEP    = 4     # SMALL_START: force escape if still 1 planet here

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
SEARCH_SELECT_LIMIT    = 1      # default cap; opening bundle logic relaxes this dynamically
SEARCH_MIN_SCORE       = -35.0  # reject proposals whose search score < this
SEARCH_TIME_BUDGET     = 0.10   # seconds budget for the full search pass
SEARCH_RELAY_HORIZON   = 20.0   # max ETA for relay capture to count toward relay value
BEAM_EXPANSION_DEPTH   = 4      # short opening/midgame sequence depth
BEAM_EXPANSION_WIDTH   = 10     # enough breadth to find tempo without burning clock
BEAM_EXPANSION_RADIUS  = 58.0   # local expansion search radius around owned/beam nodes
BEAM_OBLIGATION_GAP    = 5      # max turns between early capture attempts when behind

# ── fleet packet discipline ───────────────────────────────────────────────────
MIN_SEND_SHIPS   = 10   # no fleet below this size; prevents panic micro-fleets
SEND_GRANULARITY = 5    # every launch size must be a multiple of this value

# ── map control phases (planet % based, step number is secondary) ─────────────
PHASE_OPENING_PCT  = 0.12   # my control below this → OPENING_EXPANSION
PHASE_SWEEP_PCT    = 0.28   # my control below this → LOCAL_SWEEP
PHASE_EXPAND_PCT   = 0.45   # my control below this → EXPANSION_CONTROL
PHASE_INITIAL_MAX   = 0.20   # control-based INITIAL_EXPANSION ceiling
PHASE_MIDGAME_MAX   = 0.50   # control-based MIDGAME_CONTROL ceiling
PHASE_COLLAPSE_MIN  = 0.65   # control-based COLLAPSE trigger
# above 0.45 → CONTACT or COLLAPSE based on enemy proximity

# ── bridge planet detection ───────────────────────────────────────────────────
BRIDGE_RELAY_DIST   = 50.0  # max distance from bridge to next useful capture target
BRIDGE_MIN_SHORTCUT = 0.15  # bridge must shorten direct route by at least this fraction

# ── enemy-attack scoring nudges ──────────────────────────────────────────────
ENEMY_GATE_NEUTRAL_PCT     = 0.35  # neutral density where enemy attacks get a soft score penalty
ENEMY_GATE_MAX_MY_PCT      = 0.28  # my control below which neutrals are softly preferred
ENEMY_GATE_WEAK_SHIPS      = 12    # <= this ships → very weak enemy
ENEMY_GATE_WEAK_LOCAL      = 15    # <= this ships → "weak" for LOCAL_SWEEP phase rule

# ── always-on capture opportunity engine ──────────────────────────────────────
CAPTURE_OPP_MAX_DIST      = 55.0  # max cluster_distance to scan for opportunities
CAPTURE_OPP_MAX_ETA       = 30.0  # max ETA for an opportunity candidate
CAPTURE_OPP_MIN_SCORE     = -60.0 # discard opportunities below this score
CAPTURE_OPP_MAX_PROPOSALS = 5     # max proposals returned per turn
CAPTURE_OPP_DRAINED_DROP  = 10    # ship drop (vs expected) to count as recently drained
LOCAL_DIRECT_OPENING_DIST = 58.0  # direct attacks should stay in the local corner early
LOCAL_DIRECT_MIDGAME_DIST = 78.0  # midgame local cluster direct-attack radius
FAR_DIRECT_MAX_ETA        = 24.0  # far direct shots must still arrive quickly
# Soft-priority penalties; lower absolute value = gentler deduction
CAPTURE_OPP_4P_EARLY_PEN  = 18.0  # 4-player + early + non-local enemy
CAPTURE_OPP_NEUTRAL_PEN   = 15.0  # many neutrals remain + enemy + not urgent

# ── launchpad-chain strategy ───────────────────────────────────────────────────
RADIUS_SMALL         = 1.2    # radius <= this → STORAGE role
RADIUS_LARGE         = 2.3    # radius >= this → LAUNCHPAD role (BRIDGE otherwise)
LAUNCHPAD_RESERVE    = 45     # ships kept on large launchpad planets
LAUNCHPAD_GUARD_SUPPORT_RADIUS = 72.0
FRONT_ATTACKER_FLOW_RADIUS = 58.0
BRIDGE_RESERVE       = 25     # ships kept on medium bridge planets
STORAGE_RESERVE      = 12     # ships kept on small storage/battery planets
CHAIN_FORCE_MIN      = 100    # minimum grouped force for chain counterattack
CHAIN_RADIUS         = 50.0   # scan radius for chain targets and rally sources
CAMPAIGN_RADIUS      = 55.0   # early expansion campaign scan radius
ATTACK_SOURCE_RADIUS = 55.0   # radius when searching for the attacker source
CHAIN_RETRIGGER_FRAC = 0.25   # max fraction of total fleet for cheap recapture
CHAIN_LOCAL_FRAC     = 0.60   # max fraction of local nearby surplus for recapture
CHAIN_BANK_MAX_TURNS = 10     # never wait longer than this many steps to attack

# ── two-axis / geometric expansion ─────────────────────────────────────────────
EARLY_NEAREST_SWEEP_STEP_MAX = 70
EARLY_NEAREST_SWEEP_DIST     = 54.0
EARLY_NEAREST_SWEEP_BURST    = 4
PARALLEL_OPENING_SWEEP_DIST  = 66.0
MULTI_AXIS_FRONT_RADIUS      = 72.0
SAME_STEP_BUNDLE_MAX         = 4
SAME_STEP_BUNDLE_MIN         = 2

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

# ── mid-game fleet stability ─────────────────────────────────────────────────
SIGNIFICANT_PROD_LEAD_MULT = 1.25    # only split captures when production lead is clearly safe
STAGING_CRITICAL_MASS_MULT = 0.75    # staging planets may start below final strike mass when backup is inbound
FINAL_CAPTURE_CRITICAL_MASS_MULT = 1.05  # final capture still needs overmatch at target arrival
HUB_SECURITY_PROD_THRESHOLD = 3       # production-3+ owned planets are protected as hubs
HUB_SECURITY_BASE_GARRISON = LOCAL_HUB_SHIPS  # minimum ships to hold on productive hubs
HUB_SECURITY_PROD_GARRISON_MULT = 15  # prod-scaled hub ship buffer
STAGING_FLEET_RATIO_TRIGGER = 0.80    # begin staging when total fleet is close to enemy fleet
STAGING_MIN_STEPS = 5                 # bank at launchpads for this many steps before striking
STAGING_RELEASE_STEPS = 5             # allow strikes after staging before restarting the bank
TACTICAL_STALL_TURNS = 7              # force conversion if captures have stalled
LOW_TERRITORY_PRESSURE_THRESHOLD = 0  # last-25-turn net planet growth must stay positive

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

def aim_solution_metrics(src, tgt, ang_vel, ships):
    tx, ty = tgt.x, tgt.y
    d_est = dp(src, tgt)
    for _ in range(18):
        d_est = dist(src.x, src.y, tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    angle = safe_angle(src.x, src.y, tx, ty)
    aimed_x = src.x + d_est * math.cos(angle)
    aimed_y = src.y + d_est * math.sin(angle)
    miss = 0.0 if is_idle(tgt) else dist(aimed_x, aimed_y, tx, ty)
    return angle, t_est, miss

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


def estimate_capture_window_bonus(world, src, target, ships):
    """
    For rotating targets: sample 13 nearby future positions and reward
    targets that have wide sun-free intercept windows.
    Returns a bonus score 0..8.0; 0.0 for idle (non-rotating) targets.
    """
    if is_idle(target):
        return 0.0
    safe_count = 0
    for offset in range(-6, 7):
        fx, fy = predict_pos(target, world.ang_vel, offset)
        if not hits_sun(src.x, src.y, fx, fy):
            safe_count += 1
    return (safe_count / 13.0) * 8.0


def radius_class(p):
    if p.radius <= SMALL_RADIUS:
        return "SMALL"
    if p.radius >= LARGE_RADIUS:
        return "LARGE"
    return "MEDIUM"


def is_static_planet(p):
    return is_idle(p)


def rotating_target_approach_score(source, target, world=None):
    """
    Positive when a rotating target is moving toward source over the next
    5/10/15 turns; negative when it is pulling away.
    """
    if source is None or target is None or is_static_planet(target):
        return 0.0
    ang_vel = world.ang_vel if world is not None else 0.0
    current = dp(source, target)
    score = 0.0
    for horizon, weight in ((5, 1.4), (10, 1.0), (15, 0.7)):
        tx, ty = predict_pos(target, ang_vel, horizon)
        future = dist(source.x, source.y, tx, ty)
        score += (current - future) * weight
    return score


def _rotating_planet_moving_toward_target(world, rotating_planet):
    """Best non-owned planet the rotating body is geometrically approaching."""
    if rotating_planet is None or is_static_planet(rotating_planet):
        return None
    current_pos = (rotating_planet.x, rotating_planet.y)
    future_pos = predict_pos(rotating_planet, world.ang_vel, 10)
    best = None
    for target in world.normal_planets:
        if target.id == rotating_planet.id or target.owner == world.player or world.is_comet(target):
            continue
        now = dist(current_pos[0], current_pos[1], target.x, target.y)
        later = dist(future_pos[0], future_pos[1], target.x, target.y)
        closing = now - later
        if closing <= 0:
            continue
        item = (closing, -now, int(target.production), -int(target.ships), target.id, target)
        if best is None or item > best:
            best = item
    return best[-1] if best is not None else None


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
    Uses launch-adjusted arrivals; final tight sync windows are enforced when
    proposals are synchronized, committed, and released.
    Returns (ok, reason).
    """
    if not planned:
        return False, "no sources"
    total = sum(s for _, s, _, _ in planned)
    eta_vals = [
        max(0, _proposal_launch_step(world, launch_step) - world.step) + float(e)
        for _, _, launch_step, e in planned
    ]
    spread = max(eta_vals) - min(eta_vals) if len(eta_vals) >= 2 else 0.0
    max_eta = max(eta_vals)
    planning_slack = max(ETA_SYNC_WINDOW, BREACH_ETA_SYNC)
    if tgt.owner == -1 and spread > planning_slack:
        return False, f"neutral spread={spread:.1f}>{planning_slack}"
    if tgt.owner not in (-1, world.player) and spread > planning_slack:
        return False, f"enemy spread={spread:.1f}>{planning_slack}"
    eval_turn = max(1, int(math.ceil(max_eta)))
    extra = tuple(
        (max(1, int(math.ceil(max(0, _proposal_launch_step(world, ls) - world.step) + e))), world.player, int(s))
        for _, s, ls, e in planned if int(s) > 0
    )
    owner_after, _ = world.projected_state(tgt.id, eval_turn, extra_arrivals=extra)
    if owner_after != world.player:
        return False, f"won't flip at t={eval_turn}"
    return True, ""


def _sync_release_exempt(mission_type, reason=""):
    mission_type = canonical_mission_type(mission_type)
    reason_l = (reason or "").lower()
    if mission_type in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK", "DOOMED_EVACUATION", "FINISH_ZERO_CAPTURE", "FINAL_DRAIN"):
        return True
    return any(token in reason_l for token in ("emergency", "urgent", "save_under_attack", "finish_zero"))


def _sync_window_for_mission(world, target, mission_type, reason=""):
    mission_type = canonical_mission_type(mission_type)
    reason_l = (reason or "").lower()
    if _sync_release_exempt(mission_type, reason):
        return None
    if any(token in reason_l for token in ("staging", "two_stage", "relay", "backup_to_staging")):
        return STAGING_SYNC_WINDOW
    if target is not None and target.owner not in (-1, world.player):
        return ENEMY_ATTACK_SYNC_WINDOW
    if mission_type in ("SYNC_ATTACK", "BREACH_KILL", "COLLAPSE"):
        return ENEMY_ATTACK_SYNC_WINDOW
    if mission_type in ("HIGH_VALUE_NEUTRAL_RACE", "LOCAL_PRODUCTION_CAPTURE", "SNIPE_NEUTRAL"):
        return CONTESTED_NEUTRAL_SYNC_WINDOW
    if target is not None and target.owner == -1:
        contested = (
            int(target.production) >= LOCAL_PRODUCTION_MIN_PROD
            or world.enemy_incoming_to_targets.get(target.id, 0) > 0
            or is_contested_neutral(world, target)
        )
        return CONTESTED_NEUTRAL_SYNC_WINDOW if contested else NORMAL_NEUTRAL_SYNC_WINDOW
    return NORMAL_NEUTRAL_SYNC_WINDOW


def _planned_arrival_steps(world, planned_sources):
    return [
        _proposal_launch_step(world, launch_step) + max(1, int(math.ceil(eta)))
        for _src_id, _ships, launch_step, eta in (planned_sources or [])
    ]


def _sync_window_ok(world, target, mission_type, planned_sources, reason=""):
    if len(planned_sources or []) < 2:
        return True, 0.0, _sync_window_for_mission(world, target, mission_type, reason)
    window = _sync_window_for_mission(world, target, mission_type, reason)
    if window is None:
        return True, 0.0, None
    arrivals = _planned_arrival_steps(world, planned_sources)
    spread = max(arrivals) - min(arrivals) if arrivals else 0.0
    return spread <= window, float(spread), window


def _sync_managed_group(prop):
    if prop is None or len(getattr(prop, "planned_sources", []) or []) < 2:
        return False
    if _sync_release_exempt(getattr(prop, "kind", ""), getattr(prop, "reason", "")):
        return False
    reason_l = (getattr(prop, "reason", "") or "").lower()
    return prop.kind in OFFENSIVE_MISSIONS or any(
        token in reason_l for token in ("staging", "two_stage", "relay", "backup_to_staging")
    )


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
    INITIAL_EXPANSION = "INITIAL_EXPANSION"
    MIDGAME_CONTROL   = "MIDGAME_CONTROL"
    DOMINANCE_PHASE   = "DOMINANCE_PHASE"
    COLLAPSE_PHASE    = "COLLAPSE_PHASE"
    OPENING_EXPANSION = "OPENING_EXPANSION"   # my_pct < 12% or <= 3 planets
    LOCAL_SWEEP       = "LOCAL_SWEEP"         # 12%–28%, many neutrals remain
    EXPANSION_CONTROL = "EXPANSION_CONTROL"   # 28%–45%, cluster stable
    CONTACT           = "CONTACT"             # frontier reached or enemy near
    COLLAPSE          = "COLLAPSE"            # dominant control, finish opponent
    CONSOLIDATE       = "CONSOLIDATE"         # production lead in late game → hold, don't over-expand


class BoardPhase:
    """Primary strategic phase driven by planet occupancy and production, not step number."""
    LOW_CONTROL_MODE = "LOW_CONTROL_MODE"    # < 20% occupancy: forced expansion
    BUILD_CHAIN_MODE = "BUILD_CHAIN_MODE"    # 20%-50%: keep expanding, defend, punish drains
    PRESSURE_MODE    = "PRESSURE_MODE"       # > 50% OR my_prod > enemy_prod * 1.25: attack/collapse


def _is_low_control(world):
    """BoardPhase.LOW_CONTROL_MODE: my planet occupancy < 20%."""
    return len(world.my_planets) / max(1, len(world.normal_planets)) < 0.20


def _is_pressure_mode(world):
    """BoardPhase.PRESSURE_MODE: my occupancy > 50% OR production lead > 25%."""
    occ = len(world.my_planets) / max(1, len(world.normal_planets))
    return occ > 0.50 or world.my_prod > world.enemy_prod * 1.25


def _is_build_chain(world):
    """BoardPhase.BUILD_CHAIN_MODE: between LOW_CONTROL and PRESSURE."""
    return not _is_low_control(world) and not _is_pressure_mode(world)


def board_phase_selected(world):
    if _is_low_control(world):
        phase = BoardPhase.LOW_CONTROL_MODE
    elif _is_pressure_mode(world):
        phase = BoardPhase.PRESSURE_MODE
    else:
        phase = BoardPhase.BUILD_CHAIN_MODE
    world.add_debug(f"BOARD_PHASE {phase} occ={len(world.my_planets)}/{max(1, len(world.normal_planets))}")
    return phase


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
    planned_sources: list   # [(src_id, ships, launch_step, eta), ...]; no cached trajectory
    eta_min: float
    eta_max: float
    reason: str
    priority_tier: str = "IMPORTANT"

    def __post_init__(self):
        normalized = []
        for item in self.planned_sources:
            if len(item) >= 4:
                src_id = item[0]
                ships = item[1]
                third = item[2]
                # Older builders passed (src_id, ships, angle, eta). Angles are
                # deliberately discarded here so execution must aim just in time.
                launch_step = third if isinstance(third, int) else 0
                normalized.append((int(src_id), int(ships), int(launch_step), float(item[3])))
            elif len(item) == 3:
                normalized.append((int(item[0]), int(item[1]), int(item[2]), 0.0))
            elif len(item) == 2:
                normalized.append((int(item[0]), int(item[1]), 0, 0.0))
        self.planned_sources = normalized

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
    priority_tier: str = "IMPORTANT"


MISSION_TYPE_ALIASES = {
    "DEFEND": "DEFEND_HOLD",
    "REINFORCE": "REINFORCE_CAPTURE",
    "HOLD_CAPTURE": "REINFORCE_CAPTURE",
    "CAPTURE_HIGH_PROD_NEUTRAL": "LOCAL_PRODUCTION_CAPTURE",
    "REINFORCE_FRONTIER_HUB": "REINFORCE_CAPTURE",
    "RECAPTURE_LOST_PLANET": "RECAPTURE_LOST",
    "CORE_CHAIN_RECOVERY": "CORE_CHAIN_RECOVERY",
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
    "CORE_CHAIN_RECOVERY",
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
    "FINISH_ZERO_CAPTURE",
    "DOOMED_EVACUATION",
}

REINFORCEMENT_MISSIONS = {
    "DEFEND_HOLD",
    "SAVE_UNDER_ATTACK",
    "REINFORCE_CAPTURE",
}

OFFENSIVE_MISSIONS = {
    "CORE_CHAIN_RECOVERY",
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
_opponent_model_memory: dict = {}   # player -> owner -> previous opponent snapshot
_midgame_dominance_last: dict = {}  # player -> last step a dominance attack launched
_shot_history: dict = {}            # player -> list of recent launched shot records
_bad_shot_patterns: dict = {}       # player -> pattern -> {fails, banned_until}
_beam_expansion_last: dict = {}     # player -> last step beam/campaign launched a capture
_last_capture_step: dict = {}       # player -> last step any capture was committed
_midgame_conversion_memory: dict = {}  # player -> planet-count growth/stall memory
_recent_launch_history: dict = {}      # player -> recent launch destination/mode records
_adaptive_meta_controllers: dict = {}  # player -> AdaptiveMetaController
_pending_mission_launches: dict = {}   # player -> queued source launches awaiting JIT aim
_pending_delayed_missions: dict = {}   # player -> mission_id -> delayed synchronized burst
_expansion_obligation_cooldown: dict = {}  # player -> step when obligation cooldown expires
_staging_controller_memory: dict = {}  # player -> {active_until, release_until, started_step}
_territory_conversion_history: dict = {}  # player -> recent per-turn planet gain/loss records


def canonical_mission_type(kind):
    mission_type = MISSION_TYPE_ALIASES.get(kind, kind)
    return mission_type if mission_type in MISSION_TYPES else "SYNC_ATTACK"


def mission_allows_small_packet(mission_type):
    return False


def _chaotic_endgame_packet_allowed(world, src, tgt, mission_type, ships):
    if world is None or src is None or tgt is None:
        return False
    if int(ships) != 5:
        return False
    if not (world.step > 430 or world.remaining < 60):
        world.add_debug(f"SMALL_PACKET_REJECTED_NOT_ENDGAME mission={mission_type} ships={ships}")
        return False
    if mission_type not in ("FINAL_DRAIN", "FINISH_ZERO_CAPTURE"):
        return False
    if not (int(tgt.ships) <= 3 or len(world.enemy_planets) <= 2):
        return False
    if world.real_incoming_threat(src)["deficit"] > 0:
        return False
    world.add_debug("CHAOTIC_ENDGAME_PACKET_RELAXATION_ACTIVE")
    world.add_debug(
        f"SMALL_PACKET_ENDGAME_ALLOWED mission={mission_type} src=p{src.id} target=p{tgt.id} ships={ships}"
    )
    return True


def valid_packet_size(mission_type, ships, world=None, src=None, tgt=None):
    ships = int(ships)
    if ships < 5 or ships % SEND_GRANULARITY != 0:
        return False
    if ships >= MIN_SEND_SHIPS:
        return True
    return _chaotic_endgame_packet_allowed(world, src, tgt, mission_type, ships)


def _mission_priority_tier(mission_type, reason="", default="IMPORTANT"):
    mission_type = canonical_mission_type(mission_type)
    reason_l = (reason or "").lower()
    if mission_type in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK", "DOOMED_EVACUATION"):
        return "CRITICAL"
    if mission_type in ("COLLAPSE", "BREACH_KILL", "FINISH_ZERO_CAPTURE", "CORE_CHAIN_RECOVERY"):
        return "CRITICAL"
    if any(token in reason_l for token in ("emergency", "under_attack", "frontline", "verified winning", "collapse")):
        return "CRITICAL"
    if "hub_security_buffer" in reason_l:
        return "IMPORTANT"
    if any(token in reason_l for token in ("staging", "relay", "rally", "backup_to_staging", "zero_capital")):
        return "FLEXIBLE"
    if mission_type == "REINFORCE_CAPTURE":
        return "FLEXIBLE"
    return default


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

    def create(self, mission_type, target_id, source_ids, required_ships, expected_arrival_steps, reason, priority_tier="IMPORTANT"):
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
                entry.priority_tier = _mission_priority_tier(mission_type, reason or entry.reason, priority_tier)
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
            priority_tier=_mission_priority_tier(mission_type, reason, priority_tier),
        )
        self.entries[mission_id] = entry
        self.world.add_debug(
            f"MISSION_SELECT {mission_type} id={mission_id} target=p{target_id} "
            f"sources={list(source_ids)} required={int(required_ships)} "
            f"eta_spread={(max(expected_arrival_steps) - min(expected_arrival_steps)) if expected_arrival_steps else 0:.1f} "
            f"tier={entry.priority_tier} "
            f"reason={reason}"
        )
        return mission_id

    def create_from_proposal(self, prop):
        expected_arrivals = []
        for _src_id, _ships, launch_step, eta in prop.planned_sources:
            launch = _proposal_launch_step(self.world, launch_step)
            expected_arrivals.append(max(1, launch - self.world.step + int(math.ceil(eta))))
        if prop.kind in OFFENSIVE_MISSIONS and expected_arrivals:
            self.world.add_debug(
                f"MISSION_LEDGER_ARRIVAL_TIMING mission={prop.kind} target_id={prop.target_id} "
                f"arrival_steps={[self.world.step + a for a in expected_arrivals]} "
                f"delays={[max(0, _proposal_launch_step(self.world, ls) - self.world.step) for _sid, _s, ls, _e in prop.planned_sources]}"
            )
        return self.create(
            prop.kind,
            prop.target_id,
            [src_id for src_id, _, _, _ in prop.planned_sources],
            prop.required_ships,
            expected_arrivals or [eta for _, _, _, eta in prop.planned_sources],
            prop.reason,
            prop.priority_tier,
        )

    def cancel_flexible(self, reason):
        cancelled = []
        for entry in list(self.entries.values()):
            if entry.status not in ("planned", "active"):
                continue
            tier = getattr(entry, "priority_tier", _mission_priority_tier(entry.mission_type, entry.reason))
            if tier != "FLEXIBLE":
                continue
            entry.status = "invalidated"
            cancelled.append(entry.mission_id)
            self.world.add_debug(
                f"FLEXIBLE_MISSION_CANCELLED id={entry.mission_id} mission={entry.mission_type} "
                f"target=p{entry.target_id} reason={reason}"
            )
        if cancelled:
            pending = _pending_mission_launches.get(self.world.player, [])
            _pending_mission_launches[self.world.player] = [
                rec for rec in pending if rec.get("mission_id") not in set(cancelled)
            ]
            self.world.add_debug(
                f"TACTICAL_INTERRUPT_OVERRIDES_LEDGER cancelled={cancelled} reason={reason}"
            )
        return cancelled

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
        self.staging_reserved_ships = {}
        for rec in _pending_mission_launches.get(self.player, []):
            if "two_stage_final" not in (rec.get("reason") or ""):
                continue
            if int(rec.get("launch_step", self.step)) < self.step:
                continue
            sid = int(rec.get("source_id", -1))
            self.staging_reserved_ships[sid] = self.staging_reserved_ships.get(sid, 0) + int(rec.get("ships", 0))
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
        self._update_shot_memory()
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
            if hits_sun(fl.x, fl.y, tgt.x, tgt.y):
                self.add_debug(
                    f"SUN_DOOMED_FLEET_IGNORED owner={fl.owner} ships={int(fl.ships)} target=p{tgt.id}"
                )
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
            if zero_capital_backline_safe(self, p):
                keep = 0
                self.add_debug(f"ZERO_CAPITAL_BACKLINE_DRAIN p{p.id} reserve=0")
            else:
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
        incoming_threat_count = sum(1 for p in self.my_planets if self.strategic_incoming_threat(p)["deficit"] > 0)
        control_ratio = len(self.my_planets) / max(1, len(self.normal_planets))
        control_final = (
            self.step >= FINAL_STEPS
            or self.remaining < 45
            or (control_ratio >= PHASE_COLLAPSE_MIN and bool(self.enemy_planets))
            or (control_ratio >= PHASE_MIDGAME_MAX and self.my_total_ships > self.enemy_total_ships * 1.5)
            or (control_ratio >= PHASE_MIDGAME_MAX and self.my_prod > self.enemy_prod * 1.4)
        )
        self.features = {
            "prod_ratio": self.my_prod / max(1, self.enemy_prod),
            "ship_ratio": self.my_total_ships / max(1, self.enemy_total_ships),
            "control_ratio": control_ratio,
            "leader_ahead": self.leader is not None and self.leader_score > self.my_score * 1.22,
            "neutral_count": len(self.neutral_planets),
            "high_neutral_count": high_neutrals,
            "nearest_enemy": nearest_enemy,
            "enemy_avg_ships": enemy_avg_ships,
            "incoming_threat_count": incoming_threat_count,
            "ahead": self.my_score > max(1, self.leader_score) * 1.18 or self.my_prod > max(1, self.enemy_prod) * 1.25,
            "behind": self.my_score * 1.25 < max(1, self.leader_score) or self.my_prod * 1.3 < max(1, self.enemy_prod),
            "late": _is_pressure_mode(self) or self.remaining < 120,
            "final": control_final,
        }

    def aim(self, src, tgt, ships):
        key = (src.id, tgt.id, int(ships), int(self.step))
        if key not in self.shot_cache:
            self.shot_cache[key] = aim_valid(src, tgt, self.ang_vel, int(ships))
        return self.shot_cache[key]

    def _shot_pattern_key(self, src, tgt, angle):
        return (int(src.id), int(tgt.id), round(float(angle), 1))

    def _update_shot_memory(self):
        history = _shot_history.setdefault(self.player, [])
        bad = _bad_shot_patterns.setdefault(self.player, {})
        remaining = []
        for rec in history:
            if self.step < rec["arrival_step"]:
                remaining.append(rec)
                continue
            tgt = self.planet_by_id.get(rec["target_id"])
            if tgt is None:
                continue
            if rec.get("offensive") and tgt.owner != self.player:
                pattern = rec["pattern"]
                entry = bad.setdefault(pattern, {"fails": 0, "banned_until": -1})
                entry["fails"] += 1
                if entry["fails"] >= 3:
                    entry["banned_until"] = self.step + 30
                    self.add_debug(
                        f"REPEATED_MISS_TARGET_BANNED src=p{pattern[0]} target=p{pattern[1]} angle={pattern[2]}"
                    )
                else:
                    self.add_debug(
                        f"RECENT_MISS_PATTERN_DETECTED src=p{pattern[0]} target=p{pattern[1]} angle={pattern[2]}"
                    )
            elif rec.get("offensive"):
                pattern = rec["pattern"]
                if pattern in bad:
                    bad[pattern]["fails"] = max(0, bad[pattern]["fails"] - 1)
        _shot_history[self.player] = [
            rec for rec in remaining if self.step - rec.get("launch_step", self.step) <= 80
        ]
        for pattern in list(bad.keys()):
            if bad[pattern].get("banned_until", -1) < self.step and bad[pattern].get("fails", 0) <= 0:
                del bad[pattern]

    def aim_confidence_check(self, src, tgt, ships, mission_type):
        self.add_debug(f"AIM_CONFIDENCE_CHECK src=p{src.id} target=p{tgt.id} ships={int(ships)}")
        angle, eta_guess, miss = aim_solution_metrics(src, tgt, self.ang_vel, int(ships))
        pattern = self._shot_pattern_key(src, tgt, angle)
        bad = _bad_shot_patterns.setdefault(self.player, {})
        entry = bad.get(pattern)
        if entry and entry.get("banned_until", -1) >= self.step:
            self.add_debug(f"REPEATED_BAD_SHOT_BLOCKED src=p{src.id} target=p{tgt.id} angle={pattern[2]}")
            self.add_debug(f"SAME_ANGLE_SPAM_REJECTED src=p{src.id} target=p{tgt.id}")
            self.add_debug(f"RETARGET_AFTER_FAILED_SHOT src=p{src.id} target=p{tgt.id}")
            return False, "repeated bad shot pattern"
        if is_idle(tgt):
            return True, ""
        margin = tgt.radius + HIT_MARGIN
        if eta_guess > 25:
            margin = tgt.radius + min(1.5, HIT_MARGIN)
        if eta_guess > 28:
            margin = tgt.radius + 1.0
        if miss > margin:
            self.add_debug(
                f"LOW_CONFIDENCE_AIM_REJECTED src=p{src.id} target=p{tgt.id} "
                f"eta={eta_guess:.1f} miss={miss:.2f} margin={margin:.2f}"
            )
            return False, "low confidence moving-target aim"
        self.add_debug(
            f"MOVING_TARGET_AIM_VERIFIED src=p{src.id} target=p{tgt.id} "
            f"eta={eta_guess:.1f} miss={miss:.2f}"
        )
        return True, ""

    def record_shot_launch(self, src, tgt, ships, angle, eta, mission_type):
        pattern = self._shot_pattern_key(src, tgt, angle)
        _shot_history.setdefault(self.player, []).append({
            "launch_step": self.step,
            "arrival_step": self.step + max(1, int(math.ceil(eta))),
            "source_id": src.id,
            "target_id": tgt.id,
            "angle": float(angle),
            "ships": int(ships),
            "pattern": pattern,
            "offensive": tgt.owner != self.player and mission_type in OFFENSIVE_MISSIONS,
        })
        launch_history = _recent_launch_history.setdefault(self.player, [])
        launch_history.append({
            "step": self.step,
            "source_id": src.id,
            "target_id": tgt.id,
            "target_owned": tgt.owner == self.player,
            "mission_type": mission_type,
            "ships": int(ships),
            "offensive": tgt.owner != self.player and mission_type in OFFENSIVE_MISSIONS,
        })
        _recent_launch_history[self.player] = [
            rec for rec in launch_history if self.step - rec.get("step", self.step) <= 50
        ][-80:]

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

    def strategic_incoming_threat(self, p, horizon=DEFENSE_ETA_HORIZON):
        """Like real_incoming_threat but ignores enemy fleets too small to flip the planet.
        Used only for strategic panic/phase decisions, never for actual defense."""
        friendly = 0
        enemy = 0
        flip_threshold = max(DECOY_FLEET_THRESHOLD, int(p.ships) // 4)
        for eta, owner, ships in self.arrivals_by_target.get(p.id, []):
            if eta > horizon:
                continue
            if owner == self.player:
                friendly += ships
            elif ships >= flip_threshold:
                enemy += ships
            else:
                self.add_debug(f"DECOY_FLEET_IGNORED_FOR_PANIC_ONLY target=p{p.id} ships={ships}")
        net = int(p.ships) + friendly - enemy
        return {"friendly": friendly, "enemy": enemy, "net": net, "deficit": max(0, DEFEND_NET - net) if enemy > friendly else 0}

    def nearest_enemy_distance(self, p):
        planet_dist = min((dp(p, e) for e in self.enemy_planets), default=999.0)
        fleet_dist = min((dist(p.x, p.y, f.x, f.y) for f in self.enemy_fleets), default=999.0)
        if fleet_dist < planet_dist:
            self.add_debug(f"ENEMY_FLEET_THREAT_DISTANCE_INCLUDED p{p.id} fleet_dist={fleet_dist:.1f}")
        return min(planet_dist, fleet_dist)

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
        if zero_capital_backline_safe(self, p):
            self.add_debug(f"ZERO_CAPITAL_BACKLINE_DRAIN p{p.id} reserve=0")
            return 0
        if hasattr(self, "keep_needed_map") and p.id in self.keep_needed_map:
            cap = max(20, int(p.ships * 0.55))
            keep = int(self.keep_needed_map[p.id])
            if is_storage_planet(p):
                keep = max(keep, int(p.ships * 0.65), 6 + int(p.production) * 2)
                self.add_debug(f"SMALL_STORAGE_RESERVE_HELD p{p.id} reserve={keep}")
                return keep
            return min(keep, cap)
        if _is_low_control(self):
            base = min(3, 1 + int(p.production))
        elif not _is_pressure_mode(self):
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
        cap = max(20, int(p.ships * (0.45 if _is_low_control(self) else 0.55)))
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
            need = _neutral_capture_fast_need(self, src, tgt, eta)
            if need <= 0:
                return 0
        else:
            need = _enemy_capture_need_with_eta(self, src, tgt, eta)
            if need <= 0:
                return 0
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
        small_start_escape = (
            _small_start_escape_mode(self, src)
            and "small_start_escape" in (mission_reason or "")
            and tgt is not None
            and _small_start_escape_target_value(self, tgt)
        )
        early_neutral_capture = _early_neutral_reserve_relaxation_allowed(self, src, tgt, mission_reason)
        if small_start_escape:
            reserve = min(reserve, 3)
            self.add_debug(f"SMALL_START_RESERVE_RELAXED p{src.id} reserve={reserve}")
        elif early_neutral_capture:
            reserve = min(reserve, _early_neutral_min_source_reserve(self, src))
            self.add_debug(
                f"EARLY_NEUTRAL_RESERVE_RELAXED src=p{src.id} target=p{tgt.id} reserve={reserve}"
            )
            self.add_debug(
                f"FULL_SOURCE_ALLOWED_FOR_NEUTRAL_CAPTURE src=p{src.id} target=p{tgt.id} ships={int(ships)}"
            )
        elif (
            (
                "expansion_campaign" in (mission_reason or "")
                or "expansion_obligation" in (mission_reason or "")
            )
            and (len(self.my_planets) < 6 or not _is_pressure_mode(self))
            and threat["deficit"] <= 0
        ):
            reserve = min(reserve, max(2, int(src.production) + 3))
            self.add_debug(f"EARLY_RESERVE_RELAXED_FOR_EXPANSION src=p{src.id} reserve={reserve}")
        if remaining < reserve and not critical and not evacuating:
            return False, f"source unsafe: below reserve {remaining}<{reserve}"
        if self.is_recently_reinforced(src) and mission_type in OFFENSIVE_MISSIONS and not evacuating:
            return False, "source unsafe: recently reinforced cooldown"
        if is_storage_planet(src) and mission_type in OFFENSIVE_MISSIONS and not evacuating and not small_start_escape and not early_neutral_capture:
            allowed_storage_release = mission_type in (
                "CORE_CHAIN_RECOVERY", "RECAPTURE_LOST", "FINISH_ZERO_CAPTURE", "FINAL_DRAIN"
            )
            target_role = radius_class(tgt) if tgt is not None else "SMALL"
            launchpad_escape = (
                mission_type in ("CAPTURE_NEUTRAL", "LOCAL_PRODUCTION_CAPTURE", "HIGH_VALUE_NEUTRAL_RACE")
                and tgt is not None
                and target_role in ("MEDIUM", "LARGE")
                and dp(src, tgt) <= CAPTURE_OPP_MAX_DIST
            )
            local_support = tgt is not None and dp(src, tgt) <= CHEAP_RECAPTURE_LOCAL_DIST
            if allowed_storage_release:
                if mission_type in ("CORE_CHAIN_RECOVERY", "RECAPTURE_LOST"):
                    self.add_debug(f"SMALL_STORAGE_RELEASE_RECAPTURE src=p{src.id} target=p{getattr(tgt, 'id', '?')}")
            elif launchpad_escape:
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

    def valid_fleet_launch(self, src, tgt, ships, mission_type, mission_entry=None, planned_sources=None, mission_reason="", validate_aim=True):
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

        if validate_aim:
            aim_ok, aim_reason = self.aim_confidence_check(src, tgt, ships, mission_type)
            if not aim_ok:
                return False, aim_reason

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
            if tgt.owner == self.player and mission_type not in ("CORE_CHAIN_RECOVERY", "RECAPTURE_LOST"):
                return False, "mission invalidated: target already mine"
            owner_at, _, _ = self.target_owner_at_arrival(tgt, eta, planned=planned_arrivals)
            if mission_type in ("CORE_CHAIN_RECOVERY", "RECAPTURE_LOST") and tgt.owner == self.player and owner_at == self.player:
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
                    ok_sync, spread, window = _sync_window_ok(self, tgt, mission_type, planned_sources, mission_reason)
                    if not ok_sync:
                        self.add_debug(
                            f"SYNC_WINDOW_TOO_LOOSE_REJECTED target=p{tgt.id} spread={spread:.1f} window={window}"
                        )
                        return False, "trickle blocked: eta spread"
            if self.incoming_to_targets.get(tgt.id, 0) >= self.required_ships_to_capture(tgt, src):
                return False, "target already doomed"
        return True, ""

    def capture_due_soon(self, horizon=15):
        for pid, arrivals in self.arrivals_by_target.items():
            tgt = self.planet_by_id.get(pid)
            if tgt is None or tgt.owner == self.player or self.is_comet(tgt):
                continue
            soon = [
                (eta, ships)
                for eta, owner, ships in arrivals
                if owner == self.player and eta <= horizon
            ]
            if not soon:
                continue
            incoming = sum(ships for _eta, ships in soon)
            src = min(self.my_planets, key=lambda p: dp(p, tgt), default=None)
            if src is not None and incoming >= self.required_ships_to_capture(tgt, src):
                return True
        return False

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
        mission_entry = self.mission_ledger.get(mission_id)
        mission_reason = mission_entry.reason if mission_entry is not None else ""

        if (
            getattr(self, "midgame_conversion_active", False)
            and mission_type == "REINFORCE_CAPTURE"
            and tgt is not None
            and tgt.owner == self.player
            and self.real_incoming_threat(tgt)["deficit"] <= 0
            and not self.capture_due_soon(15)
            and "relay_backup_to_staging" not in mission_reason
            and "zero_capital_backline_drain" not in mission_reason
        ):
            self.add_debug(
                f"OWN_PLANET_ROTATION_BLOCKED src=p{src.id} target=p{tgt.id} "
                f"ships={ships} reason=midgame_conversion"
            )
            self.add_debug("NO_MORE_PASSIVE_REINFORCE_LOOP")
            return False

        if ships > available:
            self.add_debug(
                f"COMMIT_REJECT_INVALID_PACKET mission={mission_type} src=p{src.id} "
                f"ships={ships} available={available} reason=unavailable"
            )
            return False
        if not valid_packet_size(mission_type, ships):
            if ships < MIN_SEND_SHIPS:
                self.add_debug(
                    f"TINY_PACKET_REJECTED mission={mission_type} src=p{src.id} ships={ships}"
                )
            self.add_debug(
                f"COMMIT_REJECT_INVALID_PACKET mission={mission_type} src=p{src.id} "
                f"ships={ships} reason=packet_size"
            )
            self.add_debug(
                f"INVALID_PACKET_REJECTED mission={mission_type} src=p{src.id} ships={ships}"
            )
            return False
        self.add_debug(f"COMMIT_VALID_PACKET mission={mission_type} src=p{src.id} ships={ships}")
        self.add_debug(f"UNIVERSAL_PACKET_RULE_APPLIED mission={mission_type} src=p{src.id} ships={ships}")

        ok_launch, reason = self.valid_fleet_launch(
            src, tgt, ships, mission_type, mission_entry=mission_entry,
            planned_sources=planned_sources,
            mission_reason=mission_reason,
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
        self.record_shot_launch(src, tgt, ships, angle, self.eta(src, tgt, ships), mission_type)
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
      2. Always allow: recently-mine, actively-threatening, core recovery,
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
    if mission_type in ("CORE_CHAIN_RECOVERY", "RECAPTURE_LOST", "FINISH_ZERO_CAPTURE"):
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: active agent uses build_chain_retrigger_response()."""
    return None
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
        allow_small_packets=False,
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: panic recapture is disabled in the chain agent."""
    return False
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
        world.add_debug("COUNTERATTACK_PREFERRED_OVER_RECAPTURE")
        coordinate_missions(world, counter_props, moves, fleet_ratio, deadline)
        return len(moves) > before_moves

    if recapture_props:
        if counter_props:
            rec_send = min(sum(s for _, s, _, _ in p.planned_sources) for p in recapture_props)
            ctr_send = sum(s for _, s, _, _ in counter_props[0].planned_sources)
            ctr_tgt = world.planet_by_id.get(counter_props[0].target_id)
            rec_tgt = world.planet_by_id.get(recapture_props[0].target_id)
            counter_holdable = True
            if ctr_tgt is not None and counter_props[0].planned_sources:
                counter_holdable = world.can_hold_after_capture(
                    ctr_tgt,
                    max(eta for _sid, _s, _a, eta in counter_props[0].planned_sources),
                    ctr_send,
                )
            counter_weak = ctr_tgt is not None and int(ctr_tgt.ships) <= max(18, int(ctr_tgt.production) * 5)
            counter_drained = ctr_tgt is not None and _prev_ships.get(ctr_tgt.id, int(ctr_tgt.ships)) > int(ctr_tgt.ships) + max(8, int(ctr_tgt.production) * 2)
            counter_near = ctr_tgt is not None and min((dp(m, ctr_tgt) for m in world.my_planets), default=999.0) <= CHEAP_RECAPTURE_LOCAL_DIST + 15
            counter_high_prod = ctr_tgt is not None and int(ctr_tgt.production) >= max(3, int(getattr(rec_tgt, "production", 0)))
            rec_high_value = rec_tgt is not None and int(rec_tgt.production) >= 3
            rec_extremely_cheap = rec_send <= max(MIN_SEND_SHIPS, ctr_send * 0.55)
            counter_viable = counter_holdable and (counter_weak or counter_drained or counter_near or counter_high_prod)
            if counter_viable and not (rec_high_value or rec_extremely_cheap):
                world.add_debug("COUNTERATTACK_PREFERRED_OVER_RECAPTURE")
                coordinate_missions(world, counter_props, moves, fleet_ratio, deadline)
                return len(moves) > before_moves
            world.add_debug("CHEAP_RECAPTURE_SELECTED_ONLY_IF_HIGH_VALUE_OR_CHEAP")
        coordinate_missions(world, recapture_props[:1], moves, fleet_ratio, deadline)
        return len(moves) > before_moves

    world.add_debug("CHEAP_RECAPTURE_SKIP_COUNTERATTACK")
    if counter_props:
        world.add_debug("COUNTERATTACK_PREFERRED_OVER_RECAPTURE")
        coordinate_missions(world, counter_props, moves, fleet_ratio, deadline)
        return len(moves) > before_moves

    world.add_debug("NO_SOURCE_LOCK_WITHOUT_SELECTED_MISSION")
    return False


def choose_strategy_mode(world, idle_turns):
    """Choose strategic mode for legacy proposal layers that still feed the active agent."""
    f = world.features
    if f["final"]:
        return StrategyMode.FINAL_DRAIN
    if (world.is_four_player and _is_low_control(world)
            and world.neutral_planets and f["incoming_threat_count"] == 0):
        return StrategyMode.FOUR_PLAYER_EXPAND_FIRST
    if _is_low_control(world) or len(world.my_planets) < 3:
        return StrategyMode.OPENING_TEMPO
    if idle_turns >= 9 and world.my_total_ships > 250:
        return StrategyMode.FORCE_WAVE
    if f["leader_ahead"]:
        return StrategyMode.ANTI_LEADER
    if f["behind"]:
        return StrategyMode.BEHIND_STEAL
    if f["ahead"] and (_is_pressure_mode(world) or not world.neutral_planets):
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


def planet_control_ratio(world):
    ratio = len(world.my_planets) / max(1, len(world.normal_planets))
    world.add_debug(f"PLANET_CONTROL_RATIO {ratio:.2f}")
    return ratio


def enemy_planets_total(world):
    return len([p for p in world.enemy_planets if not world.is_comet(p)])


def collapse_phase_triggered(world, control_ratio=None):
    control_ratio = planet_control_ratio(world) if control_ratio is None else control_ratio
    return (
        control_ratio >= PHASE_COLLAPSE_MIN
        or enemy_planets_total(world) <= 6
        or world.my_prod > world.enemy_prod * 1.6
        or world.my_total_ships > world.enemy_total_ships * 1.7
    )


def control_phase_selected(world):
    control_ratio = planet_control_ratio(world)
    if collapse_phase_triggered(world, control_ratio):
        phase = ControlPhase.COLLAPSE_PHASE
        world.add_debug("COLLAPSE_PHASE_BY_PLANET_CONTROL")
    elif control_ratio >= PHASE_MIDGAME_MAX:
        phase = ControlPhase.DOMINANCE_PHASE
        world.add_debug("DOMINANCE_PHASE_TRIGGERED")
    elif control_ratio >= PHASE_INITIAL_MAX:
        phase = ControlPhase.MIDGAME_CONTROL
        world.add_debug("MIDGAME_CONTROL_PHASE")
    else:
        phase = ControlPhase.INITIAL_EXPANSION
        world.add_debug("INITIAL_EXPANSION_PHASE")
    world.add_debug(f"CONTROL_PHASE_SELECTED {phase}")
    return phase, control_ratio


def control_based_final_phase(world, phase=None, control_ratio=None):
    if control_ratio is None:
        control_ratio = len(world.my_planets) / max(1, len(world.normal_planets))
    return (
        world.remaining < 45
        or (control_ratio >= PHASE_COLLAPSE_MIN and enemy_planets_total(world) > 0)
        or (control_ratio >= PHASE_MIDGAME_MAX and world.my_total_ships > world.enemy_total_ships * 1.5)
        or (control_ratio >= PHASE_MIDGAME_MAX and world.my_prod > world.enemy_prod * 1.4)
        or phase == ControlPhase.COLLAPSE_PHASE
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

    # Phase Transition Lock: production lead in late game → consolidate holdings
    # rather than launching new risky expansion missions.
    if (world.step >= LATE_GAME_STEPS
            and world.my_prod > world.enemy_prod
            and my_pct >= PHASE_EXPAND_PCT):
        world.add_debug(
            f"PHASE_TRANSITION_LOCK step={world.step} my_prod={world.my_prod} "
            f"enemy_prod={world.enemy_prod} my_pct={my_pct:.2f}"
        )
        return ControlPhase.CONSOLIDATE

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
    strategic_position_bonus += _rotating_source_static_target_bonus(world, src, tgt)
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
        if _planet_role(tgt) == ROLE_STORAGE and _small_radius_target_allowed(world, tgt, src):
            enemy_core_bonus += 85.0
        if mode in (StrategyMode.COLLAPSE,):
            enemy_core_bonus += 85.0
        if len(world.enemy_planets) <= 6:
            enemy_core_bonus += 45.0

    enemy_t = min((world.eta(e, tgt, max(1, int(e.ships))) for e in world.enemy_planets), default=999.0)
    overextension_penalty = 0.0
    if world.is_four_player and _is_low_control(world) and tgt.owner not in (-1, world.player):
        overextension_penalty += 140.0
    if world.cluster_distance(tgt) > MIDGAME_CONTEST_MAX_DIST:
        overextension_penalty += (world.cluster_distance(tgt) - MIDGAME_CONTEST_MAX_DIST) * 3.0
    if tgt.owner == -1 and enemy_t < eta - 3.0:
        overextension_penalty += 80.0

    travel_penalty = eta * 5.0 + d * 1.2
    ship_cost_penalty = need * (1.15 if prod >= 4 else 1.55)
    window_bonus = estimate_capture_window_bonus(world, src, tgt, need)
    return (
        production_value_bonus + chain_bonus + strategic_position_bonus
        + recapture_bonus + enemy_core_bonus + window_bonus
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
    buffer = 1 if _is_low_control(world) else min(4, max(1, int(tgt.production)))
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
        if not _is_pressure_mode(world) and not validate_initial_target_choice(world, src, tgt):
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
        close_bonus = max(0.0, NEAREST_LOCK_DIST - closest_dist) * (5.0 if not _is_pressure_mode(world) else 3.0)
        bridge_bonus = route_bridge_value(world, src, tgt)
        weak_bonus = max(0.0, 24.0 - need) * 2.0
        static_bonus = 18.0 if is_idle(tgt) else 0.0
        contest_risk = max(0.0, closest_eta - enemy_eta + 1.0) if enemy_eta < 999 else 0.0
        missing_penalty = 120.0 if grouped_pool < need else 0.0
        early_distance_weight = 16.0 if not _is_pressure_mode(world) or len(world.my_planets) < 4 else 9.0
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
        if int(tgt.production) <= 1 and _is_low_control(world) and len(world.my_planets) > 1:
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
        and _is_low_control(world)
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: active agent uses chain-aware fallback only."""
    return False
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
        if not _is_pressure_mode(world):
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
        world.add_debug(f"DEFEND_BEFORE_ESCAPE target=p{tgt.id} deficit={threat['deficit']}")

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
            world.add_debug(f"ESCAPE_ONLY_IF_UNSAVABLE target=p{tgt.id}")
            world.add_debug(f"SKIP SAVE_UNDER_ATTACK p{tgt.id} step={world.step} reason=no_planned_sources need={need}")
            continue

        full_tl = world.simulate_planet_timeline(
            tgt,
            DEFENSE_ETA_HORIZON,
            planned=[(eta, world.player, ships) for _, ships, _, eta in planned],
        )
        if full_tl["fall_turn"] is not None:
            world.mark_doomed(tgt)
            world.add_debug(f"ESCAPE_ONLY_IF_UNSAVABLE target=p{tgt.id}")
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
        world.add_debug(f"COUNTERATTACK_AFTER_DEFENSE target=p{tgt.id}")
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
            world.add_debug(f"DEFEND_BEFORE_ESCAPE target=p{src.id}")
            continue
        world.add_debug(f"ESCAPE_ONLY_IF_UNSAVABLE target=p{src.id} fall={fall_turn}")
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
                if tgt.owner not in (-1, world.player):
                    world.add_debug(f"DOOMED_EVACUATION_LAST_RESORT countercapture=p{tgt.id}")
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
            world.add_debug(f"DOOMED_EVACUATION_LAST_RESORT src=p{src.id} target=p{chosen.id}")
        angle, ok = world.aim(src, chosen, send)
        if not ok:
            continue
        eta = world.eta(src, chosen, send)
        proposals.append(MissionProposal(
            kind="DOOMED_EVACUATION",
            target_id=chosen.id,
            priority=58.0 + max(0, DOOMED_EVAC_HORIZON - fall_turn),
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
    if len(world.enemy_planets) <= 5 and is_breach_kill_mode(world):
        world.add_debug(
            f"EXPANSION_SKIP reason=breach_kill_priority enemies={len(world.enemy_planets)}"
        )
        return proposals
    candidates = nearest_neutral_candidates(world, deadline)
    for c in candidates[:4]:
        tgt = world.planet_by_id.get(c.target_id)
        if tgt is None:
            continue
        if not _is_pressure_mode(world):
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
    targets = world.enemy_planets + (world.neutral_planets if not _is_pressure_mode(world) else [])
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
    if not _is_pressure_mode(world) and world.remaining > PROTECT_LEAD_REMAINING:
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
    if not _is_pressure_mode(world):
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
    if not control_based_final_phase(world) and world.remaining > 45:
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: active agent commits funded chain proposals directly."""
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
        and _is_low_control(world)
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
        not _is_low_control(world)
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
    """Light reserve during expansion: avoids over-holding before pressure phase."""
    if _is_low_control(world):
        reserve = 2
    elif not _is_pressure_mode(world):
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
                world.add_debug(f"MEDIUM_OPERATOR_MARKED p{current.id} radius={current.radius:.1f}")
            else:
                world.add_debug(f"SMALL_STORAGE_MARKED p{current.id} radius={current.radius:.1f}")
    return _start_type_cache[world.player]


def mark_launchpad_after_capture(world, p):
    store = _primary_launchpads.setdefault(world.player, {})
    role = radius_class(p)
    start = _start_planet_current(world)
    start_type = _start_type_cache.get(world.player) or get_start_type(world)
    low_increment_start = (
        start is not None
        and start.id != p.id
        and start_type == "SMALL"
        and (float(p.radius) > float(start.radius) or int(p.production) > int(start.production))
    )
    if role == "LARGE":
        if p.id not in store:
            store[p.id] = world.step
            world.add_debug(f"PRIMARY_LAUNCHPAD_MARKED p{p.id} radius={p.radius:.1f}")
            if low_increment_start:
                world.add_debug(f"PRIMARY_LAUNCHPAD_REASSIGNED from=p{start.id} to=p{p.id}")
                world.add_debug(f"PRIMARY_BASE_REASSIGNED_AFTER_SMALL_START from=p{start.id} to=p{p.id}")
            if not is_static_planet(p):
                world.add_debug(f"ROTATING_LAUNCHPAD_MARKED p{p.id}")
        return True
    if role == "MEDIUM":
        world.add_debug(f"MEDIUM_OPERATOR_MARKED p{p.id} radius={p.radius:.1f}")
        large_owned = any(radius_class(m) == "LARGE" for m in world.my_planets)
        if start_type == "SMALL" and not large_owned and p.id not in store:
            store[p.id] = world.step
            world.add_debug(f"PRIMARY_LAUNCHPAD_MARKED p{p.id} radius={p.radius:.1f}")
            if low_increment_start:
                world.add_debug(f"PRIMARY_LAUNCHPAD_REASSIGNED from=p{start.id} to=p{p.id}")
                world.add_debug(f"PRIMARY_BASE_REASSIGNED_AFTER_SMALL_START from=p{start.id} to=p{p.id}")
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
        score -= 140.0
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
    elif role == "LARGE":
        score += 120.0
    elif role == "MEDIUM":
        score += 75.0
    else:
        score -= 140.0

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
        if _is_low_control(world) and not is_local_enemy_opportunity(world, target):
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
    if radius_class(target) == "SMALL":
        return False
    if target.owner == -1:
        return True
    if not _is_low_control(world):
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: corner preference is folded into chain route scoring."""
    if not world.is_four_player or _is_pressure_mode(world):
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: active opening is launchpad-chain route scoring."""
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
        score -= 25.0
        if d <= 18.0 and need <= 10:
            score += 20.0
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
    if prod <= 1 and _is_low_control(world):
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
    if _is_low_control(world) and world.my_planets:
        score += start_aware_opening_score_adjustment(world, src, tgt, get_start_type(world))

    score += estimate_capture_window_bonus(world, src, tgt, need)

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
    if prod <= 1 and _is_low_control(world):
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
    if _is_low_control(world) and d <= 35.0 and world.my_planets:
        if get_start_type(world) == "LARGE":
            return True  # bypass remaining prod/race filters
    # Skip enemy-favored low-production neutrals early game
    if not _is_pressure_mode(world) and prod < 4:
        status, _, _ = neutral_race_status(world, tgt)
        if status == "ENEMY_FAVORED":
            world.add_debug(f"VALIDATE_REJECT p{tgt.id} reason=enemy_favored_low_prod prod={prod}")
            return False
    return True


def is_sunk_cost_target(world, tgt):
    """Return True if we've committed ships to tgt but should cut losses and move on."""
    if not _is_low_control(world):
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
        opening_reserve = 1 if _is_low_control(world) else 2
        available = int(src.ships) - world.committed.get(src.id, 0) - opening_reserve
        if available <= 0:
            continue
        for tgt in world.neutral_planets:
            need = int(tgt.ships) + 1 - world.incoming_to_targets.get(tgt.id, 0)
            if need <= 0:
                continue
            if _is_low_control(world):
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
    if not _is_low_control(world) and len(world.my_planets) >= 2:
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
    if _is_pressure_mode(world) and len(world.my_planets) >= 4:
        return False
    if not world.neutral_planets:
        return False

    # General early-prod force: step >= 30 with <3 planets (any start type)
    if world.my_planets and len(world.my_planets) < 3:
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: active opening is launchpad-chain route scoring."""
    return False
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
    if count < 3 or (_is_pressure_mode(world) and len(world.neutral_planets) < 3):
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
    if not world.neutral_planets or not world.my_planets:
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
            base_priority += PRIORITY_HV_PREMIER_STEP if not _is_low_control(world) else PRIORITY_HV_PREMIER_EARLY

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
    small_packets = False
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
            picked = (src.id, send, 0, world.eta(src, target, send))
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
    dynamic_fleet_cap = getattr(world, "dynamic_config", {}).get("fleet_ratio_cap", FLEET_RATIO_HARD)
    if post_ratio > dynamic_fleet_cap and mission_type not in CRITICAL_MISSIONS | {"HIGH_VALUE_NEUTRAL_RACE", "FINAL_DRAIN"}:
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
    if _is_pressure_mode(world) or world.features.get("final"):
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


LOCAL_SMASH_RADIUS     = 50.0   # max distance from my planet to smash target
LOCAL_SMASH_RATIO      = 0.95   # my nearby surplus must cover this fraction of target ships


def generate_local_smash_missions(world, states, deadline):
    """
    LOCAL_SMASH_OPPORTUNITY: enemy planets where my grouped nearby surplus is
    overwhelming (>= target.ships * LOCAL_SMASH_RATIO). Generates a MissionProposal
    with full grouped validation and hold check; never direct-commits.
    """
    proposals = []
    if not world.enemy_planets or not world.my_planets:
        return proposals
    fleet_ratio = compute_fleet_ratio(world)
    if fleet_ratio > FLEET_RATIO_SOFT:
        return proposals
    for tgt in sorted(world.enemy_planets, key=lambda p: int(p.ships)):
        if time.perf_counter() > deadline:
            break
        if world.is_comet(tgt):
            continue
        if not should_allow_enemy_attack(world, tgt, "SYNC_ATTACK", "local_smash"):
            continue
        nearby_sources = [
            p for p in world.my_planets
            if dp(p, tgt) <= LOCAL_SMASH_RADIUS
            and world.real_incoming_threat(p)["deficit"] <= 0
            and world.surplus(p) >= MIN_SEND_SHIPS
        ]
        nearby_surplus = sum(world.surplus(p) for p in nearby_sources)
        if nearby_surplus < int(tgt.ships) * LOCAL_SMASH_RATIO:
            continue
        prop = build_capture_plan(world, tgt, "SYNC_ATTACK", nearby_sources, max_sources=4, eta_spread_limit=6.0)
        if prop is None:
            continue
        eta = prop.eta_max
        if not world.can_hold_after_capture(tgt, eta, prop.required_ships):
            world.add_debug(f"LOCAL_SMASH_OPPORTUNITY_SKIP p{tgt.id} reason=not_holdable")
            continue
        prop.priority = PRIORITY_OPPORTUNISTIC_STRIKE + 12.0 + int(tgt.production) * 5.0
        prop.reason = f"local_smash p{tgt.id} surplus={nearby_surplus} ships={int(tgt.ships)}"
        world.add_debug(f"LOCAL_SMASH_OPPORTUNITY_ALLOWED p{tgt.id} ships={int(tgt.ships)} surplus={nearby_surplus}")
        proposals.append(prop)
        if len(proposals) >= 2:
            break
    return proposals


def generate_opportunistic_strike_missions(world, fleet_ratio, deadline, arbiter_fired=False):
    """
    Attacks enemy planets that recently launched fleets and are now thin.
    Only fires when: close enough, fleet_ratio safe, grouped attack flips ownership.
    All targets routed through should_allow_enemy_attack().
    """
    proposals = []
    if fleet_ratio > FLEET_RATIO_SOFT:
        return proposals
    if _is_build_chain(world) and fleet_ratio > MIDGAME_FLEET_SOFT:
        world.add_debug(f"NO_PANIC_BLOCK OPPORTUNISTIC_STRIKE fleet_ratio={fleet_ratio:.2f} build_chain")
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
    if len(world.my_planets) < 4 or len(world.enemy_planets) < 4:
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
        world.add_debug(f"FAR_DIRECT_TARGET_REJECTED p{target.id} d={cluster_d:.1f}")
        return False, f"too_far cluster_d={cluster_d:.1f}"
    if world.step <= MIDGAME_END_STEP and cluster_d > _local_direct_limit(world):
        bridge = _select_bridge_route_target(world, build_planet_states(world), target, getattr(world, "_active_chain_plan", []))
        if bridge is not None:
            world.add_debug(f"BRIDGE_ROUTE_REQUIRED final=p{target.id} bridge=p{bridge.id}")
        world.add_debug(f"FAR_DIRECT_TARGET_REJECTED p{target.id} d={cluster_d:.1f}")
        return False, f"far_direct_requires_bridge d={cluster_d:.1f}"

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

    if not is_static_planet(target):
        approach = rotating_target_approach_score(src, target, world)
        if approach > 0:
            s += min(85.0, approach * 2.8)
            world.add_debug(f"ROTATING_APPROACH_TARGET_SELECTED p{target.id} score={approach:.1f}")
        elif approach < -2.0:
            s -= min(120.0, abs(approach) * 3.2)
            world.add_debug(f"ROTATING_MOVING_AWAY_REJECTED p{target.id} score={approach:.1f}")

    # ── Anti-waiting: bonus when bot owns few planets ─────────────────────────
    if len(world.my_planets) < 4:
        s += 20.0                             # aggressively expand with few planets

    # ── Soft deductions (priority reduction, never veto) ─────────────────────
    my_pct, _, neutral_pct = compute_control_pct(world)

    # 4-player early game: slight de-priority for non-local enemy attacks
    if (world.is_four_player and _is_low_control(world)
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

    s += estimate_capture_window_bonus(world, src, target, need)

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
    if getattr(world, "aggressiveness_mode", "BALANCED") != "AGGRESSIVE" and _staging_controller_active(world):
        world.add_debug("STAGING_CONTROLLER_BLOCK find_capture_opportunities suppressed")
        return proposals

    # Hub Security: block all offensive proposals if any prod-3+ hub is under-garrisoned.
    hub_violated, hub_detail = _hub_security_violated(world)
    if hub_violated:
        world.add_debug(
            f"HUB_SECURITY_BLOCK find_capture_opportunities suppressed - {hub_detail}"
        )
        return proposals

    world.add_debug(
        f"CAPTURE_OPPORTUNITY_SCAN step={world.step} "
        f"neutrals={len(world.neutral_planets)} enemies={len(world.enemy_planets)} "
        f"my_planets={len(world.my_planets)} fleet_ratio={fleet_ratio:.2f}"
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
        if not _proposal_passes_capture_constraints(world, prop):
            continue

        # Emit context-specific debug markers for enemy captures
        if is_enemy:
            neutral_pct = len(world.neutral_planets) / max(1, len(world.normal_planets))
            if world.is_four_player and _is_low_control(world):
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
    """DEPRECATED_OLD_PIPELINE_UNREACHABLE: active agent commits funded chain proposals directly."""
    return [], False
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

    # Dynamic cap: force single-mission focus when production parity is lost.
    opening_limit = (
        EARLY_NEAREST_SWEEP_BURST
        if world.step <= EARLY_NEAREST_SWEEP_STEP_MAX and fleet_ratio <= FLEET_RATIO_SOFT
        else SEARCH_SELECT_LIMIT
    )
    effective_limit = 1 if world.my_prod <= world.enemy_prod and opening_limit <= 1 else opening_limit
    if opening_limit > SEARCH_SELECT_LIMIT:
        world.add_debug(f"SEARCH_LIMIT_RELAXED_FOR_OPENING limit={opening_limit}")
    if effective_limit < SEARCH_SELECT_LIMIT:
        world.add_debug(
            f"SEARCH_MISSION_CAP limit=1 my_prod={world.my_prod} enemy_prod={world.enemy_prod}"
        )

    selected: list = []
    for score, prop in scored:
        if score < SEARCH_MIN_SCORE:
            world.add_debug(
                f"SEARCH_REJECT_NEGATIVE {prop.kind}->p{prop.target_id} score={score:.1f} "
                f"target_id={prop.target_id} required_ships={prop.required_ships} "
                f"available_surplus={sum(world.surplus(p) for p in world.my_planets)}"
            )
            continue
        if len(selected) >= effective_limit:
            world.add_debug(
                f"SEARCH_REJECT_CAP {prop.kind}->p{prop.target_id} score={score:.1f} "
                f"limit={effective_limit} reason=not_top_{effective_limit}"
            )
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
                f"score={score:.1f} reason=not_top_{effective_limit}"
            )

    # Any candidates scored → block other offensive fallback this turn
    blocked = len(scored) > 0
    return selected, blocked


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCHPAD-CHAIN STRATEGY BRAIN
# Replaces the old scattered-planner agent with one clean strategic pipeline:
#   stable launchpad chain → production bank → rolling capture wave →
#   chain-retrigger response on loss → square-chain counterattack
# ══════════════════════════════════════════════════════════════════════════════

# ── Planet roles ──────────────────────────────────────────────────────────────

ROLE_LAUNCHPAD = "LAUNCHPAD"
ROLE_BRIDGE    = "BRIDGE"
ROLE_STORAGE   = "STORAGE"


def _planet_role(p):
    if p.radius >= RADIUS_LARGE:
        return ROLE_LAUNCHPAD
    if p.radius > RADIUS_SMALL:
        return ROLE_BRIDGE
    return ROLE_STORAGE


def _start_planet_current(world):
    start = next((p for p in world.initial_planets.values() if p.owner == world.player), None)
    return world.planet_by_id.get(start.id) if start is not None else (world.my_planets[0] if world.my_planets else None)


def _small_start_escape_mode(world, src=None):
    if world.step >= 60 or len(world.my_planets) > 2:
        return False
    start = _start_planet_current(world)
    if start is None or start.owner != world.player:
        return False
    if src is not None and src.id != start.id:
        return False
    small_or_low = start.radius <= RADIUS_SMALL or int(start.production) <= 2
    if not small_or_low:
        return False
    anchor_owned = any(
        p.id != start.id
        and (
            _planet_role(p) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or int(p.production) > int(start.production)
            or int(p.production) >= 3
        )
        for p in world.my_planets
    )
    return not anchor_owned


def _small_start_escape_target_value(world, target):
    if target is None or target.owner == world.player or world.is_comet(target):
        return False
    role = _planet_role(target)
    if role in (ROLE_BRIDGE, ROLE_LAUNCHPAD):
        return True
    if is_idle(target):
        return True
    if int(target.production) >= 3:
        return True
    return role == ROLE_STORAGE and _chain_small_has_value(world, target)


def _role_reserve(role, is_frontline=False):
    base = {ROLE_LAUNCHPAD: LAUNCHPAD_RESERVE,
            ROLE_BRIDGE:    BRIDGE_RESERVE,
            ROLE_STORAGE:   STORAGE_RESERVE}[role]
    return base + (10 if is_frontline else 0)


def zero_capital_backline_safe(world, p, critical_radius=72.0):
    if p is None or p.owner != world.player or len(world.my_planets) <= 1:
        return False
    if _small_start_escape_mode(world, p):
        return False
    if world.real_incoming_threat(p)["deficit"] > 0:
        return False
    nearest_enemy = world.nearest_enemy_distance(p)
    if nearest_enemy <= critical_radius:
        return False
    friendly_neighbors = sum(
        1 for q in world.my_planets
        if q.id != p.id and dp(p, q) <= 48.0
    )
    frontline_neighbors = sum(
        1 for q in world.my_planets
        if q.id != p.id and world.nearest_enemy_distance(q) < critical_radius
    )
    return friendly_neighbors >= 2 and frontline_neighbors >= 1


@dataclass
class PlanetState:
    """Per-planet snapshot built at the start of each turn."""
    planet_id:    int
    role:         str
    is_static:    bool
    ships:        int
    production:   int
    radius:       float
    reserve:      int
    safe_surplus: int
    threatened:   bool
    cluster_d:    float


def build_planet_states(world):
    """Build a role/surplus snapshot for every owned planet."""
    states = {}
    for p in world.my_planets:
        role      = _planet_role(p)
        static    = is_idle(p)
        front     = world.nearest_enemy_distance(p) < FRONTLINE_DIST
        reserve   = _role_reserve(role, front)
        if zero_capital_backline_safe(world, p):
            reserve = 0
            world.add_debug(f"ZERO_CAPITAL_BACKLINE_DRAIN p{p.id} reserve=0")
        elif _small_start_escape_mode(world, p):
            reserve = min(3, max(0, int(p.production)))
            world.add_debug(f"SMALL_START_ESCAPE_MODE p{p.id}")
            world.add_debug(f"SMALL_START_RESERVE_RELAXED p{p.id} reserve={reserve}")
        elif (len(world.my_planets) < 4 or world.step < 50) and not front:
            reserve = min(reserve, max(3, int(p.production) * 2 + 3))
        elif len(world.my_planets) > 1:
            start = _start_planet_current(world)
            if start is not None and p.id == start.id and role == ROLE_STORAGE:
                world.add_debug(f"SMALL_START_RECLASSIFIED_STORAGE p{p.id}")
        surplus   = max(0, int(p.ships) - world.committed.get(p.id, 0) - reserve)
        threatened = world.real_incoming_threat(p)["deficit"] > 0
        states[p.id] = PlanetState(
            planet_id    = p.id,
            role         = role,
            is_static    = static,
            ships        = int(p.ships),
            production   = int(p.production),
            radius       = p.radius,
            reserve      = reserve,
            safe_surplus = surplus,
            threatened   = threatened,
            cluster_d    = world.cluster_distance(p),
        )
    world.add_debug("CURRENT_TIMELINE_BUILT")
    return states


# ── Prediction timeline ───────────────────────────────────────────────────────

def build_prediction_timeline(world, horizons=(5, 10, 15, 20)):
    """Forecast ship counts and owner for each planet at each horizon."""
    result = {}
    for p in world.normal_planets:
        result[p.id] = {}
        for h in horizons:
            tl = world.simulate_planet_timeline(p, h)
            result[p.id][h] = {
                "owner": tl["owner_at"].get(h, p.owner),
                "ships": tl["ships_at"].get(h, int(p.ships)),
                "holds": tl["holds"],
            }
    world.add_debug("PREDICTION_TIMELINE_BUILT")
    return result


def forecast_opponent_power(world, opp_id, horizon=10):
    """Estimate an opponent's total projected ships + production."""
    opp_p = [p for p in world.enemy_planets if p.owner == opp_id]
    opp_f = [f for f in world.enemy_fleets  if f.owner == opp_id]
    ships   = sum(int(p.ships) for p in opp_p) + sum(int(f.ships) for f in opp_f)
    prod    = sum(int(p.production) for p in opp_p)
    nearest = min((min(dp(m, e) for m in world.my_planets) for e in opp_p), default=999.0)
    return {"projected_ships": ships + prod * horizon,
            "production": prod, "nearest_d": nearest}


# ── Chain planet scoring ──────────────────────────────────────────────────────

def _chain_planet_score(world, planet):
    """
    Score a planet as a desired node in the launchpad chain.
    Higher = more desirable.  Works for owned, neutral, and enemy planets.
    """
    prod     = int(planet.production)
    role     = _planet_role(planet)
    static   = is_idle(planet)
    cd       = world.cluster_distance(planet)
    ships    = int(planet.ships)
    is_enemy = planet.owner not in (-1, world.player)

    s = 0.0

    # Role / radius
    if role == ROLE_LAUNCHPAD:
        s += 80.0
    elif role == ROLE_BRIDGE:
        s += 30.0

    # Non-orbiting bonus (huge asset)
    if static:
        s += 90.0
        if role == ROLE_LAUNCHPAD:
            world.add_debug(f"STATIC_LAUNCHPAD_PRIORITY p{planet.id} r={planet.radius:.1f}")

    # Production
    s += prod * 28.0

    # Proximity to cluster
    s += max(0.0, CHAIN_RADIUS - cd) * 1.4

    # Enemy: extra value (weakens opponent, gains their production)
    if is_enemy:
        s += prod * 15.0

    # Small bridge value check
    if role == ROLE_STORAGE:
        connects = any(
            dp(planet, q) <= 40.0
            and _planet_role(q) != ROLE_STORAGE
            and q.id != planet.id
            for q in world.normal_planets
        )
        if not connects:
            s -= 40.0   # isolated storage: low priority

    # Capture cost penalty
    if planet.owner != world.player:
        s -= max(1, ships + 1) * 0.9

    return s


def _chain_small_has_value(world, planet, anchor=None):
    if _planet_role(planet) != ROLE_STORAGE:
        return True
    if planet.owner not in (-1, world.player):
        nearest = min((dp(m, planet) for m in world.my_planets), default=999.0)
        if nearest <= 70.0 or enemy_planets_total(world) <= 8:
            return True
    nearby_value = [
        q for q in world.normal_planets
        if q.id != planet.id
        and not world.is_comet(q)
        and dp(planet, q) <= CHAIN_RADIUS
        and (
            _planet_role(q) == ROLE_LAUNCHPAD
            or is_idle(q)
            or int(q.production) >= 4
            or _prev_owners.get(q.id) == world.player
        )
    ]
    if anchor is not None and nearby_value:
        return any(dp(anchor, q) > dp(planet, q) + 8.0 for q in nearby_value)
    return bool(nearby_value) or _prev_owners.get(planet.id) == world.player


def _opening_chain_bonus(world, planet):
    if world.step > FORCED_OPENING_STEP:
        return 0.0
    start_type = get_start_type(world)
    role = _planet_role(planet)
    static = is_idle(planet)
    prod = int(planet.production)
    bonus = 0.0
    if start_type == "SMALL":
        if role == ROLE_LAUNCHPAD:
            bonus += 260.0
        elif role == ROLE_BRIDGE:
            bonus += 150.0
        else:
            bonus -= 80.0
    elif start_type == "MEDIUM":
        if role == ROLE_LAUNCHPAD:
            bonus += 230.0
        elif role == ROLE_BRIDGE:
            bonus += 70.0
        else:
            bonus -= 60.0
    else:
        if role == ROLE_LAUNCHPAD:
            bonus += 190.0
        elif role == ROLE_BRIDGE:
            bonus += 105.0
        else:
            bonus -= 35.0
    if static and role in (ROLE_LAUNCHPAD, ROLE_BRIDGE):
        bonus += 130.0
    bonus += prod * 12.0
    return bonus


def _route_node_score(world, planet, anchor, selected, rotating_launchpad_used):
    role = _planet_role(planet)
    if role == ROLE_STORAGE and not _chain_small_has_value(world, planet, anchor):
        world.add_debug(f"CHAIN_REJECT_ISOLATED_SMALL p{planet.id}")
        return -1e9
    score = _chain_planet_score(world, planet) + _opening_chain_bonus(world, planet)
    if anchor is not None:
        score -= dp(anchor, planet) * 2.0
        if dp(anchor, planet) <= CHAIN_RADIUS:
            score += 85.0
    if role == ROLE_LAUNCHPAD and is_idle(planet):
        score += 260.0
    elif role == ROLE_BRIDGE and is_idle(planet):
        score += 145.0
    elif role == ROLE_LAUNCHPAD and not is_idle(planet):
        score += 70.0 if not rotating_launchpad_used else -110.0
    if int(planet.production) >= 4:
        score += 120.0
    if role == ROLE_BRIDGE:
        future_static = [
            q for q in world.normal_planets
            if q.id not in selected
            and q.id != planet.id
            and not world.is_comet(q)
            and is_idle(q)
            and _planet_role(q) in (ROLE_LAUNCHPAD, ROLE_BRIDGE)
        ]
        if future_static:
            score += max(0.0, CHAIN_RADIUS - min(dp(planet, q) for q in future_static)) * 2.0
    if world.is_four_player:
        zone = classify_corner_zone(world, planet)
        if zone == "my_start_corner":
            score += 70.0
        elif zone in ("clockwise_adjacent_corner", "counterclockwise_adjacent_corner"):
            score += 95.0 if any(classify_corner_zone(world, world.planet_by_id[pid]) == "my_start_corner" for pid in selected if pid in world.planet_by_id) else 15.0
        elif zone == "opposite_corner":
            score -= 80.0
    return score


def build_launchpad_chain_plan(world, states=None):  # noqa: states kept for API compat
    """
    Return an ordered route: owned anchor → bridge/storage if useful →
    static launchpad/high-production node → adjacent/frontier/enemy source.
    """
    owned = [p for p in world.my_planets if not world.is_comet(p)]
    if not owned:
        return []
    anchor = max(
        owned,
        key=lambda p: (
            _planet_role(p) == ROLE_LAUNCHPAD,
            is_idle(p),
            int(p.production),
            states.get(p.id).safe_surplus if states and p.id in states else int(p.ships),
        ),
    )
    chain = [anchor.id]
    selected = {anchor.id}
    rotating_launchpad_used = not is_idle(anchor) and _planet_role(anchor) == ROLE_LAUNCHPAD

    while len(chain) < 14:
        scored = []
        for p in world.normal_planets:
            if p.id in selected or world.is_comet(p):
                continue
            score = _route_node_score(world, p, anchor, selected, rotating_launchpad_used)
            if score <= -1e8:
                continue
            scored.append((score, p))
        if not scored:
            break
        scored.sort(key=lambda x: -x[0])
        score, node = scored[0]
        chain.append(node.id)
        selected.add(node.id)
        if _planet_role(node) == ROLE_LAUNCHPAD and not is_idle(node):
            rotating_launchpad_used = True
        world.add_debug(
            f"CHAIN_ROUTE_NODE_SELECTED p{node.id} role={_planet_role(node)} "
            f"static={is_idle(node)} prod={int(node.production)} score={score:.1f}"
        )
        anchor = node

    for pid in chain[:5]:
        p = world.planet_by_id.get(pid)
        if p is None:
            continue
        s = _chain_planet_score(world, p)
        world.add_debug(
            f"PLANET_ROLE_CLASSIFIED p{p.id} role={_planet_role(p)} "
            f"static={is_idle(p)} prod={int(p.production)} "
            f"r={p.radius:.1f} score={s:.1f}"
        )
    world.add_debug("LAUNCHPAD_CHAIN_SELECTED")
    return chain


# ── Staging launchpad ─────────────────────────────────────────────────────────

def choose_staging_launchpad(world, states, chain_plan):
    """Choose the best owned planet to use as the current staging / rally point."""
    if not world.my_planets:
        return None
    scored = []
    for p in world.my_planets:
        st = states.get(p.id)
        if st is None or st.threatened:
            continue
        s = 0.0
        if st.role == ROLE_LAUNCHPAD: s += 60.0
        if st.is_static:              s += 50.0
        s += int(p.production) * 20.0
        s += st.safe_surplus * 0.35
        # Proximity to nearest unowned chain target
        for pid in chain_plan:
            tgt = world.planet_by_id.get(pid)
            if tgt is not None and tgt.owner != world.player:
                s -= dp(p, tgt) * 0.25
                break
        scored.append((s, p.id))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    sid = scored[0][1]
    world.add_debug(f"STAGING_LAUNCHPAD_SELECTED p{sid}")
    return sid


# ── Grouped funding (valid packets only) ──────────────────────────────────────

def _is_opening_capture_reason(reason):
    reason = reason or ""
    return any(token in reason for token in (
        "parallel_opening_sweep",
        "early_nearest_sweep",
        "small_start_escape",
        "early_direct_expansion",
        "expansion_campaign",
        "expansion_obligation",
        "nearest_useful",
        "fast_heuristic_fallback",
    ))


def _early_neutral_reserve_relaxation_allowed(world, src, tgt, mission_reason=""):
    if world is None or src is None or tgt is None:
        return False
    if tgt.owner != -1 or world.step > 70 or world.is_comet(tgt):
        return False
    if dp(src, tgt) > max(PARALLEL_OPENING_SWEEP_DIST, EARLY_NEAREST_SWEEP_DIST + 12.0):
        return False
    if world.real_incoming_threat(src)["deficit"] > 0:
        return False
    if not _is_opening_capture_reason(mission_reason):
        return False
    return True


def _early_neutral_min_source_reserve(world, src):
    if world.real_incoming_threat(src)["deficit"] > 0:
        return 5
    if world.nearest_enemy_distance(src) <= FRONTLINE_DIST + 8:
        return 3
    return 0


def _neutral_capture_fast_need(world, src, target, eta=None, mission_reason=""):
    base = max(0, int(target.ships) + 1 - world.incoming_to_targets.get(target.id, 0))
    if target.owner != -1:
        return base
    margin = 0
    my_eta, enemy_eta = world.reaction_times(target)
    if enemy_eta <= my_eta + 2.0:
        margin = 3
        world.add_debug(f"CONTESTED_NEUTRAL_SMALL_MARGIN_ONLY target=p{target.id} margin={margin}")
    if enemy_eta < my_eta - 3.0:
        margin = 5
        world.add_debug(f"CONTESTED_NEUTRAL_SMALL_MARGIN_ONLY target=p{target.id} margin={margin}")
    if world.step <= 70:
        world.add_debug(
            f"NEUTRAL_CAPTURE_FAST_NEED_USED target=p{target.id} ships={int(target.ships)} margin={margin}"
        )
        world.add_debug(f"NEUTRAL_CAPTURE_NO_EXTRA_WAIT target=p{target.id}")
    return max(0, int(target.ships) + 1 + margin - world.incoming_to_targets.get(target.id, 0))


def _enemy_capture_hold_buffer(world, target, eta, mission_reason=""):
    buffer = 8
    if world.nearest_enemy_distance(target) <= FRONTLINE_DIST + 18:
        buffer += 8
    if int(target.production) >= 4:
        buffer += 8
    elif int(target.production) >= 3:
        buffer += 5
    if _planet_role(target) == ROLE_LAUNCHPAD:
        buffer += 8
    enemy_reinf = sum(
        ships for arrival_eta, owner, ships in world.arrivals_by_target.get(target.id, [])
        if owner != world.player and arrival_eta <= max(eta, 1) + 8
    )
    if enemy_reinf > 0:
        buffer += min(20, int(enemy_reinf * 0.5))
    reason = mission_reason or ""
    recently_drained = target.id in {
        item["source"].id
        for item in getattr(world, "_cached_enemy_actions", {}).get("drained", [])
    } if hasattr(world, "_cached_enemy_actions") else False
    if recently_drained or "drained" in reason or "same_step_bundle" in reason or len(world.enemy_planets) <= 1:
        buffer = max(5, int(buffer * 0.55))
    world.add_debug(f"ENEMY_CAPTURE_HOLD_BUFFER_INCLUDED target=p{target.id} buffer={buffer}")
    return int(buffer)


def _enemy_capture_need_with_eta(world, src, target, eta, mission_reason=""):
    eta_turns = max(1, int(math.ceil(eta)))
    hold_buffer = _enemy_capture_hold_buffer(world, target, eta_turns, mission_reason)
    need = (
        int(target.ships)
        + eta_turns * int(target.production)
        + 1
        + hold_buffer
        + world.enemy_incoming_to_targets.get(target.id, 0)
        - world.incoming_to_targets.get(target.id, 0)
    )
    rounded = normalize_send_amount(max(0, need))
    world.add_debug(
        f"ENEMY_CAPTURE_ETA_PRODUCTION_INCLUDED target=p{target.id} ships={int(target.ships)} "
        f"prod={int(target.production)} eta={eta_turns} raw_need={int(need)}"
    )
    world.add_debug(f"ENEMY_CAPTURE_NEED_ROUNDED_TO_PACKET target=p{target.id} need={rounded}")
    return rounded


def _fund_capture(
    world,
    target,
    states,
    max_sources=6,
    mission_reason="",
    hold_margin_override=None,
    source_radius=None,
):
    """
    Build a grouped capture plan for `target` using safe nearby sources.
    Every source contribution is a valid fleet packet (>=10, multiple of 5).
    Returns (planned_sources, total_send, ok).
    Callers must still pass this through validate_grouped_launch and commit().
    """
    if world.is_comet(target):
        return [], 0, False

    src_near = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if src_near is None:
        return [], 0, False

    opening_neutral = target.owner == -1 and world.step <= 70 and _is_opening_capture_reason(mission_reason)
    pool = 0
    for pid, st in states.items():
        if st.threatened:
            continue
        p = world.planet_by_id.get(pid)
        if p is None:
            continue
        if opening_neutral and _early_neutral_reserve_relaxation_allowed(world, p, target, mission_reason):
            min_reserve = _early_neutral_min_source_reserve(world, p)
            pool += max(0, int(p.ships) - world.committed.get(pid, 0) - min_reserve)
        else:
            pool += min(st.safe_surplus, int(p.ships) - world.committed.get(pid, 0) - st.reserve)
    expansion_reason = (
        "expansion_campaign" in (mission_reason or "")
        or "expansion_obligation" in (mission_reason or "")
    )
    if target.owner == -1 and (expansion_reason or opening_neutral):
        need = _neutral_capture_fast_need(world, src_near, target, mission_reason=mission_reason)
    else:
        need = world.ships_needed_to_capture(src_near, target, max(1, pool))
    if need <= 0:
        return [], 0, False

    if opening_neutral:
        hold_margin = 0 if hold_margin_override is None else min(hold_margin_override, 3)
    elif target.owner not in (-1, world.player):
        hold_margin = 0
    else:
        hold_margin = hold_margin_override if hold_margin_override is not None else max(5, int(target.production) * 2)
    required     = normalize_send_amount(need + hold_margin)
    source_radius = source_radius if source_radius is not None else CHAIN_RADIUS + 8

    candidates = sorted(
        [p for p in world.my_planets
         if p.id in states
         and not states[p.id].threatened
         and dp(p, target) <= source_radius],
        key=lambda p: (dp(p, target), -states[p.id].safe_surplus),
    )[:max_sources]

    planned   = []
    remaining = required
    mission_type = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
    for src in candidates:
        if remaining <= 0:
            break
        st    = states[src.id]
        if opening_neutral and _early_neutral_reserve_relaxation_allowed(world, src, target, mission_reason):
            min_reserve = _early_neutral_min_source_reserve(world, src)
            avail = max(0, int(src.ships) - world.committed.get(src.id, 0) - min_reserve)
            world.add_debug(
                f"SURPLUS_GATE_BYPASSED_FOR_OPENING_CAPTURE src=p{src.id} target=p{target.id} "
                f"avail={avail} reserve={min_reserve}"
            )
        else:
            avail = min(
                st.safe_surplus,
                int(src.ships) - world.committed.get(src.id, 0) - st.reserve,
            )
        raw_contrib = min(avail, remaining)
        contrib     = round_down_to_granularity(raw_contrib)
        if contrib < MIN_SEND_SHIPS and avail >= remaining:
            contrib = normalize_send_amount(remaining)
        if contrib < MIN_SEND_SHIPS:
            world.add_debug(f"TINY_PACKET_REJECTED src=p{src.id} avail={avail}")
            continue
        if contrib > avail:
            world.add_debug(f"INVALID_PACKET_REJECTED src=p{src.id} ships={contrib} avail={avail}")
            continue
        if not valid_packet_size(mission_type, contrib):
            world.add_debug(f"INVALID_PACKET_REJECTED src=p{src.id} ships={contrib}")
            continue
        min_after = _early_neutral_min_source_reserve(world, src) if opening_neutral else st.reserve
        if int(src.ships) - world.committed.get(src.id, 0) - contrib < min_after:
            world.add_debug(f"PARTIAL_PACKET_REJECT_NO_CONVERSION src=p{src.id}")
            continue
        safe, reason = world.source_is_safe_for(src, target, mission_type, contrib, mission_reason=mission_reason)
        if not safe:
            world.add_debug(f"PARTIAL_PACKET_REJECT_NO_CONVERSION src=p{src.id} reason={reason}")
            continue
        structure = getattr(world, "_active_structure", None)
        if (
            structure
            and 80 <= world.step <= 320
            and mission_type in OFFENSIVE_MISSIONS
            and mission_type not in ("FINAL_DRAIN", "COLLAPSE")
        ):
            labels = structure.get(src.id, set())
            if STRUCT_ANCHOR in labels or STRUCT_BRIDGE in labels:
                desired = _structure_desired_reserve(world, src, labels)
                remaining_after = int(src.ships) - world.committed.get(src.id, 0) - contrib
                if remaining_after < desired:
                    if STRUCT_ANCHOR in labels:
                        world.add_debug(f"ANCHOR_RESERVE_PRESERVED p{src.id} reserve={desired}")
                    else:
                        world.add_debug(f"BRIDGE_RESERVE_PRESERVED p{src.id} reserve={desired}")
                    continue
        if world.eta(src, target, contrib) > world.remaining - 1:
            continue
        eta = world.eta(src, target, contrib)
        planned.append((src.id, contrib, 0, eta))
        remaining -= contrib

    total  = sum(s for _, s, _, _ in planned)
    funded = total >= need and remaining <= 0
    if funded:
        world.add_debug(f"UNIVERSAL_PACKET_RULE_APPLIED target=p{target.id} total={total}")
    return planned, total, funded


def _top_up_capture_to_critical_mass(world, target, states, planned, mission_type, mission_reason, source_radius, max_sources):
    if not planned:
        return planned
    if target.owner == -1 and world.step <= 70 and _is_opening_capture_reason(mission_reason):
        total = sum(int(s) for _sid, s, _a, _e in planned)
        src = world.planet_by_id.get(planned[0][0])
        required = normalize_send_amount(_neutral_capture_fast_need(world, src, target, mission_reason=mission_reason))
        if total >= required:
            world.add_debug(f"NEUTRAL_CAPTURE_NO_EXTRA_WAIT target=p{target.id} total={total} required={required}")
            return planned
    _owner_at, projected_defense, arrival_turn = _capture_projected_defense(
        world, target, max(e for _sid, _s, _a, e in planned)
    )
    required = _critical_mass_required(projected_defense)
    total = sum(int(s) for _sid, s, _a, _e in planned)
    if total >= required:
        return planned
    planned = list(planned)
    by_source = {}
    for src_id, ships, _angle, _eta in planned:
        by_source[src_id] = by_source.get(src_id, 0) + int(ships)
    candidates = sorted(
        [
            p for p in world.my_planets
            if p.id in states
            and not states[p.id].threatened
            and dp(p, target) <= source_radius
        ],
        key=lambda p: (0 if p.id in by_source else 1, dp(p, target), -states[p.id].safe_surplus),
    )[:max_sources]
    for src in candidates:
        if total >= required:
            break
        st = states[src.id]
        already = by_source.get(src.id, 0)
        avail = min(
            st.safe_surplus,
            int(src.ships) - world.committed.get(src.id, 0) - st.reserve,
        ) - already
        if avail < MIN_SEND_SHIPS:
            continue
        add = round_down_to_granularity(min(avail, required - total))
        if add < MIN_SEND_SHIPS and avail >= required - total:
            add = normalize_send_amount(required - total)
        if add < MIN_SEND_SHIPS or add > avail:
            continue
        safe, reason = world.source_is_safe_for(src, target, mission_type, add, mission_reason=mission_reason)
        if not safe:
            world.add_debug(f"CRITICAL_MASS_TOPUP_SKIP src=p{src.id} target=p{target.id} reason={reason}")
            continue
        eta = world.eta(src, target, add)
        for idx, (src_id, ships, _angle, _old_eta) in enumerate(planned):
            if src_id == src.id:
                combined = int(ships) + add
                planned[idx] = (src.id, combined, 0, world.eta(src, target, combined))
                break
        else:
            planned.append((src.id, add, 0, eta))
        by_source[src.id] = by_source.get(src.id, 0) + add
        total += add
    world.add_debug(
        f"CRITICAL_MASS_TOPUP target=p{target.id} projected_defense={projected_defense} "
        f"required={required} total={total} arrival_turn={arrival_turn}"
    )
    return planned


# ── Execute a funded proposal ─────────────────────────────────────────────────

def _proposal_launch_step(world, raw_launch_step):
    launch_step = int(raw_launch_step)
    return world.step if launch_step <= 0 else launch_step


def _queue_mission_launch(world, mission_id, prop, src_id, ships, launch_step):
    launches = _pending_mission_launches.setdefault(world.player, [])
    target = world.planet_by_id.get(prop.target_id)
    arrival_steps = _planned_arrival_steps(world, prop.planned_sources)
    sync_window = _sync_window_for_mission(world, target, prop.kind, prop.reason)
    scheduled_arrival = (
        _proposal_launch_step(world, launch_step)
        + max(1, int(math.ceil(next((eta for sid, _s, _ls, eta in prop.planned_sources if sid == src_id), 0))))
    )
    launches.append({
        "mission_id": mission_id,
        "mission_type": canonical_mission_type(prop.kind),
        "target_id": int(prop.target_id),
        "source_id": int(src_id),
        "ships": int(ships),
        "launch_step": int(launch_step),
        "scheduled_arrival_step": int(scheduled_arrival),
        "sync_window": sync_window,
        "group_source_count": len(prop.planned_sources),
        "group_required_ships": int(prop.required_ships),
        "group_arrival_step": max(arrival_steps) if arrival_steps else int(scheduled_arrival),
        "planned_sources": list(prop.planned_sources),
        "reason": prop.reason,
    })
    world.add_debug(
        f"MISSION_QUEUED_JIT {prop.kind} id={mission_id} src=p{src_id} "
        f"target=p{prop.target_id} ships={int(ships)} launch_step={int(launch_step)}"
    )
    if _sync_managed_group(prop) and int(launch_step) > world.step:
        world.add_debug(
            f"SYNC_ATTACK_DELAYED_SOURCE id={mission_id} src=p{src_id} target=p{prop.target_id} "
            f"delay={int(launch_step) - world.step} arrival_step={int(scheduled_arrival)}"
        )


def _pending_records_for_mission(world, pending, mission_id):
    return [
        rec for rec in pending
        if rec.get("mission_id") == mission_id and int(rec.get("launch_step", world.step)) >= world.step
    ]


def _cancel_pending_group(world, remaining, pending, mission_id, marker, reason):
    if mission_id is not None:
        entry = world.mission_ledger.get(mission_id)
        if entry is not None:
            world.mission_ledger.invalidate(mission_id, reason)
    target_id = None
    for rec in pending:
        if rec.get("mission_id") == mission_id:
            target_id = rec.get("target_id")
            break
    world.add_debug(f"{marker} id={mission_id} target=p{target_id} reason={reason}")
    return [
        rec for rec in remaining
        if rec.get("mission_id") != mission_id
    ]


def _pending_group_sources_as_offsets(world, pending_records):
    planned = []
    for rec in pending_records:
        src = world.planet_by_id.get(rec.get("source_id"))
        tgt = world.planet_by_id.get(rec.get("target_id"))
        ships = int(rec.get("ships", 0))
        if src is None or tgt is None or ships <= 0:
            continue
        launch_step = int(rec.get("launch_step", world.step))
        eta = world.eta(src, tgt, ships)
        planned.append((src.id, ships, launch_step, eta))
    return _planned_sources_as_arrival_offsets(world, planned)


def _pending_group_still_flips(world, tgt, pending_records):
    if tgt is None or world.is_comet(tgt):
        return False, "target invalid"
    if not pending_records:
        return True, ""
    mission_type = canonical_mission_type(pending_records[0].get("mission_type", "SYNC_ATTACK"))
    if mission_type not in OFFENSIVE_MISSIONS or _sync_release_exempt(mission_type, pending_records[0].get("reason", "")):
        return True, ""
    planned_offsets = _pending_group_sources_as_offsets(world, pending_records)
    already_in_flight = world.incoming_to_targets.get(tgt.id, 0)
    if len(planned_offsets) < 2 and int(pending_records[0].get("group_source_count", 1)) >= 2 and already_in_flight <= 0:
        return False, "group would trickle"
    ok_sync, spread, window = _sync_window_ok(world, tgt, mission_type, planned_offsets, pending_records[0].get("reason", ""))
    if not ok_sync:
        world.add_debug(f"SYNC_WINDOW_TOO_LOOSE_REJECTED target=p{tgt.id} spread={spread:.1f} window={window}")
        return False, f"sync spread {spread:.1f}>{window}"
    if planned_offsets:
        friendly_arrivals = [
            eta for eta, owner, ships in world.arrivals_by_target.get(tgt.id, [])
            if owner == world.player and int(ships) > 0
        ]
        eval_turn = max(
            1,
            int(math.ceil(max(
                [e for _sid, _s, _ls, e in planned_offsets] + friendly_arrivals
            ))),
        )
        extra = tuple((max(1, int(math.ceil(e))), world.player, int(s)) for _sid, s, _ls, e in planned_offsets)
        owner_after, _ships_after = world.projected_state(tgt.id, eval_turn, extra_arrivals=extra)
        if owner_after != world.player:
            return False, f"no longer flips at t={eval_turn}"
    return True, ""


def _delayed_mission_store(world):
    return _pending_delayed_missions.setdefault(world.player, {})


def _prop_uses_emergency_defense(prop):
    reason_l = (getattr(prop, "reason", "") or "").lower()
    return (
        getattr(prop, "kind", "") in ("DEFEND_HOLD", "SAVE_UNDER_ATTACK", "FINISH_ZERO_CAPTURE", "DOOMED_EVACUATION")
        or any(token in reason_l for token in ("emergency", "urgent", "under_attack", "save_under_attack", "finish_zero"))
    )


def _delayed_source_plan(world, mission):
    target = world.planet_by_id.get(mission.get("target_id"))
    if target is None:
        return []
    planned = []
    for rec in mission.get("sources", []):
        if rec.get("released"):
            continue
        src = world.planet_by_id.get(rec.get("src_id"))
        ships = int(rec.get("ships", 0))
        if src is None or ships <= 0:
            continue
        eta = world.eta(src, target, ships)
        planned.append((src.id, ships, int(rec.get("scheduled_launch_step", world.step)), eta))
    return planned


def _delayed_arrival_spread(world, mission):
    arrivals = _planned_arrival_steps(world, _delayed_source_plan(world, mission))
    if len(arrivals) < 2:
        return 0.0
    return float(max(arrivals) - min(arrivals))


def _delayed_group_still_flips(world, mission):
    target = world.planet_by_id.get(mission.get("target_id"))
    if target is None or world.is_comet(target):
        return False, "target invalid"
    kind = canonical_mission_type(mission.get("kind", "SYNC_ATTACK"))
    if kind not in OFFENSIVE_MISSIONS:
        return True, ""
    planned = _delayed_source_plan(world, mission)
    if not planned:
        return True, ""
    adjusted = _planned_sources_as_arrival_offsets(world, planned)
    ok_sync, spread, window = _sync_window_ok(world, target, kind, adjusted, mission.get("reason", ""))
    if not ok_sync:
        world.add_debug(f"SYNC_WINDOW_TOO_LOOSE_REJECTED target=p{target.id} spread={spread:.1f} window={window}")
        return False, f"sync spread {spread:.1f}>{window}"
    eval_turn = max(1, int(math.ceil(max(e for _sid, _s, _ls, e in adjusted))))
    extra = tuple((max(1, int(math.ceil(e))), world.player, int(s)) for _sid, s, _ls, e in adjusted)
    owner_after, _ships_after = world.projected_state(target.id, eval_turn, extra_arrivals=extra)
    if owner_after != world.player:
        return False, f"no longer flips at t={eval_turn}"
    return True, ""


def schedule_synchronized_launch(world, prop):
    """
    Convert a wide-spread grouped proposal into an absolute-step delayed burst.
    Returns True when the proposal was captured by the delayed controller.
    """
    if prop is None or len(getattr(prop, "planned_sources", []) or []) < 2:
        return False
    if _prop_uses_emergency_defense(prop) or _sync_release_exempt(prop.kind, prop.reason):
        return False
    target = world.planet_by_id.get(prop.target_id)
    if target is None or world.is_comet(target) or target.owner == world.player:
        return False
    if not _sync_managed_group(prop):
        return False
    if prop.kind in OFFENSIVE_MISSIONS and not _proposal_passes_capture_constraints(world, prop):
        return False
    sync_window = _sync_window_for_mission(world, target, prop.kind, prop.reason)
    if sync_window is None:
        return False
    etas = [float(eta) for _src_id, _ships, _launch_step, eta in prop.planned_sources]
    if not etas:
        return False
    latest_eta = max(etas)
    earliest_eta = min(etas)
    if latest_eta - earliest_eta <= sync_window:
        return False

    scheduled_sources = []
    delayed_planned = []
    for src_id, ships, _launch_step, eta in prop.planned_sources:
        src = world.planet_by_id.get(src_id)
        ships = int(ships)
        if src is None or src.owner != world.player or not valid_packet_size(prop.kind, ships, world, src, target):
            return False
        delay_turns = max(0, int(math.ceil(latest_eta - float(eta))))
        scheduled_launch_step = int(world.step + delay_turns)
        delayed_planned.append((src_id, ships, scheduled_launch_step, float(eta)))
        scheduled_sources.append({
            "src_id": int(src_id),
            "ships": ships,
            "eta": float(eta),
            "scheduled_launch_step": scheduled_launch_step,
            "released": False,
        })

    ok_sync, spread, window = _sync_window_ok(world, target, prop.kind, delayed_planned, prop.reason)
    if not ok_sync:
        world.add_debug(f"SYNC_WINDOW_TOO_LOOSE_REJECTED target=p{target.id} spread={spread:.1f} window={window}")
        return False
    if prop.kind in OFFENSIVE_MISSIONS:
        ok_grp, reason = validate_grouped_launch(world, target, delayed_planned)
        if not ok_grp:
            world.add_debug(f"SYNC_ATTACK_CANCELLED_NO_LONGER_FLIPS id=None target=p{target.id} reason={reason}")
            return False

    delayed_prop = MissionProposal(
        kind=prop.kind,
        target_id=prop.target_id,
        priority=prop.priority,
        required_ships=prop.required_ships,
        planned_sources=delayed_planned,
        eta_min=min(etas),
        eta_max=max(etas),
        reason=f"{prop.reason} delayed_sync_group",
        priority_tier=prop.priority_tier,
    )
    mission_id = world.mission_ledger.create_from_proposal(delayed_prop)
    entry = world.mission_ledger.get(mission_id)
    if entry is not None:
        entry.launch_step = min(src["scheduled_launch_step"] for src in scheduled_sources)
    _delayed_mission_store(world)[mission_id] = {
        "target_id": int(prop.target_id),
        "kind": canonical_mission_type(prop.kind),
        "latest_eta": float(latest_eta),
        "created_step": int(world.step),
        "reason": delayed_prop.reason,
        "sync_window": float(sync_window),
        "required_ships": int(prop.required_ships),
        "sources": scheduled_sources,
    }
    world.add_debug(
        f"SYNC_ATTACK_LOCK_CREATED id={mission_id} mission={prop.kind} target=p{prop.target_id} "
        f"latest_eta={latest_eta:.1f} spread={latest_eta - earliest_eta:.1f} window={sync_window}"
    )
    for rec in scheduled_sources:
        if rec["scheduled_launch_step"] > world.step:
            world.add_debug(
                f"SYNC_ATTACK_DELAYED_SOURCE id={mission_id} src=p{rec['src_id']} target=p{prop.target_id} "
                f"delay={rec['scheduled_launch_step'] - world.step} scheduled_launch_step={rec['scheduled_launch_step']}"
            )
    world.add_debug(
        f"GROUPED_BURST_SCHEDULE_CREATED id={mission_id} target=p{prop.target_id} "
        f"sources={[rec['src_id'] for rec in scheduled_sources]}"
    )
    return True


def _cancel_delayed_mission(world, mission_id, marker, reason):
    store = _delayed_mission_store(world)
    mission = store.pop(mission_id, None)
    target_id = mission.get("target_id") if mission else "?"
    if world.mission_ledger.get(mission_id) is not None:
        world.mission_ledger.invalidate(mission_id, reason)
    world.add_debug(f"{marker} id={mission_id} target=p{target_id} reason={reason}")


def process_delayed_launches(world, moves):
    store = _delayed_mission_store(world)
    if not store:
        return False
    launched_any = False
    for mission_id, mission in list(store.items()):
        target = world.planet_by_id.get(mission.get("target_id"))
        kind = canonical_mission_type(mission.get("kind", "SYNC_ATTACK"))
        if target is None or world.is_comet(target):
            _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_TARGET_CHANGED", "target missing/comet")
            continue
        if kind in OFFENSIVE_MISSIONS and target.owner == world.player:
            _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_ALREADY_CAPTURED", "target already mine")
            continue
        if world.step - int(mission.get("created_step", world.step)) > 45:
            _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_TARGET_CHANGED", "stale delayed mission")
            continue

        for rec in mission.get("sources", []):
            if rec.get("released"):
                continue
            src = world.planet_by_id.get(rec.get("src_id"))
            ships = int(rec.get("ships", 0))
            if src is None or src.owner != world.player or ships <= 0:
                _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_SOURCE_UNSAFE", "source missing/lost")
                break
            if not valid_packet_size(kind, ships, world, src, target):
                _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_SOURCE_UNSAFE", "invalid packet")
                break
            safe, safe_reason = world.source_is_safe_for(src, target, kind, ships, mission_reason=mission.get("reason", ""))
            if not safe:
                _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_SOURCE_UNSAFE", safe_reason)
                break
            aim_ok, aim_reason = world.aim_confidence_check(src, target, ships, kind)
            if not aim_ok:
                _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_SOURCE_UNSAFE", aim_reason)
                break
        else:
            ok_flip, flip_reason = _delayed_group_still_flips(world, mission)
            if not ok_flip:
                _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_NO_LONGER_FLIPS", flip_reason)
                continue

            for rec in mission.get("sources", []):
                scheduled_launch_step = int(rec.get("scheduled_launch_step", world.step))
                if rec.get("released") or not (world.step >= scheduled_launch_step):
                    continue
                src = world.planet_by_id.get(rec.get("src_id"))
                ships = int(rec.get("ships", 0))
                planned = _planned_sources_as_arrival_offsets(world, _delayed_source_plan(world, mission))
                ok_launch, launch_reason = world.valid_fleet_launch(
                    src,
                    target,
                    ships,
                    kind,
                    mission_entry=world.mission_ledger.get(mission_id),
                    planned_sources=planned,
                    mission_reason=mission.get("reason", ""),
                )
                if not ok_launch:
                    _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_SOURCE_UNSAFE", launch_reason)
                    break
                angle, aim_ok = world.aim(src, target, ships)
                if not aim_ok:
                    _cancel_delayed_mission(world, mission_id, "SYNC_ATTACK_CANCELLED_SOURCE_UNSAFE", "no_safe_intercept")
                    break
                moves.append([src.id, angle, ships])
                rec["released"] = True
                launched_any = True
                world.committed[src.id] = world.committed.get(src.id, 0) + ships
                eta = world.eta(src, target, ships)
                world.record_shot_launch(src, target, ships, angle, eta, kind)
                if target.owner != world.player:
                    world.incoming_to_targets[target.id] = world.incoming_to_targets.get(target.id, 0) + ships
                    world.offensive_ships += ships
                world.mission_ledger.record_launch(mission_id, src.id, ships, eta=eta)
                world.add_debug(
                    f"SYNC_ATTACK_RELEASED id={mission_id} src=p{src.id} target=p{target.id} "
                    f"ships={ships} scheduled_launch_step={scheduled_launch_step}"
                )
                world.add_debug(
                    f"GROUPED_BURST_ARRIVAL_CONFIRMED id={mission_id} target=p{target.id} "
                    f"arrival_step={world.step + max(1, int(math.ceil(eta)))}"
                )
            if mission_id in store and all(rec.get("released") for rec in mission.get("sources", [])):
                del store[mission_id]
            continue
        continue
    return launched_any


def execute_pending_missions(world, moves):
    """Execute launches due this frame with just-in-time trajectory calculation."""
    pending = _pending_mission_launches.get(world.player, [])
    if not pending:
        return False

    remaining = []
    launched_any = False
    cancelled_missions = set()
    for rec in pending:
        mission_id = rec.get("mission_id")
        if mission_id in cancelled_missions:
            continue
        mission_type = canonical_mission_type(rec.get("mission_type", "SYNC_ATTACK"))
        tgt = world.planet_by_id.get(rec.get("target_id"))
        if tgt is None or world.is_comet(tgt) or (mission_type in OFFENSIVE_MISSIONS and tgt.owner == world.player):
            remaining = _cancel_pending_group(
                world,
                remaining,
                pending,
                mission_id,
                "SYNC_ATTACK_CANCELLED_TARGET_CHANGED",
                "jit abort: target changed",
            )
            cancelled_missions.add(mission_id)
            continue
        sync_managed_record = (
            mission_type in OFFENSIVE_MISSIONS
            or any(token in (rec.get("reason", "") or "").lower() for token in ("staging", "two_stage", "relay", "backup_to_staging"))
        ) and not _sync_release_exempt(mission_type, rec.get("reason", ""))
        if sync_managed_record and mission_type in OFFENSIVE_MISSIONS:
            group_records = _pending_records_for_mission(world, pending, mission_id)
            ok_group, group_reason = _pending_group_still_flips(world, tgt, group_records)
            if not ok_group:
                remaining = _cancel_pending_group(
                    world,
                    remaining,
                    pending,
                    mission_id,
                    "SYNC_ATTACK_CANCELLED_NO_LONGER_FLIPS",
                    group_reason,
                )
                cancelled_missions.add(mission_id)
                continue
        launch_step = int(rec.get("launch_step", world.step))
        if launch_step > world.step:
            remaining.append(rec)
            continue
        if launch_step < world.step:
            world.add_debug(
                f"MISSION_ABORT_STALE_JIT id={rec.get('mission_id')} src=p{rec.get('source_id')} "
                f"target=p{rec.get('target_id')} launch_step={launch_step} step={world.step}"
            )
            continue

        src = world.planet_by_id.get(rec.get("source_id"))
        ships = int(rec.get("ships", 0))
        if src is None or tgt is None or src.owner != world.player or ships <= 0:
            world.add_debug(
                f"MISSION_ABORT_JIT id={mission_id} reason=missing_source_or_target "
                f"src=p{rec.get('source_id')} target=p{rec.get('target_id')}"
            )
            if mission_id is not None:
                world.mission_ledger.invalidate(mission_id, "jit abort: missing source or target")
            continue
        if not valid_packet_size(mission_type, ships, world, src, tgt):
            world.add_debug(f"MISSION_ABORT_JIT id={mission_id} reason=invalid_packet ships={ships}")
            if mission_id is not None:
                world.mission_ledger.invalidate(mission_id, "jit abort: invalid packet")
            continue
        available = int(src.ships) - world.committed.get(src.id, 0)
        if ships > available:
            world.add_debug(
                f"MISSION_ABORT_JIT id={mission_id} reason=unavailable src=p{src.id} "
                f"ships={ships} available={available}"
            )
            if mission_id is not None:
                world.mission_ledger.invalidate(mission_id, "jit abort: source unavailable")
            continue

        group_records = _pending_records_for_mission(world, pending, mission_id)
        validation_sources = _pending_group_sources_as_offsets(world, group_records)
        if not validation_sources:
            planned_sources = rec.get("planned_sources") or [(src.id, ships, world.step, world.eta(src, tgt, ships))]
            validation_sources = _planned_sources_as_arrival_offsets(world, planned_sources)
        ok_launch, reason = world.valid_fleet_launch(
            src,
            tgt,
            ships,
            mission_type,
            mission_entry=world.mission_ledger.get(mission_id),
            planned_sources=validation_sources,
            mission_reason=rec.get("reason", ""),
        )
        if not ok_launch:
            world.add_debug(
                f"MISSION_ABORT_JIT id={mission_id} mission={mission_type} "
                f"src=p{src.id} target=p{tgt.id} ships={ships} reason={reason}"
            )
            if mission_id is not None:
                world.mission_ledger.invalidate(mission_id, reason)
            continue

        actual_angle, aim_ok = world.aim(src, tgt, ships)
        if not aim_ok:
            world.add_debug(
                f"MISSION_ABORT_JIT id={mission_id} mission={mission_type} "
                f"src=p{src.id} target=p{tgt.id} ships={ships} reason=no_safe_intercept"
            )
            if mission_id is not None:
                world.mission_ledger.invalidate(mission_id, "jit abort: no safe intercept")
            continue

        moves.append([src.id, actual_angle, ships])
        launched_any = True
        world.committed[src.id] = world.committed.get(src.id, 0) + ships
        eta = world.eta(src, tgt, ships)
        world.record_shot_launch(src, tgt, ships, actual_angle, eta, mission_type)
        if tgt.owner != world.player:
            world.incoming_to_targets[tgt.id] = world.incoming_to_targets.get(tgt.id, 0) + ships
            world.offensive_ships += ships
        else:
            _recently_reinforced.setdefault(world.player, {})[tgt.id] = world.step
            world.recently_reinforced_planets[tgt.id] = world.step
        world.mission_ledger.record_launch(mission_id, src.id, ships, eta=eta)
        if sync_managed_record:
            world.add_debug(
                f"SYNC_ATTACK_RELEASED id={mission_id} src=p{src.id} target=p{tgt.id} "
                f"ships={ships} eta={eta:.1f}"
            )
        world.add_debug(
            f"MISSION_EXECUTE_JIT id={mission_id} mission={mission_type} "
            f"src=p{src.id} target=p{tgt.id} ships={ships} angle={actual_angle:.3f}"
        )

    _pending_mission_launches[world.player] = [
        rec for rec in remaining if rec.get("mission_id") not in cancelled_missions
    ]
    return launched_any


def _commit_proposal(world, prop, moves):
    """Create ledger entry and queue all sources for JIT execution."""
    tgt = world.planet_by_id.get(prop.target_id)
    if tgt is None:
        world.add_debug(
            f"COMMIT_REJECTED_NO_TARGET mission={prop.kind} target_id={prop.target_id} "
            f"required_ships={prop.required_ships}"
        )
        return False
    if schedule_synchronized_launch(world, prop):
        return True
    prop = _synchronize_offensive_proposal_timing(world, prop)
    if _sync_managed_group(prop):
        ok_sync, spread, window = _sync_window_ok(world, tgt, prop.kind, prop.planned_sources, prop.reason)
        if not ok_sync:
            world.add_debug(
                f"SYNC_WINDOW_TOO_LOOSE_REJECTED target=p{tgt.id} spread={spread:.1f} window={window}"
            )
            return False
        if prop.kind in OFFENSIVE_MISSIONS:
            adjusted_sources = _planned_sources_as_arrival_offsets(world, prop.planned_sources)
            ok_grp, grp_reason = validate_grouped_launch(world, tgt, adjusted_sources)
            if not ok_grp:
                world.add_debug(f"SYNC_ATTACK_CANCELLED_NO_LONGER_FLIPS id=None target=p{tgt.id} reason={grp_reason}")
                return False
        world.add_debug(
            f"GROUPED_BURST_ARRIVAL_CONFIRMED target=p{tgt.id} "
            f"arrivals={_planned_arrival_steps(world, prop.planned_sources)}"
        )
    if (
        prop.kind in ("SYNC_ATTACK", "BREACH_KILL", "COLLAPSE", "HIGH_VALUE_NEUTRAL_RACE")
        and len(prop.planned_sources) < 2
        and any(token in (prop.reason or "") for token in ("grouped", "coordinated", "same_frame", "sync"))
        and prop.kind != "FINISH_ZERO_CAPTURE"
    ):
        world.add_debug(f"SYNC_ATTACK_CANCELLED_NO_LONGER_FLIPS id=None target=p{tgt.id} reason=group_required")
        return False
    if prop.kind in OFFENSIVE_MISSIONS:
        active_offense = _active_offensive_capture_missions(world)
        geometric_neutral_burst = (
            tgt.owner == -1
            and (
                (
                    world.step <= EARLY_NEAREST_SWEEP_STEP_MAX
                    and (
                        "early_nearest_sweep" in (prop.reason or "")
                        or "parallel_opening_sweep" in (prop.reason or "")
                    )
                )
                or (
                    world.step <= MIDGAME_END_STEP
                    and "multi_axis_front" in (prop.reason or "")
                )
            )
        )
        same_step_bundle = "same_step_bundle" in (prop.reason or "")
        if (
            active_offense
            and not _has_significant_production_lead(world)
            and not _is_two_stage_final(prop)
            and not geometric_neutral_burst
            and not same_step_bundle
        ):
            world.add_debug(
                f"MISSION_BLOCK {prop.kind} target=p{tgt.id} reason=active_capture_concentration "
                f"active={[(e.mission_type, e.target_id) for e in active_offense]} "
                f"my_prod={world.my_prod} enemy_prod={world.enemy_prod}"
            )
            return False
        if not _proposal_passes_capture_constraints(world, prop):
            world.add_debug(f"MISSION_BLOCK {prop.kind} target=p{tgt.id} reason=critical_mass_or_cluster")
            return False
        total = _proposal_total_ships(prop)
        arrival_turn = max(1, _offensive_proposal_arrival_step(world, prop) - world.step)
        _owner_at, projected_defense, _arrival_turn = _capture_projected_defense(world, tgt, arrival_turn)
        required_mass = _critical_mass_required(projected_defense)
        _log_capture_force(
            world, "MISSION_COMMIT_FORCE", prop.kind, tgt, projected_defense, total, required_mass, arrival_turn
        )
    available_surplus = sum(world.surplus(p) for p in world.my_planets)
    for src_id, ships, launch_step, _eta in prop.planned_sources:
        src = world.planet_by_id.get(src_id)
        if src is None or not valid_packet_size(prop.kind, ships, world, src, tgt):
            world.add_debug(
                f"INVALID_PACKET_REJECTED mission={prop.kind} src=p{src_id} ships={ships} "
                f"target_id={prop.target_id} required_ships={prop.required_ships} "
                f"available_surplus={available_surplus} tgt_ships={int(tgt.ships)}"
            )
            return False
        if _is_two_stage_final(prop):
            if world.real_incoming_threat(src)["deficit"] > 0:
                world.add_debug(
                    f"MISSION_BLOCK {prop.kind} target=p{tgt.id} src=p{src.id} ships={ships} "
                    f"reason=staging_source_under_threat"
                )
                return False
            world.add_debug(
                f"STAGING_FORCE_RESERVED staging=p{src.id} target=p{tgt.id} final={int(ships)} launch_step={_proposal_launch_step(world, launch_step)}"
            )
        else:
            ok, reason = world.valid_fleet_launch(
                src,
                tgt,
                ships,
                prop.kind,
                planned_sources=_planned_sources_as_arrival_offsets(world, prop.planned_sources),
                mission_reason=prop.reason,
                validate_aim=False,
            )
            if not ok:
                world.add_debug(
                    f"MISSION_BLOCK {prop.kind} target=p{tgt.id} src=p{src.id} ships={ships} "
                    f"required_ships={prop.required_ships} available_surplus={available_surplus} "
                    f"tgt_ships={int(tgt.ships)} tgt_prod={int(tgt.production)} reason={reason}"
                )
                return False
    mid       = world.mission_ledger.create_from_proposal(prop)
    entry = world.mission_ledger.get(mid)
    launch_steps = [_proposal_launch_step(world, launch_step) for _, _, launch_step, _ in prop.planned_sources]
    if entry is not None and launch_steps:
        entry.launch_step = min(launch_steps)
    queued = False
    for src_id, ships, launch_step, _eta in prop.planned_sources:
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        if _is_two_stage_final(prop):
            world.staging_reserved_ships[src_id] = world.staging_reserved_ships.get(src_id, 0) + int(ships)
        _queue_mission_launch(world, mid, prop, src_id, ships, _proposal_launch_step(world, launch_step))
        queued = True
    return queued


def _small_start_followup_count(world, target):
    return sum(
        1 for q in world.normal_planets
        if q.id != target.id
        and q.owner != world.player
        and not world.is_comet(q)
        and dp(target, q) <= CAMPAIGN_RADIUS
        and (
            _planet_role(q) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or is_idle(q)
            or int(q.production) >= 3
        )
    )


def _small_start_escape_rank(world, target):
    role = _planet_role(target)
    followups = _small_start_followup_count(world, target)
    if role == ROLE_LAUNCHPAD and int(target.production) >= 5:
        category = 0
    elif role in (ROLE_BRIDGE, ROLE_LAUNCHPAD) and int(target.production) >= 4:
        category = 1
    elif is_idle(target) and role in (ROLE_BRIDGE, ROLE_LAUNCHPAD):
        category = 2
        world.add_debug(f"STATIC_ESCAPE_TARGET_SELECTED p{target.id}")
    elif role == ROLE_BRIDGE:
        category = 3
    elif int(target.production) >= 3:
        category = 4
    elif role == ROLE_STORAGE and _chain_small_has_value(world, target):
        category = 5
    else:
        category = 9
    src = world.my_planets[0] if world.my_planets else target
    # Bonus for targets equal/bigger than our start — prefer escaping upward
    bigger_planet_bonus = 0
    if (float(target.radius) >= float(src.radius) * 0.8
            and int(target.production) >= int(src.production) * 0.8):
        bigger_planet_bonus = -1   # sort earlier (lower = higher priority)
        world.add_debug(f"SMALL_START_BIGGER_PLANET_ESCAPE_BONUS p{target.id} radius={target.radius:.1f} prod={target.production}")
    return (
        category + bigger_planet_bonus,
        dp(src, target),
        -followups,
        -int(target.production),
        int(target.ships),
    )


def run_small_start_escape(world, states, moves, deadline):
    if not _small_start_escape_mode(world) or moves or time.perf_counter() > deadline:
        return False
    src = _start_planet_current(world)
    if src is None or src.owner != world.player:
        return False
    world.add_debug(f"SMALL_START_ESCAPE_MODE p{src.id}")
    candidates = [
        t for t in world.normal_planets
        if t.owner != world.player
        and not world.is_comet(t)
        and dp(src, t) <= 70.0
        and _small_start_escape_target_value(world, t)
    ]
    candidates.sort(key=lambda t: _small_start_escape_rank(world, t))
    for target in candidates[:8]:
        if time.perf_counter() > deadline:
            break
        if target.owner not in (-1, world.player):
            continue
        if dp(src, target) > 45.0 and any(dp(src, n) <= 45.0 and _small_start_escape_target_value(world, n) for n in world.neutral_planets if not world.is_comet(n)):
            world.add_debug(f"SMALL_START_SKIP_FAR_TARGET p{target.id} d={dp(src, target):.1f}")
            continue
        # Try single-source first; fall back to grouped (2 sources) for bigger targets
        plan, total, ok = _fund_capture(
            world,
            target,
            states,
            max_sources=1,
            mission_reason="small_start_escape",
        )
        if not ok:
            # Grouped fallback — two sources is still conservative enough for small-start
            bigger = (
                float(target.radius) >= float(src.radius) * 0.8
                or int(target.production) >= max(2, int(src.production))
                or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            )
            if bigger:
                plan, total, ok = _fund_capture(
                    world,
                    target,
                    states,
                    max_sources=2,
                    mission_reason="small_start_escape_grouped",
                    hold_margin_override=2,
                )
                if ok:
                    world.add_debug(f"SMALL_START_AGGRESSIVE_ESCAPE grouped p{target.id}")
        if not ok:
            continue
        eta_vals = [eta for _, _, _, eta in plan]
        ok_grp, reason = validate_grouped_launch(world, target, plan)
        if not ok_grp:
            world.add_debug(f"SMALL_START_ESCAPE_SKIP p{target.id} reason={reason}")
            continue
        if not world.can_hold_after_capture(target, max(eta_vals), total):
            world.add_debug(f"SMALL_START_ESCAPE_SKIP p{target.id} reason=not_holdable")
            continue
        mission_type = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = MissionProposal(
            kind=mission_type,
            target_id=target.id,
            priority=140.0,
            required_ships=total,
            planned_sources=plan,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"small_start_escape p{src.id}->p{target.id}",
        )
        if _commit_proposal(world, prop, moves):
            if _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD) or int(target.production) > int(src.production):
                world.add_debug(f"SMALL_START_BIGGER_PLANET_ESCAPE p{target.id}")
                world.add_debug(f"SMALL_START_BIGGER_TARGET_SELECTED p{target.id}")
                world.add_debug(
                    f"SMALL_START_TO_MEDIUM_ANCHOR target=p{target.id} role={_planet_role(target)} prod={int(target.production)}"
                )
                world.add_debug(f"SMALL_START_ESCAPE_COMPLETED p{target.id}")
            _last_capture_step[world.player] = world.step
            return True
    return False


# ── Beam expansion / opponent forecast layer ──────────────────────────────────

def forecast_enemy_actions(world, horizons=(5, 10, 15, 20)):
    """
    Lightweight opponent forecast: identify likely captures, threats, and drained
    enemy sources without replacing WorldModel/projected_state.
    """
    actions = []
    drained = []
    threats = []
    for src in world.enemy_planets:
        if world.is_comet(src):
            continue
        prev = _prev_ships.get(src.id)
        drop = 0
        if prev is not None and _prev_owners.get(src.id) == src.owner:
            drop = max(0, int(prev) + int(src.production) - int(src.ships))
            if drop >= max(15, int(src.production) * 4):
                drained.append({"source": src, "drop": drop})
                world.add_debug(f"DRAINED_ENEMY_TARGET_FOUND p{src.id} drop={drop}")

        safe_send = round_down_to_granularity(max(0, int(src.ships) - max(8, int(src.production) * 3)))
        if safe_send < MIN_SEND_SHIPS:
            continue

        best = None
        for target in world.neutral_planets + world.my_planets:
            if target.id == src.id or world.is_comet(target):
                continue
            d = dp(src, target)
            if d > 78.0:
                continue
            eta = world.eta(src, target, safe_send)
            if eta > max(horizons):
                continue
            if target.owner == world.player:
                need = max(1, int(target.ships) + int(target.production) * max(1, int(math.ceil(eta))) + 1)
            else:
                need = max(1, int(target.ships) + 1)
            if safe_send < need:
                continue
            value = (
                int(target.production) * 42.0
                + (80.0 if target.owner == world.player and _planet_role(target) in (ROLE_LAUNCHPAD, ROLE_BRIDGE) else 0.0)
                + (60.0 if is_idle(target) else 0.0)
                + max(0.0, 80.0 - d) * 1.4
                - int(target.ships) * 0.8
                - eta * 2.0
            )
            item = (value, target, eta, need, safe_send)
            if best is None or item[0] > best[0]:
                best = item
        if best is None:
            continue
        value, target, eta, need, send = best
        action = {
            "source": src,
            "target": target,
            "eta": eta,
            "need": need,
            "send": send,
            "score": value,
            "drained_drop": drop,
        }
        actions.append(action)
        if target.owner == world.player:
            threats.append(action)
            world.add_debug(
                f"ENEMY_ATTACK_FORECASTED src=p{src.id} target=p{target.id} eta={eta:.1f} send={send}"
            )

    actions.sort(key=lambda a: (-a["score"], a["eta"]))
    threats.sort(key=lambda a: (a["eta"], -a["score"]))
    world.add_debug(
        f"ENEMY_ACTION_FORECAST_BUILT actions={len(actions)} threats={len(threats)} drained={len(drained)}"
    )
    return {"actions": actions, "threats": threats, "drained": drained}


def predictive_defense(world, states, enemy_actions, moves, deadline):
    """Pre-reinforce valuable planets before a forecast attack can land."""
    if moves or time.perf_counter() > deadline:
        return False
    threats = (enemy_actions or {}).get("threats", [])
    if not threats:
        return False
    launched = False
    for action in threats[:5]:
        if time.perf_counter() > deadline:
            break
        target = action["target"]
        if target.owner != world.player or world.is_comet(target):
            continue
        st = states.get(target.id)
        important = (
            st is not None
            and (st.role in (ROLE_LAUNCHPAD, ROLE_BRIDGE) or st.production >= 3 or st.is_static)
        )
        if not important and action["eta"] > 10:
            continue
        projected_deficit = max(
            0,
            int(action["send"]) + 1 - int(target.ships) - int(target.production) * max(0, int(math.floor(action["eta"]))),
        )
        current_deficit = world.real_incoming_threat(target, horizon=max(DEFENSE_ETA_HORIZON, int(action["eta"]) + 2))["deficit"]
        need = normalize_send_amount(max(projected_deficit, current_deficit, MIN_SEND_SHIPS))
        if need < MIN_SEND_SHIPS:
            continue
        srcs = sorted(
            [
                p for p in world.my_planets
                if p.id != target.id
                and p.id in states
                and not states[p.id].threatened
                and states[p.id].safe_surplus >= MIN_SEND_SHIPS
                and world.eta(p, target, MIN_SEND_SHIPS) <= action["eta"] + 2
            ],
            key=lambda p: (world.eta(p, target, min(states[p.id].safe_surplus, need)), dp(p, target)),
        )
        sent = 0
        for src in srcs[:4]:
            if sent >= need or time.perf_counter() > deadline:
                break
            raw = min(states[src.id].safe_surplus, need - sent)
            send = round_down_to_granularity(raw)
            if send < MIN_SEND_SHIPS and states[src.id].safe_surplus >= need - sent:
                send = normalize_send_amount(need - sent)
            if send < MIN_SEND_SHIPS or send > states[src.id].safe_surplus:
                continue
            if world.commit(src, target, send, moves, mission_type="DEFEND_HOLD"):
                sent += send
                launched = True
                world.add_debug("PREDICTIVE_DEFENSE_TRIGGERED")
                world.add_debug(f"PRE_REINFORCE_PLANET target=p{target.id} src=p{src.id} send={send}")
        if sent >= need:
            world.add_debug(f"PLANET_SAVED_BY_FORECAST p{target.id} sent={sent}")
            break
    return launched


def _beam_expansion_needed(world, control_ratio):
    if _is_low_control(world):
        return True
    if control_ratio < PHASE_MIDGAME_MAX:
        return True
    if len(world.my_planets) < 8:
        return True
    return False


def _beam_small_start_active(world):
    start = _start_planet_current(world)
    return (
        start is not None
        and start.owner == world.player
        and len(world.my_planets) <= 2
        and _is_low_control(world)
        and (float(start.radius) <= RADIUS_SMALL or int(start.production) <= 1)
    )


def _beam_target_allowed(world, target, anchor, small_start=False):
    if target is None or target.owner == world.player or world.is_comet(target):
        return False
    if small_start:
        if _small_start_escape_target_value(world, target):
            return True
        bigger_near = any(
            _small_start_escape_target_value(world, q)
            and dp(target, q) <= BEAM_EXPANSION_RADIUS
            for q in world.normal_planets
            if q.id != target.id and q.owner != world.player and not world.is_comet(q)
        )
        return _planet_role(target) == ROLE_STORAGE and bigger_near and dp(anchor, target) <= 36.0
    if target.owner not in (-1, world.player) and not (
        is_local_enemy_opportunity(world, target)
        or _small_radius_target_allowed(world, target, anchor)
        or int(target.production) >= 3
    ):
        return False
    if _planet_role(target) == ROLE_STORAGE and target.owner == -1:
        return _small_radius_target_allowed(world, target, anchor)
    return _useful_target_value(world, target, anchor)


def _beam_target_score(world, anchor, target, captured_ids, enemy_actions=None, small_start=False):
    d = dp(anchor, target)
    role = _planet_role(target)
    followups = _campaign_followup_options(world, target)
    score = 0.0
    score += 130.0 if target.id not in captured_ids else -200.0
    score += int(target.production) * 58.0
    if int(target.production) >= 4:
        score += 90.0
    if role == ROLE_LAUNCHPAD:
        score += 115.0
    elif role == ROLE_BRIDGE:
        score += 75.0
    elif target.owner not in (-1, world.player):
        score += 55.0
        world.add_debug("SMALL_PLANET_CAPTURE_ALLOWED_FOR_CONTROL")
    else:
        score -= 25.0
    if is_idle(target):
        score += 95.0
    score += min(4, len(followups)) * 48.0
    score += max(0.0, BEAM_EXPANSION_RADIUS - d) * 4.0
    score += _rotating_source_static_target_bonus(world, anchor, target)
    score -= int(target.ships) * 1.35
    score -= max(0.0, d - 38.0) * 3.0
    if role == ROLE_STORAGE and _small_radius_target_allowed(world, target, anchor):
        score += 70.0 if target.owner not in (-1, world.player) else 35.0
    if small_start:
        if _small_start_escape_target_value(world, target):
            score += 240.0
        elif role == ROLE_STORAGE:
            score -= 80.0
    if enemy_actions:
        for action in enemy_actions.get("actions", [])[:8]:
            if action["target"].id == target.id:
                score += 85.0
                break
    # Capture-efficiency bonus: prefer targets with high ROI (cheap relative to value)
    rough_need = max(MIN_SEND_SHIPS, int(target.ships) + 1)
    roi = capture_conversion_score(world, target, anchor, rough_need)
    score += min(60.0, roi * 22.0)
    return score


def _beam_next_candidates(world, anchors, captured_ids, enemy_actions, small_start=False):
    candidates = []
    for anchor in anchors:
        for target in world.normal_planets:
            if target.id in captured_ids or not _beam_target_allowed(world, target, anchor, small_start=small_start):
                continue
            d = dp(anchor, target)
            if d > BEAM_EXPANSION_RADIUS:
                continue
            score = _beam_target_score(world, anchor, target, captured_ids, enemy_actions, small_start=small_start)
            candidates.append((score, d, int(target.ships), anchor, target))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[:BEAM_EXPANSION_WIDTH]


def _beam_select_path(world, enemy_actions=None, small_start=False, deadline=None):
    anchors = list(world.my_planets)
    beam = [{
        "score": 0.0,
        "anchors": anchors,
        "captured": set(),
        "path": [],
        "prod": world.my_prod,
        "planets": len(world.my_planets),
    }]
    best = None
    depth = 3 if small_start else BEAM_EXPANSION_DEPTH
    for level in range(depth):
        if deadline is not None and time.perf_counter() > deadline:
            break
        nxt = []
        for state in beam:
            for raw_score, dist, _ships, anchor, target in _beam_next_candidates(
                world, state["anchors"], state["captured"], enemy_actions, small_start=small_start and level == 0
            ):
                eta_guess = world.eta(anchor, target, max(MIN_SEND_SHIPS, normalize_send_amount(int(target.ships) + 1)))
                new_score = (
                    state["score"]
                    + raw_score
                    + 70.0
                    + int(target.production) * 20.0
                    - eta_guess * 3.0
                    - level * 18.0
                )
                if target.owner not in (-1, world.player):
                    new_score += 45.0
                virtual_anchor = target
                new_state = {
                    "score": new_score,
                    "anchors": state["anchors"] + [virtual_anchor],
                    "captured": set(state["captured"]) | {target.id},
                    "path": state["path"] + [(anchor.id, target.id, new_score)],
                    "prod": state["prod"] + int(target.production),
                    "planets": state["planets"] + 1,
                }
                world.add_debug(
                    f"BEAM_STATE_EVALUATED depth={level + 1} target=p{target.id} score={new_score:.1f}"
                )
                nxt.append(new_state)
        if not nxt:
            break
        nxt.sort(key=lambda s: -s["score"])
        beam = nxt[:BEAM_EXPANSION_WIDTH]
        if best is None or beam[0]["score"] > best["score"]:
            best = beam[0]
    return best


def _beam_first_target_prop(world, states, path, small_start=False):
    if not path:
        return None, None
    _anchor_id, target_id, score = path[0]
    target = world.planet_by_id.get(target_id)
    if target is None:
        return None, None
    if small_start and not _small_start_escape_target_value(world, target):
        world.add_debug(f"SMALL_START_SKIP_FAR_TARGET p{target.id}")
        if not _chain_small_has_value(world, target):
            return None, target
    hold = 1 if target.owner == -1 and world.step < 40 else (2 if target.owner == -1 else max(8, int(target.production) * 3))
    prop = _main35_make_capture_prop(
        world,
        states,
        target,
        "beam_expansion_engine_small_start" if small_start else "beam_expansion_engine",
        166.0 + score * 0.02,
        max_sources=4,
        source_radius=BEAM_EXPANSION_RADIUS + 6,
        hold_margin=hold,
        require_hold=target.owner != -1,
    )
    return prop, target


def build_grouped_relay_attack(world, target, states, staging_id=None):
    """
    Build the first step of a relay: nearby surplus backs up the best staging
    source so it can launch a larger grouped wave later. No partial target poke.
    """
    if target is None or target.owner == world.player or world.is_comet(target):
        return None
    direct_plan, direct_total, direct_ok = _fund_capture(
        world,
        target,
        states,
        max_sources=5,
        mission_reason="relay_direct_probe",
        hold_margin_override=max(8, int(target.production) * 3),
        source_radius=BEAM_EXPANSION_RADIUS + 10,
    )
    if direct_ok:
        return None
    staging = world.planet_by_id.get(staging_id) if staging_id else None
    if staging is None or staging.owner != world.player:
        staging = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if staging is None or staging.id not in states:
        return None
    safe_on_stage = max(0, states[staging.id].safe_surplus)
    rough_need = world.ships_needed_to_capture(staging, target, max(MIN_SEND_SHIPS, safe_on_stage + world.my_prod * 8))
    target_force = normalize_send_amount(rough_need + max(12, int(target.production) * 4))
    if safe_on_stage >= target_force:
        return None
    needed_backup = target_force - safe_on_stage
    backup_sources = sorted(
        [
            p for p in world.my_planets
            if p.id != staging.id
            and p.id in states
            and not states[p.id].threatened
            and states[p.id].safe_surplus >= MIN_SEND_SHIPS
            and dp(p, staging) <= 42.0
        ],
        key=lambda p: (world.eta(p, staging, min(states[p.id].safe_surplus, needed_backup)), -states[p.id].safe_surplus),
    )
    planned = []
    total_backup = 0
    for src in backup_sources[:5]:
        if total_backup >= needed_backup:
            break
        spare = min(states[src.id].safe_surplus, needed_backup - total_backup)
        send = round_down_to_granularity(spare)
        if send < MIN_SEND_SHIPS:
            continue
        eta = world.eta(src, staging, send)
        if eta > 6.0:
            continue
        planned.append((src.id, send, 0, eta))
        total_backup += send
    if not planned or safe_on_stage + total_backup < min(target_force, rough_need):
        return None
    world.add_debug(f"MULTI_SOURCE_RELAY_REQUIRED target=p{target.id} staging=p{staging.id}")
    world.add_debug(
        f"BACKUP_TO_STAGING_PLANNED staging=p{staging.id} target=p{target.id} backup={total_backup} force={safe_on_stage + total_backup}"
    )
    if safe_on_stage + total_backup >= target_force:
        world.add_debug(f"STAGING_FORCE_READY staging=p{staging.id} target=p{target.id}")
    return MissionProposal(
        kind="DEFEND_HOLD",
        target_id=staging.id,
        priority=118.0,
        required_ships=total_backup,
        planned_sources=planned,
        eta_min=min(e for _, _, _, e in planned),
        eta_max=max(e for _, _, _, e in planned),
        reason=f"relay_backup_to_staging target=p{target.id}",
    )


def run_beam_expansion_engine(world, states, chain_plan, enemy_actions, moves, deadline, *, staging_id=None, control_ratio=None):
    if moves or time.perf_counter() > deadline:
        return False
    control_ratio = planet_control_ratio(world) if control_ratio is None else control_ratio
    small_start = _beam_small_start_active(world)
    obligation = (
        control_ratio < PHASE_INITIAL_MAX
        and world.step - _beam_expansion_last.get(world.player, -999) >= BEAM_OBLIGATION_GAP
    )
    if not (small_start or obligation or _beam_expansion_needed(world, control_ratio)):
        return False
    world.add_debug("BEAM_EXPANSION_ENGINE_ACTIVE")
    if small_start:
        world.add_debug("SMALL_START_ESCAPE_FORCED")
        world.add_debug("NO_PASSIVE_OPENING_ALLOWED")
    if obligation:
        world.add_debug("EXPANSION_OBLIGATION_ACTIVE")
        world.add_debug("NO_PASSIVE_OPENING_ALLOWED")

    path_state = _beam_select_path(world, enemy_actions=enemy_actions, small_start=small_start, deadline=deadline)
    if not path_state or not path_state["path"]:
        return False
    prop, target = _beam_first_target_prop(world, states, path_state["path"], small_start=small_start)
    if prop is not None and _commit_proposal(world, prop, moves):
        world.add_debug(
            f"BEAM_PATH_SELECTED path={[pid for _src, pid, _score in path_state['path'][:BEAM_EXPANSION_DEPTH]]}"
        )
        world.add_debug(f"BEAM_EXPANSION_SELECTED p{prop.target_id} total={prop.required_ships}")
        if obligation:
            world.add_debug(f"FORCED_NEAREST_CAPTURE p{prop.target_id}")
        _beam_expansion_last[world.player] = world.step
        return True

    if target is not None and not small_start:
        relay = build_grouped_relay_attack(world, target, states, staging_id=staging_id)
        if relay is not None and _commit_proposal(world, relay, moves):
            world.add_debug(f"RELAY_ATTACK_LAUNCHED staging=p{relay.target_id} future=p{target.id}")
            return True
    return False


def run_drained_enemy_counterattack(world, states, enemy_actions, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    drained = (enemy_actions or {}).get("drained", [])
    if not drained:
        return False
    for item in sorted(drained, key=lambda x: (-x["drop"], min((dp(m, x["source"]) for m in world.my_planets), default=999.0)))[:5]:
        target = item["source"]
        if target.owner in (-1, world.player) or world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > 88.0:
            continue
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            "drained_enemy_counterattack",
            178.0 + item["drop"],
            max_sources=5,
            source_radius=90.0,
            hold_margin=max(10, int(target.production) * 4),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug(f"DRAINED_ENEMY_TARGET_FOUND p{target.id} drop={item['drop']}")
            world.add_debug(f"DRAINED_SOURCE_COUNTERATTACK p{target.id}")
            return True
    return False


def _campaign_followup_options(world, target):
    opts = [
        q for q in world.normal_planets
        if q.id != target.id
        and q.owner != world.player
        and not world.is_comet(q)
        and dp(target, q) <= CAMPAIGN_RADIUS
        and (
            _planet_role(q) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or is_idle(q)
            or int(q.production) >= 3
            or _chain_small_has_value(world, q, target)
        )
    ]
    return sorted(opts, key=lambda q: (dp(target, q), -int(q.production), int(q.ships)))[:4]


def _campaign_target_score(world, source, target, chain_plan):
    role = _planet_role(target)
    d = dp(source, target)
    need = max(1, int(target.ships) + 1) if target.owner == -1 else world.ships_needed_to_capture(source, target, max(1, int(source.ships)))
    eta = world.eta(source, target, normalize_send_amount(need))
    followups = _campaign_followup_options(world, target)
    score = 0.0
    score += max(0.0, CAMPAIGN_RADIUS - d) * 4.0
    score += max(0.0, 25.0 - eta) * 3.0
    score += _rotating_source_static_target_bonus(world, source, target)
    score -= need * 1.6
    score += int(target.production) * 38.0
    if int(target.production) >= 3:
        score += 55.0
    if role == ROLE_LAUNCHPAD:
        score += 95.0
    elif role == ROLE_BRIDGE:
        score += 65.0
    elif role == ROLE_STORAGE and _small_radius_target_allowed(world, target, source, chain_plan):
        score += 70.0 if target.owner not in (-1, world.player) else 25.0
    elif _chain_small_has_value(world, target, source):
        score += 18.0
    else:
        score -= 35.0
    if is_idle(target):
        score += 85.0
    score += min(4, len(followups)) * 42.0
    if target.id in set(chain_plan[:10]):
        score += 60.0
    nearest_enemy = min((dp(target, e) for e in world.enemy_planets), default=999.0)
    if 22.0 <= nearest_enemy <= 65.0:
        score += 30.0
    return score, followups, need, eta


def _campaign_sources(world, states):
    return [
        p for p in world.my_planets
        if p.id in states
        and not states[p.id].threatened
        and states[p.id].safe_surplus >= MIN_SEND_SHIPS
    ]


def has_affordable_campaign_capture(world, states):
    for src in _campaign_sources(world, states):
        for target in world.normal_planets:
            if target.owner == world.player or world.is_comet(target):
                continue
            if dp(src, target) > CAMPAIGN_RADIUS:
                continue
            if target.owner not in (-1, world.player) and not (
                is_local_enemy_opportunity(world, target)
                or _small_radius_target_allowed(world, target, src)
            ):
                continue
            if not (
                _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
                or is_idle(target)
                or int(target.production) >= 3
                or _chain_small_has_value(world, target, src)
                or len(_campaign_followup_options(world, target)) >= 1
                or _small_radius_target_allowed(world, target, src)
            ):
                continue
            _plan, _total, ok = _fund_capture(
                world,
                target,
                states,
                max_sources=3,
                mission_reason="expansion_obligation_probe",
                hold_margin_override=2 if target.owner == -1 else max(6, int(target.production) * 3),
                source_radius=CAMPAIGN_RADIUS,
            )
            if ok:
                return True
    return False


def _opponent_planet_lead(world):
    counts = {}
    for p in world.enemy_planets:
        counts[p.owner] = counts.get(p.owner, 0) + 1
    return max(counts.values(), default=0) - len(world.my_planets)


def _max_opponent_planets(world):
    counts = {}
    for p in world.enemy_planets:
        counts[p.owner] = counts.get(p.owner, 0) + 1
    return max(counts.values(), default=0)


def _nearby_neutral_exists(world, radius=CAMPAIGN_RADIUS):
    return any(
        n.owner == -1
        and not world.is_comet(n)
        and min((dp(m, n) for m in world.my_planets), default=999.0) <= radius
        for n in world.neutral_planets
    )


def _expansion_floor_missed(world):
    if world.step >= 80 and len(world.my_planets) < 8 and _nearby_neutral_exists(world):
        return True
    if world.step >= 50 and len(world.my_planets) <= 5 and _nearby_neutral_exists(world):
        return True
    if world.step >= 30 and len(world.my_planets) < 3 and _nearby_neutral_exists(world):
        return True
    if world.step >= 25 and len(world.my_planets) <= 2 and _nearby_neutral_exists(world):
        return True
    return False


def expansion_obligation_active(world):
    lead = _opponent_planet_lead(world)
    if lead >= 2:
        world.add_debug(f"OPPONENT_EXPANSION_LEAD_DETECTED lead={lead}")
    active = (
        world.step < 100
        or len(world.my_planets) < 8
        or _nearby_neutral_exists(world)
        or lead >= 2
    )
    if active:
        world.add_debug("EXPANSION_OBLIGATION_MODE")
    if _expansion_floor_missed(world):
        world.add_debug("EXPANSION_FLOOR_MISSED")
    if world.step >= 80 and _max_opponent_planets(world) >= max(1, len(world.my_planets) * 2):
        world.add_debug("UNDER_EXPANSION_EMERGENCY")
    return active


def _stalled_expansion_force_active(world):
    if world.step >= 80 and _max_opponent_planets(world) >= max(1, len(world.my_planets) * 2):
        return True
    if world.step >= 50 and len(world.my_planets) <= 5:
        return True
    if world.step >= 25 and len(world.my_planets) <= 2:
        return True
    return False


def _campaign_chain_built_for_bank(world):
    useful = [
        p for p in world.my_planets
        if _planet_role(p) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
        or int(p.production) >= 3
        or is_idle(p)
    ]
    if len(useful) >= 4:
        return True
    strong_launchpad = any(
        _planet_role(p) == ROLE_LAUNCHPAD and (is_idle(p) or int(p.production) >= 4)
        for p in world.my_planets
    )
    support_nodes = sum(
        1 for p in world.my_planets
        if _planet_role(p) in (ROLE_BRIDGE, ROLE_STORAGE) and not (_planet_role(p) == ROLE_STORAGE and not _chain_small_has_value(world, p))
    )
    return strong_launchpad and support_nodes >= 2


def run_expansion_campaign(world, states, chain_plan, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    if not expansion_obligation_active(world):
        return False
    world.add_debug("EXPANSION_CAMPAIGN_MODE")
    stalled = _stalled_expansion_force_active(world)
    if stalled:
        world.add_debug("STALLED_EXPANSION_FORCE_MODE")
    max_missions = 3 if stalled and world.step >= 50 else 1
    launched = 0

    while launched < max_missions and time.perf_counter() < deadline:
        proposals = []
        sources = _campaign_sources(world, states)
        for src in sources:
            for target in world.normal_planets:
                if time.perf_counter() > deadline:
                    break
                if target.owner == world.player or world.is_comet(target):
                    continue
                if dp(src, target) > CAMPAIGN_RADIUS:
                    continue
                if world.incoming_to_targets.get(target.id, 0) >= world.required_ships_to_capture(target, src):
                    continue
                if target.owner not in (-1, world.player) and not (
                    is_local_enemy_opportunity(world, target)
                    or _small_radius_target_allowed(world, target, src, chain_plan)
                ):
                    continue
                role = _planet_role(target)
                followups = _campaign_followup_options(world, target)
                if (
                    role == ROLE_STORAGE
                    and target.owner == -1
                    and len(followups) < 1
                    and int(target.production) < 3
                    and not _small_radius_target_allowed(world, target, src, chain_plan)
                ):
                    world.add_debug(f"CAMPAIGN_REJECT_UNAFFORDABLE target=p{target.id} reason=isolated_small")
                    continue
                score, followups, _need, _eta = _campaign_target_score(world, src, target, chain_plan)
                if stalled:
                    score += max(0.0, CAMPAIGN_RADIUS - dp(src, target)) * 1.8
                    if target.owner == -1:
                        score += 45.0
                hold = 2 if target.owner == -1 else max(6, int(target.production) * 3)
                plan, total, ok = _fund_capture(
                    world,
                    target,
                    states,
                    max_sources=4 if stalled else 3,
                    mission_reason="expansion_obligation_campaign",
                    hold_margin_override=hold,
                    source_radius=CAMPAIGN_RADIUS,
                )
                if not ok:
                    world.add_debug(f"CAMPAIGN_REJECT_UNAFFORDABLE target=p{target.id}")
                    continue
                eta_vals = [eta for _, _, _, eta in plan]
                ok_grp, reason = validate_grouped_launch(world, target, plan)
                if not ok_grp:
                    world.add_debug(f"CAMPAIGN_REJECT_NO_CONVERSION target=p{target.id} reason={reason}")
                    continue
                if target.owner != -1 and not world.can_hold_after_capture(target, max(eta_vals), total):
                    world.add_debug(f"CAMPAIGN_REJECT_NO_CONVERSION target=p{target.id} reason=not_holdable")
                    continue
                proposals.append((score, target, plan, total, eta_vals, followups))

        if not proposals:
            break
        proposals.sort(key=lambda item: (-item[0], min(dp(world.planet_by_id[src_id], item[1]) for src_id, _, _, _ in item[2]), int(item[1].ships)))
        score, target, plan, total, eta_vals, followups = proposals[0]
        world.add_debug(
            f"CAMPAIGN_BRANCH_TARGET_SELECTED p{target.id} score={score:.1f} "
            f"role={_planet_role(target)} prod={int(target.production)} send={total}"
        )
        world.add_debug(
            f"CAMPAIGN_FOLLOWUP_LIST_BUILT p{target.id} options={[p.id for p in followups]}"
        )
        world.add_debug(
            f"CAMPAIGN_TARGET_SELECTED p{target.id} score={score:.1f} "
            f"role={_planet_role(target)} prod={int(target.production)} send={total}"
        )
        world.add_debug(
            f"CAMPAIGN_FOLLOWUP_OPTIONS p{target.id} options={[p.id for p in followups]}"
        )
        mission_type = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = MissionProposal(
            kind=mission_type,
            target_id=target.id,
            priority=145.0 + score * 0.05,
            required_ships=total,
            planned_sources=plan,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"expansion_obligation p{target.id} followups={[p.id for p in followups]}",
        )
        if not _commit_proposal(world, prop, moves):
            break
        world.add_debug(f"CAMPAIGN_LAUNCH_APPROVED p{target.id} total={total}")
        world.add_debug(f"VALID_CAPTURE_EXISTS_NO_EMPTY_TURN target=p{target.id}")
        launched += 1
        if not stalled:
            break

    return launched > 0


# ── Idle army pressure / surplus conversion ───────────────────────────────────

def _parked_army_stats(world):
    parked = sum(int(p.ships) for p in world.my_planets)
    flying = sum(int(f.ships) for f in world.my_fleets)
    total = max(1, parked + flying)
    return parked, flying, flying / total


def _safe_surplus_sources_for_pressure(world, states):
    sources = [
        p for p in world.my_planets
        if p.id in states
        and not states[p.id].threatened
        and states[p.id].safe_surplus >= MIN_SEND_SHIPS
        and world.real_incoming_threat(p)["deficit"] <= 0
    ]
    sources.sort(
        key=lambda p: (
            0 if states[p.id].role == ROLE_LAUNCHPAD else 1 if states[p.id].role == ROLE_BRIDGE else 2,
            -states[p.id].safe_surplus,
            -int(p.production),
        )
    )
    return sources


def _pressure_enemy_reachable(world, target, sources, max_dist=82.0):
    return any(dp(src, target) <= max_dist for src in sources)


def _surplus_enemy_target_score(world, target, sources, chain_plan):
    nearest = min((dp(src, target) for src in sources), default=999.0)
    followups = _campaign_followup_options(world, target)
    role = _planet_role(target)
    prev = _prev_ships.get(target.id)
    drained = 0
    if prev is not None and _prev_owners.get(target.id) == target.owner:
        drained = max(0, int(prev) + int(target.production) - int(target.ships))

    score = 0.0
    score += max(0.0, 90.0 - nearest) * 2.2
    score += max(0, 80 - int(target.ships)) * 1.4
    score += int(target.production) * 70.0
    if int(target.production) >= 4:
        score += 130.0
    if role == ROLE_LAUNCHPAD:
        score += 130.0
    elif role == ROLE_BRIDGE:
        score += 70.0
    if is_idle(target):
        score += 80.0
    if drained >= 20:
        score += drained * 2.5
    score += min(4, len(followups)) * 38.0
    if target.id in set(chain_plan[:14]):
        score += 70.0
    nearest_my = min((dp(p, target) for p in world.my_planets), default=999.0)
    nearest_enemy = min((dp(e, target) for e in world.enemy_planets if e.id != target.id), default=999.0)
    if nearest_my <= 70.0 and nearest_enemy >= 18.0:
        score += 35.0
    score -= int(target.ships) * 0.9
    return score


def _surplus_pressure_targets(world, states, chain_plan):
    sources = _safe_surplus_sources_for_pressure(world, states)
    targets = []
    for target in world.enemy_planets:
        if world.is_comet(target):
            continue
        if not _pressure_enemy_reachable(world, target, sources):
            continue
        if not (
            int(target.production) >= 3
            or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or is_local_enemy_opportunity(world, target)
            or len(_campaign_followup_options(world, target)) >= 2
        ):
            continue
        targets.append((_surplus_enemy_target_score(world, target, sources, chain_plan), target))
    targets.sort(key=lambda item: -item[0])
    return sources, [target for _, target in targets]


def _surplus_pressure_context(world, states, chain_plan, emit=False):
    parked, flying, flying_ratio = _parked_army_stats(world)
    threshold = 700 if world.step < 160 else 1200
    has_chain = len(world.my_planets) >= 6 or _campaign_chain_built_for_bank(world)
    sources, targets = _surplus_pressure_targets(world, states, chain_plan)
    emergency = any(world.real_incoming_threat(p)["deficit"] > 0 for p in world.my_planets)
    wave_floor = 300 if world.step < 160 else 500
    big_staging_wave_ready = any(
        states[p.id].role == ROLE_LAUNCHPAD and states[p.id].safe_surplus >= wave_floor
        for p in sources
    )
    active = (
        has_chain
        and parked >= threshold
        and flying_ratio < 0.15
        and (len(sources) >= 3 or big_staging_wave_ready)
        and bool(targets)
        and not emergency
    )
    if emit and active:
        world.add_debug("IDLE_ARMY_PRESSURE_MODE")
        world.add_debug(
            f"PARKED_SURPLUS_DETECTED parked={parked} flying={flying} "
            f"ratio={flying_ratio:.2f} sources={len(sources)} wave_floor={wave_floor}"
        )
    return active, sources, targets, parked, flying_ratio, threshold


def _build_surplus_attack_proposal(world, target, states, sources):
    if not sources:
        return None
    src_near = min(sources, key=lambda p: dp(p, target))
    safe_pool = sum(states[p.id].safe_surplus for p in sources if p.id in states)
    capture_need = world.ships_needed_to_capture(src_near, target, max(1, safe_pool))
    enemy_incoming_buffer = sum(
        ships for eta, owner, ships in world.arrivals_by_target.get(target.id, [])
        if owner != world.player and eta <= 18
    )
    hold_margin = max(18, int(target.production) * 5)
    required_total = capture_need + hold_margin + enemy_incoming_buffer

    planned, total, ok = _fund_capture(
        world,
        target,
        states,
        max_sources=6,
        mission_reason="surplus_conversion",
        hold_margin_override=hold_margin + enemy_incoming_buffer,
        source_radius=82.0,
    )
    if not ok:
        return None
    if len(planned) < 3:
        wave_floor = 300 if world.step < 160 else 500
        has_launchpad_wave = any(
            states.get(src_id) is not None and states[src_id].role == ROLE_LAUNCHPAD
            for src_id, _, _, _ in planned
        )
        if total < wave_floor or not has_launchpad_wave:
            return None
    eta_vals = [eta for _, _, _, eta in planned]
    ok_grp, reason = validate_grouped_launch(world, target, planned)
    if not ok_grp:
        world.add_debug(f"CAMPAIGN_REJECT_NO_CONVERSION target=p{target.id} reason={reason}")
        return None
    if not world.can_hold_after_capture(target, max(eta_vals), total):
        return None
    if total < normalize_send_amount(required_total):
        return None
    return MissionProposal(
        kind="SYNC_ATTACK",
        target_id=target.id,
        priority=170.0 + int(target.production) * 10.0,
        required_ships=total,
        planned_sources=planned,
        eta_min=min(eta_vals),
        eta_max=max(eta_vals),
        reason=f"surplus_conversion p{target.id} required={required_total}",
    )


def _run_surplus_extra_pickup(world, states, chain_plan, moves, deadline, blocked_target_id=None):
    candidates = sorted(
        [t for t in world.normal_planets
         if t.owner != world.player
         and t.id != blocked_target_id
         and not world.is_comet(t)
         and min((dp(p, t) for p in world.my_planets), default=999.0) <= 58.0],
        key=lambda t: (
            0 if t.owner == -1 else 1,
            min((dp(p, t) for p in world.my_planets), default=999.0),
            -int(t.production),
            int(t.ships),
        ),
    )
    for target in candidates[:10]:
        if time.perf_counter() > deadline:
            return False
        src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
        if target.owner not in (-1, world.player) and not (
            is_local_enemy_opportunity(world, target)
            or _small_radius_target_allowed(world, target, src, chain_plan)
        ):
            continue
        if not (
            _fallback_chain_value(world, target, chain_plan, None)
            or len(_campaign_followup_options(world, target)) >= 1
            or _small_radius_target_allowed(world, target, src, chain_plan)
        ):
            continue
        planned, total, ok = _fund_capture(
            world,
            target,
            states,
            max_sources=3,
            mission_reason="surplus_conversion_pickup",
            hold_margin_override=2 if target.owner == -1 else max(8, int(target.production) * 3),
            source_radius=58.0,
        )
        if not ok:
            continue
        eta_vals = [eta for _, _, _, eta in planned]
        ok_grp, _ = validate_grouped_launch(world, target, planned)
        if not ok_grp:
            continue
        if not world.can_hold_after_capture(target, max(eta_vals), total):
            continue
        mission_type = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = MissionProposal(
            kind=mission_type,
            target_id=target.id,
            priority=92.0,
            required_ships=total,
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"surplus_conversion_pickup p{target.id}",
        )
        if _commit_proposal(world, prop, moves):
            return True
    return False


def _run_short_surplus_rally(world, states, staging_id, moves, deadline):
    staging = world.planet_by_id.get(staging_id) if staging_id else None
    if staging is None:
        return False
    rally_sources = []
    for src in _safe_surplus_sources_for_pressure(world, states):
        if src.id == staging.id:
            continue
        spare = min(
            states[src.id].safe_surplus,
            int(src.ships) - world.committed.get(src.id, 0) - states[src.id].reserve,
        )
        if states[src.id].role == ROLE_STORAGE:
            spare = min(spare, 45)
        send = round_down_to_granularity(spare)
        if send < MIN_SEND_SHIPS:
            continue
        eta = world.eta(src, staging, send)
        if eta > 5:
            world.add_debug(f"LONG_RALLY_REJECTED src=p{src.id} staging=p{staging.id} eta={eta:.1f}")
            continue
        if dp(src, staging) > 42.0:
            world.add_debug(f"LONG_RALLY_REJECTED src=p{src.id} staging=p{staging.id} dist={dp(src, staging):.1f}")
            continue
        rally_sources.append((eta, src, send))

    rally_sources.sort(key=lambda item: (item[0], -item[2]))
    acted = False
    for _eta, src, send in rally_sources[:6]:
        if time.perf_counter() > deadline:
            break
        if world.commit(src, staging, send, moves, mission_type="REINFORCE_CAPTURE"):
            acted = True
    if acted:
        world.add_debug(f"RALLY_TO_STAGING_SELECTED staging=p{staging.id}")
        world.add_debug(f"SHORT_RALLY_APPROVED staging=p{staging.id}")
    return acted


def run_surplus_conversion_mode(world, states, chain_plan, staging_id, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    active, sources, targets, parked, _flying_ratio, threshold = _surplus_pressure_context(
        world, states, chain_plan, emit=True
    )
    if not active:
        return False

    world.add_debug("SURPLUS_CONVERSION_MODE")
    world.add_debug("PRODUCTION_BANK_BLOCKED_BY_PARKED_SURPLUS")
    world.add_debug(f"SAFE_SURPLUS_SOURCES_SELECTED sources={[p.id for p in sources[:8]]}")
    for target in targets[:8]:
        if time.perf_counter() > deadline:
            break
        prop = _build_surplus_attack_proposal(world, target, states, sources)
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug(f"DIRECT_GROUPED_ATTACK_SELECTED target=p{target.id} total={prop.required_ships}")
            world.add_debug(f"PARKED_ARMY_ATTACK_APPROVED target=p{target.id}")
            world.add_debug("NO_IDLE_ARMY_WHEN_VALID_ATTACK_EXISTS")
            if parked >= threshold * 1.8 and time.perf_counter() < deadline:
                if _run_surplus_extra_pickup(world, states, chain_plan, moves, deadline, blocked_target_id=target.id):
                    world.add_debug("MULTI_TARGET_PRESSURE_APPROVED")
            return True

    if _run_short_surplus_rally(world, states, staging_id, moves, deadline):
        return True
    return False


# ── Surplus rally ─────────────────────────────────────────────────────────────

def rally_to_staging(world, staging_id, states, moves):
    """
    Move safe surplus from storage/bridge planets to the staging launchpad.
    Never hollows any source.  Each transfer is a valid packet.
    """
    staging = world.planet_by_id.get(staging_id)
    if staging is None:
        return False
    acted = False
    for p in world.my_planets:
        if p.id == staging_id:
            continue
        st = states.get(p.id)
        if st is None or st.threatened:
            continue
        if st.role == ROLE_LAUNCHPAD and int(p.ships) > LAUNCHPAD_RESERVE * 1.6:
            continue   # large launchpads keep their force
        if dp(p, staging) > CHAIN_RADIUS:
            continue
        send = round_down_to_granularity(
            min(st.safe_surplus,
                int(p.ships) - world.committed.get(p.id, 0) - st.reserve)
        )
        if send < MIN_SEND_SHIPS:
            continue
        if int(p.ships) - send < st.reserve:
            world.add_debug(
                f"RALLY_REJECT_WOULD_HOLLOW src=p{p.id} "
                f"ships={int(p.ships)} send={send} reserve={st.reserve}"
            )
            continue
        if world.commit(p, staging, send, moves, mission_type="REINFORCE_CAPTURE"):
            world.add_debug(
                f"SURPLUS_RALLY_TO_STAGING src=p{p.id} "
                f"staging=p{staging_id} send={send}"
            )
            acted = True
    return acted


# ── Rolling capture chain ─────────────────────────────────────────────────────

def _forecast_chain_force(world, states, turns=5):
    return sum(st.safe_surplus for st in states.values()) + sum(int(p.production) for p in world.my_planets) * turns


def _next_unowned_after(world, chain_plan, pid):
    seen = False
    for next_pid in chain_plan:
        if next_pid == pid:
            seen = True
            continue
        if not seen:
            continue
        nxt = world.planet_by_id.get(next_pid)
        if nxt is not None and nxt.owner != world.player and not world.is_comet(nxt):
            return nxt
    return None


def _chain_next_node_verified(world, current, next_node, states, forecast_force):
    if next_node is None:
        return True
    if dp(current, next_node) > CHAIN_RADIUS + 14:
        world.add_debug(
            f"ROLLING_CHAIN_REJECT_BREAKS_AFTER_ONE current=p{current.id} next=p{next_node.id} reason=distance"
        )
        return False
    role = _planet_role(current)
    if role == ROLE_STORAGE and not _chain_small_has_value(world, current, next_node):
        world.add_debug(
            f"ROLLING_CHAIN_REJECT_BREAKS_AFTER_ONE current=p{current.id} next=p{next_node.id} reason=storage"
        )
        return False
    projected_need = world.ships_needed_to_capture(current, next_node, max(MIN_SEND_SHIPS, forecast_force))
    hold_margin = max(5, int(next_node.production) * 2)
    if projected_need + hold_margin > max(MIN_SEND_SHIPS, forecast_force):
        world.add_debug(
            f"ROLLING_CHAIN_REJECT_BREAKS_AFTER_ONE current=p{current.id} next=p{next_node.id} "
            f"need={projected_need + hold_margin} forecast={forecast_force}"
        )
        return False
    if not world.can_hold_after_capture(next_node, min(30, world.remaining), projected_need + hold_margin):
        world.add_debug(
            f"ROLLING_CHAIN_REJECT_BREAKS_AFTER_ONE current=p{current.id} next=p{next_node.id} reason=hold"
        )
        return False
    world.add_debug(f"ROLLING_CHAIN_NEXT_NODE_VERIFIED current=p{current.id} next=p{next_node.id}")
    return True


def build_rolling_capture_chain(world, staging_id, states, chain_plan, fleet_ratio, prediction=None, forecasts=None):
    """
    Build proposals only when target_1, target_2, and target_3 form a verified route.
    The agent still commits at most one funded mission this turn.
    """
    proposals = []
    if fleet_ratio > FLEET_RATIO_SOFT:
        return proposals

    staging = world.planet_by_id.get(staging_id)
    if staging is None:
        return proposals
    if forecasts:
        strongest = max((f["projected_ships"] for f in forecasts.values()), default=0)
        world.add_debug(f"CHAIN_RETRIGGER_FORCE_FORECAST rolling_safe={_forecast_chain_force(world, states, 5)} opp_max={strongest}")

    for pid in chain_plan:
        target = world.planet_by_id.get(pid)
        if (target is None
                or target.owner == world.player
                or world.is_comet(target)):
            continue
        if world.incoming_to_targets.get(pid, 0) >= world.required_ships_to_capture(target):
            continue

        planned, total, ok = _fund_capture(world, target, states)
        if not ok:
            world.add_debug(
                f"ROLLING_CHAIN_LAUNCH_REJECTED target=p{pid} "
                f"total={total} reason=cannot_fund_valid_packets"
            )
            continue

        eta_vals = [e for _, _, _, e in planned]
        ok_grp, grp_reason = validate_grouped_launch(world, target, planned)
        if not ok_grp:
            world.add_debug(
                f"ROLLING_CHAIN_LAUNCH_REJECTED target=p{pid} reason={grp_reason}"
            )
            continue
        if not world.can_hold_after_capture(target, max(eta_vals), total):
            world.add_debug(
                f"ROLLING_CHAIN_LAUNCH_REJECTED target=p{pid} reason=not_holdable"
            )
            continue
        next_2 = _next_unowned_after(world, chain_plan, pid)
        next_3 = _next_unowned_after(world, chain_plan, next_2.id) if next_2 is not None else None
        forecast_5 = _forecast_chain_force(world, states, 5)
        forecast_10 = _forecast_chain_force(world, states, 10)
        campaign_phase = len(world.my_planets) < 6 or world.step < 80
        if next_2 is not None and not _chain_next_node_verified(world, target, next_2, states, forecast_5):
            if campaign_phase:
                world.add_debug(f"CAMPAIGN_FOLLOWUP_WEAK_BUT_ALLOWED target=p{target.id} next=p{next_2.id}")
            else:
                continue
        if next_2 is not None and next_3 is not None and not _chain_next_node_verified(world, next_2, next_3, states, forecast_10):
            if campaign_phase:
                world.add_debug(f"CAMPAIGN_FOLLOWUP_WEAK_BUT_ALLOWED target=p{target.id} next=p{next_3.id}")
            else:
                continue

        mtype = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop  = MissionProposal(
            kind            = mtype,
            target_id       = pid,
            priority        = _chain_planet_score(world, target) + 50.0,
            required_ships  = total,
            planned_sources = planned,
            eta_min         = min(eta_vals),
            eta_max         = max(eta_vals),
            reason          = f"rolling_chain staging=p{staging_id}",
        )
        world.add_debug(
            f"ROLLING_CHAIN_PLAN_BUILT target=p{pid} total={total} "
            f"sources={len(planned)} eta={max(eta_vals):.1f}"
        )
        proposals.append(prop)
        if len(proposals) >= 2:
            break

    if proposals:
        world.add_debug("ROLLING_CHAIN_LAUNCH_APPROVED")
    return proposals


# ── Attack source identification ──────────────────────────────────────────────

def identify_attack_source(world, lost_planet):
    """
    Identify the most likely enemy planet that launched the capture of lost_planet.
    Uses proximity, recent ship drops, and production capacity.
    """
    world.add_debug(
        f"ATTACK_SOURCE_IDENTIFIED scanning for attacker of p{lost_planet.id}"
    )
    best       = None
    best_score = -1e9
    for e in world.enemy_planets:
        if world.is_comet(e):
            continue
        d = dp(e, lost_planet)
        if d > ATTACK_SOURCE_RADIUS:
            continue
        s = max(0.0, ATTACK_SOURCE_RADIUS - d) * 2.0 + int(e.production) * 8.0
        prev = _prev_ships.get(e.id)
        if prev is not None:
            drop = (prev + int(e.production)) - int(e.ships)
            if drop >= 10:
                s += drop * 1.5
                world.add_debug(
                    f"ATTACK_SOURCE_CLUSTER_IDENTIFIED p{e.id} drop={drop}"
                )
        if s > best_score:
            best_score = s
            best = e
    if best:
        world.add_debug(
            f"ATTACK_SOURCE_IDENTIFIED p{best.id} score={best_score:.1f}"
        )
    return best


# ── Chain retrigger response ──────────────────────────────────────────────────

def _is_core_chain_planet(world, planet, chain_plan=None):
    role = _planet_role(planet)
    return (
        role == ROLE_LAUNCHPAD
        or (role == ROLE_BRIDGE and (is_idle(planet) or int(planet.production) >= 3))
        or int(planet.production) >= 4
        or (chain_plan is not None and planet.id in set(chain_plan[:5]))
    )


def build_chain_retrigger_response(world, lost_planet, states, chain_plan=None, prediction=None, forecasts=None):
    """
    When a planet is lost, attack the likely source/base first when safe.
    Cheap recapture is only a core emergency fallback.

    Returns a MissionProposal (or None if banking is chosen).
    """
    world.add_debug(f"PLANET_LOST_CHAIN_RETRIGGERED p{lost_planet.id}")
    world.add_debug(f"CORE_CHAIN_RECOVERY_POLICY_SOURCE_FIRST p{lost_planet.id}")

    source = identify_attack_source(world, lost_planet)
    safe_surplus = sum(st.safe_surplus for st in states.values())
    chain_prod_5 = sum(int(p.production) for p in world.my_planets) * 5
    chain_prod_10 = sum(int(p.production) for p in world.my_planets) * 10
    chain_force_5 = safe_surplus + chain_prod_5
    chain_force_10 = safe_surplus + chain_prod_10
    strongest_opp = max((f["projected_ships"] for f in (forecasts or {}).values()), default=0)

    world.add_debug(
        f"CHAIN_RETRIGGER_FORCE_FORECAST safe={safe_surplus} "
        f"force5={chain_force_5} force10={chain_force_10} opp_max={strongest_opp}"
    )

    if source is not None:
        source_need_src = min(world.my_planets, key=lambda p: dp(p, source), default=None)
        source_need = world.ships_needed_to_capture(source_need_src, source, max(chain_force_10, 1)) if source_need_src else chain_force_10 + 1
        source_hold = max(8, int(source.production) * 4)
        enough_force = chain_force_5 >= CHAIN_FORCE_MIN or chain_force_10 >= source_need + source_hold
        if enough_force:
            planned, total, ok = _fund_capture(world, source, states)
            if ok:
                eta_vals = [e for _, _, _, e in planned]
                ok_grp, grp_reason = validate_grouped_launch(world, source, planned)
                if ok_grp and world.can_hold_after_capture(source, max(eta_vals), total):
                    world.add_debug(f"ATTACK_SOURCE_AS_CHAIN_TARGET p{source.id}")
                    world.add_debug(f"CHAIN_RETRIGGER_ROUTE_BUILT p{source.id}")
                    if dp(source, lost_planet) <= CHAIN_RADIUS + 14:
                        world.add_debug(
                            f"SQUARE_CHAIN_RECOVERY_AFTER_SOURCE source=p{source.id} lost=p{lost_planet.id}"
                        )
                        world.add_debug(f"LOST_PLANET_RECOVERED_BY_CHAIN p{lost_planet.id}")
                    return MissionProposal(
                        kind="SYNC_ATTACK",
                        target_id=source.id,
                        priority=125.0,
                        required_ships=total,
                        planned_sources=planned,
                        eta_min=min(eta_vals),
                        eta_max=max(eta_vals),
                        reason=f"chain_source_attack lost=p{lost_planet.id} source=p{source.id}",
                    )
                world.add_debug(f"ATTACK_SOURCE_REJECT_NO_FOLLOWUP_CHAIN reason={grp_reason}")
            else:
                world.add_debug(
                    f"ATTACK_SOURCE_REJECT_FORCE_TOO_SMALL p{source.id} cannot_fund total={total}"
                )
        else:
            world.add_debug(
                f"ATTACK_SOURCE_REJECT_FORCE_TOO_SMALL p{source.id} "
                f"need={source_need + source_hold} force10={chain_force_10}"
            )
    else:
        world.add_debug(f"ATTACK_SOURCE_IDENTIFIED none lost=p{lost_planet.id}")

    if not _is_core_chain_planet(world, lost_planet, chain_plan):
        return None

    world.add_debug(f"CORE_CHAIN_RECOVERY_ALLOWED p{lost_planet.id}")
    nearby = sorted(
        [p for p in world.my_planets
         if dp(p, lost_planet) <= 34.0
         and p.id in states
         and states[p.id].safe_surplus >= MIN_SEND_SHIPS
         and not states[p.id].threatened],
        key=lambda p: dp(p, lost_planet),
    )
    near_pool = sum(states[p.id].safe_surplus for p in nearby)
    src_near = min(world.my_planets, key=lambda p: dp(p, lost_planet), default=None)
    need = world.ships_needed_to_capture(src_near, lost_planet, near_pool) if src_near else near_pool + 1
    if need <= 0 or need > near_pool:
        return None
    near_states = {p.id: states[p.id] for p in nearby}
    planned, total, ok = _fund_capture(world, lost_planet, near_states)
    if not ok:
        return None
    eta_vals = [e for _, _, _, e in planned]
    ok_grp, _ = validate_grouped_launch(world, lost_planet, planned)
    if not ok_grp or not world.can_hold_after_capture(lost_planet, max(eta_vals), total):
        return None
    world.add_debug(f"CORE_CHAIN_RECOVERY_SELECTED p{lost_planet.id} total={total}")
    return MissionProposal(
        kind="CORE_CHAIN_RECOVERY",
        target_id=lost_planet.id,
        priority=111.0,
        required_ships=total,
        planned_sources=planned,
        eta_min=min(eta_vals),
        eta_max=max(eta_vals),
        reason=f"core_chain_recovery p{lost_planet.id}",
    )


# ── Emergency defense (simplified, packet-safe) ───────────────────────────────

def emergency_defense_chain(world, states, moves):
    """
    Reinforce only planets that can actually be saved with valid packets.
    Never hollows launchpads for low-value storage planets.
    """
    for tgt in sorted(world.my_planets,
                      key=lambda p: -world.real_incoming_threat(p)["deficit"]):
        deficit = world.real_incoming_threat(tgt)["deficit"]
        if deficit <= 0:
            continue
        st = states.get(tgt.id)
        # Skip tiny storage planets; do not hollow launchpads for them
        if st and st.production <= 1 and st.role == ROLE_STORAGE:
            continue
        needed = normalize_send_amount(deficit)
        srcs = sorted(
            [p for p in world.my_planets
             if p.id != tgt.id
             and p.id in states
             and states[p.id].safe_surplus >= MIN_SEND_SHIPS
             and not states[p.id].threatened
             and dp(p, tgt) <= CHAIN_RADIUS],
            key=lambda p: dp(p, tgt),
        )
        sent = 0
        for src in srcs[:4]:
            if sent >= needed:
                break
            raw = min(states[src.id].safe_surplus, needed - sent)
            wanted = normalize_send_amount(raw)
            contrib = wanted if states[src.id].safe_surplus >= wanted else round_down_to_granularity(raw)
            if contrib < MIN_SEND_SHIPS:
                continue
            if world.commit(src, tgt, contrib, moves, mission_type="DEFEND_HOLD"):
                sent += contrib
                world.add_debug(
                    f"EMERGENCY_DEFENSE src=p{src.id} tgt=p{tgt.id} send={contrib}"
                )


# ── Endgame drain ─────────────────────────────────────────────────────────────

def _final_drain_target_value(world, target, chain_plan):
    if target.owner == world.player or world.is_comet(target):
        return False
    if target.id in set(chain_plan[:10]):
        return True
    if target.owner not in (-1, world.player):
        return True
    return (
        int(target.production) >= 4
        or (is_idle(target) and _planet_role(target) in (ROLE_LAUNCHPAD, ROLE_BRIDGE))
        or _chain_small_has_value(world, target)
    )


def _final_drain_chain(world, moves, chain_plan=None):
    """Endgame drain only into chain/collapse targets that can convert before time."""
    chain_plan = chain_plan or []
    for src in sorted(world.my_planets, key=lambda p: -int(p.ships)):
        spare = round_down_to_granularity(
            max(0, int(src.ships) - world.committed.get(src.id, 0) - 1)
        )
        if spare < MIN_SEND_SHIPS:
            continue
        targets = [
            tgt for tgt in world.enemy_planets + world.neutral_planets
            if _final_drain_target_value(world, tgt, chain_plan)
        ]
        for tgt in sorted(
            targets,
            key=lambda t: (
                0 if t.id in set(chain_plan[:10]) else 1,
                0 if t.owner not in (-1, world.player) else 1,
                world.eta(src, t, spare),
                -int(t.production),
            ),
        ):
            if world.is_comet(tgt):
                continue
            if world.eta(src, tgt, spare) > world.remaining - 1:
                continue
            need = world.ships_needed_to_capture(src, tgt, spare)
            send = normalize_send_amount(need)
            if send < MIN_SEND_SHIPS or send > spare:
                continue
            if not world.can_hold_after_capture(tgt, world.eta(src, tgt, send), send, final_all_in=True):
                continue
            if world.commit(src, tgt, send, moves, mission_type="FINAL_DRAIN"):
                break


def _fallback_chain_value(world, target, chain_plan, prediction=None):
    if target.owner == world.player or world.is_comet(target):
        return False
    if target.id in chain_plan[:8]:
        return True
    if _prev_owners.get(target.id) == world.player and _is_core_chain_planet(world, target, chain_plan):
        return True
    role = _planet_role(target)
    followups = _campaign_followup_options(world, target)
    if role == ROLE_STORAGE and not _chain_small_has_value(world, target) and len(followups) < 1:
        return False
    if len(followups) >= 2:
        return True
    if target.owner == -1 and len(followups) >= 1 and min((dp(m, target) for m in world.my_planets), default=999.0) <= CAMPAIGN_RADIUS:
        return True
    if is_idle(target) and role in (ROLE_LAUNCHPAD, ROLE_BRIDGE):
        return True
    if int(target.production) >= 4:
        return True
    if target.owner not in (-1, world.player) and is_local_enemy_opportunity(world, target):
        return True
    return False


# ── main35 hybrid tempo imports from main33 ───────────────────────────────────

def _main35_tempo_states(world, states):
    """
    Use main34 PlanetState shape, but relax early expansion surplus accounting.
    Source safety still runs later through world.valid_fleet_launch().
    """
    if not (world.step < 100 or len(world.my_planets) < 6):
        return states
    tempo = dict(states)
    small_escape_start = _start_planet_current(world) if _small_start_escape_needed(world) else None
    for p in world.my_planets:
        st = states.get(p.id)
        if st is None or st.threatened:
            continue
        reserve = st.reserve
        if _small_start_escape_mode(world, p) or (small_escape_start is not None and p.id == small_escape_start.id):
            reserve = min(reserve, 3)
        else:
            reserve = min(reserve, max(2, int(p.production) + 3))
        safe_surplus = max(0, int(p.ships) - world.committed.get(p.id, 0) - reserve)
        if safe_surplus > st.safe_surplus:
            tempo[p.id] = PlanetState(
                planet_id=st.planet_id,
                role=st.role,
                is_static=st.is_static,
                ships=st.ships,
                production=st.production,
                radius=st.radius,
                reserve=reserve,
                safe_surplus=safe_surplus,
                threatened=st.threatened,
                cluster_d=st.cluster_d,
            )
    return tempo


def _main35_make_capture_prop(
    world,
    states,
    target,
    mission_reason,
    priority,
    *,
    mission_kind=None,
    max_sources=4,
    source_radius=CAMPAIGN_RADIUS,
    hold_margin=None,
    require_hold=True,
):
    if target is None or target.owner == world.player or world.is_comet(target):
        return None
    if target.owner not in (-1, world.player):
        if not should_allow_enemy_attack(world, target, "SYNC_ATTACK", mission_reason):
            return None
    if not _distance_discipline_allows_target(world, states, target, mission_reason):
        return None
    if not _structure_safe_for_deep_capture(world, states, getattr(world, "_active_chain_plan", []), target, mission_reason):
        return None
    local_states = _main35_tempo_states(world, states)
    margin = hold_margin
    if margin is None:
        margin = 2 if target.owner == -1 else max(8, int(target.production) * 3)
    margin = _dynamic_hold_margin(world, target, margin)
    planned, total, ok = _fund_capture(
        world,
        target,
        local_states,
        max_sources=max_sources,
        mission_reason=f"expansion_obligation_main35_{mission_reason}",
        hold_margin_override=margin,
        source_radius=source_radius,
    )
    if not ok:
        return None
    kind = mission_kind or ("CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK")
    planned = _top_up_capture_to_critical_mass(
        world,
        target,
        local_states,
        planned,
        kind,
        f"expansion_obligation_main35_{mission_reason}",
        source_radius,
        max_sources,
    )
    total = sum(s for _, s, _, _ in planned)
    for _src_id, ships, _angle, _eta in planned:
        if not valid_packet_size(kind, ships):
            world.add_debug(f"INVALID_PACKET_REJECTED_MAIN35 target=p{target.id} ships={ships}")
            return None
    ok_grp, reason = validate_grouped_launch(world, target, planned)
    if not ok_grp:
        world.add_debug(f"MAIN35_CAPTURE_REJECT target=p{target.id} reason={reason}")
        return None
    eta_vals = [eta for _, _, _, eta in planned]
    if require_hold and not world.can_hold_after_capture(target, max(eta_vals), total):
        world.add_debug(f"MAIN35_CAPTURE_REJECT target=p{target.id} reason=not_holdable")
        return None
    return MissionProposal(
        kind=kind,
        target_id=target.id,
        priority=priority,
        required_ships=total,
        planned_sources=planned,
        eta_min=min(eta_vals),
        eta_max=max(eta_vals),
        reason=f"expansion_obligation_main35_{mission_reason} p{target.id}",
    )


def _useful_target_value(world, target, anchor=None):
    if target is None or target.owner == world.player or world.is_comet(target):
        return False
    if _small_radius_target_allowed(world, target, anchor):
        return True
    if int(target.production) >= 2:
        return True
    if _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD):
        return True
    if is_idle(target):
        return True
    if _chain_small_has_value(world, target, anchor):
        return True
    if len(_campaign_followup_options(world, target)) >= 1:
        return True
    if target.owner not in (-1, world.player) and is_local_enemy_opportunity(world, target):
        return True
    return False


def _nearest_useful_target_candidate(world, states, chain_plan, exclude_target_id=None, build_prop=False):
    best = None
    for src in world.my_planets:
        st = states.get(src.id)
        if st is None or st.threatened or st.safe_surplus < MIN_SEND_SHIPS:
            continue
        for target in world.normal_planets:
            if target.id == exclude_target_id or target.owner == world.player or world.is_comet(target):
                continue
            if not _useful_target_value(world, target, src):
                continue
            if target.owner not in (-1, world.player) and not (
                is_local_enemy_opportunity(world, target)
                or _small_radius_target_allowed(world, target, src, chain_plan)
            ):
                continue
            d = dp(src, target)
            if d > 42.0:
                continue
            rough_need = max(MIN_SEND_SHIPS, normalize_send_amount(world.required_ships_to_capture(target, src)))
            eta = world.eta(src, target, rough_need)
            if not (eta <= 18.0 or d <= 32.0):
                continue
            hold = 2 if target.owner == -1 else max(8, int(target.production) * 3)
            plan, total, ok = _fund_capture(
                world,
                target,
                states,
                max_sources=4,
                mission_reason="nearest_useful_target_lock",
                hold_margin_override=hold,
                source_radius=42.0,
            )
            if not ok:
                continue
            eta_vals = [e for _, _, _, e in plan]
            ok_grp, _reason = validate_grouped_launch(world, target, plan)
            if not ok_grp:
                continue
            if not world.can_hold_after_capture(target, max(eta_vals), total):
                continue
            roi = capture_conversion_score(world, target, src, total)
            score = (
                d * 5.0
                + eta * 4.0
                + int(target.ships) * 0.6
                - int(target.production) * 18.0
                - (55.0 if is_idle(target) else 0.0)
                - (35.0 if _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD) else 0.0)
                - min(45.0, roi * 18.0)   # high ROI → lower sort-key → preferred
            )
            if is_idle(target):
                world.add_debug(f"STATIC_TARGET_PRIORITY p{target.id}")
            if roi > 1.5:
                world.add_debug(f"CAPTURE_CONVERSION_PRIORITY_ACTIVE p{target.id} roi={roi:.2f}")
            prop = None
            if build_prop:
                kind = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
                prop = MissionProposal(
                    kind=kind,
                    target_id=target.id,
                    priority=152.0 - score * 0.02,
                    required_ships=total,
                    planned_sources=plan,
                    eta_min=min(eta_vals),
                    eta_max=max(eta_vals),
                    reason=f"nearest_useful_target_lock p{target.id}",
                )
            item = (score, d, eta, target, prop)
            if best is None or item[:3] < best[:3]:
                best = item
    return best


def _local_direct_limit(world):
    if world.step <= FORCED_OPENING_STEP or len(world.my_planets) <= 4:
        return LOCAL_DIRECT_OPENING_DIST
    if world.step <= MIDGAME_END_STEP:
        return LOCAL_DIRECT_MIDGAME_DIST
    return CAPTURE_OPP_MAX_DIST + 12.0


def _nearest_direct_source(world, target):
    return min(world.my_planets, key=lambda p: dp(p, target), default=None)


def _closer_useful_target_exists(world, states, target, chain_plan):
    near = _nearest_useful_target_candidate(world, states, chain_plan, exclude_target_id=target.id)
    if near is None:
        return False, None
    src = _nearest_direct_source(world, target)
    target_d = dp(src, target) if src is not None else 999.0
    return near[1] + 6.0 < target_d, near


def _select_bridge_route_target(world, states, final_target, chain_plan):
    src = _nearest_direct_source(world, final_target)
    if src is None:
        return None
    direct = dp(src, final_target)
    candidates = []
    for node in world.normal_planets:
        if node.id == final_target.id or node.owner == world.player or world.is_comet(node):
            continue
        d1 = dp(src, node)
        d2 = dp(node, final_target)
        if d1 > _local_direct_limit(world) or d2 > max(BRIDGE_RELAY_DIST, direct * 0.72):
            continue
        if d1 + d2 >= direct * 1.05:
            continue
        if not _useful_target_value(world, node, src):
            continue
        if node.owner not in (-1, world.player) and not is_local_enemy_opportunity(world, node):
            continue
        role = _planet_role(node)
        score = (
            (direct - (d1 + d2)) * 4.0
            + int(node.production) * 55.0
            + (95.0 if role == ROLE_LAUNCHPAD else 55.0 if role == ROLE_BRIDGE else 0.0)
            + (70.0 if is_static_planet(node) else 0.0)
            + (45.0 if node.id in set(chain_plan[:18]) else 0.0)
            - int(node.ships) * 0.8
        )
        candidates.append((score, d1, int(node.ships), node.id, node))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][4] if candidates else None


def _far_direct_attack_blocked(world, states, target, mission_reason):
    """Opening/midgame gate: far targets need high confidence and no local work."""
    reason = mission_reason or ""
    if any(tag in reason for tag in ("defense", "recovery", "final", "collapse", "game_winning")):
        return False
    if world.step > MIDGAME_END_STEP and len(world.my_planets) >= 8:
        return False
    src = _nearest_direct_source(world, target)
    if src is None:
        return False
    rough_need = max(MIN_SEND_SHIPS, normalize_send_amount(world.required_ships_to_capture(target, src)))
    eta = world.eta(src, target, rough_need)
    nearest = dp(src, target)
    if nearest <= _local_direct_limit(world):
        world.add_debug(f"LOCAL_CLUSTER_TARGET_SELECTED p{target.id} d={nearest:.1f}")
        return False
    chain_plan = getattr(world, "_active_chain_plan", [])
    closer_exists, near = _closer_useful_target_exists(world, states, target, chain_plan)
    critical = (
        target.owner not in (-1, world.player)
        and (target.owner == world.leader or int(target.production) >= 4 or _planet_role(target) == ROLE_LAUNCHPAD)
    ) or int(target.production) >= 5
    aim_ok, aim_reason = world.aim_confidence_check(
        src, target, rough_need, "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
    )
    if aim_ok and eta <= FAR_DIRECT_MAX_ETA and critical and not closer_exists:
        return False
    if near is not None:
        world.add_debug(
            f"FAR_DIRECT_TARGET_REJECTED p{target.id} near=p{near[3].id} "
            f"d={nearest:.1f} eta={eta:.1f}"
        )
    else:
        world.add_debug(
            f"FAR_DIRECT_TARGET_REJECTED p{target.id} d={nearest:.1f} eta={eta:.1f} aim={aim_reason}"
        )
    bridge = _select_bridge_route_target(world, states, target, chain_plan)
    if bridge is not None:
        world.add_debug(f"BRIDGE_ROUTE_REQUIRED final=p{target.id} bridge=p{bridge.id}")
    return True


def _distance_discipline_allows_target(world, states, target, mission_reason):
    if target is None:
        return True
    if target.owner != world.player and _far_direct_attack_blocked(world, states, target, mission_reason):
        return False
    if world.step > 120:
        return True
    control_ratio = len(world.my_planets) / max(1, len(world.normal_planets))
    if control_ratio < PHASE_INITIAL_MAX and target.owner not in (-1, world.player):
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > 55.0 or not is_local_enemy_opportunity(world, target):
            world.add_debug(f"INITIAL_EXPANSION_BLOCKED_FAR_TARGET p{target.id}")
            return False
    if any(tag in (mission_reason or "") for tag in ("nearest_useful", "small_start_escape", "defense", "recovery", "final", "collapse")):
        return True
    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if src is None:
        return True
    rough_need = max(MIN_SEND_SHIPS, normalize_send_amount(world.required_ships_to_capture(target, src)))
    eta = world.eta(src, target, rough_need)
    d = dp(src, target)
    if eta <= 22.0 and d <= 42.0:
        return True
    world.add_debug("OPENING_DISTANCE_DISCIPLINE")
    near = _nearest_useful_target_candidate(world, states, getattr(world, "_active_chain_plan", []), exclude_target_id=target.id)
    if near is None:
        if eta > 28.0 and not is_idle(target):
            world.add_debug(f"FAR_EXPANSION_BLOCKED target=p{target.id} eta={eta:.1f}")
            return False
        return True
    if int(target.production) >= 5 and world.can_hold_after_capture(target, eta, rough_need):
        aim_ok, _ = world.aim_confidence_check(src, target, rough_need, "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK")
        if aim_ok and _target_improves_control(world, target, getattr(world, "_active_chain_plan", [])):
            world.add_debug(f"FAR_TARGET_ALLOWED_HIGH_VALUE target=p{target.id} eta={eta:.1f}")
            return True
    world.add_debug(f"FAR_TARGET_REJECTED_NEARER_EXISTS target=p{target.id} near=p{near[3].id}")
    world.add_debug("LOCAL_EXPANSION_FIRST")
    if eta > 22.0:
        world.add_debug(f"FAR_EXPANSION_BLOCKED target=p{target.id} eta={eta:.1f}")
    return False


def run_nearest_useful_target_lock(world, states, chain_plan, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    candidate = _nearest_useful_target_candidate(world, states, chain_plan, build_prop=True)
    if candidate is None:
        return False
    _score, _d, _eta, target, prop = candidate
    if prop is None:
        return False
    world.add_debug("NEAREST_USEFUL_TARGET_LOCK")
    world.add_debug(f"NEAREST_TARGET_SELECTED p{target.id} d={_d:.1f} eta={_eta:.1f}")
    if _commit_proposal(world, prop, moves):
        return True
    return False


def main35_opening_tempo(world, states, chain_plan, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    if world.step > FORCED_OPENING_STEP and len(world.my_planets) > FORCED_OPENING_PLANETS:
        return False
    if not world.neutral_planets:
        return False

    target_order = []
    chain_prop = opening_chain_plan(world, deadline)
    if chain_prop is not None:
        target_order.append(world.planet_by_id.get(chain_prop.target_id))
    best = choose_best_opening_target(world)
    if best is not None:
        _src, tgt, _need, _angle = best
        target_order.append(tgt)

    scored = []
    for src in world.my_planets:
        for tgt in world.neutral_planets:
            if tgt in target_order or world.is_comet(tgt):
                continue
            if dp(src, tgt) > 62.0:
                continue
            if not validate_initial_target_choice(world, src, tgt):
                continue
            score = early_target_score(world, src, tgt)
            score += len(_campaign_followup_options(world, tgt)) * 35.0
            if int(tgt.production) >= 3:
                score += 45.0
            scored.append((score, dp(src, tgt), int(tgt.ships), tgt))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    target_order.extend(tgt for _score, _d, _ships, tgt in scored)

    seen = set()
    for tgt in target_order[:12]:
        if tgt is None or tgt.id in seen:
            continue
        seen.add(tgt.id)
        prop = _main35_make_capture_prop(
            world,
            states,
            tgt,
            "opening_tempo",
            PRIORITY_CHAIN_PLAN_BASE + int(tgt.production) * 8,
            max_sources=3,
            source_radius=64.0,
            hold_margin=1 if world.step < 35 else 2,
            require_hold=False,
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug("MAIN33_OPENING_TEMPO_SELECTED")
            world.add_debug(f"MAIN33_EXPANSION_IMPORTED opening target=p{tgt.id} total={prop.required_ships}")
            world.add_debug("EXPANSION_BEFORE_BANKING")
            return True
    return False


def main35_nearest_occupiable_arbiter(world, states, moves, deadline):
    if moves or time.perf_counter() > deadline or not world.my_planets:
        return False
    nearest = _find_best_nearest_for_arbiter(world)
    candidates = []
    if nearest is not None:
        tgt, src, need, score, _status = nearest
        candidates.append((score + 80.0, dp(src, tgt), tgt))

    for tgt in world.normal_planets:
        if time.perf_counter() > deadline:
            break
        if tgt.owner == world.player or world.is_comet(tgt):
            continue
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        d = dp(src, tgt)
        if d > ARBITER_NEAREST_MAX_DIST + 18:
            continue
        if tgt.owner not in (-1, world.player) and not (
            is_local_enemy_opportunity(world, tgt)
            or _small_radius_target_allowed(world, tgt, src)
        ):
            continue
        need = world.required_ships_to_capture(tgt, src)
        score = (
            max(0.0, 50.0 - d) * 5.5
            - need * 1.2
            + int(tgt.production) * 62.0
            + len(_campaign_followup_options(world, tgt)) * 35.0
        )
        if _planet_role(tgt) in (ROLE_BRIDGE, ROLE_LAUNCHPAD):
            score += 55.0
        if is_idle(tgt):
            score += 45.0
        candidates.append((score, d, tgt))

    candidates.sort(key=lambda item: (-item[0], item[1], int(item[2].ships)))
    for score, _d, tgt in candidates[:8]:
        prop = _main35_make_capture_prop(
            world,
            states,
            tgt,
            "nearest_occupiable",
            PRIORITY_NEAREST_OCCUPIABLE + score * 0.05,
            max_sources=4,
            source_radius=ARBITER_NEAREST_MAX_DIST + 18,
            hold_margin=2 if tgt.owner == -1 else max(8, int(tgt.production) * 3),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug("MAIN33_NEAREST_OCCUPIABLE_SELECTED")
            world.add_debug(f"EXPANSION_BEFORE_BANKING target=p{tgt.id}")
            return True
    return False


def main35_high_value_neutral_race(world, states, moves, deadline):
    if moves or time.perf_counter() > deadline or not world.neutral_planets:
        return False
    valuable = [
        n for n in world.neutral_planets
        if not world.is_comet(n)
        and int(n.production) >= LOCAL_PRODUCTION_MIN_PROD
        and min((dp(p, n) for p in world.my_planets), default=999.0) <= max(LOCAL_PRODUCTION_MAX_DIST, CAMPAIGN_RADIUS + 18)
    ]
    valuable.sort(
        key=lambda n: (
            -int(n.production),
            0 if int(n.production) >= LOCAL_PRODUCTION_PREMIER_PROD else 1,
            min((dp(p, n) for p in world.my_planets), default=999.0),
            int(n.ships),
        )
    )
    for tgt in valuable[:8]:
        my_eta, enemy_eta = world.reaction_times(tgt)
        race = world.enemy_incoming_to_targets.get(tgt.id, 0) > 0 or enemy_eta <= my_eta + LOCAL_PRODUCTION_RACE_MARGIN
        kind = "HIGH_VALUE_NEUTRAL_RACE" if race else "LOCAL_PRODUCTION_CAPTURE"
        prop = _main35_make_capture_prop(
            world,
            states,
            tgt,
            "hv_neutral_race",
            (PRIORITY_HV_RACE_BASE if race else PRIORITY_HV_CAPTURE_BASE) + int(tgt.production) * 12,
            mission_kind=kind,
            max_sources=5,
            source_radius=max(LOCAL_PRODUCTION_MAX_DIST, CAMPAIGN_RADIUS + 18),
            hold_margin=max(3, int(tgt.production) * 2),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug("MAIN33_HV_NEUTRAL_SELECTED")
            world.add_debug(f"EXPANSION_BEFORE_BANKING target=p{tgt.id}")
            return True
    return False


def main35_expand_from_hub(world, states, chain_plan, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    hubs = command_hubs(world)[:5]
    if not hubs:
        return False
    candidates = []
    for hub in hubs:
        if hub.id not in states or states[hub.id].threatened:
            continue
        for tgt in world.normal_planets:
            if tgt.owner == world.player or world.is_comet(tgt):
                continue
            d = dp(hub, tgt)
            if d > LOCAL_HUB_RADIUS + 15:
                continue
            if tgt.owner not in (-1, world.player) and not (
                is_local_enemy_opportunity(world, tgt)
                or _small_radius_target_allowed(world, tgt, hub, chain_plan)
            ):
                continue
            score = launchpad_target_score(world, hub, tgt, StrategyMode.EXPAND_CHAIN)
            score += len(_campaign_followup_options(world, tgt)) * 28.0
            if tgt.id in chain_plan[:12]:
                score += 35.0
            candidates.append((score, d, tgt))
    candidates.sort(key=lambda item: (-item[0], item[1], int(item[2].ships)))
    for score, _d, tgt in candidates[:6]:
        prop = _main35_make_capture_prop(
            world,
            states,
            tgt,
            "expand_from_hub",
            PRIORITY_EXPAND_FROM_HUB + score * 0.04,
            max_sources=4,
            source_radius=LOCAL_HUB_RADIUS + 15,
            hold_margin=2 if tgt.owner == -1 else max(8, int(tgt.production) * 3),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug(f"MAIN33_EXPANSION_IMPORTED hub_expand target=p{tgt.id}")
            world.add_debug("EXPANSION_BEFORE_BANKING")
            return True
    return False


def main35_opportunity_attack(world, states, moves, deadline):
    if moves or time.perf_counter() > deadline or not world.enemy_planets:
        return False
    candidates = []
    for tgt, weakness in detect_enemy_weakness(world)[:8]:
        if not should_allow_enemy_attack(world, tgt, "SYNC_ATTACK", "main35_opportunity"):
            continue
        nearest = min((dp(p, tgt) for p in world.my_planets), default=999.0)
        if nearest > BREACH_KILL_DIST + 12:
            continue
        score = weakness + int(tgt.production) * 35.0 + max(0.0, 70.0 - nearest) * 1.5
        if is_local_enemy_opportunity(world, tgt):
            score += 60.0
        candidates.append((score, nearest, tgt))
    candidates.sort(key=lambda item: (-item[0], item[1], int(item[2].ships)))
    for score, _nearest, tgt in candidates[:4]:
        prop = _main35_make_capture_prop(
            world,
            states,
            tgt,
            "opportunity_attack",
            PRIORITY_OPPORTUNISTIC_STRIKE + score * 0.06,
            max_sources=5,
            source_radius=BREACH_KILL_DIST + 12,
            hold_margin=max(12, int(tgt.production) * 5),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug("MAIN33_OPPORTUNITY_ATTACK_SELECTED")
            world.add_debug(f"EXPANSION_BEFORE_BANKING target=p{tgt.id}")
            return True
    return False


def run_four_player_expand_first(world, moves, deadline):
    if time.perf_counter() > deadline:
        return False
    if not (
        world.is_four_player
        and _is_low_control(world)
        and world.neutral_planets
        and world.features.get("incoming_threat_count", 0) == 0
    ):
        return False
    prop = find_4p_corner_expansion_target(world)
    if prop is None:
        return False
    target = world.planet_by_id.get(prop.target_id)
    if target is None or world.is_comet(target):
        return False
    zone = classify_corner_zone(world, target)
    if zone == "opposite_corner" and not (
        _adjacent_corner_secured(world, "clockwise_adjacent_corner")
        and _adjacent_corner_secured(world, "counterclockwise_adjacent_corner")
    ):
        world.add_debug(f"FOUR_PLAYER_EXPAND_FIRST_SKIP_OPPOSITE p{target.id}")
        return False
    if target.owner not in (-1, world.player):
        eta = prop.eta_max or max((e for _, _, _, e in prop.planned_sources), default=20)
        total = sum(ships for _, ships, _, _ in prop.planned_sources)
        if not is_local_enemy_opportunity(world, target):
            return False
        if not world.can_hold_after_capture(target, eta, total):
            world.add_debug(f"FOUR_PLAYER_EXPAND_FIRST_SKIP_ENEMY_NOT_HOLDABLE p{target.id}")
            return False
    if _commit_proposal(world, prop, moves):
        world.add_debug("FOUR_PLAYER_EXPAND_FIRST")
        world.add_debug(f"FOUR_PLAYER_EXPAND_FIRST_SELECTED p{target.id} zone={zone}")
        return True
    return False


# ── main36 opponent model / adaptive tempo ────────────────────────────────────

@dataclass
class OpponentModel:
    owner: int
    planet_count: int
    production: int
    total_ships: int
    fleet_ships: int
    fleet_ratio: float
    expansion_speed: int
    production_growth: int
    planet_count_growth: int
    grouped_fleets: bool
    scattered_fleets: bool
    early_aggression: bool
    turtle_behavior: bool
    drained_sources: list
    preferred_target_zones: list
    nearest_threatening_launchpads: list
    weak_nearby: list


def build_opponent_model(world):
    models = {}
    owner_ids = set(p.owner for p in world.enemy_planets) | set(f.owner for f in world.enemy_fleets)
    memory = _opponent_model_memory.setdefault(world.player, {})
    for owner in owner_ids:
        planets = [p for p in world.enemy_planets if p.owner == owner]
        fleets = [f for f in world.enemy_fleets if f.owner == owner]
        planet_count = len(planets)
        production = sum(int(p.production) for p in planets)
        fleet_ships = sum(int(f.ships) for f in fleets)
        stationed = sum(int(p.ships) for p in planets)
        total_ships = stationed + fleet_ships
        prev = memory.get(owner, {})
        planet_growth = planet_count - int(prev.get("planet_count", planet_count))
        prod_growth = production - int(prev.get("production", production))
        expansion_speed = max(planet_growth, 0)
        fleet_ratio = fleet_ships / max(1, total_ships)
        avg_fleet = fleet_ships / max(1, len(fleets))
        grouped = len(fleets) > 0 and avg_fleet >= 20
        scattered = len(fleets) >= 4 and avg_fleet < 18
        nearest_fleet_threat = min(
            (min(dp(f, m) for m in world.my_planets) for f in fleets),
            default=999.0,
        )
        early_aggression = world.step < 120 and fleet_ships >= 25 and nearest_fleet_threat <= 55.0
        turtle = (
            world.step > 60
            and planet_growth <= 0
            and fleet_ratio < 0.12
            and stationed >= max(120, production * 8)
        )
        drained_sources = []
        for p in planets:
            prev_ships = _prev_ships.get(p.id)
            if prev_ships is None:
                continue
            drop = int(prev_ships) + int(p.production) - int(p.ships)
            if drop >= max(20, int(p.production) * 5):
                drained_sources.append((p, drop))
        drained_sources.sort(key=lambda item: (-item[1], dp(item[0], min(world.my_planets, key=lambda m: dp(m, item[0]), default=item[0]))))
        preferred_targets = []
        for f in fleets:
            tgt = min(world.my_planets, key=lambda p: dp(f, p), default=None)
            if tgt is not None and dp(f, tgt) <= 50.0:
                preferred_targets.append(tgt.id)
        threatening_launchpads = sorted(
            [
                p for p in planets
                if _planet_role(p) == ROLE_LAUNCHPAD
                and min((dp(p, m) for m in world.my_planets), default=999.0) <= 80.0
            ],
            key=lambda p: min((dp(p, m) for m in world.my_planets), default=999.0),
        )[:3]
        weak_nearby = sorted(
            [
                p for p in planets
                if int(p.ships) <= max(25, int(p.production) * 6)
                and min((dp(m, p) for m in world.my_planets), default=999.0) <= 70.0
            ],
            key=lambda p: (int(p.ships), -int(p.production)),
        )[:4]
        models[owner] = OpponentModel(
            owner=owner,
            planet_count=planet_count,
            production=production,
            total_ships=total_ships,
            fleet_ships=fleet_ships,
            fleet_ratio=fleet_ratio,
            expansion_speed=expansion_speed,
            production_growth=prod_growth,
            planet_count_growth=planet_growth,
            grouped_fleets=grouped,
            scattered_fleets=scattered,
            early_aggression=early_aggression,
            turtle_behavior=turtle,
            drained_sources=drained_sources,
            preferred_target_zones=preferred_targets[:5],
            nearest_threatening_launchpads=threatening_launchpads,
            weak_nearby=weak_nearby,
        )
        memory[owner] = {
            "planet_count": planet_count,
            "production": production,
            "total_ships": total_ships,
            "fleet_ships": fleet_ships,
        }
    world.add_debug("OPPONENT_MODEL_BUILT")
    return models


def select_adaptive_strategy_mode(world, opponent_models):
    best_mode = None
    best_model = None
    best_score = -1e9
    for model in opponent_models.values():
        modes = []
        if model.drained_sources:
            modes.append(("DRAINED_AFTER_ATTACK", 130.0 + model.drained_sources[0][1]))
        if model.early_aggression:
            modes.append(("EARLY_ATTACKER", 120.0 + model.fleet_ships * 0.2))
        if model.production >= world.my_prod + 8 or model.planet_count >= len(world.my_planets) + 3:
            modes.append(("LEADER_SNOWBALL", 105.0 + max(0, model.production - world.my_prod) * 4.0))
        if model.expansion_speed >= 2 or (world.step < 120 and model.planet_count >= len(world.my_planets) + 2):
            modes.append(("FAST_EXPANDER", 100.0 + model.expansion_speed * 15.0))
        if model.scattered_fleets:
            modes.append(("SCATTER_ATTACKER", 92.0 + len(model.preferred_target_zones) * 4.0))
        if model.turtle_behavior:
            modes.append(("TURTLE", 85.0 + model.production * 2.0))
        if model.weak_nearby:
            modes.append(("WEAK_NEARBY", 80.0 + int(model.weak_nearby[0].production) * 8.0))
        for mode, score in modes:
            if score > best_score:
                best_score = score
                best_mode = mode
                best_model = model
    if best_mode is not None:
        world.add_debug(f"ADAPTIVE_MODE_SELECTED {best_mode} owner={best_model.owner}")
    return best_mode, best_model


@dataclass
class OpponentProfile:
    aggression_index: float = 0.0
    expansion_rate: float = 0.0
    hoarding_ratio: float = 0.0
    fleet_ratio: float = 0.0
    pressure_on_neutrals: float = 0.0
    pressure_on_me: float = 0.0


@dataclass
class MetaStrategy:
    name: str
    weights: dict


META_STRATEGIES = (
    MetaStrategy("Aggressive", {
        "production": 0.95,
        "ships": 0.90,
        "control": 1.10,
        "enemy_capture": 1.55,
        "neutral_capture": 0.85,
        "vulnerability": 0.85,
        "vulture": 0.90,
    }),
    MetaStrategy("Greedy Macro", {
        "production": 1.35,
        "ships": 0.95,
        "control": 1.10,
        "enemy_capture": 0.80,
        "neutral_capture": 1.45,
        "vulnerability": 0.80,
        "vulture": 0.85,
    }),
    MetaStrategy("Defensive", {
        "production": 0.95,
        "ships": 1.15,
        "control": 0.95,
        "enemy_capture": 0.75,
        "neutral_capture": 0.85,
        "vulnerability": 1.65,
        "vulture": 0.90,
    }),
    MetaStrategy("Vulture", {
        "production": 1.00,
        "ships": 0.95,
        "control": 1.00,
        "enemy_capture": 1.15,
        "neutral_capture": 0.95,
        "vulnerability": 1.00,
        "vulture": 1.70,
    }),
)


class AdaptiveMetaController:
    """Tiny online learner for beam-search score multipliers.

    The controller keeps a 10-turn opponent behavior window and uses UCB over
    four hand-shaped strategy arms. Updates are intentionally O(arms + fleets)
    and run only once per turn.
    """

    def __init__(self, update_interval=18, window=10, exploration_c=0.85):
        self.update_interval = int(update_interval)
        self.window = int(window)
        self.exploration_c = float(exploration_c)
        self.history = []
        self.counts = {strategy.name: 0 for strategy in META_STRATEGIES}
        self.values = {strategy.name: 0.0 for strategy in META_STRATEGIES}
        self.active_name = "Greedy Macro"
        self.last_eval_step = None
        self.last_margin = None
        self.last_my_score = None
        self.last_enemy_score = None
        self.profile = OpponentProfile()

    def reset(self):
        self.__init__(self.update_interval, self.window, self.exploration_c)

    def observe_opponent(self, world):
        enemy_fleet_ships = 0
        ships_to_me = 0
        ships_to_neutral = 0
        ships_to_enemy = 0
        targets = world.normal_planets
        for fl in world.enemy_fleets:
            ships = int(fl.ships)
            enemy_fleet_ships += ships
            target = fleet_target(fl, targets, world.ang_vel)
            if target is None:
                continue
            if target.owner == world.player:
                ships_to_me += ships
            elif target.owner == -1:
                ships_to_neutral += ships
            else:
                ships_to_enemy += ships

        enemy_stationed = sum(int(p.ships) for p in world.enemy_planets)
        enemy_total = max(1, enemy_stationed + enemy_fleet_ships)
        snapshot = {
            "step": world.step,
            "ships_to_me": ships_to_me,
            "ships_to_neutral": ships_to_neutral,
            "ships_to_enemy": ships_to_enemy,
            "enemy_fleet_ships": enemy_fleet_ships,
            "enemy_stationed": enemy_stationed,
            "enemy_total": enemy_total,
            "enemy_planets": len(world.enemy_planets),
            "enemy_prod": world.enemy_prod,
        }
        self.history.append(snapshot)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]
        self.profile = self._profile_from_history()
        world.add_debug(
            f"OPP_PROFILE aggression={self.profile.aggression_index:.2f} "
            f"expansion={self.profile.expansion_rate:.2f} hoard={self.profile.hoarding_ratio:.2f}"
        )
        return self.profile

    def _profile_from_history(self):
        if not self.history:
            return OpponentProfile()
        fleet_ships = sum(h["enemy_fleet_ships"] for h in self.history)
        ships_to_me = sum(h["ships_to_me"] for h in self.history)
        ships_to_neutral = sum(h["ships_to_neutral"] for h in self.history)
        latest = self.history[-1]
        oldest = self.history[0]
        denom = max(1, fleet_ships)
        planet_growth = max(0, latest["enemy_planets"] - oldest["enemy_planets"])
        prod_growth = max(0, latest["enemy_prod"] - oldest["enemy_prod"])
        expansion = min(1.0, ships_to_neutral / denom + (planet_growth * 0.12) + (prod_growth * 0.025))
        return OpponentProfile(
            aggression_index=min(1.0, ships_to_me / denom),
            expansion_rate=expansion,
            hoarding_ratio=latest["enemy_stationed"] / max(1, latest["enemy_total"]),
            fleet_ratio=latest["enemy_fleet_ships"] / max(1, latest["enemy_total"]),
            pressure_on_neutrals=min(1.0, ships_to_neutral / denom),
            pressure_on_me=min(1.0, ships_to_me / denom),
        )

    def _context_bias(self, strategy_name, profile):
        if strategy_name == "Defensive":
            return profile.aggression_index * 0.18 + (1.0 - profile.hoarding_ratio) * 0.04
        if strategy_name == "Greedy Macro":
            return profile.expansion_rate * 0.10 + profile.hoarding_ratio * 0.06
        if strategy_name == "Aggressive":
            return profile.hoarding_ratio * 0.14 + max(0.0, 0.35 - profile.aggression_index) * 0.08
        if strategy_name == "Vulture":
            return profile.pressure_on_neutrals * 0.14 + profile.fleet_ratio * 0.06
        return 0.0

    def _current_margin(self, world):
        enemy_score = max(world.leader_score, world.enemy_prod * 95 + world.enemy_total_ships)
        my_score = world.my_score
        return float(my_score - enemy_score), float(my_score), float(enemy_score)

    def update(self, world):
        profile = self.observe_opponent(world)
        margin, my_score, enemy_score = self._current_margin(world)
        if self.last_eval_step is None:
            self.last_eval_step = world.step
            self.last_margin = margin
            self.last_my_score = my_score
            self.last_enemy_score = enemy_score
            self.counts[self.active_name] += 1
            world.add_debug(f"META_INIT strategy={self.active_name}")
            return self.active_name, self.active_weights()

        if world.step - self.last_eval_step >= self.update_interval:
            my_delta = my_score - self.last_my_score
            enemy_delta = enemy_score - self.last_enemy_score
            margin_delta = margin - self.last_margin
            reward = (margin_delta + 0.45 * (my_delta - enemy_delta)) / 250.0
            reward = max(-1.0, min(1.0, reward))
            n = self.counts[self.active_name]
            old = self.values[self.active_name]
            self.values[self.active_name] = old + (reward - old) / max(1, n)
            self.last_eval_step = world.step
            self.last_margin = margin
            self.last_my_score = my_score
            self.last_enemy_score = enemy_score
            self.active_name = self.select_strategy(profile)
            self.counts[self.active_name] += 1
            world.add_debug(
                f"META_UPDATE reward={reward:.3f} next={self.active_name} "
                f"counts={self.counts[self.active_name]} value={self.values[self.active_name]:.3f}"
            )
        else:
            world.add_debug(f"META_HOLD strategy={self.active_name}")
        return self.active_name, self.active_weights()

    def select_strategy(self, profile):
        total = max(1, sum(self.counts.values()))
        best_name = self.active_name
        best_score = -1e9
        for strategy in META_STRATEGIES:
            pulls = self.counts[strategy.name]
            if pulls <= 0:
                ucb = 1.0
            else:
                ucb = self.values[strategy.name] + self.exploration_c * math.sqrt(math.log(total + 1) / pulls)
            score = ucb + self._context_bias(strategy.name, profile)
            if score > best_score:
                best_score = score
                best_name = strategy.name
        return best_name

    def active_weights(self):
        for strategy in META_STRATEGIES:
            if strategy.name == self.active_name:
                return strategy.weights
        return META_STRATEGIES[1].weights


def adaptive_meta_controller_for(world):
    controller = _adaptive_meta_controllers.get(world.player)
    if controller is None:
        controller = AdaptiveMetaController()
        _adaptive_meta_controllers[world.player] = controller
    return controller


def _commit_adaptive_target(world, states, target, reason, priority, moves, deadline, *, hold_margin=None, source_radius=80.0):
    if time.perf_counter() > deadline or target is None or target.owner == world.player or world.is_comet(target):
        return False
    kind = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
    prop = _main35_make_capture_prop(
        world,
        states,
        target,
        reason,
        priority,
        mission_kind=kind,
        max_sources=6,
        source_radius=source_radius,
        hold_margin=hold_margin,
    )
    if prop is None:
        return False
    if _commit_proposal(world, prop, moves):
        return True
    return False


def _small_radius_target_allowed(world, target, anchor=None, chain_plan=None):
    """Small planets are lower priority, not forbidden; ownership and route value decide."""
    if target is None or target.owner == world.player or world.is_comet(target):
        return False
    if _planet_role(target) != ROLE_STORAGE:
        return False
    nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
    followups = len(_campaign_followup_options(world, target))
    chain_ids = set((chain_plan or [])[:16])
    bridge_value = (
        _chain_small_has_value(world, target, anchor)
        or followups >= 1
        or target.id in chain_ids
    )
    cheap_near = nearest <= 42.0 and int(target.ships) <= max(20, int(target.production) * 7 + 12)
    enemy_owned = target.owner not in (-1, world.player)
    if enemy_owned and (
        nearest <= 72.0
        or bridge_value
        or cheap_near
        or is_local_enemy_opportunity(world, target)
        or enemy_planets_total(world) <= 8
    ):
        world.add_debug(f"SMALL_RADIUS_TARGET_ALLOWED p{target.id}")
        world.add_debug(f"SMALL_ENEMY_PLANET_NOT_SKIPPED p{target.id}")
        return True
    if target.owner == -1 and (bridge_value or cheap_near or nearest <= 30.0):
        world.add_debug(f"SMALL_RADIUS_TARGET_ALLOWED p{target.id}")
        world.add_debug(f"SMALL_NEUTRAL_ALLOWED_IF_BRIDGE p{target.id}")
        return True
    return False


def _rotating_source_static_target_bonus(world, src, target):
    if src is None or target is None or is_static_planet(src) or not is_static_planet(target):
        return 0.0
    if target.owner == world.player or world.is_comet(target):
        return 0.0
    useful_static = (
        target.owner not in (-1, world.player)
        or int(target.production) >= 2
        or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
        or _chain_small_has_value(world, target, src)
    )
    if not useful_static:
        return 0.0
    world.add_debug(f"ROTATING_SOURCE_STATIC_TARGET_PRIORITY source=p{src.id} target=p{target.id}")
    world.add_debug(f"ROTATING_TO_STATIC_ANCHOR_CAPTURE source=p{src.id} target=p{target.id}")
    return 130.0 if target.owner not in (-1, world.player) else 90.0


def run_adaptive_strategy_mode(world, states, chain_plan, opponent_models, adaptive_mode, adaptive_model, moves, deadline):
    if moves or adaptive_mode is None or adaptive_model is None or time.perf_counter() > deadline:
        return False
    if adaptive_mode == "DRAINED_AFTER_ATTACK":
        for target, drop in adaptive_model.drained_sources[:3]:
            if _commit_adaptive_target(
                world, states, target, "drained_source_counterattack",
                PRIORITY_OPPORTUNISTIC_STRIKE + drop, moves, deadline,
                hold_margin=max(12, int(target.production) * 5),
            ):
                world.add_debug("DRAINED_SOURCE_COUNTERATTACK")
                return True
    if adaptive_mode in ("EARLY_ATTACKER", "SCATTER_ATTACKER"):
        for target, drop in adaptive_model.drained_sources[:3]:
            if _commit_adaptive_target(
                world, states, target, "early_attacker_punish_source",
                PRIORITY_OPPORTUNISTIC_STRIKE + drop + 30.0, moves, deadline,
                hold_margin=max(12, int(target.production) * 5),
            ):
                world.add_debug("EARLY_ATTACKER_PUNISH_SOURCE")
                return True
        if adaptive_model.weak_nearby:
            target = adaptive_model.weak_nearby[0]
            if _commit_adaptive_target(world, states, target, "scatter_attacker_base_counter", 118.0, moves, deadline):
                world.add_debug("EARLY_ATTACKER_PUNISH_SOURCE")
                return True
    if adaptive_mode == "FAST_EXPANDER":
        world.add_debug("FAST_EXPANDER_RESPONSE")
        targets = sorted(
            [
                n for n in world.neutral_planets
                if not world.is_comet(n)
                and (int(n.production) >= 3 or _planet_role(n) in (ROLE_BRIDGE, ROLE_LAUNCHPAD))
                and min((dp(m, n) for m in world.my_planets), default=999.0) <= CAMPAIGN_RADIUS + 18
            ],
            key=lambda n: (-int(n.production), min((dp(m, n) for m in world.my_planets), default=999.0), int(n.ships)),
        )
        for target in targets[:6]:
            if _commit_adaptive_target(world, states, target, "fast_expander_contest", 132.0 + int(target.production) * 8, moves, deadline, hold_margin=max(2, int(target.production))):
                return True
    if adaptive_mode == "LEADER_SNOWBALL":
        world.add_debug("LEADER_SNOWBALL_PRESSURE")
        targets = sorted(
            adaptive_model.weak_nearby + adaptive_model.nearest_threatening_launchpads,
            key=lambda p: (-int(p.production), int(p.ships)),
        )
        for target in targets[:5]:
            if _commit_adaptive_target(world, states, target, "leader_snowball_pressure", 140.0 + int(target.production) * 10, moves, deadline, hold_margin=max(15, int(target.production) * 5)):
                return True
    if adaptive_mode == "TURTLE":
        world.add_debug("TURTLE_BREACH_MODE")
        for target in adaptive_model.nearest_threatening_launchpads + adaptive_model.weak_nearby:
            if _commit_adaptive_target(world, states, target, "turtle_breach", 120.0 + int(target.production) * 8, moves, deadline, hold_margin=max(18, int(target.production) * 6)):
                return True
    if adaptive_mode == "WEAK_NEARBY":
        for target in adaptive_model.weak_nearby[:4]:
            if _commit_adaptive_target(world, states, target, "weak_nearby_chain_attack", 122.0 + int(target.production) * 8, moves, deadline, hold_margin=max(12, int(target.production) * 5)):
                return True
    return False


def _target_improves_control(world, target, chain_plan):
    if target.owner == world.player or world.is_comet(target):
        return False
    if _small_radius_target_allowed(world, target, chain_plan=chain_plan):
        return True
    if target.owner not in (-1, world.player) and _planet_role(target) == ROLE_STORAGE:
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest <= 72.0 or enemy_planets_total(world) <= 8:
            world.add_debug(f"SMALL_PLANET_CONTROL_BONUS p{target.id}")
            return True
    if int(target.production) >= 2:
        return True
    if _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD):
        return True
    if target.id in chain_plan[:14]:
        return True
    if is_idle(target):
        return True
    if _chain_small_has_value(world, target):
        if target.owner == -1 and _planet_role(target) == ROLE_STORAGE:
            world.add_debug(f"SMALL_NEUTRAL_CONNECTOR_SELECTED p{target.id}")
        return True
    if len(_campaign_followup_options(world, target)) >= 1:
        return True
    if target.owner not in (-1, world.player) and is_local_enemy_opportunity(world, target):
        return True
    if target.owner == -1 and _planet_role(target) == ROLE_STORAGE:
        world.add_debug(f"SMALL_NEUTRAL_REJECTED_NOT_USEFUL p{target.id}")
    return False


def _map_center(world):
    pts = world.normal_planets or world.planets
    if not pts:
        return 0.0, 0.0
    return (
        sum(float(p.x) for p in pts) / max(1, len(pts)),
        sum(float(p.y) for p in pts) / max(1, len(pts)),
    )


def _owned_centroid(world):
    pts = world.my_planets or world.normal_planets or world.planets
    if not pts:
        return _map_center(world)
    return (
        sum(float(p.x) for p in pts) / max(1, len(pts)),
        sum(float(p.y) for p in pts) / max(1, len(pts)),
    )


def _axis_bucket_from(center, p):
    dx = float(p.x) - center[0]
    dy = float(p.y) - center[1]
    if abs(dx) >= abs(dy):
        return "E" if dx >= 0 else "W"
    return "N" if dy >= 0 else "S"


def _perpendicular_axis(axis):
    return ("N", "S") if axis in ("E", "W") else ("E", "W")


def _target_axis_width_bonus(world, target):
    center = _owned_centroid(world)
    owned_axes = {_axis_bucket_from(center, p) for p in world.my_planets}
    axis = _axis_bucket_from(center, target)
    if axis not in owned_axes:
        return 75.0
    opposite = {"E": "W", "W": "E", "N": "S", "S": "N"}[axis]
    if opposite in owned_axes:
        return 35.0
    return 10.0


def _target_l_shape_bonus(world, src, target):
    if src is None or not is_static_planet(src):
        return 0.0
    center = (float(src.x), float(src.y))
    src_axis = _axis_bucket_from(_owned_centroid(world), src)
    target_axis = _axis_bucket_from(center, target)
    if target_axis in _perpendicular_axis(src_axis):
        return 95.0
    return 20.0 if is_static_planet(target) or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD) else 0.0


def _target_x_shape_bonus(world, src, target):
    if src is None or is_static_planet(src):
        return 0.0
    dx = abs(float(target.x) - float(src.x))
    dy = abs(float(target.y) - float(src.y))
    diagonal = min(dx, dy) / max(1.0, max(dx, dy))
    bonus = 85.0 * diagonal
    if not is_static_planet(target) and _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD):
        bonus += 25.0
    return bonus


def _nearest_sweep_target_score(world, target, src, chain_plan):
    d = dp(src, target)
    role = _planet_role(target)
    radius_gain = max(0.0, float(target.radius) - float(src.radius))
    prod = int(target.production)
    score = (
        max(0.0, EARLY_NEAREST_SWEEP_DIST + 8.0 - d) * 4.8
        - int(target.ships) * 1.15
        + prod * 58.0
        + float(target.radius) * 26.0
        + radius_gain * 145.0
        + _target_axis_width_bonus(world, target)
        + _target_l_shape_bonus(world, src, target)
        + _target_x_shape_bonus(world, src, target)
        + _rotating_source_static_target_bonus(world, src, target)
        + min(3, len(_campaign_followup_options(world, target))) * 28.0
    )
    if prod >= 5:
        score += 190.0
    elif prod >= 4:
        score += 145.0
    elif prod >= 3:
        score += 85.0
    if is_idle(target):
        score += 110.0
    if role == ROLE_LAUNCHPAD:
        score += 120.0
    elif role == ROLE_BRIDGE:
        score += 80.0
    elif role == ROLE_STORAGE and target.owner not in (-1, world.player) and _small_radius_target_allowed(world, target, src, chain_plan):
        score += 95.0
    elif role == ROLE_STORAGE and d <= 34.0 and _chain_small_has_value(world, target, src):
        score += 36.0
    elif role == ROLE_STORAGE and prod <= 1 and target.owner == -1:
        score -= 120.0
    if target.id in set(chain_plan[:14]):
        score += 45.0
    if target.owner not in (-1, world.player):
        if role == ROLE_STORAGE and _small_radius_target_allowed(world, target, src, chain_plan):
            score += 65.0
        else:
            score += 30.0 if is_local_enemy_opportunity(world, target) else -80.0
    return score


def _low_increment_start_detected(world):
    start = _start_planet_current(world)
    if start is None or start.owner != world.player:
        return False
    low = (
        radius_class(start) == "SMALL"
        or float(start.radius) <= RADIUS_SMALL
        or int(start.production) <= 2
    )
    if low:
        world.add_debug(
            f"LOW_INCREMENT_START_DETECTED start=p{start.id} radius={float(start.radius):.1f} prod={int(start.production)}"
        )
    return low


def _small_start_escape_needed(world):
    if world.step >= 80:
        return False
    start = _start_planet_current(world)
    if start is None or not _low_increment_start_detected(world):
        return False
    better_anchor_owned = any(
        p.id != start.id
        and (
            float(p.radius) > float(start.radius) + 0.05
            or _planet_role(p) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or int(p.production) > int(start.production)
            or int(p.production) >= 3
        )
        for p in world.my_planets
    )
    return not better_anchor_owned


def _small_start_escape_candidate_score(world, start, target):
    d = dp(start, target)
    role = _planet_role(target)
    bigger_radius = float(target.radius) > float(start.radius) + 0.05
    bigger_role = role in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
    prod_gain = max(0, int(target.production) - int(start.production))
    category = 0 if bigger_radius and bigger_role else 1 if bigger_radius else 2 if prod_gain > 0 else 3
    score = (
        -category * 500.0
        + max(0.0, 72.0 - d) * 7.0
        + float(target.radius) * 70.0
        + int(target.production) * 82.0
        + prod_gain * 55.0
        + (125.0 if role == ROLE_LAUNCHPAD else 70.0 if role == ROLE_BRIDGE else 0.0)
        + (85.0 if is_static_planet(target) else 0.0)
        + _rotating_source_static_target_bonus(world, start, target)
        + min(3, len(_campaign_followup_options(world, target))) * 30.0
        - int(target.ships) * 1.25
    )
    if target.owner not in (-1, world.player):
        score -= 90.0
    approach = rotating_target_approach_score(start, target, world)
    if approach > 0:
        score += min(110.0, approach * 3.0)
    elif approach < 0:
        score += max(-160.0, approach * 4.0)
    return score


def opening_capture_deficit_active(world):
    deficit = (
        (world.step >= 20 and len(world.my_planets) <= 2)
        or (world.step >= 40 and len(world.my_planets) <= 3)
        or (world.step >= 70 and len(world.my_planets) <= 5)
    )
    if deficit:
        world.add_debug(
            f"OPENING_CAPTURE_DEFICIT_DETECTED step={world.step} owned={len(world.my_planets)}"
        )
        world.add_debug("EARLY_CAPTURE_TEMPO_REQUIRED")
    return deficit


def build_small_start_escape_props(world, states, chain_plan, deadline):
    """Highest-priority opening escape from a low-increment starting planet."""
    if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
        return ()
    if not _small_start_escape_needed(world):
        return ()
    start = _start_planet_current(world)
    if start is None or start.owner != world.player:
        return ()
    world.add_debug("SMALL_START_ESCAPE_ACTIVE")
    if world.step <= SMALL_START_STALL_STEP:
        world.add_debug("SMALL_START_ESCAPE_IMMEDIATE")

    candidates = []
    nearby_better_neutral = False
    nearest_bigger_anchor = None
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        d = dp(start, target)
        if d > 72.0:
            if _small_start_escape_target_value(world, target):
                world.add_debug(f"SMALL_START_FAR_TARGET_REJECTED p{target.id} d={d:.1f}")
            continue
        bigger_or_better = (
            float(target.radius) > float(start.radius) + 0.05
            or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or int(target.production) > int(start.production)
            or is_static_planet(target)
        )
        if not bigger_or_better:
            if _planet_role(target) == ROLE_STORAGE:
                world.add_debug(f"SMALL_START_SKIP_SMALL_SPAM p{target.id} d={d:.1f}")
            continue
        if target.owner not in (-1, world.player) and not is_local_enemy_opportunity(world, target):
            continue
        if target.owner == -1 and d <= 52.0 and (
            float(target.radius) > float(start.radius) + 0.05
            or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or int(target.production) > int(start.production)
        ):
            nearby_better_neutral = True
        if nearest_bigger_anchor is None or d < nearest_bigger_anchor[0]:
            nearest_bigger_anchor = (d, target)
        score = _small_start_escape_candidate_score(world, start, target)
        score -= d * 8.0
        candidates.append((-score, d, int(target.ships), target.id, target))
    candidates.sort()
    if nearest_bigger_anchor is not None:
        anchor_d, anchor = nearest_bigger_anchor
        world.add_debug(f"SMALL_START_NEAREST_BIGGER_LOCK p{anchor.id} d={anchor_d:.1f}")

    props = []
    for _neg_score, d, _ships, _tid, target in candidates[:10]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if nearby_better_neutral and (target.owner != -1 or d > 52.0):
            world.add_debug(f"OPENING_FAR_TARGET_REJECTED p{target.id} d={d:.1f}")
            world.add_debug(f"SMALL_START_FAR_TARGET_REJECTED p{target.id} d={d:.1f}")
            continue
        if nearest_bigger_anchor is not None and target.id != nearest_bigger_anchor[1].id and d > nearest_bigger_anchor[0] + 8.0:
            world.add_debug(
                f"SMALL_START_FAR_TARGET_REJECTED p{target.id} nearest=p{nearest_bigger_anchor[1].id} d={d:.1f}"
            )
            continue
        world.add_debug(
            f"BIGGER_RADIUS_TARGET_SELECTED p{target.id} radius={float(target.radius):.1f} "
            f"prod={int(target.production)} d={d:.1f}"
        )
        world.add_debug(f"SMALL_START_BIGGER_PLANET_ESCAPE p{target.id}")
        world.add_debug(f"SMALL_START_ESCAPE_TO_ANCHOR p{target.id}")
        if float(target.radius) > float(start.radius) + 0.05:
            world.add_debug(f"BIGGER_RADIUS_FIRST_SELECTED p{target.id}")
        kind = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            "small_start_escape",
            PRIORITY_SMALL_START_ESCAPE + max(0.0, 60.0 - d),
            mission_kind=kind,
            max_sources=1,
            source_radius=72.0,
            hold_margin=1 if target.owner == -1 else max(8, int(target.production) * 3),
            require_hold=target.owner != -1,
        )
        if prop is None:
            prop = _main35_make_capture_prop(
                world,
                states,
                target,
                "small_start_escape_grouped",
                PRIORITY_SMALL_START_ESCAPE + max(0.0, 60.0 - d) - 4.0,
                mission_kind=kind,
                max_sources=3,
                source_radius=72.0,
                hold_margin=1 if target.owner == -1 else max(8, int(target.production) * 3),
                require_hold=target.owner != -1,
            )
            if prop is not None:
                world.add_debug(f"OPENING_GROUPED_CAPTURE_USED p{target.id} sources={len(prop.planned_sources)}")
        if prop is None:
            continue
        prop.reason = f"{prop.reason} low_increment_escape start=p{start.id}"
        props.append(prop)
        break
    return tuple(props)


def build_early_nearest_sweep_props(world, states, chain_plan, deadline):
    """Opening-only burst of nearest useful captures, still through proposals."""
    if _small_start_escape_needed(world):
        return ()
    if world.step > EARLY_NEAREST_SWEEP_STEP_MAX and len(world.my_planets) >= 5:
        return ()
    if not world.my_planets or time.perf_counter() > deadline - BEAM_TIME_BUFFER:
        return ()
    world.add_debug("EARLY_NEAREST_SWEEP_ACTIVE")
    better_nearby_ids = set()
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        nearest_src = min(
            (p for p in world.my_planets if p.id in states and not states[p.id].threatened),
            key=lambda p: dp(p, target),
            default=None,
        )
        if nearest_src is None or dp(nearest_src, target) > EARLY_NEAREST_SWEEP_DIST + 8.0:
            continue
        if (
            _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            or int(target.production) >= 3
            or is_static_planet(target)
            or float(target.radius) > float(nearest_src.radius) + 0.05
        ):
            better_nearby_ids.add(target.id)
    candidates = []
    for target in world.normal_planets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if target.owner == world.player or world.is_comet(target):
            continue
        nearest_src = min(
            (p for p in world.my_planets if p.id in states and not states[p.id].threatened),
            key=lambda p: dp(p, target),
            default=None,
        )
        if nearest_src is None:
            continue
        if target.owner not in (-1, world.player) and not (
            is_local_enemy_opportunity(world, target)
            or _small_radius_target_allowed(world, target, nearest_src, chain_plan)
        ):
            continue
        nearest = dp(nearest_src, target)
        if nearest > EARLY_NEAREST_SWEEP_DIST:
            if nearest <= EARLY_NEAREST_SWEEP_DIST + 24.0:
                world.add_debug(f"OPENING_FAR_TARGET_REJECTED p{target.id} d={nearest:.1f}")
            continue
        role = _planet_role(target)
        if (
            better_nearby_ids
            and target.id not in better_nearby_ids
            and role == ROLE_STORAGE
            and int(target.production) <= 1
            and target.owner == -1
            and not (_chain_small_has_value(world, target, nearest_src) and nearest <= 24.0)
        ):
            world.add_debug(f"LOW_PROD_SMALL_TARGET_DEFERRED p{target.id}")
            continue
        useful_small = (
            role != ROLE_STORAGE
            or int(target.production) >= 1
            or nearest <= 28.0
            or _chain_small_has_value(world, target, nearest_src)
            or len(_campaign_followup_options(world, target)) >= 1
            or _small_radius_target_allowed(world, target, nearest_src, chain_plan)
        )
        if not useful_small:
            continue
        if role == ROLE_STORAGE and int(target.production) <= 1:
            world.add_debug(f"OPENING_SMALL_BRIDGE_ALLOWED p{target.id}")
        score = _nearest_sweep_target_score(world, target, nearest_src, chain_plan)
        candidates.append((-score, nearest, int(target.ships), target.id, target))
    candidates.sort()
    props = []
    seen_targets = set()
    used_axes = set()
    for _neg_score, nearest, _ships, _tid, target in candidates[:12]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if target.id in seen_targets:
            continue
        hold = 1 if target.owner == -1 else max(8, int(target.production) * 3)
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            "early_nearest_sweep",
            PRIORITY_EARLY_NEAREST_SWEEP - len(props) * 4.0 + max(0.0, 45.0 - nearest),
            mission_kind="CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK",
            max_sources=4,
            source_radius=EARLY_NEAREST_SWEEP_DIST + 8.0,
            hold_margin=hold,
            require_hold=target.owner != -1,
        )
        if prop is None:
            continue
        if world.step < 10:
            world.add_debug(f"OPENING_ATTACK_STARTED_BEFORE_STEP_10 target=p{target.id}")
        if _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD) or int(target.production) >= 3 or float(target.radius) > float(nearest_src.radius) + 0.05:
            world.add_debug(f"BIGGER_RADIUS_FIRST_SELECTED p{target.id}")
        prop.reason = f"{prop.reason} nearest_sweep_axis={_axis_bucket_from(_owned_centroid(world), target)}"
        props.append(prop)
        seen_targets.add(target.id)
        used_axes.add(_axis_bucket_from(_owned_centroid(world), target))
        world.add_debug(
            f"NEAREST_PLANET_CAPTURE_SELECTED p{target.id} d={nearest:.1f} "
            f"prod={int(target.production)} sources={len(prop.planned_sources)}"
        )
        if len(prop.planned_sources) > 1:
            world.add_debug(f"OPENING_GROUPED_CAPTURE_USED p{target.id} sources={len(prop.planned_sources)}")
        if is_static_planet(target):
            world.add_debug(f"L_SHAPE_EXPANSION_ACTIVE seed=p{target.id}")
        elif not is_idle(target):
            world.add_debug(f"X_SHAPE_EXPANSION_ACTIVE seed=p{target.id}")
        if len(props) >= EARLY_NEAREST_SWEEP_BURST:
            break
    if len(props) >= 2:
        world.add_debug(f"OPENING_MULTI_CAPTURE_BURST n={len(props)} axes={sorted(used_axes)}")
    return tuple(props)


def _parallel_opening_target_score(world, target, src, chain_plan, deficit_mode=False):
    d = dp(src, target)
    role = _planet_role(target)
    prod = int(target.production)
    need_hint = max(MIN_SEND_SHIPS, int(target.ships) + 1)
    eta = world.eta(src, target, need_hint)
    cheap_bridge = role == ROLE_STORAGE and d <= 28.0 and (
        _chain_small_has_value(world, target, src)
        or len(_campaign_followup_options(world, target)) >= 1
    )
    score = (
        max(0.0, PARALLEL_OPENING_SWEEP_DIST - d) * 5.2
        - eta * 5.5
        - int(target.ships) * 1.0
        + prod * 72.0
        + max(0.0, float(target.radius) - float(src.radius)) * 110.0
        + float(target.radius) * 18.0
        + _target_axis_width_bonus(world, target)
        + _target_l_shape_bonus(world, src, target)
        + _target_x_shape_bonus(world, src, target)
        + _rotating_source_static_target_bonus(world, src, target)
        + min(4, len(_campaign_followup_options(world, target))) * 34.0
    )
    if prod >= 5:
        score += 190.0
    elif prod >= 4:
        score += 145.0
    elif prod >= 3:
        score += 90.0
    if role == ROLE_LAUNCHPAD:
        score += 120.0
    elif role == ROLE_BRIDGE:
        score += 85.0
    elif role == ROLE_STORAGE and target.owner not in (-1, world.player) and _small_radius_target_allowed(world, target, src, chain_plan):
        score += 110.0
    elif cheap_bridge:
        score += 70.0
    elif role == ROLE_STORAGE and prod <= 1 and target.owner == -1:
        score -= 45.0 if deficit_mode else 80.0
    if is_idle(target):
        score += 95.0
    if target.id in set(chain_plan[:16]):
        score += 45.0
    if target.owner not in (-1, world.player):
        if role == ROLE_STORAGE and _small_radius_target_allowed(world, target, src, chain_plan):
            score += 70.0
        else:
            score += 35.0 if is_local_enemy_opportunity(world, target) else -100.0
    return score


def _make_parallel_opening_capture_prop(world, states, chain_plan, target, score, deadline):
    if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
        return None
    kind = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
    hold = 0 if target.owner == -1 and world.step <= 70 else (1 if target.owner == -1 else max(6, int(target.production) * 2))
    prop = _main35_make_capture_prop(
        world,
        states,
        target,
        "parallel_opening_sweep",
        PRIORITY_PARALLEL_OPENING_SWEEP + score * 0.035,
        mission_kind=kind,
        max_sources=1,
        source_radius=PARALLEL_OPENING_SWEEP_DIST,
        hold_margin=hold,
        require_hold=target.owner != -1,
    )
    if prop is None:
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            "parallel_opening_sweep_grouped",
            PRIORITY_PARALLEL_OPENING_SWEEP + score * 0.03 - 6.0,
            mission_kind=kind,
            max_sources=3,
            source_radius=PARALLEL_OPENING_SWEEP_DIST,
            hold_margin=hold,
            require_hold=target.owner != -1,
        )
        if prop is not None:
            world.add_debug(f"OPENING_GROUPED_CAPTURE_USED p{target.id} sources={len(prop.planned_sources)}")
    if prop is None:
        return None
    prop.reason = f"{prop.reason} parallel_opening_sweep"
    if world.step < 10:
        world.add_debug(f"OPENING_ATTACK_STARTED_BEFORE_STEP_10 target=p{target.id}")
    return prop


def build_parallel_opening_sweep_props(world, states, chain_plan, deadline):
    if world.step > 70 or time.perf_counter() > deadline - BEAM_TIME_BUFFER:
        return ()
    if not world.my_planets:
        return ()
    deficit_mode = opening_capture_deficit_active(world)
    world.add_debug("PARALLEL_SWEEP_ACTIVE")
    world.add_debug("EARLY_CAPTURE_TEMPO_REQUIRED")
    world.add_debug(f"SEARCH_LIMIT_RELAXED_FOR_OPENING limit={EARLY_NEAREST_SWEEP_BURST}")

    scored = []
    for target in world.normal_planets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if target.owner == world.player or world.is_comet(target):
            continue
        nearby_sources = []
        for p in world.my_planets:
            if p.id not in states or states[p.id].threatened or dp(p, target) > PARALLEL_OPENING_SWEEP_DIST:
                continue
            if target.owner == -1 and _early_neutral_reserve_relaxation_allowed(world, p, target, "parallel_opening_sweep"):
                avail = int(p.ships) - world.committed.get(p.id, 0) - _early_neutral_min_source_reserve(world, p)
            else:
                avail = states[p.id].safe_surplus
            if avail >= MIN_SEND_SHIPS:
                nearby_sources.append(p)
        if not nearby_sources:
            continue
        src = min(nearby_sources, key=lambda p: (dp(p, target), -states[p.id].safe_surplus))
        if target.owner not in (-1, world.player) and not (
            is_local_enemy_opportunity(world, target)
            or _small_radius_target_allowed(world, target, src, chain_plan)
        ):
            continue
        role = _planet_role(target)
        useful_small = (
            role != ROLE_STORAGE
            or int(target.production) >= 1
            or dp(src, target) <= 30.0
            or _chain_small_has_value(world, target, src)
            or len(_campaign_followup_options(world, target)) >= 1
            or _small_radius_target_allowed(world, target, src, chain_plan)
        )
        if not useful_small:
            continue
        score = _parallel_opening_target_score(world, target, src, chain_plan, deficit_mode=deficit_mode)
        scored.append((score, dp(src, target), int(target.ships), target.id, target))
    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    props = []
    seen = set()
    for score, _d, _ships, _tid, target in scored[:18]:
        if len(props) >= EARLY_NEAREST_SWEEP_BURST * 2 or time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if target.id in seen:
            continue
        prop = _make_parallel_opening_capture_prop(world, states, chain_plan, target, score, deadline)
        if prop is None:
            continue
        props.append(prop)
        seen.add(target.id)
    if props:
        world.add_debug(f"ATTACK_BUNDLE_CANDIDATES_BUILT parallel_opening={len(props)}")
    return tuple(props)


def identify_front_clusters(world):
    center = _owned_centroid(world)
    clusters = {}
    for p in world.my_planets:
        axis = _axis_bucket_from(center, p)
        info = clusters.setdefault(axis, {
            "axis": axis,
            "launchpads": [],
            "bridges": [],
            "staging": [],
            "planets": [],
            "pressure": 0.0,
        })
        info["planets"].append(p)
        if _planet_role(p) == ROLE_LAUNCHPAD or is_static_planet(p) or int(p.production) >= 3:
            info["launchpads"].append(p)
        if _planet_role(p) == ROLE_BRIDGE:
            info["bridges"].append(p)
        if world.nearest_enemy_distance(p) <= MULTI_AXIS_FRONT_RADIUS:
            info["staging"].append(p)
            info["pressure"] += max(0.0, MULTI_AXIS_FRONT_RADIUS - world.nearest_enemy_distance(p))
    fronts = list(clusters.values())
    fronts.sort(key=lambda f: (-len(f["launchpads"]), -len(f["planets"]), f["axis"]))
    if len(fronts) >= 2:
        world.add_debug(f"MULTI_AXIS_FRONT_CREATED axes={[f['axis'] for f in fronts[:4]]}")
    return fronts


def assign_planets_to_fronts(world):
    fronts = identify_front_clusters(world)
    assignments = {f["axis"]: [] for f in fronts}
    center = _owned_centroid(world)
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        axis = _axis_bucket_from(center, target)
        if axis in assignments:
            assignments[axis].append(target)
    for axis in assignments:
        assignments[axis].sort(
            key=lambda t: (
                min((dp(p, t) for p in world.my_planets), default=999.0),
                -int(t.production),
                int(t.ships),
            )
        )
    return assignments


def build_multi_axis_routes(world):
    fronts = identify_front_clusters(world)
    assignments = assign_planets_to_fronts(world)
    routes = []
    for front in fronts[:4]:
        axis = front["axis"]
        anchors = front["launchpads"] or front["bridges"] or front["planets"]
        if not anchors:
            continue
        targets = []
        for target in assignments.get(axis, [])[:8]:
            nearest = min((dp(a, target) for a in anchors), default=999.0)
            if nearest <= MULTI_AXIS_FRONT_RADIUS:
                targets.append(target)
        if targets:
            routes.append({"axis": axis, "anchors": anchors, "targets": targets, "front": front})
    if routes:
        world.add_debug(f"DIRECTIONAL_SCAN_BUILT axes={[r['axis'] for r in routes]}")
    return routes


def build_multi_axis_expansion_props(world, states, chain_plan, deadline):
    if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
        return ()
    routes = build_multi_axis_routes(world)
    if len(routes) < 2:
        return ()
    pressure_axis = None
    if world.enemy_fleets:
        threatened = []
        center = _owned_centroid(world)
        for fl in world.enemy_fleets:
            tgt = fleet_target(fl, world.normal_planets, world.ang_vel)
            if tgt is not None and tgt.owner == world.player:
                threatened.append(_axis_bucket_from(center, tgt))
        if threatened:
            pressure_axis = sorted(set(threatened), key=lambda axis: (-threatened.count(axis), axis))[0]
            world.add_debug(f"DUAL_FRONT_RECOVERY attacked_axis={pressure_axis}")
            world.add_debug("FRONT_REBALANCE_ACTIVE")
    props = []
    used_axes = set()
    for route in routes[:4]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        axis = route["axis"]
        if axis in used_axes:
            continue
        best = None
        for target in route["targets"][:6]:
            anchor = min(route["anchors"], key=lambda p: dp(p, target), default=None)
            if anchor is None:
                continue
            rough_need = normalize_send_amount(world.required_ships_to_capture(target, anchor))
            if rough_need > sum(max(0, states.get(p.id).safe_surplus if states.get(p.id) else world.surplus(p)) for p in world.my_planets if dp(p, target) <= MULTI_AXIS_FRONT_RADIUS):
                world.add_debug(f"UNACHIEVABLE_TARGET_SKIPPED p{target.id} axis={axis}")
                continue
            if target.owner not in (-1, world.player) and not (
                is_local_enemy_opportunity(world, target)
                or _small_radius_target_allowed(world, target, anchor, chain_plan)
            ):
                continue
            score = _nearest_sweep_target_score(world, target, anchor, chain_plan)
            score += 70.0 if target.owner not in (-1, world.player) else 0.0
            score += 45.0 if is_static_planet(target) else 0.0
            score += 30.0 if _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD) else 0.0
            if pressure_axis is not None and axis != pressure_axis:
                score += 85.0
            if is_static_planet(anchor):
                score += _target_l_shape_bonus(world, anchor, target)
            else:
                score += _target_x_shape_bonus(world, anchor, target)
            item = (score, target, anchor)
            if best is None or item[0] > best[0]:
                best = item
        if best is None:
            continue
        score, target, anchor = best
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            f"multi_axis_front axis={axis}",
            PRIORITY_MULTI_AXIS_EXPAND + score * 0.025,
            mission_kind="CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK",
            max_sources=5,
            source_radius=MULTI_AXIS_FRONT_RADIUS,
            hold_margin=2 if target.owner == -1 else max(10, int(target.production) * 4),
            require_hold=target.owner != -1,
        )
        if prop is None:
            continue
        props.append(prop)
        used_axes.add(axis)
        world.add_debug(f"ACHIEVABLE_TARGET_PRIORITY axis={axis} target=p{target.id}")
        if pressure_axis is not None and axis != pressure_axis:
            world.add_debug(f"COUNTER_AXIS_ATTACK axis={axis} target=p{target.id}")
            world.add_debug(f"SECONDARY_FRONT_PRESSURE axis={axis} target=p{target.id}")
        if is_static_planet(anchor):
            world.add_debug(f"L_SHAPE_EXPANSION_ACTIVE anchor=p{anchor.id} target=p{target.id}")
            if axis in ("E", "W"):
                world.add_debug(f"STATIC_BRANCH_A axis={axis} target=p{target.id}")
            else:
                world.add_debug(f"STATIC_BRANCH_B axis={axis} target=p{target.id}")
            if len(used_axes) >= 2:
                world.add_debug("CORNER_CONTROL_ESTABLISHED")
        else:
            world.add_debug(f"X_SHAPE_EXPANSION_ACTIVE anchor=p{anchor.id} target=p{target.id}")
            world.add_debug(f"ROTATING_DIAGONAL_CAPTURE source=p{anchor.id} target=p{target.id}")
            if len(_campaign_followup_options(world, target)) >= 1:
                world.add_debug(f"ROTATIONAL_CHAIN_EXTENDED target=p{target.id}")
    if len(props) >= 2:
        world.add_debug(f"TWO_AXIS_EXPANSION_SELECTED axes={sorted(used_axes)}")
        world.add_debug(f"SIMULTANEOUS_FRONT_CAPTURE n={len(props)}")
    return tuple(props[:3])


def useful_capturable_target_exists(world, states, chain_plan):
    for target in sorted(
        [t for t in world.normal_planets if t.owner != world.player and not world.is_comet(t)],
        key=lambda t: (min((dp(m, t) for m in world.my_planets), default=999.0), -int(t.production), int(t.ships)),
    )[:14]:
        if not _target_improves_control(world, target, chain_plan):
            continue
        prop = _main35_make_capture_prop(
            world, states, target, "tempo_probe", 1.0,
            max_sources=4,
            source_radius=CAMPAIGN_RADIUS + 18,
            hold_margin=2 if target.owner == -1 else max(8, int(target.production) * 3),
        )
        if prop is not None:
            return True
    return False


def run_tempo_override_capture(world, states, chain_plan, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    candidates = []
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > CAMPAIGN_RADIUS + 18:
            continue
        if not _target_improves_control(world, target, chain_plan):
            continue
        score = (
            max(0.0, 75.0 - nearest) * 2.4
            + int(target.production) * 55.0
            + len(_campaign_followup_options(world, target)) * 38.0
            - int(target.ships) * 0.9
        )
        if target.owner != -1:
            score += 30.0
        if target.id in chain_plan[:14]:
            score += 45.0
        candidates.append((score, nearest, target))
    candidates.sort(key=lambda item: (-item[0], item[1], int(item[2].ships)))
    for score, _nearest, target in candidates[:8]:
        prop = _main35_make_capture_prop(
            world, states, target, "tempo_override", 150.0 + score * 0.04,
            max_sources=5,
            source_radius=CAMPAIGN_RADIUS + 18,
            hold_margin=2 if target.owner == -1 else max(10, int(target.production) * 4),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug("TEMPO_OVERRIDE_CAPTURE")
            world.add_debug(f"EXPANSION_BEFORE_BANKING target=p{target.id}")
            return True
    return False


def _sequence_target_score(world, first, second=None):
    score = 0.0
    targets = [first] + ([second] if second is not None else [])
    for idx, target in enumerate(targets):
        decay = 1.0 if idx == 0 else 0.65
        score += decay * int(target.production) * 90.0
        score += decay * (120.0 if _planet_role(target) == ROLE_LAUNCHPAD else 65.0 if _planet_role(target) == ROLE_BRIDGE else 18.0)
        score += decay * (80.0 if is_idle(target) else 0.0)
        score += decay * len(_campaign_followup_options(world, target)) * 45.0
        if target.owner not in (-1, world.player):
            score += decay * int(target.production) * 50.0
        score -= decay * int(target.ships) * 0.7
    if second is not None:
        score += max(0.0, 60.0 - dp(first, second)) * 2.0
    return score


def run_search_sequence_planner(world, states, chain_plan, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    world.add_debug("SEARCH_SEQUENCE_PLANNER_ACTIVE")
    candidates = []
    for first in world.normal_planets:
        if time.perf_counter() > deadline:
            break
        if first.owner == world.player or world.is_comet(first):
            continue
        if min((dp(m, first) for m in world.my_planets), default=999.0) > CAMPAIGN_RADIUS + 18:
            continue
        if not _target_improves_control(world, first, chain_plan):
            continue
        prop = _main35_make_capture_prop(
            world, states, first, "sequence_first_probe", 1.0,
            max_sources=5,
            source_radius=CAMPAIGN_RADIUS + 18,
            hold_margin=2 if first.owner == -1 else max(10, int(first.production) * 4),
        )
        if prop is None:
            continue
        best_second = None
        best_score = _sequence_target_score(world, first)
        for second in _campaign_followup_options(world, first):
            if second.owner == world.player or world.is_comet(second):
                continue
            s = _sequence_target_score(world, first, second)
            if s > best_score:
                best_score = s
                best_second = second
        candidates.append((best_score, first, best_second, prop))
    if not candidates:
        return False
    candidates.sort(key=lambda item: -item[0])
    score, first, second, prop = candidates[0]
    prop.priority = max(prop.priority, 155.0 + score * 0.025)
    prop.reason = f"search_sequence first=p{first.id} second=p{getattr(second, 'id', 'none')} score={score:.1f}"
    if _commit_proposal(world, prop, moves):
        world.add_debug("SEQUENCE_CAPTURE_CHAIN_SELECTED")
        if second is not None:
            world.add_debug(f"SEQUENCE_NEXT_NODE p{second.id}")
        return True
    return False


STRUCT_ANCHOR = "ANCHOR_LAUNCHPAD"
STRUCT_FRONTLINE = "FRONTLINE"
STRUCT_BRIDGE = "BRIDGE_CRITICAL"
STRUCT_REAR = "REAR_SAFE"
STRUCT_STORAGE = "STORAGE"
STRUCT_VULNERABLE = "VULNERABLE_ISOLATED"
STRUCT_RECENT = "RECENT_CAPTURE"
STRUCT_RECOVERY = "RECOVERY_TARGET"


def _owned_support_count(world, p, radius=42.0):
    return sum(1 for q in world.my_planets if q.id != p.id and dp(p, q) <= radius)


def _nearby_support_ships(world, p, radius=42.0):
    return sum(int(q.ships) for q in world.my_planets if q.id != p.id and dp(p, q) <= radius)


def _nearest_enemy_to_planet(world, p):
    return min((dp(p, e) for e in world.enemy_planets), default=999.0)


def classify_planet_structure(world, states, chain_plan):
    structure = {}
    chain_set = set(chain_plan[:16])
    for p in world.my_planets:
        labels = set()
        st = states.get(p.id)
        enemy_d = _nearest_enemy_to_planet(world, p)
        support = _owned_support_count(world, p)
        support_ships = _nearby_support_ships(world, p)
        role = _planet_role(p)

        anchor = (
            role == ROLE_LAUNCHPAD
            or is_static_planet(p)
            or int(p.production) >= 4
            or p.id in _primary_launchpads.get(world.player, {})
        )
        if anchor:
            labels.add(STRUCT_ANCHOR)
            world.add_debug(f"ANCHOR_LAUNCHPAD_IDENTIFIED p{p.id}")
        if enemy_d <= FRONTLINE_DIST + 22:
            labels.add(STRUCT_FRONTLINE)
        if (
            role == ROLE_BRIDGE
            or p.id in chain_set
            or (support >= 2 and enemy_d <= 80.0 and world.cluster_distance(p) <= CHAIN_RADIUS + 18)
        ):
            labels.add(STRUCT_BRIDGE)
            world.add_debug(f"BRIDGE_CRITICAL_IDENTIFIED p{p.id}")
        if enemy_d > 75.0 and support >= 2:
            labels.add(STRUCT_REAR)
        if role == ROLE_STORAGE and STRUCT_REAR in labels:
            labels.add(STRUCT_STORAGE)
        if _prev_owners and _prev_owners.get(p.id) != world.player:
            labels.add(STRUCT_RECENT)

        weak_for_role = False
        if st is not None:
            desired = _structure_desired_reserve(world, p, labels)
            weak_for_role = int(p.ships) < desired
        if (
            support < 2
            or (enemy_d <= 62.0 and support_ships < max(35, int(p.ships) * 0.7))
            or weak_for_role
        ) and (STRUCT_FRONTLINE in labels or STRUCT_BRIDGE in labels or STRUCT_RECENT in labels):
            labels.add(STRUCT_VULNERABLE)
            world.add_debug(f"VULNERABLE_ISOLATED_DETECTED p{p.id} support={support} enemy_d={enemy_d:.1f}")
        structure[p.id] = labels

    for lost in world.enemy_planets:
        if world.is_comet(lost):
            continue
        if _prev_owners.get(lost.id) == world.player and _is_core_chain_planet(world, lost, chain_plan):
            structure[lost.id] = {STRUCT_RECOVERY}
            world.add_debug(f"TERRITORY_RECOVERY_TRIGGERED p{lost.id}")
    return structure


def _structure_desired_reserve(world, p, labels):
    role = _planet_role(p)
    reserve = STORAGE_RESERVE if role == ROLE_STORAGE else BRIDGE_RESERVE if role == ROLE_BRIDGE else LAUNCHPAD_RESERVE
    if STRUCT_ANCHOR in labels:
        reserve = max(reserve, 50 if role == ROLE_LAUNCHPAD else 30 if role == ROLE_BRIDGE else 18)
    if STRUCT_BRIDGE in labels:
        reserve = max(reserve, 30, int(p.production) * 6)
    if STRUCT_FRONTLINE in labels:
        reserve += 15
    if STRUCT_RECENT in labels:
        reserve = max(reserve, 25, int(p.production) * 5)
    if STRUCT_VULNERABLE in labels:
        reserve = max(reserve, 35, int(p.production) * 7)
    return int(reserve)


def _frontline_fragmented(world, structure):
    frontline = [pid for pid, labels in structure.items() if STRUCT_FRONTLINE in labels]
    if len(frontline) <= 1:
        return False
    weak_links = 0
    for pid in frontline:
        p = world.planet_by_id.get(pid)
        if p is None:
            continue
        if _owned_support_count(world, p, radius=48.0) < 2:
            weak_links += 1
    fragmented = weak_links >= max(1, len(frontline) // 2)
    if fragmented:
        world.add_debug("FRONTLINE_FRAGMENTED")
    return fragmented


def _midgame_structure_context(world, states, chain_plan):
    structure = classify_planet_structure(world, states, chain_plan)
    vulnerable = [
        world.planet_by_id[pid] for pid, labels in structure.items()
        if pid in world.planet_by_id and STRUCT_VULNERABLE in labels
    ]
    weak_bridges = [
        p for p in vulnerable
        if STRUCT_BRIDGE in structure.get(p.id, set()) or STRUCT_ANCHOR in structure.get(p.id, set())
    ]
    recent = [
        world.planet_by_id[pid] for pid, labels in structure.items()
        if pid in world.planet_by_id and STRUCT_RECENT in labels
    ]
    fragmented = _frontline_fragmented(world, structure)
    enemy_near = min((_nearest_enemy_to_planet(world, p) for p in world.my_planets), default=999.0) <= 95.0
    active = (
        80 <= world.step <= 320
        and len(world.my_planets) >= 7
        and enemy_near
        and (recent or vulnerable or weak_bridges or fragmented)
    )
    return {
        "active": active,
        "structure": structure,
        "vulnerable": vulnerable,
        "weak_bridges": weak_bridges,
        "recent": recent,
        "fragmented": fragmented,
    }


def _structure_safe_for_deep_capture(world, states, chain_plan, target, mission_reason):
    if not (80 <= world.step <= 320 and len(world.my_planets) >= 7):
        return True
    if any(tag in (mission_reason or "") for tag in ("recovery", "defense", "final", "collapse")):
        return True
    ctx = _midgame_structure_context(world, states, chain_plan)
    if not ctx["active"]:
        return True
    nearest = min((dp(p, target) for p in world.my_planets), default=999.0)
    allowed_connected = (
        nearest <= 58.0
        or target.id in set(chain_plan[:12])
        or int(target.production) >= 4
        or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
        or (target.owner not in (-1, world.player) and is_local_enemy_opportunity(world, target))
    )
    if allowed_connected and len(_campaign_followup_options(world, target)) >= 1:
        return True
    if allowed_connected and nearest <= 50.0:
        return True
    world.add_debug(f"EXPANSION_BLOCKED_STRUCTURE_UNSAFE target=p{target.id} d={nearest:.1f}")
    return False


def _structure_source_spare(world, src, states, structure):
    st = states.get(src.id)
    if st is None or st.threatened:
        return 0
    labels = structure.get(src.id, set())
    reserve = max(st.reserve, _structure_desired_reserve(world, src, labels))
    spare = int(src.ships) - world.committed.get(src.id, 0) - reserve
    if STRUCT_ANCHOR in labels and spare < MIN_SEND_SHIPS:
        world.add_debug(f"ANCHOR_RESERVE_PRESERVED p{src.id} reserve={reserve}")
    return max(0, spare)


def _reinforce_structure_target(world, states, structure, target, moves, deadline):
    labels = structure.get(target.id, set())
    desired = _structure_desired_reserve(world, target, labels)
    deficit = desired - int(target.ships) - world.incoming_to_targets.get(target.id, 0)
    if deficit < MIN_SEND_SHIPS:
        return False
    sources = sorted(
        [
            p for p in world.my_planets
            if p.id != target.id
            and _structure_source_spare(world, p, states, structure) >= MIN_SEND_SHIPS
            and dp(p, target) <= 55.0
            and world.real_incoming_threat(p)["deficit"] <= 0
        ],
        key=lambda p: (dp(p, target), 0 if STRUCT_REAR in structure.get(p.id, set()) else 1, -_structure_source_spare(world, p, states, structure)),
    )
    if not sources:
        return False
    needed = round_up_to_granularity(deficit)
    sent = 0
    for src in sources[:4]:
        if time.perf_counter() > deadline or sent >= needed:
            break
        spare = _structure_source_spare(world, src, states, structure)
        send = round_down_to_granularity(min(spare, needed - sent))
        if send < MIN_SEND_SHIPS:
            continue
        if world.commit(src, target, send, moves, mission_type="DEFEND_HOLD"):
            sent += send
    if sent >= MIN_SEND_SHIPS:
        if STRUCT_BRIDGE in labels:
            world.add_debug(f"BRIDGE_REINFORCED p{target.id} sent={sent}")
        if STRUCT_RECOVERY in labels or STRUCT_BRIDGE in labels:
            world.add_debug(f"CHAIN_STRUCTURE_RECONNECTED p{target.id}")
        return True
    return False


def _run_grouped_front_pressure_if_stable(world, states, chain_plan, moves, deadline, structure):
    idle = _midgame_dominance_surplus(world, states)
    if idle < 120:
        return False
    targets = sorted(
        [
            t for t in world.enemy_planets + world.neutral_planets
            if not world.is_comet(t)
            and t.owner != world.player
            and min((dp(p, t) for p in world.my_planets), default=999.0) <= 82.0
            and _target_improves_control(world, t, chain_plan)
        ],
        key=lambda t: (
            0 if t.owner not in (-1, world.player) else 1,
            -int(t.production),
            min((dp(p, t) for p in world.my_planets), default=999.0),
            int(t.ships),
        ),
    )
    for target in targets[:6]:
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            "grouped_front_pressure",
            142.0 + int(target.production) * 8,
            max_sources=5,
            source_radius=82.0,
            hold_margin=3 if target.owner == -1 else max(12, int(target.production) * 5),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug("GROUPED_FRONT_PRESSURE")
            world.add_debug("IDLE_FORCE_CONVERTED")
            return True
    return False


def run_midgame_stabilization_mode(world, states, chain_plan, moves, deadline):
    if time.perf_counter() > deadline:
        return False
    ctx = _midgame_structure_context(world, states, chain_plan)
    if not ctx["active"]:
        return False
    world.add_debug("MIDGAME_STABILIZATION_MODE")
    if ctx["fragmented"] or ctx["vulnerable"] or ctx["weak_bridges"]:
        world.add_debug("FRONT_STABILIZATION_TRIGGERED")
    structure = ctx["structure"]
    targets = sorted(
        ctx["weak_bridges"] + [p for p in ctx["vulnerable"] if p not in ctx["weak_bridges"]],
        key=lambda p: (
            0 if STRUCT_ANCHOR in structure.get(p.id, set()) else 1,
            0 if STRUCT_BRIDGE in structure.get(p.id, set()) else 1,
            int(p.ships),
        ),
    )
    for target in targets[:4]:
        if time.perf_counter() > deadline:
            break
        if _reinforce_structure_target(world, states, structure, target, moves, deadline):
            return True
    if not ctx["vulnerable"] and not ctx["fragmented"]:
        world.add_debug("STABILIZED_EMPIRE_READY_FOR_PUSH")
        return _run_grouped_front_pressure_if_stable(world, states, chain_plan, moves, deadline, structure)
    return False


def log_missed_opportunities(world, states, chain_plan):
    parked = sum(int(p.ships) for p in world.my_planets)
    flying = sum(int(f.ships) for f in world.my_fleets)
    if parked - flying > 70:
        world.add_debug("IDLE_ARMY_PRESSURE_TRIGGERED")
    for target in sorted(world.normal_planets, key=lambda t: (min((dp(m, t) for m in world.my_planets), default=999.0), -int(t.production)))[:10]:
        if target.owner == world.player or world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest <= CAMPAIGN_RADIUS and _target_improves_control(world, target, chain_plan):
            world.add_debug(f"MISSED_NEARBY_CAPTURE_LOGGED target=p{target.id} d={nearest:.1f} prod={int(target.production)}")
            break
    for target in world.neutral_planets:
        if not world.is_comet(target) and int(target.production) >= 4:
            nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
            if nearest <= CAMPAIGN_RADIUS + 20:
                world.add_debug(f"MISSED_HIGH_PROD_NEUTRAL target=p{target.id} prod={int(target.production)}")
                break
    for target, _score in detect_enemy_weakness(world)[:3]:
        if min((dp(m, target) for m in world.my_planets), default=999.0) <= 75.0:
            world.add_debug(f"MISSED_WEAK_ENEMY target=p{target.id} prod={int(target.production)} ships={int(target.ships)}")
            break
    for enemy in world.enemy_planets:
        prev_ships = _prev_ships.get(enemy.id)
        if prev_ships is None:
            continue
        drop = int(prev_ships) + int(enemy.production) - int(enemy.ships)
        if drop >= max(20, int(enemy.production) * 5):
            world.add_debug(f"MISSED_DRAINED_SOURCE_NOT_ATTACKED target=p{enemy.id} drop={drop}")
            break


def _midgame_dominance_surplus(world, states):
    return sum(
        max(0, min(st.safe_surplus, int(world.planet_by_id[pid].ships) - world.committed.get(pid, 0) - st.reserve))
        for pid, st in states.items()
        if pid in world.planet_by_id and not st.threatened
    )


def _midgame_dominance_eligible(world, states):
    if not (160 <= world.step <= 440):
        return False, 0
    if not (world.my_total_ships > world.enemy_total_ships * 1.4 or world.my_prod >= world.enemy_prod):
        return False, 0
    if len(world.my_planets) < 8:
        return False, 0
    if any(world.real_incoming_threat(p)["deficit"] > 0 for p in world.my_planets):
        return False, 0
    surplus = _midgame_dominance_surplus(world, states)
    if surplus < 150:
        return False, surplus
    return True, surplus


def _midgame_dominance_category(world, target, chain_plan):
    is_enemy = target.owner not in (-1, world.player)
    prod = int(target.production)
    weak = int(target.ships) <= max(45, prod * 8)
    nearest = min((dp(p, target) for p in world.my_planets), default=999.0)
    near_chain = target.id in set(chain_plan[:16]) or nearest <= 78.0
    frontier = is_local_enemy_opportunity(world, target) or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
    if is_enemy and weak and prod >= 3:
        return 0
    if is_enemy and near_chain:
        return 1
    if is_enemy and frontier:
        return 2
    if target.owner == -1 and prod >= 3:
        return 3
    if nearest <= 85.0 and _target_improves_control(world, target, chain_plan):
        return 4
    return 9


def _midgame_dominance_score(world, target, chain_plan):
    nearest = min((dp(p, target) for p in world.my_planets), default=999.0)
    prod = int(target.production)
    is_enemy = target.owner not in (-1, world.player)
    score = 0.0
    score += prod * (95.0 if is_enemy else 70.0)
    score += max(0.0, 105.0 - nearest) * 2.0
    score += len(_campaign_followup_options(world, target)) * 34.0
    score -= int(target.ships) * (0.8 if is_enemy else 0.55)
    if _planet_role(target) == ROLE_LAUNCHPAD:
        score += 115.0
    elif _planet_role(target) == ROLE_BRIDGE:
        score += 70.0
    if is_idle(target):
        score += 45.0
    if target.id in set(chain_plan[:16]):
        score += 85.0
    if is_enemy and is_local_enemy_opportunity(world, target):
        score += 75.0
    return score


def _midgame_dominance_make_prop(world, states, chain_plan, target):
    if target.owner == world.player or world.is_comet(target):
        return None
    category = _midgame_dominance_category(world, target, chain_plan)
    if category >= 9:
        return None
    mission_type = "LOCAL_PRODUCTION_CAPTURE" if target.owner == -1 else "SYNC_ATTACK"
    hold = max(5, int(target.production) * 2) if target.owner == -1 else max(18, int(target.production) * 6)
    planned, total, ok = _fund_capture(
        world,
        target,
        states,
        max_sources=6,
        mission_reason="midgame_dominance_attack",
        hold_margin_override=hold,
        source_radius=96.0,
    )
    if not ok:
        return None
    for _src_id, ships, _angle, _eta in planned:
        if not valid_packet_size(mission_type, ships):
            world.add_debug(f"INVALID_PACKET_REJECTED midgame_dominance target=p{target.id} ships={ships}")
            return None
    ok_grp, reason = validate_grouped_launch(world, target, planned)
    if not ok_grp:
        world.add_debug(f"MIDGAME_DOMINANCE_REJECT target=p{target.id} reason={reason}")
        return None
    eta_vals = [eta for _, _, _, eta in planned]
    if not world.can_hold_after_capture(target, max(eta_vals), total):
        world.add_debug(f"MIDGAME_DOMINANCE_REJECT target=p{target.id} reason=not_holdable")
        return None
    score = _midgame_dominance_score(world, target, chain_plan)
    return MissionProposal(
        kind=mission_type,
        target_id=target.id,
        priority=175.0 + score * 0.03,
        required_ships=total,
        planned_sources=planned,
        eta_min=min(eta_vals),
        eta_max=max(eta_vals),
        reason=f"midgame_dominance category={category} score={score:.1f}",
    )


def midgame_dominance_attack_props(world, states, chain_plan, deadline, *, limit=2):
    eligible, surplus = _midgame_dominance_eligible(world, states)
    if not eligible:
        return [], surplus
    candidates = []
    for target in world.enemy_planets + world.neutral_planets:
        if time.perf_counter() > deadline:
            break
        if target.owner == world.player or world.is_comet(target):
            continue
        category = _midgame_dominance_category(world, target, chain_plan)
        if category >= 9:
            continue
        nearest = min((dp(p, target) for p in world.my_planets), default=999.0)
        if nearest > 105.0:
            continue
        candidates.append((category, -_midgame_dominance_score(world, target, chain_plan), nearest, int(target.ships), target))
    candidates.sort()
    props = []
    seen = set()
    for _category, _neg_score, _nearest, _ships, target in candidates[:14]:
        if time.perf_counter() > deadline or len(props) >= limit:
            break
        if target.id in seen:
            continue
        prop = _midgame_dominance_make_prop(world, states, chain_plan, target)
        if prop is None:
            continue
        props.append(prop)
        seen.add(target.id)
    return props, surplus


def run_midgame_dominance_attack_mode(world, states, chain_plan, moves, deadline):
    if time.perf_counter() > deadline:
        return False
    props, surplus = midgame_dominance_attack_props(world, states, chain_plan, deadline, limit=2)
    if not props:
        return False
    world.add_debug("MIDGAME_DOMINANCE_ATTACK_TRIGGERED")
    world.add_debug(f"MIDGAME_IDLE_ARMY_DETECTED surplus={surplus} total={world.my_total_ships}")
    last = _midgame_dominance_last.get(world.player, -999)
    if world.step - last < 10 and surplus < 650:
        return False
    max_missions = 2 if surplus >= 650 or world.step - last >= 25 else 1
    launched = 0
    for prop in props[:max_missions]:
        if time.perf_counter() > deadline:
            break
        if _commit_proposal(world, prop, moves):
            launched += 1
            _midgame_dominance_last[world.player] = world.step
            world.add_debug(f"MIDGAME_ATTACK_SELECTED target=p{prop.target_id} total={prop.required_ships}")
            world.add_debug("MIDGAME_FORCE_CONVERTED_BEFORE_FINAL_DRAIN")
            world.add_debug("NO_WAIT_UNTIL_FINAL_DRAIN")
    return launched > 0


def _small_enemy_planet_score(world, target, chain_plan, phase):
    nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
    score = max(0.0, 95.0 - nearest) * 2.0 - int(target.ships) * 1.2
    if target.id in set(chain_plan[:18]) or len(_campaign_followup_options(world, target)) >= 1:
        score += 95.0
    if is_local_enemy_opportunity(world, target):
        score += 75.0
    if enemy_planets_total(world) <= 8:
        score += 180.0
    if phase in (ControlPhase.DOMINANCE_PHASE, ControlPhase.COLLAPSE_PHASE):
        score += 120.0
    if target.owner == world.leader:
        score += 45.0
    return score


def run_small_opponent_planet_capture_mode(world, states, chain_plan, phase, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    candidates = []
    for target in world.enemy_planets:
        if world.is_comet(target) or _planet_role(target) != ROLE_STORAGE:
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > 82.0 and phase != ControlPhase.COLLAPSE_PHASE:
            world.add_debug(f"SMALL_PLANET_REJECTED_FAR_ISOLATED p{target.id} d={nearest:.1f}")
            continue
        route_value = (
            target.id in set(chain_plan[:18])
            or len(_campaign_followup_options(world, target)) >= 1
            or is_local_enemy_opportunity(world, target)
            or phase in (ControlPhase.DOMINANCE_PHASE, ControlPhase.COLLAPSE_PHASE)
            or enemy_planets_total(world) <= 8
        )
        if not route_value:
            world.add_debug(f"SMALL_PLANET_REJECTED_FAR_ISOLATED p{target.id} d={nearest:.1f}")
            continue
        score = _small_enemy_planet_score(world, target, chain_plan, phase)
        candidates.append((score, nearest, int(target.ships), target))
    if not candidates:
        return False
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    world.add_debug("SMALL_OPPONENT_PLANET_CAPTURE_MODE")
    for score, _nearest, _ships, target in candidates[:6]:
        if target.id in set(chain_plan[:18]) or len(_campaign_followup_options(world, target)) >= 1:
            world.add_debug(f"SMALL_PLANET_BRIDGE_BONUS p{target.id}")
        if enemy_planets_total(world) <= 8:
            world.add_debug(f"SMALL_PLANET_LAST_ENEMY_BONUS p{target.id}")
        world.add_debug(f"SMALL_PLANET_CONTROL_BONUS p{target.id}")
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            "small_opponent_planet_capture",
            136.0 + score * 0.04,
            max_sources=4,
            source_radius=92.0 if phase == ControlPhase.COLLAPSE_PHASE else 76.0,
            hold_margin=max(8, int(target.production) * 3),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug(f"SMALL_ENEMY_PLANET_TARGETED p{target.id}")
            return True
    return False


def run_control_phase_attack_mode(world, states, chain_plan, phase, control_ratio, moves, deadline):
    if moves or time.perf_counter() > deadline:
        return False
    if phase not in (ControlPhase.DOMINANCE_PHASE, ControlPhase.COLLAPSE_PHASE):
        return False
    if phase == ControlPhase.DOMINANCE_PHASE:
        world.add_debug("DOMINANCE_PHASE_TRIGGERED")
        world.add_debug("CONTROL_BASED_FINAL_PHASE")
        world.add_debug("STEP_BASED_FINAL_PHASE_BYPASSED")
    else:
        world.add_debug("COLLAPSE_PHASE_BY_PLANET_CONTROL")
    candidates = []
    for target in world.enemy_planets:
        if world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        role = _planet_role(target)
        category = 0
        if int(target.production) >= 3:
            category = 0
        elif role == ROLE_LAUNCHPAD:
            category = 1
        elif role == ROLE_BRIDGE or is_local_enemy_opportunity(world, target):
            category = 2
        elif role == ROLE_STORAGE:
            category = 3
        if phase == ControlPhase.COLLAPSE_PHASE and enemy_planets_total(world) <= 6:
            category = min(category, 2)
        score = (
            int(target.production) * 80.0
            + max(0.0, 110.0 - nearest) * 1.8
            - int(target.ships) * 0.75
            + (90.0 if role == ROLE_STORAGE else 0.0)
            + (160.0 if enemy_planets_total(world) <= 6 else 0.0)
        )
        candidates.append((category, -score, nearest, int(target.ships), target))
    candidates.sort()
    for _category, neg_score, _nearest, _ships, target in candidates[:10]:
        reason = "control_based_collapse_attack" if phase == ControlPhase.COLLAPSE_PHASE else "dominance_phase_attack"
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            reason,
            170.0 + (-neg_score) * 0.03,
            max_sources=6,
            source_radius=120.0 if phase == ControlPhase.COLLAPSE_PHASE else 95.0,
            hold_margin=max(12, int(target.production) * 5),
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            if phase == ControlPhase.COLLAPSE_PHASE:
                world.add_debug(f"CONTROL_BASED_COLLAPSE_ATTACK p{target.id}")
                world.add_debug("FINAL_DRAIN_STARTED_EARLY_BY_CONTROL")
            else:
                world.add_debug(f"DOMINANCE_ATTACK_SELECTED p{target.id}")
            world.add_debug("NO_WAIT_UNTIL_FINAL_STEPS")
            return True
    return False


# ── midgame conversion mode ───────────────────────────────────────────────────

def _recent_launches_mostly_to_owned(world, horizon=20):
    records = [
        rec for rec in _recent_launch_history.get(world.player, [])
        if world.step - rec.get("step", world.step) <= horizon
    ]
    if len(records) < 3:
        return False, 0.0, len(records)
    owned = sum(1 for rec in records if rec.get("target_owned"))
    ratio = owned / max(1, len(records))
    return ratio >= 0.55, ratio, len(records)


def _active_offensive_capture_incoming(world, horizon=15):
    for pid, arrivals in world.arrivals_by_target.items():
        tgt = world.planet_by_id.get(pid)
        if tgt is None or tgt.owner == world.player or world.is_comet(tgt):
            continue
        soon = [
            (eta, ships)
            for eta, owner, ships in arrivals
            if owner == world.player and eta <= horizon
        ]
        if not soon:
            continue
        incoming = sum(ships for _eta, ships in soon)
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is not None and incoming >= world.required_ships_to_capture(tgt, src):
            return True
    return False


def midgame_conversion_context(world, control_ratio):
    rec = _midgame_conversion_memory.setdefault(world.player, {
        "best_my_count": len(world.my_planets),
        "last_increase_step": world.step,
        "prev_my_count": len(world.my_planets),
        "prev_enemy_max": _max_opponent_planets(world),
    })
    my_count = len(world.my_planets)
    enemy_max = _max_opponent_planets(world)
    prev_my = int(rec.get("prev_my_count", my_count))
    prev_enemy = int(rec.get("prev_enemy_max", enemy_max))
    my_growth = my_count - prev_my
    enemy_growth = enemy_max - prev_enemy

    if my_count > int(rec.get("best_my_count", my_count)):
        rec["best_my_count"] = my_count
        rec["last_increase_step"] = world.step
    stall_turns = world.step - int(rec.get("last_increase_step", world.step))

    own_rotation, own_ratio, launch_count = _recent_launches_mostly_to_owned(world)
    enemy_faster = enemy_growth > max(0, my_growth)
    enemy_more = enemy_max > my_count
    neutral_pressure = bool(world.neutral_planets) and control_ratio < PHASE_EXPAND_PCT
    # Trigger faster when stall is 15+ turns (was 20) or hard stall at 10+ with no neutrals nearby
    hard_stall = stall_turns >= 10 and not _nearby_neutral_exists(world) and enemy_more
    active = (
        world.step > 60
        and (
            stall_turns >= 15
            or hard_stall
            or enemy_faster
            or neutral_pressure
            or enemy_more
            or own_rotation
        )
    )

    if enemy_more:
        world.add_debug(
            f"ENEMY_PLANET_COUNT_PRESSURE enemy={enemy_max} mine={my_count}"
        )
    if own_rotation:
        world.add_debug(
            f"NO_MORE_PASSIVE_REINFORCE_LOOP owned_launch_ratio={own_ratio:.2f} launches={launch_count}"
        )
    if active:
        world.add_debug(
            f"MIDGAME_CONVERSION_MODE_ACTIVE stall={stall_turns} "
            f"my_growth={my_growth} enemy_growth={enemy_growth} "
            f"neutral_pressure={int(neutral_pressure)} enemy_more={int(enemy_more)}"
        )

    rec["prev_my_count"] = my_count
    rec["prev_enemy_max"] = enemy_max
    return {
        "active": active,
        "stall_turns": stall_turns,
        "enemy_faster": enemy_faster,
        "enemy_more": enemy_more,
        "neutral_pressure": neutral_pressure,
        "own_rotation": own_rotation,
        "own_rotation_ratio": own_ratio,
        "launch_count": launch_count,
        "enemy_max": enemy_max,
    }


def _midgame_conversion_candidate_score(world, target, chain_plan, context, enemy_actions):
    nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    role = _planet_role(target)
    followups = _campaign_followup_options(world, target)
    drained = 0
    for item in (enemy_actions or {}).get("drained", []):
        if item["source"].id == target.id:
            drained = max(drained, int(item["drop"]))
    score = 0.0
    score += max(0.0, 78.0 - nearest) * 3.0
    score += int(target.production) * 60.0
    score += min(4, len(followups)) * 42.0
    score -= int(target.ships) * 1.15
    if target.owner == -1:
        score += 85.0
        if nearest <= 46.0:
            score += 65.0
        if int(target.production) >= 3:
            score += 85.0
    else:
        if is_local_enemy_opportunity(world, target):
            score += 105.0
        if int(target.ships) <= max(18, int(target.production) * 5):
            score += 75.0
        if context.get("enemy_more"):
            score += 130.0
        if drained:
            score += drained * 3.0
    if role in (ROLE_BRIDGE, ROLE_LAUNCHPAD):
        score += 80.0
    if is_idle(target):
        score += 65.0
    if target.id in set(chain_plan[:16]):
        score += 70.0
    if role == ROLE_STORAGE and target.owner == -1 and not _chain_small_has_value(world, target):
        score -= 65.0
    # ROI bonus: prefer cheap high-return captures
    if src is not None:
        rough_need = max(MIN_SEND_SHIPS, int(target.ships) + 1)
        roi = capture_conversion_score(world, target, src, rough_need)
        score += min(80.0, roi * 28.0)
    return score


def run_midgame_conversion_mode(world, states, chain_plan, enemy_actions, context, moves, deadline):
    if moves or time.perf_counter() > deadline or not context.get("active"):
        return False
    world.add_debug("MIDGAME_CONVERSION_MODE_ACTIVE")
    if not world.neutral_planets:
        world.add_debug("NEAREST_WEAK_ENEMY_PRESSURE")

    capture_due = _active_offensive_capture_incoming(world, horizon=15)
    force_due = (
        context.get("stall_turns", 0) >= 20
        or context.get("enemy_faster")
        or context.get("enemy_more")
        or (world.step - _last_capture_step.get(world.player, -999) >= 5 and not capture_due)
    )
    if force_due:
        world.add_debug("FORCED_CAPTURE_AFTER_STALL")

    candidates = []
    for target in world.neutral_planets + world.enemy_planets:
        if time.perf_counter() > deadline:
            break
        if target.owner == world.player or world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > 82.0:
            continue
        if target.owner not in (-1, world.player) and not (
            is_local_enemy_opportunity(world, target)
            or context.get("enemy_more")
            or not world.neutral_planets
            or int(target.production) >= 3
        ):
            continue
        if target.owner == -1 and _planet_role(target) == ROLE_STORAGE and not (
            _chain_small_has_value(world, target)
            or len(_campaign_followup_options(world, target)) >= 1
            or nearest <= 30.0
        ):
            continue
        score = _midgame_conversion_candidate_score(world, target, chain_plan, context, enemy_actions)
        candidates.append(( -score, nearest, int(target.ships), target))

    candidates.sort()
    for neg_score, nearest, _ships, target in candidates[:12]:
        if time.perf_counter() > deadline:
            break
        hold = 2 if target.owner == -1 else max(8, int(target.production) * 3)
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            "midgame_conversion_force",
            184.0 + (-neg_score) * 0.03,
            max_sources=5,
            source_radius=86.0,
            hold_margin=hold,
            require_hold=target.owner != -1,
        )
        if prop is None:
            continue
        if _commit_proposal(world, prop, moves):
            world.add_debug(
                f"FORCED_CAPTURE_AFTER_STALL target=p{target.id} "
                f"nearest={nearest:.1f} total={prop.required_ships}"
            )
            if target.owner not in (-1, world.player) and not world.neutral_planets:
                world.add_debug("NEAREST_WEAK_ENEMY_PRESSURE")
            _last_capture_step[world.player] = world.step
            _beam_expansion_last[world.player] = world.step
            return True

    return False


# ── expansion tempo additions ────────────────────────────────────────────────

def capture_conversion_score(world, target, src, need):
    """
    True capture ROI: (net production value over hold horizon) / fleet cost.
    Accounts for ETA time cost, hold probability, and remaining game turns.
    Returns efficiency ratio: higher = better return on ships invested.
    0.0 if need <= 0 or target is a comet.
    """
    if need <= 0 or world.is_comet(target):
        return 0.0
    prod = max(1, int(target.production))
    eta = world.eta(src, target, need)
    arrival = max(1, int(math.ceil(eta)))
    future = max(1, world.remaining - arrival)

    can_hold = world.can_hold_after_capture(target, eta, need)
    hold_factor = 1.0 if can_hold else 0.25

    # Production that arrives over the game after capture
    gross_value = prod * future * hold_factor
    # Ships tied up in transit also have an opportunity cost
    time_cost = eta * (prod * 0.4)
    net_value = gross_value - time_cost

    return net_value / max(1.0, float(need))


def _over_rotation_active(world):
    """
    True when idle in-flight ships are wasting resources AND expansion is stalled.

    Uses flying_ship_breakdown() to measure actually-wasted (idle) fleets rather
    than raw fleet ratio, so legitimate defense/relay flights are not counted.

    Triggers when:
      idle_flying / total_force > 0.18
      AND no capture committed in the last 8 turns.
    """
    if not world.my_planets:
        return False
    useful_flying, idle_flying = world.flying_ship_breakdown(world.player)
    if idle_flying < MIN_SEND_SHIPS:
        return False
    stationed = sum(int(p.ships) for p in world.my_planets)
    total_force = max(1, stationed + useful_flying + idle_flying)
    idle_ratio = idle_flying / total_force
    if idle_ratio < 0.18:
        return False
    last = _last_capture_step.get(world.player, 0)
    if world.step - last < 8:
        return False
    world.add_debug(
        f"OVER_ROTATION_CORRECTION_ACTIVE idle_ratio={idle_ratio:.2f} "
        f"idle_ships={idle_flying} stall_turns={world.step - last}"
    )
    return True


def apply_over_rotation_correction(world, states, moves, deadline):
    """
    Active over-rotation correction: fires a relaxed expansion pass targeting the
    nearest convertible planet. Runs even when planets >= 4 (up to 6) so mid-game
    fleet spinning is also corrected.
    Resets _beam_expansion_last so beam will fire again next turn if needed.
    """
    if moves or time.perf_counter() > deadline:
        return False
    if not _over_rotation_active(world):
        return False
    if not world.neutral_planets and not world.enemy_planets:
        return False

    fleet_ratio = compute_fleet_ratio(world)
    if fleet_ratio > FLEET_RATIO_HARD:
        return False

    world.add_debug("OVER_ROTATION_CORRECTION_ACTIVE")
    world.add_debug("EXPANSION_OBLIGATION_FORCE_CAPTURE")

    # Relax beam obligation gap so it fires again next turn
    _beam_expansion_last[world.player] = world.step - BEAM_OBLIGATION_GAP - 1

    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < MIN_SEND_SHIPS:
        return False

    candidates = []
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        if target.owner not in (-1, world.player):
            if not should_allow_enemy_attack(world, target, "SYNC_ATTACK", "over_rotation_fix"):
                continue
            if int(target.ships) > 22 and int(target.production) < 3:
                continue
        src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
        if src is None:
            continue
        d = dp(src, target)
        if d > 62.0:
            continue
        need = world.ships_needed_to_capture(src, target, pool)
        if need <= 0 or pool < need:
            continue
        eta = world.eta(src, target, normalize_send_amount(need))
        if eta > 48.0:
            continue
        roi = capture_conversion_score(world, target, src, need)
        score = (
            roi * 130.0
            + int(target.production) * 50.0
            + (90.0 if is_idle(target) else 0.0)
            + (70.0 if _planet_role(target) in (ROLE_LAUNCHPAD, ROLE_BRIDGE) else 0.0)
            + max(0.0, 32.0 - d) * 5.0
            - d * 1.5
            - need * 0.8
            - eta * 3.0
        )
        candidates.append((score, d, need, target, src))

    if not candidates:
        return False

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    for score, d, need, target, _src in candidates[:6]:
        if time.perf_counter() > deadline:
            break
        hold = 1 if target.owner == -1 else max(5, int(target.production) * 2)
        plan, total, ok = _fund_capture(
            world, target, states,
            max_sources=4,
            mission_reason="over_rotation_correction",
            hold_margin_override=hold,
            source_radius=66.0,
        )
        if not ok:
            continue
        ok_grp, _ = validate_grouped_launch(world, target, plan)
        if not ok_grp:
            continue
        eta_vals = [e for _, _, _, e in plan]
        kind = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = MissionProposal(
            kind=kind,
            target_id=target.id,
            priority=182.0 + score * 0.04,
            required_ships=total,
            planned_sources=plan,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"over_rotation_correction p{target.id} roi={capture_conversion_score(world, target, _src, total):.2f}",
        )
        if _commit_proposal(world, prop, moves):
            world.add_debug(
                f"OVER_ROTATION_CORRECTION_ACTIVE target=p{target.id} "
                f"prod={int(target.production)} d={d:.1f}"
            )
            _last_capture_step[world.player] = world.step
            return True
    return False


def run_early_direct_expansion_mode(world, states, moves, deadline):
    """
    EARLY_DIRECT_EXPANSION_MODE: fast nearest-first expansion when planets < 4.
    Fires before beam expansion and strategic chain logic.
    Priority: nearest capturable neutral > weak enemy > prod4/5 > bridge > route-fill.
    Less hesitation, direct commits, relaxed hold margins.
    """
    if world.step >= 100 or len(world.my_planets) >= 4:
        return False
    if moves or time.perf_counter() > deadline:
        return False
    if not world.neutral_planets and not world.enemy_planets:
        return False

    fleet_ratio = compute_fleet_ratio(world)
    if fleet_ratio > FLEET_RATIO_SOFT:
        return False

    world.add_debug("EARLY_DIRECT_EXPANSION_MODE_ACTIVE")

    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < MIN_SEND_SHIPS:
        return False

    candidates = []
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        if target.owner not in (-1, world.player):
            if not should_allow_enemy_attack(world, target, "SYNC_ATTACK", "early_direct_expansion"):
                continue
            # skip heavily fortified low-value enemies in the very early game
            if int(target.ships) > 18 and int(target.production) < 3:
                continue

        src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
        if src is None:
            continue
        d = dp(src, target)
        if d > 58.0:
            continue

        need = world.ships_needed_to_capture(src, target, pool)
        if need <= 0 or pool < need:
            continue
        need_norm = normalize_send_amount(need)
        eta = world.eta(src, target, need_norm)
        if eta > 42.0:
            continue

        prod = int(target.production)
        my_eta, enemy_eta = world.reaction_times(target)
        conv = capture_conversion_score(world, target, src, need_norm)
        radius_gain = max(0.0, float(target.radius) - float(src.radius))
        score = (
            conv * 110.0
            + prod * 60.0
            + radius_gain * 140.0
            + (190.0 if prod >= 5 else 145.0 if prod >= 4 else 85.0 if prod >= 3 else 0.0)
            + (110.0 if is_idle(target) else 0.0)
            + (120.0 if _planet_role(target) == ROLE_LAUNCHPAD else 80.0 if _planet_role(target) == ROLE_BRIDGE else 0.0)
            + (32.0 if target.owner not in (-1, world.player) and int(target.ships) <= ENEMY_GATE_WEAK_LOCAL else 0.0)
            + max(0.0, 32.0 - d) * 5.0
            + min(4, len(_campaign_followup_options(world, target))) * 22.0
            - d * 1.6
            - need_norm * 0.8
            - eta * 3.2
        )
        if prod <= 1 and d > 28.0:
            score -= 55.0
        if enemy_eta < my_eta - 4.0:
            score -= 45.0   # enemy-favored race: deprioritise but don't skip
        candidates.append((score, d, need_norm, target, src))

    if not candidates:
        return False

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    for score, d, need_norm, target, primary_src in candidates[:8]:
        if time.perf_counter() > deadline:
            break
        hold = 1 if target.owner == -1 else max(5, int(target.production) * 2)
        plan, total, ok = _fund_capture(
            world,
            target,
            states,
            max_sources=3,
            mission_reason="early_direct_expansion",
            hold_margin_override=hold,
            source_radius=62.0,
        )
        if not ok:
            continue
        ok_grp, grp_reason = validate_grouped_launch(world, target, plan)
        if not ok_grp:
            world.add_debug(f"EARLY_DIRECT_SKIP p{target.id} reason={grp_reason}")
            continue
        eta_vals = [e for _, _, _, e in plan]
        if not world.can_hold_after_capture(target, max(eta_vals), total):
            world.add_debug(f"EARLY_DIRECT_SKIP p{target.id} reason=not_holdable")
            continue
        kind = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = MissionProposal(
            kind=kind,
            target_id=target.id,
            priority=188.0 + score * 0.05,
            required_ships=total,
            planned_sources=plan,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"early_direct_expansion p{target.id} d={d:.1f}",
        )
        if _commit_proposal(world, prop, moves):
            world.add_debug(
                f"EARLY_DIRECT_EXPANSION_MODE_ACTIVE target=p{target.id} "
                f"prod={int(target.production)} d={d:.1f} conv={capture_conversion_score(world, target, primary_src, total):.2f}"
            )
            if world.step < 10:
                world.add_debug("OPENING_ATTACK_STARTED_BEFORE_STEP_10")
            if float(target.radius) > float(primary_src.radius) + 0.05:
                world.add_debug(f"BIGGER_RADIUS_FIRST_SELECTED p{target.id}")
            world.add_debug("EARLY_NEAREST_EXPANSION_ACTIVE")
            world.add_debug("FIRST_CAPTURE_360_ACTIVE")
            _last_capture_step[world.player] = world.step
            return True

    return False


def run_expansion_obligation_force(world, states, moves, deadline):
    """
    EXPANSION_OBLIGATION_FORCE_CAPTURE: if no capture in last N turns and
    my_pct < 0.22, forcibly reduce hold margins and capture the nearest cheap target.
    Prevents overthinking / passive waiting / excessive rallying.
    """
    if moves or time.perf_counter() > deadline:
        return False
    my_pct = len(world.my_planets) / max(1, len(world.normal_planets))
    if my_pct >= 0.22:
        return False

    # Cooldown gate: when losing production and not in final steps, enforce a
    # 20-turn wait between obligation attempts to avoid greedy overextension.
    OBLIGATION_PROD_DEFICIT_COOLDOWN = 20
    FINAL_STEPS_THRESHOLD = 80
    if world.my_prod <= world.enemy_prod and world.remaining > FINAL_STEPS_THRESHOLD:
        cooldown_until = _expansion_obligation_cooldown.get(world.player, 0)
        if world.step < cooldown_until:
            world.add_debug(
                f"EXPANSION_OBLIGATION_COOLDOWN blocked step={world.step} "
                f"cooldown_until={cooldown_until} my_prod={world.my_prod} enemy_prod={world.enemy_prod}"
            )
            return False

    last = _last_capture_step.get(world.player, 0)
    turns_idle = world.step - last
    threshold = 10 if world.step < 80 else 16
    if turns_idle < threshold:
        return False
    if not world.neutral_planets and not world.enemy_planets:
        return False

    world.add_debug(f"EXPANSION_OBLIGATION_FORCE_CAPTURE idle={turns_idle} my_pct={my_pct:.2f}")
    # Set the cooldown timestamp whenever we enter an obligation attempt so the
    # next attempt is gated even if this one fails to commit.
    if world.my_prod <= world.enemy_prod and world.remaining > FINAL_STEPS_THRESHOLD:
        _expansion_obligation_cooldown[world.player] = world.step + OBLIGATION_PROD_DEFICIT_COOLDOWN
        world.add_debug(
            f"EXPANSION_OBLIGATION_COOLDOWN_SET until={world.step + OBLIGATION_PROD_DEFICIT_COOLDOWN} "
            f"my_prod={world.my_prod} enemy_prod={world.enemy_prod}"
        )
    fleet_ratio = compute_fleet_ratio(world)
    if fleet_ratio > FLEET_RATIO_HARD:
        return False

    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < MIN_SEND_SHIPS:
        return False

    candidates = []
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        if target.owner not in (-1, world.player):
            if not should_allow_enemy_attack(world, target, "SYNC_ATTACK", "expansion_obligation"):
                continue
        src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
        if src is None:
            continue
        d = dp(src, target)
        if d > 68.0:
            continue
        need = world.ships_needed_to_capture(src, target, pool)
        if need <= 0 or pool < need:
            continue
        eta = world.eta(src, target, normalize_send_amount(need))
        if eta > 50.0:
            continue
        my_eta, enemy_eta = world.reaction_times(target)
        conv = capture_conversion_score(world, target, src, need)
        score = (
            conv * 90.0
            + int(target.production) * 45.0
            + (80.0 if is_idle(target) else 0.0)
            + max(0.0, 30.0 - d) * 4.0
            - d * 1.5
            - need * 0.7
            - eta * 3.0
        )
        if enemy_eta < my_eta - 5.0:
            score -= 40.0
        candidates.append((score, d, need, target, src))

    if not candidates:
        return False

    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    for score, d, need, target, _src in candidates[:6]:
        if time.perf_counter() > deadline:
            break
        # Relaxed margins — obligated to expand
        hold = 0 if target.owner == -1 else max(4, int(target.production) * 2)
        plan, total, ok = _fund_capture(
            world,
            target,
            states,
            max_sources=4,
            mission_reason="expansion_obligation_force",
            hold_margin_override=hold,
            source_radius=72.0,
        )
        if not ok:
            available = sum(world.surplus(p) for p in world.my_planets)
            need_est = world.ships_needed_to_capture(
                min(world.my_planets, key=lambda p: dp(p, target), default=None) or world.my_planets[0],
                target, available,
            )
            world.add_debug(
                f"OBLIGATION_FORCE_FUND_FAIL target=p{target.id} "
                f"prod={int(target.production)} tgt_ships={int(target.ships)} "
                f"required_ships={need_est} available_surplus={available}"
            )
            continue
        ok_grp, grp_reason = validate_grouped_launch(world, target, plan)
        if not ok_grp:
            world.add_debug(
                f"OBLIGATION_FORCE_SKIP p{target.id} reason={grp_reason} "
                f"required_ships={total} available_surplus={sum(world.surplus(p) for p in world.my_planets)}"
            )
            continue
        eta_vals = [e for _, _, _, e in plan]
        kind = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = MissionProposal(
            kind=kind,
            target_id=target.id,
            priority=175.0 + score * 0.04,
            required_ships=total,
            planned_sources=plan,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"expansion_obligation_force p{target.id} idle={turns_idle}",
        )
        if _commit_proposal(world, prop, moves):
            world.add_debug(
                f"EXPANSION_OBLIGATION_FORCE_CAPTURE committed p{target.id} "
                f"prod={int(target.production)} idle={turns_idle}"
            )
            _last_capture_step[world.player] = world.step
            return True

    return False


# ── Source lock disabled ──────────────────────────────────────────────────────
# The new strategy does not use backyard source locks.
# world.backyard_locked_sources stays empty; no sources are locked by threat tracking.

NO_SOURCE_LOCK_WITHOUT_SELECTED_MISSION = True   # debug flag; signals intent


# ── Main agent ────────────────────────────────────────────────────────────────

def _legacy_rule_agent(obs, config=None):
    """
    Launchpad-chain strategy agent.

    Pipeline:
      1. WorldModel + planet-state snapshot
      2. Emergency defense (savable planets only, packet-safe)
      3. Small-start escape / chain-retrigger if needed
      4. Build launchpad chain plan
      5. Main33 opening/arbiter/HV-neutral/local attack tempo
      6. Midgame territory stabilization if structure is fragile
      7. Opponent model adaptive response / tempo override / sequence planner
      8. Expansion obligation / campaign while neutrals or planet deficit remain
      9. Midgame dominance conversion before banking/final drain
      10. Idle army pressure / surplus conversion before banking
      11. Production bank / rally only after useful chain is built
      12. Rolling capture chain from staging
      13. Expansion fallback campaign
      14. Final drain (endgame only)

    No random fallback trickle.  No micro-fleet exceptions.
    Every fleet >= 10 ships and a multiple of 5.
    """
    start       = time.perf_counter()
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    deadline    = start + min(SOFT_DEADLINE, max(0.55, act_timeout * 0.82))
    world       = WorldModel(obs)

    if not world.my_planets:
        update_ownership_memory(world)
        return []

    world.add_debug("STRATEGY_REWRITE_ACTIVE")
    world.add_debug("MAIN35_HYBRID_ACTIVE")
    world.add_debug("MAIN36_RATING_CLIMB_ACTIVE")
    world.add_debug("MAIN33_EXPANSION_IMPORTED")
    world.add_debug("MAIN34_PACKET_RULE_PRESERVED")
    world.add_debug("MAIN34_CHAIN_ENGINE_PRESERVED")

    # Reset per-game state at turn 0/1
    if not hasattr(agent, "_chain_bank_turns"):
        agent._chain_bank_turns = {}
    if world.step <= 1:
        _prev_owners.clear()
        _prev_ships.clear()
        _opponent_model_memory.clear()
        _midgame_dominance_last.clear()
        _shot_history.clear()
        _bad_shot_patterns.clear()
        _beam_expansion_last.clear()
        _last_capture_step.clear()
        _midgame_conversion_memory.clear()
        _recent_launch_history.clear()
        _adaptive_meta_controllers.clear()
        _pending_mission_launches.clear()
        _pending_delayed_missions.clear()
        _expansion_obligation_cooldown.clear()
        _staging_controller_memory.clear()
        _territory_conversion_history.clear()
        agent._chain_bank_turns = {}

    moves       = []
    fleet_ratio = compute_fleet_ratio(world)

    # ── 1. Planet state snapshot ──────────────────────────────────────────────
    states = build_planet_states(world)
    update_rotational_hubs(world)
    prediction = build_prediction_timeline(world, horizons=(5, 10, 15, 20, 30, 40))
    opponent_forecasts = {
        owner: forecast_opponent_power(world, owner, horizon=10)
        for owner in set(world.enemy_prod_by_owner) | set(world.enemy_ships_by_owner)
    }
    enemy_actions = forecast_enemy_actions(world, horizons=(5, 10, 15, 20))
    opponent_models = build_opponent_model(world)
    adaptive_mode, adaptive_model = select_adaptive_strategy_mode(world, opponent_models)
    world.add_debug("PLANET_ROLE_CLASSIFIED")

    # ── 2. Predictive + emergency defense ─────────────────────────────────────
    if time.perf_counter() < deadline:
        predictive_defense(world, states, enemy_actions, moves, deadline)

    if time.perf_counter() < deadline:
        emergency_defense_chain(world, states, moves)

    # ── 2a. Small-start escape launcher ──────────────────────────────────────
    if not moves and time.perf_counter() < deadline:
        run_small_start_escape(world, states, moves, deadline)

    # ── 2b. Early direct expansion (step < 100, planets < 4) ─────────────────
    # Fast nearest-first expansion before beam/chain strategic logic.
    # Behaves like main40 opening tempo: direct, low-hesitation, chain second.
    if not moves and time.perf_counter() < deadline:
        run_early_direct_expansion_mode(world, states, moves, deadline)

    # ── 2c. Over-rotation correction (active idle-fleet detector) ─────────────
    # When idle in-flight ratio > 0.18 AND expansion stalled: fire direct capture
    # and reset beam obligation gap so beam fires immediately next turn too.
    if not moves and time.perf_counter() < deadline:
        apply_over_rotation_correction(world, states, moves, deadline)

    # ── 2d. Expansion obligation force (stall detector) ──────────────────────
    # If no capture in threshold turns and low occupancy, force the cheapest expand.
    if not moves and time.perf_counter() < deadline:
        run_expansion_obligation_force(world, states, moves, deadline)

    # ── 3. Chain retrigger: recent planet loss → attacker source/base first
    recent_lost = [
        p for p in world.enemy_planets
        if not world.is_comet(p)
        and _prev_owners.get(p.id) == world.player
    ]
    if recent_lost and time.perf_counter() < deadline:
        chain_plan_tmp = build_launchpad_chain_plan(world, states)
        for lost in recent_lost[:1]:
            prop = build_chain_retrigger_response(
                world, lost, states, chain_plan_tmp, prediction, opponent_forecasts
            )
            if prop is not None and time.perf_counter() < deadline:
                if _commit_proposal(world, prop, moves):
                    world.add_debug("CHAIN_RETRIGGER_RALLY_TO_STAGING")
                    break

    # ── 4. Build chain plan ───────────────────────────────────────────────────
    chain_plan = build_launchpad_chain_plan(world, states)
    world._active_chain_plan = chain_plan
    world._active_structure = classify_planet_structure(world, states, chain_plan)
    control_phase, control_ratio = control_phase_selected(world)
    midgame_conversion = midgame_conversion_context(world, control_ratio)
    world.midgame_conversion_active = bool(midgame_conversion.get("active"))

    # ── 5. Choose staging launchpad ───────────────────────────────────────────
    staging_id = choose_staging_launchpad(world, states, chain_plan)
    staging    = world.planet_by_id.get(staging_id) if staging_id else None

    # ── 5a. Beam expansion dominates early/mid local growth ──────────────────
    if not moves and time.perf_counter() < deadline:
        run_beam_expansion_engine(
            world, states, chain_plan, enemy_actions, moves, deadline,
            staging_id=staging_id, control_ratio=control_ratio,
        )

    if not moves and time.perf_counter() < deadline:
        run_drained_enemy_counterattack(world, states, enemy_actions, moves, deadline)

    if not moves and time.perf_counter() < deadline:
        run_midgame_conversion_mode(
            world, states, chain_plan, enemy_actions, midgame_conversion, moves, deadline
        )

    small_start_escape_pending = _beam_small_start_active(world) and not moves
    if small_start_escape_pending:
        world.add_debug("SMALL_START_ESCAPE_FORCED")
        world.add_debug("NO_PASSIVE_OPENING_ALLOWED")

    # ── 5b. Main33 tempo layers, funded/committed by main34 packet rules ─────
    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_nearest_useful_target_lock(world, states, chain_plan, moves, deadline)
        if moves and control_phase == ControlPhase.INITIAL_EXPANSION:
            world.add_debug("INITIAL_EXPANSION_NEAREST_TARGET")

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_midgame_stabilization_mode(world, states, chain_plan, moves, deadline)
        if moves and control_phase == ControlPhase.MIDGAME_CONTROL:
            world.add_debug("MIDGAME_STRUCTURE_STABILIZE")

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_small_opponent_planet_capture_mode(world, states, chain_plan, control_phase, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_four_player_expand_first(world, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        main35_opening_tempo(world, states, chain_plan, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        main35_nearest_occupiable_arbiter(world, states, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        main35_high_value_neutral_race(world, states, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        main35_expand_from_hub(world, states, chain_plan, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        main35_opportunity_attack(world, states, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        smash_props = generate_local_smash_missions(world, states, deadline)
        for prop in smash_props:
            if time.perf_counter() > deadline:
                break
            if _commit_proposal(world, prop, moves):
                world.add_debug("LOCAL_SMASH_COMMITTED")
                break

    # ── 7. Adaptive response, tempo override, and short sequence planner ─────
    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_adaptive_strategy_mode(world, states, chain_plan, opponent_models, adaptive_mode, adaptive_model, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_tempo_override_capture(world, states, chain_plan, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_search_sequence_planner(world, states, chain_plan, moves, deadline)

    # ── 8. Expansion campaign: keep converting affordable connected planets ──
    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_expansion_campaign(world, states, chain_plan, moves, deadline)

    # ── 9. Midgame dominance: convert large advantage before final drain ─────
    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_control_phase_attack_mode(world, states, chain_plan, control_phase, control_ratio, moves, deadline)

    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_midgame_dominance_attack_mode(world, states, chain_plan, moves, deadline)

    midgame_attack_blocked = False
    if not moves and time.perf_counter() < deadline:
        midgame_props, _midgame_surplus = midgame_dominance_attack_props(
            world, states, chain_plan, deadline, limit=1
        )
        if midgame_props:
            midgame_attack_blocked = True
            world.add_debug("MIDGAME_BANKING_BLOCKED_BY_ATTACK")

    # ── 10. Idle army pressure: convert parked surplus before banking ────────
    if not moves and not small_start_escape_pending and time.perf_counter() < deadline:
        run_surplus_conversion_mode(world, states, chain_plan, staging_id, moves, deadline)

    parked_bank_blocked = False
    if not moves and time.perf_counter() < deadline:
        parked_active, _parked_sources, _parked_targets, _parked, _flying_ratio, _threshold = _surplus_pressure_context(
            world, states, chain_plan, emit=False
        )
        if parked_active:
            parked_bank_blocked = True
            world.add_debug("PRODUCTION_BANK_BLOCKED_BY_PARKED_SURPLUS")

    # Track offensive capture launches only; reinforcement loops should not reset stall memory.
    if any(
        rec.get("step") == world.step and rec.get("offensive")
        for rec in _recent_launch_history.get(world.player, [])
    ):
        _last_capture_step[world.player] = world.step

    # ── 11. Production bank / rally ───────────────────────────────────────────
    affordable_expansion_exists = (not moves and has_affordable_campaign_capture(world, states))
    useful_capture_exists = (not moves and useful_capturable_target_exists(world, states, chain_plan))
    small_escape_blocked = not moves and _small_start_escape_mode(world)
    over_rotation_blocked = not moves and affordable_expansion_exists and _over_rotation_active(world)
    expansion_need_blocked = affordable_expansion_exists and (
        len(world.my_planets) < 6
        or _nearby_neutral_exists(world)
        or _opponent_planet_lead(world) > 0
    )
    conversion_bank_blocked = (
        not moves
        and midgame_conversion.get("active")
        and (affordable_expansion_exists or useful_capture_exists or midgame_conversion.get("enemy_more"))
    )
    if small_escape_blocked:
        world.add_debug("SMALL_START_ESCAPE_MODE")
        world.add_debug("BANKING_SKIPPED_USEFUL_CAPTURE_EXISTS")
    if expansion_need_blocked:
        world.add_debug("PRODUCTION_BANK_BLOCKED_BY_EXPANSION_NEED")
        world.add_debug("BANKING_SKIPPED_USEFUL_CAPTURE_EXISTS")
        if world.step <= EARLY_NEAREST_SWEEP_STEP_MAX:
            world.add_debug("EARLY_BANKING_BLOCKED_CAPTURE_EXISTS")
    if useful_capture_exists:
        world.add_debug("BANKING_BLOCKED_CAPTURE_EXISTS")
        world.add_debug("BANKING_SKIPPED_USEFUL_CAPTURE_EXISTS")
        if world.step <= EARLY_NEAREST_SWEEP_STEP_MAX:
            world.add_debug("EARLY_BANKING_BLOCKED_CAPTURE_EXISTS")
    if over_rotation_blocked:
        world.add_debug("OVER_ROTATION_CORRECTION_ACTIVE")
        world.add_debug("BANKING_BLOCKED_OVER_ROTATION")
    if conversion_bank_blocked:
        world.add_debug("MIDGAME_CONVERSION_MODE_ACTIVE")
        world.add_debug("NO_MORE_PASSIVE_REINFORCE_LOOP")
        world.add_debug("BANKING_SKIPPED_USEFUL_CAPTURE_EXISTS")
    bank_blocked = small_escape_blocked or midgame_attack_blocked or parked_bank_blocked or expansion_need_blocked or useful_capture_exists or over_rotation_blocked or conversion_bank_blocked or (
        not _campaign_chain_built_for_bank(world)
        and not moves
        and affordable_expansion_exists
    )
    if bank_blocked and not parked_bank_blocked and not expansion_need_blocked:
        world.add_debug("PRODUCTION_BANK_BLOCKED_UNTIL_CHAIN_BUILT")
    if (staging is not None
            and not moves
            and not small_start_escape_pending
            and not bank_blocked
            and fleet_ratio <= FLEET_RATIO_SOFT
            and time.perf_counter() < deadline):
        # Find next unowned chain target
        next_tgts = [
            world.planet_by_id[pid]
            for pid in chain_plan
            if pid in world.planet_by_id
            and world.planet_by_id[pid].owner != world.player
        ]
        chain_surplus = sum(st.safe_surplus for st in states.values())

        if next_tgts:
            tgt0    = next_tgts[0]
            need0   = world.ships_needed_to_capture(staging, tgt0, chain_surplus)
            prod5   = sum(int(p.production) for p in world.my_planets) * 5
            prod10  = sum(int(p.production) for p in world.my_planets) * 10
            bank_no = agent._chain_bank_turns.get(world.player, 0)
            opp_max = max((f["projected_ships"] for f in opponent_forecasts.values()), default=0)
            opp_near = min((f["nearest_d"] for f in opponent_forecasts.values()), default=999.0)

            if need0 > chain_surplus and bank_no < CHAIN_BANK_MAX_TURNS:
                # Worth banking a few more steps?
                can_bank_force = chain_surplus + prod10 >= need0
                opponent_breaks_first = opp_near < 24.0 and opp_max > max(1, chain_surplus + prod5) * 1.35
                if can_bank_force and not opponent_breaks_first:
                    world.add_debug(
                        f"PRODUCTION_BANK_MODE staging=p{staging_id} "
                        f"need={need0} have={chain_surplus} force10={chain_surplus + prod10}"
                    )
                    world.add_debug("PRODUCTION_BANK_WAIT_APPROVED")
                    agent._chain_bank_turns[world.player] = bank_no + 1
                    # Rally storage to staging while banking
                    rally_to_staging(world, staging_id, states, moves)
                else:
                    world.add_debug(
                        f"PRODUCTION_BANK_WAIT_REJECTED need={need0} force10={chain_surplus + prod10} "
                        f"opp_near={opp_near:.1f} opp_max={opp_max}"
                    )
            else:
                # Reset bank counter and attack
                agent._chain_bank_turns[world.player] = 0
        else:
            world.add_debug("PRODUCTION_BANK_WAIT_REJECTED")

    # ── 12. Rolling capture chain ─────────────────────────────────────────────
    if (staging_id
            and not moves
            and not small_start_escape_pending
            and fleet_ratio <= FLEET_RATIO_SOFT
            and time.perf_counter() < deadline):
        props = build_rolling_capture_chain(
            world, staging_id, states, chain_plan, fleet_ratio, prediction, opponent_forecasts
        )
        for prop in props:
            if time.perf_counter() > deadline:
                break
            if _commit_proposal(world, prop, moves):
                world.add_debug("ROLLING_CHAIN_LAUNCH_APPROVED")
                break

    # ── 13. Fallback: expansion campaign only (no trickle) ───────────────────
    if (not moves
            and not small_start_escape_pending
            and (fleet_ratio <= FLEET_RATIO_SOFT or expansion_obligation_active(world))
            and time.perf_counter() < deadline):
        world.add_debug("EXPANSION_FALLBACK_CAMPAIGN")
        fallback_tgts = sorted(
            [t for t in world.normal_planets
             if t.owner != world.player
             and not world.is_comet(t)
             and min((dp(m, t) for m in world.my_planets), default=999.0) <= CAMPAIGN_RADIUS + 12],
            key=lambda t: (
                min((dp(m, t) for m in world.my_planets), default=999.0),
                0 if len(_campaign_followup_options(world, t)) >= 2 else 1,
                -int(t.production),
                int(t.ships),
            ),
        )
        for tgt in fallback_tgts[:12]:
            if time.perf_counter() > deadline:
                break
            if not (_fallback_chain_value(world, tgt, chain_plan, prediction) or _small_start_escape_target_value(world, tgt)):
                world.add_debug(f"FALLBACK_REJECT_NO_CHAIN_VALUE target=p{tgt.id}")
                continue
            if tgt.owner not in (-1, world.player) and opponent_forecasts:
                owner_forecast = opponent_forecasts.get(tgt.owner, {})
                world.add_debug(
                    f"CHAIN_RETRIGGER_FORCE_FORECAST fallback_enemy=p{tgt.id} "
                    f"opp_projected={owner_forecast.get('projected_ships', 0)}"
                )
            planned, total, ok = _fund_capture(
                world,
                tgt,
                states,
                max_sources=3,
                mission_reason="expansion_campaign_fallback",
                hold_margin_override=2 if tgt.owner == -1 else max(6, int(tgt.production) * 3),
                source_radius=CAMPAIGN_RADIUS + 12,
            )
            if not ok:
                world.add_debug(f"CAMPAIGN_REJECT_UNAFFORDABLE target=p{tgt.id} reason=fallback")
                continue
            eta_vals = [e for _, _, _, e in planned]
            ok_grp, reason = validate_grouped_launch(world, tgt, planned)
            if not ok_grp:
                world.add_debug(f"CAMPAIGN_REJECT_NO_CONVERSION target=p{tgt.id} reason={reason}")
                continue
            if not world.can_hold_after_capture(tgt, max(eta_vals), total):
                world.add_debug(f"CAMPAIGN_REJECT_NO_CONVERSION target=p{tgt.id} reason=fallback_not_holdable")
                continue
            mtype = "CAPTURE_NEUTRAL" if tgt.owner == -1 else "SYNC_ATTACK"
            prop  = MissionProposal(
                kind=mtype,
                target_id=tgt.id,
                priority=40.0,
                required_ships=total,
                planned_sources=planned,
                eta_min=min(eta_vals),
                eta_max=max(eta_vals),
                reason=f"fallback_chain p{tgt.id}",
            )
            if _commit_proposal(world, prop, moves):
                world.add_debug(
                    f"NO_FALLBACK_TRICKLE target=p{tgt.id} total={total}"
                )
                world.add_debug(f"NO_EMPTY_TURN_WITH_VALID_CAPTURE target=p{tgt.id}")
                world.add_debug(f"VALID_CAPTURE_EXISTS_NO_EMPTY_TURN target=p{tgt.id}")
                break

    if not moves:
        log_missed_opportunities(world, states, chain_plan)

    # ── 14. Final drain (endgame only) ────────────────────────────────────────
    if control_based_final_phase(world, control_phase, control_ratio) and time.perf_counter() < deadline:
        if world.remaining >= 45:
            world.add_debug("FINAL_DRAIN_STARTED_EARLY_BY_CONTROL")
            world.add_debug("CONTROL_BASED_FINAL_PHASE")
            world.add_debug("CONTROL_BASED_FINAL_DRAIN")
        _final_drain_chain(world, moves, chain_plan)

    if DEBUG:
        for event in world.debug_events:
            print(event)
    update_ownership_memory(world)
    return moves


# ── Rolling-horizon beam-search decision loop ────────────────────────────────

BEAM_ROOT_COMPONENT_LIMIT = 34
BEAM_ROOT_COMBO_LIMIT = 40
BEAM_SEARCH_DEPTH = 14
BEAM_SEARCH_WIDTH = 6
BEAM_TIME_BUFFER = 0.005

# Fleet clustering: minimum fleet size before any offensive mission is allowed.
# Prevents the agent from scattering small, inefficient packets.
MIN_COMMIT_THRESHOLD_BASE  = 20    # absolute floor for any offensive send
MIN_COMMIT_DEFENSE_MULT    = 1.4   # threshold = max(BASE, ceil(ships * MULT))
MIN_COMMIT_HIGH_PROD       = 4     # production level above which a tighter multiplier applies
MIN_COMMIT_HIGH_PROD_MULT  = 2.0   # multiplier for high-production targets


def _proposal_source_totals(prop):
    totals = {}
    for src_id, ships, _angle, _eta in prop.planned_sources:
        totals[src_id] = totals.get(src_id, 0) + int(ships)
    return totals


def _min_commit_threshold(target):
    """Dynamic minimum fleet size required before launching an offensive mission.

    Scales with the target's current garrison so we never send a packet
    that is too small to overcome the defense.  High-production targets
    use a tighter multiplier to prevent cheap harassment raids.
    """
    defense = int(target.ships)
    prod = int(target.production)
    if prod >= MIN_COMMIT_HIGH_PROD:
        threshold = max(MIN_COMMIT_THRESHOLD_BASE, int(defense * MIN_COMMIT_HIGH_PROD_MULT) + prod * 2)
    else:
        threshold = max(MIN_COMMIT_THRESHOLD_BASE, int(defense * MIN_COMMIT_DEFENSE_MULT))
    return threshold


def _proposal_total_ships(prop):
    return sum(int(ships) for _src_id, ships, _angle, _eta in prop.planned_sources)


def _hub_security_garrison(p):
    """Conservative current-ship floor for production hubs."""
    return max(HUB_SECURITY_BASE_GARRISON, int(p.production) * HUB_SECURITY_PROD_GARRISON_MULT)


def _hub_security_violated(world):
    """Return (violated, details_str) when a productive hub is below its buffer."""
    for p in world.my_planets:
        if int(p.production) >= HUB_SECURITY_PROD_THRESHOLD:
            floor = _hub_security_garrison(p)
            if int(p.ships) < floor:
                return True, (
                    f"p{p.id} prod={int(p.production)} ships={int(p.ships)} "
                    f"garrison_floor={floor} surplus={world.surplus(p)}"
                )
    return False, ""


def _active_offensive_capture_missions(world):
    active = []
    for entry in world.mission_ledger.entries.values():
        if entry.status not in ("planned", "active") or entry.mission_type not in OFFENSIVE_MISSIONS:
            continue
        target = world.planet_by_id.get(entry.target_id)
        if target is None or target.owner == world.player or world.is_comet(target):
            continue
        active.append(entry)
    return active


def _has_significant_production_lead(world):
    if world.enemy_prod <= 0:
        return world.my_prod > 0
    return world.my_prod >= world.enemy_prod * SIGNIFICANT_PROD_LEAD_MULT


def _capture_projected_defense(world, target, eta_max):
    arrival_turn = max(1, int(math.ceil(eta_max)))
    owner_at, projected_defense = target_ships_at_arrival_frame(world, target, arrival_turn)
    travel_time = max(1, arrival_turn)
    growth_defense = int(target.ships) + int(target.production) * travel_time
    return owner_at, max(int(projected_defense), growth_defense), arrival_turn


def _critical_mass_required(projected_defense, mult=FINAL_CAPTURE_CRITICAL_MASS_MULT):
    return normalize_send_amount(int(math.ceil(max(0, projected_defense) * mult)) + 1)


def _is_two_stage_final(prop):
    return prop is not None and "two_stage_final" in (getattr(prop, "reason", "") or "")


def _is_two_stage_backup(prop):
    return prop is not None and "two_stage_backup" in (getattr(prop, "reason", "") or "")


def _two_stage_key(prop):
    reason = getattr(prop, "reason", "") or ""
    for token in reason.split():
        if token.startswith("stage_key="):
            return token.split("=", 1)[1]
    return ""


def _log_capture_force(world, label, kind, target, projected_defense, committed, required, arrival_turn):
    world.add_debug(
        f"{label} kind={kind} target_id={target.id} projected_defense={int(projected_defense)} "
        f"total_ships_committed={int(committed)} required={int(required)} arrival_step={world.step + int(arrival_turn)}"
    )


def _proposal_passes_capture_constraints(world, prop):
    if prop.kind not in OFFENSIVE_MISSIONS:
        return True
    target = world.planet_by_id.get(prop.target_id)
    if target is None or target.owner == world.player or world.is_comet(target):
        return False
    total_ships = _proposal_total_ships(prop)
    if total_ships == 5 and prop.kind in ("FINAL_DRAIN", "FINISH_ZERO_CAPTURE"):
        if all(
            valid_packet_size(prop.kind, ships, world, world.planet_by_id.get(src_id), target)
            for src_id, ships, _ls, _eta in prop.planned_sources
        ):
            _owner_at, projected_defense, arrival_turn = _capture_projected_defense(world, target, prop.eta_max)
            _log_capture_force(world, "CAPTURE_FORCE_CHECK", prop.kind, target, projected_defense, total_ships, 5, arrival_turn)
            return True
    if not _is_two_stage_final(prop):
        threshold = _min_commit_threshold(target)
        early_neutral = target.owner == -1 and world.step <= 70 and _is_opening_capture_reason(prop.reason)
        if early_neutral:
            src = world.planet_by_id.get(prop.planned_sources[0][0]) if prop.planned_sources else None
            required_fast = normalize_send_amount(_neutral_capture_fast_need(world, src, target, mission_reason=prop.reason))
            world.add_debug(
                f"NEUTRAL_CAPTURE_FAST_NEED_USED target=p{target.id} committed={total_ships} required={required_fast}"
            )
            if total_ships < required_fast:
                return False
            world.add_debug(f"NEUTRAL_CAPTURE_NO_EXTRA_WAIT target=p{target.id}")
            return True
        if total_ships < threshold:
            world.add_debug(
                f"FLEET_CLUSTER_REJECT kind={prop.kind} target=p{target.id} "
                f"ships={total_ships} threshold={threshold} defense={int(target.ships)} "
                f"prod={int(target.production)}"
            )
            return False
    arrival_turn = max(1, _offensive_proposal_arrival_step(world, prop) - world.step)
    _owner_at, projected_defense, _arrival_turn = _capture_projected_defense(world, target, arrival_turn)
    required_mass = _critical_mass_required(projected_defense)
    _log_capture_force(world, "CAPTURE_FORCE_CHECK", prop.kind, target, projected_defense, total_ships, required_mass, arrival_turn)
    if total_ships < required_mass:
        world.add_debug(
            f"CRITICAL_MASS_REJECT kind={prop.kind} target=p{target.id} "
            f"ships={total_ships} projected_defense={projected_defense} "
            f"required_1_25x={required_mass} arrival_step={world.step + arrival_turn}"
        )
        return False
    return True


def _offensive_proposal_arrival_step(world, prop):
    return max(
        (
            _proposal_launch_step(world, launch_step) + max(1, int(math.ceil(eta)))
            for _src_id, _ships, launch_step, eta in prop.planned_sources
        ),
        default=world.step + max(1, int(math.ceil(prop.eta_max))),
    )


def _synchronize_offensive_proposal_timing(world, prop):
    if not _sync_managed_group(prop):
        return prop
    target = world.planet_by_id.get(prop.target_id)
    latest_eta = max(float(eta) for _src_id, _ships, _launch_step, eta in prop.planned_sources)
    arrival_step = world.step + max(1, int(math.ceil(latest_eta)))
    synchronized = []
    arrival_steps = []
    for src_id, ships, _launch_step, eta in prop.planned_sources:
        travel_steps = max(1, int(math.ceil(eta)))
        launch_step = max(world.step, arrival_step - travel_steps)
        synchronized.append((src_id, ships, launch_step, eta))
        arrival_steps.append(launch_step + travel_steps)
    prop.planned_sources = synchronized
    prop.eta_min = min((eta for _sid, _s, _ls, eta in synchronized), default=prop.eta_min)
    prop.eta_max = max((eta for _sid, _s, _ls, eta in synchronized), default=prop.eta_max)
    ok_sync, spread, window = _sync_window_ok(world, target, prop.kind, prop.planned_sources, prop.reason)
    if not ok_sync:
        world.add_debug(
            f"SYNC_WINDOW_TOO_LOOSE_REJECTED target=p{prop.target_id} spread={spread:.1f} window={window}"
        )
        return prop
    world.add_debug(
        f"ARRIVAL_TIMING_SYNC mission={prop.kind} target_id={prop.target_id} "
        f"arrival_step={arrival_step} source_arrivals={arrival_steps} "
        f"launch_steps={[ls for _sid, _s, ls, _eta in synchronized]}"
    )
    world.add_debug(
        f"SYNC_ATTACK_LOCK_CREATED mission={prop.kind} target=p{prop.target_id} "
        f"arrival_step={arrival_step} spread={spread:.1f} window={window}"
    )
    if spread <= (window if window is not None else ETA_SYNC_WINDOW):
        world.add_debug(
            f"GROUPED_BURST_ARRIVAL_CONFIRMED target=p{prop.target_id} arrivals={arrival_steps}"
        )
    return prop


def _planned_sources_as_arrival_offsets(world, planned_sources):
    adjusted = []
    for src_id, ships, launch_step, eta in planned_sources or []:
        launch = _proposal_launch_step(world, launch_step)
        adjusted_eta = max(0, launch - world.step) + float(eta)
        adjusted.append((src_id, ships, 0, adjusted_eta))
    return adjusted


def _staging_fleet_ratio(world):
    return world.my_total_ships / max(1, world.enemy_total_ships)


def _staging_controller_active(world):
    if world.enemy_total_ships <= 0 or not world.enemy_planets:
        return False
    ratio = _staging_fleet_ratio(world)
    rec = _staging_controller_memory.setdefault(
        world.player,
        {"active_until": -999, "release_until": -999, "started_step": -999},
    )
    active_until = int(rec.get("active_until", -999))
    release_until = int(rec.get("release_until", -999))
    if world.step <= active_until:
        active = True
    elif world.step <= release_until:
        active = False
        world.add_debug(
            f"STAGING_CONTROLLER_RELEASED ratio={ratio:.2f} release_until={release_until}"
        )
    else:
        active = False
        if ratio > STAGING_FLEET_RATIO_TRIGGER:
            rec["started_step"] = world.step
            rec["active_until"] = world.step + STAGING_MIN_STEPS - 1
            rec["release_until"] = int(rec["active_until"]) + STAGING_RELEASE_STEPS
            active = True
    if active:
        world.add_debug(
            f"STAGING_CONTROLLER_ACTIVE ratio={ratio:.2f} "
            f"started={rec.get('started_step')} active_until={rec.get('active_until')} "
            f"release_until={rec.get('release_until')}"
        )
    return active


def _active_two_stage_sequence_pending(world):
    return any(
        "two_stage" in (rec.get("reason") or "")
        and int(rec.get("launch_step", world.step)) >= world.step
        for rec in _pending_mission_launches.get(world.player, [])
    )


def _launchpad_targets_for_staging(world, states, chain_plan):
    pads = [
        p for p in world.my_planets
        if p.id in states
        and (
            _planet_role(p) == ROLE_LAUNCHPAD
            or int(p.ships) >= LOCAL_HUB_SHIPS
            or p.id in set(chain_plan[:8])
        )
    ]
    if not pads:
        pads = [
            p for p in world.my_planets
            if p.id in states and int(p.production) >= HUB_SECURITY_PROD_THRESHOLD
        ]
    pads.sort(key=lambda p: (_planet_role(p) != ROLE_LAUNCHPAD, -int(p.production), -int(p.ships)))
    return pads[:3]


def _build_staging_controller_props(world, states, chain_plan, deadline):
    pads = _launchpad_targets_for_staging(world, states, chain_plan)
    if not pads:
        world.add_debug("STAGING_CONTROLLER_NO_LAUNCHPAD")
        return []
    buckets = {p.id: [] for p in pads}
    for src in sorted(world.my_planets, key=lambda p: -states.get(p.id, PlanetState(p.id, "", False, 0, 0, 0.0, 0, 0, False, 0.0)).safe_surplus):
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        st = states.get(src.id)
        if st is None or st.threatened or st.safe_surplus < MIN_SEND_SHIPS:
            continue
        target = min((p for p in pads if p.id != src.id), key=lambda p: dp(src, p), default=None)
        if target is None:
            continue
        send = round_down_to_granularity(st.safe_surplus)
        if send < MIN_SEND_SHIPS:
            continue
        safe, _reason = world.source_is_safe_for(src, target, "REINFORCE_CAPTURE", send, mission_reason="staging_controller")
        if not safe:
            continue
        eta = world.eta(src, target, send)
        buckets[target.id].append((src.id, send, 0, eta))
        world.add_debug(
            f"STAGING_ROUTE_SURPLUS src=p{src.id} launchpad=p{target.id} ships={send} eta={eta:.1f}"
        )
    proposals = []
    for target in pads:
        planned = buckets.get(target.id, [])
        if not planned:
            continue
        total = sum(ships for _sid, ships, _ls, _eta in planned)
        proposals.append(MissionProposal(
            kind="REINFORCE_CAPTURE",
            target_id=target.id,
            priority=225.0 + total * 0.4 + int(target.production) * 12.0,
            required_ships=total,
            planned_sources=planned,
            eta_min=min(eta for _sid, _s, _ls, eta in planned),
            eta_max=max(eta for _sid, _s, _ls, eta in planned),
            reason=f"staging_controller launchpad=p{target.id}",
            priority_tier="FLEXIBLE",
        ))
        world.add_debug(
            f"STAGING_GROUP_READY launchpad=p{target.id} sources={[sid for sid, _s, _ls, _eta in planned]} total={total}"
        )
    return proposals[:3]


def _staging_launchpad_score(world, staging, target, states, chain_plan):
    st = states.get(staging.id)
    if st is None or st.threatened or world.is_comet(staging):
        return None
    if world.real_incoming_threat(staging)["deficit"] > 0:
        return None
    enemy_d = world.nearest_enemy_distance(staging)
    if enemy_d < FRONTLINE_DIST * 0.75:
        return None
    role = _planet_role(staging)
    score = 0.0
    if radius_class(staging) == "LARGE":
        score += 90.0
    elif radius_class(staging) == "MEDIUM":
        score += 45.0
    if is_static_planet(staging):
        score += 55.0
    if role == ROLE_LAUNCHPAD:
        score += 70.0
    if role == ROLE_BRIDGE:
        score += 30.0
    if staging.id in set(chain_plan[:12]):
        score += 35.0
    score += max(0.0, 70.0 - abs(enemy_d - 45.0)) * 0.8
    score -= dp(staging, target) * 0.7
    score += st.safe_surplus * 0.25
    return score


def _select_two_stage_staging_planet(world, target, states, chain_plan, staging_id=None):
    preferred = world.planet_by_id.get(staging_id) if staging_id else None
    candidates = []
    for staging in world.my_planets:
        if staging.id == target.id:
            continue
        if preferred is not None and staging.id != preferred.id and dp(staging, target) > 76.0:
            continue
        score = _staging_launchpad_score(world, staging, target, states, chain_plan)
        if score is None:
            continue
        if preferred is not None and staging.id == preferred.id:
            score += 30.0
        candidates.append((score, staging))
    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[0])
    staging = candidates[0][1]
    world.add_debug(
        f"STAGING_LAUNCHPAD_SELECTED staging=p{staging.id} target=p{target.id} score={candidates[0][0]:.1f}"
    )
    return staging


def build_two_stage_grouped_assault_props(world, target, states, chain_plan, deadline, staging_id=None):
    if target is None or target.owner == world.player or world.is_comet(target):
        return []
    staging = _select_two_stage_staging_planet(world, target, states, chain_plan, staging_id=staging_id)
    if staging is None or staging.id not in states:
        return []
    if _active_two_stage_sequence_pending(world):
        world.add_debug("OVER_STAGING_REJECTED")
        return []

    stage_state = states[staging.id]
    eta_final = world.eta(staging, target, max(MIN_SEND_SHIPS, stage_state.safe_surplus))
    final_arrival_turn = max(1, int(math.ceil(eta_final)) + 1)
    _owner_at, projected_defense = target_ships_at_arrival_frame(world, target, final_arrival_turn)
    projected_defense = max(projected_defense, int(target.ships) + int(target.production) * final_arrival_turn)
    final_required = _critical_mass_required(projected_defense, FINAL_CAPTURE_CRITICAL_MASS_MULT)
    staging_required = _critical_mass_required(projected_defense, STAGING_CRITICAL_MASS_MULT)
    safe_on_stage = max(0, stage_state.safe_surplus - world.staging_reserved_ships.get(staging.id, 0))
    if safe_on_stage < staging_required:
        return []
    if safe_on_stage >= final_required:
        return []

    backup_needed = final_required - safe_on_stage
    backup_sources = sorted(
        [
            p for p in world.my_planets
            if p.id != staging.id
            and p.id in states
            and not states[p.id].threatened
            and states[p.id].safe_surplus >= MIN_SEND_SHIPS
            and dp(p, staging) <= 52.0
        ],
        key=lambda p: (world.eta(p, staging, min(states[p.id].safe_surplus, backup_needed)), dp(p, staging), -states[p.id].safe_surplus),
    )
    backup_plan = []
    backup_total = 0
    for src in backup_sources[:5]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER or backup_total >= backup_needed:
            break
        spare = min(states[src.id].safe_surplus, backup_needed - backup_total)
        send = round_down_to_granularity(spare)
        if send < MIN_SEND_SHIPS and states[src.id].safe_surplus >= backup_needed - backup_total:
            send = normalize_send_amount(backup_needed - backup_total)
        if send < MIN_SEND_SHIPS or send > states[src.id].safe_surplus:
            continue
        safe, _reason = world.source_is_safe_for(src, staging, "REINFORCE_CAPTURE", send, mission_reason="two_stage_backup")
        if not safe:
            continue
        backup_plan.append((src.id, send, 0, world.eta(src, staging, send)))
        backup_total += send
    if not backup_plan:
        return []
    backup_etas = [eta for _sid, _s, _ls, eta in backup_plan]
    if max(backup_etas) - min(backup_etas) > ETA_SYNC_WINDOW:
        world.add_debug(
            f"STAGING_ETA_SYNC_REJECTED staging=p{staging.id} target=p{target.id} spread={max(backup_etas) - min(backup_etas):.1f}"
        )
        return []
    stage_horizon = max(1, int(math.ceil(max(backup_etas))) + 1)
    stage_tl = world.simulate_planet_timeline(staging, min(SIM_HORIZON, stage_horizon))
    if stage_tl["fall_turn"] is not None and stage_tl["fall_turn"] <= stage_horizon:
        return []
    final_eta_est = world.eta(staging, target, max(MIN_SEND_SHIPS, safe_on_stage + backup_total))
    final_total_turn = stage_horizon + max(1, int(math.ceil(final_eta_est)))
    _owner_at, projected_defense = target_ships_at_arrival_frame(world, target, final_total_turn)
    projected_defense = max(projected_defense, int(target.ships) + int(target.production) * final_total_turn)
    final_required = _critical_mass_required(projected_defense, FINAL_CAPTURE_CRITICAL_MASS_MULT)
    staging_required = _critical_mass_required(projected_defense, STAGING_CRITICAL_MASS_MULT)
    staged_total = safe_on_stage + backup_total
    if safe_on_stage < staging_required or staged_total < final_required:
        return []
    world.add_debug(
        f"STAGING_VALIDATION_PASSED staging=p{staging.id} target=p{target.id} "
        f"safe={safe_on_stage} backup={backup_total} staging_required={staging_required} final_required={final_required}"
    )

    final_launch_step = world.step + stage_horizon
    final_eta = world.eta(staging, target, final_required)
    final_plan = [(staging.id, final_required, final_launch_step, final_eta)]
    final_arrival_turn = stage_horizon + max(1, int(math.ceil(final_eta)))
    owner_after, _ships_after = world.projected_state(
        target.id,
        final_arrival_turn,
        extra_arrivals=((final_arrival_turn, world.player, final_required),),
    )
    if owner_after != world.player:
        return []
    if not world.can_hold_after_capture(target, final_arrival_turn, final_required):
        return []
    if world.source_is_safe_for(staging, target, "SYNC_ATTACK", min(final_required, max(MIN_SEND_SHIPS, safe_on_stage)), mission_reason="two_stage_final_probe")[0] is False:
        return []
    world.add_debug(
        f"FINAL_CAPTURE_VALIDATION_PASSED staging=p{staging.id} target=p{target.id} "
        f"projected_defense={projected_defense} final_required={final_required} launch_step={final_launch_step}"
    )

    stage_key = f"{staging.id}-{target.id}-{world.step}"
    backup_prop = MissionProposal(
        kind="REINFORCE_CAPTURE",
        target_id=staging.id,
        priority=222.0 + int(target.production) * 18.0 + _conversion_momentum_bonus(world, target),
        required_ships=backup_total,
        planned_sources=backup_plan,
        eta_min=min(backup_etas),
        eta_max=max(backup_etas),
        reason=f"two_stage_backup stage_key={stage_key} target=p{target.id} staging=p{staging.id}",
        priority_tier="IMPORTANT",
    )
    final_prop = MissionProposal(
        kind="SYNC_ATTACK",
        target_id=target.id,
        priority=232.0 + int(target.production) * 20.0 + _conversion_momentum_bonus(world, target),
        required_ships=final_required,
        planned_sources=final_plan,
        eta_min=final_eta,
        eta_max=final_eta,
        reason=f"two_stage_final stage_key={stage_key} staging=p{staging.id}",
        priority_tier="IMPORTANT",
    )
    world.add_debug(
        f"STAGING_FORCE_RESERVED staging=p{staging.id} target=p{target.id} incoming={backup_total} final={final_required}"
    )
    world.add_debug(
        f"MULTI_STAGE_ASSAULT_LOCKED staging=p{staging.id} target=p{target.id} stage_key={stage_key}"
    )
    return [backup_prop, final_prop]


def update_territory_conversion_history(world):
    if not _prev_owners:
        return
    gained = lost = enemy_gained = enemy_lost = 0
    for p in world.normal_planets:
        prev = _prev_owners.get(p.id)
        if prev is None or prev == p.owner:
            continue
        if prev != world.player and p.owner == world.player:
            gained += 1
        elif prev == world.player and p.owner != world.player:
            lost += 1
        if prev in (-1, world.player) and p.owner not in (-1, world.player):
            enemy_gained += 1
        elif prev not in (-1, world.player) and p.owner in (-1, world.player):
            enemy_lost += 1
    hist = _territory_conversion_history.setdefault(world.player, [])
    if not hist or hist[-1].get("step") != world.step:
        hist.append({
            "step": world.step,
            "gained": gained,
            "lost": lost,
            "enemy_gained": enemy_gained,
            "enemy_lost": enemy_lost,
        })
    _territory_conversion_history[world.player] = [
        rec for rec in hist if world.step - rec.get("step", world.step) <= 25
    ][-30:]


def territory_conversion_score(world):
    hist = [
        rec for rec in _territory_conversion_history.get(world.player, [])
        if world.step - rec.get("step", world.step) <= 25
    ]
    gained = sum(int(rec.get("gained", 0)) for rec in hist)
    lost = sum(int(rec.get("lost", 0)) for rec in hist)
    enemy_gained = sum(int(rec.get("enemy_gained", 0)) for rec in hist)
    enemy_lost = sum(int(rec.get("enemy_lost", 0)) for rec in hist)
    score = gained - lost
    pressure = {
        "score": score,
        "gained": gained,
        "lost": lost,
        "enemy_net": enemy_gained - enemy_lost,
        "low": score <= LOW_TERRITORY_PRESSURE_THRESHOLD,
    }
    if pressure["low"] and world.step > 20:
        world.add_debug(
            f"LOW_TERRITORY_CONVERSION_PRESSURE score={score} gained={gained} "
            f"lost={lost} enemy_net={pressure['enemy_net']}"
        )
    return pressure


def configure_aggressiveness_controller(world, control_ratio, conversion_pressure):
    enemy_prod_values = list(world.enemy_prod_by_owner.values())
    avg_enemy_prod = sum(enemy_prod_values) / max(1, len(enemy_prod_values))
    enemy_growing_faster = conversion_pressure.get("enemy_net", 0) > conversion_pressure.get("score", 0)
    neutral_high_while_behind = (
        len(world.neutral_planets) >= max(3, len(world.normal_planets) * 0.20)
        and world.my_prod < max(1, world.enemy_prod)
    )
    if control_ratio >= PHASE_COLLAPSE_MIN or world.features.get("final"):
        mode = "COLLAPSE"
    elif (
        world.my_prod < avg_enemy_prod
        or enemy_growing_faster
        or neutral_high_while_behind
        or conversion_pressure.get("low")
    ):
        mode = "AGGRESSIVE"
    elif world.features.get("incoming_threat_count", 0) >= 2 and world.my_prod >= world.enemy_prod:
        mode = "DEFENSIVE"
    else:
        mode = "BALANCED"
    world.aggressiveness_mode = mode
    if mode == "AGGRESSIVE":
        world.add_debug("AGGRESSIVENESS_MODE_AGGRESSIVE")
        world.add_debug("EARLY_CAPTURE_PRIORITY_ACTIVE" if world.step < 100 else "TACTICAL_CAPTURE_PRIORITY_ACTIVE")
    return mode


def _reinforcement_loop_active(world):
    recent = [
        rec for rec in _recent_launch_history.get(world.player, [])
        if world.step - rec.get("step", world.step) <= 12
    ]
    if len(recent) < 3:
        return False
    own_moves = sum(1 for rec in recent if rec.get("target_owned"))
    offensive = sum(1 for rec in recent if rec.get("offensive"))
    no_recent_capture = world.step - _last_capture_step.get(world.player, -999) >= TACTICAL_STALL_TURNS
    active = no_recent_capture and own_moves >= max(3, offensive * 2)
    if active:
        world.add_debug("REINFORCEMENT_LOOP_SUPPRESSED")
    return active


def _cancel_flexible_planning_for_interrupt(world, reason):
    cancelled = world.mission_ledger.cancel_flexible(reason)
    if cancelled:
        world.add_debug("MISSION_INTERRUPTED_FOR_TACTICAL_CAPTURE")
    rec = _staging_controller_memory.setdefault(
        world.player,
        {"active_until": -999, "release_until": -999, "started_step": -999},
    )
    if rec.get("active_until", -999) >= world.step:
        rec["active_until"] = world.step - 1
        rec["release_until"] = max(int(rec.get("release_until", -999)), world.step + STAGING_RELEASE_STEPS)
        world.add_debug("PLANNING_INTERRUPTED_FOR_FRONTLINE_PRESSURE")
    return cancelled


def _build_immediate_capture_prop(world, states, chain_plan, target, reason, priority):
    if target is None or target.owner == world.player or world.is_comet(target):
        return None
    prop = _main35_make_capture_prop(
        world,
        states,
        target,
        reason,
        priority,
        max_sources=6 if getattr(world, "aggressiveness_mode", "BALANCED") == "AGGRESSIVE" else 5,
        source_radius=72.0,
        hold_margin=1 if target.owner == -1 else max(5, int(target.production) * 2),
        require_hold=target.owner != -1 and getattr(world, "aggressiveness_mode", "BALANCED") != "AGGRESSIVE",
    )
    if prop is None:
        sources = sorted(
            [
                p for p in world.my_planets
                if p.id in states
                and not states[p.id].threatened
                and states[p.id].safe_surplus >= MIN_SEND_SHIPS
                and dp(p, target) <= 72.0
            ],
            key=lambda p: (dp(p, target), -states[p.id].safe_surplus),
        )[:6]
        mission_type = "CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK"
        prop = build_capture_plan(world, target, mission_type, sources, max_sources=len(sources), eta_spread_limit=6.0)
        if prop is not None:
            prop.priority = priority
            prop.reason = reason
    if prop is None:
        return None
    prop.priority += _conversion_momentum_bonus(world, target, emit=True)
    prop.priority_tier = "IMPORTANT"
    prop = _synchronize_offensive_proposal_timing(world, prop)
    return prop if _proposal_passes_capture_constraints(world, prop) else None


def _conversion_momentum_bonus(world, target, emit=False):
    if target is None:
        return 0.0
    followups = [
        p for p in world.normal_planets
        if p.id != target.id
        and p.owner != world.player
        and not world.is_comet(p)
        and dp(target, p) <= 38.0
        and (
            int(p.production) >= 2
            or is_idle(p)
            or _planet_role(p) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
        )
    ]
    if not followups:
        return 0.0
    bonus = min(3, len(followups)) * 55.0
    if emit:
        world.add_debug(
            f"CONVERSION_MOMENTUM_CHAIN target=p{target.id} followups={[p.id for p in followups[:3]]} bonus={bonus:.1f}"
        )
    return bonus


def adaptive_constant_tuner(world, control_ratio, conversion_pressure, deadline):
    time_left = max(0.0, deadline - time.perf_counter())
    behind_prod = world.my_prod < max(1, world.enemy_prod)
    under_attack = any(
        int(p.production) >= HUB_SECURITY_PROD_THRESHOLD and world.real_incoming_threat(p)["deficit"] > 0
        for p in world.my_planets
    )
    many_neutrals = len(world.neutral_planets) >= max(3, len(world.normal_planets) * 0.18)
    no_neutrals = not world.neutral_planets
    mode = getattr(world, "aggressiveness_mode", "BALANCED")

    hold_margin_mult = 0.78 if behind_prod or mode == "AGGRESSIVE" else 1.0
    if under_attack and not behind_prod:
        hold_margin_mult = max(hold_margin_mult, 1.18)
    fleet_ratio_cap = MIDGAME_FLEET_HARD
    if behind_prod:
        fleet_ratio_cap += 0.08
    elif under_attack:
        fleet_ratio_cap -= 0.05

    if time_left < 0.08:
        beam_depth, beam_width = 1, 3
    elif world.step < 100:
        beam_depth, beam_width = 2, 5
    else:
        beam_depth, beam_width = 3, 8
    if time_left < 0.08:
        world.add_debug("BEAM_DEPTH_REDUCED_FOR_TIME")
        world.add_debug("BEAM_WIDTH_REDUCED_FOR_TIME")

    cfg = {
        "fleet_ratio_cap": fleet_ratio_cap,
        "hold_margin_mult": hold_margin_mult,
        "expansion_aggression": 1.35 if (behind_prod or many_neutrals or conversion_pressure.get("low")) else 1.0,
        "beam_depth": beam_depth,
        "beam_width": beam_width,
        "reserve_mult": 1.20 if under_attack else 0.85 if behind_prod else 1.0,
        "enemy_attack_threshold": 0.80 if no_neutrals else 1.0,
        "time_left": time_left,
        "under_attack": under_attack,
        "many_neutrals": many_neutrals,
        "no_neutrals": no_neutrals,
    }
    world.dynamic_config = cfg
    world.add_debug("ADAPTIVE_CONSTANT_TUNER_ACTIVE")
    if hold_margin_mult != 1.0:
        world.add_debug(f"DYNAMIC_HOLD_MARGIN_APPLIED mult={hold_margin_mult:.2f}")
    world.add_debug(f"DYNAMIC_BEAM_WIDTH_APPLIED width={beam_width}")
    if cfg["expansion_aggression"] != 1.0:
        world.add_debug(f"DYNAMIC_AGGRESSION_APPLIED value={cfg['expansion_aggression']:.2f}")
    if world.step % 25 == 0:
        world.add_debug(
            f"CONSTANT_AUDIT_REPORT fleet_ratio_cap={fleet_ratio_cap:.2f} "
            f"hold_margin_mult={hold_margin_mult:.2f} beam_depth={beam_depth} beam_width={beam_width} "
            f"aggression_mode={mode} reserve_mult={cfg['reserve_mult']:.2f} "
            f"expansion_pressure={cfg['expansion_aggression']:.2f}"
        )
    return cfg


def _dynamic_hold_margin(world, target, base):
    mult = getattr(world, "dynamic_config", {}).get("hold_margin_mult", 1.0)
    adjusted = max(1, int(math.ceil(base * mult)))
    if adjusted != base:
        world.add_debug(f"DYNAMIC_HOLD_MARGIN_APPLIED target=p{target.id} base={base} adjusted={adjusted}")
    return adjusted


def build_pressure_map(world):
    """Per-turn planet graph influence map."""
    pressure = {
        p.id: {
            "my_pressure": 0.0,
            "enemy_pressure": 0.0,
            "net_pressure": 0.0,
            "contested": False,
            "enemy_dominant": False,
            "my_dominant": False,
        }
        for p in world.normal_planets
    }
    for source in world.normal_planets:
        if source.owner == -1:
            continue
        strength = int(source.ships) + int(source.production) * 5
        if strength <= 0:
            continue
        for target in world.normal_planets:
            if target.id == source.id:
                continue
            decay = strength / (dp(source, target) + 8.0)
            if source.owner == world.player:
                pressure[target.id]["my_pressure"] += decay
            else:
                pressure[target.id]["enemy_pressure"] += decay
    for target in world.normal_planets:
        entry = pressure[target.id]
        for eta, owner, ships in world.arrivals_by_target.get(target.id, []):
            incoming = int(ships) * 0.35
            if owner == world.player:
                entry["my_pressure"] += incoming
            elif owner != -1:
                entry["enemy_pressure"] += incoming
        entry["net_pressure"] = entry["my_pressure"] - entry["enemy_pressure"]
        total = entry["my_pressure"] + entry["enemy_pressure"]
        entry["contested"] = total > 0 and abs(entry["net_pressure"]) <= max(6.0, total * 0.22)
        entry["enemy_dominant"] = entry["enemy_pressure"] > entry["my_pressure"] * 1.30 + 8.0
        entry["my_dominant"] = entry["my_pressure"] > entry["enemy_pressure"] * 1.25 + 6.0
    world.add_debug(f"PRESSURE_MAP_BUILT planets={len(pressure)}")
    return pressure


def _pressure_entry(world, target):
    return getattr(world, "pressure_map", {}).get(
        target.id,
        {
            "my_pressure": 0.0,
            "enemy_pressure": 0.0,
            "net_pressure": 0.0,
            "contested": False,
            "enemy_dominant": False,
            "my_dominant": False,
        },
    )


def _target_connected_to_pressure(world, target):
    if target is None:
        return False
    if min((dp(p, target) for p in world.my_planets), default=999.0) <= 48.0:
        return True
    if any(
        dp(p, target) <= 60.0
        and (_planet_role(p) in (ROLE_BRIDGE, ROLE_LAUNCHPAD) or is_static_planet(p) or int(p.production) >= 3)
        for p in world.my_planets
    ):
        return True
    if len(_campaign_followup_options(world, target)) >= 1:
        return True
    return False


def pressure_score_bonus(world, prop, states=None, chain_plan=None):
    if prop.kind not in OFFENSIVE_MISSIONS:
        return 0.0
    target = world.planet_by_id.get(prop.target_id)
    if target is None or target.owner == world.player or world.is_comet(target):
        return 0.0
    states = states or {}
    chain_plan = chain_plan or getattr(world, "_active_chain_plan", [])
    entry = _pressure_entry(world, target)
    adj = 0.0
    nearest = min((dp(p, target) for p in world.my_planets), default=999.0)

    if entry["contested"] or (entry["my_pressure"] > 0 and entry["enemy_pressure"] > 0 and nearest <= 72.0):
        adj += 18.0
        world.add_debug(
            f"PRESSURE_TARGET_BONUS target=p{target.id} contested={entry['contested']} net={entry['net_pressure']:.1f}"
        )
    if _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD) or is_bridge_planet(world, target):
        adj += 18.0
        world.add_debug(f"PRESSURE_BRIDGE_TARGET_SELECTED target=p{target.id}")
    if target.owner not in (-1, world.player) and int(target.ships) <= max(22, int(target.production) * 6):
        if entry["my_pressure"] >= entry["enemy_pressure"] * 0.65:
            adj += 22.0
            world.add_debug(
                f"PRESSURE_TARGET_BONUS target=p{target.id} weak_enemy_zone my={entry['my_pressure']:.1f} enemy={entry['enemy_pressure']:.1f}"
            )
    isolated = nearest > 78.0 and not _target_connected_to_pressure(world, target)
    if isolated and entry["enemy_dominant"]:
        adj -= 35.0
        world.add_debug(
            f"PRESSURE_ISOLATED_TARGET_PENALTY target=p{target.id} nearest={nearest:.1f} enemy_pressure={entry['enemy_pressure']:.1f}"
        )
    if nearest > _local_direct_limit(world):
        bridge = _select_bridge_route_target(world, states, target, chain_plan)
        if bridge is None:
            adj -= 28.0
            world.add_debug(f"PRESSURE_ISOLATED_TARGET_PENALTY target=p{target.id} reason=no_bridge_route")
    return max(-50.0, min(50.0, adj))


def hold_continue_check(world, prop):
    target = world.planet_by_id.get(prop.target_id)
    if target is None or prop.kind not in OFFENSIVE_MISSIONS:
        return True, 0.0
    adjusted_sources = _planned_sources_as_arrival_offsets(world, prop.planned_sources)
    total = sum(int(s) for _sid, s, _ls, _eta in adjusted_sources)
    if not adjusted_sources:
        return True, 0.0
    arrival = max(1, int(math.ceil(max(eta for _sid, _s, _ls, eta in adjusted_sources))))
    extra = tuple((max(1, int(math.ceil(eta))), world.player, int(ships)) for _sid, ships, _ls, eta in adjusted_sources)
    owner_after, _ships_after = world.projected_state(target.id, arrival, extra_arrivals=extra)
    if owner_after != world.player:
        world.add_debug(f"HOLD_CONTINUE_FAIL target=p{target.id} reason=no_flip")
        return False, -50.0
    owner_5, _ships_5 = world.projected_state(target.id, min(SIM_HORIZON, arrival + 5), extra_arrivals=extra)
    owner_10, _ships_10 = world.projected_state(target.id, min(SIM_HORIZON, arrival + 10), extra_arrivals=extra)
    hold_5 = owner_5 == world.player
    hold_10 = owner_10 == world.player
    connected = _target_connected_to_pressure(world, target)
    supports_next = (
        len(_campaign_followup_options(world, target)) >= 1
        or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
        or int(target.production) >= 3
        or is_static_planet(target)
    )
    if hold_5 and connected and supports_next:
        world.add_debug(f"HOLD_CONTINUE_PASS target=p{target.id} horizon=5 connected={connected}")
        return True, 12.0 if hold_10 else 4.0
    penalty = 0.0
    if not hold_5:
        penalty -= 25.0
    if not connected:
        penalty -= 15.0
    if not supports_next:
        penalty -= 10.0
    world.add_debug(
        f"HOLD_CONTINUE_FAIL target=p{target.id} hold5={hold_5} hold10={hold_10} connected={connected} supports_next={supports_next}"
    )
    return True, max(-50.0, penalty)


def adversarial_candidate_adjustment(world, prop, states=None, chain_plan=None):
    if prop.kind not in OFFENSIVE_MISSIONS:
        return 0.0, False
    target = world.planet_by_id.get(prop.target_id)
    if target is None or target.owner == world.player or world.is_comet(target):
        return 0.0, True
    states = states or {}
    chain_plan = chain_plan or getattr(world, "_active_chain_plan", [])
    entry = _pressure_entry(world, target)
    adjusted_sources = _planned_sources_as_arrival_offsets(world, prop.planned_sources)
    arrival = max((eta for _sid, _s, _ls, eta in adjusted_sources), default=prop.eta_max)
    nearest_src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if target.owner not in (-1, world.player):
        enemy_eta = min(
            (world.eta(e, target, max(1, int(e.ships) // 2)) for e in world.enemy_planets if e.id != target.id),
            default=999.0,
        )
    else:
        enemy_eta = enemy_earliest_capture_turn(world, target)
    adjustment = 0.0

    if enemy_eta <= arrival + 3.0:
        adjustment -= 18.0
        world.add_debug(f"ENEMY_RESPONSE_PREDICTED target=p{target.id} enemy_eta={enemy_eta:.1f} my_arrival={arrival:.1f}")
        world.add_debug(f"ENEMY_REINFORCEMENT_PENALTY target=p{target.id} penalty=18")

    ok_hold, hold_adj = hold_continue_check(world, prop)
    adjustment += hold_adj

    for src_id, ships, _ls, _eta in prop.planned_sources:
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        role = _planet_role(src)
        launchpad_like = (
            role == ROLE_LAUNCHPAD
            or src.id in _primary_launchpads.get(world.player, {})
            or int(src.production) >= HUB_SECURITY_PROD_THRESHOLD
        )
        if not launchpad_like:
            continue
        reserve = world.reserve_for(src)
        remaining = int(src.ships) - world.committed.get(src.id, 0) - int(ships)
        if remaining < reserve:
            penalty = min(24.0, (reserve - remaining) * 1.2)
            adjustment -= penalty
            world.add_debug(
                f"DRAINED_LAUNCHPAD_PENALTY src=p{src.id} target=p{target.id} remaining={remaining} reserve={reserve}"
            )

    isolated = not _target_connected_to_pressure(world, target)
    enemy_faster = nearest_src is not None and enemy_eta <= arrival + 2.0
    if isolated and entry["enemy_dominant"]:
        adjustment -= 22.0
        world.add_debug(f"PRESSURE_ISOLATED_TARGET_PENALTY target=p{target.id} reason=enemy_dominant")
    if not ok_hold or (isolated and entry["enemy_dominant"] and enemy_faster):
        world.add_debug(
            f"SUICIDE_CAPTURE_REJECTED target=p{target.id} isolated={isolated} enemy_dominant={entry['enemy_dominant']} enemy_eta={enemy_eta:.1f}"
        )
        return -50.0, True

    world.add_debug(f"ADVERSARIAL_FILTER_APPLIED target=p{target.id} adjustment={adjustment:.1f}")
    return max(-50.0, min(50.0, adjustment)), False


def apply_pressure_adversarial_adjustments(world, proposals, states=None, chain_plan=None):
    adjusted = []
    for prop in list(proposals or ()):
        if prop is None:
            continue
        if getattr(prop, "_pressure_adversarial_applied", False):
            adjusted.append(prop)
            continue
        target = world.planet_by_id.get(prop.target_id)
        if target is None:
            continue
        if prop.kind in OFFENSIVE_MISSIONS:
            pressure_adj = pressure_score_bonus(world, prop, states, chain_plan)
            adv_adj, reject = adversarial_candidate_adjustment(world, prop, states, chain_plan)
            total_adj = max(-50.0, min(50.0, pressure_adj + adv_adj))
            prop.priority += total_adj
            prop._pressure_adversarial_applied = True
            if reject:
                continue
            if abs(total_adj) > 0.01:
                world.add_debug(
                    f"PRESSURE_TARGET_BONUS target=p{target.id} total_adjustment={total_adj:.1f} priority={prop.priority:.1f}"
                )
        adjusted.append(prop)
    return tuple(adjusted)


def run_tactical_interrupt_layer(world, states=None, chain_plan=None, enemy_actions=None, conversion_pressure=None, deadline=None):
    states = states or {}
    chain_plan = chain_plan or []
    enemy_actions = enemy_actions or {}
    conversion_pressure = conversion_pressure or {"low": False, "enemy_net": 0, "score": 0}
    deadline = deadline or (time.perf_counter() + 0.02)

    triggers = []
    props = []

    threatened_hubs = [
        p for p in world.my_planets
        if int(p.production) >= HUB_SECURITY_PROD_THRESHOLD
        and world.real_incoming_threat(p)["deficit"] > 0
    ]
    if threatened_hubs:
        triggers.append("high_prod_threat")

    no_recent_capture = world.step - _last_capture_step.get(world.player, -999) >= TACTICAL_STALL_TURNS
    if no_recent_capture:
        triggers.append("capture_stall")

    if conversion_pressure.get("enemy_net", 0) > conversion_pressure.get("score", 0):
        triggers.append("enemy_expanding_faster")

    drained_items = (enemy_actions or {}).get("drained", [])
    if drained_items:
        triggers.append("enemy_drained")

    loop_active = _reinforcement_loop_active(world)
    if loop_active:
        triggers.append("reinforcement_loop")

    # Immediate weak-enemy punishment: nearby, thin, recently drained/launched, ETA <= 25.
    drained_ids = {item["source"].id for item in drained_items}
    for target in sorted(world.enemy_planets, key=lambda p: (int(p.ships), min((dp(m, p) for m in world.my_planets), default=999.0)))[:10]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        src = min(world.my_planets, key=lambda m: dp(m, target), default=None)
        if src is None:
            continue
        rough_need = max(MIN_SEND_SHIPS, normalize_send_amount(world.required_ships_to_capture(target, src)))
        eta = world.eta(src, target, rough_need)
        recently_drained = target.id in drained_ids
        weak = int(target.ships) <= max(18, int(target.production) * 5)
        if nearest <= 62.0 and eta <= 25.0 and (weak or recently_drained):
            prop = _build_immediate_capture_prop(
                world, states, chain_plan, target, "weak_enemy_immediate_punish", 265.0
            )
            if prop is not None:
                triggers.append("weak_enemy")
                world.add_debug(
                    f"WEAK_ENEMY_IMMEDIATE_PUNISH target=p{target.id} ships={int(target.ships)} eta={eta:.1f}"
                )
                props.append(prop)
                break

    # Neutral production race: take valuable nearby neutrals before enemy ETA beats us.
    for target in sorted(world.neutral_planets, key=lambda p: (-int(p.production), int(p.ships)))[:10]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if int(target.production) < 2 and not no_recent_capture:
            continue
        src = min(world.my_planets, key=lambda m: dp(m, target), default=None)
        if src is None:
            continue
        need = max(MIN_SEND_SHIPS, normalize_send_amount(world.required_ships_to_capture(target, src)))
        my_eta = world.eta(src, target, need)
        enemy_eta = enemy_earliest_capture_turn(world, target)
        race_losing = enemy_eta <= my_eta + 3.0
        local_stall = no_recent_capture and my_eta <= 24.0
        if (race_losing or local_stall or conversion_pressure.get("low")) and my_eta <= 28.0:
            prop = _build_immediate_capture_prop(
                world, states, chain_plan, target, "tactical_neutral_conversion", 240.0 + int(target.production) * 12.0
            )
            if prop is not None:
                triggers.append("neutral_race" if race_losing else "forced_conversion")
                props.append(prop)
                break

    # Nearby chain opportunity: convert the first reachable unowned chain node.
    for pid in chain_plan[:8]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        target = world.planet_by_id.get(pid)
        if target is None or target.owner == world.player or world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > 58.0:
            continue
        prop = _build_immediate_capture_prop(
            world, states, chain_plan, target, "tactical_chain_conversion", 232.0
        )
        if prop is not None:
            triggers.append("nearby_chain")
            props.append(prop)
            break

    if not triggers or not props:
        return ()

    world.add_debug(f"TACTICAL_INTERRUPT_ACTIVE triggers={sorted(set(triggers))}")
    world.add_debug("PLANNING_INTERRUPTED_FOR_FRONTLINE_PRESSURE")
    _cancel_flexible_planning_for_interrupt(world, "tactical_capture")

    dedup = {}
    for prop in props:
        key = prop.target_id
        if key not in dedup or prop.priority > dedup[key].priority:
            dedup[key] = prop
    selected = sorted(dedup.values(), key=lambda p: -p.priority)[:2 if getattr(world, "aggressiveness_mode", "BALANCED") in ("AGGRESSIVE", "COLLAPSE") else 1]
    return tuple(selected)


def detect_game_winning_opportunities(world, states, chain_plan, enemy_actions, deadline):
    props = []
    drained_ids = {item["source"].id for item in (enemy_actions or {}).get("drained", [])}
    current_lead = world.my_prod > world.enemy_prod or len(world.my_planets) > max(1, len(world.enemy_planets))
    for target in sorted(world.enemy_planets, key=lambda p: (-int(p.production), int(p.ships)))[:10]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        enemy_last = len(world.enemy_planets) <= 1
        high_prod_weak = int(target.production) >= 3 and int(target.ships) <= max(20, int(target.production) * 6)
        drained = target.id in drained_ids
        gives_lead = (
            world.my_prod + int(target.production) > max(0, world.enemy_prod - int(target.production))
            and not current_lead
        )
        route_block = target.id in set(chain_plan[:10]) or _planet_role(target) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
        if not (enemy_last or high_prod_weak or drained or gives_lead or route_block):
            continue
        prop = _build_immediate_capture_prop(
            world, states, chain_plan, target, "game_winning_opportunity", 310.0 + int(target.production) * 18.0
        )
        if prop is None:
            continue
        prop.priority_tier = "CRITICAL" if enemy_last or gives_lead else "IMPORTANT"
        world.add_debug(
            f"GAME_WINNING_OPPORTUNITY_DETECTED target=p{target.id} enemy_last={enemy_last} "
            f"high_prod_weak={high_prod_weak} drained={drained} gives_lead={gives_lead}"
        )
        props.append(prop)
    return tuple(sorted(props, key=lambda p: -p.priority)[:2])


def build_nuisance_interrupt_props(world, states, enemy_actions, deadline):
    if not (world.step > 340 or world.remaining < 120 or not world.neutral_planets):
        return ()
    props = []
    drained_ids = {item["source"].id for item in (enemy_actions or {}).get("drained", [])}
    targets = [
        p for p in world.enemy_planets
        if not world.is_comet(p)
        and (
            int(p.production) >= 3
            or p.id in drained_ids
            or len(world.enemy_planets) <= 2
            or int(p.ships) <= 12
        )
    ]
    targets.sort(key=lambda p: (-int(p.production), int(p.ships)))
    for target in targets[:8]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if int(target.ships) <= 3 and (world.step > 430 or world.remaining < 60):
            src = min(
                [
                    p for p in world.my_planets
                    if p.id in states and not states[p.id].threatened and states[p.id].safe_surplus >= 5
                ],
                key=lambda p: dp(p, target),
                default=None,
            )
            if src is not None and valid_packet_size("FINISH_ZERO_CAPTURE", 5, world, src, target):
                eta = world.eta(src, target, 5)
                prop = MissionProposal(
                    kind="FINISH_ZERO_CAPTURE",
                    target_id=target.id,
                    priority=260.0,
                    required_ships=5,
                    planned_sources=[(src.id, 5, 0, eta)],
                    eta_min=eta,
                    eta_max=eta,
                    reason="nuisance_endgame_finish_zero",
                    priority_tier="CRITICAL",
                )
                world.add_debug(f"NUISANCE_INTERRUPT_FLEET_SELECTED target=p{target.id} ships=5")
                world.add_debug("VALID_NUISANCE_PACKET_SENT")
                props.append(prop)
                break
        sources = sorted(
            [
                p for p in world.my_planets
                if p.id in states
                and not states[p.id].threatened
                and states[p.id].safe_surplus >= MIN_SEND_SHIPS
                and dp(p, target) <= 70.0
            ],
            key=lambda p: (dp(p, target), -states[p.id].safe_surplus),
        )
        if not sources:
            continue
        prop = _build_immediate_capture_prop(world, states, [], target, "nuisance_interrupt", 205.0)
        if prop is None:
            continue
        total = _proposal_total_ships(prop)
        if total > 20 and len(world.enemy_planets) > 2:
            continue
        world.add_debug(f"NUISANCE_INTERRUPT_FLEET_SELECTED target=p{target.id} ships={total}")
        world.add_debug("VALID_NUISANCE_PACKET_SENT")
        props.append(prop)
        break

    for target in sorted(world.neutral_planets, key=lambda p: enemy_earliest_capture_turn(world, p))[:4]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        enemy_eta = enemy_earliest_capture_turn(world, target)
        if enemy_eta > 12:
            continue
        prop = _build_immediate_capture_prop(world, states, [], target, "nuisance_neutral_interrupt", 198.0)
        if prop is not None:
            world.add_debug(f"NUISANCE_INTERRUPT_FLEET_SELECTED target=p{target.id} enemy_eta={enemy_eta:.1f}")
            world.add_debug("VALID_NUISANCE_PACKET_SENT")
            props.append(prop)
            break
    return tuple(props[:1])


def activate_offense_first_objective(world):
    world.add_debug("OFFENSE_FIRST_OBJECTIVE_ACTIVE")
    if world.enemy_planets:
        world.add_debug("ENEMY_CAPTURE_GOAL_ACTIVE")
    if world.my_prod <= world.enemy_prod or world.my_total_ships <= world.enemy_total_ships:
        world.add_debug("FLEET_ADVANTAGE_GOAL_ACTIVE")
        world.add_debug("BUILD_FLEET_ADVANTAGE_FOR_ATTACK")
    elif world.enemy_planets:
        world.add_debug("ADVANTAGE_TO_ENEMY_CAPTURE_TRANSITION")


def _launchpad_network_hubs(world):
    pads = owned_launchpads(world)
    seen = {p.id for p in pads}
    for hub in command_hubs(world):
        if hub.id not in seen and (
            _planet_role(hub) == ROLE_LAUNCHPAD
            or int(hub.production) >= HUB_SECURITY_PROD_THRESHOLD
            or is_static_planet(hub)
        ):
            pads.append(hub)
            seen.add(hub.id)
    return pads


def build_launchpad_defense_network(world):
    network = {}
    for pad in _launchpad_network_hubs(world):
        guards = sorted(
            [
                p for p in world.my_planets
                if p.id != pad.id
                and dp(p, pad) <= LAUNCHPAD_GUARD_SUPPORT_RADIUS
                and not world.is_comet(p)
            ],
            key=lambda p: (dp(p, pad), -int(p.production), -int(p.ships)),
        )
        if guards:
            network[pad.id] = guards
            world.add_debug(f"LAUNCHPAD_GUARD_NETWORK_ACTIVE launchpad=p{pad.id} guards={[g.id for g in guards[:5]]}")
            for guard in guards[:3]:
                world.add_debug(f"SURROUNDING_PLANET_GUARD_ASSIGNED launchpad=p{pad.id} guard=p{guard.id}")
                world.add_debug(f"FRONT_ATTACKER_CAN_REINFORCE_BACK guard=p{guard.id} launchpad=p{pad.id}")
    return network


def assign_guard_planets_to_launchpad(world):
    return build_launchpad_defense_network(world)


def reinforce_launchpad_from_surroundings(world, states, deadline):
    proposals = []
    network = assign_guard_planets_to_launchpad(world)
    for pad_id, guards in network.items():
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        pad = world.planet_by_id.get(pad_id)
        if pad is None or pad.owner != world.player:
            continue
        threat = world.real_incoming_threat(pad)
        if threat["deficit"] <= 0:
            continue
        world.add_debug(f"INCOMING_ATTACK_DEFENSE_NETWORK_TRIGGERED launchpad=p{pad.id} deficit={threat['deficit']}")
        world.add_debug(f"DEFEND_BEFORE_ESCAPE launchpad=p{pad.id}")
        latest_enemy_eta = max(
            [eta for eta, owner, _ships in world.arrivals_by_target.get(pad.id, []) if owner != world.player]
            or [DEFENSE_ETA_HORIZON]
        )
        need = normalize_send_amount(max(MIN_SEND_SHIPS, threat["deficit"] + 2))
        planned = []
        sent = 0
        for src in sorted(guards, key=lambda g: (world.eta(g, pad, MIN_SEND_SHIPS), dp(g, pad)))[:6]:
            st = states.get(src.id)
            if st is None or st.threatened or st.safe_surplus < MIN_SEND_SHIPS:
                continue
            raw = min(st.safe_surplus, need - sent)
            send = round_down_to_granularity(raw)
            if send < MIN_SEND_SHIPS and st.safe_surplus >= need - sent:
                send = normalize_send_amount(need - sent)
            if send < MIN_SEND_SHIPS or send > st.safe_surplus:
                continue
            eta = world.eta(src, pad, send)
            if eta > latest_enemy_eta + 2:
                continue
            ok, _reason = world.source_is_safe_for(
                src, pad, "DEFEND_HOLD", send, mission_reason="launchpad_guard_network"
            )
            if not ok:
                continue
            planned.append((src.id, send, 0, eta))
            sent += send
            world.add_debug(f"GUARD_PLANET_REINFORCEMENT_SENT src=p{src.id} launchpad=p{pad.id} ships={send}")
            if sent >= need:
                break
        if sent < need or not planned:
            continue
        eta_vals = [eta for _sid, _ships, _a, eta in planned]
        proposals.append(MissionProposal(
            kind="DEFEND_HOLD",
            target_id=pad.id,
            priority=310.0 + threat["deficit"] * 6.0 + int(pad.production) * 14.0,
            required_ships=sent,
            planned_sources=planned,
            eta_min=min(eta_vals),
            eta_max=max(eta_vals),
            reason=f"launchpad_guard_network defend p{pad.id} deficit={threat['deficit']}",
            priority_tier="CRITICAL",
        ))
        world.add_debug(f"LAUNCHPAD_REINFORCED_FROM_GUARDS launchpad=p{pad.id} ships={sent}")
        world.add_debug(f"REINFORCEMENT_FROM_LAUNCHPAD_SENT target=p{pad.id} ships={sent}")
    return tuple(proposals)


def build_front_attacker_flow_props(world, states, chain_plan, deadline):
    props = []
    network = assign_guard_planets_to_launchpad(world)
    for pad_id, guards in network.items():
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        pad = world.planet_by_id.get(pad_id)
        st = states.get(pad_id) if pad is not None else None
        if pad is None or st is None or st.threatened or st.safe_surplus < MIN_SEND_SHIPS:
            continue
        if world.real_incoming_threat(pad)["deficit"] > 0:
            world.add_debug(f"LAUNCHPAD_NOT_LEFT_EMPTY launchpad=p{pad.id}")
            continue
        front_guards = [
            g for g in guards
            if g.id in states
            and not states[g.id].threatened
            and dp(pad, g) <= FRONT_ATTACKER_FLOW_RADIUS
            and (
                world.nearest_enemy_distance(g) < world.nearest_enemy_distance(pad)
                or g.id in set(chain_plan[:16])
                or _planet_role(g) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
            )
        ]
        if not front_guards:
            continue
        target = min(
            front_guards,
            key=lambda g: (world.nearest_enemy_distance(g), -int(g.production), dp(pad, g)),
        )
        reserve = max(st.reserve, LAUNCHPAD_RESERVE if _planet_role(pad) == ROLE_LAUNCHPAD else st.reserve)
        send = round_down_to_granularity(min(st.safe_surplus, max(0, int(pad.ships) - world.committed.get(pad.id, 0) - reserve)))
        if send < MIN_SEND_SHIPS:
            world.add_debug(f"LAUNCHPAD_NOT_LEFT_EMPTY launchpad=p{pad.id}")
            continue
        eta = world.eta(pad, target, send)
        ok, _reason = world.source_is_safe_for(
            pad, target, "REINFORCE_CAPTURE", send, mission_reason="launchpad_to_front_attacker_flow"
        )
        if not ok:
            continue
        props.append(MissionProposal(
            kind="REINFORCE_CAPTURE",
            target_id=target.id,
            priority=125.0 + min(40.0, send * 0.25) + int(target.production) * 4.0,
            required_ships=send,
            planned_sources=[(pad.id, send, 0, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"launchpad_to_front_attacker_flow launchpad=p{pad.id} front=p{target.id}",
            priority_tier="FLEXIBLE",
        ))
        world.add_debug(f"LAUNCHPAD_TO_FRONT_ATTACKER_FLOW launchpad=p{pad.id} front=p{target.id} ships={send}")
        world.add_debug(f"FRONT_ATTACKER_RECEIVED_LAUNCHPAD_FLEET front=p{target.id} ships={send}")
    return tuple(props[:3])


def build_front_attacker_push_props(world, states, chain_plan, deadline):
    props = []
    network = assign_guard_planets_to_launchpad(world)
    guard_ids = {guard.id for guards in network.values() for guard in guards}
    for src in sorted(
        [p for p in world.my_planets if p.id in guard_ids and p.id in states and not states[p.id].threatened],
        key=lambda p: (world.nearest_enemy_distance(p), -states[p.id].safe_surplus),
    )[:8]:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if states[src.id].safe_surplus < MIN_SEND_SHIPS:
            continue
        targets = [
            t for t in world.enemy_planets + world.neutral_planets
            if not world.is_comet(t)
            and dp(src, t) <= FRONT_ATTACKER_FLOW_RADIUS
            and (
                t.owner not in (-1, world.player)
                or t.id in set(chain_plan[:18])
                or _planet_role(t) in (ROLE_BRIDGE, ROLE_LAUNCHPAD)
                or int(t.production) >= 2
                or _small_radius_target_allowed(world, t, src, chain_plan)
            )
        ]
        if not targets:
            continue
        targets.sort(
            key=lambda t: (
                0 if t.owner not in (-1, world.player) else 1,
                dp(src, t),
                -int(t.production),
                int(t.ships),
            )
        )
        for target in targets[:4]:
            prop = _main35_make_capture_prop(
                world,
                states,
                target,
                "front_attacker_push",
                165.0 + (45.0 if target.owner not in (-1, world.player) else 0.0) + int(target.production) * 9.0,
                mission_kind="CAPTURE_NEUTRAL" if target.owner == -1 else "SYNC_ATTACK",
                max_sources=2,
                source_radius=36.0,
                hold_margin=2 if target.owner == -1 else max(8, int(target.production) * 3),
                require_hold=target.owner != -1,
            )
            if prop is None:
                continue
            props.append(prop)
            world.add_debug(f"FRONT_ATTACKER_PUSH_ACTIVE src=p{src.id} target=p{target.id}")
            break
    return tuple(props[:4])


def build_counterattack_after_defense_props(world, states, enemy_actions, deadline):
    props = []
    drained = list((enemy_actions or {}).get("drained", []))
    for defended in world.my_planets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if world.real_incoming_threat(defended)["enemy"] <= 0:
            continue
        candidates = []
        for item in drained:
            src = item.get("source")
            if src is not None and src.owner not in (-1, world.player) and dp(defended, src) <= ATTACK_SOURCE_RADIUS + 22:
                candidates.append((item.get("drop", 0), src))
        if not candidates:
            candidates = [(0, e) for e in world.enemy_planets if dp(defended, e) <= ATTACK_SOURCE_RADIUS]
        candidates.sort(key=lambda item: (-item[0], int(item[1].ships), -int(item[1].production)))
        for _drop, target in candidates[:3]:
            prop = _main35_make_capture_prop(
                world,
                states,
                target,
                "counterattack_after_defense",
                215.0 + int(target.production) * 10.0,
                mission_kind="SYNC_ATTACK",
                max_sources=4,
                source_radius=ATTACK_SOURCE_RADIUS + 22,
                hold_margin=max(8, int(target.production) * 3),
                require_hold=True,
            )
            if prop is None:
                continue
            props.append(prop)
            world.add_debug(f"COUNTERATTACK_AFTER_DEFENSE defended=p{defended.id} target=p{target.id}")
            world.add_debug(f"ATTACKER_SOURCE_TARGETED_AFTER_HOLD source=p{target.id}")
            break
    return tuple(props[:2])


def _proposal_arbiter_score(world, prop):
    target = world.planet_by_id.get(prop.target_id)
    if target is None:
        return -1e18
    score = float(prop.priority)
    score += _component_move_sort_score(world, prop) * 0.25
    if prop.kind in OFFENSIVE_MISSIONS and target.owner != world.player:
        score += 95.0
        score += int(target.production) * 45.0
        score += _conversion_momentum_bonus(world, target, emit=False)
        if target.owner not in (-1, world.player):
            score += 80.0
        if "small_start_escape" in (prop.reason or ""):
            score += 900.0
        if "game_winning" in (prop.reason or ""):
            score += 1000.0
    if prop.kind in REINFORCEMENT_MISSIONS and target.owner == world.player:
        threat = world.real_incoming_threat(target)["deficit"]
        score += threat * 18.0
        if "launchpad_guard_network" in (prop.reason or ""):
            score += 190.0
        if "launchpad_to_front_attacker_flow" in (prop.reason or ""):
            score += 45.0
        if threat <= 0:
            score -= 140.0
            if "launchpad_to_front_attacker_flow" not in (prop.reason or ""):
                score -= 60.0
    if prop.kind == "DOOMED_EVACUATION":
        score -= 180.0
    return score


def _proposal_front_axis(world, prop):
    target = world.planet_by_id.get(prop.target_id)
    if target is None:
        return "?"
    return _axis_bucket_from(_owned_centroid(world), target)


def _proposal_is_too_isolated_for_bundle(world, prop):
    target = world.planet_by_id.get(prop.target_id)
    if target is None or world.is_comet(target):
        return True
    nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
    if nearest <= MULTI_AXIS_FRONT_RADIUS:
        return False
    if target.id in set(getattr(world, "_active_chain_plan", [])[:16]):
        return False
    if len(_campaign_followup_options(world, target)) >= 1:
        return False
    if target.owner not in (-1, world.player) and is_local_enemy_opportunity(world, target):
        return False
    return True


def _bundle_post_fleet_ratio(world, missions):
    launched = sum(
        _proposal_total_ships(prop)
        for prop in missions
        if prop.kind in OFFENSIVE_MISSIONS
    )
    current_fleet = sum(int(f.ships) for f in world.my_fleets)
    planet_ships = sum(int(p.ships) for p in world.my_planets)
    return (current_fleet + launched) / max(1, current_fleet + planet_ships)


def validate_attack_bundle(world, missions, states=None):
    missions = tuple(missions or ())
    if len(missions) < SAME_STEP_BUNDLE_MIN:
        return False
    caps = _component_source_caps(world, states or build_planet_states(world))
    if not _combo_is_compatible(missions, caps, world):
        world.add_debug("BUNDLE_REJECT_SOURCE_CONFLICT")
        return False
    used = {}
    for prop in missions:
        target = world.planet_by_id.get(prop.target_id)
        if target is None or world.is_comet(target):
            return False
        if prop.kind in OFFENSIVE_MISSIONS:
            if _proposal_is_too_isolated_for_bundle(world, prop):
                world.add_debug(f"BUNDLE_REJECT_ISOLATED target=p{prop.target_id}")
                return False
            if not _proposal_passes_capture_constraints(world, prop):
                return False
        for src_id, ships, _launch_step, _eta in prop.planned_sources:
            src = world.planet_by_id.get(src_id)
            if src is None or not valid_packet_size(prop.kind, ships, world, src, target):
                return False
            if _is_two_stage_final(prop):
                continue
            ok, _reason = world.valid_fleet_launch(
                src,
                target,
                ships,
                prop.kind,
                planned_sources=_planned_sources_as_arrival_offsets(world, prop.planned_sources),
                mission_reason=prop.reason,
                validate_aim=False,
            )
            if not ok:
                world.add_debug("BUNDLE_REJECT_UNSAFE_AFTER_COMBO")
                return False
            used[src_id] = used.get(src_id, 0) + int(ships)
            if used[src_id] > caps.get(src_id, 0):
                world.add_debug("BUNDLE_REJECT_SOURCE_CONFLICT")
                return False
            remaining_cap = caps.get(src_id, 0) - used[src_id]
            if remaining_cap < 0:
                world.add_debug("BUNDLE_REJECT_UNSAFE_AFTER_COMBO")
                return False
    ratio = _bundle_post_fleet_ratio(world, missions)
    cap = getattr(world, "dynamic_config", {}).get("fleet_ratio_cap", FLEET_RATIO_HARD)
    if ratio > cap and not any("game_winning" in (p.reason or "") for p in missions):
        world.add_debug(f"BUNDLE_REJECT_UNSAFE_AFTER_COMBO ratio={ratio:.2f} cap={cap:.2f}")
        return False
    world.add_debug(
        f"ATTACK_BUNDLE_VALIDATED n={len(missions)} "
        f"targets={[p.target_id for p in missions]} ratio={ratio:.2f}"
    )
    return True


def build_same_step_attack_bundle(world, states=None, proposal_groups=None, deadline=None):
    world.add_debug("SAME_STEP_MULTI_ATTACK_ACTIVE")
    deadline = deadline if deadline is not None else time.perf_counter() + 0.02
    states = states or build_planet_states(world)
    proposals = []
    for group in proposal_groups or ():
        proposals.extend(list(group or ()))
    proposals = [p for p in proposals if p is not None]
    proposals = list(apply_pressure_adversarial_adjustments(
        world,
        proposals,
        states,
        getattr(world, "_active_chain_plan", []),
    ))
    if len(proposals) < SAME_STEP_BUNDLE_MIN:
        return ()

    candidates = []
    for prop in proposals:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        target = world.planet_by_id.get(prop.target_id)
        if target is None or world.is_comet(target):
            continue
        if prop.kind not in OFFENSIVE_MISSIONS | {"REINFORCE_CAPTURE"}:
            continue
        if prop.kind in REINFORCEMENT_MISSIONS and prop.priority_tier == "CRITICAL":
            continue
        if prop.kind in OFFENSIVE_MISSIONS and not _proposal_passes_capture_constraints(world, prop):
            continue
        if _proposal_is_too_isolated_for_bundle(world, prop):
            continue
        score = _proposal_arbiter_score(world, prop)
        axis = _proposal_front_axis(world, prop)
        reason = prop.reason or ""
        if "parallel_opening_sweep" in reason:
            score += 240.0
        if "early_nearest_sweep" in reason:
            score += 180.0
        if "multi_axis_front" in reason:
            score += 130.0
        if "same_frame" in reason or "coordinated" in reason:
            score += 90.0
        candidates.append((score, axis, prop))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2].target_id))
    world.add_debug(f"ATTACK_BUNDLE_CANDIDATES_BUILT n={len(candidates)}")
    if len(candidates) < SAME_STEP_BUNDLE_MIN:
        return ()

    bundle = []
    used_axes = set()
    caps = _component_source_caps(world, states)
    for _score, axis, prop in candidates:
        if len(bundle) >= SAME_STEP_BUNDLE_MAX or time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        tentative = tuple(bundle + [prop])
        if not _combo_is_compatible(tentative, caps, world):
            world.add_debug("BUNDLE_REJECT_SOURCE_CONFLICT")
            continue
        # Prefer multiple fronts first; after two axes are represented, allow
        # extra local captures from the same axis for opening bursts.
        if axis in used_axes and len(used_axes) < 2 and not any(
            ("early_nearest_sweep" in (p.reason or "") or "parallel_opening_sweep" in (p.reason or ""))
            for p in tentative
        ):
            continue
        if len(tentative) < SAME_STEP_BUNDLE_MIN or validate_attack_bundle(world, tentative, states):
            bundle.append(prop)
            used_axes.add(axis)
    if len(bundle) < SAME_STEP_BUNDLE_MIN:
        return ()
    if not validate_attack_bundle(world, bundle, states):
        return ()
    reasons = " ".join((p.reason or "") for p in bundle)
    axes = [_proposal_front_axis(world, p) for p in bundle]
    if any(("early_nearest_sweep" in (p.reason or "") or "parallel_opening_sweep" in (p.reason or "")) for p in bundle):
        world.add_debug(f"SAME_STEP_OPENING_BURST n={len(bundle)}")
        world.add_debug(f"OPENING_BURST_MULTI_CAPTURE n={len(bundle)}")
        world.add_debug(f"OPENING_BUNDLE_SELECTED n={len(bundle)}")
    if len(set(axes)) >= 2:
        world.add_debug(f"MULTI_AXIS_FRONT_CREATED axes={sorted(set(axes))}")
    if any(
        "multi_axis_front" in (p.reason or "")
        or "early_nearest_sweep" in (p.reason or "")
        or "parallel_opening_sweep" in (p.reason or "")
        for p in bundle
    ):
        world.add_debug("SAME_STEP_BRANCH_A_ATTACK")
        world.add_debug("SAME_STEP_BRANCH_B_ATTACK")
    if "ROTATING" in reasons or "x_shape" in reasons or any(not is_static_planet(world.planet_by_id.get(src_id, world.my_planets[0])) for p in bundle for src_id, _s, _ls, _e in p.planned_sources if world.my_planets):
        world.add_debug("SAME_STEP_X_SHAPE_ATTACKS")
    for prop in bundle:
        if "same_step_bundle" not in (prop.reason or ""):
            prop.reason = f"{prop.reason} same_step_bundle"
    return tuple(bundle)


def commit_attack_bundle(world, missions, moves, states=None, deadline=None):
    missions = tuple(missions or ())
    if not validate_attack_bundle(world, missions, states):
        return False
    committed = 0
    for prop in missions:
        if deadline is not None and time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if _commit_proposal(world, prop, moves):
            committed += 1
            if prop.kind in OFFENSIVE_MISSIONS:
                _last_capture_step[world.player] = world.step
        else:
            world.add_debug(f"BEAM_COMMIT_REJECTED {prop.kind}->p{prop.target_id}")
    if committed >= SAME_STEP_BUNDLE_MIN:
        world.add_debug(
            f"MULTI_FRONT_ATTACK_COMMITTED n={committed} targets={[p.target_id for p in missions[:committed]]}"
        )
        if any(("early_nearest_sweep" in (p.reason or "") or "parallel_opening_sweep" in (p.reason or "")) for p in missions[:committed]):
            world.add_debug(f"SAME_STEP_OPENING_ATTACKS_COMMITTED n={committed}")
            world.add_debug(f"OPENING_MULTI_CAPTURE_COMMITTED n={committed}")
            if world.step < 10:
                world.add_debug("OPENING_ATTACK_STARTED_BEFORE_STEP_10")
        return True
    return committed > 0


def priority_arbiter(world, states, proposal_groups, deadline):
    world.add_debug("PRIORITY_ARBITER_ACTIVE")
    proposals = []
    for group in proposal_groups:
        proposals.extend(list(group or ()))
    proposals = [p for p in proposals if p is not None]
    proposals = list(apply_pressure_adversarial_adjustments(
        world,
        proposals,
        states,
        getattr(world, "_active_chain_plan", []),
    ))
    if not proposals:
        return ()

    for prop in proposals:
        target = world.planet_by_id.get(prop.target_id)
        if target is None:
            continue
        if prop.kind in REINFORCEMENT_MISSIONS and target.owner == world.player:
            threat = world.real_incoming_threat(target)["deficit"]
            is_key_launchpad = (
                int(target.production) >= HUB_SECURITY_PROD_THRESHOLD
                or target.id in _primary_launchpads.get(world.player, {})
                or _planet_role(target) == ROLE_LAUNCHPAD
            )
            if threat > 0 and is_key_launchpad:
                world.add_debug(f"EMERGENCY_DEFENSE_SELECTED target=p{target.id} deficit={threat}")
                return (prop,)

    game_winning = [p for p in proposals if "game_winning" in (p.reason or "")]
    if game_winning:
        best = max(game_winning, key=lambda p: _proposal_arbiter_score(world, p))
        world.add_debug(f"GAME_WINNING_CAPTURE_OVERRIDE target=p{best.target_id}")
        world.add_debug("GAME_WINNING_OPPORTUNITY_SELECTED")
        if any(p.kind in REINFORCEMENT_MISSIONS for p in proposals):
            world.add_debug("MINOR_DEFENSE_DEFERRED_FOR_CAPTURE")
        return (best,)

    small_escape = [p for p in proposals if "small_start_escape" in (p.reason or "")]
    if small_escape:
        best = max(small_escape, key=lambda p: _proposal_arbiter_score(world, p))
        world.add_debug(f"SMALL_START_ESCAPE_ACTIVE target=p{best.target_id}")
        return (best,)

    scored = sorted(proposals, key=lambda p: -_proposal_arbiter_score(world, p))
    chosen = []
    caps = _component_source_caps(world, states)
    opening_burst_available = (
        world.step <= EARLY_NEAREST_SWEEP_STEP_MAX
        and any(
            "early_nearest_sweep" in (p.reason or "") or "parallel_opening_sweep" in (p.reason or "")
            for p in scored
        )
    )
    multi_axis_available = any("multi_axis_front" in (p.reason or "") for p in scored)
    chosen_limit = 2 if getattr(world, "aggressiveness_mode", "BALANCED") in ("AGGRESSIVE", "COLLAPSE") else 1
    if opening_burst_available:
        chosen_limit = max(chosen_limit, EARLY_NEAREST_SWEEP_BURST)
        world.add_debug(f"SEARCH_LIMIT_RELAXED_FOR_OPENING limit={chosen_limit}")
    elif multi_axis_available:
        chosen_limit = max(chosen_limit, 2)
    for prop in scored:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if prop.kind in OFFENSIVE_MISSIONS:
            low_defense = False
            for q in scored[:4]:
                q_target = world.planet_by_id.get(q.target_id)
                if q.kind in REINFORCEMENT_MISSIONS and q_target is not None and world.real_incoming_threat(q_target)["deficit"] <= 0:
                    low_defense = True
                    break
            if low_defense:
                world.add_debug(f"OPPORTUNISTIC_CAPTURE_OVERRIDES_LOW_DEFENSE target=p{prop.target_id}")
        tentative = tuple(chosen + [prop])
        if _combo_is_compatible(tentative, caps, world):
            chosen.append(prop)
        if len(chosen) >= chosen_limit:
            break
    return tuple(chosen)


def fast_heuristic_fallback(world, states, chain_plan, enemy_actions, deadline):
    defenses = _build_defense_component_proposals(world, states, deadline)
    if defenses:
        world.add_debug("FAST_HEURISTIC_FALLBACK_USED")
        return (max(defenses, key=lambda p: p.priority),)
    candidates = []
    for target in list(world.neutral_planets) + list(world.enemy_planets):
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        if world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > 64.0:
            continue
        if target.owner not in (-1, world.player) and int(target.ships) > max(20, int(target.production) * 6):
            continue
        candidates.append((nearest - int(target.production) * 6 + int(target.ships) * 0.2, target))
    candidates.sort()
    for _score, target in candidates[:6]:
        prop = _build_immediate_capture_prop(world, states, chain_plan, target, "fast_heuristic_fallback", 190.0)
        if prop is not None:
            adjusted = apply_pressure_adversarial_adjustments(world, (prop,), states, chain_plan)
            if not adjusted:
                continue
            prop = adjusted[0]
            world.add_debug("FAST_HEURISTIC_FALLBACK_USED")
            return (prop,)
    return ()


def time_budget_guard(world, states, chain_plan, enemy_actions, deadline, safety_buffer=0.055):
    if deadline - time.perf_counter() > safety_buffer:
        return ()
    world.add_debug("TIME_BUDGET_FAILSAFE_ACTIVE")
    return fast_heuristic_fallback(world, states, chain_plan, enemy_actions, deadline)


def _component_move_sort_score(world, prop):
    target = world.planet_by_id.get(prop.target_id)
    if target is None:
        return -1e9
    nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
    role_bonus = 0.0
    if _planet_role(target) == ROLE_LAUNCHPAD:
        role_bonus += 100.0
    elif _planet_role(target) == ROLE_BRIDGE:
        role_bonus += 55.0
    if is_idle(target):
        role_bonus += 70.0
    if prop.kind in REINFORCEMENT_MISSIONS:
        threat = world.real_incoming_threat(target)["deficit"] if target.owner == world.player else 0
        score = 200.0 + threat * 8.0 - _proposal_total_ships(prop) * 0.25
        if getattr(world, "aggressiveness_mode", "BALANCED") == "AGGRESSIVE" and threat <= 0:
            score -= 70.0
        if getattr(world, "_reinforcement_loop_active", False) and threat <= 0:
            score -= 120.0
        return score
    enemy_bonus = 70.0 if target.owner not in (-1, world.player) else 0.0
    weak_bonus = max(0.0, 35.0 - int(target.ships)) * (1.5 if target.owner not in (-1, world.player) else 0.8)
    momentum_bonus = _conversion_momentum_bonus(world, target) * 0.45
    aggro_bonus = 0.0
    if getattr(world, "aggressiveness_mode", "BALANCED") == "AGGRESSIVE":
        aggro_bonus += 55.0 if target.owner == -1 else 75.0
        if int(target.ships) <= max(18, int(target.production) * 5):
            aggro_bonus += 45.0
    early_bonus = 0.0
    if world.step < 100:
        early_bonus += int(target.production) * 35.0
        early_bonus += max(0.0, 42.0 - int(target.ships)) * 1.2
    return (
        float(prop.priority)
        + int(target.production) * 55.0
        + role_bonus
        + enemy_bonus
        + weak_bonus
        + momentum_bonus
        + aggro_bonus
        + early_bonus
        + max(0.0, 72.0 - nearest) * 2.0
        - _proposal_total_ships(prop) * 0.8
        - max(0.0, prop.eta_max - 18.0) * 4.0
    )


def _build_defense_component_proposals(world, states, deadline):
    proposals = []
    for target in sorted(world.my_planets, key=lambda p: -world.real_incoming_threat(p)["deficit"]):
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        deficit = world.real_incoming_threat(target)["deficit"]
        if deficit <= 0:
            continue
        world.add_debug(f"INCOMING_ATTACK_DEFENSE_NETWORK_TRIGGERED target=p{target.id} deficit={deficit}")
        world.add_debug(f"DEFEND_BEFORE_ESCAPE target=p{target.id}")
        need = normalize_send_amount(max(MIN_SEND_SHIPS, deficit))
        planned = []
        sent = 0
        sources = sorted(
            [
                p for p in world.my_planets
                if p.id != target.id
                and p.id in states
                and not states[p.id].threatened
                and states[p.id].safe_surplus >= MIN_SEND_SHIPS
            ],
            key=lambda p: (world.eta(p, target, min(states[p.id].safe_surplus, need)), dp(p, target)),
        )
        for src in sources[:5]:
            if sent >= need:
                break
            raw = min(states[src.id].safe_surplus, need - sent)
            send = round_down_to_granularity(raw)
            if send < MIN_SEND_SHIPS and states[src.id].safe_surplus >= need - sent:
                send = normalize_send_amount(need - sent)
            if send < MIN_SEND_SHIPS or send > states[src.id].safe_surplus:
                continue
            ok, _reason = world.source_is_safe_for(src, target, "DEFEND_HOLD", send)
            if not ok:
                continue
            planned.append((src.id, send, 0, world.eta(src, target, send)))
            if src.id in _primary_launchpads.get(world.player, {}) or _planet_role(src) == ROLE_LAUNCHPAD:
                world.add_debug(f"REINFORCEMENT_FROM_LAUNCHPAD_SENT src=p{src.id} target=p{target.id} ships={send}")
            else:
                world.add_debug(f"GUARD_PLANET_REINFORCEMENT_SENT src=p{src.id} target=p{target.id} ships={send}")
            sent += send
        if sent >= need and planned:
            proposals.append(MissionProposal(
                kind="DEFEND_HOLD",
                target_id=target.id,
                priority=240.0 + deficit * 5.0,
                required_ships=sent,
                planned_sources=planned,
                eta_min=min(e for _sid, _s, _a, e in planned),
                eta_max=max(e for _sid, _s, _a, e in planned),
                reason=f"beam_defense deficit={deficit}",
                priority_tier="CRITICAL",
            ))
            world.add_debug(f"COUNTERATTACK_AFTER_DEFENSE target=p{target.id}")
    return proposals


def _build_hub_security_reinforce_props(world, states, deadline):
    proposals = []
    hubs = sorted(
        [
            p for p in world.my_planets
            if int(p.production) >= HUB_SECURITY_PROD_THRESHOLD
            and int(p.ships) < _hub_security_garrison(p)
        ],
        key=lambda p: (_hub_security_garrison(p) - int(p.ships), int(p.production)),
        reverse=True,
    )
    for target in hubs:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        floor = _hub_security_garrison(target)
        deficit = max(0, floor - int(target.ships))
        need = normalize_send_amount(deficit)
        planned = []
        sent = 0
        sources = sorted(
            [
                p for p in world.my_planets
                if p.id != target.id
                and p.id in states
                and not states[p.id].threatened
                and states[p.id].safe_surplus >= MIN_SEND_SHIPS
            ],
            key=lambda p: (world.eta(p, target, min(states[p.id].safe_surplus, need)), dp(p, target)),
        )
        for src in sources[:5]:
            if sent >= need:
                break
            raw = min(states[src.id].safe_surplus, need - sent)
            send = round_down_to_granularity(raw)
            if send < MIN_SEND_SHIPS and states[src.id].safe_surplus >= need - sent:
                send = normalize_send_amount(need - sent)
            if send < MIN_SEND_SHIPS or send > states[src.id].safe_surplus:
                continue
            safe, _reason = world.source_is_safe_for(
                src, target, "REINFORCE_CAPTURE", send, mission_reason="hub_security_buffer"
            )
            if not safe:
                continue
            planned.append((src.id, send, 0, world.eta(src, target, send)))
            sent += send
        if sent <= 0 or not planned:
            world.add_debug(
                f"HUB_SECURITY_REINFORCE_UNFUNDED target=p{target.id} "
                f"ships={int(target.ships)} floor={floor} deficit={deficit}"
            )
            continue
        proposals.append(MissionProposal(
            kind="REINFORCE_CAPTURE",
            target_id=target.id,
            priority=260.0 + deficit * 5.0 + int(target.production) * 20.0,
            required_ships=sent,
            planned_sources=planned,
            eta_min=min(e for _sid, _s, _a, e in planned),
            eta_max=max(e for _sid, _s, _a, e in planned),
            reason=f"hub_security_buffer floor={floor} deficit={deficit}",
        ))
        world.add_debug(
            f"HUB_SECURITY_REINFORCE_READY target=p{target.id} prod={int(target.production)} "
            f"ships={int(target.ships)} floor={floor} send={sent}"
        )
    return proposals


def _candidate_targets_for_beam(world, chain_plan, enemy_actions, control_ratio):
    scored = []
    drained_ids = {item["source"].id for item in (enemy_actions or {}).get("drained", [])}
    small_start = _beam_small_start_active(world)
    phase_locked = getattr(world, "_phase_transition_locked", False)
    for target in world.normal_planets:
        if target.owner == world.player or world.is_comet(target):
            continue
        nearest = min((dp(m, target) for m in world.my_planets), default=999.0)
        if nearest > (94.0 if control_ratio >= PHASE_MIDGAME_MAX else 76.0):
            if small_start:
                world.add_debug(f"SMALL_START_FAR_TARGET_REJECTED p{target.id} d={nearest:.1f}")
            continue
        if small_start and not (_small_start_escape_target_value(world, target) or _chain_small_has_value(world, target)):
            if _planet_role(target) == ROLE_STORAGE:
                world.add_debug(f"SMALL_START_SKIP_SMALL_SPAM p{target.id}")
            continue
        # Phase Transition Lock: when we hold a production lead in the late game,
        # skip low-value neutral grabs; only pursue high-production or enemy targets.
        if phase_locked and target.owner == -1:
            is_high_value = int(target.production) >= 4 or target.id in drained_ids
            if not is_high_value:
                world.add_debug(
                    f"PHASE_LOCK_SKIP target=p{target.id} prod={int(target.production)}"
                )
                continue
        if target.owner not in (-1, world.player) and not (
            is_local_enemy_opportunity(world, target)
            or target.id in drained_ids
            or control_ratio >= PHASE_MIDGAME_MAX
            or int(target.production) >= 3
            or _small_radius_target_allowed(world, target, chain_plan=chain_plan)
        ):
            continue
        if target.owner == -1 and _planet_role(target) == ROLE_STORAGE and not (
            _small_radius_target_allowed(world, target, chain_plan=chain_plan)
            or control_ratio >= PHASE_MIDGAME_MAX
        ):
            continue
        small_allowed_bonus = 0.0
        if _planet_role(target) == ROLE_STORAGE and _small_radius_target_allowed(world, target, chain_plan=chain_plan):
            small_allowed_bonus = 70.0 if target.owner not in (-1, world.player) else 35.0
        score = (
            int(target.production) * 70.0
            + max(0.0, 80.0 - nearest) * 2.6
            + (120.0 if target.id in drained_ids else 0.0)
            + (120.0 if target.owner not in (-1, world.player) and control_ratio >= PHASE_MIDGAME_MAX else 0.0)
            + (95.0 if is_idle(target) else 0.0)
            + (90.0 if _planet_role(target) == ROLE_LAUNCHPAD else 45.0 if _planet_role(target) == ROLE_BRIDGE else 0.0)
            + small_allowed_bonus
            + (70.0 if target.id in set(chain_plan[:18]) else 0.0)
            + min(4, len(_campaign_followup_options(world, target))) * 32.0
            - int(target.ships) * 1.1
        )
        if not is_static_planet(target):
            src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
            approach = rotating_target_approach_score(src, target, world)
            if approach > 0:
                score += min(120.0, approach * 3.0)
                world.add_debug(f"ROTATING_APPROACH_TARGET_SELECTED p{target.id} score={approach:.1f}")
            elif approach < -2.0:
                relay = _rotating_planet_moving_toward_target(world, target)
                score -= min(180.0, abs(approach) * 4.0)
                world.add_debug(f"ROTATING_MOVING_AWAY_REJECTED p{target.id} score={approach:.1f}")
                if relay is not None:
                    world.add_debug(f"ROTATING_RELAY_NEXT_TARGET_SELECTED rotating=p{target.id} relay=p{relay.id}")
        scored.append((score, nearest, int(target.ships), target))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [target for _score, _nearest, _ships, target in scored[:BEAM_ROOT_COMPONENT_LIMIT]]


def _turn_for_eta(eta):
    return max(1, int(math.ceil(eta)))


def target_ships_at_arrival_frame(world, target, arrival_turn, extra_arrivals=()):
    owner, ships = world.projected_state(target.id, arrival_turn, extra_arrivals=extra_arrivals)
    return owner, max(0, int(ships))


def _find_single_source_frame_send(world, src, target, states, mission_type, *, desired_turn=None, max_surplus_slack=3):
    st = states.get(src.id)
    if st is None or st.threatened:
        return None
    max_send = round_down_to_granularity(st.safe_surplus)
    if max_send < MIN_SEND_SHIPS:
        return None
    best = None
    for send in range(MIN_SEND_SHIPS, max_send + 1, SEND_GRANULARITY):
        if not valid_packet_size(mission_type, send):
            continue
        eta = world.eta(src, target, send)
        turn = _turn_for_eta(eta)
        if desired_turn is not None and turn != desired_turn:
            continue
        owner_at, ships_at = target_ships_at_arrival_frame(world, target, turn)
        if owner_at == world.player:
            continue
        projected_defense = max(int(ships_at), int(target.ships) + int(target.production) * max(1, turn))
        required = _critical_mass_required(projected_defense)
        if send < required:
            continue
        safe, _reason = world.source_is_safe_for(src, target, mission_type, send, mission_reason="frame_perfect_snipe")
        if not safe:
            continue
        surplus = send - required
        if surplus > max_surplus_slack and send - SEND_GRANULARITY >= required:
            continue
        item = (surplus, eta, send, turn, required)
        if best is None or item < best:
            best = item
            if surplus <= max_surplus_slack:
                break
    if best is None:
        return None
    surplus, eta, send, turn, required = best
    return {
        "src": src,
        "send": int(send),
        "eta": eta,
        "turn": turn,
        "required": int(required),
        "surplus": int(surplus),
        "owner_at": owner_at,
    }


def build_frame_perfect_snipe_props(world, states, chain_plan, deadline):
    props = []
    targets = sorted(
        [
            t for t in world.neutral_planets
            if not world.is_comet(t)
            and min((dp(m, t) for m in world.my_planets), default=999.0) <= 80.0
        ],
        key=lambda t: (
            min((dp(m, t) for m in world.my_planets), default=999.0),
            -int(t.production),
            int(t.ships),
        ),
    )[:14]
    for target in targets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        sources = sorted(
            [p for p in world.my_planets if p.id in states and not states[p.id].threatened],
            key=lambda p: (dp(p, target), -states[p.id].safe_surplus),
        )
        best_plan = None
        for src in sources[:8]:
            plan = _find_single_source_frame_send(world, src, target, states, "HIGH_VALUE_NEUTRAL_RACE")
            if plan is None:
                continue
            if best_plan is None or (plan["surplus"], plan["eta"], plan["send"]) < (best_plan["surplus"], best_plan["eta"], best_plan["send"]):
                best_plan = plan
        if best_plan is None:
            continue
        planned = [(best_plan["src"].id, best_plan["send"], 0, best_plan["eta"])]
        mission_kind = "HIGH_VALUE_NEUTRAL_RACE" if best_plan["owner_at"] not in (-1, world.player) else "CAPTURE_NEUTRAL"
        props.append(MissionProposal(
            kind=mission_kind,
            target_id=target.id,
            priority=215.0 + int(target.production) * 18.0 - best_plan["surplus"] * 8.0,
            required_ships=best_plan["send"],
            planned_sources=planned,
            eta_min=best_plan["eta"],
            eta_max=best_plan["eta"],
            reason=(
                f"frame_perfect_snipe p{target.id} turn={best_plan['turn']} "
                f"required={best_plan['required']} surplus={best_plan['surplus']}"
            ),
        ))
        world.add_debug(
            f"FRAME_PERFECT_SNIPE_READY p{target.id} src=p{best_plan['src'].id} "
            f"turn={best_plan['turn']} send={best_plan['send']} required={best_plan['required']} surplus={best_plan['surplus']}"
        )
    return props


def _source_max_same_turn_send(world, src, target, states, mission_type, arrival_turn):
    st = states.get(src.id)
    if st is None or st.threatened:
        return None
    max_send = round_down_to_granularity(st.safe_surplus)
    if max_send < MIN_SEND_SHIPS:
        return None
    best = None
    for send in range(MIN_SEND_SHIPS, max_send + 1, SEND_GRANULARITY):
        eta = world.eta(src, target, send)
        if _turn_for_eta(eta) != arrival_turn:
            continue
        safe, _reason = world.source_is_safe_for(src, target, mission_type, send, mission_reason="coordinated_same_frame_attack")
        if not safe:
            continue
        best = (src.id, send, 0, eta)
    return best


def _trim_same_turn_plan(world, target, planned, required_total, mission_type, arrival_turn):
    trimmed = list(planned)
    changed = True
    while changed:
        changed = False
        for i in range(len(trimmed) - 1, -1, -1):
            src_id, send, _angle, _eta = trimmed[i]
            if send - SEND_GRANULARITY < MIN_SEND_SHIPS:
                continue
            if sum(s for _sid, s, _a, _e in trimmed) - SEND_GRANULARITY < required_total:
                continue
            src = world.planet_by_id.get(src_id)
            if src is None:
                continue
            lower = send - SEND_GRANULARITY
            if _turn_for_eta(world.eta(src, target, lower)) != arrival_turn:
                continue
            safe, _reason = world.source_is_safe_for(src, target, mission_type, lower, mission_reason="coordinated_same_frame_attack")
            if not safe:
                continue
            trimmed[i] = (src_id, lower, 0, world.eta(src, target, lower))
            changed = True
    return trimmed


def build_coordinated_launch_props(world, states, chain_plan, deadline):
    props = []
    targets = sorted(
        [
            t for t in world.enemy_planets
            if not world.is_comet(t)
            and (
                int(t.production) >= 3
                or _planet_role(t) in (ROLE_LAUNCHPAD, ROLE_BRIDGE)
                or t.id in set(chain_plan[:18])
                or is_local_enemy_opportunity(world, t)
            )
        ],
        key=lambda t: (
            0 if int(t.production) >= 4 else 1,
            min((dp(m, t) for m in world.my_planets), default=999.0),
            int(t.ships),
        ),
    )[:8]
    for target in targets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        sources = [
            p for p in world.my_planets
            if p.id in states
            and not states[p.id].threatened
            and states[p.id].safe_surplus >= MIN_SEND_SHIPS
            and dp(p, target) <= 115.0
        ]
        if len(sources) < 2:
            continue
        nearby_pool = sum(round_down_to_granularity(states[p.id].safe_surplus) for p in sources[:5])
        if nearby_pool > int(target.ships):
            world.add_debug(
                f"MULTI_SOURCE_OVERMATCH_DETECTED p{target.id} pool={nearby_pool} defense={int(target.ships)}"
            )
            world.add_debug(f"WAITING_FOR_SINGLE_SOURCE_REPLACED p{target.id}")
            world.add_debug(f"WAITING_FOR_SINGLE_SOURCE_REPLACED_BY_SYNC_GROUP p{target.id}")
        min_turn = min((_turn_for_eta(world.eta(p, target, MIN_SEND_SHIPS)) for p in sources), default=99)
        best_prop = None
        best_score = -1e18
        for arrival_turn in range(min_turn, min(min_turn + 12, BEAM_SEARCH_DEPTH + 18)):
            owner_at, ships_at = target_ships_at_arrival_frame(world, target, arrival_turn)
            if owner_at == world.player:
                continue
            growth_defense = int(target.ships) + int(target.production) * max(1, arrival_turn)
            projected_defense = max(int(ships_at), growth_defense)
            required_total = _critical_mass_required(projected_defense)
            planned = []
            for src in sorted(sources, key=lambda p: (-states[p.id].safe_surplus, dp(p, target)))[:5]:
                item = _source_max_same_turn_send(world, src, target, states, "SYNC_ATTACK", arrival_turn)
                if item is not None:
                    planned.append(item)
                if sum(s for _sid, s, _a, _e in planned) >= required_total:
                    break
            if sum(s for _sid, s, _a, _e in planned) < required_total or len(planned) < 2:
                continue
            planned = _trim_same_turn_plan(world, target, planned, required_total, "SYNC_ATTACK", arrival_turn)
            total = sum(s for _sid, s, _a, _e in planned)
            if total < required_total:
                continue
            if any(_turn_for_eta(eta) != arrival_turn for _sid, _s, _a, eta in planned):
                continue
            ok_grp, reason = validate_grouped_launch(world, target, planned)
            if not ok_grp:
                world.add_debug(f"COORDINATED_ATTACK_REJECT p{target.id} reason={reason}")
                continue
            score = (
                int(target.production) * 80.0
                + len(planned) * 45.0
                - (total - required_total) * 5.0
                - arrival_turn * 3.0
                + (90.0 if _planet_role(target) == ROLE_LAUNCHPAD else 0.0)
            )
            if score > best_score:
                best_score = score
                best_prop = MissionProposal(
                    kind="SYNC_ATTACK",
                    target_id=target.id,
                    priority=230.0 + score * 0.04,
                    required_ships=total,
                    planned_sources=planned,
                    eta_min=min(e for _sid, _s, _a, e in planned),
                    eta_max=max(e for _sid, _s, _a, e in planned),
                    reason=(
                        f"coordinated_same_frame_attack p{target.id} "
                        f"arrival_turn={arrival_turn} projected_defense={projected_defense} "
                        f"required={required_total} surplus={total - required_total}"
                    ),
                )
        if best_prop is not None:
            world.add_debug(
                f"COORDINATED_ATTACK_READY p{target.id} turn={_turn_for_eta(best_prop.eta_max)} "
                f"sources={len(best_prop.planned_sources)} total={best_prop.required_ships}"
            )
            world.add_debug(f"GROUPED_ONE_SHOT_ATTACK_BUILT p{target.id} sources={len(best_prop.planned_sources)}")
            world.add_debug(f"COORDINATED_ATTACK_RELEASED p{target.id}")
            props.append(best_prop)
    return props


def build_bridge_route_required_props(world, states, chain_plan, deadline):
    """Capture a local bridge first when a valuable target is too far for direct play."""
    if time.perf_counter() > deadline - BEAM_TIME_BUFFER or not world.my_planets:
        return []
    props = []
    far_targets = sorted(
        [
            t for t in world.normal_planets
            if t.owner != world.player
            and not world.is_comet(t)
            and (
                int(t.production) >= 3
                or _planet_role(t) in (ROLE_LAUNCHPAD, ROLE_BRIDGE)
                or (t.owner not in (-1, world.player) and is_local_enemy_opportunity(world, t))
            )
        ],
        key=lambda t: (
            min((dp(m, t) for m in world.my_planets), default=999.0),
            -int(t.production),
            int(t.ships),
        ),
    )[:10]
    seen_bridges = set()
    for final in far_targets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        src = _nearest_direct_source(world, final)
        if src is None or dp(src, final) <= _local_direct_limit(world):
            continue
        bridge = _select_bridge_route_target(world, states, final, chain_plan)
        if bridge is None or bridge.id in seen_bridges:
            continue
        prop = _main35_make_capture_prop(
            world,
            states,
            bridge,
            f"bridge_route_required final=p{final.id}",
            185.0 + int(bridge.production) * 12.0,
            mission_kind="CAPTURE_NEUTRAL" if bridge.owner == -1 else "SYNC_ATTACK",
            max_sources=4,
            source_radius=_local_direct_limit(world) + 8.0,
            hold_margin=2 if bridge.owner == -1 else max(8, int(bridge.production) * 3),
            require_hold=bridge.owner != -1,
        )
        if prop is None:
            continue
        prop.reason = f"{prop.reason} bridge_route final=p{final.id}"
        world.add_debug(f"BRIDGE_ROUTE_REQUIRED final=p{final.id} bridge=p{bridge.id}")
        world.add_debug(f"BRIDGE_ROUTE_SELECTED src=p{src.id} bridge=p{bridge.id} final=p{final.id}")
        props.append(prop)
        seen_bridges.add(bridge.id)
        if len(props) >= 2:
            break
    return props


def build_third_party_intercept_props(world, states, deadline):
    props = []
    for fl in world.enemy_fleets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        target = fleet_target(fl, world.planets, world.ang_vel)
        if (
            target is None
            or world.is_comet(target)
            or target.owner in (-1, world.player, fl.owner)
        ):
            continue
        eta_enemy = travel_turns(dist(fl.x, fl.y, target.x, target.y), max(1, int(fl.ships)))
        collision_turn = _turn_for_eta(eta_enemy)
        intercept_turn = collision_turn + 1
        if intercept_turn > min(world.remaining - 1, 36):
            continue
        owner_after, ships_after = target_ships_at_arrival_frame(world, target, intercept_turn)
        if owner_after == world.player:
            continue
        projected_defense = max(int(ships_after), int(target.ships) + int(target.production) * max(1, intercept_turn))
        required = _critical_mass_required(projected_defense)
        best = None
        for src in sorted(world.my_planets, key=lambda p: (abs(_turn_for_eta(world.eta(p, target, MIN_SEND_SHIPS)) - intercept_turn), dp(p, target)))[:10]:
            plan = _find_single_source_frame_send(
                world, src, target, states, "SYNC_ATTACK",
                desired_turn=intercept_turn,
                max_surplus_slack=6,
            )
            if plan is None or plan["send"] < required:
                continue
            current_guard_need = world.required_ships_to_capture(target, src)
            if plan["send"] < current_guard_need * MIN_WAVE_FRACTION:
                # Existing launch validation requires enough force for the current hostile target too.
                continue
            item = (plan["surplus"], plan["eta"], plan)
            if best is None or item < best:
                best = item
        if best is None:
            continue
        _surplus, _eta, plan = best
        props.append(MissionProposal(
            kind="SYNC_ATTACK",
            target_id=target.id,
            priority=245.0 + int(target.production) * 14.0,
            required_ships=plan["send"],
            planned_sources=[(plan["src"].id, plan["send"], 0, plan["eta"])],
            eta_min=plan["eta"],
            eta_max=plan["eta"],
            reason=(
                f"third_party_intercept p{target.id} enemy={fl.owner} "
                f"collision={collision_turn} intercept={intercept_turn}"
            ),
        ))
        world.add_debug(
            f"THIRD_PARTY_INTERCEPT_READY p{target.id} src=p{plan['src'].id} "
            f"collision={collision_turn} intercept={intercept_turn} send={plan['send']}"
        )
    return props


def _frontline_staging_planet(world, src, chain_plan):
    candidates = [
        p for p in world.my_planets
        if p.id != src.id
        and (
            world.nearest_enemy_distance(p) <= 82.0
            or p.id in set(chain_plan[:12])
            or _planet_role(p) in (ROLE_LAUNCHPAD, ROLE_BRIDGE)
        )
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: (
            0 if world.nearest_enemy_distance(p) <= 82.0 else 1,
            0 if _planet_role(p) == ROLE_LAUNCHPAD else 1,
            dp(src, p),
            -int(p.production),
        )
    )
    return candidates[0]


def build_zero_capital_backline_rally_props(world, states, chain_plan, deadline):
    props = []
    for src in sorted(world.my_planets, key=lambda p: -states.get(p.id, PlanetState(p.id, "", False, 0, 0, 0.0, 0, 0, False, 0.0)).safe_surplus):
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        st = states.get(src.id)
        if st is None or not zero_capital_backline_safe(world, src):
            continue
        send = round_down_to_granularity(st.safe_surplus)
        if send < MIN_SEND_SHIPS:
            continue
        staging = _frontline_staging_planet(world, src, chain_plan)
        if staging is None or staging.id == src.id:
            continue
        if world.eta(src, staging, send) > 16:
            continue
        props.append(MissionProposal(
            kind="DEFEND_HOLD",
            target_id=staging.id,
            priority=150.0 + send * 0.3,
            required_ships=send,
            planned_sources=[(src.id, send, 0, world.eta(src, staging, send))],
            eta_min=world.eta(src, staging, send),
            eta_max=world.eta(src, staging, send),
            reason=f"zero_capital_backline_drain src=p{src.id} staging=p{staging.id}",
        ))
        world.add_debug(
            f"ZERO_CAPITAL_RALLY_READY src=p{src.id} staging=p{staging.id} send={send}"
        )
    return props[:8]


def generate_atomic_component_moves(world, states, chain_plan, enemy_actions, control_ratio, deadline, staging_id=None):
    components = []
    components.extend(_build_defense_component_proposals(world, states, deadline))
    components.extend(reinforce_launchpad_from_surroundings(world, states, deadline))
    components.extend(_build_hub_security_reinforce_props(world, states, deadline))
    staging_active = (
        getattr(world, "aggressiveness_mode", "BALANCED") != "AGGRESSIVE"
        and not getattr(world, "_reinforcement_loop_active", False)
        and not (world.step <= 70 and opening_capture_deficit_active(world))
        and _staging_controller_active(world)
    )
    if staging_active:
        components.extend(_build_staging_controller_props(world, states, chain_plan, deadline))
    components.extend(build_third_party_intercept_props(world, states, deadline))
    components.extend(build_frame_perfect_snipe_props(world, states, chain_plan, deadline))
    components.extend(build_front_attacker_push_props(world, states, chain_plan, deadline))
    components.extend(build_counterattack_after_defense_props(world, states, enemy_actions, deadline))
    components.extend(build_coordinated_launch_props(world, states, chain_plan, deadline))
    components.extend(build_bridge_route_required_props(world, states, chain_plan, deadline))
    components.extend(build_front_attacker_flow_props(world, states, chain_plan, deadline))
    components.extend(build_zero_capital_backline_rally_props(world, states, chain_plan, deadline))

    # Hub Security: if any prod-3+ friendly planet is below minimum garrison,
    # skip all new offensive proposals so ships stay home to reinforce.
    hub_violated, hub_detail = _hub_security_violated(world)
    if hub_violated:
        world.add_debug(
            f"HUB_SECURITY_BLOCK offensive expansion suppressed - {hub_detail}"
        )
        components = [prop for prop in components if prop.kind not in OFFENSIVE_MISSIONS]
        components.sort(key=lambda prop: -_component_move_sort_score(world, prop))
        world.add_debug(f"BEAM_COMPONENT_MOVES_GENERATED n={len(components)} (hub_security_active)")
        return components[:BEAM_ROOT_COMPONENT_LIMIT]

    if staging_active:
        world.add_debug("STAGING_CONTROLLER_BLOCK offensive proposals suppressed")
        components = [prop for prop in components if prop.kind not in OFFENSIVE_MISSIONS]
        components.sort(key=lambda prop: -_component_move_sort_score(world, prop))
        world.add_debug(f"BEAM_COMPONENT_MOVES_GENERATED n={len(components)} (staging_controller_active)")
        return components[:BEAM_ROOT_COMPONENT_LIMIT]
    if getattr(world, "_reinforcement_loop_active", False):
        components = [
            prop for prop in components
            if prop.kind not in REINFORCEMENT_MISSIONS
            or prop.priority_tier == "CRITICAL"
            or "hub_security_buffer" in (prop.reason or "")
        ]

    active_offense = _active_offensive_capture_missions(world)
    if active_offense and not _has_significant_production_lead(world):
        world.add_debug(
            f"FLEET_CONCENTRATION_BLOCK active_offense={[(e.mission_type, e.target_id) for e in active_offense]} "
            f"my_prod={world.my_prod} enemy_prod={world.enemy_prod} lead_mult={SIGNIFICANT_PROD_LEAD_MULT:.2f}"
        )
        components = [prop for prop in components if prop.kind not in OFFENSIVE_MISSIONS]
        components.sort(key=lambda prop: -_component_move_sort_score(world, prop))
        world.add_debug(f"BEAM_COMPONENT_MOVES_GENERATED n={len(components)} (active_capture_concentrating)")
        return components[:BEAM_ROOT_COMPONENT_LIMIT]

    targets = _candidate_targets_for_beam(world, chain_plan, enemy_actions, control_ratio)
    seen = {(p.kind, p.target_id, tuple(src_id for src_id, _s, _a, _e in p.planned_sources)) for p in components}
    for target in targets:
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        hold = 1 if target.owner == -1 and world.step < 80 else (2 if target.owner == -1 else max(8, int(target.production) * 3))
        reason = "beam_atomic_capture"
        if _beam_small_start_active(world):
            reason = "beam_atomic_small_start_escape"
        prop = _main35_make_capture_prop(
            world,
            states,
            target,
            reason,
            150.0,
            max_sources=5,
            source_radius=90.0,
            hold_margin=hold,
            require_hold=target.owner != -1,
        )
        if prop is None and target.owner not in (-1, world.player):
            staged_props = build_two_stage_grouped_assault_props(
                world, target, states, chain_plan, deadline, staging_id=staging_id
            )
            if staged_props:
                for staged_prop in staged_props:
                    key = (
                        staged_prop.kind,
                        staged_prop.target_id,
                        tuple(src_id for src_id, _s, _a, _e in staged_prop.planned_sources),
                    )
                    if key not in seen:
                        seen.add(key)
                        components.append(staged_prop)
                continue
            relay = build_grouped_relay_attack(world, target, states, staging_id=staging_id)
            if relay is not None:
                prop = relay
        if prop is None:
            continue
        prop = _synchronize_offensive_proposal_timing(world, prop)
        if not _proposal_passes_capture_constraints(world, prop):
            continue
        key = (prop.kind, prop.target_id, tuple(src_id for src_id, _s, _a, _e in prop.planned_sources))
        if key in seen:
            continue
        seen.add(key)
        components.append(prop)

    dominance, _surplus = midgame_dominance_attack_props(world, states, chain_plan, deadline, limit=3)
    for prop in dominance:
        prop = _synchronize_offensive_proposal_timing(world, prop)
        if not _proposal_passes_capture_constraints(world, prop):
            continue
        key = (prop.kind, prop.target_id, tuple(src_id for src_id, _s, _a, _e in prop.planned_sources))
        if key not in seen:
            seen.add(key)
            components.append(prop)

    components = [
        _synchronize_offensive_proposal_timing(world, prop)
        for prop in components
    ]
    components = [prop for prop in components if _proposal_passes_capture_constraints(world, prop)]
    components = list(apply_pressure_adversarial_adjustments(world, components, states, chain_plan))
    components.sort(key=lambda prop: -_component_move_sort_score(world, prop))
    world.add_debug(f"BEAM_COMPONENT_MOVES_GENERATED n={len(components)}")
    return components[:BEAM_ROOT_COMPONENT_LIMIT]


def _component_source_caps(world, states):
    caps = {}
    for p in world.my_planets:
        st = states.get(p.id)
        if st is None:
            caps[p.id] = max(0, world.surplus(p))
        else:
            caps[p.id] = max(0, min(st.safe_surplus, int(p.ships) - world.committed.get(p.id, 0) - st.reserve))
        caps[p.id] = max(0, caps[p.id] - world.staging_reserved_ships.get(p.id, 0))
    return caps


def _combo_is_compatible(combo, caps, world=None):
    used = {}
    offensive_targets = set()
    stage_keys = {_two_stage_key(prop) for prop in combo if _is_two_stage_backup(prop)}
    for prop in combo:
        if _is_two_stage_final(prop) and _two_stage_key(prop) not in stage_keys:
            return False
        if prop.kind in OFFENSIVE_MISSIONS:
            if prop.target_id in offensive_targets:
                return False
            offensive_targets.add(prop.target_id)
        for src_id, ships in _proposal_source_totals(prop).items():
            if _is_two_stage_final(prop):
                continue
            used[src_id] = used.get(src_id, 0) + ships
            cap = caps.get(src_id, 0)
            if world is not None:
                src = world.planet_by_id.get(src_id)
                tgt = world.planet_by_id.get(prop.target_id)
                if _early_neutral_reserve_relaxation_allowed(world, src, tgt, prop.reason):
                    cap = max(
                        cap,
                        int(src.ships) - world.committed.get(src.id, 0) - _early_neutral_min_source_reserve(world, src),
                    )
            if used[src_id] > cap:
                return False
    return True


def _active_offensive_limit(world):
    """Dynamic cap on simultaneous offensive missions the beam may commit per turn.

    Returns 1 when production parity is lost or when the midgame state is not
    purely expansionary — forcing the agent to commit fully to one high-value
    capture rather than splitting fleets across two lower-value ones.
    Returns 2 (the default) otherwise.
    """
    if not _has_significant_production_lead(world):
        return 1
    mg_state = getattr(world, "_cached_midgame_state", None)
    if mg_state is not None and mg_state not in (MidgameState.STABLE_EXPAND,):
        return 1
    return 2


def generate_action_combinations(world, states, components, deadline):
    caps = _component_source_caps(world, states)
    offense_limit = _active_offensive_limit(world)
    combos = []
    for prop in components:
        if _combo_is_compatible((prop,), caps, world):
            combos.append((prop,))
    for i, first in enumerate(components[:18]):
        if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
            break
        for second in components[i + 1:24]:
            combo = (first, second)
            # Mission Cap: if the limit is 1, forbid combos where both proposals
            # are offensive so the agent commits fully to a single capture.
            if offense_limit < 2:
                offensive_count = sum(1 for p in combo if p.kind in OFFENSIVE_MISSIONS)
                if offensive_count >= 2:
                    world.add_debug(
                        f"MISSION_CAP_REJECT combo=({first.kind}->p{first.target_id},"
                        f"{second.kind}->p{second.target_id}) "
                        f"limit={offense_limit} my_prod={world.my_prod} enemy_prod={world.enemy_prod} "
                        f"mg_state={getattr(world, '_cached_midgame_state', 'N/A')}"
                    )
                    continue
            if _combo_is_compatible(combo, caps, world):
                combos.append(combo)
    combos.sort(key=lambda combo: -sum(_component_move_sort_score(world, prop) for prop in combo))
    world.add_debug(f"BEAM_ACTION_COMBINATIONS n={len(combos)} offense_limit={offense_limit}")
    return combos[:BEAM_ROOT_COMBO_LIMIT]


def _initial_abstract_beam_state(world):
    my_ids = {p.id for p in world.my_planets}
    enemy_ids = {p.id for p in world.enemy_planets}
    neutral_ids = {p.id for p in world.neutral_planets}
    planet_value = {}
    for p in world.normal_planets:
        nearest_enemy = min((dp(p, e) for e in world.enemy_planets), default=80.0)
        nearest_my = min((dp(p, m) for m in world.my_planets), default=80.0)
        bottleneck = max(0.0, 60.0 - abs(nearest_enemy - nearest_my)) * 0.4
        planet_value[p.id] = (
            int(p.production) * 12.0
            + (20.0 if _planet_role(p) == ROLE_LAUNCHPAD else 10.0 if _planet_role(p) == ROLE_BRIDGE else 2.0)
            + (12.0 if is_idle(p) else 0.0)
            + bottleneck
        )
    conflict_opportunity = 0.0
    for fl in world.enemy_fleets:
        target = fleet_target(fl, world.normal_planets, world.ang_vel)
        if target is not None and target.owner not in (world.player, -1):
            conflict_opportunity += int(fl.ships)
    return {
        "turn": 0,
        "my_ids": my_ids,
        "enemy_ids": enemy_ids,
        "neutral_ids": neutral_ids,
        "my_prod": sum(int(world.planet_by_id[pid].production) for pid in my_ids),
        "enemy_prod": sum(int(world.planet_by_id[pid].production) for pid in enemy_ids),
        "my_ships": int(world.my_total_ships),
        "enemy_ships": int(world.enemy_total_ships),
        "events": [],
        "opponent_events": [],
        "planet_value": planet_value,
        "vulnerability": 0.0,
        "conflict_opportunity": conflict_opportunity,
        "conversion_momentum": 0.0,
        "root_combo": (),
        "score": 0.0,
    }


def _opponent_events_from_forecast(world, enemy_actions):
    events = []
    for action in (enemy_actions or {}).get("actions", [])[:10]:
        target = action["target"]
        if world.is_comet(target):
            continue
        turn = max(1, min(BEAM_SEARCH_DEPTH, int(math.ceil(action["eta"]))))
        events.append({
            "turn": turn,
            "target_id": target.id,
            "source_id": action["source"].id,
            "send": int(action["send"]),
        })
    return events


def _apply_root_combo_to_state(world, base_state, combo, enemy_actions):
    state = {
        key: (set(value) if isinstance(value, set) else list(value) if isinstance(value, list) else value)
        for key, value in base_state.items()
    }
    state["events"] = []
    state["opponent_events"] = _opponent_events_from_forecast(world, enemy_actions)
    state["root_combo"] = combo
    launched = 0
    for prop in combo:
        total = _proposal_total_ships(prop)
        launched += total
        if prop.kind in OFFENSIVE_MISSIONS:
            target = world.planet_by_id.get(prop.target_id)
            momentum = _conversion_momentum_bonus(world, target, emit=True) if target is not None else 0.0
            state["conversion_momentum"] = float(state.get("conversion_momentum", 0.0)) + momentum
            state["events"].append({
                "turn": max(1, min(BEAM_SEARCH_DEPTH, _offensive_proposal_arrival_step(world, prop) - world.step)),
                "target_id": prop.target_id,
                "ships": total,
                "kind": prop.kind,
            })
        elif prop.kind in REINFORCEMENT_MISSIONS:
            state["vulnerability"] = max(0.0, float(state.get("vulnerability", 0.0)) - total * 0.08)
            if not any(token in (prop.reason or "") for token in ("deficit", "under_attack", "hub_security")):
                state["conversion_momentum"] = float(state.get("conversion_momentum", 0.0)) - total * 0.25
    state["my_ships"] = max(0, int(state["my_ships"]) - launched)
    return state


def simulate_abstract_beam_step(world, state):
    state = {
        key: (set(value) if isinstance(value, set) else list(value) if isinstance(value, list) else value)
        for key, value in state.items()
    }
    state["turn"] = int(state["turn"]) + 1
    state["my_ships"] += int(state["my_prod"])
    state["enemy_ships"] += int(state["enemy_prod"])

    for event in list(state.get("events", [])):
        if event["turn"] != state["turn"]:
            continue
        pid = event["target_id"]
        planet = world.planet_by_id.get(pid)
        if planet is None:
            continue
        was_enemy = pid in state["enemy_ids"]
        if pid not in state["my_ids"]:
            state["my_ids"].add(pid)
            state["neutral_ids"].discard(pid)
            state["enemy_ids"].discard(pid)
            state["my_prod"] += int(planet.production)
            if was_enemy:
                state["enemy_prod"] = max(0, int(state["enemy_prod"]) - int(planet.production))

    for event in list(state.get("opponent_events", [])):
        if event["turn"] != state["turn"]:
            continue
        pid = event["target_id"]
        planet = world.planet_by_id.get(pid)
        if planet is None:
            continue
        if pid in state["my_ids"]:
            rough_defense = int(planet.ships) + int(planet.production) * max(1, state["turn"])
            if event["send"] > rough_defense + 6:
                state["my_ids"].discard(pid)
                state["enemy_ids"].add(pid)
                state["my_prod"] = max(0, int(state["my_prod"]) - int(planet.production))
                state["enemy_prod"] += int(planet.production)
                state["vulnerability"] = float(state.get("vulnerability", 0.0)) + 80.0
        elif pid in state["neutral_ids"]:
            state["neutral_ids"].discard(pid)
            state["enemy_ids"].add(pid)
            state["enemy_prod"] += int(planet.production)

    vuln = 0.0
    for pid in state["my_ids"]:
        planet = world.planet_by_id.get(pid)
        if planet is None:
            continue
        nearest_enemy = min((dp(planet, e) for e in world.enemy_planets), default=999.0)
        if nearest_enemy < 34.0 and int(planet.ships) < max(8, int(planet.production) * 3):
            vuln += (34.0 - nearest_enemy) * 1.5 + max(0, 12 - int(planet.ships))
    state["vulnerability"] = float(state.get("vulnerability", 0.0)) + vuln * 0.03
    return state


def simulate_step(world, state):
    """Beam-search projection step; keeps WorldModel immutable and safety checks on commit."""
    return simulate_abstract_beam_step(world, state)


DEFAULT_BEAM_EVAL_WEIGHTS = {
    "production": 1.0,
    "ships": 1.0,
    "control": 1.0,
    "enemy_capture": 1.0,
    "neutral_capture": 1.0,
    "vulnerability": 1.0,
    "vulture": 1.0,
    "conversion_momentum": 1.0,
}


def evaluate_board_state(world_state, weights=None):
    weights = weights or DEFAULT_BEAM_EVAL_WEIGHTS
    if isinstance(world_state, WorldModel):
        my_ids = {p.id for p in world_state.my_planets}
        enemy_ids = {p.id for p in world_state.enemy_planets}
        neutral_ids = {p.id for p in world_state.neutral_planets}
        my_prod = world_state.my_prod
        enemy_prod = world_state.enemy_prod
        my_ships = world_state.my_total_ships
        enemy_ships = world_state.enemy_total_ships
        planet_value = {
            p.id: int(p.production) * 12.0 + (20.0 if _planet_role(p) == ROLE_LAUNCHPAD else 8.0)
            for p in world_state.normal_planets
        }
        vulnerability = sum(
            1 for p in world_state.my_planets
            if world_state.nearest_enemy_distance(p) < 34.0 and int(p.ships) < max(8, int(p.production) * 3)
        ) * 35.0
        conflict_opportunity = sum(
            int(f.ships)
            for f in world_state.enemy_fleets
            if (fleet_target(f, world_state.normal_planets, world_state.ang_vel) or f).owner not in (world_state.player, -1)
        )
        conversion_momentum = 0.0
        # Strategic Defensive Priority: heavy penalty for high-production hubs that are
        # under active threat.  This biases the beam toward defending before expanding.
        high_prod_threat_penalty = sum(
            int(p.production) * 55.0
            for p in world_state.my_planets
            if int(p.production) >= 3
            and world_state.real_incoming_threat(p)["deficit"] > 0
        ) * weights.get("vulnerability", 1.0)
    else:
        my_ids = world_state["my_ids"]
        enemy_ids = world_state["enemy_ids"]
        neutral_ids = world_state.get("neutral_ids", set())
        my_prod = world_state["my_prod"]
        enemy_prod = world_state["enemy_prod"]
        my_ships = world_state["my_ships"]
        enemy_ships = world_state["enemy_ships"]
        planet_value = world_state.get("planet_value", {})
        vulnerability = world_state.get("vulnerability", 0.0)
        conflict_opportunity = world_state.get("conflict_opportunity", 0.0)
        high_prod_threat_penalty = world_state.get("high_prod_threat_penalty", 0.0)
        conversion_momentum = world_state.get("conversion_momentum", 0.0)

    production_dominance = (my_prod - enemy_prod) * 95.0 * weights.get("production", 1.0)
    ship_pool_health = (my_ships - enemy_ships) * 0.42 * weights.get("ships", 1.0)
    enemy_capture_pressure = -sum(planet_value.get(pid, 0.0) for pid in enemy_ids) * 0.16
    neutral_capture_pressure = -sum(planet_value.get(pid, 0.0) for pid in neutral_ids) * 0.10
    strategic_control = (
        len(my_ids) * 55.0
        - len(enemy_ids) * 38.0
        + sum(planet_value.get(pid, 0.0) for pid in my_ids)
        - sum(planet_value.get(pid, 0.0) * 0.55 for pid in enemy_ids)
    ) * weights.get("control", 1.0)
    vulnerability_penalty = float(vulnerability) * 5.0 * weights.get("vulnerability", 1.0)
    enemy_capture_bonus = enemy_capture_pressure * weights.get("enemy_capture", 1.0)
    neutral_capture_bonus = neutral_capture_pressure * weights.get("neutral_capture", 1.0)
    vulture_bonus = float(conflict_opportunity) * 1.8 * weights.get("vulture", 1.0)
    conversion_momentum_bonus = float(conversion_momentum) * weights.get("conversion_momentum", 1.0)
    return (
        production_dominance
        + ship_pool_health
        + strategic_control
        + enemy_capture_bonus
        + neutral_capture_bonus
        + vulture_bonus
        + conversion_momentum_bonus
        - vulnerability_penalty
        - high_prod_threat_penalty
    )


def beam_search_orchestrator(world, states, chain_plan, enemy_actions, control_ratio, deadline, staging_id=None, eval_weights=None):
    soft_deadline = deadline - BEAM_TIME_BUFFER
    cfg = getattr(world, "dynamic_config", {})
    beam_depth = int(cfg.get("beam_depth", BEAM_SEARCH_DEPTH))
    beam_width = int(cfg.get("beam_width", BEAM_SEARCH_WIDTH))
    if deadline - time.perf_counter() < 0.08:
        beam_depth = min(beam_depth, 1)
        beam_width = min(beam_width, 3)
        world.add_debug("BEAM_DEPTH_REDUCED_FOR_TIME")
        world.add_debug("BEAM_WIDTH_REDUCED_FOR_TIME")
    elif world.step < 100:
        beam_depth = min(beam_depth, 2)
        beam_width = min(beam_width, 5)
    else:
        beam_depth = min(beam_depth, 3)
        beam_width = min(beam_width, 8)
    world.add_debug(
        f"CLOCK_AWARE_BEAM_CONFIG depth={beam_depth} width={beam_width} time_left={deadline - time.perf_counter():.3f}"
    )

    # Strategic Defensive Priority: if high-production hubs are under threat or
    # total production is behind, switch to Defensive eval weights so the beam
    # values staying alive over capturing new territory.
    threatened_hubs = [
        p for p in world.my_planets
        if int(p.production) >= 3 and world.real_incoming_threat(p)["deficit"] > 0
    ]
    prod_declining = world.my_prod < world.enemy_prod
    if getattr(world, "aggressiveness_mode", "BALANCED") == "AGGRESSIVE":
        aggressive_override = {
            "production": 1.35,
            "ships": 0.85,
            "control": 1.45,
            "enemy_capture": 1.35,
            "neutral_capture": 1.65,
            "vulnerability": 0.85 if not threatened_hubs else 1.65,
            "vulture": 1.25,
            "conversion_momentum": 2.20,
        }
        eval_weights = aggressive_override
    elif threatened_hubs or prod_declining:
        _defensive_override = {
            "production": 1.10,
            "ships": 1.20,
            "control": 0.90,
            "enemy_capture": 0.70,
            "neutral_capture": 0.75,
            "vulnerability": 2.80,
            "vulture": 0.80,
            "conversion_momentum": 0.75,
        }
        eval_weights = _defensive_override
        hub_ids = [p.id for p in threatened_hubs]
        world.add_debug(
            f"DEFENSIVE_PRIORITY_OVERRIDE threatened_hubs={hub_ids} "
            f"prod_declining={prod_declining} my_prod={world.my_prod} enemy_prod={world.enemy_prod}"
        )
    if world.step < 100 and getattr(world, "aggressiveness_mode", "BALANCED") != "DEFENSIVE":
        early_weights = dict(eval_weights or DEFAULT_BEAM_EVAL_WEIGHTS)
        early_weights.update({
            "production": max(early_weights.get("production", 1.0), 1.45),
            "control": max(early_weights.get("control", 1.0), 1.65),
            "neutral_capture": max(early_weights.get("neutral_capture", 1.0), 1.85),
            "enemy_capture": max(early_weights.get("enemy_capture", 1.0), 1.35),
            "ships": min(early_weights.get("ships", 1.0), 0.90),
            "conversion_momentum": max(early_weights.get("conversion_momentum", 1.0), 2.35),
        })
        eval_weights = early_weights
        world.add_debug("EARLY_CAPTURE_PRIORITY_ACTIVE")

    root_components = generate_atomic_component_moves(
        world, states, chain_plan, enemy_actions, control_ratio, soft_deadline, staging_id=staging_id
    )
    if time.perf_counter() > soft_deadline:
        world.add_debug("BEAM_SEARCH_BEST_SO_FAR_USED")
        return (root_components[0],) if root_components else ()
    combos = generate_action_combinations(world, states, root_components, soft_deadline)
    if not combos:
        world.add_debug("BEAM_SEARCH_NO_COMPONENTS")
        return ()

    base = _initial_abstract_beam_state(world)
    best_combo = combos[0]
    best_score = -1e18
    beam = []
    for combo in combos:
        if time.perf_counter() > soft_deadline:
            break
        state = _apply_root_combo_to_state(world, base, combo, enemy_actions)
        state["score"] = evaluate_board_state(state, eval_weights) + sum(_component_move_sort_score(world, p) for p in combo) * 0.15
        beam.append(state)
        if state["score"] > best_score:
            best_score = state["score"]
            best_combo = combo
    beam.sort(key=lambda s: -s["score"])
    beam = beam[:beam_width]

    for _depth in range(beam_depth):
        if time.perf_counter() > soft_deadline:
            world.add_debug("BEAM_SEARCH_TIME_CUTOFF")
            world.add_debug("BEAM_SEARCH_BEST_SO_FAR_USED")
            break
        next_beam = []
        for state in beam:
            projected = simulate_step(world, state)
            projected["score"] = evaluate_board_state(projected, eval_weights)
            next_beam.append(projected)
            if projected["score"] > best_score:
                best_score = projected["score"]
                best_combo = projected.get("root_combo", best_combo)
        if not next_beam:
            break
        next_beam.sort(key=lambda s: -s["score"])
        beam = next_beam[:beam_width]

    world.add_debug(
        f"BEAM_SEARCH_SELECTED score={best_score:.1f} combo={[f'{p.kind}->p{p.target_id}' for p in best_combo]}"
    )
    return best_combo


def agent(obs, config=None):
    """Rolling-horizon beam-search agent using existing safety/geometry guards."""
    start = time.perf_counter()
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    deadline = start + min(SOFT_DEADLINE, max(0.55, act_timeout * 0.82))
    world = WorldModel(obs)

    if not world.my_planets:
        update_ownership_memory(world)
        return []

    world.add_debug("ROLLING_HORIZON_BEAM_AGENT_ACTIVE")
    world.add_debug("MAIN34_PACKET_RULE_PRESERVED")
    world.add_debug("MAIN34_CHAIN_ENGINE_PRESERVED")
    world.add_debug("TACTICAL_CONSTANT_TUNING_APPLIED")
    world.add_debug("PACKET_DISCIPLINE_PRESERVED")

    if world.step <= 1:
        _prev_owners.clear()
        _prev_ships.clear()
        _opponent_model_memory.clear()
        _midgame_dominance_last.clear()
        _shot_history.clear()
        _bad_shot_patterns.clear()
        _beam_expansion_last.clear()
        _last_capture_step.clear()
        _midgame_conversion_memory.clear()
        _recent_launch_history.clear()
        _adaptive_meta_controllers.clear()
        _pending_mission_launches.clear()
        _pending_delayed_missions.clear()
        _expansion_obligation_cooldown.clear()
        _staging_controller_memory.clear()
        _territory_conversion_history.clear()

    moves = []
    states = build_planet_states(world)
    world.pressure_map = build_pressure_map(world)
    update_rotational_hubs(world)
    prediction = build_prediction_timeline(world, horizons=(5, 10, 15, 20, 30, 40))
    enemy_actions = forecast_enemy_actions(world, horizons=(5, 10, 15, 20))
    chain_plan = build_launchpad_chain_plan(world, states)
    world._active_chain_plan = chain_plan
    activate_offense_first_objective(world)
    world._active_structure = classify_planet_structure(world, states, chain_plan)
    _control_phase, control_ratio = control_phase_selected(world)
    update_territory_conversion_history(world)
    conversion_pressure = territory_conversion_score(world)
    configure_aggressiveness_controller(world, control_ratio, conversion_pressure)
    world._reinforcement_loop_active = _reinforcement_loop_active(world)
    adaptive_constant_tuner(world, control_ratio, conversion_pressure, deadline)
    midgame_conversion = midgame_conversion_context(world, control_ratio)
    world.midgame_conversion_active = bool(midgame_conversion.get("active"))
    staging_id = choose_staging_launchpad(world, states, chain_plan)
    meta_controller = adaptive_meta_controller_for(world)
    active_meta, eval_weights = meta_controller.update(world)
    world.add_debug(f"META_STRATEGY_ACTIVE {active_meta}")

    delayed_released = process_delayed_launches(world, moves)
    if delayed_released:
        _last_capture_step[world.player] = world.step
        execute_pending_missions(world, moves)
        if DEBUG:
            for event in world.debug_events:
                print(event)
        update_ownership_memory(world)
        return moves

    # Phase Transition Lock: classify the current strategic phase and attach a
    # flag the beam and target-selection can read to restrict risky expansion.
    _strategic_phase = classify_strategic_phase(world)
    world._strategic_phase = _strategic_phase
    world._phase_transition_locked = (_strategic_phase == ControlPhase.CONSOLIDATE)
    if world._phase_transition_locked:
        world.add_debug(
            f"PHASE_TRANSITION_LOCKED strategic_phase={_strategic_phase} "
            f"step={world.step} my_prod={world.my_prod} enemy_prod={world.enemy_prod}"
        )

    # Cache midgame state on world so _active_offensive_limit() can read it
    # without recomputing fleet_ratio a second time.
    _fleet_ratio_now = compute_fleet_ratio(world)
    world._cached_midgame_state = classify_midgame_state(world, _fleet_ratio_now)
    world.add_debug(
        f"MIDGAME_STATE {world._cached_midgame_state} "
        f"fleet_ratio={_fleet_ratio_now:.2f} my_prod={world.my_prod} enemy_prod={world.enemy_prod}"
    )

    failsafe_combo = time_budget_guard(world, states, chain_plan, enemy_actions, deadline)
    bundle_selected = False
    if failsafe_combo:
        combo = failsafe_combo
    else:
        emergency_props = tuple(_build_defense_component_proposals(world, states, deadline))
        launchpad_guard_props = reinforce_launchpad_from_surroundings(world, states, deadline)
        game_winning_props = detect_game_winning_opportunities(world, states, chain_plan, enemy_actions, deadline)
        small_escape_props = build_small_start_escape_props(world, states, chain_plan, deadline)
        parallel_sweep_props = build_parallel_opening_sweep_props(world, states, chain_plan, deadline)
        if world.step <= 70 and parallel_sweep_props:
            world.add_debug("EARLY_WAIT_FORBIDDEN_CAPTURE_EXISTS")
            world.add_debug("OPENING_IDLE_REJECTED")
        early_sweep_props = build_early_nearest_sweep_props(world, states, chain_plan, deadline)
        multi_axis_props = build_multi_axis_expansion_props(world, states, chain_plan, deadline)
        front_push_props = build_front_attacker_push_props(world, states, chain_plan, deadline)
        counter_after_defense_props = build_counterattack_after_defense_props(world, states, enemy_actions, deadline)
        front_flow_props = build_front_attacker_flow_props(world, states, chain_plan, deadline)
        tactical_combo = run_tactical_interrupt_layer(
            world, states, chain_plan, enemy_actions, conversion_pressure, deadline
        )
        nuisance_props = build_nuisance_interrupt_props(world, states, enemy_actions, deadline)
        beam_combo = ()
        if deadline - time.perf_counter() > 0.07:
            beam_combo = beam_search_orchestrator(
                world, states, chain_plan, enemy_actions, control_ratio, deadline, staging_id=staging_id, eval_weights=eval_weights
            )
        else:
            world.add_debug("TIME_BUDGET_FAILSAFE_ACTIVE")
        proposal_groups = (
            emergency_props,
            launchpad_guard_props,
            game_winning_props,
            small_escape_props,
            parallel_sweep_props,
            early_sweep_props,
            multi_axis_props,
            front_push_props,
            counter_after_defense_props,
            front_flow_props,
            tactical_combo,
            nuisance_props,
            beam_combo,
        )
        emergency_must_win = any(
            (
                (tgt := world.planet_by_id.get(prop.target_id)) is not None
                and prop.kind in REINFORCEMENT_MISSIONS
                and int(tgt.production) >= HUB_SECURITY_PROD_THRESHOLD
                and world.real_incoming_threat(tgt)["deficit"] > 0
            )
            for prop in tuple(emergency_props) + tuple(launchpad_guard_props)
        )
        same_step_bundle = () if emergency_must_win else build_same_step_attack_bundle(
            world, states, proposal_groups, deadline
        )
        if same_step_bundle:
            combo = same_step_bundle
            bundle_selected = True
        else:
            combo = priority_arbiter(
                world,
                states,
                proposal_groups,
                deadline,
            )
        if not combo:
            combo = fast_heuristic_fallback(world, states, chain_plan, enemy_actions, deadline)

    if bundle_selected:
        commit_attack_bundle(world, combo, moves, states, deadline)
    else:
        for prop in combo:
            if time.perf_counter() > deadline - BEAM_TIME_BUFFER:
                world.add_debug("BEAM_COMMIT_TIME_CUTOFF")
                break
            if _commit_proposal(world, prop, moves):
                if prop.kind in OFFENSIVE_MISSIONS:
                    _last_capture_step[world.player] = world.step
                    if world.step < 10 and (
                        "early_nearest_sweep" in (prop.reason or "")
                        or "small_start_escape" in (prop.reason or "")
                        or "early_direct_expansion" in (prop.reason or "")
                    ):
                        world.add_debug("OPENING_ATTACK_STARTED_BEFORE_STEP_10")
            else:
                world.add_debug(f"BEAM_COMMIT_REJECTED {prop.kind}->p{prop.target_id}")

    execute_pending_missions(world, moves)

    if not moves:
        # If no root combo survived commit validation, do not fall back to the
        # old hierarchy. Log diagnostics and let memory update normally.
        log_missed_opportunities(world, states, chain_plan)

    if DEBUG:
        for event in world.debug_events:
            print(event)
    update_ownership_memory(world)
    return moves
