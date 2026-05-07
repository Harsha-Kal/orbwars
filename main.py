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
import time
import heapq
from dataclasses import dataclass
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

SUN_X, SUN_Y   = 50.0, 50.0
SUN_R          = 10.0
MAX_SPEED      = 6.0
ROTATION_LIMIT = 50.0
TOTAL_STEPS    = 500

DEFEND_NET = 8      # reinforce if net projected ships < this (was 5)
INCOMING_T = 0.85

HIT_MARGIN              = 3.0   # acceptable aimed-point miss beyond planet radius

EARLY_STEPS            = 55   # lighter reserves and looser rotating timing before this step
FRONTLINE_DIST        = 25.0  # distance to nearest enemy below which = frontline

# Endgame
LATE_GAME_STEPS = 350   # prefer enemy planets over neutrals from this step on
FINAL_STEPS     = 460   # drain pass: send all idle ships to any reachable target

# Opponent style detection
TURTLE_SHIP_THRESH     = 40    # avg ships per enemy planet above this → turtle classifier

# Defense reactivity
DEFENSE_ETA_HORIZON   = 30    # only count inbound fleets arriving within this many turns

# Search planner margins
HOSTILE_MARGIN_BASE = 5
HOSTILE_MARGIN_CAP = 16

# Proactive expansion.  Keep these early and explicit: the beam planner is good
# at weighing options, but nearest safe neutrals need a hard tempo bias.
DEBUG = False
NEAREST_LOCK_DIST = 28.0
NEAREST_LOCK_ETA = 18.0
NEAREST_LOCK_MAX_SOURCES = 3
PROACTIVE_EXPANSION_MAX_SOURCES = 4
STALL_FORCE_TURNS = 6
STALL_FORCE_SHIPS = 180
MAX_WAVE_WAIT = 3
MIN_OFFENSIVE_SHIPS = 5
MIN_CAMPAIGN_SHIPS = 10
MIN_WAVE_FRACTION = 0.90

_wave_reservation = {
    "target_id": None,
    "source_ids": [],
    "started_step": -1,
    "launch_by_step": -1,
    "required_ships": 0,
    "reason": ""
}

# Mission Coordinator
LOCAL_HUB_SHIPS = 60        # planet with >= this many ships acts as a command hub
LOCAL_HUB_RADIUS = 40.0     # command hub targets within this distance first
ETA_SYNC_WINDOW = 8         # max turn spread for a sync attack to be valid
FLEET_RATIO_SOFT = 0.55     # above this: block new speculative attacks
FLEET_RATIO_HARD = 0.70     # above this: only defense/reinforce/finish allowed


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

# ── Search / simulation architecture ─────────────────────────────────────────

SOFT_DEADLINE = 0.82
SIM_HORIZON = 100
BEAM_WIDTH = 8
BEAM_DEPTH = 3
MAX_CANDIDATES = 48
MAX_GROUP_SOURCES = 7


class StrategyMode:
    OPENING_TEMPO = "OPENING_TEMPO"
    SAFE_EXPANSION = "SAFE_EXPANSION"
    CONTEST_NEUTRALS = "CONTEST_NEUTRALS"
    ANTI_LEADER = "ANTI_LEADER"
    BEHIND_STEAL = "BEHIND_STEAL"
    TURTLE_BREAKER = "TURTLE_BREAKER"
    COLLAPSE = "COLLAPSE"
    FORCE_WAVE = "FORCE_WAVE"
    FINAL_DRAIN = "FINAL_DRAIN"
    FOUR_PLAYER_EXPAND_FIRST = "FOUR_PLAYER_EXPAND_FIRST"


@dataclass
class ShotOption:
    score: float
    src_id: int
    target_id: int
    ships: int
    angle: float
    eta: float
    mission: str


@dataclass
class Mission:
    kind: str
    target_id: int
    score: float
    options: list


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
    kind: str           # DEFEND REINFORCE FINISH_CAPTURE CAPTURE_NEUTRAL LOCAL_STRIKE SYNC_ATTACK COLLAPSE
    target_id: int
    priority: float
    required_ships: int
    planned_sources: list   # [(src_id, ships, angle, eta), ...]
    eta_min: float
    eta_max: float
    reason: str


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
        self.owned_planets = self.my_planets
        self.neutral_planets = [p for p in self.normal_planets if p.owner == -1]
        self.enemy_planets = [p for p in self.normal_planets if p.owner not in (-1, self.player)]
        self.my_fleets = [f for f in self.fleets if f.owner == self.player]
        self.enemy_fleets = [f for f in self.fleets if f.owner != self.player]
        self.static_planets = [p for p in self.planets if is_idle(p)]
        self.rotating_planets = [p for p in self.planets if not is_idle(p)]
        self.remaining = max(1, TOTAL_STEPS - self.step)

        self.my_total_ships = sum(int(p.ships) for p in self.my_planets) + sum(int(f.ships) for f in self.my_fleets)
        self.enemy_total_ships = sum(int(p.ships) for p in self.enemy_planets) + sum(int(f.ships) for f in self.enemy_fleets)
        self.my_prod = sum(int(p.production) for p in self.my_planets)
        self.enemy_prod = sum(int(p.production) for p in self.enemy_planets)
        self.enemy_prod_by_owner = {}
        self.enemy_ships_by_owner = {}
        self.enemy_planets_by_owner = {}
        for p in self.enemy_planets:
            self.enemy_prod_by_owner[p.owner] = self.enemy_prod_by_owner.get(p.owner, 0) + int(p.production)
            self.enemy_ships_by_owner[p.owner] = self.enemy_ships_by_owner.get(p.owner, 0) + int(p.ships)
            self.enemy_planets_by_owner[p.owner] = self.enemy_planets_by_owner.get(p.owner, 0) + 1
        for f in self.enemy_fleets:
            self.enemy_ships_by_owner[f.owner] = self.enemy_ships_by_owner.get(f.owner, 0) + int(f.ships)

        self.leader = None
        self.leader_score = 0
        for owner in set(self.enemy_prod_by_owner) | set(self.enemy_ships_by_owner):
            score = (
                self.enemy_prod_by_owner.get(owner, 0) * 18
                + self.enemy_ships_by_owner.get(owner, 0)
                + self.enemy_planets_by_owner.get(owner, 0) * 12
            )
            if score > self.leader_score:
                self.leader = owner
                self.leader_score = score
        self.my_score = self.my_prod * 18 + self.my_total_ships + len(self.my_planets) * 12

        self.arrivals_by_target = {}
        self.incoming_to_targets = {}
        self.enemy_incoming_to_targets = {}
        self._build_arrivals()
        self.shot_cache = {}
        self.reaction_cache = {}
        self.timeline_cache = {}
        self.committed = {}
        self.offensive_ships = 0
        self.wave_attempted = False
        self.debug_events = []
        self.features = {}
        self._compute_features()
        self._detect_game_type()

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

    def planet_pos(self, p, turns):
        base = self.initial_planets.get(p.id, p)
        return predict_pos(base, self.ang_vel, turns)

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
        cap = max(20, int(p.ships * (0.45 if self.step < EARLY_STEPS else 0.55)))
        return min(int(raw), cap)

    def surplus(self, p):
        return max(0, int(p.ships) - self.committed.get(p.id, 0) - self.reserve_for(p))

    def available_ships_after_reserve(self, p):
        return self.surplus(p)

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

    def production_value(self, p):
        return int(p.production) * max(1, self.remaining)

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

    def enemy_reaches_faster(self, tgt, my_eta, margin=1.5):
        _, enemy_t = self.reaction_times(tgt)
        return enemy_t + margin < my_eta

    def add_debug(self, message):
        if DEBUG:
            self.debug_events.append(message)

    def commit(self, src, tgt, ships, moves):
        ships = int(min(ships, int(src.ships) - self.committed.get(src.id, 0)))
        if ships <= 0:
            return False
        angle, ok = self.aim(src, tgt, ships)
        if not ok:
            return False
        moves.append([src.id, angle, ships])
        self.committed[src.id] = self.committed.get(src.id, 0) + ships
        if tgt.owner != self.player:
            self.incoming_to_targets[tgt.id] = self.incoming_to_targets.get(tgt.id, 0) + ships
            self.offensive_ships += ships
        return True


BACKYARD_DIST = 22.0          # enemy planet within this distance of my nearest = backyard
FOUR_P_ATTACK_STEP = 80       # step below which 4-player early-attack block is active
FOUR_P_EXPAND_STEP = 100      # step below which FOUR_PLAYER_EXPAND_FIRST mode activates


def is_in_my_backyard(world, p):
    """True if planet p is very close to my cluster (enemy planted inside my territory)."""
    if not world.my_planets:
        return False
    return min(dp(m, p) for m in world.my_planets) < BACKYARD_DIST


def should_block_early_enemy_attack_4p(world, target):
    """Return True when a 4-player early-game enemy attack should be deferred for expansion."""
    if not world.is_four_player:
        return False
    if world.step >= FOUR_P_ATTACK_STEP:
        return False
    if target.owner in (-1, world.player):
        return False
    if is_in_my_backyard(world, target):
        return False
    if any(world.real_incoming_threat(p)["deficit"] > 0 for p in world.my_planets):
        return False
    if not world.neutral_planets:
        return False
    world.add_debug(f"FOUR_PLAYER_ATTACK_BLOCK target=p{target.id} reason=expand_neutrals_first")
    return True


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
    if mode == StrategyMode.COLLAPSE and tgt.owner not in (-1, world.player):
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
    selected, total, goal = estimate_grouped_sources(world, tgt, need, max_sources=max_sources)
    if total < need:
        return False
    if not force and total < max(need, min(goal, need + 1)):
        return False
    sent = 0
    for src, send, _angle in selected:
        if sent >= goal:
            break
        n = min(send, goal - sent)
        if n <= 0:
            continue
        if world.commit(src, tgt, n, moves):
            sent += n
    if sent >= need:
        world.wave_attempted = True
        world.add_debug(f"grouped_capture target=p{tgt.id} need={need} sent={sent} sources={[s.id for s, _, _ in selected]}")
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


def route_search_to_enemy(world, mode, deadline):
    routes = []
    if not world.enemy_planets:
        return routes
    targets = world.enemy_planets
    bridges = [n for n in world.neutral_planets if n.production >= 3]
    for enemy in targets:
        if time.perf_counter() > deadline:
            break
        best_src = min(world.my_planets, key=lambda p: dp(p, enemy), default=None)
        if best_src is not None:
            direct_need = world.ships_needed_to_capture(best_src, enemy)
            direct_cost = world.eta(best_src, enemy, direct_need) + direct_need * 0.18
            direct_score = score_target(world, best_src, enemy, mode) - direct_cost
            routes.append(RoutePlan(enemy.id, enemy.id, direct_score, direct_cost, [enemy.id]))
        if time.perf_counter() < deadline:
            astar_plan = astar_route_to_target(world, enemy, deadline)
            if astar_plan is not None:
                routes.append(astar_plan)
        for bridge in bridges:
            if time.perf_counter() > deadline:
                break
            src = min(world.my_planets, key=lambda p: dp(p, bridge), default=None)
            if src is None:
                continue
            need = world.ships_needed_to_capture(src, bridge)
            to_bridge = world.eta(src, bridge, need)
            after = travel_turns(dp(bridge, enemy), max(1, need))
            progress = max(0, dp(src, enemy) - dp(bridge, enemy))
            cost = to_bridge + after * 0.45 + need * 0.14
            reward = bridge.production * 30 + enemy.production * 25 + progress * 1.2
            if enemy.owner == world.leader:
                reward += 40
            routes.append(RoutePlan(enemy.id, bridge.id, reward - cost, cost, [bridge.id, enemy.id]))
    return sorted(routes, key=lambda r: -r.score)[:10]


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
        if should_wait_for_better_wave(world, tgt):
            reserve_wave(world, tgt, sources)
        else:
            world.add_debug(f"LOW_VALUE_NOW_SKIP target=p{tgt.id} reason=cant_reach_need pool={pool} need={need}")
        world.add_debug(f"TRICKLE_BLOCK target=p{tgt.id} ships={pool} need={need}")
        return 0
    goal = need if pool >= need else (int(pool * 0.32) if allow_partial else 0)
    if goal < min(8, need):
        return 0
    if sync and len(sources) >= 2:
        etas = [(world.eta(p, tgt, max(1, min(world.surplus(p), goal))), p) for p in sources]
        median_eta = sorted(e for e, _ in etas)[len(etas) // 2]
        sources = [p for eta, p in etas if abs(eta - median_eta) <= 4] or sources[:3]
    sent = 0
    for src in sources:
        if sent >= goal:
            break
        n = min(world.surplus(src), goal - sent)
        if n <= 0:
            continue
        if n < 3 and sent + n < goal:
            continue
        if world.commit(src, tgt, n, moves):
            sent += n
    if sent > 0 and tgt.owner != world.player:
        world.wave_attempted = True
    return sent


def make_shot_option(world, src, tgt, mode, mission):
    need = world.ships_needed_to_capture(src, tgt)
    if need <= 0:
        return None
    surplus = world.surplus(src)
    if surplus <= 0:
        return None
    send = min(surplus, need)
    if send < need and mission not in ("force", "pressure"):
        return None
    if send < 5 and send < need:
        return None
    angle, ok = world.aim(src, tgt, send)
    if not ok:
        return None
    eta = world.eta(src, tgt, send)
    if eta > world.remaining - 3:
        return None
    final_all_in = mode in (StrategyMode.COLLAPSE, StrategyMode.FINAL_DRAIN)
    if not world.can_hold_after_capture(tgt, eta, send, final_all_in=final_all_in) and not final_all_in:
        return None
    score = score_target(world, src, tgt, mode)
    return ShotOption(score, src.id, tgt.id, int(send), angle, eta, mission)


def generate_candidate_missions(world, mode, deadline):
    missions = []
    target_pool = []
    if mode in (StrategyMode.OPENING_TEMPO, StrategyMode.FOUR_PLAYER_EXPAND_FIRST):
        target_pool = sorted(world.neutral_planets, key=lambda p: (dp(world.my_planets[0], p) if world.my_planets else 999, p.ships))[:12]
    elif mode == StrategyMode.BEHIND_STEAL:
        target_pool = world.neutral_planets + [e for e in world.enemy_planets if e.ships <= 35 or e.owner == world.leader]
    elif mode in (StrategyMode.ANTI_LEADER, StrategyMode.TURTLE_BREAKER):
        target_pool = [e for e in world.enemy_planets if e.owner == world.leader] or world.enemy_planets
    elif mode in (StrategyMode.COLLAPSE, StrategyMode.FINAL_DRAIN):
        target_pool = world.enemy_planets + world.neutral_planets
    else:
        target_pool = world.neutral_planets + world.enemy_planets

    routes = route_search_to_enemy(world, mode, deadline)
    for route in routes[:5]:
        tgt = world.planet_by_id.get(route.first_target_id)
        if tgt is not None and tgt not in target_pool:
            target_pool.insert(0, tgt)

    for src in world.my_planets:
        if time.perf_counter() > deadline:
            break
        if world.surplus(src) <= 0:
            continue
        for tgt in target_pool:
            if time.perf_counter() > deadline:
                break
            if tgt.id == src.id or tgt.owner == world.player:
                continue
            option = make_shot_option(world, src, tgt, mode, "capture")
            if option is not None:
                missions.append(Mission("single", tgt.id, option.score, [option]))

    # Multi-source missions for targets too large for one planet.
    for tgt in target_pool[:18]:
        if time.perf_counter() > deadline:
            break
        if tgt.owner == world.player:
            continue
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        need = world.ships_needed_to_capture(src, tgt, world.my_total_ships)
        sources = sorted(world.my_planets, key=lambda p: dp(p, tgt))[:MAX_GROUP_SOURCES]
        pool = sum(world.surplus(p) for p in sources)
        if pool >= need or (mode == StrategyMode.FORCE_WAVE and pool >= 25):
            score = score_target(world, src, tgt, mode) * (1.08 if tgt.owner != -1 else 1.0)
            missions.append(Mission("group", tgt.id, score, []))

    missions.sort(key=lambda m: -m.score)
    return missions[:MAX_CANDIDATES]


def beam_search_planner(world, mode, deadline):
    missions = generate_candidate_missions(world, mode, deadline)
    if not missions:
        return []
    beam = [([], 0.0, {}, set())]
    best = beam[0]
    for _ in range(BEAM_DEPTH if mode != StrategyMode.OPENING_TEMPO else 4):
        if time.perf_counter() > deadline:
            break
        next_beam = []
        for chosen, score, spent, targets in beam:
            for mission in missions:
                if mission.target_id in targets:
                    continue
                if mission.kind == "single":
                    opt = mission.options[0]
                    if spent.get(opt.src_id, 0) + opt.ships > world.planet_by_id[opt.src_id].ships:
                        continue
                    new_spent = dict(spent)
                    new_spent[opt.src_id] = new_spent.get(opt.src_id, 0) + opt.ships
                    next_beam.append((chosen + [mission], score + mission.score, new_spent, targets | {mission.target_id}))
                else:
                    next_beam.append((chosen + [mission], score + mission.score * 0.96, dict(spent), targets | {mission.target_id}))
        if not next_beam:
            break
        next_beam.sort(key=lambda item: -item[1])
        beam = next_beam[:BEAM_WIDTH]
        if beam[0][1] > best[1]:
            best = beam[0]
    return best[0] if best[0] else missions[:1]


def emergency_defense(world, moves):
    acted = False
    for tgt in sorted(world.my_planets, key=lambda p: -world.real_incoming_threat(p)["deficit"]):
        deficit = world.real_incoming_threat(tgt)["deficit"]
        if deficit <= 0:
            continue
        for src in sorted([p for p in world.my_planets if p.id != tgt.id], key=lambda p: dp(p, tgt)):
            n = min(world.surplus(src), deficit)
            if n <= 0:
                continue
            if world.commit(src, tgt, n, moves):
                deficit -= n
                acted = True
            if deficit <= 0:
                break
    return acted


def finish_started_captures(world, moves):
    acted = False
    targets = [p for p in world.neutral_planets + world.enemy_planets if world.incoming_to_targets.get(p.id, 0) > 0]
    for tgt in sorted(targets, key=lambda p: world.target_need_now(p)):
        need = world.target_need_now(tgt)
        if need <= 0 or need > 18:
            continue
        if build_grouped_wave(world, tgt, need, moves, max_sources=4, allow_partial=False) > 0:
            acted = True
    return acted


def anti_leader_overlay(world, moves, deadline):
    if world.is_four_player and world.step < FOUR_P_ATTACK_STEP and world.neutral_planets:
        return False
    if world.leader is None or world.leader_score <= world.my_score * 1.22:
        return False
    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < 15:
        return False
    targets = [p for p in world.enemy_planets if p.owner == world.leader]
    if not targets:
        return False
    tgt = max(targets, key=lambda p: p.production * 12 + max(0, 45 - p.ships) - min(dp(m, p) for m in world.my_planets))
    src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
    if src is None:
        return False
    need = world.ships_needed_to_capture(src, tgt, pool)
    if pool < need:
        world.add_debug(f"ANTI_LEADER_SKIP pool={pool} need={need}")
        return False
    return build_grouped_wave(world, tgt, need, moves, max_sources=6, allow_partial=False, sync=True) > 0


def force_wave_if_inactive(world, moves, idle_turns):
    if idle_turns < 9 or world.my_total_ships <= 250 or world.step < 120 or world.step > 430:
        return False
    targets = world.enemy_planets + world.neutral_planets
    if not targets:
        return False
    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < 25:
        return False
    tgt = max(targets, key=lambda p: max(score_target(world, min(world.my_planets, key=lambda m: dp(m, p)), p, StrategyMode.FORCE_WAVE), -1e8))
    src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
    need = world.ships_needed_to_capture(src, tgt, pool) if src else max(20, pool)
    return build_grouped_wave(world, tgt, need, moves, max_sources=7, allow_partial=False, sync=tgt.owner != -1) > 0


def force_action_if_stalling(world, moves, idle_turns, deadline):
    """Prefer forced nearest expansion before falling back to enemy pressure."""
    if idle_turns < STALL_FORCE_TURNS or world.my_total_ships < STALL_FORCE_SHIPS:
        return False
    if world.neutral_planets and nearest_expansion_plan(world, moves, StrategyMode.FORCE_WAVE, deadline, force=True):
        return True
    if idle_turns < 8 or world.my_total_ships <= 230:
        return False
    if world.is_four_player and world.step < FOUR_P_ATTACK_STEP and world.neutral_planets:
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


def final_drain(world, moves):
    if world.step < FINAL_STEPS and world.remaining > 45:
        return False
    acted = False
    for src in sorted(world.my_planets, key=lambda p: -int(p.ships)):
        spare = max(0, int(src.ships) - world.committed.get(src.id, 0) - 1)
        if spare <= 0:
            continue
        targets = sorted(world.enemy_planets + world.neutral_planets, key=lambda t: world.eta(src, t, spare))
        for tgt in targets:
            if world.eta(src, tgt, spare) > world.remaining - 1:
                continue
            need = world.ships_needed_to_capture(src, tgt, spare)
            if 0 < need <= spare and world.commit(src, tgt, need, moves):
                acted = True
                break
    return acted


def execute_missions(world, missions, moves, mode):
    acted = False
    for mission in missions:
        tgt = world.planet_by_id.get(mission.target_id)
        if tgt is None or tgt.owner == world.player:
            continue
        if mission.kind == "single" and mission.options:
            opt = mission.options[0]
            src = world.planet_by_id.get(opt.src_id)
            if src is not None and world.commit(src, tgt, opt.ships, moves):
                acted = True
        elif mission.kind == "group":
            src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
            if src is None:
                continue
            need = world.ships_needed_to_capture(src, tgt, world.my_total_ships)
            allow_partial = mode == StrategyMode.FINAL_DRAIN
            if build_grouped_wave(world, tgt, need, moves, allow_partial=allow_partial, sync=tgt.owner != -1) > 0:
                acted = True
        if len(moves) >= 12:
            break
    return acted


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
        if 0 < need <= surplus and world.commit(src, tgt, need, moves):
            acted = True
    return acted


# ── Mission Coordinator ───────────────────────────────────────────────────────

def compute_fleet_ratio(world):
    planet_ships = sum(int(p.ships) for p in world.my_planets)
    fleet_ships = sum(int(f.ships) for f in world.my_fleets)
    return fleet_ships / max(1, planet_ships + fleet_ships)


def generate_finish_capture_missions(world):
    proposals = []
    targets = [p for p in world.neutral_planets + world.enemy_planets
               if world.incoming_to_targets.get(p.id, 0) > 0]
    for tgt in sorted(targets, key=lambda p: world.target_need_now(p)):
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
    return proposals


def generate_expansion_missions(world, deadline):
    proposals = []
    if not world.neutral_planets or world.remaining < 35:
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
            if should_block_early_enemy_attack_4p(world, tgt):
                continue
            need = world.ships_needed_to_capture(src, tgt)
            if need <= 0 or need > surplus:
                continue
            angle, ok = world.aim(src, tgt, need)
            if not ok:
                continue
            eta = world.eta(src, tgt, need)
            score = tgt.production * 15.0 - d * 0.8 - need * 0.3
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
            priority=72.0 + score_val * 0.4,
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
        if should_block_early_enemy_attack_4p(world, tgt):
            continue
        pool_srcs = sorted(world.my_planets, key=lambda p: dp(p, tgt))[:6]
        if not pool_srcs:
            continue
        pool = sum(world.surplus(p) for p in pool_srcs)
        need = world.ships_needed_to_capture(min(pool_srcs, key=lambda p: dp(p, tgt)), tgt, pool)
        if need <= 0 or pool < need:
            continue
        source_etas = []
        for src in pool_srcs:
            av = world.surplus(src)
            if av < 8:
                continue
            send = min(av, max(8, int(av * 0.6)))
            angle, ok = world.aim(src, tgt, send)
            if not ok:
                continue
            eta = world.eta(src, tgt, send)
            source_etas.append((eta, src, send, angle))
        if len(source_etas) < 2:
            continue
        source_etas.sort()
        anchor_eta = source_etas[-1][0]
        synced = [(eta, src, send, angle) for eta, src, send, angle in source_etas
                  if anchor_eta - eta <= ETA_SYNC_WINDOW]
        if sum(s for _, _, s, _ in synced) < need:
            continue
        planned = [(src.id, send, angle, eta) for eta, src, send, angle in synced]
        base_score = score_target(world, synced[0][1], tgt, mode)
        proposals.append(MissionProposal(
            kind="SYNC_ATTACK",
            target_id=tgt.id,
            priority=62.0 + base_score * 0.25,
            required_ships=need,
            planned_sources=planned,
            eta_min=source_etas[0][0],
            eta_max=anchor_eta,
            reason=f"sync p{tgt.id} need={need} sources={len(synced)} spread={anchor_eta - source_etas[0][0]:.1f}",
        ))
    return proposals


def generate_anti_leader_missions(world):
    proposals = []
    if world.is_four_player and world.step < FOUR_P_ATTACK_STEP and world.neutral_planets:
        return proposals
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
    src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
    if src is None:
        return proposals
    need = world.ships_needed_to_capture(src, tgt, pool)
    if pool < need:
        world.add_debug(f"ANTI_LEADER_WAIT target=p{tgt.id} pool={pool} need={need}")
        return proposals
    send = min(world.surplus(src), need)
    if send <= 0:
        return proposals
    angle, ok = world.aim(src, tgt, send)
    if not ok:
        return proposals
    eta = world.eta(src, tgt, send)
    proposals.append(MissionProposal(
        kind="SYNC_ATTACK",
        target_id=tgt.id,
        priority=55.0 + tgt.production * 3,
        required_ships=need,
        planned_sources=[(src.id, send, angle, eta)],
        eta_min=eta,
        eta_max=eta,
        reason=f"anti_leader p{tgt.id} prod={tgt.production} leader={world.leader}",
    ))
    return proposals


def generate_collapse_missions(world, mode):
    proposals = []
    if mode not in (StrategyMode.COLLAPSE, StrategyMode.FORCE_WAVE):
        return proposals
    pool = sum(world.surplus(p) for p in world.my_planets)
    if pool < 20:
        return proposals
    targets = world.enemy_planets + (world.neutral_planets if world.step <= LATE_GAME_STEPS else [])
    for tgt in sorted(targets, key=lambda t: -(t.production * 10 - t.ships))[:4]:
        src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
        if src is None:
            continue
        need = world.ships_needed_to_capture(src, tgt, pool)
        if need <= 0:
            continue
        send = min(world.surplus(src), need)
        if send <= 0:
            continue
        angle, ok = world.aim(src, tgt, send)
        if not ok:
            continue
        eta = world.eta(src, tgt, send)
        proposals.append(MissionProposal(
            kind="COLLAPSE",
            target_id=tgt.id,
            priority=52.0 + tgt.production * 4,
            required_ships=need,
            planned_sources=[(src.id, send, angle, eta)],
            eta_min=eta,
            eta_max=eta,
            reason=f"collapse p{tgt.id} prod={tgt.production} need={need}",
        ))
    return proposals


def generate_final_drain_missions(world):
    proposals = []
    if world.step < FINAL_STEPS and world.remaining > 45:
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
                        kind="COLLAPSE",
                        target_id=tgt.id,
                        priority=40.0 + tgt.production,
                        required_ships=need,
                        planned_sources=[(src.id, need, angle, eta)],
                        eta_min=eta,
                        eta_max=eta,
                        reason=f"drain src=p{src.id} tgt=p{tgt.id}",
                    ))
                    break
    return proposals


def coordinate_missions(world, proposals, moves, fleet_ratio, deadline):
    """Select best non-conflicting missions. Enforce fleet ratio cap. Prevent scatter."""
    proposals.sort(key=lambda p: -p.priority)

    spent: dict = {}         # src_id -> ships committed this coordinator pass
    used_targets: set = set()
    defense_count = 0
    expansion_count = 0
    attack_count = 0
    breach_count = 0

    for prop in proposals:
        if time.perf_counter() > deadline:
            break
        if prop.target_id in used_targets:
            continue

        # Fleet ratio guards — BREACH_KILL bypasses hard cap (it's a kill, not speculative)
        if world.step < FINAL_STEPS:
            if fleet_ratio > FLEET_RATIO_HARD and prop.kind not in (
                "DEFEND", "REINFORCE", "FINISH_CAPTURE", "BREACH_KILL"
            ):
                world.add_debug(f"COORD_SKIP ratio={fleet_ratio:.2f} blocked {prop.kind}")
                continue
            if fleet_ratio > FLEET_RATIO_SOFT and prop.kind in ("LOCAL_STRIKE", "SYNC_ATTACK", "COLLAPSE"):
                continue

        # Per-kind limits: prevent scatter
        if prop.kind == "DEFEND" and defense_count >= len(world.my_planets):
            continue
        if prop.kind in ("CAPTURE_NEUTRAL", "FINISH_CAPTURE") and expansion_count >= 2:
            continue
        if prop.kind == "BREACH_KILL" and breach_count >= 1:
            continue
        if prop.kind in ("LOCAL_STRIKE", "SYNC_ATTACK") and attack_count >= 1:
            continue
        if prop.kind == "COLLAPSE" and attack_count >= 2:
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
        if total_send < 5 and prop.kind not in ("FINISH_CAPTURE", "DEFEND"):
            continue

        # Block speculative offensive moves from wave-reserved sources
        if _wave_reservation["target_id"] is not None and prop.kind in (
            "LOCAL_STRIKE", "SYNC_ATTACK", "COLLAPSE"
        ) and prop.target_id != _wave_reservation["target_id"]:
            rsv_set = set(_wave_reservation["source_ids"])
            if any(src_id in rsv_set for src_id, _, _, _ in prop.planned_sources):
                world.add_debug(
                    f"WAVE_RESERVE_BLOCK {prop.kind} p{prop.target_id} "
                    f"blocked for wave on p{_wave_reservation['target_id']}"
                )
                continue

        # Commit via world.commit() which handles re-aiming and tracking
        tgt = world.planet_by_id.get(prop.target_id)
        if tgt is None:
            continue
        committed_any = False
        for src_id, ships, _, _ in prop.planned_sources:
            src = world.planet_by_id.get(src_id)
            if src is None:
                continue
            available = int(src.ships) - world.committed.get(src_id, 0) - spent.get(src_id, 0)
            send = min(ships, available)
            if send <= 0:
                continue
            if world.commit(src, tgt, send, moves):
                spent[src_id] = spent.get(src_id, 0) + send
                committed_any = True

        if committed_any:
            world.wave_attempted = True
            used_targets.add(prop.target_id)
            world.add_debug(f"COORD {prop.kind} p{prop.target_id} send={total_send} {prop.reason}")
            if prop.kind == "DEFEND":
                defense_count += 1
            elif prop.kind in ("CAPTURE_NEUTRAL", "FINISH_CAPTURE"):
                expansion_count += 1
            elif prop.kind == "BREACH_KILL":
                breach_count += 1
            elif prop.kind in ("LOCAL_STRIKE", "SYNC_ATTACK", "COLLAPSE"):
                attack_count += 1


BREACH_KILL_DIST = 45.0      # forward planet is within this distance of an enemy
BREACH_KILL_STEP_MIN = 70    # activate after this step (or sooner if expansion is done)
BREACH_ETA_SYNC = 10         # max ETA spread for a breach grouped attack


def is_breach_kill_mode(world):
    """True when we have a forward planet inside enemy territory and can press the kill."""
    if not world.enemy_planets or not world.my_planets:
        return False
    if world.is_four_player and world.step < FOUR_P_ATTACK_STEP and world.neutral_planets:
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

            score = (
                int(tgt.production) * 120.0
                + max(0, 50 - int(tgt.ships)) * 2.0
                + max(0.0, BREACH_KILL_DIST - d_to_enemy) * 2.0
                - need * 0.5
            )
            if tgt.owner == world.leader:
                score += 60.0

            planned = [(src.id, send, angle, eta) for eta, src, send, angle in synced]
            proposals.append(MissionProposal(
                kind="BREACH_KILL",
                target_id=tgt.id,
                priority=85.0 + score * 0.1,
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


def wave_readiness_score(world, target, wait_turns=0):
    """Pool / need ratio, optionally projecting production over wait_turns."""
    src = min(world.my_planets, key=lambda p: dp(p, target), default=None)
    if src is None:
        return 0.0
    need = max(1, world.ships_needed_to_capture(src, target, world.my_total_ships))
    pool = sum(world.surplus(p) for p in world.my_planets)
    if wait_turns > 0:
        pool += sum(int(p.production) * wait_turns for p in world.my_planets)
    return pool / need


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


def is_trickle_attack(world, mission):
    """True if a mission would send far fewer ships than needed (weak partial pressure)."""
    tgt = world.planet_by_id.get(mission.target_id)
    if tgt is None or tgt.owner in (-1, world.player):
        return False
    src = min(world.my_planets, key=lambda p: dp(p, tgt), default=None)
    if src is None:
        return False
    need = world.ships_needed_to_capture(src, tgt, world.my_total_ships)
    total_send = sum(s for _, s, _, _ in mission.planned_sources)
    return need > 0 and total_send < need * MIN_WAVE_FRACTION


# ── Early Target Intelligence ─────────────────────────────────────────────────

def _connects_good_neutral(world, tgt, radius=28.0, min_prod=2):
    return any(
        n.id != tgt.id and dp(tgt, n) <= radius and int(n.production) >= min_prod
        for n in world.neutral_planets
    )


def _support_source_count(world, tgt, radius=42.0):
    return sum(1 for m in world.my_planets if dp(m, tgt) <= radius and world.surplus(m) > 0)


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

    my_nearest = d
    enemy_nearest = min((dp(e, tgt) for e in world.enemy_planets), default=999.0)

    score = prod * 55.0
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
        score += 20.0

    if _support_source_count(world, tgt) >= 2:
        score += 20.0

    score -= need * 1.4
    score -= eta * 5.0
    score -= d * 1.5

    if prod <= 1 and d > 30.0:
        score -= 60.0

    if enemy_nearest < my_nearest - 2.0:
        score -= 80.0

    if enemy_nearest < my_nearest - 10.0:
        score -= 100.0

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
    Return (True, new_tgt_id) if a significantly better opening target exists.
    Only active before step 60.
    """
    if world.step >= 60 or not world.neutral_planets or not world.my_planets:
        return False, None
    current_tgt = world.planet_by_id.get(current_target_id)
    if current_tgt is None:
        return False, None
    src = min(world.my_planets, key=lambda p: dp(p, current_tgt), default=None)
    if src is None:
        return False, None
    current_score = early_target_score(world, src, current_tgt)
    best_alt = None
    best_alt_score = current_score + 40.0
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


def first_capture_360(world, moves):
    """Steps 0-70 or <2 planets: strategic 360-degree opening capture.
    Uses early_target_score to pick the best local tempo/value/position target."""
    if world.step > 70 and len(world.my_planets) >= 2:
        return False
    if not world.neutral_planets:
        return False

    result = choose_best_opening_target(world)
    if result is None:
        return False

    src, tgt, need, angle = result
    moves.append([src.id, angle, need])
    world.committed[src.id] = world.committed.get(src.id, 0) + need
    world.incoming_to_targets[tgt.id] = world.incoming_to_targets.get(tgt.id, 0) + need
    world.offensive_ships += need
    world.wave_attempted = True
    d = dp(src, tgt)
    eta_val = world.eta(src, tgt, need)
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
        if av >= need:
            if world.commit(src, tgt, need, moves):
                world.add_debug(
                    f"EARLY_EXPANSION_360 src={src.id} tgt={tgt.id} score={score:.1f} d={d:.1f} eta={eta_val:.1f} need={need}"
                )
                return True
        else:
            pool_srcs = sorted(
                [p for p in world.my_planets if world.real_incoming_threat(p)["deficit"] <= 0],
                key=lambda p: dp(p, tgt),
            )[:3]
            if sum(early_surplus(world, p) for p in pool_srcs) >= need:
                sent = 0
                for psrc in pool_srcs:
                    if sent >= need:
                        break
                    sn = min(early_surplus(world, psrc), need - sent)
                    if sn < 3 and sent + sn < need:
                        continue
                    if world.commit(psrc, tgt, sn, moves):
                        sent += sn
                if sent >= need:
                    world.wave_attempted = True
                    world.add_debug(
                        f"EARLY_EXPANSION_360_GROUPED tgt={tgt.id} score={score:.1f} d={d:.1f} need={need} sent={sent}"
                    )
                    return True
    return False


def astar_route_to_target(world, goal_planet, deadline, max_depth=4):
    """True A* for specific enemy/leader/collapse targets only. Not used in early expansion."""
    goal = goal_planet

    def heuristic(node):
        d = dp(node, goal)
        time_est = d / fleet_speed(max(1, int(node.ships)))
        capture_cost = world.ships_needed_to_capture(node, goal, max(1, int(node.ships))) * 0.45
        pressure = world.enemy_pressure_near(goal, radius=25.0) * 0.3
        sun_pen = 12.0 if hits_sun(node.x, node.y, goal.x, goal.y) else 0.0
        overextension = max(0.0, 18.0 - world.nearest_enemy_distance(goal)) * 0.6
        return time_est * 3.5 + capture_cost + pressure + sun_pen + overextension

    queue: list = []
    best_cost: dict = {}
    for src in world.my_planets:
        h = heuristic(src)
        heapq.heappush(queue, (h, 0.0, src.id, src.id, ()))
        best_cost[(src.id, ())] = 0.0

    best_plan = None
    while queue and time.perf_counter() < deadline:
        _, g_cost, node_id, source_id, route = heapq.heappop(queue)
        node = world.planet_by_id.get(node_id)
        if node is None:
            continue
        if node_id == goal.id and route:
            if best_plan is None or g_cost < best_plan.cost:
                best_plan = RoutePlan(
                    target_id=goal.id,
                    first_target_id=route[0],
                    score=-g_cost,
                    cost=g_cost,
                    route=list(route),
                )
            continue
        if len(route) >= max_depth:
            continue
        for nxt in world.neutral_planets + [goal]:
            if nxt.id == node_id or nxt.id in route:
                continue
            edge = route_edge_cost(world, node, nxt)
            new_g = g_cost + edge
            new_route = route + (nxt.id,)
            key = (nxt.id, new_route)
            if new_g + 1e-6 >= best_cost.get(key, 1e18):
                continue
            best_cost[key] = new_g
            h = heuristic(nxt) if nxt.id != goal.id else 0.0
            heapq.heappush(queue, (new_g + h, new_g, nxt.id, source_id, new_route))

    return best_plan


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
    count = _missed_skipped.get(tgt.id, 0)
    if count < 3 or world.step >= 200:
        return False
    any_threatened = any(world.real_incoming_threat(p)["deficit"] > 0 for p in world.my_planets)
    if any_threatened:
        return False
    pool_srcs = sorted(
        [p for p in world.my_planets if world.real_incoming_threat(p)["deficit"] <= 0],
        key=lambda p: dp(p, tgt),
    )[:3]
    sent = 0
    for psrc in pool_srcs:
        if sent >= need:
            break
        sn = min(world.surplus(psrc), need - sent)
        if sn <= 0:
            continue
        if world.commit(psrc, tgt, sn, moves):
            sent += sn
    if sent >= need:
        _missed_skipped[tgt.id] = 0
        world.add_debug(f"FORCE_MISSED_NEUTRAL p{tgt.id} after {count} skips sent={sent}")
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


def agent(obs, config=None):
    start = time.perf_counter()
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    deadline = start + min(SOFT_DEADLINE, max(0.55, act_timeout * 0.82))
    world = WorldModel(obs)
    if not world.my_planets:
        return []

    if not hasattr(agent, "_last_meaningful") or world.step <= 1:
        agent._last_meaningful = {}
        _wave_reservation["target_id"] = None
    last_meaningful = agent._last_meaningful.get(world.player, world.step)
    idle_turns = world.step - last_meaningful
    mode = choose_strategy_mode(world, idle_turns)

    moves = []
    emergency_defense(world, moves)

    # Opening retarget check: before step 60, block more ships to a bad in-flight target
    if world.step < 60 and world.step > 0:
        for tgt_id, incoming in list(world.incoming_to_targets.items()):
            if incoming > 0:
                retarget, _ = should_retarget_opening(world, tgt_id)
                if retarget:
                    # Zero out further commitment to this target so no new ships go there
                    world.incoming_to_targets[tgt_id] = max(incoming, 999)

    # Phase 1: Strategic 360-degree opening capture (steps 0-70 or <2 planets)
    if time.perf_counter() < deadline:
        if first_capture_360(world, moves):
            if world.step < 60 or len(world.my_planets) < 2:
                if world.offensive_ships >= 15 or world.wave_attempted:
                    agent._last_meaningful[world.player] = world.step
                if DEBUG:
                    for event in world.debug_events:
                        print(event)
                return moves

    # Phase 2: Early nearest expansion (steps 0-140 or <4 planets)
    if time.perf_counter() < deadline:
        if early_nearest_expansion_360(world, moves):
            if world.step < 100 or len(world.my_planets) < 3:
                if world.offensive_ships >= 15 or world.wave_attempted:
                    agent._last_meaningful[world.player] = world.step
                if DEBUG:
                    for event in world.debug_events:
                        print(event)
                return moves

    # Phase 3+: Mission Coordinator
    fleet_ratio = compute_fleet_ratio(world)
    world.add_debug(f"fleet_ratio={fleet_ratio:.2f} mode={mode} idle={idle_turns}")

    proposals: list = []
    if time.perf_counter() < deadline:
        proposals += generate_finish_capture_missions(world)
    if time.perf_counter() < deadline:
        proposals += generate_breach_kill_missions(world)
    if time.perf_counter() < deadline:
        proposals += generate_expansion_missions(world, deadline)
    if time.perf_counter() < deadline:
        proposals += generate_local_strike_missions(world)
    if time.perf_counter() < deadline:
        proposals += generate_sync_attack_missions(world, mode, deadline)
    if time.perf_counter() < deadline:
        proposals += generate_anti_leader_missions(world)
    if time.perf_counter() < deadline:
        proposals += generate_collapse_missions(world, mode)
    if time.perf_counter() < deadline:
        proposals += generate_final_drain_missions(world)

    # Inject organized wave if reservation is ready or overdue
    if _wave_reservation["target_id"] is not None and time.perf_counter() < deadline:
        rsv_tgt = world.planet_by_id.get(_wave_reservation["target_id"])
        if rsv_tgt is None or rsv_tgt.owner == world.player:
            _wave_reservation["target_id"] = None
        else:
            rsv_pool = sum(world.surplus(p) for p in world.my_planets
                          if p.id in set(_wave_reservation["source_ids"]))
            rsv_need = _wave_reservation["required_ships"]
            force_launch = world.step >= _wave_reservation["launch_by_step"]
            if rsv_pool >= rsv_need or force_launch:
                wave_prop = generate_organized_wave_mission(world, rsv_tgt)
                if wave_prop is not None:
                    proposals.insert(0, wave_prop)
                _wave_reservation["target_id"] = None

    if proposals and time.perf_counter() < deadline:
        coordinate_missions(world, proposals, moves, fleet_ratio, deadline)

    # Anti-stall fallback — only when coordinator produced nothing
    if not moves and time.perf_counter() < deadline:
        force_action_if_stalling(world, moves, idle_turns, deadline)

    if not moves:
        fallback_tempo(world, moves)

    missed_opportunity_detector(world, moves)

    if world.offensive_ships >= 15 or world.wave_attempted:
        agent._last_meaningful[world.player] = world.step
    if DEBUG:
        for event in world.debug_events:
            print(event)
    return moves
