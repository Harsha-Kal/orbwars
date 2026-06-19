"""
Orbit Wars – Quadrant A* Agent
================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

Architecture:
  - Map divided into 4 quadrants (NW/NE/SW/SE relative to sun)
  - Weighted A* for traffic-aware path planning with intermediate checkpoints
  - Goal 1: Fastest occupiable planet – pool resources like Google Maps routing
             with traffic; intermediate planets act as refuelling checkpoints
  - Goal 2: Opponent occupation via Coordinated Encirclement Attack –
             strike the same enemy from 2+ quadrant fronts simultaneously
  - Emergency defense: reinforce endangered high-value planets first
  - Attacking in multiples of 5 ships, minimum 10 (enforced everywhere)
"""

import math
import heapq
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# ── Physics / game constants ──────────────────────────────────────────────────
SUN_X, SUN_Y   = 50.0, 50.0
SUN_R          = 10.0
MAX_SPEED      = 6.0
ROTATION_LIMIT = 50.0
TOTAL_STEPS    = 500
HIT_MARGIN     = 3.0
INCOMING_T     = 0.85

# ── Packet discipline (enforced on every launch) ──────────────────────────────
MIN_SEND_SHIPS   = 10
SEND_GRANULARITY = 5   # every fleet must be a multiple of this

DEBUG = False

# ── Quadrant labels (0=SE, 1=SW, 2=NW, 3=NE) ─────────────────────────────────
QUAD_SE, QUAD_SW, QUAD_NW, QUAD_NE = 0, 1, 2, 3
QUAD_NAMES = {0: "SE", 1: "SW", 2: "NW", 3: "NE"}

# ── A* tuning ─────────────────────────────────────────────────────────────────
ASTAR_MAX_HOPS          = 4     # max intermediate planets considered per route
ASTAR_NEIGHBOR_LIMIT    = 12    # planets checked per expansion step
ASTAR_THREAT_RADIUS     = 28.0  # corridor-threat sampling radius
ASTAR_FLEET_THREAT_RAD  = 22.0
ASTAR_MAX_TRAFFIC_MULT  = 3.5   # cap on traffic slow-down factor
ASTAR_FRIENDLY_DISCOUNT = 0.55  # travel through friendly territory is faster

# ── Expansion engine ──────────────────────────────────────────────────────────
EXPAND_MAX_SOURCES   = 8     # planets pooled per expansion target
EXPAND_MAX_MISSIONS  = 6     # expansion missions per turn
EXPAND_RACE_MARGIN   = 1.15  # my ETA must be this much better than enemy ETA
EARLY_GAME_STEPS     = 80    # steps during which early reserve discounts apply
EARLY_RESERVE_MULT   = 0.45  # reserve multiplier during early land-grab phase
EARLY_PROD_WEIGHT    = 3.5   # production score multiplier in early expansion scorer

# ── Pressure / encirclement engine ───────────────────────────────────────────
PRESSURE_MAX_MISSIONS   = 4
PRESSURE_FRONTS         = 2    # attack from this many quadrant directions
PRESSURE_MIN_PROD       = 0    # minimum enemy production worth hitting
PRESSURE_ENCIRCLE_RANGE = 50.0 # radius to find flanking sources

# ── Defense ───────────────────────────────────────────────────────────────────
DEFENSE_ETA_HORIZON  = 20    # only count threats arriving within this many turns
DEFENSE_SAVE_HELPERS = 4     # max planets pulled for a single save

# ── Planet roles ──────────────────────────────────────────────────────────────
ROLE_LAUNCHPAD = "LAUNCHPAD"
ROLE_BRIDGE    = "BRIDGE"
ROLE_STORAGE   = "STORAGE"


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY & PHYSICS  (unchanged from spec)
# ═══════════════════════════════════════════════════════════════════════════════

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
    return d_val / fleet_speed(max(1, int(ships)))

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

def predict_pos(p, ang_vel, turns):
    """Predict where planet p will be after `turns` steps."""
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= ROTATION_LIMIT:
        return p.x, p.y
    cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
    new_ang = cur_ang + ang_vel * turns
    return SUN_X + orb_r * math.cos(new_ang), SUN_Y + orb_r * math.sin(new_ang)

def is_idle(p):
    return dist(p.x, p.y, SUN_X, SUN_Y) + p.radius >= ROTATION_LIMIT

def compute_aim(src, tgt, ships, ang_vel):
    """
    Iterative predictive aim.
    Returns (angle, hit_ok, eta).  hit_ok is False when the aimed point
    misses a rotating target by > HIT_MARGIN.
    """
    ships = max(1, int(ships))
    tx, ty = tgt.x, tgt.y
    eta = travel_turns(dp(src, tgt), ships)
    for _ in range(12):
        tx, ty = predict_pos(tgt, ang_vel, eta)
        eta = travel_turns(dist(src.x, src.y, tx, ty), ships)
    angle = safe_angle(src.x, src.y, tx, ty)
    if is_idle(tgt):
        return angle, True, eta
    d_aimed = dist(src.x, src.y, tx, ty)
    aimed_x = src.x + d_aimed * math.cos(angle)
    aimed_y = src.y + d_aimed * math.sin(angle)
    ok = dist(aimed_x, aimed_y, tx, ty) <= float(tgt.radius) + HIT_MARGIN
    return angle, ok, eta


# ═══════════════════════════════════════════════════════════════════════════════
#  PACKET HELPERS  (multiples of 5, minimum 10)
# ═══════════════════════════════════════════════════════════════════════════════

def pkt_up(ships):
    """Round up to nearest valid packet size (min 10, multiple of 5)."""
    ships = max(0, int(math.ceil(ships)))
    if ships <= 0:
        return 0
    return max(MIN_SEND_SHIPS, ((ships + SEND_GRANULARITY - 1) // SEND_GRANULARITY) * SEND_GRANULARITY)

def pkt_down(ships):
    """Round down to nearest valid packet size (min 10, multiple of 5). 0 if < 10."""
    ships = int(max(0, ships))
    ships -= ships % SEND_GRANULARITY
    return ships if ships >= MIN_SEND_SHIPS else 0

def valid_pkt(ships):
    ships = int(ships)
    return ships >= MIN_SEND_SHIPS and ships % SEND_GRANULARITY == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  QUADRANT MAP
#  The map is divided into 4 quadrants using the sun (50, 50) as origin.
#  SE = right+lower, SW = left+lower, NW = left+upper, NE = right+upper
# ═══════════════════════════════════════════════════════════════════════════════

def quadrant_of(x, y):
    right = x >= SUN_X
    lower = y >= SUN_Y
    if right and lower:
        return QUAD_SE
    if not right and lower:
        return QUAD_SW
    if not right and not lower:
        return QUAD_NW
    return QUAD_NE


class QuadrantMap:
    """
    Partitions all planets into 4 quadrants.
    Tracks per-quadrant ownership to drive directional expansion.
    """

    def __init__(self, snap):
        self.snap = snap
        self.quads        = {q: [] for q in range(4)}
        self.my_quads     = {q: [] for q in range(4)}
        self.enemy_quads  = {q: [] for q in range(4)}
        self.neutral_quads = {q: [] for q in range(4)}
        self.planet_quad  = {}  # planet_id -> quadrant

        for p in snap.normal_planets:
            q = quadrant_of(p.x, p.y)
            self.planet_quad[p.id] = q
            self.quads[q].append(p)
            if p.owner == snap.player:
                self.my_quads[q].append(p)
            elif p.owner == -1:
                self.neutral_quads[q].append(p)
            else:
                self.enemy_quads[q].append(p)

        # control[q] in [-1, +1]: +1 = fully mine, -1 = fully enemy
        self.control = {}
        for q in range(4):
            total = len(self.quads[q])
            if total == 0:
                self.control[q] = 0.0
            else:
                self.control[q] = (len(self.my_quads[q]) - len(self.enemy_quads[q])) / total

        # My start quadrant = quadrant of my first planet
        self.my_quad = (
            quadrant_of(snap.my_planets[0].x, snap.my_planets[0].y)
            if snap.my_planets else QUAD_SE
        )

    def quad_of(self, p):
        return self.planet_quad.get(p.id, quadrant_of(p.x, p.y))

    def adjacent_quads(self, q):
        return [(q + 1) % 4, (q - 1) % 4]

    def opposite_quad(self, q):
        return (q + 2) % 4

    def pressure_score_for_quad(self, q):
        """
        How urgently should we expand into quadrant q?
        Higher = we need to act there more.
        """
        ctrl = self.control.get(q, 0.0)
        n_neutral = len(self.neutral_quads[q])
        n_enemy   = len(self.enemy_quads[q])
        # Untouched quadrants (neutral-heavy) and contested ones score high
        return n_neutral * 2.0 + n_enemy * 1.5 - max(0.0, ctrl) * 10.0


# ═══════════════════════════════════════════════════════════════════════════════
#  GAME SNAPSHOT  (lightweight per-turn world model)
# ═══════════════════════════════════════════════════════════════════════════════

def _read(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class GameSnapshot:
    """
    Builds the world model for one turn.
    Tracks committed ships so engines don't double-spend.
    """

    def __init__(self, obs):
        self.player  = int(_read(obs, "player", 0))
        self.step    = int(_read(obs, "step", 0))
        self.ang_vel = float(_read(obs, "angular_velocity",
                                    _read(obs, "angularVelocity", 0.033)))
        self.planets   = [Planet(*p) for p in (_read(obs, "planets", []) or [])]
        self.fleets    = [Fleet(*f)  for f in (_read(obs, "fleets",  []) or [])]
        self.comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

        self.planet_by_id   = {p.id: p for p in self.planets}
        self.normal_planets = [p for p in self.planets if p.id not in self.comet_ids]
        self.my_planets     = [p for p in self.normal_planets if p.owner == self.player]
        self.enemy_planets  = [p for p in self.normal_planets if p.owner not in (-1, self.player)]
        self.neutral_planets = [p for p in self.normal_planets if p.owner == -1]
        self.my_fleets      = [f for f in self.fleets if f.owner == self.player]
        self.enemy_fleets   = [f for f in self.fleets if f.owner != self.player]

        self.my_prod    = sum(int(p.production) for p in self.my_planets)
        self.enemy_prod = sum(int(p.production) for p in self.enemy_planets)
        self.remaining  = max(1, TOTAL_STEPS - self.step)

        # Ships already scheduled for launch this turn (prevents double-spend)
        self.committed: dict = {}

        # Per-planet incoming fleet map
        self.incoming: dict = {}  # pid -> {friendly, enemy, arrivals}
        self._build_incoming()

        # Quadrant map
        self.qmap = QuadrantMap(self)

        # Planet roles (derived from radius distribution)
        self._roles = self._classify_roles()

    # ── internals ────────────────────────────────────────────────────────────

    def _classify_roles(self):
        if not self.normal_planets:
            return {}
        radii = sorted(float(p.radius) for p in self.normal_planets)
        small_cut = radii[max(0, len(radii) // 3 - 1)]
        large_cut = radii[min(len(radii) - 1, (len(radii) * 2) // 3)]
        roles = {}
        for p in self.normal_planets:
            if int(p.production) >= 4 or float(p.radius) >= large_cut:
                roles[p.id] = ROLE_LAUNCHPAD
            elif int(p.production) >= 2 or float(p.radius) >= small_cut:
                roles[p.id] = ROLE_BRIDGE
            else:
                roles[p.id] = ROLE_STORAGE
        return roles

    def _fleet_target(self, fl):
        """Match a fleet to its most likely target planet."""
        best, best_diff = None, 0.95
        for p in self.normal_planets:
            d    = dist(fl.x, fl.y, p.x, p.y)
            eta  = travel_turns(d, max(1, int(fl.ships)))
            tx, ty = predict_pos(p, self.ang_vel, eta)
            ea   = math.atan2(ty - fl.y, tx - fl.x)
            diff = abs(math.atan2(
                math.sin(float(fl.angle) - ea),
                math.cos(float(fl.angle) - ea)
            ))
            if diff < best_diff:
                best, best_diff = p, diff
        return best

    def _build_incoming(self):
        for p in self.normal_planets:
            self.incoming[p.id] = {"friendly": 0, "enemy": 0, "arrivals": []}
        for fl in self.fleets:
            tgt = self._fleet_target(fl)
            if tgt is None or tgt.id in self.comet_ids:
                continue
            eta = travel_turns(dist(fl.x, fl.y, tgt.x, tgt.y), max(1, int(fl.ships)))
            rec = self.incoming.setdefault(tgt.id, {"friendly": 0, "enemy": 0, "arrivals": []})
            rec["arrivals"].append((eta, int(fl.owner), int(fl.ships)))
            if fl.owner == self.player:
                rec["friendly"] += int(fl.ships)
            else:
                rec["enemy"]    += int(fl.ships)

    # ── public API ────────────────────────────────────────────────────────────

    def role(self, p):
        return self._roles.get(p.id, ROLE_STORAGE)

    def surplus(self, p):
        """
        Available ships on p that can be launched without losing the planet.
        Reserve scales with role and local enemy threat.
        In the early land-grab phase (< EARLY_GAME_STEPS) reserves are cut by
        EARLY_RESERVE_MULT so more ships flow into expansion captures.
        """
        inc = self.incoming.get(p.id, {})
        enemy_near = sum(
            sh for eta, own, sh in inc.get("arrivals", [])
            if own != self.player and eta <= DEFENSE_ETA_HORIZON
        )
        r = self.role(p)
        if r == ROLE_LAUNCHPAD:
            base_reserve = max(20, int(p.ships) // 5)
        elif r == ROLE_BRIDGE:
            base_reserve = max(12, int(p.ships) // 6)
        else:
            base_reserve = max(6, int(p.ships) // 8)
        # Early game: discount reserves to fuel aggressive expansion
        if self.step < EARLY_GAME_STEPS and enemy_near == 0:
            base_reserve = max(5, int(base_reserve * EARLY_RESERVE_MULT))
        reserve = max(base_reserve, enemy_near + 5)
        return max(0, int(p.ships) - self.committed.get(p.id, 0) - reserve)

    def capture_need(self, src, tgt, ships_hint=40):
        """
        Minimum ships needed from src to flip tgt.
        Accounts for tgt's current garrison, production growth during transit,
        and existing friendly/enemy fleets already inbound.
        """
        inc = self.incoming.get(tgt.id, {})
        eta, _, _ = compute_aim(src, tgt, ships_hint, self.ang_vel)
        eta = max(1.0, eta)
        # Enemy planet: grows during transit
        growth = (
            int(tgt.production) * int(math.ceil(eta))
            if tgt.owner not in (-1, self.player)
            else 0
        )
        base = int(tgt.ships) + growth + inc.get("enemy", 0) - inc.get("friendly", 0) + 1
        return max(0, base)

    def projected_owner(self, planet, horizon, extra=()):
        """
        Simulate ownership after `horizon` turns.
        extra = [(eta, owner, ships), ...] for planned arrivals.
        """
        horizon = max(1, min(120, int(math.ceil(horizon))))
        owner = int(planet.owner)
        ships = int(planet.ships)
        inc   = self.incoming.get(planet.id, {})
        buckets: dict = {}
        for eta, who, cnt in list(inc.get("arrivals", [])) + list(extra):
            t = max(1, min(horizon, int(math.ceil(eta))))
            buckets.setdefault(t, []).append((who, cnt))
        for t in range(1, horizon + 1):
            if owner != -1:
                ships += int(planet.production)
            for who, cnt in buckets.get(t, []):
                forces = {owner: max(0, ships)}
                forces[who] = forces.get(who, 0) + max(0, cnt)
                ranked = sorted(forces.items(), key=lambda x: x[1], reverse=True)
                if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
                    owner, ships = -1, 0
                else:
                    owner = ranked[0][0]
                    ships = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0)
        return owner, ships

    def commit(self, src_id, ships):
        self.committed[src_id] = self.committed.get(src_id, 0) + int(ships)

    def dies_if_sent(self, p, send):
        """True if sending `send` ships would leave p indefensible."""
        inc = self.incoming.get(p.id, {})
        enemy_soon = sum(sh for eta, own, sh in inc.get("arrivals", [])
                         if own != self.player and eta <= 10)
        if enemy_soon <= 0:
            return False
        remaining_ships = int(p.ships) - self.committed.get(p.id, 0) - int(send)
        return remaining_ships <= enemy_soon


# ═══════════════════════════════════════════════════════════════════════════════
#  WEIGHTED A*  (traffic-aware path planner)
#
#  Planets are graph nodes.  The edge weight between two planets is:
#    base_travel_time × traffic_multiplier
#
#  traffic_multiplier:
#    • 1.0  – clear corridor (no enemies nearby)
#    • up to ASTAR_MAX_TRAFFIC_MULT  – corridor crosses enemy territory
#    • discounted to ASTAR_FRIENDLY_DISCOUNT  – both endpoints are mine
#
#  Intermediate planets ("checkpoints") are real planets we may capture on
#  the way to the final target – exactly like Google Maps via-points.
# ═══════════════════════════════════════════════════════════════════════════════

class WeightedAStarPlanner:

    def __init__(self, snap: GameSnapshot):
        self.snap = snap

    # ── edge weight ──────────────────────────────────────────────────────────

    def _edge_cost(self, pa, pb, ships=30):
        """Traffic-weighted travel time from pa to pb."""
        d         = dp(pa, pb)
        base_time = travel_turns(d, ships)

        mid_x = (pa.x + pb.x) / 2
        mid_y = (pa.y + pb.y) / 2

        threat = 0.0
        for ep in self.snap.enemy_planets:
            cd = dist(mid_x, mid_y, ep.x, ep.y)
            if cd < ASTAR_THREAT_RADIUS:
                threat += int(ep.ships) / max(1.0, cd)
        for ef in self.snap.enemy_fleets:
            cd = dist(mid_x, mid_y, ef.x, ef.y)
            if cd < ASTAR_FLEET_THREAT_RAD:
                threat += int(ef.ships) / max(1.0, cd)

        traffic = 1.0 + min(ASTAR_MAX_TRAFFIC_MULT - 1.0, threat / 60.0)

        # Safe corridor: both endpoints are friendly
        if pa.owner == self.snap.player and pb.owner == self.snap.player:
            traffic = max(ASTAR_FRIENDLY_DISCOUNT, traffic * 0.5)

        return base_time * traffic

    def _heuristic(self, p, goal):
        return dp(p, goal) / MAX_SPEED

    # ── path finding ─────────────────────────────────────────────────────────

    def find_path(self, start_id, goal_id):
        """
        A* from start_id to goal_id through intermediate normal planets.
        Returns a list of planet IDs (path), starting at start_id.
        Falls back to [start_id, goal_id] if A* finds no improvement.
        """
        snap = self.snap
        pb   = snap.planet_by_id
        start = pb.get(start_id)
        goal  = pb.get(goal_id)
        if start is None or goal is None:
            return [start_id, goal_id]

        # (f_score, g_score, planet_id, path_list)
        open_heap = [(self._heuristic(start, goal), 0.0, start_id, [start_id])]
        best_g: dict = {}

        while open_heap:
            f, g, cur_id, path = heapq.heappop(open_heap)
            if cur_id in best_g and best_g[cur_id] <= g:
                continue
            best_g[cur_id] = g
            if cur_id == goal_id:
                return path
            if len(path) - 1 >= ASTAR_MAX_HOPS:
                continue
            cur_p = pb.get(cur_id)
            if cur_p is None:
                continue
            # Nearest planets as neighbors
            neighbors = sorted(
                (p for p in snap.normal_planets if p.id not in snap.comet_ids and p.id != cur_id),
                key=lambda p: dp(cur_p, p)
            )[:ASTAR_NEIGHBOR_LIMIT]
            for nb in neighbors:
                edge  = self._edge_cost(cur_p, nb)
                new_g = g + edge
                if nb.id in best_g and best_g[nb.id] <= new_g:
                    continue
                new_f = new_g + self._heuristic(nb, goal)
                heapq.heappush(open_heap, (new_f, new_g, nb.id, path + [nb.id]))

        return [start_id, goal_id]

    def route_cost(self, path):
        """Sum of edge costs along a path (list of planet IDs)."""
        total = 0.0
        pb = self.snap.planet_by_id
        for i in range(len(path) - 1):
            pa = pb.get(path[i])
            pb_ = pb.get(path[i + 1])
            if pa and pb_:
                total += self._edge_cost(pa, pb_)
        return total

    def best_routes(self, sources, target):
        """
        Compute (source, path, cost) for each source planet to target.
        Returned sorted cheapest-first (best route first).
        """
        results = []
        for src in sources:
            path = self.find_path(src.id, target.id)
            cost = self.route_cost(path)
            results.append((src, path, cost))
        results.sort(key=lambda x: x[2])
        return results

    def first_checkpoint(self, path):
        """
        Return the first unowned planet in path after the source.
        This is the planet we should actually strike next (the checkpoint).
        If the path goes through only my planets, the final target is returned.
        """
        snap = self.snap
        pb   = snap.planet_by_id
        for pid in path[1:]:
            p = pb.get(pid)
            if p and p.owner != snap.player and p.id not in snap.comet_ids:
                return p
        return pb.get(path[-1])


# ═══════════════════════════════════════════════════════════════════════════════
#  GOAL 1 – EXPANSION ENGINE
#  "Fastest occupiable planet with available resources"
#
#  Scoring = (production × remaining_turns) / A*_route_cost
#  Like Google Maps: prefers fast routes (low traffic), can route through
#  intermediate neutral planets as stepping stones (checkpoints).
#
#  Directional pressure: we score planets in under-controlled quadrants higher,
#  ensuring we expand in all 4 directions, not just toward one corner.
# ═══════════════════════════════════════════════════════════════════════════════

class ExpansionEngine:

    def __init__(self, snap: GameSnapshot, astar: WeightedAStarPlanner):
        self.snap  = snap
        self.astar = astar

    def _pool_for_target(self, target):
        """Nearby my-planets with surplus available, sorted by distance."""
        snap = self.snap
        return sorted(
            [p for p in snap.my_planets
             if dp(p, target) <= 110.0 and snap.surplus(p) >= MIN_SEND_SHIPS],
            key=lambda p: dp(p, target)
        )[:EXPAND_MAX_SOURCES]

    def _expansion_score(self, target, route_cost, my_fastest_eta, enemy_fastest_eta):
        """
        Value of capturing this target, discounted by route cost.
        Higher is better.
        """
        snap = self.snap
        prod  = max(0, int(target.production))
        ships = int(target.ships)

        # Race factor: if we're clearly losing the race, skip
        if my_fastest_eta > enemy_fastest_eta * EXPAND_RACE_MARGIN:
            race = 0.15  # still consider it but heavily discounted
        else:
            race = 1.0

        turns_yielding = max(1, snap.remaining - my_fastest_eta)
        # Early game: heavily weight production to grab high-prod planets first
        prod_mult = EARLY_PROD_WEIGHT if snap.step < EARLY_GAME_STEPS else 1.0
        value = prod * prod_mult * turns_yielding * race

        # Quadrant pressure bonus: expand into neglected quadrants
        q = self.snap.qmap.quad_of(target)
        quad_urgency = self.snap.qmap.pressure_score_for_quad(q)
        value += quad_urgency * 8.0

        # Cheaper / closer = higher score
        score = value / max(1.0, route_cost)

        # Penalize far low-production storage planets (relaxed early to avoid
        # anchoring on nearby low-value planets at the cost of good far ones)
        if prod <= 1 and route_cost > 30.0 and snap.step >= EARLY_GAME_STEPS:
            score *= 0.25

        return score

    def _build_plan(self, actual_target, sources):
        """
        Allocate ships from sources to capture actual_target.
        Returns [(src, ships, angle, eta), ...] or None.
        """
        snap = self.snap
        if not sources:
            return None

        need = snap.capture_need(sources[0], actual_target)
        need = pkt_up(max(need, MIN_SEND_SHIPS))

        plan     = []
        leftover = need
        for src in sources:
            if leftover <= 0:
                break
            avail = pkt_down(snap.surplus(src))
            if avail <= 0:
                continue
            send = min(avail, pkt_up(leftover))
            send = pkt_down(send)
            if not valid_pkt(send):
                continue
            if snap.dies_if_sent(src, send):
                continue
            angle, ok, eta = compute_aim(src, actual_target, send, snap.ang_vel)
            if not ok:
                continue
            plan.append((src, send, angle, eta))
            leftover -= send

        if not plan:
            return None
        total = sum(s for _, s, _, _ in plan)
        if total < need:
            return None

        # Verify the combined fleet will actually flip ownership
        extra = tuple(
            (eta, snap.player, ships)
            for _, ships, _, eta in plan
        )
        max_eta = max(eta for _, _, _, eta in plan)
        owner_after, _ = snap.projected_owner(actual_target, max_eta, extra)
        if owner_after != snap.player:
            return None

        return plan

    def missions(self, max_missions=EXPAND_MAX_MISSIONS):
        """
        Return up to max_missions expansion plans, sorted best-first.
        Each plan is [(src_planet, ships, angle, eta), ...].
        """
        snap = self.snap
        if not snap.my_planets:
            return []

        candidates = snap.neutral_planets + snap.enemy_planets
        candidates = [c for c in candidates if c.id not in snap.comet_ids]

        scored = []
        for target in candidates:
            pool = self._pool_for_target(target)
            if not pool:
                continue

            nearest_src = pool[0]
            path        = self.astar.find_path(nearest_src.id, target.id)
            route_cost  = self.astar.route_cost(path)

            # Actual immediate target = first checkpoint on path
            actual = self.astar.first_checkpoint(path)
            if actual is None:
                actual = target

            my_eta = route_cost
            enemy_eta = min(
                (travel_turns(dp(ep, target), 30) for ep in snap.enemy_planets),
                default=999.0
            )
            score = self._expansion_score(target, route_cost, my_eta, enemy_eta)
            if score <= 0:
                continue

            pool_for_actual = self._pool_for_target(actual)
            scored.append((score, actual, pool_for_actual, target.id))

        scored.sort(reverse=True, key=lambda x: x[0])

        missions      = []
        taken_targets = set()
        for score, actual_target, pool, orig_id in scored:
            if actual_target.id in taken_targets or orig_id in taken_targets:
                continue
            if len(missions) >= max_missions:
                break
            plan = self._build_plan(actual_target, pool)
            if plan:
                missions.append(plan)
                taken_targets.add(actual_target.id)
                taken_targets.add(orig_id)
                for src, ships, _, _ in plan:
                    snap.commit(src.id, ships)

        return missions


# ═══════════════════════════════════════════════════════════════════════════════
#  GOAL 2 – PRESSURE ENGINE  (Coordinated Encirclement Attack)
#
#  Attack Strategy: Encirclement
#  ─────────────────────────────
#  1. Score each enemy planet by its exposure: isolated planets (far from
#     enemy support, surrounded by my territory) score highest.
#  2. For the top candidate, identify attack fronts from 2+ different quadrant
#     directions (like a military pincer movement).
#  3. Allocate ships from each front proportionally to its available surplus.
#  4. All ships aim at the same target, launched immediately.
#
#  The 2-front simultaneous strike prevents the enemy from funnelling
#  reinforcements from a single direction and guarantees our combined fleet
#  is larger than any single-direction counter.
# ═══════════════════════════════════════════════════════════════════════════════

class PressureEngine:

    def __init__(self, snap: GameSnapshot, astar: WeightedAStarPlanner):
        self.snap  = snap
        self.astar = astar

    def _exposure_score(self, ep):
        """
        How exposed/attackable is enemy planet ep?
        Higher = better encirclement candidate.
        """
        snap = self.snap
        nearest_my   = min((dp(mp, ep) for mp in snap.my_planets), default=999.0)
        nearest_supp = min(
            (dp(oe, ep) for oe in snap.enemy_planets if oe.id != ep.id),
            default=999.0
        )
        my_coverage  = sum(1 for mp in snap.my_planets if dp(mp, ep) <= 50.0)
        prod  = int(ep.production)
        ships = int(ep.ships)

        score  = prod * 45.0
        score += min(60.0, nearest_supp * 1.2)    # isolated from support → easier
        score += my_coverage * 18.0               # surrounded by my planets → easier
        score += max(0.0, 60.0 - nearest_my) * 2.0  # close to me → fast strike
        score -= ships * 0.6                       # fewer defenders → easier
        score -= max(0.0, nearest_my - 40.0) * 3.0  # too far → penalty

        # Bonus if enemy ships are depleted (recently fought elsewhere)
        expected_ships = prod * max(1, snap.step // 10)
        if ships < expected_ships * 0.5:
            score += 30.0

        return score

    def _quadrant_fronts(self, target):
        """
        Group my surplus planets by quadrant.
        Returns up to PRESSURE_FRONTS groups: [(quad_id, [planets]), ...]
        sorted by total surplus descending.
        """
        snap = self.snap
        groups: dict = {}
        for mp in snap.my_planets:
            if snap.surplus(mp) < MIN_SEND_SHIPS:
                continue
            if dp(mp, target) > PRESSURE_ENCIRCLE_RANGE:
                continue
            q = snap.qmap.quad_of(mp)
            groups.setdefault(q, []).append(mp)

        fronts = [
            (q, sorted(ps, key=lambda p: dp(p, target)))
            for q, ps in groups.items()
        ]
        # Sort fronts by total available surplus (strongest front first)
        fronts.sort(
            key=lambda f: -sum(snap.surplus(p) for p in f[1])
        )
        return fronts[:PRESSURE_FRONTS]

    def _build_encirclement(self, target, fronts):
        """
        Allocate ships across fronts to capture target.
        Each front contributes proportionally to its strength.
        Returns [(src, ships, angle, eta), ...] or None.
        """
        snap = self.snap
        need = snap.capture_need(
            min(snap.my_planets, key=lambda p: dp(p, target)),
            target
        )
        # Attack with extra weight to account for production growth in transit
        need = pkt_up(max(need, MIN_SEND_SHIPS * 2))

        total_strength = sum(
            sum(snap.surplus(p) for p in ps)
            for _, ps in fronts
        )
        if total_strength < need:
            return None  # can't fund the attack

        plan      = []
        remaining = need

        for q, sources in fronts:
            if remaining <= 0:
                break
            front_pool = sum(snap.surplus(p) for p in sources)
            # Proportional share of the need from this front
            front_share = pkt_up(int(math.ceil(need * front_pool / max(1, total_strength))))
            front_left  = front_share

            for src in sources:
                if front_left <= 0:
                    break
                avail = pkt_down(snap.surplus(src))
                if avail <= 0:
                    continue
                send = min(avail, pkt_up(front_left))
                send = pkt_down(send)
                if not valid_pkt(send):
                    continue
                if snap.dies_if_sent(src, send):
                    continue
                angle, ok, eta = compute_aim(src, target, send, snap.ang_vel)
                if not ok:
                    continue
                plan.append((src, send, angle, eta))
                front_left -= send
                remaining  -= send

        if not plan:
            return None
        total = sum(s for _, s, _, _ in plan)
        if total < need:
            return None

        # Ownership verification
        extra = tuple((eta, snap.player, ships) for _, ships, _, eta in plan)
        max_eta = max(eta for _, _, _, eta in plan)
        owner_after, _ = snap.projected_owner(target, max_eta, extra)
        if owner_after != snap.player:
            return None

        return plan

    def missions(self, max_missions=PRESSURE_MAX_MISSIONS):
        """
        Return up to max_missions encirclement attack plans.
        Each plan is [(src_planet, ships, angle, eta), ...].
        """
        snap = self.snap
        if not snap.my_planets or not snap.enemy_planets:
            return []

        # Only attack when we have enough resources to not over-extend
        my_total = sum(int(p.ships) for p in snap.my_planets)
        if my_total < 40 and len(snap.my_planets) < 3:
            return []

        scored = [
            (self._exposure_score(ep), ep)
            for ep in snap.enemy_planets
            if ep.id not in snap.comet_ids and int(ep.production) >= PRESSURE_MIN_PROD
        ]
        scored.sort(reverse=True)

        result  = []
        launched = set()
        for score, target in scored:
            if target.id in launched or len(result) >= max_missions:
                break
            fronts = self._quadrant_fronts(target)
            if not fronts:
                continue
            plan = self._build_encirclement(target, fronts)
            if plan:
                result.append(plan)
                launched.add(target.id)
                for src, ships, _, _ in plan:
                    snap.commit(src.id, ships)

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DEFENSE ENGINE
#  Emergency: protect high-production planets under imminent attack.
#  Pulls reinforcements from nearby surplus planets.
# ═══════════════════════════════════════════════════════════════════════════════

class DefenseEngine:

    def __init__(self, snap: GameSnapshot):
        self.snap = snap

    def emergency_saves(self, moves):
        """
        Detect my planets about to fall and pull reinforcements.
        Modifies moves in place; returns True if any save was attempted.
        """
        snap   = self.snap
        saved  = False

        # Sort by production (save the most valuable planets first)
        at_risk = []
        for p in snap.my_planets:
            inc = snap.incoming.get(p.id, {})
            enemy_near = sum(
                sh for eta, own, sh in inc.get("arrivals", [])
                if own != snap.player and eta <= DEFENSE_ETA_HORIZON
            )
            if enemy_near <= 0:
                continue
            deficit = enemy_near - int(p.ships) + 6
            if deficit > 0:
                at_risk.append((int(p.production), deficit, p))

        at_risk.sort(reverse=True)  # highest production first

        for _prod, deficit, endangered in at_risk:
            helpers = sorted(
                [mp for mp in snap.my_planets
                 if mp.id != endangered.id and snap.surplus(mp) >= MIN_SEND_SHIPS],
                key=lambda mp: dp(mp, endangered)
            )
            sent = 0
            for helper in helpers[:DEFENSE_SAVE_HELPERS]:
                if sent >= deficit:
                    break
                send = pkt_down(min(snap.surplus(helper), deficit - sent))
                if not valid_pkt(send):
                    continue
                if snap.dies_if_sent(helper, send):
                    continue
                angle, ok, _ = compute_aim(helper, endangered, send, snap.ang_vel)
                if not ok:
                    continue
                moves.append([helper.id, angle, send])
                snap.commit(helper.id, send)
                sent  += send
                saved  = True

        return saved


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def agent(obs, config=None):
    """
    Decision order each turn:
      1. Emergency defense – save endangered high-production planets
      2. Pressure goal     – encirclement attack on most exposed enemy planet
      3. Expansion goal    – fastest-occupiable planet in each quadrant
                             (A* routing with traffic, checkpoint stepping stones)
      4. Fallback          – brute-force nearest capture if nothing else fired
    """
    snap = GameSnapshot(obs)
    if not snap.my_planets:
        return []

    astar   = WeightedAStarPlanner(snap)
    moves   = []

    # 1. Emergency defense
    defense = DefenseEngine(snap)
    defense.emergency_saves(moves)

    # 2. Pressure (encirclement) – only when we have a meaningful foothold
    if len(snap.my_planets) >= 2:
        pressure = PressureEngine(snap, astar)
        for plan in pressure.missions():
            for src, ships, angle, _ in plan:
                if valid_pkt(ships):
                    moves.append([src.id, angle, ships])

    # 3. Expansion – all-directions quadrant growth
    expansion = ExpansionEngine(snap, astar)
    for plan in expansion.missions():
        for src, ships, angle, _ in plan:
            if valid_pkt(ships):
                moves.append([src.id, angle, ships])

    # 4. Fallback – if nothing launched, hit the nearest affordable target
    if not moves:
        all_targets = [
            t for t in snap.neutral_planets + snap.enemy_planets
            if t.id not in snap.comet_ids
        ]
        all_targets.sort(key=lambda t: min(
            (dp(mp, t) for mp in snap.my_planets), default=999.0
        ))
        for target in all_targets:
            src = min(snap.my_planets, key=lambda p: dp(p, target), default=None)
            if src is None:
                break
            need = pkt_up(snap.capture_need(src, target))
            avail = pkt_down(snap.surplus(src))
            send  = pkt_down(min(avail, need))
            if valid_pkt(send) and not snap.dies_if_sent(src, send):
                angle, ok, _ = compute_aim(src, target, send, snap.ang_vel)
                if ok:
                    moves.append([src.id, angle, send])
                    break

    if DEBUG:
        print(f"[step {snap.step}] moves={len(moves)} "
              f"my={len(snap.my_planets)} enemy={len(snap.enemy_planets)} "
              f"neutral={len(snap.neutral_planets)}")

    return moves
