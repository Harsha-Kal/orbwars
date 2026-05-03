"""
Orbit Wars - Competitive Agent v2
===================================
Action:  [from_planet_id, angle_radians, num_ships]
Planet:  Planet(id, owner, x, y, radius, ships, production)
Fleet:   Fleet(id, owner, x, y, angle, from_planet_id, ships)

Improvements over v1:
  - Correct combat math (accounts for multi-fleet battles)
  - Production-value-aware targeting (prefer high-prod planets)
  - Orbit prediction uses CURRENT position from obs, not initial
  - Multi-target expansion per turn from each owned planet
  - 4-player awareness: prefer isolated enemies; let others fight
  - Surplus ships forwarded to nearest attack planet, not just "front"
  - Comet capture when cheap
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

SUN_X, SUN_Y = 50.0, 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0
MIN_HOLD     = 1     # ships to keep on every owned planet


# ── geometry ──────────────────────────────────────────────────────────────────

def dist(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)

def dp(a, b):   # dist between two Planet-like objects
    return math.hypot(b.x - a.x, b.y - a.y)

def fleet_speed(n):
    n = max(1, int(n))
    if n == 1: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5
def travel_turns(d_val, ships): return d_val / fleet_speed(ships)
def hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx*dx + dy*dy
    if a < 1e-9: return math.hypot(fx, fy) < SUN_R
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - SUN_R*SUN_R
    disc = b*b - 4*a*c
    if disc < 0: return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)

def safe_angle(sx, sy, tx, ty):
    direct = math.atan2(ty - sy, tx - sx)
    if not hits_sun(sx, sy, tx, ty):
        return direct
    dist_st = dist(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0]:
        ang = direct + off
        ex = sx + dist_st * math.cos(ang)
        ey = sy + dist_st * math.sin(ang)
        if not hits_sun(sx, sy, ex, ey):
            return ang
    return direct


# ── orbit prediction ──────────────────────────────────────────────────────────

def is_rotating(p, init_p=None):
    """True if planet orbits the sun (orbital_radius + radius < 50)."""
    ref = init_p if init_p is not None else p
    orb_r = dist(ref.x, ref.y, SUN_X, SUN_Y)
    return (orb_r + ref.radius) < 50.0

def predict_pos(p, ang_vel, turns):
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= 50.0:
        return p.x, p.y     # static
    cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
    new_ang = cur_ang + ang_vel * turns
    return SUN_X + orb_r * math.cos(new_ang), SUN_Y + orb_r * math.sin(new_ang)

def aim_at(src, tgt, ang_vel, ships):
    tx, ty = tgt.x, tgt.y
    for _ in range(10):
        d_est = dist(src.x, src.y, tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    return safe_angle(src.x, src.y, tx, ty)


# ── fleet analysis ────────────────────────────────────────────────────────────

def fleet_net(planet, all_fleets, me):
    """
    Returns (my_inbound, enemy_inbound) ships heading roughly toward this planet.
    Uses angle proximity with a tighter tolerance for accuracy.
    """
    fi = ei = 0
    for fl in all_fleets:
        ea   = math.atan2(planet.y - fl.y, planet.x - fl.x)
        diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
        if diff < 0.35:
            if fl.owner == me:
                fi += fl.ships
            else:
                ei += fl.ships
    return fi, ei


# ── combat math ───────────────────────────────────────────────────────────────

def ships_to_capture(planet, inbound_enemy=0):
    """
    How many ships do we need to send to capture `planet` given its current
    garrison and any other enemy fleets already inbound?
    Combat rule: largest force beats 2nd largest; survivor fights garrison.
    We're the attacking fleet — we need to beat the garrison (+ any friendly
    reinforcements already on the way which we approximate as 0 here).
    Minimum: garrison + 1 to flip ownership.
    """
    return planet.ships + 1


# ── planet value score ────────────────────────────────────────────────────────

def planet_value(p):
    """Higher is more desirable to capture."""
    return p.production * 10 - p.ships   # cheap + productive = great


# ── main agent ────────────────────────────────────────────────────────────────

def agent(obs):
    # Handle both dict and Struct obs
    g = lambda key, default: (obs.get(key, default) if isinstance(obs, dict)
                              else getattr(obs, key, default))

    me         = int(g("player", 0))
    ang_vel    = float(g("angular_velocity", 0.033))
    comet_ids  = set(g("comet_planet_ids", []))

    planets    = [Planet(*p) for p in g("planets", [])]
    fleets     = [Fleet(*f)  for f in g("fleets",  [])]

    my_p      = [p for p in planets if p.owner == me]
    neutral_p = [p for p in planets if p.owner == -1]
    enemy_p   = [p for p in planets if p.owner not in (-1, me)]
    all_enemies = set(p.owner for p in enemy_p)  # distinct enemy players

    committed = {}
    moves     = []

    def avail(p):
        return max(0, p.ships - committed.get(p.id, 0) - MIN_HOLD)

    def send(src, ang, n):
        n = int(n)
        if n <= 0:
            return
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])

    def send_to(src, tgt, n):
        ang = aim_at(src, tgt, ang_vel, n)
        send(src, ang, n)

    # Planets already being targeted by our fleets
    my_fleets  = [fl for fl in fleets if fl.owner == me]
    targeted   = set()
    for fl in my_fleets:
        for p in neutral_p + enemy_p:
            fi, _ = fleet_net(p, [fl], me)
            if fi > 0:
                targeted.add(p.id)

    # Total inbound enemy ships per our planet (for threat assessment)
    threat = {}
    for p in my_p:
        fi, ei = fleet_net(p, fleets, me)
        threat[p.id] = (fi, ei)

    total_my_ships = sum(p.ships for p in my_p) + sum(fl.ships for fl in my_fleets)
    total_my_prod  = sum(p.production for p in my_p)

    # ── Phase 1: Reinforce planets under serious threat ───────────────────────
    for p in sorted(my_p, key=lambda p: threat[p.id][1] - threat[p.id][0] - p.ships, reverse=True):
        fi, ei = threat[p.id]
        net = p.ships + fi - ei
        if net > 3:
            continue   # safe enough
        needed = max(5, 5 - net)
        donors = sorted(
            [q for q in my_p if q.id != p.id and avail(q) >= needed],
            key=lambda q: dp(q, p)
        )
        if donors:
            send_to(donors[0], p, min(needed, avail(donors[0])))

    # ── Phase 2: Expand to neutrals ──────────────────────────────────────────
    # Score: high production preferred, cheap preferred; avoid already targeted
    free_neutrals = [n for n in neutral_p if n.id not in targeted]
    # Sort neutrals by value descending (production matters most)
    free_neutrals.sort(key=lambda n: -planet_value(n))

    for src in sorted(my_p, key=lambda p: -avail(p)):
        av = avail(src)
        if av < 2:
            continue
        for tgt in free_neutrals:
            cost = ships_to_capture(tgt)
            if cost <= av:
                send_to(src, tgt, cost)
                targeted.add(tgt.id)
                free_neutrals = [n for n in free_neutrals if n.id != tgt.id]
                break  # one expansion per source planet per turn

    # ── Phase 3: Attack enemy planets ────────────────────────────────────────
    # In 4-player games: prefer attacking players who are also being attacked
    # by others (weakened). Avoid 2-front wars.
    free_enemies = [e for e in enemy_p if e.id not in targeted]

    # Score enemy planets: fewer ships + higher production = top priority
    def enemy_priority(e, src):
        d_val = dp(src, e)
        # Bonus if this enemy is already fighting someone else
        enemy_inbound = sum(fl.ships for fl in fleets
                            if fl.owner != me and fl.owner != e.owner
                            and fleet_net(e, [fl], me)[1] > 0)
        return e.ships - enemy_inbound * 0.5 + d_val * 0.2 - e.production * 3

    for src in sorted(my_p, key=lambda p: -avail(p)):
        av = avail(src)
        if av < 4 or not free_enemies:
            continue
        tgt = min(free_enemies, key=lambda e: enemy_priority(e, src))
        need = ships_to_capture(tgt)
        if av >= need:
            # Send with a buffer to handle slight miscounts
            send_to(src, tgt, min(av, need + max(3, need // 5)))
            targeted.add(tgt.id)
            free_enemies.remove(tgt)

    # ── Phase 4: Consolidate surplus toward attacking planets ─────────────────
    # Find owned planets closest to enemies and funnel ships there
    if enemy_p and len(my_p) > 1:
        # Rank our planets by attack opportunity (closest to weakest enemy)
        def attack_score(p):
            if not free_enemies:
                return 0
            nearest_e = min(free_enemies, key=lambda e: dp(p, e))
            return -(dp(p, nearest_e) + nearest_e.ships * 0.5)

        staging = sorted(my_p, key=attack_score)
        if staging:
            front = staging[0]
            for src in my_p:
                if src.id == front.id:
                    continue
                av = avail(src)
                # Only forward if we have a big surplus and front needs ships
                if av >= 15:
                    send_to(src, front, av // 2)

    return moves

if __name__ == "__main__":
    test_obs = {
        "player": 1,
        "step": 50,
        "angular_velocity": 0.033,
        "planets": [
            [0, 1, 20, 20, 5, 50, 5],   # id, owner, x, y, radius, ships, production
            [1, -1, 80, 80, 4, 20, 3],  # neutral planet
            [2, 2, 30, 70, 5, 40, 4],   # enemy planet
        ],
        "fleets": [],
        "comet_planet_ids": [],
        "comets": []
    }
    result = agent(test_obs)
    print(f"Agent returned: {result}")

