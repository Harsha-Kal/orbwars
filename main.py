import math
import numpy as np

# <<LEARNED_PARAMS_START>>
LEARNED_PARAMS = {
    "min_hold": 3,
    "defend_threshold": 11,
    "reinforce_min_send": 4,
    "reinforce_target_net": 8,
    "neutral_prod_mult": 42,
    "attack_dist_weight": 0.067,
    "attack_prod_weight": 5.804,
    "attack_buffer": 25,
    "consolidate_threshold": 17,
}
# <<LEARNED_PARAMS_END>>

# ─────────────────────────────────────────────────────────────────────────────
SUN_X, SUN_Y = 50.0, 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0

class Planet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id, self.owner, self.x, self.y = id, owner, x, y
        self.radius, self.ships, self.production = radius, ships, production

class Fleet:
    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id, self.owner, self.x, self.y = id, owner, x, y
        self.angle, self.from_planet_id, self.ships = angle, from_planet_id, ships

def dist(ax, ay, bx, by): return math.hypot(bx - ax, by - ay)
def dp(a, b): return math.hypot(b.x - a.x, b.y - a.y)

def fleet_speed(n):
    n = max(1, int(n))
    if n == 1: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5

def travel_turns(d_val, ships):
    return d_val / fleet_speed(ships)

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
    t1, t2 = (-b - sq) / (2*a), (-b + sq) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)

def safe_angle(sx, sy, tx, ty):
    direct = math.atan2(ty - sy, tx - sx)
    if not hits_sun(sx, sy, tx, ty): return direct
    dist_st = dist(sx, sy, tx, ty)
    for off in [0.5, -0.5, 1.0, -1.0]:
        ang = direct + off
        ex, ey = sx + dist_st * math.cos(ang), sy + dist_st * math.sin(ang)
        if not hits_sun(sx, sy, ex, ey): return ang
    return direct

def predict_pos(p, ang_vel, turns):
    orb_r = dist(p.x, p.y, SUN_X, SUN_Y)
    if orb_r + p.radius >= 50.0: return p.x, p.y
    cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
    new_ang = cur_ang + ang_vel * turns
    return SUN_X + orb_r * math.cos(new_ang), SUN_Y + orb_r * math.sin(new_ang)

def aim_at(src, tgt, ang_vel, ships):
    tx, ty = tgt.x, tgt.y
    for _ in range(8):
        d_est = dist(src.x, src.y, tx, ty)
        t_est = travel_turns(d_est, ships)
        tx, ty = predict_pos(tgt, ang_vel, t_est)
    return safe_angle(src.x, src.y, tx, ty)

def fleet_net(planet, all_fleets, me):
    fi = ei = 0
    for fl in all_fleets:
        ea = math.atan2(planet.y - fl.y, planet.x - fl.x)
        diff = abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea)))
        if diff < 0.35:
            if fl.owner == me: fi += fl.ships
            else: ei += fl.ships
    return fi, ei

def agent(obs):
    g = lambda key, default: (obs.get(key, default) if isinstance(obs, dict) else getattr(obs, key, default))
    me = int(g("player", 0)); ang_vel = float(g("angular_velocity", 0.033))
    planets_raw = g("planets", []); fleets_raw = g("fleets", [])
    planets = [Planet(int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), int(p[6])) for p in planets_raw]
    fleets = [Fleet(int(f[0]), int(f[1]), float(f[2]), float(f[3]), float(f[4]), int(f[5]), int(f[6])) for f in fleets_raw]
    my_p = [p for p in planets if p.owner == me]; neutral_p = [p for p in planets if p.owner == -1]; enemy_p = [p for p in planets if p.owner not in (-1, me)]
    if not my_p: return []
    committed, moves = {}, []
    def avail(p): return max(0, p.ships - committed.get(p.id, 0) - LEARNED_PARAMS["min_hold"])
    def send_to(src, tgt, n):
        n = int(n)
        if n <= 0: return
        ang = aim_at(src, tgt, ang_vel, n)
        committed[src.id] = committed.get(src.id, 0) + n
        moves.append([src.id, ang, n])
    targeted = set()
    for fl in fleets:
        if fl.owner == me:
            for p in planets:
                if p.owner == me: continue
                ea = math.atan2(p.y - fl.y, p.x - fl.x)
                if abs(math.atan2(math.sin(fl.angle - ea), math.cos(fl.angle - ea))) < 0.35:
                    targeted.add(p.id)
    # 1. Reinforcement
    threat = {p.id: fleet_net(p, fleets, me) for p in my_p}
    for p in sorted(my_p, key=lambda p: threat[p.id][1] - threat[p.id][0] - p.ships, reverse=True):
        fi, ei = threat[p.id]
        net = p.ships + fi - ei
        if net <= LEARNED_PARAMS["defend_threshold"]:
            needed = max(LEARNED_PARAMS["reinforce_min_send"], LEARNED_PARAMS["reinforce_target_net"] - net)
            donors = sorted([q for q in my_p if q.id != p.id and avail(q) >= needed], key=lambda q: dp(q, p))
            if donors: send_to(donors[0], p, min(needed, avail(donors[0])))
    # 2. Expansion
    free_neutrals = sorted([n for n in neutral_p if n.id not in targeted], key=lambda n: -(n.production * LEARNED_PARAMS["neutral_prod_mult"] - n.ships))
    for src in sorted(my_p, key=lambda p: -avail(p)):
        for tgt in list(free_neutrals):
            av = avail(src); cost = tgt.ships + 1
            if cost <= av:
                send_to(src, tgt, cost); targeted.add(tgt.id); free_neutrals.remove(tgt)
            if avail(src) < 2: break
    # 3. Attacks
    free_enemies = [e for e in enemy_p if e.id not in targeted]
    for src in sorted(my_p, key=lambda p: -avail(p)):
        av = avail(src)
        if av < 5 or not free_enemies: continue
        tgt = min(free_enemies, key=lambda e: e.ships + dp(src, e) * LEARNED_PARAMS["attack_dist_weight"] - e.production * LEARNED_PARAMS["attack_prod_weight"])
        t_est = travel_turns(dp(src, tgt), av); need = tgt.ships + int(tgt.production * t_est) + 1
        if av >= need:
            send_to(src, tgt, min(av, need + LEARNED_PARAMS["attack_buffer"])); targeted.add(tgt.id); free_enemies.remove(tgt)
    # 4. Consolidate
    if enemy_p and len(my_p) > 1:
        front = min(my_p, key=lambda p: min(dp(p, e) for e in enemy_p))
        for src in my_p:
            if src.id != front.id:
                av = avail(src)
                if av >= LEARNED_PARAMS["consolidate_threshold"]: send_to(src, front, av // 2)
    return moves
