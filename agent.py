"""
Orbit Wars Agent - Competition Submission
=========================================
Planet format: [id, owner, x, y, radius, ships, production]
Fleet format:  [id, owner, x, y, angle, from_planet_id, ships]
Obs attributes: .player, .planets, .fleets, .angular_velocity, .step

Strategy:
  1. Reinforce any threatened planet.
  2. Expand: capture cheapest neutral, skipping already-targeted ones.
  3. Attack weakest enemy when we have overwhelming surplus.
  4. Forward-mass large surpluses toward the front.
"""

import math

MIN_HOLD = 1   # minimum ships to keep on each planet


def dist2d(p, q):
    return math.hypot(p["x"] - q["x"], p["y"] - q["y"])


def parse_planets(raw):
    """[id, owner, x, y, radius, ships, production]"""
    out = []
    for p in raw:
        out.append({"id": int(p[0]), "owner": int(p[1]),
                    "x": float(p[2]),  "y": float(p[3]),
                    "radius": float(p[4]), "ships": int(p[5]),
                    "prod": int(p[6])})
    return out


def parse_fleets(raw):
    """[id, owner, x, y, angle, from_planet_id, ships]"""
    out = []
    for f in raw:
        out.append({"id": int(f[0]), "owner": int(f[1]),
                    "x": float(f[2]),  "y": float(f[3]),
                    "angle": float(f[4]), "from": int(f[5]),
                    "ships": int(f[6])})
    return out


def fleet_heading_to(fl, planet):
    """True if fleet appears to be heading roughly toward this planet."""
    ea = math.atan2(planet["y"] - fl["y"], planet["x"] - fl["x"])
    diff = abs(math.atan2(math.sin(fl["angle"] - ea),
                          math.cos(fl["angle"] - ea)))
    return diff < 0.45


def agent(obs, cfg=None):
    me      = int(obs.player)
    planets = parse_planets(obs.planets)
    fleets  = parse_fleets(obs.fleets)

    my_p      = [p for p in planets if p["owner"] == me]
    neutral_p = [p for p in planets if p["owner"] == -1]
    enemy_p   = [p for p in planets if p["owner"] not in (-1, me)]

    # Track which neutral planets already have our fleets heading there
    my_fleets = [fl for fl in fleets if fl["owner"] == me]
    already_targeted = set()
    for fl in my_fleets:
        for p in neutral_p:
            if fleet_heading_to(fl, p):
                already_targeted.add(p["id"])

    moves     = []
    committed = {}   # planet_id → ships earmarked this turn

    def avail(p):
        return max(0, p["ships"] - committed.get(p["id"], 0) - MIN_HOLD)

    def send(src, tgt, n):
        n = int(n)
        if n <= 0:
            return
        committed[src["id"]] = committed.get(src["id"], 0) + n
        moves.append([src["id"], tgt["id"], n])

    # ── Phase 1: Reinforce threatened planets ─────────────────────────────────
    for p in my_p:
        fi = sum(fl["ships"] for fl in my_fleets if fleet_heading_to(fl, p))
        ei = sum(fl["ships"] for fl in fleets
                 if fl["owner"] != me and fleet_heading_to(fl, p))
        net = p["ships"] + fi - ei
        if net <= 2:
            needed = max(5, 5 - net)
            donors = sorted(
                [q for q in my_p if q["id"] != p["id"] and avail(q) >= needed],
                key=lambda q: dist2d(q, p)
            )
            if donors:
                send(donors[0], p, min(needed, avail(donors[0])))

    # ── Phase 2: Expand to cheap, untargeted neutral planets ──────────────────
    free_neutrals = [n for n in neutral_p if n["id"] not in already_targeted]

    for p in sorted(my_p, key=lambda x: -x["prod"]):   # richest first
        av = avail(p)
        if av < 2:
            continue
        affordable = [n for n in free_neutrals if n["ships"] + 1 <= av]
        if affordable:
            # Cheapest first, ties broken by distance
            tgt = min(affordable, key=lambda n: (n["ships"], dist2d(p, n)))
            cost = tgt["ships"] + 1
            send(p, tgt, cost)
            already_targeted.add(tgt["id"])
            # Remove from free_neutrals so next planet picks a different one
            free_neutrals = [n for n in free_neutrals if n["id"] != tgt["id"]]

    # ── Phase 3: Attack weakest reachable enemy ───────────────────────────────
    already_attacked = set()
    for fl in my_fleets:
        for e in enemy_p:
            if fleet_heading_to(fl, e):
                already_attacked.add(e["id"])

    free_enemies = [e for e in enemy_p if e["id"] not in already_attacked]

    for p in sorted(my_p, key=lambda x: -avail(x)):
        av = avail(p)
        if av < 5 or not free_enemies:
            continue
        tgt = min(free_enemies, key=lambda e: e["ships"] + dist2d(p, e) * 0.3)
        need = tgt["ships"] + 3
        if av >= need:
            send(p, tgt, min(av, need + 5))
            already_attacked.add(tgt["id"])
            free_enemies = [e for e in free_enemies if e["id"] != tgt["id"]]

    # ── Phase 4: Forward-mass large surplus to front-line ─────────────────────
    if enemy_p and len(my_p) > 1:
        front = min(my_p, key=lambda p: min(dist2d(p, e) for e in enemy_p))
        for p in my_p:
            if p["id"] == front["id"]:
                continue
            av = avail(p)
            if av >= 20:
                send(p, front, av // 2)

    return moves
