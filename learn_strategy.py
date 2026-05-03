"""
Orbit Wars Strategy Learner
============================
Reads agent logs, runs local simulations, trains ML models, and rewrites
the LEARNED_PARAMS block in main.py with optimised strategy constants.

Models trained
--------------
  RandomForestClassifier        – win-probability predictor
  GradientBoostingRegressor     – planet-value scorer + survival predictor
  DecisionTreeClassifier        – interpretable fleet-divergence explainer
  ExtraTreesClassifier          – robust feature importance

Post-mortem analysis
--------------------
  GamePostMortem     – per-game: why planets were lost, survival length
  SurvivalOptimizer  – GBR: early features → survival_steps, nudges params
  FleetGrowthAnalyzer– DT: why enemy fleet grew while ours didn't

Usage
-----
  pip install scikit-learn pandas prophet kaggle-environments
  python learn_strategy.py [--games 30] [--no-update]
"""

import json, math, os, pickle, re, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from kaggle_environments import make as kag_make
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False

ROOT     = Path(__file__).parent
LOG_DIR  = ROOT / "agent logs"
PKL_PATH = ROOT / "strategy_data.pkl"
MAIN_PY  = ROOT / "main.py"

FEATURE_COLS = [
    "turn_frac",
    "my_ships", "enemy_ships", "neutral_ships",
    "ship_ratio", "fleet_ratio",
    "my_prod", "enemy_prod", "prod_ratio",
    "my_planets", "enemy_planets", "neutral_planets",
    "my_fleets_ships", "enemy_fleets_ships",
    "max_threat", "total_threat",
    "cheapest_neutral", "best_neutral_prod", "neutral_opportunity",
    "weakest_enemy", "best_attack_score",
    "dominance", "leading",
    "early_game", "mid_game", "late_game",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 – LOG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class LogAnalyzer:
    """Parse Kaggle agent-log JSON files (timing + stderr/stdout)."""

    def analyze(self) -> Dict:
        if not LOG_DIR.exists():
            print(f"[LogAnalyzer] Log dir not found: {LOG_DIR}")
            return {}

        files = sorted(LOG_DIR.glob("*.json"))
        print(f"\n[LogAnalyzer] {len(files)} log files found")

        all_dur, all_errors, slow = [], [], []

        for fpath in files:
            with open(fpath) as fp:
                data = json.load(fp)

            durs = []
            for step_idx, step in enumerate(data):
                for entry in step:
                    if not isinstance(entry, dict):
                        continue
                    d = entry.get("duration", 0.0)
                    durs.append(d)
                    all_dur.append(d)
                    if d > 0.1:
                        slow.append({"file": fpath.name, "step": step_idx, "ms": d * 1000})
                    if entry.get("stderr"):
                        all_errors.append({"file": fpath.name, "step": step_idx,
                                           "msg": entry["stderr"][:200]})

            if durs:
                print(f"  {fpath.name}: steps={len(data)}  avg={np.mean(durs)*1000:.2f}ms  "
                      f"max={np.max(durs)*1000:.2f}ms  p95={np.percentile(durs,95)*1000:.2f}ms")

        if all_dur:
            print(f"\n  Global: avg={np.mean(all_dur)*1000:.2f}ms  "
                  f"max={np.max(all_dur)*1000:.2f}ms  "
                  f"slow_turns(>100ms)={len(slow)}  errors={len(all_errors)}")
        if all_errors:
            print(f"  First error: {all_errors[0]}")

        return {
            "files": len(files),
            "avg_ms": np.mean(all_dur) * 1000 if all_dur else 0,
            "max_ms": np.max(all_dur) * 1000 if all_dur else 0,
            "p95_ms": np.percentile(all_dur, 95) * 1000 if all_dur else 0,
            "slow_turns": slow,
            "errors": all_errors,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 – GAME SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _sg(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _fleet_heads_to(fl_dict, planet_dict, tol=0.40) -> bool:
    ea = math.atan2(planet_dict["y"] - fl_dict["y"], planet_dict["x"] - fl_dict["x"])
    diff = abs(math.atan2(
        math.sin(fl_dict["angle"] - ea),
        math.cos(fl_dict["angle"] - ea)
    ))
    return diff < tol


def _parse_obs(obs) -> Tuple[List, List, int]:
    planets_raw = _sg(obs, "planets", [])
    fleets_raw  = _sg(obs, "fleets",  [])
    player      = int(_sg(obs, "player", 0))
    planets = [
        {"id": int(p[0]), "owner": int(p[1]),
         "x": float(p[2]), "y": float(p[3]),
         "radius": float(p[4]), "ships": int(p[5]), "prod": int(p[6])}
        for p in planets_raw
    ]
    fleets = [
        {"id": int(f[0]), "owner": int(f[1]),
         "x": float(f[2]), "y": float(f[3]),
         "angle": float(f[4]), "from": int(f[5]), "ships": int(f[6])}
        for f in fleets_raw
    ]
    return planets, fleets, player


def make_difficulty_agent(base_module, difficulty: str):
    param_overrides = {
        "passive": {
            "aggressive_ship_ratio": 3.0,
            "min_attack_avail": 15,
            "consolidate_avail": 30,
        },
        "aggressive": {
            "aggressive_ship_ratio": 1.1,
            "attack_buffer_ratio": 0.05,
            "consolidate_avail": 8,
            "min_attack_avail": 3,
        },
    }
    overrides = param_overrides.get(difficulty, {})

    def agent(obs):
        saved = {k: base_module.LEARNED_PARAMS[k] for k in overrides}
        base_module.LEARNED_PARAMS.update(overrides)
        result = base_module.agent(obs)
        base_module.LEARNED_PARAMS.update(saved)
        return result

    agent.__name__ = f"agent_{difficulty}"
    return agent


def run_single_game(main_agent, opponent="random", seed: Optional[int] = None) -> Tuple[List, int, List]:
    cfg = {"seed": seed} if seed is not None else {}
    env = kag_make("orbit_wars", configuration=cfg, debug=False)
    opponent_agent = main_agent if opponent == "self" else opponent
    steps = env.run([main_agent, opponent_agent])

    rows = []
    for turn_idx, step in enumerate(steps):
        for player_idx, pstate in enumerate(step):
            obs    = _sg(pstate, "observation")
            reward = _sg(pstate, "reward")
            if obs is None:
                continue
            rows.append({"turn": turn_idx, "player": player_idx,
                         "obs": obs, "reward": reward,
                         "max_turn": len(steps)})

    final = steps[-1]
    final_rewards = [_sg(ps, "reward") or 0 for ps in final]
    winner = int(np.argmax(final_rewards)) if final_rewards else 0
    return rows, winner, final_rewards


def run_multiple_games(n_games: int = 20, opponents=None) -> Tuple[pd.DataFrame, List, List]:
    """
    Returns (feature_df, ship_series_list, postmortem_list).
    postmortem_list: one GamePostMortem per game (player 0 perspective).
    """
    import importlib, main as main_module

    if opponents is None:
        opponents = [
            "random", "random", "self",
            make_difficulty_agent(main_module, "passive"),
            make_difficulty_agent(main_module, "aggressive"),
        ]

    all_rows    = []
    ship_series = []
    postmortems = []

    for i in range(n_games):
        opponent  = opponents[i % len(opponents)]
        opp_label = getattr(opponent, "__name__", str(opponent))
        seed      = 42 + i * 7
        print(f"  Game {i+1}/{n_games}  opponent={opp_label}  seed={seed}", end="  ")
        try:
            rows, winner, rewards = run_single_game(
                main_module.agent, opponent=opponent, seed=seed)
            print(f"winner={winner}  rewards={[round(r,1) for r in rewards]}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        turns_p0 = [r for r in rows if r["player"] == 0]
        game_feat_rows = []
        my_ships_series = []

        for r in turns_p0:
            feats = _extract_features(r["obs"], r["turn"], r["max_turn"])
            if feats is None:
                continue
            feats["won"]    = 1 if winner == 0 else 0
            feats["game_i"] = i
            all_rows.append(feats)
            game_feat_rows.append(feats)
            my_ships_series.append(feats["my_ships"] + feats["my_fleets_ships"])

        if my_ships_series:
            ship_series.append({
                "game_i": i, "won": 1 if winner == 0 else 0,
                "series": my_ships_series,
            })

        if game_feat_rows:
            pm = GamePostMortem(game_feat_rows, game_i=i,
                                won=(winner == 0), n_turns=len(steps if False else turns_p0))
            postmortems.append(pm)

    df = pd.DataFrame(all_rows)
    return df, ship_series, postmortems


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 – FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_features(obs, turn: int, max_turn: int) -> Optional[Dict]:
    try:
        planets, fleets, me = _parse_obs(obs)
    except Exception:
        return None

    my_p  = [p for p in planets if p["owner"] == me]
    en_p  = [p for p in planets if p["owner"] not in (-1, me)]
    ne_p  = [p for p in planets if p["owner"] == -1]
    my_fl = [f for f in fleets  if f["owner"] == me]
    en_fl = [f for f in fleets  if f["owner"] != me]

    my_ships    = sum(p["ships"] for p in my_p)
    en_ships    = sum(p["ships"] for p in en_p)
    ne_ships    = sum(p["ships"] for p in ne_p)
    my_fl_ships = sum(f["ships"] for f in my_fl)
    en_fl_ships = sum(f["ships"] for f in en_fl)
    my_prod     = sum(p["prod"] for p in my_p)
    en_prod     = sum(p["prod"] for p in en_p)

    threat_vals = []
    for p in my_p:
        inbound_en = sum(f["ships"] for f in en_fl if _fleet_heads_to(f, p))
        inbound_my = sum(f["ships"] for f in my_fl if _fleet_heads_to(f, p))
        net = p["ships"] + inbound_my - inbound_en
        threat_vals.append(max(0, -net))

    max_threat   = max(threat_vals) if threat_vals else 0
    total_threat = sum(threat_vals)

    cheapest_ne  = min((p["ships"] for p in ne_p), default=0)
    best_ne_prod = max((p["prod"]  for p in ne_p), default=0)
    ne_opp       = sum(p["prod"] / (p["ships"] + 1) for p in ne_p) if ne_p else 0

    weakest_en   = min((p["ships"] for p in en_p), default=0)
    best_attack  = max((p["prod"] / (p["ships"] + 1) for p in en_p), default=0)

    ship_ratio  = (my_ships + my_fl_ships) / max(1, en_ships + en_fl_ships)
    fleet_ratio = my_fl_ships / max(1, en_fl_ships)
    prod_ratio  = my_prod / max(1, en_prod)
    dominance   = (len(my_p) - len(en_p)) / max(1, len(planets))
    leading     = 1.0 if (my_ships + my_fl_ships) > (en_ships + en_fl_ships) else 0.0

    turn_frac = turn / max(1, max_turn)
    early = 1.0 if turn_frac < 0.20 else 0.0
    mid   = 1.0 if 0.20 <= turn_frac < 0.70 else 0.0
    late  = 1.0 if turn_frac >= 0.70 else 0.0

    return {
        "turn_frac": turn_frac,
        "my_ships": my_ships, "enemy_ships": en_ships, "neutral_ships": ne_ships,
        "ship_ratio": min(ship_ratio, 10.0),
        "fleet_ratio": min(fleet_ratio, 10.0),
        "my_prod": my_prod, "enemy_prod": en_prod,
        "prod_ratio": min(prod_ratio, 10.0),
        "my_planets": len(my_p), "enemy_planets": len(en_p), "neutral_planets": len(ne_p),
        "my_fleets_ships": my_fl_ships, "enemy_fleets_ships": en_fl_ships,
        "max_threat": max_threat, "total_threat": total_threat,
        "cheapest_neutral": cheapest_ne, "best_neutral_prod": best_ne_prod,
        "neutral_opportunity": min(ne_opp, 5.0),
        "weakest_enemy": weakest_en, "best_attack_score": min(best_attack, 5.0),
        "dominance": dominance, "leading": leading,
        "early_game": early, "mid_game": mid, "late_game": late,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 – POST-MORTEM: WHY DID WE LOSE PLANETS / DIE EARLY?
# ═══════════════════════════════════════════════════════════════════════════════

class GamePostMortem:
    """
    Per-game diagnosis of planet losses, survival length, and fleet divergence.
    Input: sorted list of feature-row dicts for one game, player 0.
    """

    CAUSES = ("overwhelmed", "threat_spike", "starvation", "expansion_lag", "unknown")

    def __init__(self, rows: List[Dict], game_i: int = 0,
                 won: bool = False, n_turns: int = 0):
        self.rows    = rows
        self.game_i  = game_i
        self.won     = won
        self.n_turns = n_turns or len(rows)

    @property
    def survival_steps(self) -> int:
        """Last turn (0-500) where we still held ≥1 planet."""
        for row in reversed(self.rows):
            if row.get("my_planets", 0) > 0:
                return round(row["turn_frac"] * 500)
        return 0

    @property
    def reached_500(self) -> bool:
        return self.survival_steps >= 490

    def planet_loss_events(self) -> List[Dict]:
        """One record per turn where our planet count dropped."""
        events = []
        for i in range(1, len(self.rows)):
            prev = self.rows[i - 1]
            curr = self.rows[i]
            if curr.get("my_planets", 0) < prev.get("my_planets", 1):
                events.append({
                    "game_i":         self.game_i,
                    "turn_idx":       i,
                    "turn_frac":      curr["turn_frac"],
                    "planets_before": prev["my_planets"],
                    "planets_after":  curr["my_planets"],
                    "cause":          self._diagnose(prev),
                    # snapshot of key indicators just before loss
                    "ship_ratio":     prev.get("ship_ratio", 1.0),
                    "prod_ratio":     prev.get("prod_ratio", 1.0),
                    "max_threat":     prev.get("max_threat", 0),
                    "total_threat":   prev.get("total_threat", 0),
                    "my_ships":       prev.get("my_ships", 0),
                    "enemy_ships":    prev.get("enemy_ships", 0),
                    "leading":        prev.get("leading", 0),
                })
        return events

    def _diagnose(self, row: Dict) -> str:
        if row.get("ship_ratio", 1.0) < 0.40:
            return "overwhelmed"
        if row.get("max_threat", 0) > 15 or row.get("total_threat", 0) > 40:
            return "threat_spike"
        if row.get("my_prod", 0) < 5 and row.get("my_ships", 0) < 20:
            return "starvation"
        if (row.get("neutral_opportunity", 0) > 1.0
                and row.get("my_planets", 1) <= 2
                and row.get("turn_frac", 0) < 0.25):
            return "expansion_lag"
        return "unknown"

    def fleet_divergence_turn(self) -> Optional[int]:
        """Index of first turn in a 5-turn window where ship_ratio stays < 0.85."""
        for i in range(5, len(self.rows)):
            if all(r.get("ship_ratio", 1.0) < 0.85 for r in self.rows[i - 5:i]):
                return i - 5
        return None

    def early_features(self) -> Optional[Dict]:
        """Mean feature values over the first 20% of turns."""
        early = [r for r in self.rows if r.get("turn_frac", 1.0) <= 0.20]
        if not early:
            return None
        return {k: float(np.mean([r.get(k, 0) for r in early])) for k in FEATURE_COLS}


def summarise_postmortems(pms: List[GamePostMortem]) -> Dict:
    """
    Aggregate stats across all post-mortems:
      - avg/min survival steps
      - cause distribution
      - games that reached 500
    """
    if not pms:
        return {}

    survivals = [pm.survival_steps for pm in pms]
    all_events = [ev for pm in pms for ev in pm.planet_loss_events()]
    cause_counts: Dict[str, int] = {}
    for ev in all_events:
        cause_counts[ev["cause"]] = cause_counts.get(ev["cause"], 0) + 1

    reached = sum(1 for pm in pms if pm.reached_500)

    print(f"\n[PostMortem] {len(pms)} games  "
          f"avg_survival={np.mean(survivals):.0f}/500  "
          f"min={min(survivals)}  max={max(survivals)}  "
          f"reached_500={reached}/{len(pms)}")
    print(f"  Planet-loss events: {len(all_events)} total")
    for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print(f"    {cause:<20} {cnt:>4} losses")

    return {
        "avg_survival":   float(np.mean(survivals)),
        "min_survival":   int(min(survivals)),
        "max_survival":   int(max(survivals)),
        "reached_500":    reached,
        "total_games":    len(pms),
        "cause_counts":   cause_counts,
        "total_losses":   len(all_events),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5 – SURVIVAL OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class SurvivalOptimizer:
    """
    GBR trained on early-game features → survival_steps.
    Identifies mortality drivers and nudges LEARNED_PARAMS to reach 500 turns.
    """

    # Feature → [(param_name, fractional_direction), ...]
    # direction > 0: increase param; < 0: decrease param
    PARAM_FEATURE_MAP: Dict[str, List[Tuple[str, float]]] = {
        "max_threat":          [("defend_threshold",    -0.10),
                                 ("min_hold_threat",     +0.15)],
        "total_threat":        [("min_hold_threat",     +0.10),
                                 ("defend_threshold",    -0.05)],
        "ship_ratio":          [("min_attack_avail",    +0.20),
                                 ("attack_buffer_ratio", -0.10)],
        "prod_ratio":          [("prod_weight",         +0.08),
                                 ("min_expand_avail",    -0.10)],
        "neutral_opportunity": [("min_expand_avail",    -0.15),
                                 ("prod_target_early",   +0.05)],
        "dominance":           [("min_attack_avail",    +0.10)],
        "leading":             [("consolidate_avail",   -0.10)],
    }

    PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
        "defend_threshold":    (1.0,  10.0),
        "min_hold_threat":     (2.0,  20.0),
        "min_attack_avail":    (2.0,  20.0),
        "attack_buffer_ratio": (0.05,  0.50),
        "prod_weight":         (3.0,  20.0),
        "min_expand_avail":    (1.0,  10.0),
        "prod_target_early":   (0.8,   2.0),
        "consolidate_avail":   (5.0,  30.0),
    }

    def __init__(self):
        self.regressor  = None
        self.importances: Dict[str, float] = {}

    def fit(self, postmortems: List[GamePostMortem]):
        """Build training set: early-game mean features → survival_steps."""
        X_rows, y = [], []
        for pm in postmortems:
            ef = pm.early_features()
            if ef is None:
                continue
            X_rows.append([ef.get(c, 0) for c in FEATURE_COLS])
            y.append(float(pm.survival_steps))

        if len(X_rows) < 10:
            print("  [Survival] not enough games to fit (<10)")
            return

        X = np.array(X_rows)
        y = np.array(y)

        self.regressor = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=3, random_state=42
        )
        self.regressor.fit(X, y)
        r2 = self.regressor.score(X, y)
        self.importances = dict(zip(FEATURE_COLS, self.regressor.feature_importances_))

        print(f"\n[SurvivalOptimizer] GBR  R²={r2:.3f}  "
              f"mean_survival={y.mean():.0f}/500  min={y.min():.0f}  max={y.max():.0f}")
        top = sorted(self.importances.items(), key=lambda x: -x[1])[:6]
        print("  Early-game predictors of survival:")
        for feat, imp in top:
            print(f"    {feat:<35} {imp:.4f}")

    def suggest_adjustments(self, params: Dict, avg_survival: float) -> Dict:
        """Return updated params dict with nudges targeting 500 survival steps."""
        if not self.importances:
            return params

        shortfall = max(0.0, (500 - avg_survival) / 500)
        if shortfall < 0.05:
            print("  [Survival] avg survival ≥475 – no param changes needed")
            return params

        print(f"\n  [Survival] shortfall={shortfall:.1%}  nudging params…")
        top_feats = sorted(self.importances, key=lambda f: -self.importances[f])
        updated   = dict(params)

        for feat in top_feats[:6]:
            if feat not in self.PARAM_FEATURE_MAP:
                continue
            for param_name, direction in self.PARAM_FEATURE_MAP[feat]:
                if param_name not in updated:
                    continue
                cur   = float(updated[param_name])
                delta = abs(cur) * abs(direction) * shortfall * (1 if direction > 0 else -1)
                new_val = float(np.clip(cur + delta,
                                        *self.PARAM_BOUNDS.get(param_name, (-1e9, 1e9))))
                if abs(new_val - cur) > 1e-4:
                    print(f"    {param_name:<30} {cur:.3f} → {new_val:.3f}  (driver: {feat})")
                    updated[param_name] = round(new_val, 3)

        return updated


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 6 – FLEET GROWTH ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class FleetGrowthAnalyzer:
    """
    Decision Tree trained on features at fleet-divergence moments to explain
    why enemy ships grow faster than ours and whether we can recover.
    """

    def __init__(self):
        self.dt          = None
        self.importances: Dict[str, float] = {}
        self.n_events    = 0

    def fit(self, postmortems: List[GamePostMortem]):
        div_rows = []
        for pm in postmortems:
            dt_idx = pm.fleet_divergence_turn()
            if dt_idx is None or dt_idx >= len(pm.rows):
                continue
            feat = {k: pm.rows[dt_idx].get(k, 0) for k in FEATURE_COLS}
            # Did we recover? ship_ratio > 1.0 within 50 turns after divergence
            recovered = any(
                pm.rows[j].get("ship_ratio", 0) > 1.0
                for j in range(dt_idx, min(dt_idx + 50, len(pm.rows)))
            )
            feat["recovered"] = int(recovered)
            div_rows.append(feat)

        self.n_events = len(div_rows)
        if self.n_events < 10:
            print(f"\n[FleetGrowth] Only {self.n_events} divergence events – skipping DT")
            return

        df_div = pd.DataFrame(div_rows)
        X = df_div[FEATURE_COLS].fillna(0).values
        y = df_div["recovered"].values

        if len(np.unique(y)) < 2:
            print(f"\n[FleetGrowth] All {self.n_events} divergences ended the same way – skipping")
            return

        self.dt = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42
        )
        self.dt.fit(X, y)
        self.importances = dict(zip(FEATURE_COLS, self.dt.feature_importances_))

        recovery_rate = y.mean()
        print(f"\n[FleetGrowth] DT on {self.n_events} divergence events  "
              f"(recovery rate={recovery_rate:.1%})")
        print("  What determines recovery after enemy outpaces us:")
        print(export_text(self.dt, feature_names=FEATURE_COLS, max_depth=3))

    def report(self) -> str:
        if not self.importances:
            return "  [FleetGrowth] insufficient divergence data"
        top = sorted(self.importances.items(), key=lambda x: -x[1])[:5]
        lines = [f"  Fleet-divergence key factors ({self.n_events} events):"]
        for f, imp in top:
            lines.append(f"    {f:<35} {imp:.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 7 – ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyLearner:
    def __init__(self):
        self.win_rf   = None
        self.win_gbc  = None
        self.dt       = None
        self.et       = None
        self.scaler   = StandardScaler()
        self.feat_imp = {}

    def train_win_predictor(self, df: pd.DataFrame) -> Dict:
        X = df[FEATURE_COLS].fillna(0).values
        y = df["won"].values.astype(int)

        if len(np.unique(y)) < 2:
            print("  [ML] Only one class in labels – skipping model training.")
            return {}

        X_s = self.scaler.fit_transform(X)

        self.win_rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
        cv_rf = cross_val_score(
            self.win_rf, X_s, y, cv=StratifiedKFold(3), scoring="roc_auc")
        self.win_rf.fit(X_s, y)
        print(f"  [RF]  win-predictor  AUC={cv_rf.mean():.3f}±{cv_rf.std():.3f}")

        self.win_gbc = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        cv_gbc = cross_val_score(
            self.win_gbc, X, y, cv=StratifiedKFold(3), scoring="roc_auc")
        self.win_gbc.fit(X, y)
        print(f"  [GBC] win-predictor  AUC={cv_gbc.mean():.3f}±{cv_gbc.std():.3f}")

        self.dt = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10,
            class_weight="balanced", random_state=42)
        self.dt.fit(X, y)
        print("  [DT]  strategy tree trained")
        print(export_text(self.dt, feature_names=FEATURE_COLS, max_depth=3))

        self.et = ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1)
        self.et.fit(X, y)

        imp = pd.Series(self.et.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        self.feat_imp = imp.to_dict()
        print("  Top-10 features:")
        print(imp.head(10).to_string())

        return {"rf_auc": cv_rf.mean(), "gbc_auc": cv_gbc.mean()}

    def derive_params(self, df: pd.DataFrame,
                      pm_summary: Optional[Dict] = None) -> Dict:
        """
        Derive strategy constants from winning-game statistics + post-mortem causes.
        pm_summary: output of summarise_postmortems(), used to skew defend/expand params.
        """
        if df.empty or "won" not in df.columns:
            return {}

        wins  = df[df["won"] == 1]
        loses = df[df["won"] == 0]

        def pct(series, p): return float(np.percentile(series, p)) if len(series) else 0.0

        aggressive_ratio = pct(wins["ship_ratio"],  25)
        defensive_ratio  = pct(loses["ship_ratio"], 75)

        early_wins   = wins[wins["early_game"] == 1]
        prod_target  = pct(early_wins["prod_ratio"], 50) if not early_wins.empty else 1.2

        threatened_wins = wins[wins["max_threat"] > 0]
        defend_thresh   = max(1, int(pct(threatened_wins["max_threat"], 70)))

        late_wins = wins[wins["late_game"] == 1]
        consolidate_thresh = max(10, int(
            pct(late_wins["my_ships"], 25) / max(1, pct(late_wins["my_planets"], 50))
        ))

        attack_buffer = 0.25 if wins["ship_ratio"].mean() > 1.5 else 0.15
        prod_corr     = df[["prod_ratio", "won"]].corr().iloc[0, 1]
        prod_weight   = round(max(5.0, min(15.0, 10.0 * prod_corr * 2 + 8.0)), 1)

        params = {
            "prod_weight":           prod_weight,
            "ship_cost_weight":      1.0,
            "dist_weight":           0.15,
            "early_end_turn":        int(pct(wins["turn_frac"], 20) * 500),
            "late_start_turn":       int(pct(wins["turn_frac"], 70) * 500),
            "defend_threshold":      defend_thresh,
            "min_hold_base":         1,
            "min_hold_threat":       max(3, defend_thresh // 2),
            "attack_buffer_ratio":   round(attack_buffer, 2),
            "min_attack_avail":      4,
            "min_expand_avail":      2,
            "consolidate_avail":     max(10, consolidate_thresh),
            "consolidate_frac":      0.5,
            "aggressive_ship_ratio": round(max(1.1, aggressive_ratio), 2),
            "defensive_ship_ratio":  round(max(0.5, min(1.0, defensive_ratio)), 2),
            "prod_target_early":     round(max(1.0, prod_target), 2),
        }

        # Adjust defend/expand params based on dominant planet-loss cause
        if pm_summary:
            causes = pm_summary.get("cause_counts", {})
            total  = sum(causes.values()) or 1
            dominant = max(causes, key=causes.get) if causes else None

            if dominant == "threat_spike":
                # React to threats earlier and hold more ships
                params["defend_threshold"] = max(1, params["defend_threshold"] - 1)
                params["min_hold_threat"]  = min(20, params["min_hold_threat"] + 2)
                print("  [Params] Tightening defence (dominant loss cause: threat_spike)")
            elif dominant == "overwhelmed":
                # Build a bigger buffer before attacking
                params["min_attack_avail"]    = min(20, params["min_attack_avail"] + 2)
                params["attack_buffer_ratio"] = min(0.5, params["attack_buffer_ratio"] + 0.05)
                print("  [Params] Raising attack threshold (dominant loss cause: overwhelmed)")
            elif dominant in ("starvation", "expansion_lag"):
                # Expand more aggressively early
                params["min_expand_avail"] = max(1, params["min_expand_avail"] - 1)
                params["prod_weight"]      = min(20.0, params["prod_weight"] + 1.0)
                print(f"  [Params] Boosting expansion (dominant loss cause: {dominant})")

        return params


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 8 – PROPHET SHIP-COUNT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def prophet_analysis(ship_series_list: List[Dict]) -> Dict:
    if not HAS_PROPHET:
        print("  [Prophet] not installed – skipping")
        return {}
    if not ship_series_list:
        return {}

    won_series  = [s for s in ship_series_list if s["won"] == 1]
    lost_series = [s for s in ship_series_list if s["won"] == 0]
    insights    = {}

    for label, series_list in [("won", won_series), ("lost", lost_series)]:
        if not series_list:
            continue
        max_len = max(len(s["series"]) for s in series_list)
        stacked = []
        for s in series_list:
            arr    = np.array(s["series"], dtype=float)
            interp = np.interp(np.linspace(0, 1, max_len),
                               np.linspace(0, 1, len(arr)), arr)
            stacked.append(interp)
        mean_series = np.mean(stacked, axis=0)

        df_p = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=max_len, freq="h"),
            "y":  mean_series,
        })
        m = Prophet(changepoint_prior_scale=0.3, yearly_seasonality=False,
                    weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_p)

        cp_turns = []
        for cp in m.changepoints:
            idx = (cp - df_p["ds"].iloc[0]).total_seconds() / 3600
            cp_turns.append(int(round(idx)))

        n20        = max_len // 5
        early_rate = (mean_series[n20] - mean_series[0]) / max(1, n20)
        late_rate  = (mean_series[-1] - mean_series[-n20]) / max(1, n20)

        print(f"  [Prophet] {label}: changepoints≈{cp_turns[:3]}  "
              f"early_rate={early_rate:.1f}/turn  late_rate={late_rate:.1f}/turn")

        insights[label] = {
            "changepoints": cp_turns,
            "early_rate":   round(early_rate, 2),
            "late_rate":    round(late_rate,  2),
            "peak_turn":    int(np.argmax(mean_series)),
        }

    if "won" in insights and "lost" in insights:
        insights["diverge_turn"] = min(
            insights["won"].get("changepoints", [100])[0], 100)

    return insights


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 9 – MAIN.PY UPDATER
# ═══════════════════════════════════════════════════════════════════════════════

_PARAMS_MARKER_START = "# <<LEARNED_PARAMS_START>>"
_PARAMS_MARKER_END   = "# <<LEARNED_PARAMS_END>>"

def update_main_py(params: Dict) -> bool:
    if not MAIN_PY.exists():
        print(f"  [Updater] {MAIN_PY} not found")
        return False

    src   = MAIN_PY.read_text()
    lines = ["LEARNED_PARAMS = {"]
    for k, v in params.items():
        lines.append(f'    "{k}": {repr(v)},')
    lines.append("}")
    new_block = (
        f"{_PARAMS_MARKER_START}\n"
        + "\n".join(lines)
        + f"\n{_PARAMS_MARKER_END}"
    )

    if _PARAMS_MARKER_START in src and _PARAMS_MARKER_END in src:
        pattern = re.compile(
            re.escape(_PARAMS_MARKER_START) + r".*?" + re.escape(_PARAMS_MARKER_END),
            re.DOTALL,
        )
        new_src = pattern.sub(new_block, src)
    else:
        print(f"  [Updater] Markers not found in {MAIN_PY}")
        return False

    MAIN_PY.write_text(new_src)
    print(f"  [Updater] main.py updated with {len(params)} parameters.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 10 – SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def print_strategy_summary(learner: StrategyLearner, params: Dict,
                            prophet: Dict, pm_stats: Dict,
                            survival_opt: SurvivalOptimizer,
                            fleet_growth: FleetGrowthAnalyzer):
    sep = "=" * 65
    print(f"\n{sep}")
    print("STRATEGY LEARNING SUMMARY")
    print(sep)

    # Win predictor features
    if learner.feat_imp:
        top = sorted(learner.feat_imp.items(), key=lambda x: -x[1])[:6]
        print("\nTop-6 win-predictor features:")
        for fname, imp in top:
            print(f"  {fname:<35} {imp:.4f}")

    # Survival stats
    if pm_stats:
        print(f"\nSurvival stats:")
        print(f"  avg={pm_stats['avg_survival']:.0f}/500  "
              f"min={pm_stats['min_survival']}  max={pm_stats['max_survival']}  "
              f"reached_500={pm_stats['reached_500']}/{pm_stats['total_games']}")
        if pm_stats.get("cause_counts"):
            print("  Planet-loss causes:")
            for cause, cnt in sorted(pm_stats["cause_counts"].items(), key=lambda x: -x[1]):
                pct = 100 * cnt / max(1, pm_stats["total_losses"])
                print(f"    {cause:<20} {cnt:>4} ({pct:.0f}%)")

    # Fleet growth
    print(f"\n{fleet_growth.report()}")

    # Survival optimizer features
    if survival_opt.importances:
        top_s = sorted(survival_opt.importances.items(), key=lambda x: -x[1])[:5]
        print("\nSurvival GBR – top drivers of early death:")
        for f, imp in top_s:
            print(f"  {f:<35} {imp:.4f}")

    # Prophet
    if prophet:
        print("\nProphet ship-count insights:")
        for label, info in prophet.items():
            if isinstance(info, dict):
                print(f"  {label}: {info}")

    # Final params
    print("\nFinal LEARNED_PARAMS:")
    for k, v in params.items():
        print(f"  {k:<35} = {v}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Orbit Wars Strategy Learner")
    parser.add_argument("--games",     type=int, default=20)
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("--load-pkl",  action="store_true")
    args = parser.parse_args()

    # ── Step 1: Log analysis ──────────────────────────────────────────────────
    log_report = LogAnalyzer().analyze()

    # ── Step 2: Game simulation ───────────────────────────────────────────────
    if not HAS_KAGGLE:
        print("\n[Sim] kaggle_environments not available – skipping simulation.")
        df, ship_series, postmortems = pd.DataFrame(), [], []
    elif args.load_pkl and PKL_PATH.exists():
        print(f"\n[Sim] Loading cached data from {PKL_PATH}")
        with open(PKL_PATH, "rb") as fp:
            cached = pickle.load(fp)
        df           = cached["df"]
        ship_series  = cached["ship_series"]
        postmortems  = cached.get("postmortems", [])
        print(f"  Loaded {len(df)} rows, {len(postmortems)} post-mortems")
    else:
        print(f"\n[Sim] Running {args.games} games…")
        df, ship_series, postmortems = run_multiple_games(n_games=args.games)
        print(f"  Captured {len(df)} feature rows, {len(postmortems)} post-mortems")
        with open(PKL_PATH, "wb") as fp:
            pickle.dump({"df": df, "ship_series": ship_series,
                         "postmortems": postmortems,
                         "log_report": log_report}, fp)
        print(f"  Saved to {PKL_PATH}")

    # ── Step 3: Post-mortem aggregation ──────────────────────────────────────
    pm_stats = summarise_postmortems(postmortems)

    # ── Step 4: Fleet growth analysis ────────────────────────────────────────
    fleet_growth = FleetGrowthAnalyzer()
    print("\n[FleetGrowth] Fitting divergence DT…")
    fleet_growth.fit(postmortems)

    # ── Step 5: Survival optimizer ────────────────────────────────────────────
    survival_opt = SurvivalOptimizer()
    print("\n[SurvivalOptimizer] Fitting GBR…")
    survival_opt.fit(postmortems)

    # ── Step 6: Train win predictor ───────────────────────────────────────────
    learner  = StrategyLearner()
    ml_stats = {}
    if not df.empty and "won" in df.columns and len(df) >= 50:
        print(f"\n[ML] Training on {len(df)} samples  (win rate={df['won'].mean():.2%})")
        ml_stats = learner.train_win_predictor(df)
    else:
        print("\n[ML] Not enough data to train win predictor (need ≥50 rows).")

    # ── Step 7: Derive base parameters ───────────────────────────────────────
    params = learner.derive_params(df, pm_summary=pm_stats) if not df.empty else {}

    defaults = {
        "prod_weight": 10.0, "ship_cost_weight": 1.0, "dist_weight": 0.15,
        "early_end_turn": 100, "late_start_turn": 350,
        "defend_threshold": 3, "min_hold_base": 1, "min_hold_threat": 4,
        "attack_buffer_ratio": 0.20, "min_attack_avail": 4,
        "min_expand_avail": 2, "consolidate_avail": 15,
        "consolidate_frac": 0.50, "aggressive_ship_ratio": 1.50,
        "defensive_ship_ratio": 0.75, "prod_target_early": 1.20,
    }
    for k, v in defaults.items():
        params.setdefault(k, v)

    # ── Step 8: Prophet refinement ────────────────────────────────────────────
    print("\n[Prophet] Analysing ship-count trajectories…")
    prophet_insights = prophet_analysis(ship_series)
    if prophet_insights.get("diverge_turn"):
        params["early_end_turn"] = max(50, prophet_insights["diverge_turn"])

    # ── Step 9: Survival-driven param nudges ──────────────────────────────────
    avg_survival = pm_stats.get("avg_survival", 500.0)
    params = survival_opt.suggest_adjustments(params, avg_survival)

    # ── Step 10: Save ─────────────────────────────────────────────────────────
    bundle = {
        "win_rf": learner.win_rf, "win_gbc": learner.win_gbc,
        "dt": learner.dt, "et": learner.et,
        "scaler": learner.scaler, "params": params,
        "feat_imp": learner.feat_imp, "prophet": prophet_insights,
        "ml_stats": ml_stats, "pm_stats": pm_stats,
    }
    with open(PKL_PATH, "wb") as fp:
        pickle.dump({**bundle, "df": df, "ship_series": ship_series,
                     "postmortems": postmortems, "log_report": log_report}, fp)
    print(f"\n[Save] Models + data saved to {PKL_PATH}")

    # ── Step 11: Summary ──────────────────────────────────────────────────────
    print_strategy_summary(learner, params, prophet_insights,
                            pm_stats, survival_opt, fleet_growth)

    # ── Step 12: Write main.py ────────────────────────────────────────────────
    if not args.no_update:
        print("\n[Update] Writing learned params to main.py…")
        update_main_py(params)
    else:
        print("\n[Update] --no-update set, skipping main.py update.")

    print("\nDone. Re-run with --load-pkl to retrain on cached data.")


if __name__ == "__main__":
    main()
