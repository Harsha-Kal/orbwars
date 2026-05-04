"""
Orbit Wars Strategy Learner
============================
Reads agent logs, runs local simulations, trains ML models, and rewrites
the LEARNED_PARAMS block in main.py with optimised strategy constants.

Models trained
--------------
  RandomForestClassifier        – win-probability predictor
  GradientBoostingRegressor     – planet-value scorer
  DecisionTreeClassifier        – interpretable strategy tree
  ExtraTreesClassifier          – robust feature importance
  Prophet (optional)            – per-game ship-count trend analysis

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

# ── optional heavy imports ────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
LOG_DIR  = ROOT / "agent logs"
PKL_PATH = ROOT / "strategy_data.pkl"
MAIN_PY  = ROOT / "main.py"

# Feature columns shared by all models
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

        all_dur, all_errors, all_out, slow = [], [], [], []

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
                        all_errors.append({"file": fpath.name, "step": step_idx, "msg": entry["stderr"][:200]})
                    if entry.get("stdout"):
                        all_out.append(entry["stdout"][:100])

            if durs:
                print(f"  {fpath.name}: steps={len(data)}  avg={np.mean(durs)*1000:.2f}ms  "
                      f"max={np.max(durs)*1000:.2f}ms  p95={np.percentile(durs,95)*1000:.2f}ms")

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
    """Safe-get from Struct or dict."""
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
    """Return (planets_list, fleets_list, player_id) as plain dicts."""
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


def run_single_game(main_agent, opponent="random", seed: Optional[int] = None) -> Tuple[List, int, List]:
    """
    Run one game, return (game_data_rows, winner_idx, final_rewards).
    game_data_rows: list of {"turn", "player", "obs", "reward"} dicts.
    """
    cfg = {"seed": seed} if seed is not None else {}
    env = kag_make("orbit_wars", configuration=cfg, debug=False)
    steps = env.run([main_agent, opponent])

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


def run_multiple_games(n_games: int = 20, opponents=("random",)) -> Tuple[pd.DataFrame, List]:
    """Run n_games, return (feature DataFrame, list of ship-count series)."""
    import importlib, main as main_module

    all_rows   = []
    ship_series = []   # for Prophet

    for i in range(n_games):
        opponent = opponents[i % len(opponents)]
        seed = 42 + i * 7
        print(f"  Game {i+1}/{n_games}  opponent={opponent}  seed={seed}", end="  ")
        try:
            rows, winner, rewards = run_single_game(
                main_module.agent, opponent=opponent, seed=seed)
            print(f"winner={winner}  final_rewards={[round(r,1) for r in rewards]}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        # Per-turn features for player 0 (our agent)
        turns_p0 = [r for r in rows if r["player"] == 0]
        my_ships_per_turn = []

        for r in turns_p0:
            feats = _extract_features(r["obs"], r["turn"], r["max_turn"])
            if feats is None:
                continue
            feats["won"]    = 1 if winner == 0 else 0
            feats["game_i"] = i
            all_rows.append(feats)
            my_ships_per_turn.append(feats["my_ships"] + feats["my_fleets_ships"])

        if my_ships_per_turn:
            ship_series.append({
                "game_i": i, "won": 1 if winner == 0 else 0,
                "series": my_ships_per_turn,
            })

    df = pd.DataFrame(all_rows)
    return df, ship_series


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

    my_ships     = sum(p["ships"] for p in my_p)
    en_ships     = sum(p["ships"] for p in en_p)
    ne_ships     = sum(p["ships"] for p in ne_p)
    my_fl_ships  = sum(f["ships"] for f in my_fl)
    en_fl_ships  = sum(f["ships"] for f in en_fl)

    my_prod  = sum(p["prod"] for p in my_p)
    en_prod  = sum(p["prod"] for p in en_p)

    # Threat: deficit on each of our planets
    threat_vals = []
    for p in my_p:
        inbound_en = sum(f["ships"] for f in en_fl if _fleet_heads_to(f, p))
        inbound_my = sum(f["ships"] for f in my_fl if _fleet_heads_to(f, p))
        net = p["ships"] + inbound_my - inbound_en
        threat_vals.append(max(0, -net))

    max_threat   = max(threat_vals) if threat_vals else 0
    total_threat = sum(threat_vals)

    cheapest_ne   = min((p["ships"] for p in ne_p), default=0)
    best_ne_prod  = max((p["prod"]  for p in ne_p), default=0)
    ne_opp        = sum(p["prod"] / (p["ships"] + 1) for p in ne_p) if ne_p else 0

    weakest_en    = min((p["ships"] for p in en_p), default=0)
    best_attack   = max((p["prod"] / (p["ships"] + 1) for p in en_p), default=0)

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
#  PART 4 – ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyLearner:
    def __init__(self):
        self.win_rf   = None  # RandomForest win predictor
        self.win_gbc  = None  # GradientBoosting win predictor
        self.dt       = None  # Decision Tree (interpretable)
        self.et       = None  # ExtraTrees (feature importance)
        self.scaler   = StandardScaler()
        self.feat_imp = {}

    # ── train win predictor ───────────────────────────────────────────────────

    def train_win_predictor(self, df: pd.DataFrame) -> Dict:
        X = df[FEATURE_COLS].fillna(0).values
        y = df["won"].values.astype(int)

        if len(np.unique(y)) < 2:
            print("  [ML] Only one class in labels – skipping model training.")
            return {}

        X_s = self.scaler.fit_transform(X)

        # ── RandomForest ──
        self.win_rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
        cv_rf = cross_val_score(
            self.win_rf, X_s, y, cv=StratifiedKFold(3), scoring="roc_auc")
        self.win_rf.fit(X_s, y)
        print(f"  [RF]  win-predictor  AUC={cv_rf.mean():.3f}±{cv_rf.std():.3f}")

        # ── GradientBoosting ──
        self.win_gbc = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        cv_gbc = cross_val_score(
            self.win_gbc, X, y, cv=StratifiedKFold(3), scoring="roc_auc")
        self.win_gbc.fit(X, y)
        print(f"  [GBC] win-predictor  AUC={cv_gbc.mean():.3f}±{cv_gbc.std():.3f}")

        # ── Decision Tree (shallow, interpretable) ──
        self.dt = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10,
            class_weight="balanced", random_state=42)
        self.dt.fit(X, y)
        print("  [DT]  strategy tree trained")
        print(export_text(self.dt, feature_names=FEATURE_COLS, max_depth=3))

        # ── ExtraTrees (feature importance) ──
        self.et = ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1)
        self.et.fit(X, y)

        imp = pd.Series(self.et.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        self.feat_imp = imp.to_dict()
        print("  Top-10 features:")
        print(imp.head(10).to_string())

        return {"rf_auc": cv_rf.mean(), "gbc_auc": cv_gbc.mean()}

    # ── extract optimal strategy parameters from data ─────────────────────────

    def derive_params(self, df: pd.DataFrame) -> Dict:
        """
        Use empirical percentile analysis on winning turns to set
        strategy constants that will be written into main.py.
        """
        if df.empty or "won" not in df.columns:
            return {}

        wins  = df[df["won"] == 1]
        loses = df[df["won"] == 0]

        def pct(series, p): return float(np.percentile(series, p)) if len(series) else 0.0

        # Ship ratio: threshold above/below which we're usually winning/losing
        aggressive_ratio = pct(wins["ship_ratio"],  25)   # even bottom-quartile winners
        defensive_ratio  = pct(loses["ship_ratio"], 75)   # top-quartile losers

        # Early-game: median turn where prod_ratio first tips positive
        early_wins = wins[wins["early_game"] == 1]
        prod_target = pct(early_wins["prod_ratio"], 50) if not early_wins.empty else 1.2

        # Defend threshold: how negative net garrison before we react
        # Look at turns where we were threatened but still won
        threatened_wins = wins[wins["max_threat"] > 0]
        defend_thresh = max(1, int(pct(threatened_wins["max_threat"], 70)))

        # Consolidation threshold (ships available before forwarding)
        late_wins = wins[wins["late_game"] == 1]
        consolidate_thresh = max(10, int(pct(late_wins["my_ships"], 25) / max(1, pct(late_wins["my_planets"], 50))))

        # Attack buffer: how much over the minimum do winning games send?
        # proxy: in winning turns with high ship_ratio, how large are outgoing fleets?
        attack_buffer = 0.25 if wins["ship_ratio"].mean() > 1.5 else 0.15

        # Production weight: how correlated is prod_ratio with winning?
        prod_corr = df[["prod_ratio", "won"]].corr().iloc[0, 1]
        prod_weight = round(max(5.0, min(15.0, 10.0 * prod_corr * 2 + 8.0)), 1)

        return {
            "prod_weight":          prod_weight,
            "ship_cost_weight":     1.0,
            "dist_weight":          0.15,
            "early_end_turn":       int(pct(wins["turn_frac"], 20) * 500),
            "late_start_turn":      int(pct(wins["turn_frac"], 70) * 500),
            "defend_threshold":     defend_thresh,
            "min_hold_base":        1,
            "min_hold_threat":      max(3, defend_thresh // 2),
            "attack_buffer_ratio":  round(attack_buffer, 2),
            "min_attack_avail":     4,
            "min_expand_avail":     2,
            "consolidate_avail":    max(10, consolidate_thresh),
            "consolidate_frac":     0.5,
            "aggressive_ship_ratio": round(max(1.1, aggressive_ratio), 2),
            "defensive_ship_ratio":  round(max(0.5, min(1.0, defensive_ratio)), 2),
            "prod_target_early":     round(max(1.0, prod_target), 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5 – PROPHET SHIP-COUNT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def prophet_analysis(ship_series_list: List[Dict]) -> Dict:
    """
    Fit Prophet to ship-count trajectories and find:
      - typical inflection turns (changepoints) for winning vs losing games
      - growth rates in early / late phases
    """
    if not HAS_PROPHET:
        print("  [Prophet] not installed – skipping")
        return {}
    if not ship_series_list:
        return {}

    won_series  = [s for s in ship_series_list if s["won"] == 1]
    lost_series = [s for s in ship_series_list if s["won"] == 0]

    insights = {}

    for label, series_list in [("won", won_series), ("lost", lost_series)]:
        if not series_list:
            continue
        # Stack all series into one long DataFrame (rebase to same length via interpolation)
        max_len = max(len(s["series"]) for s in series_list)
        stacked = []
        for s in series_list:
            arr = np.array(s["series"], dtype=float)
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

        # Find changepoints (turns where growth rate changes)
        cp_turns = []
        for cp in m.changepoints:
            idx = (cp - df_p["ds"].iloc[0]).total_seconds() / 3600
            cp_turns.append(int(round(idx)))

        # Growth rate in first 20% vs last 20%
        n20 = max_len // 5
        early_rate = (mean_series[n20] - mean_series[0]) / max(1, n20)
        late_rate  = (mean_series[-1] - mean_series[-n20]) / max(1, n20)

        print(f"  [Prophet] {label}: changepoints≈{cp_turns[:3]}  "
              f"early_rate={early_rate:.1f}/turn  late_rate={late_rate:.1f}/turn")

        insights[label] = {
            "changepoints": cp_turns,
            "early_rate": round(early_rate, 2),
            "late_rate": round(late_rate, 2),
            "peak_turn": int(np.argmax(mean_series)),
        }

    # Derive a "danger turn" – earliest turn where winning games diverge from losing
    if "won" in insights and "lost" in insights:
        insights["diverge_turn"] = min(
            insights["won"].get("changepoints", [100])[0],
            100)

    return insights


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 6 – MAIN.PY UPDATER
# ═══════════════════════════════════════════════════════════════════════════════

_PARAMS_MARKER_START = "# <<LEARNED_PARAMS_START>>"
_PARAMS_MARKER_END   = "# <<LEARNED_PARAMS_END>>"

def update_main_py(params: Dict) -> bool:
    """Replace the LEARNED_PARAMS block in main.py with new values."""
    if not MAIN_PY.exists():
        print(f"  [Updater] {MAIN_PY} not found")
        return False

    src = MAIN_PY.read_text()

    lines = ["LEARNED_PARAMS = {"]
    for k, v in params.items():
        if isinstance(v, str):
            lines.append(f'    "{k}": "{v}",')
        else:
            lines.append(f'    "{k}": {repr(v)},')
    lines.append("}")
    new_block = (
        f"{_PARAMS_MARKER_START}\n"
        + "\n".join(lines)
        + f"\n{_PARAMS_MARKER_END}"
    )

    if _PARAMS_MARKER_START in src and _PARAMS_MARKER_END in src:
        # Replace existing block
        pattern = re.compile(
            re.escape(_PARAMS_MARKER_START) + r".*?" + re.escape(_PARAMS_MARKER_END),
            re.DOTALL,
        )
        new_src = pattern.sub(new_block, src)
    else:
        print(f"  [Updater] Markers not found in {MAIN_PY}. Skipping auto-update.")
        print("  Add these markers around the LEARNED_PARAMS dict in main.py:")
        print(f"    {_PARAMS_MARKER_START}")
        print(f"    {_PARAMS_MARKER_END}")
        return False

    MAIN_PY.write_text(new_src)
    print(f"  [Updater] main.py updated with {len(params)} parameters.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 7 – DECISION TREE → PYTHON RULES (for debugging / inspection)
# ═══════════════════════════════════════════════════════════════════════════════

def print_strategy_summary(learner: StrategyLearner, params: Dict, prophet: Dict):
    print("\n" + "=" * 60)
    print("STRATEGY LEARNING SUMMARY")
    print("=" * 60)

    if learner.feat_imp:
        top = sorted(learner.feat_imp.items(), key=lambda x: -x[1])[:6]
        print("\nTop-6 features that determine winning:")
        for fname, imp in top:
            print(f"  {fname:<35} importance={imp:.4f}")

    print("\nLearned parameters:")
    for k, v in params.items():
        print(f"  {k:<35} = {v}")

    if prophet:
        print("\nProphet ship-count analysis:")
        for label, info in prophet.items():
            if isinstance(info, dict):
                print(f"  {label}: {info}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Orbit Wars Strategy Learner")
    parser.add_argument("--games",     type=int, default=20,
                        help="Number of simulation games to run (default 20)")
    parser.add_argument("--no-update", action="store_true",
                        help="Do not write learned params back to main.py")
    parser.add_argument("--load-pkl",  action="store_true",
                        help="Load existing strategy_data.pkl instead of running new games")
    args = parser.parse_args()

    # ── Step 1: Analyse existing logs ─────────────────────────────────────────
    log_report = LogAnalyzer().analyze()

    # ── Step 2: Run games ──────────────────────────────────────────────────────
    if not HAS_KAGGLE:
        print("\n[Sim] kaggle_environments not available – skipping simulation.")
        df, ship_series = pd.DataFrame(), []
    elif args.load_pkl and PKL_PATH.exists():
        print(f"\n[Sim] Loading cached data from {PKL_PATH}")
        with open(PKL_PATH, "rb") as fp:
            cached = pickle.load(fp)
        df, ship_series = cached["df"], cached["ship_series"]
        print(f"  Loaded {len(df)} rows from {df['game_i'].nunique()} games")
    else:
        print(f"\n[Sim] Running {args.games} games…")
        df, ship_series = run_multiple_games(
            n_games=args.games,
            opponents=("random", "random"),   # add more opponents here if available
        )
        print(f"  Captured {len(df)} feature rows from {len(ship_series)} games")
        with open(PKL_PATH, "wb") as fp:
            pickle.dump({"df": df, "ship_series": ship_series,
                         "log_report": log_report}, fp)
        print(f"  Saved to {PKL_PATH}")

    # ── Step 3: Train ML models ────────────────────────────────────────────────
    learner = StrategyLearner()
    ml_stats = {}

    if not df.empty and "won" in df.columns and len(df) >= 50:
        print(f"\n[ML] Training on {len(df)} samples  "
              f"(win rate={df['won'].mean():.2%})")
        ml_stats = learner.train_win_predictor(df)
    else:
        print("\n[ML] Not enough data to train models (need ≥50 rows).")

    # ── Step 4: Derive optimal parameters ─────────────────────────────────────
    params = learner.derive_params(df) if not df.empty else {}

    # Defaults for any missing params
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

    # ── Step 5: Prophet analysis ───────────────────────────────────────────────
    print("\n[Prophet] Analysing ship-count trajectories…")
    prophet_insights = prophet_analysis(ship_series)
    if prophet_insights.get("diverge_turn"):
        params["early_end_turn"] = max(50, prophet_insights["diverge_turn"])

    # ── Step 6: Save models ────────────────────────────────────────────────────
    model_bundle = {
        "win_rf": learner.win_rf,
        "win_gbc": learner.win_gbc,
        "dt": learner.dt,
        "et": learner.et,
        "scaler": learner.scaler,
        "params": params,
        "feat_imp": learner.feat_imp,
        "prophet": prophet_insights,
        "ml_stats": ml_stats,
    }
    with open(PKL_PATH, "wb") as fp:
        pickle.dump({**model_bundle, "df": df, "ship_series": ship_series,
                     "log_report": log_report}, fp)
    print(f"\n[Save] Models + data saved to {PKL_PATH}")

    # ── Step 7: Print summary ──────────────────────────────────────────────────
    print_strategy_summary(learner, params, prophet_insights)

    # ── Step 8: Update main.py ─────────────────────────────────────────────────
    if not args.no_update:
        print("\n[Update] Writing learned params to main.py…")
        update_main_py(params)
    else:
        print("\n[Update] --no-update set, skipping main.py update.")

    print("\nDone. Run  python learn_strategy.py --load-pkl  to retrain on cached data.")


if __name__ == "__main__":
    main()
