"""
Orbit Wars Strategy Learner (v2 - Surrogate Optimizer)
======================================================
Reads agent logs, runs local simulations with randomized parameters, 
trains a surrogate model (Parameters -> Win Rate), and optimizes 
the strategy via acquisition function search.

Usage:
  python learn_strategy.py [--games 50] [--no-update]
"""

import json, math, os, pickle, re, warnings, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── ML imports ────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

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

# ── Parameter Search Space ────────────────────────────────────────────────────
PARAM_SPACE = {
    "prod_weight":           (5.0, 30.0, float),
    "ship_cost_weight":      (0.5, 2.5,  float),
    "dist_weight":           (0.05, 0.5, float),
    "early_end_turn":        (50, 150,   int),
    "late_start_turn":       (250, 450,  int),
    "defend_threshold":      (1, 20,     int),
    "min_hold_base":         (0, 5,      int),
    "min_hold_threat":       (1, 10,     int),
    "attack_buffer_ratio":   (0.0, 0.5,  float),
    "min_attack_avail":      (1, 10,     int),
    "min_expand_avail":      (1, 10,     int),
    "consolidate_avail":     (20, 150,   int),
    "consolidate_frac":      (0.2, 0.8,  float),
    "aggressive_ship_ratio": (1.0, 5.0,  float),
    "defensive_ship_ratio":  (0.5, 1.5,  float),
    "prod_target_early":     (1.0, 2.5,  float),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 – LOG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class LogAnalyzer:
    """Parse Kaggle agent-log JSON files."""
    def analyze(self) -> Dict:
        if not LOG_DIR.exists():
            return {}
        files = sorted(LOG_DIR.glob("*.json"))
        all_dur = []
        for fpath in files:
            try:
                with open(fpath) as fp:
                    data = json.load(fp)
                for step in data:
                    for entry in step:
                        if isinstance(entry, dict) and "duration" in entry:
                            all_dur.append(entry["duration"])
            except: continue
        
        if not all_dur: return {}
        return {
            "avg_ms": np.mean(all_dur) * 1000,
            "max_ms": np.max(all_dur) * 1000,
            "p95_ms": np.percentile(all_dur, 95) * 1000,
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 – GAME SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _sg(obj, key, default=None):
    if obj is None: return default
    if isinstance(obj, dict): return obj.get(key, default)
    return getattr(obj, key, default)

def run_single_game(main_agent, opponent="random", seed: Optional[int] = None) -> Tuple[int, float]:
    """Run one game, return (winner_idx, final_reward)."""
    cfg = {"seed": seed} if seed is not None else {}
    env = kag_make("orbit_wars", configuration=cfg, debug=False)
    env.run([main_agent, opponent])
    final_rewards = [_sg(ps, "reward") or 0 for ps in env.steps[-1]]
    winner = int(np.argmax(final_rewards)) if final_rewards else 0
    return winner, float(final_rewards[0])

def sample_random_params() -> Dict:
    res = {}
    for k, (low, high, ptype) in PARAM_SPACE.items():
        if ptype == int:
            res[k] = int(np.random.randint(low, high + 1))
        else:
            res[k] = round(float(np.random.uniform(low, high)), 3)
    return res

def run_multiple_games(n_games: int = 50, opponents=("random",)) -> pd.DataFrame:
    """Run n_games with randomized parameters to collect training data."""
    import main as main_module
    
    results = []
    print(f"\n[Sim] Starting {n_games} games with randomized parameters...")
    
    start_time = time.time()
    for i in range(n_games):
        # 1. Sample random parameters
        params = sample_random_params()
        
        # 2. Inject into main module
        main_module.LEARNED_PARAMS = params
        
        # 3. Run game
        opponent = opponents[i % len(opponents)]
        seed = int(time.time() * 1000) % 10000 + i
        
        try:
            winner, reward = run_single_game(main_module.agent, opponent=opponent, seed=seed)
            won = 1 if winner == 0 else 0
            
            # 4. Store (Params + Result)
            row = {**params, "won": won, "reward": reward}
            results.append(row)
            
            if (i+1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i+1}/{n_games} | Win Rate: {np.mean([r['won'] for r in results]):.1%} | {elapsed:.1f}s")
        except Exception as e:
            print(f"  Error in game {i}: {e}")

    return pd.DataFrame(results)

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 – MODEL-BASED OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class SurrogateOptimizer:
    def __init__(self, space: Dict):
        self.space = space
        self.model = ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42)
        self.feature_names = list(space.keys())
        self.best_params = {}

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_names].values
        # Target is 'reward' (continuous)
        y = df["reward"].values 
        
        self.model.fit(X, y)
        print(f"  [Model] Surrogate trained on {len(df)} samples.")
        
        # Feature importance
        imp = pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        print("  Top sensitive parameters:")
        print(imp.head(5).to_string())

    def optimize(self, n_trials=10000) -> Dict:
        """Search the surrogate model for the best parameter combination."""
        print(f"  [Optimize] Searching {n_trials} candidates via surrogate model...")
        
        # 1. Generate many random candidates
        candidates = []
        for _ in range(n_trials):
            cand = []
            for k in self.feature_names:
                low, high, ptype = self.space[k]
                if ptype == int:
                    cand.append(np.random.randint(low, high + 1))
                else:
                    cand.append(np.random.uniform(low, high))
            candidates.append(cand)
        
        candidates = np.array(candidates)
        
        # 2. Predict rewards
        preds = self.model.predict(candidates)
        
        # 3. Pick best
        best_idx = np.argmax(preds)
        best_vec = candidates[best_idx]
        
        # 4. Refine with a small local search (Hill Climbing on the model)
        current_best = best_vec
        current_score = preds[best_idx]
        
        for _ in range(500):
            # Mutate
            mutation = current_best + np.random.normal(0, 0.05 * (current_best + 1e-6), size=len(current_best))
            # Clip to bounds
            for i, k in enumerate(self.feature_names):
                low, high, ptype = self.space[k]
                mutation[i] = np.clip(mutation[i], low, high)
            
            score = self.model.predict([mutation])[0]
            if score > current_score:
                current_best = mutation
                current_score = score
        
        # Format result
        res = {}
        for i, k in enumerate(self.feature_names):
            ptype = self.space[k][2]
            val = current_best[i]
            res[k] = int(round(val)) if ptype == int else round(float(val), 3)
            
        self.best_params = res
        print(f"  [Optimize] Best predicted reward: {current_score:.4f}")
        return res

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 – UPDATER & UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def update_main_py(params: Dict):
    if not MAIN_PY.exists(): return
    src = MAIN_PY.read_text()
    
    start_m, end_m = "# <<LEARNED_PARAMS_START>>", "# <<LEARNED_PARAMS_END>>"
    if start_m not in src or end_m not in src:
        print("  [Error] Markers not found in main.py")
        return

    lines = ["LEARNED_PARAMS = {"]
    for k, v in params.items():
        lines.append(f'    "{k}": {repr(v)},')
    lines.append("}")
    new_block = f"{start_m}\n" + "\n".join(lines) + f"\n{end_m}"
    
    pattern = re.compile(re.escape(start_m) + r".*?" + re.escape(end_m), re.DOTALL)
    MAIN_PY.write_text(pattern.sub(new_block, src))
    print(f"  [Updater] main.py updated with {len(params)} optimized parameters.")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Orbit Wars Strategy Learner v2")
    parser.add_argument("--games", type=int, default=50, help="Number of simulation games (default 50)")
    parser.add_argument("--no-update", action="store_true", help="Skip updating main.py")
    parser.add_argument("--load-pkl", action="store_true", help="Load existing data")
    args = parser.parse_args()

    # 1. Load or Generate Data
    if args.load_pkl and PKL_PATH.exists():
        print(f"[Data] Loading from {PKL_PATH}")
        with open(PKL_PATH, "rb") as f:
            df = pickle.load(f)
    else:
        if not HAS_KAGGLE:
            print("[Error] kaggle_environments not installed.")
            return
        df = run_multiple_games(n_games=args.games)
        with open(PKL_PATH, "wb") as f:
            pickle.dump(df, f)

    if df.empty:
        print("[Error] No data collected.")
        return

    # 2. Optimize
    opt = SurrogateOptimizer(PARAM_SPACE)
    opt.fit(df)
    best_params = opt.optimize()

    # 3. Print Summary
    print("\n" + "="*60)
    print("OPTIMIZED PARAMETERS")
    print("="*60)
    for k, v in best_params.items():
        print(f"  {k:<25} = {v}")
    print("="*60)

    # 4. Update
    if not args.no_update:
        update_main_py(best_params)

if __name__ == "__main__":
    main()
