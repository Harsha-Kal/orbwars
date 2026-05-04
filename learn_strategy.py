"""
Orbit Wars Strategy Learner (v3 - Surrogate Optimizer + Post-Mortem)
====================================================================
Reads agent logs, runs local simulations with randomized parameters, 
trains a surrogate model (Parameters -> Win Rate), and performs 
post-mortem analysis to identify mortality drivers.

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
#  PART 1 – POST-MORTEM: WHY DID WE LOSE PLANETS / DIE EARLY?
# ═══════════════════════════════════════════════════════════════════════════════

class GamePostMortem:
    def __init__(self, rows: List[Dict], game_i: int = 0, won: bool = False, n_turns: int = 0, params: Dict = None):
        self.rows    = rows
        self.game_i  = game_i
        self.won     = won
        self.n_turns = n_turns or len(rows)
        self.params  = params or {}

    @property
    def survival_steps(self) -> int:
        for row in reversed(self.rows):
            if row.get("my_planets", 0) > 0:
                return round(row["turn_frac"] * 500)
        return 0

    def planet_loss_events(self) -> List[Dict]:
        events = []
        for i in range(1, len(self.rows)):
            prev, curr = self.rows[i-1], self.rows[i]
            if curr.get("my_planets", 0) < prev.get("my_planets", 1):
                events.append({
                    "game_i": self.game_i, "turn_frac": curr["turn_frac"],
                    "cause": self._diagnose(prev),
                    "ship_ratio": prev.get("ship_ratio", 1.0),
                    "max_threat": prev.get("max_threat", 0),
                })
        return events

    def _diagnose(self, row: Dict) -> str:
        if row.get("ship_ratio", 1.0) < 0.40: return "overwhelmed"
        if row.get("max_threat", 0) > 15:     return "threat_spike"
        if row.get("my_prod", 0) < 5:        return "starvation"
        return "unknown"

def summarise_postmortems(pms: List[GamePostMortem]) -> Dict:
    if not pms: return {}
    survivals = [pm.survival_steps for pm in pms]
    all_events = [ev for pm in pms for ev in pm.planet_loss_events()]
    cause_counts = {}
    for ev in all_events:
        cause_counts[ev["cause"]] = cause_counts.get(ev["cause"], 0) + 1
    
    print(f"\n[PostMortem] {len(pms)} games analyzed")
    print(f"  Avg survival: {np.mean(survivals):.1f}/500 steps")
    for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print(f"    {cause:<15} {cnt:>4} losses")
    return {"avg_survival": np.mean(survivals), "cause_counts": cause_counts}

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 – GAME SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def _sg(obj, key, default=None):
    if obj is None: return default
    if isinstance(obj, dict): return obj.get(key, default)
    return getattr(obj, key, default)

def run_single_game(main_agent, opponent="random", seed: Optional[int] = None) -> Tuple[List, int, float]:
    cfg = {"seed": seed} if seed is not None else {}
    env = kag_make("orbit_wars", configuration=cfg, debug=False)
    steps = env.run([main_agent, opponent])
    
    rows = []
    for turn_idx, step in enumerate(steps):
        for player_idx, pstate in enumerate(step):
            obs, reward = _sg(pstate, "observation"), _sg(pstate, "reward")
            if obs and player_idx == 0:
                rows.append({"turn": turn_idx, "obs": obs, "reward": reward, "max_turn": len(steps)})
                
    final_rewards = [_sg(ps, "reward") or 0 for ps in steps[-1]]
    winner = int(np.argmax(final_rewards)) if final_rewards else 0
    return rows, winner, float(final_rewards[0])

def sample_random_params() -> Dict:
    res = {}
    for k, (low, high, ptype) in PARAM_SPACE.items():
        if ptype == int: res[k] = int(np.random.randint(low, high + 1))
        else: res[k] = round(float(np.random.uniform(low, high)), 3)
    return res

def run_multiple_games(n_games: int = 50, opponents=("random",)) -> Tuple[pd.DataFrame, List[GamePostMortem]]:
    import main as main_module
    results, postmortems = [], []
    print(f"\n[Sim] Starting {n_games} games with randomized parameters...")
    
    for i in range(n_games):
        params = sample_random_params()
        main_module.LEARNED_PARAMS = params
        opponent = opponents[i % len(opponents)]
        seed = 42 + i * 7
        
        try:
            rows, winner, reward = run_single_game(main_module.agent, opponent=opponent, seed=seed)
            won = 1 if winner == 0 else 0
            
            # Extract features for each turn for post-mortem, but only use final reward for surrogate
            feat_rows = []
            for r in rows:
                from learn_strategy import _extract_features # late import
                f = _extract_features(r["obs"], r["turn"], r["max_turn"])
                if f: feat_rows.append(f)
            
            if feat_rows:
                results.append({**params, "won": won, "reward": reward})
                postmortems.append(GamePostMortem(feat_rows, game_i=i, won=(won==1), params=params))
            
            if (i+1) % 5 == 0:
                print(f"  Progress: {i+1}/{n_games} | Win Rate: {np.mean([r['won'] for r in results]):.1%}")
        except Exception as e:
            print(f"  Error in game {i}: {e}")

    return pd.DataFrame(results), postmortems

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 – FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _fleet_heads_to(fl_dict, planet_dict, tol=0.40) -> bool:
    ea = math.atan2(planet_dict["y"] - fl_dict["y"], planet_dict["x"] - fl_dict["x"])
    diff = abs(math.atan2(math.sin(fl_dict["angle"] - ea), math.cos(fl_dict["angle"] - ea)))
    return diff < tol

def _parse_obs(obs) -> Tuple[List, List, int]:
    planets_raw = _sg(obs, "planets", [])
    fleets_raw  = _sg(obs, "fleets",  [])
    player      = int(_sg(obs, "player", 0))
    planets = [{"id": int(p[0]), "owner": int(p[1]), "radius": float(p[4]), "ships": int(p[5]), "prod": int(p[6]), "x": float(p[2]), "y": float(p[3])} for p in planets_raw]
    fleets = [{"owner": int(f[1]), "ships": int(f[6]), "x": float(f[2]), "y": float(f[3]), "angle": float(f[4])} for f in fleets_raw]
    return planets, fleets, player

def _extract_features(obs, turn: int, max_turn: int) -> Optional[Dict]:
    try: planets, fleets, me = _parse_obs(obs)
    except: return None
    my_p = [p for p in planets if p["owner"] == me]; en_p = [p for p in planets if p["owner"] not in (-1, me)]; ne_p = [p for p in planets if p["owner"] == -1]
    my_fl = [f for f in fleets if f["owner"] == me]; en_fl = [f for f in fleets if f["owner"] != me]
    my_ships = sum(p["ships"] for p in my_p); en_ships = sum(p["ships"] for p in en_p); ne_ships = sum(p["ships"] for p in ne_p)
    my_fl_ships = sum(f["ships"] for f in my_fl); en_fl_ships = sum(f["ships"] for f in en_fl)
    my_prod = sum(p["prod"] for p in my_p); en_prod = sum(p["prod"] for p in en_p)
    threat_vals = []
    for p in my_p:
        inbound_en = sum(f["ships"] for f in en_fl if _fleet_heads_to(f, p))
        inbound_my = sum(f["ships"] for f in my_fl if _fleet_heads_to(f, p))
        net = p["ships"] + inbound_my - inbound_en
        threat_vals.append(max(0, -net))
    max_threat = max(threat_vals) if threat_vals else 0
    ship_ratio = (my_ships + my_fl_ships) / max(1, en_ships + en_fl_ships)
    turn_frac = turn / max(1, max_turn)
    return {
        "turn_frac": turn_frac, "my_ships": my_ships, "enemy_ships": en_ships, "neutral_ships": ne_ships,
        "ship_ratio": min(ship_ratio, 10.0), "my_prod": my_prod, "enemy_prod": en_prod,
        "prod_ratio": min(my_prod / max(1, en_prod), 10.0), "my_planets": len(my_p),
        "enemy_planets": len(en_p), "neutral_planets": len(ne_p), "max_threat": max_threat,
        "early_game": 1.0 if turn_frac < 0.20 else 0.0, "mid_game": 1.0 if 0.20 <= turn_frac < 0.70 else 0.0,
        "late_game": 1.0 if turn_frac >= 0.70 else 0.0,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4 – MODEL-BASED OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class SurrogateOptimizer:
    def __init__(self, space: Dict):
        self.space = space
        self.model = ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42)
        self.feature_names = list(space.keys())

    def fit(self, df: pd.DataFrame):
        X, y = df[self.feature_names].values, df["reward"].values
        self.model.fit(X, y)
        print(f"  [Model] Surrogate trained. Best seen reward: {y.max():.4f}")

    def optimize(self, n_trials=10000) -> Dict:
        candidates = []
        for _ in range(n_trials):
            cand = []
            for k in self.feature_names:
                low, high, ptype = self.space[k]
                if ptype == int: cand.append(np.random.randint(low, high + 1))
                else: cand.append(np.random.uniform(low, high))
            candidates.append(cand)
        candidates = np.array(candidates)
        preds = self.model.predict(candidates)
        best_vec = candidates[np.argmax(preds)]
        res = {}
        for i, k in enumerate(self.feature_names):
            ptype = self.space[k][2]
            res[k] = int(round(best_vec[i])) if ptype == int else round(float(best_vec[i]), 3)
        return res

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5 – UPDATER & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def update_main_py(params: Dict):
    if not MAIN_PY.exists(): return
    src = MAIN_PY.read_text(encoding="utf-8")
    start_m, end_m = "# <<LEARNED_PARAMS_START>>", "# <<LEARNED_PARAMS_END>>"
    if start_m not in src or end_m not in src: return
    lines = ["LEARNED_PARAMS = {"]
    for k, v in params.items(): lines.append(f'    "{k}": {repr(v)},')
    lines.append("}")
    new_block = f"{start_m}\n" + "\n".join(lines) + f"\n{end_m}"
    pattern = re.compile(re.escape(start_m) + r".*?" + re.escape(end_m), re.DOTALL)
    MAIN_PY.write_text(pattern.sub(new_block, src), encoding="utf-8")
    print(f"  [Updater] main.py updated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--no-update", action="store_true")
    args = parser.parse_args()

    if not HAS_KAGGLE: return
    df, postmortems = run_multiple_games(n_games=args.games)
    
    summarise_postmortems(postmortems)
    
    opt = SurrogateOptimizer(PARAM_SPACE)
    opt.fit(df)
    best_params = opt.optimize()

    print("\n" + "="*60 + "\nOPTIMIZED PARAMETERS\n" + "="*60)
    for k, v in best_params.items(): print(f"  {k:<25} = {v}")
    
    if not args.no_update:
        update_main_py(best_params)
        # Save to pickle so main.py can load models + params
        with open(PKL_PATH, "wb") as fp:
            pickle.dump({
                "params": best_params,
                "df": df,
                "model": opt.model,
                "space": PARAM_SPACE
            }, fp)
        print(f"  [Save] {PKL_PATH} updated.")


if __name__ == "__main__":
    main()
