import json
import random
import os
import copy
from kaggle_environments import make
import main as agent_module

# ── Configuration ─────────────────────────────────────────────────────────────

BEST_PARAMS_FILE = "best_params.json"
GAMES_PER_EVAL = 20  # 10 vs 10 to be fair
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.2

PARAM_SPACE = {
    'MIN_HOLD': (0, 20, int),
    'REINFORCE_NET_LIMIT': (0, 20, int),
    'REINFORCE_MIN_SEND': (1, 30, int),
    'REINFORCE_TARGET_NET': (5, 50, int),
    'NEUTRAL_VALUE_PROD_MULT': (1, 50, int),
    'ATTACK_DIST_WEIGHT': (0.01, 2.0, float),
    'ATTACK_PROD_WEIGHT': (0.1, 10.0, float),
    'ATTACK_BUFFER': (0, 50, int),
    'CONSOLIDATE_MIN_SURPLUS': (5, 100, int),
    'CONSOLIDATE_RATIO': (0.1, 0.9, float),
    'ATTACK_THRESHOLD_AV': (2, 50, int),
}

# ── Helper Functions ──────────────────────────────────────────────────────────

def load_best():
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE, "r") as f:
            return json.load(f)
    return copy.deepcopy(agent_module.PARAMS)

def save_best(params):
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)

def mutate(params):
    new_params = copy.deepcopy(params)
    for p, (low, high, p_type) in PARAM_SPACE.items():
        if random.random() < MUTATION_RATE:
            if p_type == int:
                change = random.randint(-max(1, int((high-low)*MUTATION_STRENGTH)), 
                                        max(1, int((high-low)*MUTATION_STRENGTH)))
                new_params[p] = max(low, min(high, new_params[p] + change))
            else:
                change = random.uniform(-(high-low)*MUTATION_STRENGTH, 
                                        (high-low)*MUTATION_STRENGTH)
                new_params[p] = max(low, min(high, new_params[p] + change))
    return new_params

def evaluate(p_a, p_b):
    """Run p_a vs p_b for GAMES_PER_EVAL games. Return p_a's win count."""
    a_wins = 0
    
    # Half games as player 0
    for seed in range(GAMES_PER_EVAL // 2):
        env = make("orbit_wars", debug=False)
        
        # We need a way to pass DIFFERENT params to each agent in the same environment.
        # Since they are in the same process, we'll use a wrapper.
        
        def agent_a(obs, cfg):
            agent_module.PARAMS = p_a
            return agent_module.agent(obs)
            
        def agent_b(obs, cfg):
            agent_module.PARAMS = p_b
            return agent_module.agent(obs)
            
        env.run([agent_a, agent_b])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r0 > r1: a_wins += 1

    # Half games as player 1
    for seed in range(GAMES_PER_EVAL // 2):
        env = make("orbit_wars", debug=False)
        
        def agent_a(obs, cfg):
            agent_module.PARAMS = p_a
            return agent_module.agent(obs)
            
        def agent_b(obs, cfg):
            agent_module.PARAMS = p_b
            return agent_module.agent(obs)
            
        env.run([agent_b, agent_a])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r1 > r0: a_wins += 1
        
    return a_wins

# ── Main Loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    best = load_best()
    print(f"Starting evolution for 10 generations...")
    print(f"Initial params: {best}")
    
    for generation in range(1, 11):
        candidate = mutate(best)
        print(f"\nGen {generation}/10: Testing candidate mutation...")
        
        # Test candidate vs current best
        wins = evaluate(candidate, best)
        win_rate = wins / GAMES_PER_EVAL
        
        if win_rate > 0.5:
            print(f"  SUCCESS! Candidate won {wins}/{GAMES_PER_EVAL} ({win_rate:.1%}). New best saved to {BEST_PARAMS_FILE}.")
            best = candidate
            save_best(best)
        else:
            print(f"  FAIL. Candidate won {wins}/{GAMES_PER_EVAL} ({win_rate:.1%}). Keeping previous best.")
            
    print("\n" + "="*40)
    print("EVOLUTION COMPLETE")
    print("="*40)
    print(f"Final Optimized Params: {best}")
    print("\nYour 'main.py' is now using these parameters via 'best_params.json'.")
    print("To submit to Kaggle, ensure you include both files or run the submission command.")
    print("="*40)
