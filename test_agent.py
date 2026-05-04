"""
Local test runner for the Orbit Wars agent.
Run with: python test_agent.py
"""

from kaggle_environments import make
import main as my_agent  # Updated to use main.py


def run_game(opponent="random", steps=500, debug=True):
    env = make("orbit_wars", debug=debug)
    print(f"Running game: my_agent vs {opponent}")
    
    # Run the simulation
    result = env.run([my_agent.agent, opponent])

    # Print final state
    last_step = result[-1]
    print("\n=== Game Over ===")
    for i, state in enumerate(last_step):
        reward = state.get("reward", "?")
        status = state.get("status", "?")
        # In Orbit Wars, reward 1 = Win, -1 = Loss
        outcome = "WIN" if reward == 1 else "LOSS" if reward == -1 else "TIE"
        print(f"  Player {i} ({'YOU' if i==0 else opponent}): reward={reward} -> {outcome} (Status: {status})")

    return result


if __name__ == "__main__":
    # Test against the random baseline
    run_game(opponent="random", steps=500)
