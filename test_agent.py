"""
Local test runner for the Orbit Wars agent.
Run with: python test_agent.py
"""

from kaggle_environments import make
import agent as my_agent


def run_game(opponent="random", steps=200, debug=True):
    env = make("orbit_wars", debug=debug)
    print(f"Running game: my_agent vs {opponent}")
    result = env.run([my_agent.agent, opponent])

    # Print final state
    last_step = result[-1]
    print("\n=== Game Over ===")
    for i, state in enumerate(last_step):
        reward = state.get("reward", "?")
        status = state.get("status", "?")
        print(f"  Player {i}: reward={reward}, status={status}")

    return result


if __name__ == "__main__":
    result = run_game(opponent="random", steps=200)
    print("\nDone! To render in a notebook use:")
    print("  env.render(mode='ipython')")
