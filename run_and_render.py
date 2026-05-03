from kaggle_environments import make
import os

# Create the environment with a fixed seed for reproducibility
env = make("orbit_wars", configuration={"seed": 42}, debug=True)

print("Running game: main.py vs random...")
# Run the simulation (Player 0 is our main.py, Player 1 is random)
env.run(["main.py", "random"])

# View result
final = env.steps[-1]
print("\n=== FINAL RESULTS ===")
for i, s in enumerate(final):
    # reward 1 = Win, -1 = Loss
    outcome = "WIN" if s.reward == 1 else "LOSS" if s.reward == -1 else "TIE"
    print(f"Player {i}: reward={s.reward} ({outcome}), status={s.status}")

# Save as HTML for viewing in a browser (since 'ipython' only works in notebooks)
output_path = "game_replay.html"
with open(output_path, "w") as f:
    f.write(env.render(mode="html", width=800, height=600))

print(f"\nReplay saved to: {os.path.abspath(output_path)}")
print("You can open this file in any web browser to watch the game.")
