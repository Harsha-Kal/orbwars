"""
benchmark.py — run the current agent vs. the previous 12 git versions.

Usage:
    python benchmark.py [--games N]   # default: 3 games per matchup

For each historical version:
  - extracts main.py at that commit into a temp file
  - wraps the agent() function so it can run inside kaggle_environments
  - plays N games (current=player0, old=player1) + N games reversed
  - prints a win/draw/loss summary table

Results are written to benchmark_results.json for further analysis.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from kaggle_environments import make

REPO_DIR = Path(__file__).parent
CURRENT_MAIN = REPO_DIR / "main.py"


# ── helpers ───────────────────────────────────────────────────────────────────

def git_log(n=13):
    """Return list of (sha, subject) for the last n commits (index 0 = HEAD)."""
    out = subprocess.check_output(
        ["git", "log", f"-{n}", "--pretty=format:%H|%s"],
        cwd=REPO_DIR, text=True,
    )
    entries = []
    for line in out.strip().splitlines():
        sha, _, subject = line.partition("|")
        entries.append((sha.strip(), subject.strip()))
    return entries


def checkout_file(sha, path="main.py"):
    """Return the content of `path` at commit `sha` as a string."""
    return subprocess.check_output(
        ["git", "show", f"{sha}:{path}"],
        cwd=REPO_DIR, text=True,
    )


def load_agent_from_source(source: str, label: str):
    """Compile `source` into a module and return its agent() callable."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir=REPO_DIR, prefix=f"_bench_{label}_"
    ) as f:
        f.write(source)
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location(f"bench_{label}", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "agent", None), tmp_path
    except Exception:
        return None, tmp_path


def safe_agent_wrapper(fn):
    """Wrap an agent so exceptions return [] instead of crashing the env."""
    def wrapped(obs, cfg):
        try:
            return fn(obs, cfg) or []
        except Exception:
            return []
    return wrapped


def run_games(agent_a, agent_b, n_games: int):
    """
    Play n_games with agent_a as player-0 and agent_b as player-1.
    Returns (wins_a, draws, wins_b).
    """
    wins_a = draws = wins_b = 0
    for _ in range(n_games):
        try:
            env = make("orbit_wars", debug=False)
            env.run([agent_a, agent_b])
            rewards = [s.reward for s in env.state]
            r0, r1 = rewards[0], rewards[1]
            if r0 > r1:
                wins_a += 1
            elif r1 > r0:
                wins_b += 1
            else:
                draws += 1
        except Exception:
            draws += 1   # count broken games as draws
    return wins_a, draws, wins_b


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark current agent vs prior 12 versions")
    parser.add_argument("--games", type=int, default=3, help="Games per side per matchup (default 3)")
    args = parser.parse_args()
    n = args.games

    print(f"\n{'='*70}")
    print(f"  Orbit Wars benchmark  |  {n} games/side per matchup")
    print(f"{'='*70}\n")

    # Load current (HEAD) agent
    commits = git_log(13)          # 0=HEAD, 1..12 = previous 12
    head_sha, head_subject = commits[0]
    print(f"Current (HEAD): {head_sha[:8]}  {head_subject}\n")

    current_src = CURRENT_MAIN.read_text()
    current_agent, current_tmp = load_agent_from_source(current_src, "current")
    tmp_files = [current_tmp]

    if current_agent is None:
        print("ERROR: could not load agent() from current main.py")
        sys.exit(1)

    current_wrapped = safe_agent_wrapper(current_agent)

    historical = commits[1:]   # previous 12

    results = []
    col_w = 52

    header = f"{'Opponent commit':<10}  {'Subject':<{col_w}}  {'W':>4} {'D':>4} {'L':>4}  {'Win%':>6}"
    print(header)
    print("-" * len(header))

    for sha, subject in historical:
        short = sha[:8]
        try:
            src = checkout_file(sha)
        except Exception as e:
            print(f"{short}  {'(could not read file)':>{col_w}}  {'ERR':>4}")
            results.append({"sha": short, "subject": subject, "error": str(e)})
            continue

        old_agent, tmp = load_agent_from_source(src, short)
        tmp_files.append(tmp)

        if old_agent is None:
            print(f"{short}  {subject[:col_w]:<{col_w}}  {'LOAD_ERR':>4}")
            results.append({"sha": short, "subject": subject, "error": "load failed"})
            continue

        old_wrapped = safe_agent_wrapper(old_agent)

        # current=p0 vs old=p1
        w0, d0, l0 = run_games(current_wrapped, old_wrapped, n)
        # old=p0 vs current=p1  (swap to reduce first-mover bias)
        l1, d1, w1 = run_games(old_wrapped, current_wrapped, n)

        total_w = w0 + w1
        total_d = d0 + d1
        total_l = l0 + l1
        total   = total_w + total_d + total_l
        win_pct = 100.0 * total_w / total if total else 0.0

        subj_col = subject[:col_w]
        print(f"{short}  {subj_col:<{col_w}}  {total_w:>4} {total_d:>4} {total_l:>4}  {win_pct:>5.1f}%")

        results.append({
            "sha": short,
            "subject": subject,
            "games": total,
            "wins": total_w,
            "draws": total_d,
            "losses": total_l,
            "win_pct": round(win_pct, 1),
        })

    print()
    # Overall summary
    valid = [r for r in results if "error" not in r]
    if valid:
        total_w = sum(r["wins"]   for r in valid)
        total_d = sum(r["draws"]  for r in valid)
        total_l = sum(r["losses"] for r in valid)
        total   = total_w + total_d + total_l
        overall = 100.0 * total_w / total if total else 0.0
        print(f"Overall vs all 12 versions:  W={total_w}  D={total_d}  L={total_l}  Win%={overall:.1f}%")

    # Persist results
    out_path = REPO_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "current_sha": head_sha[:8],
            "current_subject": head_subject,
            "games_per_side": n,
            "matchups": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}\n")

    # Cleanup temp files
    for p in tmp_files:
        try:
            os.unlink(p)
        except OSError:
            pass


if __name__ == "__main__":
    main()
