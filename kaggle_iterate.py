"""
kaggle_iterate.py
─────────────────
Every run:
  1. Fetch recent episodes for the latest Kaggle submission
  2. Download replays of recent losses and analyze strategic metrics
  3. Diagnose the primary failure pattern
  4. Apply targeted parameter tweaks to main.py
  5. Submit the updated agent to orbit-wars
  6. Commit to git (master + srujan_test)

Run manually:   python3 kaggle_iterate.py
Scheduled by:   cron / CronCreate every 5 h
"""

import json, math, os, re, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

import kaggle
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetEpisodeReplayRequest,
    ApiListSubmissionEpisodesRequest,
)

REPO      = Path(__file__).parent
MAIN_PY   = REPO / "main.py"
LOG_FILE  = REPO / "iterate_log.jsonl"
COMP      = "orbit-wars"
MY_SUB_ID = None   # resolved dynamically from latest submission

# ── Kaggle helpers ────────────────────────────────────────────────────────────

def latest_submission():
    """Return (submission_id, version_desc, score) for the newest complete sub."""
    subs = kaggle.api.competition_submissions(COMP) or []
    for s in subs:
        if "COMPLETE" in str(getattr(s, "status", "")):
            desc  = getattr(s, "description", "") or ""
            score = float(getattr(s, "public_score", 0) or 0)
            ref   = int(getattr(s, "ref", 0))
            return ref, desc, score
    return None, "v0", 0.0


def list_episodes(submission_id, limit=40):
    return kaggle.api.competition_list_episodes(submission_id)[:limit]


def fetch_replay(ep_id):
    with kaggle.api.build_kaggle_client() as kc:
        req = ApiGetEpisodeReplayRequest()
        req.episode_id = ep_id
        r = kc.competitions.competition_api_client.get_episode_replay(req)
        return r.json()


def analyze_replay(replay, my_index):
    """
    Return per-checkpoint metrics: {t10, t25, t50, t75, t100}.
    Each entry: my_planets, opp_planets, my_prod, opp_prod, ship_ratio
    """
    steps = replay.get("steps", [])
    if not steps:
        return {}
    result = {}
    for pct, key in [(0.10,"t10"),(0.25,"t25"),(0.50,"t50"),(0.75,"t75"),(1.0,"t100")]:
        idx  = min(int(len(steps) * pct), len(steps) - 1)
        step = steps[idx]
        if not step or not isinstance(step[0], dict):
            continue
        obs     = step[0].get("observation", {})
        planets = obs.get("planets", [])
        myp  = [p for p in planets if p[1] == my_index]
        oppp = [p for p in planets if p[1] not in (-1, my_index)]
        result[key] = {
            "my_planets":  len(myp),
            "opp_planets": len(oppp),
            "my_prod":     sum(p[6] for p in myp),
            "opp_prod":    sum(p[6] for p in oppp),
            "my_ships":    sum(p[5] for p in myp),
            "opp_ships":   sum(p[5] for p in oppp),
            "ship_ratio":  sum(p[5] for p in myp) / max(1, sum(p[5] for p in oppp)),
        }
    return result


def diagnose(loss_metrics_list):
    """
    Classify the primary failure mode from a list of per-episode metrics.
    Returns a dict of {flag: fraction_of_losses}.
    """
    n = max(1, len(loss_metrics_list))
    return {
        "slow_start":       sum(1 for m in loss_metrics_list
                                if m.get("t25",{}).get("ship_ratio",1) < 0.65) / n,
        "prod_deficit_mid": sum(1 for m in loss_metrics_list
                                if m.get("t50",{}).get("my_prod",0)
                                 < m.get("t50",{}).get("opp_prod",1)) / n,
        "collapse_late":    sum(1 for m in loss_metrics_list
                                if m.get("t50",{}).get("ship_ratio",1) >= 0.7
                                and m.get("t75",{}).get("ship_ratio",1) < 0.4) / n,
        "over_attacked":    sum(1 for m in loss_metrics_list
                                if m.get("t25",{}).get("ship_ratio",1) >= 0.8
                                and m.get("t50",{}).get("ship_ratio",1) < 0.4) / n,
    }


# ── Parametric patch system ───────────────────────────────────────────────────
# Each patch adjusts a named constant in main.py by a delta.
# We keep a log so we don't drift endlessly in one direction.

PATCH_REGISTRY = {
    # (constant_name, delta, rationale)
    "slow_start": [
        ("EARLY_RESERVE_MULT", -0.05,
         "cut early reserves further when we're losing the opening land-grab"),
        ("EARLY_PROD_WEIGHT",  +0.5,
         "value high-production planets even more during early expansion"),
    ],
    "prod_deficit_mid": [
        ("PRESSURE_MIN_PROD",   -1,
         "attack even low-production enemy planets to deny their growth"),
        ("PRESSURE_MAX_MISSIONS", +1,
         "run more pressure missions when we're losing the production race"),
    ],
    "collapse_late": [
        ("PRESSURE_FRONTS",     +1,
         "open more encirclement fronts when we collapse in late game"),
        ("DEFENSE_SAVE_HELPERS", +1,
         "pull more reinforcements when we collapse late"),
    ],
    "over_attacked": [
        ("DEFENSE_ETA_HORIZON", +5,
         "look further ahead for threats when enemy attacks are overwhelming us"),
        ("DEFENSE_SAVE_HELPERS", +1,
         "pull more helpers when overrun by early attacks"),
    ],
}

# Hard bounds to prevent runaway values
PATCH_BOUNDS = {
    "EXPAND_MAX_MISSIONS":  (1, 6),
    "EXPAND_MAX_SOURCES":   (3, 8),
    "PRESSURE_MIN_PROD":    (0, 3),
    "PRESSURE_MAX_MISSIONS":(1, 4),
    "PRESSURE_FRONTS":      (1, 4),
    "DEFENSE_ETA_HORIZON":  (10, 40),
    "DEFENSE_SAVE_HELPERS": (2, 8),
    "EXPAND_RACE_MARGIN":   (1.0, 1.5),
    "EARLY_RESERVE_MULT":   (0.35, 0.70),
    "EARLY_PROD_WEIGHT":    (1.0, 5.0),
}


def _current_value(source, const_name):
    """Read the current integer/float value of a constant from source text."""
    m = re.search(rf"^{const_name}\s*=\s*([0-9.]+)", source, re.MULTILINE)
    if m:
        val = m.group(1)
        return float(val) if "." in val else int(val)
    return None


def apply_patch(source, const_name, delta):
    """Return (new_source, new_value) after clamping to PATCH_BOUNDS."""
    cur = _current_value(source, const_name)
    if cur is None:
        print(f"  [skip] {const_name} not found in source")
        return source, None
    new_val = cur + delta
    lo, hi  = PATCH_BOUNDS.get(const_name, (-999, 999))
    new_val = max(lo, min(hi, new_val))
    if new_val == cur:
        print(f"  [clip] {const_name} already at bound {cur}")
        return source, cur
    if isinstance(cur, int):
        new_str = str(int(new_val))
    else:
        new_str = f"{new_val:.2f}"
    new_source = re.sub(
        rf"^({const_name}\s*=\s*)[0-9.]+",
        rf"\g<1>{new_str}",
        source,
        flags=re.MULTILINE,
    )
    print(f"  [patch] {const_name}: {cur} → {new_val}")
    return new_source, new_val


def next_version_tag():
    """Read current version tag from git log and increment."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"], cwd=REPO, text=True
        ).strip()
        m = re.search(r"v(\d+)", out)
        return int(m.group(1)) + 1 if m else 43
    except Exception:
        return 43


def git_commit_push(version_tag, message):
    """Stage main.py, commit, push to master and srujan_test."""
    try:
        subprocess.run(["git", "add", "main.py"], cwd=REPO, check=True)
        # Check if there's actually anything to commit
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPO
        )
        if diff.returncode == 0:
            print(f"  [git] no changes to main.py — skipping commit")
            return
        subprocess.run(
            ["git", "commit", "-m", f"v{version_tag}: {message}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"],
            cwd=REPO, check=True
        )
        # Push master
        subprocess.run(["git", "push", "origin", "master"], cwd=REPO, check=True)
        # Fast-forward srujan_test
        subprocess.run(["git", "checkout", "srujan_test"], cwd=REPO, check=True)
        subprocess.run(["git", "merge", "master", "--ff-only"], cwd=REPO, check=True)
        subprocess.run(["git", "push", "origin", "srujan_test"], cwd=REPO, check=True)
        subprocess.run(["git", "checkout", "master"], cwd=REPO, check=True)
        print(f"  [git] committed and pushed v{version_tag}")
    except subprocess.CalledProcessError as e:
        print(f"  [git] error: {e}")


def submit(version_tag, message):
    """Submit main.py to Kaggle orbit-wars."""
    full_msg = f"v{version_tag}: {message}"
    # Prefer the python3.12 kaggle CLI which has working SSL
    kaggle_cli = os.path.expanduser("~/Library/Python/3.12/bin/kaggle")
    if not os.path.exists(kaggle_cli):
        kaggle_cli = "kaggle"
    try:
        result = subprocess.run(
            [kaggle_cli, "competitions", "submit", COMP, "-f", str(MAIN_PY), "-m", full_msg],
            cwd=REPO, capture_output=True, text=True, check=True
        )
        print(f"  [kaggle] submitted: {full_msg}")
        print(f"  {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [kaggle] submit error: {e.stderr}")
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_iteration():
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"  Kaggle iterate  {ts}")
    print(f"{'='*60}")

    # 1. Find latest submission
    sub_id, desc, score = latest_submission()
    if not sub_id:
        print("No complete submissions found. Submitting current main.py as baseline.")
        v = next_version_tag()
        submit(v, "initial auto-iterate submission")
        return

    print(f"Latest sub: {sub_id} ({desc})  score={score:.1f}")

    # 2. Fetch episodes
    eps = list_episodes(sub_id)
    wins   = sum(1 for ep in eps for ag in ep.agents if ag.submission_id == sub_id and ag.reward ==  1)
    losses = sum(1 for ep in eps for ag in ep.agents if ag.submission_id == sub_id and ag.reward == -1)
    draws  = sum(1 for ep in eps for ag in ep.agents if ag.submission_id == sub_id and ag.reward ==  0)
    total  = wins + losses + draws
    win_pct = 100 * wins / max(1, total)
    print(f"Episodes: {total}  W={wins} D={draws} L={losses}  WR={win_pct:.1f}%")

    # 3. Analyze recent losses (up to 6)
    loss_eps = [
        (ep, next(ag for ag in ep.agents if ag.submission_id == sub_id))
        for ep in eps
        for ag in ep.agents
        if ag.submission_id == sub_id and ag.reward == -1
    ][:6]

    loss_metrics = []
    for ep, me in loss_eps:
        try:
            replay  = fetch_replay(ep.id)
            metrics = analyze_replay(replay, me.index)
            loss_metrics.append(metrics)
        except Exception as e:
            print(f"  [warn] replay {ep.id}: {e}")

    if not loss_metrics:
        print("No replay data — submitting unchanged as new version.")
        v = next_version_tag()
        msg = f"auto-iterate no-data resubmit  wr={win_pct:.0f}pct"
        git_commit_push(v, msg)
        submit(v, msg)
        return

    diag = diagnose(loss_metrics)
    print("\nDiagnosis:")
    for flag, fraction in sorted(diag.items(), key=lambda x: -x[1]):
        print(f"  {flag}: {fraction*100:.0f}% of losses")

    # 4. Trend guard: if win rate is declining, skip parameter patches.
    #    Applying more of the same patches when things are getting worse only
    #    accelerates the decline (as seen with EARLY_RESERVE_MULT runaway cut).
    prev_win_pct = None
    if LOG_FILE.exists():
        try:
            lines = LOG_FILE.read_text().strip().splitlines()
            if lines:
                prev_record = json.loads(lines[-1])
                prev_win_pct = prev_record.get("win_pct")
        except Exception:
            pass

    declining = prev_win_pct is not None and win_pct < prev_win_pct - 3.0
    if declining:
        print(f"\n[trend-guard] Win rate fell {prev_win_pct:.1f}% → {win_pct:.1f}%. "
              f"Skipping parameter patches to avoid runaway degradation.")

    # 5. Apply patches for the two most severe failure modes (only if not declining)
    source = MAIN_PY.read_text()
    patches_applied = []
    top_flags = sorted(diag.items(), key=lambda x: -x[1])[:2]

    if not declining:
        for flag, fraction in top_flags:
            if fraction < 0.40:   # only fix if ≥40% of losses show this pattern
                continue
            for const, delta, rationale in PATCH_REGISTRY.get(flag, []):
                source, new_val = apply_patch(source, const, delta)
                if new_val is not None:
                    patches_applied.append(f"{const}→{new_val}")

    if not patches_applied:
        print("No patches to apply (all at bounds, below threshold, or trend-guarded).")
        patches_applied = ["no-param-change"]

    MAIN_PY.write_text(source)

    # 6. Commit + submit
    v   = next_version_tag()
    top_flag = top_flags[0][0] if top_flags else "unknown"
    msg = f"auto: fix {top_flag}  {','.join(patches_applied)}  wr={win_pct:.0f}pct"
    print(f"\nVersion: v{v}")
    print(f"Message: {msg}")
    git_commit_push(v, msg)
    submit(v, msg)

    # 7. Log to JSONL
    record = {
        "ts": ts,
        "version": v,
        "sub_id": sub_id,
        "score": score,
        "win_pct": win_pct,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "diagnosis": diag,
        "patches": patches_applied,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print("\nDone.")


if __name__ == "__main__":
    run_iteration()
