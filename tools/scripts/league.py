#!/usr/bin/env python3
"""P24: a two-learner league -- a main agent and a main exploiter -- driven over tools/train.py.

    PYTHONPATH=.:build/release python tools/scripts/league.py \\
        --main-name E5-02-43 --main-resume <resume_state.pt> --main-steps 610000000 --seed 43 \\
        --exploiter-name E5-03-43 --exploiter-reset <resume_state.pt> --exploiter-gen-steps 50000000 \\
        --league-dir /workspace/data/league/E5-02-43 --log-dir /workspace/data/logs/league \\
        --train-args "<architecture and recipe flags shared by both>" \\
        --main-description "..." --exploiter-description "..."

Both learners are ordinary `tools/train.py` runs (invariant 9); the league is a set of directories.

* **Main agent:** its own spread pool as usual (`--opponent-self-pool`), plus `--league-dirs
  <league-dir>`, which holds the exploiter snapshots this driver publishes, drawn with
  `--league-frac` of the pool's draws.
* **Main exploiter:** `--opponent-frac 1.0` against the main agent's newest snapshots only
  (`--league-dirs <main run dir> --league-pool-size 2`), resumed from `--exploiter-reset` -- an
  early state of the main lineage, the closest thing to AlphaStar's supervised reset point.
* **Publish:** when the exploiter writes a snapshot while winning at least `--publish-win`
  against the main agent on either seat, it is symlinked into the league directory. Per seat,
  because a weakness of the main agent is usually a weakness of one seat.
* **Reset:** once it wins `--reset-win` on either seat, its next snapshot is published and it is
  restarted from the reset state, as a new generation -- or when a generation's
  `--exploiter-gen-steps` run out, whether or not it succeeded.

The exploiter's win rate is read from its own `training_metrics.jsonl` (`opp_league_win_us` /
`_ussr`, with their game counts), so judging it needs no third GPU process. Events go to
`<league-dir>/events.jsonl`. The driver ends when the main agent's run does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINTS = "/workspace/data/checkpoints"
SNAP_RE = re.compile(r"^snapshot_(\d+)steps\.pt$")


def _event(league_dir: str, kind: str, **kw: Any) -> None:
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": kind, **kw}
    with open(os.path.join(league_dir, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[league] {kind} {kw}", flush=True)


def _launch(args: List[str], log_path: str) -> subprocess.Popen:
    log = open(log_path, "a", encoding="utf-8")
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    return subprocess.Popen([sys.executable, "tools/train.py", *args], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, env=env)


def _run_dir(name: str, started: float, timeout: float = 600.0) -> str:
    """The directory `tools/train.py --run-name <name>` created after `started`."""
    deadline = time.time() + timeout
    pat = re.compile(rf"^{re.escape(name)}_\d{{8}}_\d{{6}}$")
    while time.time() < deadline:
        cands = [os.path.join(CHECKPOINTS, d) for d in os.listdir(CHECKPOINTS) if pat.match(d)]
        cands = [d for d in cands if os.path.getmtime(d) >= started - 5]
        if cands:
            return max(cands, key=os.path.getmtime)
        time.sleep(5)
    raise RuntimeError(f"no run directory for {name} appeared within {timeout:.0f}s")


def _snapshots(run_dir: str) -> List[int]:
    return sorted(int(m.group(1)) for f in os.listdir(run_dir) if (m := SNAP_RE.match(f)))


def _last_metrics(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "training_metrics.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 65536))
        lines = f.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "opp_league_win_us" in rec:
            return rec
    return None


def _stop(proc: subprocess.Popen, grace: float = 120.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-name", required=True)
    ap.add_argument("--main-resume", required=True)
    ap.add_argument("--main-steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exploiter-name", required=True,
                    help="Generation g runs as <name>-<g>.")
    ap.add_argument("--exploiter-reset", required=True,
                    help="Resume state every exploiter generation starts from.")
    ap.add_argument("--exploiter-gen-steps", type=int, default=50_000_000)
    ap.add_argument("--league-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--train-args", required=True,
                    help="Architecture and recipe flags shared by both learners, one string.")
    ap.add_argument("--main-league-frac", type=float, default=0.5)
    ap.add_argument("--main-league-size", type=int, default=4)
    ap.add_argument("--exploiter-league-size", type=int, default=2)
    ap.add_argument("--publish-win", type=float, default=0.55)
    ap.add_argument("--reset-win", type=float, default=0.70)
    ap.add_argument("--min-games", type=float, default=300.0,
                    help="Games per seat against the current main snapshots before a rate counts.")
    ap.add_argument("--main-description", required=True)
    ap.add_argument("--exploiter-description", required=True)
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)

    os.makedirs(a.league_dir, exist_ok=True)
    os.makedirs(a.log_dir, exist_ok=True)
    common = shlex.split(a.train_args)

    t0 = time.time()
    main_proc = _launch(common + [
        "--resume", a.main_resume, "--seed", str(a.seed), "--run-name", a.main_name,
        "--train-steps", str(a.main_steps),
        "--opponent-frac", "0.3", "--opponent-self-pool", "--opponent-pool-size", "12",
        "--league-dirs", a.league_dir, "--league-pool-size", str(a.main_league_size),
        "--league-frac", str(a.main_league_frac),
        "--description", a.main_description,
    ], os.path.join(a.log_dir, f"{a.main_name}.log"))
    main_dir = _run_dir(a.main_name, t0)
    _event(a.league_dir, "main_launched", run=a.main_name, dir=main_dir, pid=main_proc.pid)
    # The exploiter plays the main agent's newest snapshots, so it waits for the first one.
    while not _snapshots(main_dir):
        if main_proc.poll() is not None:
            _event(a.league_dir, "main_exited_early", code=main_proc.returncode)
            return int(main_proc.returncode or 1)
        time.sleep(a.poll)

    m_reset = re.search(r"(\d+)steps", os.path.basename(a.exploiter_reset))
    if m_reset is None:
        raise SystemExit(f"--exploiter-reset must name a resume_<n>steps.pt state, got {a.exploiter_reset}")
    reset_steps = int(m_reset.group(1))
    gen = 0
    expl: Optional[subprocess.Popen] = None
    expl_dir = ""
    published: set = set()
    reset_pending = False

    def start_generation() -> None:
        nonlocal gen, expl, expl_dir, reset_pending
        gen += 1
        name = f"{a.exploiter_name}-{gen}"
        t = time.time()
        expl = _launch(common + [
            "--resume", a.exploiter_reset, "--seed", str(a.seed), "--run-name", name,
            "--train-steps", str(reset_steps + a.exploiter_gen_steps),
            "--opponent-frac", "1.0", "--league-dirs", main_dir,
            "--league-pool-size", str(a.exploiter_league_size),
            "--description", f"{a.exploiter_description} Generation {gen}.",
        ], os.path.join(a.log_dir, f"{name}.log"))
        expl_dir = _run_dir(name, t)
        reset_pending = False
        _event(a.league_dir, "exploiter_launched", run=name, generation=gen, dir=expl_dir,
               pid=expl.pid)

    start_generation()
    seen_snaps: set = set()
    while True:
        if main_proc.poll() is not None:
            _event(a.league_dir, "main_exited", code=main_proc.returncode)
            if expl is not None:
                _stop(expl)
                _event(a.league_dir, "exploiter_stopped", generation=gen, reason="main finished")
            return int(main_proc.returncode or 0)

        rec = _last_metrics(expl_dir)
        rates = {}
        if rec is not None:
            for seat in ("us", "ussr"):
                if rec.get(f"opp_league_games_{seat}", 0.0) >= a.min_games:
                    rates[seat] = float(rec[f"opp_league_win_{seat}"])
        best = max(rates.values()) if rates else 0.0
        if best >= a.reset_win and not reset_pending:
            reset_pending = True
            _event(a.league_dir, "exploiter_succeeded", generation=gen, rates=rates,
                   steps=rec.get("total_steps") if rec else None)

        for n in _snapshots(expl_dir):
            if n in seen_snaps or n <= reset_steps:
                continue
            seen_snaps.add(n)
            src = os.path.join(expl_dir, f"snapshot_{n}steps.pt")
            if best >= a.publish_win and src not in published:
                link = os.path.join(a.league_dir,
                                    f"{os.path.basename(expl_dir).rsplit('_', 2)[0]}_snapshot_{n}steps.pt")
                if not os.path.exists(link):
                    os.symlink(src, link)
                published.add(src)
                _event(a.league_dir, "published", generation=gen, snapshot=src, rates=rates)
            if reset_pending and expl is not None:
                _stop(expl)
                _event(a.league_dir, "exploiter_reset", generation=gen, reason="succeeded",
                       rates=rates)
                seen_snaps = set()
                start_generation()
                break
        else:
            if expl is not None and expl.poll() is not None:
                _event(a.league_dir, "exploiter_reset", generation=gen,
                       reason=f"budget spent (exit {expl.returncode})", rates=rates)
                seen_snaps = set()
                start_generation()
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
