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
* **Main agent from scratch:** omit `--main-resume`.
* **Main exploiter:** `--opponent-frac 1.0`, resumed from `--exploiter-reset`, plus
  `--exploiter-args`. With `--exploiter-target frozen` (the default) each generation trains against
  the main agent's newest snapshot *at its launch*, and is judged against that same snapshot: an
  exploit takes 30-40M steps to find, and a target that moves every 10M never lets it settle.
  `moving` trains against the main agent's two newest snapshots as they appear. An exploiter carrying the self-play recipe collapses or stalls; the one that
  exploits plays a greedy opponent with no KL anchor, no entropy bonus and learner-only
  normalisation (`research/log/P24_stage1_exploiter_collapse.md`).
* **Judged greedily:** each new exploiter snapshot plays the main agent's newest snapshot,
  greedy on both sides, `--judge-games` a side, on the CPU -- about 8 s for 200, and no third GPU
  process. A greedy tournament is what rates the main agent, so it is what an exploit must beat;
  the exploiter's own training win rate, sampled against a greedy opponent, sits far below it.
* **Publish:** a snapshot scoring at least `--publish-win` is symlinked into the league directory.
* **Reset:** one scoring at least `--reset-win` is published and the exploiter restarts from the
  reset state as a new generation -- as it also does when `--exploiter-gen-steps` run out.
  `--exploiter-reset main-latest` makes the reset state the main agent's newest tagged resume
  state at each generation's launch, so every generation starts level with its target; the first
  waits for one at least `--exploiter-start-steps` in.

Events, every judgement included, go to `<league-dir>/events.jsonl`. The driver ends when the main
agent's run does.
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai.training.run_name import is_run_name  # noqa: E402  (ROOT must be importable first)

CHECKPOINTS = "/workspace/data/checkpoints"
SNAP_RE = re.compile(r"^snapshot_(\d+)steps\.pt$")
RESUME_RE = re.compile(r"^resume_(\d+)steps\.pt$")


def latest_resume(run_dir: str, min_steps: int = 0) -> Optional[str]:
    """The newest tagged resume state (`resume_<n>steps.pt`) with n >= min_steps, or None.

    Tagged states are written whole and never rewritten, unlike `resume_state.pt`, so one can be
    read while the run that wrote it goes on."""
    found = [(int(m.group(1)), f) for f in os.listdir(run_dir) if (m := RESUME_RE.match(f))]
    found = [(n, f) for n, f in found if n >= min_steps]
    return os.path.join(run_dir, max(found)[1]) if found else None


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
    """The directory `tools/train.py --run-name <name>` created after `started`.

    The timeout counts this loop's own sleeps, not the wall clock: a host suspend stops sleep
    but not the clock, and must not read as a launch that never produced a directory."""
    pat = re.compile(rf"^{re.escape(name)}_\d{{8}}_\d{{6}}$")
    for _ in range(int(timeout / 5)):
        cands = [os.path.join(CHECKPOINTS, d) for d in os.listdir(CHECKPOINTS) if pat.match(d)]
        cands = [d for d in cands if os.path.getmtime(d) >= started - 5]
        if cands:
            return max(cands, key=os.path.getmtime)
        time.sleep(5)
    raise RuntimeError(f"no run directory for {name} appeared within {timeout:.0f}s")


def _snapshots(run_dir: str) -> List[int]:
    return sorted(int(m.group(1)) for f in os.listdir(run_dir) if (m := SNAP_RE.match(f)))


_OVERALL_RE = re.compile(r"Overall Result: .*? (\d+)W - (\d+)L - (\d+)D")
_RECORD_RE = re.compile(r"Record: (\d+)W - (\d+)L - (\d+)D")


def _judge(exploiter: str, main_snap: str, games: int) -> Optional[Dict[str, float]]:
    """Greedy head to head on the CPU: the exploiter's score overall and per seat, or None."""
    out = subprocess.run(
        [sys.executable, "tools/tournament.py", "--models", exploiter, main_snap,
         "--games-per-side", str(games), "--temperature", "0.0", "--device", "cpu"],
        cwd=ROOT, capture_output=True, text=True, env=dict(os.environ))
    text = out.stdout + out.stderr
    m = _OVERALL_RE.search(text)
    recs = _RECORD_RE.findall(text)
    if m is None or len(recs) < 2:
        return None
    score = lambda w, l, d: (int(w) + 0.5 * int(d)) / max(1, int(w) + int(l) + int(d))
    return {"overall": score(*m.groups()), "as_us": score(*recs[0]), "as_ussr": score(*recs[1])}


class _Adopted:
    """A main agent this driver did not launch: a running process, known by its PID."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        if self.returncode is None:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                self.returncode = 0          # its exit status is not ours to read
        return self.returncode


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
    ap.add_argument("--main-resume", default=None,
                    help="Resume state for the main agent; omitted, it starts from scratch.")
    ap.add_argument("--adopt-main-dir", default=None,
                    help="Take over a main agent already running in this run directory "
                         "(with --adopt-main-pid) instead of launching one.")
    ap.add_argument("--adopt-main-pid", type=int, default=None)
    ap.add_argument("--exploiter-target", choices=["frozen", "moving"], default="frozen")
    ap.add_argument("--main-steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exploiter-name", required=True,
                    help="Generation g runs as <name>-<g>.")
    ap.add_argument("--exploiter-start-steps", type=int, default=0,
                    help="With --exploiter-reset main-latest: the first generation waits until the main "
                         "agent has a tagged resume state at least this far on.")
    ap.add_argument("--exploiter-reset", required=True,
                    help="Resume state every exploiter generation starts from.")
    ap.add_argument("--exploiter-gen-steps", type=int, default=50_000_000)
    ap.add_argument("--first-generation", type=int, default=1,
                    help="Number of the first exploiter generation this driver launches. A league "
                         "continued by a new driver over the same --league-dir passes the next "
                         "number, so generation names (<exploiter-name>-<n>) and the events log "
                         "keep counting instead of starting again at 1.")
    ap.add_argument("--league-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--train-args", required=True,
                    help="Architecture and recipe flags shared by both learners, one string.")
    ap.add_argument("--main-league-frac", type=float, default=0.5)
    ap.add_argument("--main-league-size", type=int, default=4)
    ap.add_argument("--exploiter-league-size", type=int, default=2)
    ap.add_argument("--main-args", default="",
                    help="Flags for the main agent only, after --train-args (e.g. its setup credit), so the "
                         "exploiter keeps the recipe that exploits.")
    ap.add_argument("--exploiter-args", default="",
                    help="Flags for the exploiter only, one string (e.g. --opponent-temperature 0 "
                         "--eta 0 --entropy-coef 0 --adv-norm-learner-only).")
    ap.add_argument("--publish-win", type=float, default=0.55)
    ap.add_argument("--reset-win", type=float, default=0.65)
    ap.add_argument("--judge-games", type=int, default=200,
                    help="Games a side in each greedy judgement.")
    ap.add_argument("--main-description", required=True)
    ap.add_argument("--exploiter-description", required=True)
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)

    # Both names are checked up front, against the rule tools/train.py enforces, generation suffix
    # included: a non-conforming exploiter name used to fail only when its first generation
    # launched, and the driver then died while the main agent trained on alone.
    if a.first_generation < 1:
        raise SystemExit("--first-generation must be at least 1")
    for n in (a.main_name, f"{a.exploiter_name}-{a.first_generation}"):
        if not is_run_name(n):
            raise SystemExit(f"run name {n!r} is not <engine>-<attempt>-<seed>; tools/train.py would refuse it")
    os.makedirs(a.league_dir, exist_ok=True)
    os.makedirs(a.log_dir, exist_ok=True)
    common = shlex.split(a.train_args)

    t0 = time.time()
    main_proc: Any
    if a.adopt_main_dir:
        if a.adopt_main_pid is None:
            raise SystemExit("--adopt-main-dir needs --adopt-main-pid")
        main_proc = _Adopted(a.adopt_main_pid)
        main_dir = a.adopt_main_dir
        _event(a.league_dir, "main_adopted", run=a.main_name, dir=main_dir, pid=a.adopt_main_pid)
    else:
        # No --main-resume: the main agent starts from scratch.
        resume_args = ["--resume", a.main_resume] if a.main_resume else []
        main_proc = _launch(common + resume_args + [
            "--seed", str(a.seed), "--run-name", a.main_name,
            "--train-steps", str(a.main_steps),
            "--opponent-frac", "0.3", "--opponent-self-pool", "--opponent-pool-size", "12",
            "--league-dirs", a.league_dir, "--league-pool-size", str(a.main_league_size),
            "--league-frac", str(a.main_league_frac),
            "--description", a.main_description,
        ] + shlex.split(a.main_args), os.path.join(a.log_dir, f"{a.main_name}.log"))
        main_dir = _run_dir(a.main_name, t0)
        _event(a.league_dir, "main_launched", run=a.main_name, dir=main_dir, pid=main_proc.pid)
    # The exploiter plays the main agent's newest snapshots, so it waits for the first one.
    while not _snapshots(main_dir):
        if main_proc.poll() is not None:
            _event(a.league_dir, "main_exited_early", code=main_proc.returncode)
            return int(main_proc.returncode or 1)
        time.sleep(a.poll)

    follow_main = a.exploiter_reset == "main-latest"
    if follow_main:
        # Each generation restarts from the main agent's newest tagged resume state, so it starts
        # level with the target it is about to attack (P24 stage 1's recommendation). The first
        # waits until the main agent is past the basics.
        while latest_resume(main_dir, a.exploiter_start_steps) is None:
            if main_proc.poll() is not None:
                _event(a.league_dir, "main_exited_early", code=main_proc.returncode)
                return int(main_proc.returncode or 1)
            time.sleep(a.poll)
    elif re.search(r"(\d+)steps", os.path.basename(a.exploiter_reset)) is None:
        raise SystemExit(f"--exploiter-reset must name a resume_<n>steps.pt state or be main-latest, "
                         f"got {a.exploiter_reset}")
    reset_file = a.exploiter_reset
    reset_steps = 0
    gen = a.first_generation - 1
    expl: Optional[subprocess.Popen] = None
    expl_dir = ""
    gen_target = ""
    published: set = set()

    def start_generation() -> None:
        nonlocal gen, expl, expl_dir, gen_target, reset_file, reset_steps
        gen += 1
        if follow_main:
            newest = latest_resume(main_dir, a.exploiter_start_steps)
            assert newest is not None
            reset_file = newest
        m_reset = re.search(r"(\d+)steps", os.path.basename(reset_file))
        assert m_reset is not None
        reset_steps = int(m_reset.group(1))
        name = f"{a.exploiter_name}-{gen}"
        gen_target = os.path.join(main_dir, f"snapshot_{_snapshots(main_dir)[-1]}steps.pt")
        pool = (["--opponent-checkpoints", gen_target] if a.exploiter_target == "frozen" else
                ["--league-dirs", main_dir, "--league-pool-size", str(a.exploiter_league_size)])
        t = time.time()
        expl = _launch(common + [
            "--resume", reset_file, "--seed", str(a.seed), "--run-name", name,
            "--train-steps", str(reset_steps + a.exploiter_gen_steps),
            "--opponent-frac", "1.0", *pool,
            "--description", f"{a.exploiter_description} Generation {gen}"
                             + (f", against {os.path.basename(gen_target)}." if a.exploiter_target == "frozen" else "."),
        ] + shlex.split(a.exploiter_args), os.path.join(a.log_dir, f"{name}.log"))
        expl_dir = _run_dir(name, t)
        _event(a.league_dir, "exploiter_launched", run=name, generation=gen, dir=expl_dir,
               pid=expl.pid, reset=reset_file, target=gen_target if a.exploiter_target == "frozen" else "moving")

    start_generation()
    seen_snaps: set = set()
    while True:
        if main_proc.poll() is not None:
            _event(a.league_dir, "main_exited", code=main_proc.returncode)
            if expl is not None:
                _stop(expl)
                _event(a.league_dir, "exploiter_stopped", generation=gen, reason="main finished")
            return int(main_proc.returncode or 0)

        reset = False
        for n in _snapshots(expl_dir):
            if n in seen_snaps or n <= reset_steps:
                continue
            seen_snaps.add(n)
            src = os.path.join(expl_dir, f"snapshot_{n}steps.pt")
            newest = os.path.join(main_dir, f"snapshot_{_snapshots(main_dir)[-1]}steps.pt")
            target = gen_target if a.exploiter_target == "frozen" else newest
            time.sleep(20)                     # let the snapshot finish writing
            res = _judge(src, target, a.judge_games)
            info = _judge(src, newest, a.judge_games) if target != newest else res
            _event(a.league_dir, "judged", generation=gen, snapshot=src, against=target, result=res,
                   vs_main_newest=info, main_newest=newest)
            if res is None:
                continue
            if res["overall"] >= a.publish_win and src not in published:
                link = os.path.join(a.league_dir,
                                    f"{os.path.basename(expl_dir).rsplit('_', 2)[0]}_snapshot_{n}steps.pt")
                if not os.path.exists(link):
                    os.symlink(src, link)
                published.add(src)
                _event(a.league_dir, "published", generation=gen, snapshot=src, result=res)
            if res["overall"] >= a.reset_win:
                reset = True
                break
        if reset and expl is not None:
            _stop(expl)
            _event(a.league_dir, "exploiter_reset", generation=gen, reason="succeeded")
            seen_snaps = set()
            start_generation()
        elif expl is not None and expl.poll() is not None:
            _event(a.league_dir, "exploiter_reset", generation=gen,
                   reason=f"budget spent (exit {expl.returncode})")
            seen_snaps = set()
            start_generation()
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
