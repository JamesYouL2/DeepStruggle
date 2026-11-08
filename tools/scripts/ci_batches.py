#!/usr/bin/env python3
"""Run one split stage of a workflow as a series of small runs, so other jobs get their turn.

A stage split into N parts as one big matrix holds the repository's job limit (20 on a free public
repository) until it ends, and a queued run waits behind all of it. This dispatches the parts a few
at a time, as separate runs (the workflow's `part_range` input), and only while the repository's
queued and running jobs, whoever's, leave room under `--cap`. Each run pools its own parts; pool the
runs afterwards from the run ids in `--ledger`.

    tools/scripts/ci_batches.py --workflow search_bank.yml --ref exp/j1-census --parts 40 --batch 6 \\
        --ledger data/reports/x/annotate_runs.jsonl -f stage=annotate -f runners=40 -f "args=--games 1500"

Interrupted, it continues from the ledger: a batch already in it is not dispatched again.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO = "JamesYouL2/DeepStruggle"
#: A dispatched run lists no matrix jobs until its plan job has run: it counts as a whole batch.
UNPLANNED_JOBS = 6


def gh_json(args: Sequence[str]) -> Any:
    """`gh` with JSON output; retries, because the driver outlives any one network blip."""
    for attempt in range(5):
        p = subprocess.run(["gh", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return json.loads(p.stdout or "null")
        print(f"gh {' '.join(args[:3])}: {p.stderr.strip()[:200]}", file=sys.stderr, flush=True)
        time.sleep(30 * (attempt + 1))
    raise SystemExit(f"gh {' '.join(args)} kept failing")


def batches(parts: int, batch: int, start: int = 1) -> List[Tuple[int, int]]:
    """The inclusive part ranges, in order, covering `start`..`parts`."""
    return [(lo, min(lo + batch - 1, parts)) for lo in range(start, parts + 1, batch)]


def pending_jobs(job_statuses: Sequence[str]) -> int:
    """Jobs a run still holds or wants: a run with no more than its plan job listed counts as a batch."""
    if len(job_statuses) <= 1:
        return UNPLANNED_JOBS
    return sum(1 for s in job_statuses if s != "completed")


def active_jobs(repo: str) -> int:
    """The repository's queued and running jobs."""
    total = 0
    for status in ("in_progress", "queued", "waiting", "requested", "pending"):
        runs = gh_json(["api", f"repos/{repo}/actions/runs?status={status}&per_page=100"])["workflow_runs"]
        for run in runs:
            jobs = gh_json(["api", f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100"])["jobs"]
            total += pending_jobs([str(j["status"]) for j in jobs])
    return total


def workflow_runs(repo: str, workflow: str, ref: str) -> List[int]:
    rows = gh_json(["run", "list", "-R", repo, "--workflow", workflow, "--branch", ref, "--event",
                    "workflow_dispatch", "--limit", "20", "--json", "databaseId"])
    return [int(r["databaseId"]) for r in rows]


def dispatch(repo: str, workflow: str, ref: str, fields: Sequence[str], part_range: str) -> int:
    """Dispatch one batch and return its run id. Holds a lock: two drivers dispatching the same
    workflow at once could not tell their runs apart."""
    with open(os.path.join(tempfile.gettempdir(), "ci_batches.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        seen: Set[int] = set(workflow_runs(repo, workflow, ref))
        cmd = ["gh", "workflow", "run", workflow, "-R", repo, "--ref", ref, "-f", f"part_range={part_range}"]
        for f in fields:
            cmd += ["-f", f]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        for _ in range(60):
            time.sleep(5)
            fresh = [r for r in workflow_runs(repo, workflow, ref) if r not in seen]
            if fresh:
                return max(fresh)
    raise SystemExit(f"the run of parts {part_range} did not appear")


def read_ledger(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {str(r["range"]): int(r["run"]) for r in rows}


def report(repo: str, ledger: Dict[str, int]) -> int:
    """Wait for every run, then name any job that did not succeed. Exit 1 if there is one."""
    bad: List[str] = []
    for part_range, run in ledger.items():
        while True:
            jobs = gh_json(["api", f"repos/{repo}/actions/runs/{run}/jobs?per_page=100"])["jobs"]
            state = gh_json(["api", f"repos/{repo}/actions/runs/{run}"])["status"]
            if state == "completed":
                break
            time.sleep(120)
        bad += [f"{part_range} run {run}: {j['name']} {j['conclusion']}" for j in jobs
                if j["conclusion"] not in ("success", "skipped")]
    for line in bad:
        print(line, flush=True)
    print(f"{len(ledger)} runs done, {len(bad)} jobs did not succeed: {' '.join(map(str, ledger.values()))}", flush=True)
    return 1 if bad else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--parts", type=int, required=True, help="the parts the stage is split into")
    ap.add_argument("--batch", type=int, default=6, help="parts per run")
    ap.add_argument("--cap", type=int, default=14, help="dispatch only while the repository's jobs leave room under this")
    ap.add_argument("--poll", type=int, default=120, help="seconds between looks at the queue")
    ap.add_argument("--ledger", required=True, help="one line per dispatched run; continues from it")
    ap.add_argument("-f", dest="fields", action="append", default=[], help="a workflow input, name=value")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.ledger)), exist_ok=True)
    done = read_ledger(a.ledger)
    for lo, hi in batches(a.parts, a.batch):
        part_range = f"{lo}-{hi}"
        if part_range in done:
            continue
        while active_jobs(a.repo) + (hi - lo + 1) > a.cap:
            time.sleep(a.poll)
        run = dispatch(a.repo, a.workflow, a.ref, a.fields, part_range)
        done[part_range] = run
        with open(a.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"range": part_range, "run": run, "t": time.time()}) + "\n")
        print(f"parts {part_range}: run {run}", flush=True)
        time.sleep(60)  # let its plan job list the matrix before counting the queue again
    return report(a.repo, done)


if __name__ == "__main__":
    sys.exit(main())
