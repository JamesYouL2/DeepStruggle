"""Live scoring battlegrounds: does the model place for scoring, and is it right (`ai/eval/scoring_bg.py`).

    # one part: 60 greedy self-play games, every spot's branches 32 paired playouts
    PYTHONPATH=.:build/release .venv/bin/python tools/scoring_bg.py run \\
        --model data/checkpoints/<model>.onnx --games 60 --pairs 32 --seed 1 --out parts/part-1.json.gz
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

On CI: `.github/workflows/scoring_bg.yml`.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.branch_oracle import onnx_policy  # noqa: E402
from ai.eval.scoring_bg import collect, play, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--games", type=int, default=60)
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--max-spots", type=int, default=400, help="spots played out per part")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        act, _ = onnx_policy(a.model)
        sens, spots = collect(act, a.games, a.seed, max_spots=a.max_spots)
        print(f"{len(sens)} plays, {len(spots)} spots ({round(time.time() - t0)}s)", flush=True)
        rows = []
        for lo in range(0, len(spots), 50):
            rows += play(act, spots[lo:lo + 50], a.pairs, a.seed * 7_919 + lo)
            print(f"  {min(lo + 50, len(spots))}/{len(spots)} ({round(time.time() - t0)}s)", flush=True)
        meta = {"model": os.path.basename(a.model), "pairs": a.pairs, "seed": a.seed,
                "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "sens": sens, "rows": rows}, f)
        print(f"{len(rows)} spots played, {meta['seconds']}s")
    else:
        sens, rows, meta = [], [], None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            key = {k: part["meta"][k] for k in ("model", "pairs")}
            if meta not in (None, key):
                raise SystemExit(f"{fn} was run as {key}, not {meta}")
            meta = key
            sens += part["sens"]
            rows += part["rows"]
        md, summary = report(sens, rows, meta or {})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump({**summary, "rows": rows}, f)
        print(md)


if __name__ == "__main__":
    main()
