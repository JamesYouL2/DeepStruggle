"""The doctrine census: how often the model follows a strong player's card rules (`ai/eval/doctrine_census.py`).

    # one part: 500 greedy self-play games from seed 1
    PYTHONPATH=.:build/release .venv/bin/python tools/doctrine_census.py run \\
        --model data/checkpoints/E6-06-44@soup_680-760.onnx --games 500 --seed 1 --out parts/part-1.json.gz
    # pool the parts
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

On CI: `.github/workflows/doctrine_census.yml`.
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
from ai.eval.doctrine_census import collect, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--games", type=int, default=500)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--envs", type=int, default=32)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        act, _ = onnx_policy(a.model)
        data = collect(act, a.games, a.seed, envs=a.envs)
        data["meta"] = {"model": os.path.basename(a.model), "seed": a.seed, "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump(data, f)
        print(f"{data['games']} games, {len(data['plays'])} plays, {data['meta']['seconds']}s")
    else:
        pooled = {"plays": [], "headlines": [], "coups": [], "boards": [], "games": 0}
        model = None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            if model not in (None, part["meta"]["model"]):
                raise SystemExit(f"{fn} was played by {part['meta']['model']}, not {model}")
            model = part["meta"]["model"]
            seed = part["meta"]["seed"]
            for k in ("plays", "headlines", "coups", "boards"):
                for row in part[k]:
                    row["game"] = f"{seed}:{row['game']}"
                pooled[k] += part[k]
            pooled["games"] += part["games"]
        md, summary = report(pooled, {"model": model})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump(summary, f)
        print(md)


if __name__ == "__main__":
    main()
