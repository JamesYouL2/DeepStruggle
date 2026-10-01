"""Ops-efficient realignment spots at DEFCON 2, judged by paired playouts (`ai/eval/realign_spots.py`).

    # one part: collect 200 spots from self-play and play each spot's three branches 32 times
    PYTHONPATH=.:build/release .venv/bin/python tools/realign_spots.py run \\
        --model data/checkpoints/E6-06-44@soup_680-760.onnx --spots 200 --pairs 32 --seed 1 \\
        --out parts/part-1.json
    # pool the parts
    ... report --parts parts/part-*.json --out-md report.md --out-json report.json

Each part draws its own self-play games from `--seed`, so parts with different seeds hold
different spots. On CI: `.github/workflows/realign_spots.yml`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.branch_oracle import onnx_policy  # noqa: E402
from ai.eval.realign_spots import collect, play_spots, report  # noqa: E402
from tools.lib.player_agent import OnnxAgent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--spots", type=int, default=200)
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--envs", type=int, default=16, help="self-play games in parallel; few, so spots spread over many games")
    r.add_argument("--accept", type=float, default=0.03, help="probability a seen spot is kept")
    r.add_argument("--temperature", type=float, default=0.1, help="self-play sampling temperature")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        act, probs = onnx_policy(a.model)
        sampler = OnnxAgent(a.model, seed=a.seed)
        col = collect(act, probs, a.spots, a.seed, envs=a.envs, accept=a.accept,
                      sample=lambda o, m: sampler.act_batch(o, m, a.temperature, False))
        rows = play_spots(col["spots"], a.pairs, act, a.seed)
        meta = {"model": os.path.basename(a.model), "seed": a.seed, "seen": col["seen"],
                "greedy_realign": col["greedy_realign"], "games": col["games"],
                "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f)
        print(f"{len(rows)} spots, {meta['seen']} seen in {meta['games']} games, {meta['seconds']}s")
    else:
        rows, meta = [], {"model": None, "seen": 0, "greedy_realign": 0, "games": 0, "parts": 0}
        files = sorted({f for pat in a.parts for f in glob.glob(pat)})
        for fn in files:
            with open(fn) as f:
                part = json.load(f)
            rows += part["rows"]
            m = part["meta"]
            if meta["model"] not in (None, m["model"]):
                raise SystemExit(f"{fn} was played by {m['model']}, not {meta['model']}")
            meta["model"] = m["model"]
            for k in ("seen", "greedy_realign", "games"):
                meta[k] += m[k]
            meta["parts"] += 1
        md, js = report(rows, meta)
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump({**js, "rows": rows}, f)
        print(md)


if __name__ == "__main__":
    main()
