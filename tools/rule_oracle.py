"""The rule oracle: pooled tests of rules over generic features of the position (`ai/eval/rule_oracle.py`).

    # one part: 80 greedy self-play games, up to 400 spots a rule, 32 paired playouts per branch
    PYTHONPATH=.:build/release .venv/bin/python tools/rule_oracle.py run \\
        --model data/checkpoints/<model>.onnx --games 80 --pairs 32 --seed 1 --out parts/part-1.json.gz
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

On CI: `.github/workflows/rule_oracle.yml`.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.rule_oracle import collect, play, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--games", type=int, default=80)
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--rate", type=float, default=0.25, help="chance each rule is asked at a matching decision")
    r.add_argument("--cap", type=int, default=400, help="spots kept per rule per part")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        from ai.eval.branch_oracle import onnx_policy
        from tools.lib.player_agent import OnnxAgent

        t0 = time.time()
        act, probs = onnx_policy(a.model)
        agent = OnnxAgent(a.model)

        def value_fn(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
            _, v, _ = agent.session.run(None, {"obs": np.ascontiguousarray(obs, dtype=np.float32),
                                                "mask": np.ascontiguousarray(masks, dtype=np.uint8)})
            return np.asarray(v).reshape(-1)

        spots, games = collect(act, probs, value_fn, a.games, a.seed, rate=a.rate, cap=a.cap)
        counts = {k: sum(1 for s in spots if s["rule"] == k) for k in sorted({s["rule"] for s in spots})}
        print(f"{games} games, {len(spots)} spots {counts} ({round(time.time() - t0)}s)", flush=True)
        rows = []
        for lo in range(0, len(spots), 40):
            rows += play(act, spots[lo:lo + 40], a.pairs, a.seed * 7_919 + lo)
            print(f"  {min(lo + 40, len(spots))}/{len(spots)} ({round(time.time() - t0)}s)", flush=True)
        meta = {"model": os.path.basename(a.model), "pairs": a.pairs, "seed": a.seed, "games": games,
                "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "rows": rows}, f)
    else:
        rows, meta, games = [], None, 0
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            key = {k: part["meta"][k] for k in ("model", "pairs")}
            if meta not in (None, key):
                raise SystemExit(f"{fn} was run as {key}, not {meta}")
            meta = key
            rows += part["rows"]
            games += part["meta"]["games"]
        md, summary = report(rows, {**(meta or {}), "games": games})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump(summary, f)
        print(md)


if __name__ == "__main__":
    main()
