"""The reply probe: net against search, one action round at a time (`ai/eval/reply_probe.py`).

    # one part: 96 round starts, search-64 determinized on every mover decision, 32 reply pairs
    PYTHONPATH=.:build/release .venv/bin/python tools/reply_probe.py run \\
        --model data/checkpoints/<model>.onnx --rounds 96 --pairs 32 --seed 1 --out parts/part-1.json.gz
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

On CI: `.github/workflows/reply_probe.yml`.
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
from ai.eval.reply_probe import collect, opponent_replies, rounds_left, play_replies, play_rounds, report, summarise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--search", default="64:determinize",
                   help="search spec after the model path, as tools/lib/player_agent.load_agent reads it")
    r.add_argument("--rounds", type=int, default=96, help="round starts to play both branches from")
    r.add_argument("--pairs", type=int, default=32, help="reply playouts per branch per disagreement")
    r.add_argument("--batch", type=int, default=32, help="round starts searched together")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        from tools.lib.player_agent import load_agent

        t0 = time.time()
        act, _ = onnx_policy(a.model)
        search = load_agent(f"search:{a.model}:{a.search}", device="cpu")
        starts = collect(act, a.rounds, a.seed)
        print(f"{len(starts)} round starts ({round(time.time() - t0)}s)", flush=True)
        rounds = []
        for lo in range(0, len(starts), a.batch):
            rounds += play_rounds(starts[lo:lo + a.batch], act, search, a.seed * 7_919 + lo)
            print(f"  rounds {min(lo + a.batch, len(starts))}/{len(starts)} ({round(time.time() - t0)}s)", flush=True)
        differ = [x for x in rounds if [m for m, _ in x["net"]["moves"]] != [m for m, _ in x["search"]["moves"]]]
        replies = play_replies(differ, act, a.pairs, a.seed)
        rows = summarise(differ, replies)
        meta = {"model": os.path.basename(a.model), "search": f"search:{a.search}", "seed": a.seed,
                "pairs": a.pairs, "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "rounds": len(rounds), "rows": rows,
                       "all": [[rounds_left(x["start"]), opponent_replies(x["start"])] for x in rounds]}, f)
        print(f"{len(rounds)} rounds, {len(rows)} disagreements, {meta['seconds']}s")
    else:
        pooled = {"rounds": 0, "rows": [], "all": []}
        meta = None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            key = {k: part["meta"][k] for k in ("model", "search", "pairs")}
            if meta not in (None, key):
                raise SystemExit(f"{fn} was run as {key}, not {meta}")
            meta = key
            pooled["rounds"] += part["rounds"]
            pooled["rows"] += part["rows"]
            pooled["all"] += part["all"]
        md, summary = report(pooled, meta or {})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump({**summary, "rows": pooled["rows"]}, f)
        print(md)


if __name__ == "__main__":
    main()
