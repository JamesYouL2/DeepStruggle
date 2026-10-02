"""Region weight: where each side puts its influence, region by region, against humans; and whether
keeping a play in each region beats the model's own (`ai/eval/region_weight.py`).

    # one part of eight: its self-play, its share of the corpus, 120 influence plays × 16 pairs
    PYTHONPATH=.:build/release .venv/bin/python tools/region_weight.py run \\
        --model data/checkpoints/<model>.onnx --plays 120 --pairs 16 --part 1/8 --out parts/part-1.json.gz
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

The human half needs the ts-replayer corpus; `--part k/0` skips it. On CI: `.github/workflows/region_weight.yml`.
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
from ai.eval.region_weight import collect, human_counts, play, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--plays", type=int, default=120, help="influence plays played out per part")
    r.add_argument("--pairs", type=int, default=16)
    r.add_argument("--part", default="1/1", help="k/N: seed k and 1/N of the corpus (k/0: no corpus)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        k, n = (int(x) for x in a.part.split("/"))
        act, _ = onnx_policy(a.model)
        starts, bot, bot_games = collect(act, a.plays, max(1, k))
        print(f"{bot_games} games, {sum(bot.values())} points, {len(starts)} plays ({round(time.time() - t0)}s)", flush=True)
        hum, hum_games = {}, 0
        if n > 0:
            from tools.lib.corpus_paths import distinct_corpus_files
            paths, _ = distinct_corpus_files()
            mine = [p for i, p in enumerate(sorted(paths, key=str)) if i % n == k - 1]
            hum, hum_games = human_counts(mine)
        print(f"{hum_games} human games ({round(time.time() - t0)}s)", flush=True)
        rows = []
        for lo in range(0, len(starts), 20):
            rows += play(act, starts[lo:lo + 20], a.pairs, max(1, k) * 7_919 + lo)
            print(f"  {min(lo + 20, len(starts))}/{len(starts)} ({round(time.time() - t0)}s)", flush=True)
        meta = {"model": os.path.basename(a.model), "pairs": a.pairs, "part": a.part, "bot_games": bot_games,
                "human_games": hum_games, "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "bot": bot, "human": hum, "rows": rows}, f)
    else:
        bot, hum, rows, meta = {}, {}, [], None
        bg = hg = 0
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            key = {k: part["meta"][k] for k in ("model", "pairs")}
            if meta not in (None, key):
                raise SystemExit(f"{fn} was run as {key}, not {meta}")
            meta = key
            for src, dst in ((part["bot"], bot), (part["human"], hum)):
                for k, v in src.items():
                    dst[k] = dst.get(k, 0) + v
            rows += part["rows"]
            bg += part["meta"]["bot_games"]
            hg += part["meta"]["human_games"]
        md, summary = report(bot, hum, rows, {**(meta or {}), "bot_games": bg, "human_games": hg})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump(summary, f)
        print(md)


if __name__ == "__main__":
    main()
