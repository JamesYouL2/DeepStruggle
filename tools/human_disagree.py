"""The model against strong human play: where it disagrees, and who is right (`ai/eval/human_disagree.py`).

    # one part of eight: its share of the corpus's distinct games, 40 spots per kind, 32 pairs
    PYTHONPATH=.:build/release .venv/bin/python tools/human_disagree.py run \\
        --model data/checkpoints/<model>.onnx --part 1/8 --per-kind 40 --pairs 32 --out parts/part-1.json.gz
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

Needs the ts-replayer corpus (tools/download_ts_replayer.py). On CI: `.github/workflows/human_disagree.yml`.
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

from ai.eval.human_disagree import choose_spots, load_game, play, report, scan_game  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--part", default="1/1", help="k/N: this part's share of the corpus's distinct games")
    r.add_argument("--per-kind", type=int, default=40, help="disagreements played out per kind of decision")
    r.add_argument("--max-p", type=float, default=0.25, help="only where the model gave the human's move less")
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--games", type=int, default=0, help="at most this many games (0 = the whole share)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        from tools.lib.corpus_paths import distinct_corpus_files
        from tools.lib.player_agent import OnnxAgent

        t0 = time.time()
        k, n = (int(x) for x in a.part.split("/"))
        paths, _ = distinct_corpus_files()
        mine = [p for i, p in enumerate(sorted(paths, key=str)) if i % n == k - 1]
        if a.games:
            mine = mine[:a.games]
        agent = OnnxAgent(a.model)

        def act(obs, masks):
            return agent.act_batch(obs, masks, 0.0, True)

        scan, states = [], []
        for path in mine:
            rows, sts = scan_game(load_game(path), agent.logits, int(os.path.basename(str(path)).split(".")[0]))
            for row in rows:
                if "state_index" in row:
                    row["state_index"] += len(states)
            scan += rows
            states += sts
        print(f"{len(mine)} games, {len(scan)} decisions, {sum(not r['agree'] for r in scan)} disagreements "
              f"({round(time.time() - t0)}s)", flush=True)
        spots = choose_spots(scan, a.per_kind, a.max_p, seed=k)
        played = play(spots, states, act, a.pairs, seed=k)
        for row in scan:
            row.pop("state_index", None)
        meta = {"model": os.path.basename(a.model), "pairs": a.pairs, "max_p": a.max_p, "part": a.part,
                "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "scan": scan, "played": played}, f)
        print(f"{len(played)} spots played, {meta['seconds']}s")
    else:
        scan, played, meta = [], [], None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            key = {k: part["meta"][k] for k in ("model", "pairs", "max_p")}
            if meta not in (None, key):
                raise SystemExit(f"{fn} was run as {key}, not {meta}")
            meta = key
            scan += part["scan"]
            played += part["played"]
        md, summary = report(scan, played, meta or {})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump({**summary, "played": played}, f)
        print(md)


if __name__ == "__main__":
    main()
