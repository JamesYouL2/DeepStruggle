"""VP ledger and calibration: where each side's VP come from, the model against strong humans, and how
calibrated and swingy the model's win estimate is (`ai/eval/vp_ledger.py`).

    # one part of eight: 250 self-play games and 1/8 of the corpus's distinct games
    PYTHONPATH=.:build/release .venv/bin/python tools/vp_ledger.py run \\
        --model data/checkpoints/<model>.onnx --games 250 --part 1/8 --out parts/part-1.json.gz
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

The human half needs the ts-replayer corpus (tools/download_ts_replayer.py); `--part 0/0` skips it.
On CI: `.github/workflows/vp_ledger.yml`.
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

from ai.eval.vp_ledger import human_game, merge_cal, report, selfplay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--games", type=int, default=250)
    r.add_argument("--part", default="1/1", help="k/N: this part's seed and share of the corpus (0/0: no corpus)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        from tools.lib.player_agent import OnnxAgent

        t0 = time.time()
        k, n = (int(x) for x in a.part.split("/"))
        agent = OnnxAgent(a.model)

        def policy(obs: np.ndarray, masks: np.ndarray):
            lg, v, _ = agent.session.run(None, {"obs": np.ascontiguousarray(obs, dtype=np.float32),
                                                "mask": np.ascontiguousarray(masks, dtype=np.uint8)})
            scores = np.where(masks.astype(bool), np.asarray(lg, dtype=np.float32), -np.inf)
            return scores.argmax(axis=1), np.asarray(v).reshape(-1)

        bot, cal = selfplay(policy, a.games, max(1, k))
        print(f"{len(bot)} self-play games ({round(time.time() - t0)}s)", flush=True)
        human = []
        if n > 0:
            from ai.eval.human_disagree import load_game
            from tools.lib.corpus_paths import distinct_corpus_files

            paths, _ = distinct_corpus_files()
            for i, path in enumerate(sorted(paths, key=str)):
                if i % n == k - 1:
                    row = human_game(load_game(path))
                    if row is not None:
                        human.append(row)
        print(f"{len(human)} human games ({round(time.time() - t0)}s)", flush=True)
        meta = {"model": os.path.basename(a.model), "part": a.part, "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "bot": bot, "human": human, "cal": cal}, f)
    else:
        bot, human, cals, model = [], [], [], None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            if model not in (None, part["meta"]["model"]):
                raise SystemExit(f"{fn} was played by {part['meta']['model']}, not {model}")
            model = part["meta"]["model"]
            bot += part["bot"]
            human += part["human"]
            cals.append(part["cal"])
        md, summary = report(bot, human, merge_cal(cals), {"model": model})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump(summary, f)
        print(md)


if __name__ == "__main__":
    main()
