"""The position bank: positions with every candidate move played out, saved for later queries
(`ai/eval/position_bank.py`, asked with `ai/eval/bank_query.py`).

    # one part: 400 decisions from greedy self-play, 32 paired playouts per candidate
    PYTHONPATH=.:build/release .venv/bin/python tools/position_bank.py run \\
        --model data/checkpoints/<model>.onnx --decisions 400 --pairs 32 --seed 1 --out parts/bank-1.json.gz
    # or only play-mode decisions on the card-rules targets (ai/eval/card_rules.py)
    ... run --model <model>.onnx --targeted --per-card 12 --games 400 --pairs 32 --seed 1 --out parts/bank-1.json.gz
    # pool parts into one bank, and ask it every built-in rule (or --card-rules)
    ... pool --parts parts/bank-*.json.gz --out bank.json.gz
    ... query --bank bank.json.gz [more banks...] --out-md report.md --out-json report.json

On CI: `.github/workflows/position_bank.yml`. Banks from several runs of the same model can be
queried together; a bank records the model and engine it was built with.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from typing import Dict, Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def merge_counts(parts: Iterable[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for part in parts:
        for key, row in part.items():
            tgt = out.setdefault(key, {})
            for m, n in row.items():
                tgt[m] = tgt.get(m, 0) + n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--decisions", type=int, default=400)
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument("--targeted", action="store_true",
                   help="bank only play-mode decisions on ai/eval/card_rules.py's TARGETS (--per-card, --games)")
    r.add_argument("--forced", default=None, metavar="SET",
                   help="with --targeted: only a card_rules.FORCED_SETS set (vp, kd, nextvp, oss12, oss34), where its condition holds")
    r.add_argument("--per-card", type=int, default=12)
    r.add_argument("--games", type=int, default=400)
    p = sub.add_parser("pool")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("query")
    q.add_argument("--bank", nargs="+", required=True)
    q.add_argument("--out-md", required=True)
    q.add_argument("--out-json", required=True)
    q.add_argument("--card-rules", action="store_true", help="ask the card event rules (ai/eval/card_rules.py)")
    q.add_argument("--forced", default=None, metavar="SET", help="report a card_rules.FORCED_SETS set")
    a = ap.parse_args()

    if a.cmd == "run":
        from ai.eval.branch_oracle import onnx_policy
        from ai.eval.position_bank import build, collect
        from tools.lib.engine_fingerprint import fingerprint
        from tools.lib.player_agent import OnnxAgent

        t0 = time.time()
        act, probs = onnx_policy(a.model)
        agent = OnnxAgent(a.model)

        def value_fn(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
            _, v, _ = agent.session.run(None, {"obs": np.ascontiguousarray(obs, dtype=np.float32),
                                                "mask": np.ascontiguousarray(masks, dtype=np.uint8)})
            return np.asarray(v).reshape(-1)

        counts: dict = {}
        if a.targeted:
            from ai.eval.card_rules import forced_keep, target_ids
            from ai.eval.doctrine_census import cards
            from ai.eval.position_bank import collect_targets

            by_name = {str(v["name"]): k for k, v in cards().items()}
            if a.forced:
                keep = forced_keep(a.forced, by_name, act)
                positions, counts = collect_targets(act, set(keep), a.per_card, a.games, a.seed, keep=keep)
            else:
                positions, counts = collect_targets(act, target_ids(by_name), a.per_card, a.games, a.seed)
        else:
            positions = collect(act, a.decisions, a.seed)
        print(f"{len(positions)} decisions ({round(time.time() - t0)}s)", flush=True)
        records = []
        for lo in range(0, len(positions), 25):
            records += build(act, probs, value_fn, positions[lo:lo + 25], a.pairs, a.seed * 7_919 + lo)
            print(f"  {min(lo + 25, len(positions))}/{len(positions)} ({round(time.time() - t0)}s)", flush=True)
        meta = {"model": os.path.basename(a.model), "pairs": a.pairs, "seed": a.seed,
                "engine": fingerprint(), "seconds": round(time.time() - t0, 1), "targeted": a.targeted,
                "mode_counts": counts}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "records": records}, f)
        print(f"{len(records)} records, {meta['seconds']}s")
    elif a.cmd == "pool":
        records, metas = [], []
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            metas.append(part["meta"])
            records += part["records"]
        models = {m["model"] for m in metas}
        engines = {m["engine"] for m in metas}
        if len(models) > 1 or len(engines) > 1:
            raise SystemExit(f"parts from different models or engines: {models} {engines}")
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": {"model": models.pop(), "engine": engines.pop(), "parts": metas,
                                "mode_counts": merge_counts(m.get("mode_counts", {}) for m in metas)},
                       "records": records}, f)
        print(f"{len(records)} records from {len(metas)} parts -> {a.out}")
    else:
        from ai.eval.bank_query import load, report

        metas = []
        for p in a.bank:
            with gzip.open(p, "rt") as f:
                metas.append(json.load(f)["meta"])
        models = {m["model"] for m in metas}
        if len(models) > 1:
            raise SystemExit(f"banks from different models: {models}")
        records = load(a.bank)
        if a.forced:
            from ai.eval import card_rules
            from ai.eval.doctrine_census import cards

            md, summary = card_rules.forced_report(records, {k: str(v["name"]) for k, v in cards().items()}, a.forced,
                                                   {"model": models.pop(), "banks": len(a.bank)})
        elif a.card_rules:
            from ai.eval import card_rules
            from ai.eval.doctrine_census import cards

            names = {k: str(v["name"]) for k, v in cards().items()}
            counts = merge_counts(m.get("mode_counts", {}) for m in metas)
            md, summary = card_rules.report(records, names, counts, {"model": models.pop(), "banks": len(a.bank)})
        else:
            md, summary = report(records, meta={"model": models.pop(), "banks": len(a.bank)})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump(summary, f)
        print(md)


if __name__ == "__main__":
    main()
