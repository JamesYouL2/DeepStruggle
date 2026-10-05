#!/usr/bin/env python3
"""What breaking a doctrine rule costs the model, by paired playouts (ai/eval/doctrine_playouts.py).

One part (greedy self-play from seed 1, then the playouts):

    PYTHONPATH=.:build/release python tools/doctrine_playouts.py --model <ckpt.pt> \\
        --games 250 --seed 1 --pairs 64 --dump part-1.jsonl

Pool parts:

    python tools/doctrine_playouts.py --merge part-*.jsonl --output-md report.md

On CI: .github/workflows/doctrine_playouts.yml.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ai.eval import doctrine_playouts as P

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a torch checkpoint (.pt)")
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=64, help="paired playouts per position")
    ap.add_argument("--max-per-rule", type=int, default=0, help="positions measured per rule (0 = all)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rules", nargs="+", default=list(P.DEFAULT_RULES),
                    help="census rule names whose compliance is eventing the card")
    ap.add_argument("--dump")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--output-md")
    a = ap.parse_args(argv)

    if a.merge:
        rows, meta = P.merge(a.merge)
        md = P.report(rows, meta, a.rules)
        if a.output_md:
            with open(a.output_md, "w", encoding="utf-8") as fh:
                fh.write(md)
        print(md)
        return 0
    if not a.model:
        ap.error("--model is required unless --merge")

    import torch

    from tools.doctrine_census import greedy_policy
    from tools.lib.player_agent import NeuralAgent

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    act, feats = greedy_policy(a.model)
    if feats:
        raise SystemExit("paired playouts read the base observation; this model has obs features")
    got = P.collect_breaks(act, a.games, a.seed, a.rules, envs=a.envs, max_per_rule=a.max_per_rule)
    print(f"[collect] {got['games']} games, broken {got['broken']}, measuring {len(got['items'])} "
          f"in {time.time() - t0:.0f}s", flush=True)
    model = NeuralAgent.from_checkpoint(a.model, device="cpu").model.eval()
    rows = P.measure(model, got["items"], a.pairs, a.seed)
    meta = {"games": got["games"], "pairs": a.pairs, "applied": got["applied"], "broken": got["broken"]}
    print(f"[playouts] {len(rows)} positions x {a.pairs} pairs in {time.time() - t0:.0f}s", flush=True)
    if a.dump:
        P.dump(rows, meta, a.dump)
    print(P.report(rows, meta, a.rules))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
