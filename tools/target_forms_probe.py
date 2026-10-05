#!/usr/bin/env python3
"""Compare search target forms by how often they depart from the prior and whether they are right
(ai/eval/target_forms.py).

One part (a CI runner, or locally):

    PYTHONPATH=.:build/release python tools/target_forms_probe.py --model <ckpt.pt> \\
        --positions 250 --seed 1 --pairs 32 --dump part-1.jsonl

Pool the parts into a report:

    python tools/target_forms_probe.py --merge part-*.jsonl --output-md report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ai.eval import target_forms as T

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="torch checkpoint (load_agent spec)")
    ap.add_argument("--positions", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=32, help="paired playouts per departure")
    ap.add_argument("--forms", nargs="+", default=list(T.DEFAULT_FORMS))
    ap.add_argument("--gumbel-k", type=int, default=8)
    ap.add_argument("--dump", help="write this part's rows (jsonl)")
    ap.add_argument("--merge", nargs="+", help="pool these parts' rows into a report")
    ap.add_argument("--output-md", help="the pooled report")
    a = ap.parse_args(argv)

    if a.merge:
        rows: List[dict] = []
        forms: List[str] = []
        for p in a.merge:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    rows.append(r)
                    for f in r["forms"]:
                        if f not in forms:
                            forms.append(f)
        md = T.report(rows, forms, T.sims_of(forms))
        if a.output_md:
            with open(a.output_md, "w", encoding="utf-8") as fh:
                fh.write(md)
        print(md)
        return 0

    if not a.model:
        ap.error("--model is required unless --merge")
    import torch
    from tools.lib.player_agent import load_agent

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = getattr(load_agent(a.model, device="cpu"), "model")
    model.eval()
    t0 = time.time()
    pos = T.collect_positions(model, a.positions, a.seed)
    print(f"[probe] {len(pos)} positions in {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    T.TargetBuilder(model, seed=a.seed, gumbel_k=a.gumbel_k).build(pos, a.forms)
    print(f"[probe] targets ({', '.join(a.forms)}) in {time.time() - t1:.0f}s", flush=True)
    items: List[Tuple["T.Position", int, int]] = []
    keys: List[Tuple[int, int]] = []
    for i, p in enumerate(pos):
        top0 = max(p.prior, key=lambda x: p.prior[x])
        alts = set()
        for f in a.forms:
            t = p.targets[f]
            top = max(t, key=lambda x: (t[x], p.prior[x]))
            if top != top0:
                alts.add(top)
        for alt in sorted(alts):
            items.append((p, alt, top0))
            keys.append((i, alt))
    t2 = time.time()
    adv = T.paired_advantage(model, items, a.pairs, a.seed)
    print(f"[probe] {len(items)} departures x {a.pairs} pairs in {time.time() - t2:.0f}s", flush=True)
    verdicts: Dict[Tuple[int, int], Tuple[float, float]] = dict(zip(keys, adv))
    rows = T.rows(pos, a.forms, verdicts)
    if a.dump:
        T.dump(rows, a.dump)
    print(T.report(rows, a.forms, T.sims_of(a.forms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
