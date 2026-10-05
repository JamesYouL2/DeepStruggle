#!/usr/bin/env python3
"""Search's root value against the critic's, as predictors of the result (ai/eval/value_targets.py).

One part:

    PYTHONPATH=.:build/release python tools/value_targets_probe.py --model <ckpt.pt> \\
        --positions 250 --seed 1 --playouts 64 --dump part-1.jsonl

Pool parts:

    python tools/value_targets_probe.py --merge part-*.jsonl --output-md report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ai.eval import value_targets as V

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--positions", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--playouts", type=int, default=64)
    ap.add_argument("--forms", nargs="+", default=list(V.DEFAULT_FORMS))
    ap.add_argument("--dump")
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--output-md")
    a = ap.parse_args(argv)

    if a.merge:
        rows: List[dict] = []
        forms: List[str] = []
        for p in a.merge:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    rows.append(r)
                    for f in r["values"]:
                        if f != "critic" and f not in forms:
                            forms.append(f)
        md = V.report(rows, forms)
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
    rows = V.build(model, a.positions, a.seed, a.forms, a.playouts)
    print(f"[probe] {len(rows)} positions, {a.playouts} playouts each, in {time.time() - t0:.0f}s",
          flush=True)
    if a.dump:
        V.dump(rows, a.dump)
    print(V.report(rows, a.forms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
