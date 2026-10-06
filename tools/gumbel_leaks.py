#!/usr/bin/env python3
"""Where the raw network leaks: its move against a Gumbel root's, priced by paired playouts
(ai/eval/gumbel_leaks.py).

One part (a CI runner, or locally):

    tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python \\
        tools/gumbel_leaks.py --model <ckpt.pt | export.onnx> --positions 300 --seed 1 --pairs 32 --dump part-1.jsonl

Pool the parts into a report, with each leak linked into the workbench at its position:

    python tools/gumbel_leaks.py --merge part-*.jsonl --output-md report.md \\
        --workbench-model hf:mihaild/deepstruggle@main:<model>.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ai.eval import gumbel_leaks as L

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a checkpoint (.pt) or a published export (.onnx, run in ONNX Runtime)")
    ap.add_argument("--positions", type=int, default=300, help="positions sampled by this part")
    ap.add_argument("--seed", type=int, default=1, help="seeds the games, the root and the playouts")
    ap.add_argument("--pairs", type=int, default=32, help="paired playouts per departure")
    ap.add_argument("--sims", type=int, default=32, help="the root's simulations")
    ap.add_argument("--k", type=int, default=4, help="the root's candidates")
    ap.add_argument("--fpu", type=float, default=0.2, help="first-play urgency reduction")
    ap.add_argument("--temperature", type=float, default=L.DEFAULT_TEMPERATURE,
                    help="sampling temperature of the games positions are drawn from")
    ap.add_argument("--dump", help="write this part's rows (jsonl)")
    ap.add_argument("--merge", nargs="+", help="pool these parts' rows into a report")
    ap.add_argument("--output-md", help="the pooled report")
    ap.add_argument("--workbench", default=L.DEFAULT_WORKBENCH, help="workbench address the leak links open")
    ap.add_argument("--workbench-model", default="",
                    help="the workbench's model= for the links, e.g. hf:<repo>@main:<file>.onnx")
    ap.add_argument("--top", type=int, default=40, help="rows in each ranked table")
    a = ap.parse_args(argv)

    if a.merge:
        rows, metas = L.load(a.merge)
        md = L.report(rows, metas, workbench=a.workbench, workbench_model=a.workbench_model, top=a.top)
        if a.output_md:
            with open(a.output_md, "w", encoding="utf-8") as fh:
                fh.write(md)
        print(md)
        return 0

    if not a.model:
        ap.error("--model is required unless --merge")
    import torch
    from tools.lib.player_agent import load_network

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(a.seed)
    model = load_network(a.model, device="cpu")
    spec = L.RootSpec(sims=a.sims, k=a.k, fpu=a.fpu)
    rows, meta = L.run_part(model, a.positions, a.seed, a.pairs, spec, temperature=a.temperature)
    meta["model"] = os.path.basename(a.model)
    if a.dump:
        L.dump(rows, meta, a.dump)
    print(L.report(rows, [meta], workbench=a.workbench, workbench_model=a.workbench_model, top=a.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
