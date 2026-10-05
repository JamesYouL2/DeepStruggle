#!/usr/bin/env python3
"""How differently do these models play? Pairwise disagreement of their policies on shared positions.

For an opponent pool the useful members are strong models that choose differently. A census of
event rates hints at style but averages over everything; this asks every model for its move at the
same positions -- drawn from the first model's own play at the rollout temperature, 1 in 8
non-forced decisions (ai/eval/target_forms.collect_positions) -- and reports:

* the share of positions where two models' top moves differ, for every pair;
* the mean total-variation distance between their move distributions, for every pair;
* each model's disagreement with the plurality top move, by decision segment.

    PYTHONPATH=.:build/release python tools/policy_disagreement.py \\
        --models <a.pt> <b.pt> ... --positions 2000 --seed 1 --output-md report.md --dump tops.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def label(path: str) -> str:
    """A short name: the file name, or run/snapshot for a run directory's snapshot."""
    base = os.path.basename(path)[:-3] if path.endswith(".pt") else os.path.basename(path)
    if base.startswith("snapshot_") and base.endswith("steps"):
        run = os.path.basename(os.path.dirname(path)).split("_")[0]
        return f"{run if run.startswith('E') else 'snapshot'}@{int(base[9:-5]) // 1_000_000}M"
    return base


def distributions(model, positions, batch: int = 512) -> np.ndarray:
    """(positions, 220) move probabilities over each position's legal moves."""
    import torch

    from ai.eval.target_forms import _forward

    out = []
    for lo in range(0, len(positions), batch):
        chunk = positions[lo:lo + batch]
        lg, masks, _v = _forward(model, [p.state for p in chunk], "cpu")
        lg = np.where(masks > 0, lg, -np.inf)
        z = lg - lg.max(axis=1, keepdims=True)
        e = np.exp(z)
        out.append(e / e.sum(axis=1, keepdims=True))
    return np.concatenate(out)


def report(names: Sequence[str], probs: Sequence[np.ndarray], segments: Sequence[str]) -> str:
    k = len(names)
    tops = [p.argmax(axis=1) for p in probs]
    n = len(segments)
    short = [f"m{i + 1}" for i in range(k)]
    lines = ["# Policy disagreement", "",
             f"{n} positions from {names[0]}'s own play (rollout temperature, 1 in 8 non-forced "
             "decisions). Models:", ""]
    lines += [f"* **{s}** {nm}" for s, nm in zip(short, names)]
    lines += ["", "## Top move differs (% of positions)", "",
              "| | " + " | ".join(short) + " |", "|:---|" + "---:|" * k]
    for i in range(k):
        cells = ["—" if i == j else f"{100 * np.mean(tops[i] != tops[j]):.1f}" for j in range(k)]
        lines.append(f"| **{short[i]}** | " + " | ".join(cells) + " |")
    lines += ["", "## Mean total-variation distance between move distributions", "",
              "| | " + " | ".join(short) + " |", "|:---|" + "---:|" * k]
    for i in range(k):
        cells = ["—" if i == j else f"{0.5 * np.abs(probs[i] - probs[j]).sum(axis=1).mean():.3f}"
                 for j in range(k)]
        lines.append(f"| **{short[i]}** | " + " | ".join(cells) + " |")
    # Plurality top move per position, and each model's disagreement with it by segment.
    stack = np.stack(tops, axis=1)
    plural = np.array([Counter(row.tolist()).most_common(1)[0][0] for row in stack])
    segs = sorted(set(segments))
    seg_arr = np.array(segments)
    lines += ["", "## Disagreement with the plurality top move, by segment (%)", "",
              "| model | all | " + " | ".join(segs) + " |", "|:---|---:|" + "---:|" * len(segs)]
    for i in range(k):
        cells = [f"{100 * np.mean(tops[i][seg_arr == s] != plural[seg_arr == s]):.1f}" for s in segs]
        lines.append(f"| {short[i]} {names[i]} | {100 * np.mean(tops[i] != plural):.1f} | "
                     + " | ".join(cells) + " |")
    lines += ["", "Positions per segment: " + ", ".join(f"{s} {int((seg_arr == s).sum())}" for s in segs)]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="torch checkpoints; the first one's "
                    "self-play supplies the positions")
    ap.add_argument("--positions", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-md")
    ap.add_argument("--dump", help="per-position top moves of every model (json)")
    a = ap.parse_args(argv)

    import torch

    from ai.eval.target_forms import collect_positions
    from tools.lib.player_agent import load_agent

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    models = []
    for m in a.models:
        net = getattr(load_agent(m, device="cpu"), "model")
        net.eval()
        models.append(net)
    names = [label(m) for m in a.models]
    if len(set(names)) != len(names):
        names = [f"{nm}#{i + 1}" for i, nm in enumerate(names)]
    pos = collect_positions(models[0], a.positions, a.seed)
    probs = [distributions(net, pos) for net in models]
    segments = [p.segment for p in pos]
    md = report(names, probs, segments)
    if a.output_md:
        with open(a.output_md, "w", encoding="utf-8") as fh:
            fh.write(md)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as fh:
            json.dump({"models": names, "segments": segments,
                       "tops": [p.argmax(axis=1).tolist() for p in probs]}, fh)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
