"""P33d: does the rollout root need more candidates than the network's top 4?

The root at k = 8 on positions of the searched player's own self-play (the searched-leak bank: R39 +
rollout root, population-weighted), recording where its pick ranks under the network. If the root
departs to ranks 5-8 often (> ~25% of its departures), k = 6 at play time and a recall objective
(R51) are worth their cost; below 10%, the candidate channel is closed
(research/plans/P33_search_preserved_gains.md).

    PYTHONPATH=.:build/release python -m ai.eval.candidate_recall --bank bank.jsonl.gz \\
        --checkpoint swa.pt [--positions 2000] [--k 8] [--worlds 16]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.rollout_root import RolloutConfig, RolloutRoot
from tools.lib.corpus_driver import state_from_token
from tools.lib.player_agent import NeuralAgent


def load_positions(bank_path: str, n: int, seed: int) -> List[Tuple[ts.GameState, float, str]]:
    """`n` bank positions drawn uniformly (each keeps its population weight), with their decision kind."""
    rows = []
    with gzip.open(bank_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["pos"], float(r["weight"]), str(r["decision_type"])))
    random.Random(seed).shuffle(rows)
    return [(state_from_token(p), w, dt) for p, w, dt in rows[:n]]


def pick_ranks(root: RolloutRoot, states: Sequence[ts.GameState], chunk: int = 256) -> List[Optional[int]]:
    """Per position, the rank (0 = the network's own move) of the root's pick among its candidates
    in the network's order; None where the root did not search (one legal move, or no world kept)."""
    ranks: List[Optional[int]] = []
    for lo in range(0, len(states), chunk):
        part = list(states[lo:lo + chunk])
        picks = root.choose(part)
        for st, pick in zip(root.last_stats, picks):
            ranks.append(None if st["worlds"] == 0 else st["candidates"].index(pick))
    return ranks


def report(ranks: Sequence[Optional[int]], weights: Sequence[float], kinds: Sequence[str], k: int) -> str:
    rk = np.asarray([-1 if r is None else r for r in ranks])
    w = np.asarray(weights, dtype=np.float64)
    kd = np.asarray(kinds)
    out: List[str] = []

    def line(sel: np.ndarray, title: str) -> str:
        searched = sel & (rk >= 0)
        dep = searched & (rk > 0)
        far = searched & (rk >= 4)
        if not dep.any():
            return f"  {title:20s} searched {int(searched.sum()):5d}, no departures"
        share_dep = w[dep].sum() / w[searched].sum()
        share_far = w[far].sum() / w[dep].sum()
        n_far, n_dep = int(far.sum()), int(dep.sum())
        se = float(np.sqrt(max(n_far / n_dep * (1 - n_far / n_dep), 1e-12) / n_dep))
        hist = " ".join(f"{r + 1}:{100 * w[searched & (rk == r)].sum() / w[searched].sum():.1f}%" for r in range(k))
        return (f"  {title:20s} searched {int(searched.sum()):5d}  departs {100 * share_dep:5.1f}%  "
                f"departures at rank 5-{k}: {100 * share_far:5.1f}% (unweighted {100 * n_far / n_dep:.1f}% ± {100 * se:.1f}, "
                f"{n_far} of {n_dep})  picks by rank {hist}")
    out.append(f"Rollout root at k = {k}: the rank under the network of the move it picks (population-weighted)")
    out.append(line(np.ones(len(rk), dtype=bool), "all"))
    for kind in sorted(set(kinds)):
        out.append(line(kd == kind, kind))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    pos = load_positions(a.bank, a.positions, a.seed)
    root = RolloutRoot(model, RolloutConfig(k=a.k, worlds=a.worlds, horizon=4, rule="z2", seed=a.seed), device=dev)
    ranks = pick_ranks(root, [p for p, _, _ in pos])
    print(f"checkpoint {a.checkpoint}, {len(pos)} bank positions, k={a.k}, {a.worlds} worlds, 4 boundaries, z2")
    print(report(ranks, [w for _, w, _ in pos], [d for _, _, d in pos], a.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
