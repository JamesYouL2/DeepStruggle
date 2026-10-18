"""Do the confirmed leaks survive a stronger follow-up? (P34, before D3.)

P33c's confirmed leak pairs were labelled by greedy paired playouts: the better move and the player's,
each played out by the network's own greedy play on both sides. A critic that "misses" a leak may be
missing something real -- or the label may be an artefact of the greedy follow-up. Here each pair is
replayed in the same keyed worlds (bindings/keyed_chance.hpp: the same deals and the same dice wherever
the games meet the same situation) under several follow-ups, so a change in a pair's lead is the
follow-up's alone:

* greedy -- the network on both sides, as the bank's labels;
* search, window w -- at every entry decision (action-round card choices, headline cards) from the
  move's turn through w turns after it, the keyed rollout root's move (tools.value_labels.SearchFollow);
  greedy elsewhere.

Per follow-up: the mean lead of the better move over the player's (points of win probability, the
bank's population weights and unweighted), how many pairs still lead, lead significantly (z >= 2) or
flip significantly, and the agreement with the bank's own labels.

    PYTHONPATH=.:build/release python -m ai.eval.leak_followup --bank bank.jsonl.gz \\
        --playouts confirm_playouts.jsonl.gz --model <checkpoint> [--worlds 64] [--windows 0 1]
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

import ts_engine as ts
from ai.eval.leak_critic import LeakPair, load_pairs
from bindings.ts_env import model_obs_features
from tools.lib.player_agent import NeuralAgent
from tools.value_labels import GraphedPolicy, SearchFollow, _drive, policy_streams


def leads(model: Any, device: torch.device, pairs: Sequence[LeakPair], worlds: int, seed: int,
          follow: Optional[SearchFollow], capacity: int = 16384) -> Dict[str, np.ndarray]:
    """Per pair: the mean over the shared worlds of (better move's score - player's move's score),
    its standard error, and the worlds used."""
    features = model_obs_features(model)
    pls = [ts.PlayoutLabels(features=features, worlds=worlds, capacity=capacity, seed=seed, keyed=True)
           for _ in range(2)]
    streams = policy_streams(device)
    pols = [GraphedPolicy(model, capacity, pls[0].obs_width(), device, streams=streams) for _ in range(2)]
    cursor = [0]
    per = max(1, capacity // (2 * worlds))
    got: List[Dict[str, np.ndarray]] = []

    def feed(pl: Any) -> None:
        while pl.queued() < per and cursor[0] < len(pairs):
            lo = cursor[0]
            hi = min(len(pairs), lo + per)
            pl.enqueue([p.state for p in pairs[lo:hi]], list(range(lo, hi)), [[p.best, p.played] for p in pairs[lo:hi]])
            cursor[0] = hi

    def sink(pl: Any) -> None:
        g = pl.take_siblings()
        if g["id"].size:
            got.append(g)
    _drive(pls, pols, feed, sink, follow)
    ids = np.concatenate([g["id"] for g in got])
    meta = np.concatenate([g["meta"] for g in got])
    y = np.concatenate([g["cols"] for g in got])[:, 0]
    table = np.full((len(pairs), 2, worlds), np.nan)
    ok = meta[:, 2] == 0
    table[ids[ok], meta[ok, 0], meta[ok, 1]] = y[ok]
    both = ~np.isnan(table).any(axis=1)
    diff = np.where(both, table[:, 0] - table[:, 1], 0.0)
    n = both.sum(1)
    mean = diff.sum(1) / np.maximum(n, 1)
    var = np.where(both, (diff - mean[:, None]) ** 2, 0.0).sum(1) / np.maximum(n - 1, 1)
    return {"lead": mean, "se": np.sqrt(var / np.maximum(n, 1)), "worlds": n}


def report(name: str, r: Dict[str, np.ndarray], pairs: Sequence[LeakPair], seconds: float,
           searched: int = 0, deviated: int = 0) -> str:
    lead, se = r["lead"], r["se"]
    w = np.asarray([p.weight for p in pairs])
    label = np.asarray([p.label for p in pairs])
    z = np.where(se > 0, lead / np.where(se > 0, se, 1.0), 0.0)
    sem = lead.std() / np.sqrt(lead.size)
    out = (f"{name:28s} lead {100 * lead.mean():+.2f} +- {100 * sem:.2f} pts (weighted {100 * np.average(lead, weights=w):+.2f})"
           f" | still ahead {np.mean(lead > 0):.1%}, z>=2 {np.mean(z >= 2):.1%}, flipped z<=-2 {np.mean(z <= -2):.1%}"
           f" | r with the bank's label {np.corrcoef(lead, label)[0, 1]:+.3f} | {seconds:.0f}s")
    if searched:
        out += f" | searched {searched:,} decisions, {deviated / searched:.0%} not greedy's"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--playouts", required=True, help="the confirmation playouts (the bank's labels)")
    ap.add_argument("--model", required=True, help="the playouts' network")
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--windows", type=int, nargs="*", default=[0, 1],
                    help="search follow-ups to run: turns after the move's in which entry decisions are searched")
    ap.add_argument("--search-worlds", type=int, default=8)
    ap.add_argument("--search-rule", default="argmax")
    ap.add_argument("--seed", type=int, default=91)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(a.bank, a.playouts)
    model = NeuralAgent.from_checkpoint(a.model, device=str(dev)).model.eval()
    label = np.asarray([p.label for p in pairs])
    print(f"{len(pairs)} confirmed pairs; the bank's label (2,048 greedy pairs, R39): lead {100 * label.mean():+.2f} pts, "
          f"{a.worlds} keyed worlds each here; playouts by {a.model}", flush=True)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    t0 = time.time()
    results["greedy"] = leads(model, dev, pairs, a.worlds, a.seed, None)
    print(report("greedy", results["greedy"], pairs, time.time() - t0), flush=True)
    for w in a.windows:
        t0 = time.time()
        follow = SearchFollow(model, dev, a.search_worlds, w, a.seed, a.search_rule)
        key = f"search {a.search_rule}@{a.search_worlds}, window {w}"
        results[key] = leads(model, dev, pairs, a.worlds, a.seed, follow)
        print(report(key, results[key], pairs, time.time() - t0, follow.searched, follow.deviated), flush=True)
        g, s = results["greedy"]["lead"], results[key]["lead"]
        print(f"{'':28s} vs greedy: r {np.corrcoef(g, s)[0, 1]:+.3f}, mean change {100 * (s - g).mean():+.2f} pts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
