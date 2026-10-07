#!/usr/bin/env python3
"""How clear is each disagreement-bank position? A stronger search, and the three moves valued.

For every position of a disagreement bank (`disagreement_bank.py`; human-alone rows are left out by
default -- there are 21,000 of them):

* **stronger search** -- the move a Gumbel root with k = 8 candidates and 256 simulations plays
  (the strongest configuration measured, `research/log/E7_gumbel_headroom.md`), and whether it is
  the human's, the network's, the weak search's, or another move;
* **the three moves valued** -- each distinct move among human / network / weak search is applied,
  and the position after it is searched with `--sims` simulations, in each of `--worlds` worlds
  whose unseen cards are redealt from the mover's side; a move's value is the mover's mean over the
  worlds, with its standard error;
* **clarity** -- the best move's margin over the runner-up in standard errors (`z`), and whether
  the stronger search picks the same move.

`--emit-playouts N` also writes the N clearest positions as the input of `bank_playouts.py`.

    PYTHONPATH=.:build/release python tools/scripts/bank_clarity.py --bank part*.jsonl.gz \\
        --checkpoint newest.pt --out clarity.jsonl.gz --emit-playouts 1000 playout_input.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.scripts.disagreement_bank import _name, mover_of, row_id
from tools.scripts.event_play_census import state_from_token

WHO = ("human", "network", "search")
_UINT64 = 1 << 64


def move_index(st: ts.GameState, name: str) -> int:
    """The legal move a bank row names (names are unique among a decision's legal moves)."""
    hits = [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st))) if _name(st, int(a)) == name]
    if len(hits) != 1:
        raise ValueError(f"{name!r} names {len(hits)} legal moves here")
    return hits[0]


def load_rows(banks: Sequence[str], patterns: Sequence[str]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for path in banks:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["pattern"] in patterns:
                    seen.setdefault(row_id(r), r)
    return [dict(r, id=rid) for rid, r in seen.items()]


def value_moves(mcts: Any, states: Sequence[ts.GameState], moves: Sequence[List[int]], sims: int,
                worlds: int, seed: int) -> List[Dict[int, Tuple[float, float]]]:
    """Per position, each move's mean value for the mover over `worlds` redeals, and its SE."""
    out: List[Dict[int, Tuple[float, float]]] = []
    for v in value_moves_per_world(mcts, states, moves, sims, worlds, seed):
        d: Dict[int, Tuple[float, float]] = {}
        for a, xs in v.items():
            arr = np.asarray([x for x in xs if x is not None], dtype=float)
            if len(arr):
                d[a] = (float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else math.nan)
        out.append(d)
    return out


def value_moves_per_world(mcts: Any, states: Sequence[ts.GameState], moves: Sequence[List[int]], sims: int,
                          worlds: int, seed: int) -> List[Dict[int, List[Optional[float]]]]:
    """Per position, each move's value for the mover in each of `worlds` redeals (None where the
    redeal made it illegal). Every move of a position is searched in the same worlds, so the
    difference between two moves can be taken world by world."""
    from ai.search.batched_mcts import BatchedMCTS, settle
    from ai.search.dmcts import determinize

    cfg = replace(mcts.cfg, simulations=sims, determinize=False, node_filter="all", subsample=1.0,
                  gumbel_k=0, reuse_subtree=False, dirichlet_frac=0.0)
    sub = BatchedMCTS(mcts.model, device=mcts.device, config=cfg, featurise_capacity=mcts._featurise_capacity)
    rng = random.Random(seed)
    jobs: List[Tuple[int, int, int, ts.GameState]] = []
    for i, (st, ms) in enumerate(zip(states, moves)):
        for w in range(worlds):
            world = determinize(st.clone(), mover_of(st), rng)
            world.rng_state = rng.getrandbits(64) % _UINT64
            for a in ms:
                child = world.clone()
                if not np.asarray(ActionEncoder.get_legal_mask(child))[a]:
                    continue                   # the redeal made it illegal (Cambridge Five and the like)
                ts.Engine.step_flat(child, a)
                settle(child, cfg.auto_advance)
                jobs.append((i, w, a, child))
    vals: List[Dict[int, List[Optional[float]]]] = [{a: [None] * worlds for a in ms} for ms in moves]
    roots = sub._search([j[3] for j in jobs])
    for (i, w, a, child), r in zip(jobs, roots):
        if r is None:
            continue
        if r.terminal or not r.actions:
            v_us = float(r.value_us)
        else:
            v_us = (float(r.value_us) + float(sum(r.w))) / (1.0 + float(sum(r.n)))
        vals[i][a][w] = v_us if mover_of(states[i]) == ts.Player.US else -v_us
    return vals


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.lib.player_agent import load_agent

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", nargs="+", required=True, help="disagreement_bank.py outputs (JSONL.gz)")
    ap.add_argument("--checkpoint", required=True, help="the searcher's .pt (onnx_to_checkpoint.py)")
    ap.add_argument("--patterns", nargs="+", default=["network-alone", "all-differ", "search-alone"])
    ap.add_argument("--strong-sims", type=int, default=256)
    ap.add_argument("--strong-k", type=int, default=8)
    ap.add_argument("--sims", type=int, default=64, help="simulations per move and world for the values")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64, help="positions per search call")
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit-playouts", nargs=2, default=None, metavar=("N", "PATH"),
                    help="write the N clearest positions as bank_playouts.py's input")
    a = ap.parse_args(argv)

    rows = load_rows(a.bank, a.patterns)
    k, n = (int(x) for x in a.part.split("/"))
    rows = [r for i, r in enumerate(sorted(rows, key=lambda r: r["id"])) if i % n == k - 1][: a.limit or None]
    agent: Any = load_agent(f"gumbel:{a.checkpoint}:{a.strong_sims}:{a.strong_k}", device="cpu")
    agent.reseed(a.seed)
    mcts = agent.mcts
    t0 = time.time()
    out: List[Dict[str, Any]] = []
    for lo in range(0, len(rows), a.chunk):
        group = rows[lo:lo + a.chunk]
        states = [state_from_token(r["pos"]) for r in group]
        idx = [{w: move_index(st, r[w]) for w in WHO} for st, r in zip(states, group)]
        strong = [int(x) for x in mcts.best_actions(states)]
        moves = [sorted(set(d.values())) for d in idx]
        values = value_moves(mcts, states, moves, a.sims, a.worlds, a.seed * 7919 + lo)
        for r, st, d, s_act, val in zip(group, states, idx, strong, values):
            ranked = sorted(val.items(), key=lambda kv: -kv[1][0])
            best_a, (best_q, best_se) = ranked[0]
            if len(ranked) > 1:
                _, (q2, se2) = ranked[1]
                spread = math.hypot(best_se if best_se == best_se else 0.0, se2 if se2 == se2 else 0.0)
                z = (best_q - q2) / spread if spread > 0 else math.inf
            else:
                z = math.nan
            strong_is = [w for w in WHO if d[w] == s_act] or ["other"]
            out.append({
                "id": r["id"], "pattern": r["pattern"], "kind": r["kind"],
                "a": d, "strong": _name(st, s_act), "strong_is": strong_is,
                "q": {w: [round(val[d[w]][0], 4), round(val[d[w]][1], 4)] for w in WHO if d[w] in val},
                "best": [w for w in WHO if d[w] == best_a], "z": round(z, 2) if z == z and z != math.inf else None,
                "strong_agrees": s_act == best_a,
            })
        print(f"[{lo + len(group)}/{len(rows)}] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    with gzip.open(a.out, "wt") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    if a.emit_playouts:
        top = int(a.emit_playouts[0])
        by_id = {r["id"]: r for r in rows}
        clear = sorted((o for o in out if o["z"] is not None), key=lambda o: -o["z"])[:top]
        with gzip.open(a.emit_playouts[1], "wt") as f:
            for o in clear:
                f.write(json.dumps({"id": o["id"], "pos": by_id[o["id"]]["pos"], "a": o["a"]}) + "\n")
    print(f"{len(out)} positions in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
