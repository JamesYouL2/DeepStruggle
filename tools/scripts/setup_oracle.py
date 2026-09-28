#!/usr/bin/env python3
"""Opening oracle: one side's own setup against forced alternatives, over many deals.

A setup the policy plays with probability ~1 is never sampled differently in training, so the
policy never learns whether an alternative is better. This asks the game directly. Each deal is
played by the checkpoint up to `--side`'s setup (the USSR places first, so for the USSR that is the
start of the game), which is where the branches split:

* **own** -- the checkpoint places that side's setup itself;
* **each named opening** -- that side's placements are forced in the order given. For the US: the
  seven Western Europe points first, then the two bonus points, which the engine offers only where
  the US already has influence. For the USSR: its six Eastern Europe points. The other side then
  sets up as the checkpoint would, in reply.

Every branch is then played to the end once per deal by the checkpoint on both sides, greedily.
The engine RNG is re-seeded identically in every branch of a deal. What is reported is the US win
rate of `--side` and its paired difference from `own`, beside the critic's own difference at the
first decision after that side's setup (read from the mover's side), in the same units -- whether
the critic sees what the playouts see.

    PYTHONPATH=.:build/release python tools/scripts/setup_oracle.py --checkpoint <snapshot.pt> \\
        --deals 2000 --opening "human=West Germany*4,Italy*2,France,Italy,Iran"
    ... --side USSR --opening "human=East Germany,Poland*4,Yugoslavia"
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from ai.eval.ops_block import NODE_OFFSET, country_table  # noqa: E402
from bindings.settle import SettleMode, settle  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.scripts.ops_block_oracle import playouts  # noqa: E402
from tools.scripts.ops_block_probe import _obs_mask  # noqa: E402


def _country_ids() -> Dict[str, int]:
    _, _, names = country_table()
    return {str(n): c for c, n in enumerate(names)}


def parse_opening(spec: str, ids: Dict[str, int]) -> Tuple[str, List[int]]:
    """`name=Country*n,Country,...` -> (name, [country id per point, in order])."""
    name, _, body = spec.partition("=")
    out: List[int] = []
    for part in body.split(","):
        country, _, n = part.strip().partition("*")
        out += [ids[country.strip()]] * (int(n) if n else 1)
    return name.strip(), out


@torch.no_grad()
def _greedy(model: torch.nn.Module, dev: torch.device, st: "ts.GameState") -> int:
    o, m = _obs_mask([st], ts.Player(int(st.ctx().decision_player)), dev)
    return int(model(o, m)[0].float().argmax(-1).item())


def setup_start(model: torch.nn.Module, dev: torch.device, seed: int,
                side: "ts.Player" = ts.Player.US) -> "ts.GameState":
    """A fresh deal, played by the checkpoint up to `side`'s first setup placement."""
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    settle(st, SettleMode.CHANCE)
    while st.ctx().decision_player != side:
        ts.Engine.step_flat(st, _greedy(model, dev, st), False)
        settle(st, SettleMode.CHANCE)
    return st


def play_setup(model: torch.nn.Module, dev: torch.device, start: "ts.GameState",
               forced: Sequence[int], side: "ts.Player" = ts.Player.US) -> Tuple["ts.GameState", List[int]]:
    """`side`'s setup from `start`, forcing `forced` in order and the checkpoint's choice after it."""
    st = start.clone()
    queue = list(forced)
    placed: List[int] = []
    while st.ctx().decision_player == side and st.current_phase == ts.Phase.SETUP:
        mask = np.asarray(ts.Engine.get_flat_action_mask(st, False))
        if queue:
            a = NODE_OFFSET + queue.pop(0)
            if not mask[a]:
                raise RuntimeError(f"country {a - NODE_OFFSET} is not placeable after {placed}")
        else:
            a = _greedy(model, dev, st)
        placed.append(a - NODE_OFFSET)
        ts.Engine.step_flat(st, a, False)
        settle(st, SettleMode.CHANCE)
    if queue:
        raise RuntimeError(f"setup ended with {queue} unplaced")
    return st, placed


@torch.no_grad()
def _critic_us(model: torch.nn.Module, dev: torch.device, states: Sequence["ts.GameState"]) -> np.ndarray:
    """The critic's v_win from the US side for each state, read from the mover's observation (the
    only perspective the value head is trained on) and sign-converted. Negate for the USSR."""
    out = np.zeros(len(states))
    for i, st in enumerate(states):
        mover = ts.Player(int(st.ctx().decision_player))
        o, m = _obs_mask([st], mover, dev)
        v = float(model(o, m)[1].float().reshape(-1)[0].item())
        out[i] = v if mover == ts.Player.US else -v
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--opening", action="append", default=[],
                    help="name=Country*n,Country,... in placement order; repeatable")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--side", choices=["US", "USSR"], default="US",
                    help="Whose opening is varied; the other side sets up as the checkpoint would.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    ids = _country_ids()
    names = {v: k for k, v in ids.items()}
    openings = dict(parse_opening(s, ids) for s in a.opening)

    side = ts.Player.US if a.side == "US" else ts.Player.USSR
    sign = 1.0 if a.side == "US" else -1.0
    starts = [setup_start(model, dev, a.seed + g, side) for g in range(a.deals)]
    branches: Dict[str, List["ts.GameState"]] = {"own": []}
    own: Counter = Counter()
    for s in starts:
        end, placed = play_setup(model, dev, s, [], side)
        branches["own"].append(end)
        own[tuple(sorted(Counter(names[p] for p in placed).items()))] += 1
    for name, forced in openings.items():
        branches[name] = [play_setup(model, dev, s, forced, side)[0] for s in starts]
    print("own opening(s):", ", ".join(f"{dict(k)} x{n}" for k, n in own.most_common(3)))

    seeds = [7919 * g + 17 for g in range(a.deals)]
    score = {name: (sign * playouts(model, dev, sts, seeds, 0.0) + 1.0) / 2.0 for name, sts in branches.items()}
    critic = {name: sign * _critic_us(model, dev, sts) for name, sts in branches.items()}
    base, cbase = score["own"], critic["own"]
    for name, sc in score.items():
        d = sc - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        dc = critic[name] - cbase
        cse = dc.std(ddof=1) / np.sqrt(len(dc)) if len(dc) > 1 else 0.0
        # v_win is an expected result in [-1, 1]; half of it is a win-rate difference.
        print(f"{name:32s} {a.side} wins {100 * sc.mean():5.1f}%   vs own {100 * d.mean():+5.1f} ± {100 * se:.1f} pp "
              f"(paired, {len(d)} deals) | critic vs own {100 * dc.mean() / 2:+5.1f} ± {100 * cse / 2:.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
