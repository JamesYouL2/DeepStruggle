#!/usr/bin/env python3
"""P30 analysis 4: is the placement shift towards a held scoring card's region the right size?

`card_country_interaction_probe.py` found that every trained net moves 5-24 points of placement
mass into a region when its scoring card is in hand. This measures, by playing games out, whether
it should move more or less.

Positions are the starts of influence plays from the checkpoint's own self-play (P27's
collector). For each region whose countries the mover can place in, and where the checkpoint's
own greedy allocation puts **no** point in that region, two branches are played to the end:

* **policy** -- the checkpoint's greedy allocation;
* **one in** -- the first point goes to the region's country the checkpoint ranks highest, then
  the checkpoint places the rest.

Each branch is played `--pairs` times by the checkpoint on both sides, pair k of a position
re-seeding the engine identically in both, so the difference is the placement's. The positions
are split by whether the mover **holds** that region's scoring card:

* `one in - policy` when holding > 0: the net places too little in the region it holds the card for;
* the holding minus not-holding difference is what the card is worth to a point in its region,
  which is the interaction the net is supposed to have learned.

`--direction out` measures the other margin: positions where the policy **did** place in the
region, against the branch that places every point elsewhere (the checkpoint's best legal action
outside the region at each step). There `none in - policy` < 0 when holding means the policy's
points in that region were worth making, i.e. it is not over-placing.

    PYTHONPATH=.:build/release python tools/scripts/scoring_region_oracle.py \\
        --checkpoint <snapshot.pt> --starts 3000 --per-cell 100 --pairs 16 --output-json o.json --output-md o.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from ai.eval.ops_block import (NODE_OFFSET, Allocation, _ctx, _in_block, _legal, influence,  # noqa: E402
                               play_block)
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.scripts.card_country_interaction_probe import CARD_OFFSET, CARD_W, MY_HAND, REGIONS, _regions  # noqa: E402
from tools.scripts.ops_block_oracle import _pair_seed, playouts  # noqa: E402
from tools.scripts.ops_block_probe import _obs_mask, collect_starts  # noqa: E402


def one_in(start: "ts.GameState", countries: Sequence[int], logits_of: Callable[["ts.GameState"], np.ndarray],
           greedy: Callable[["ts.GameState"], int]) -> Optional[Allocation]:
    """The play whose first point goes to the best-ranked legal country of `countries`, the rest by
    the checkpoint. None if no country there is legal at the start."""
    c0 = _ctx(start)
    card, player = int(c0.pending_op_card), int(c0.decision_player)
    st = start.clone()
    legal = set(int(a) for a in _legal(st))
    cands = [NODE_OFFSET + int(c) for c in countries if NODE_OFFSET + int(c) in legal]
    if not cands:
        return None
    lg = logits_of(st)
    first = max(cands, key=lambda a: lg[a])
    acts = [first]
    ts.Engine.step_flat(st, first, True, False)
    while _in_block(st, card, player):
        a = int(greedy(st))
        acts.append(a)
        ts.Engine.step_flat(st, a, True, False)
    return Allocation(st, tuple(acts), influence(st))


def none_in(start: "ts.GameState", countries: Sequence[int], logits_of: Callable[["ts.GameState"], np.ndarray]
            ) -> Optional[Allocation]:
    """The play that puts no point in `countries`: at every step the checkpoint's best-ranked legal
    action outside them. None if at some step only those countries are legal."""
    c0 = _ctx(start)
    card, player = int(c0.pending_op_card), int(c0.decision_player)
    st = start.clone()
    banned = {NODE_OFFSET + int(c) for c in countries}
    acts: List[int] = []
    while _in_block(st, card, player):
        legal = [int(a) for a in _legal(st) if int(a) not in banned]
        if not legal:
            return None
        lg = logits_of(st)
        a = max(legal, key=lambda x: lg[x])
        acts.append(a)
        ts.Engine.step_flat(st, a, True, False)
    return Allocation(st, tuple(acts), influence(st))


def build(agent: NeuralAgent, starts: Sequence["ts.GameState"], dev: torch.device, per_cell: int,
          seed: int, direction: str = "in") -> List[Dict[str, Any]]:
    model = agent.model
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def logits_of(state: "ts.GameState") -> np.ndarray:
        persp = ts.Player(int(state.ctx().decision_player))
        o, m = _obs_mask([state], persp, dev)
        return model(o, m)[0].float()[0].cpu().numpy()

    def greedy(state: "ts.GameState") -> int:
        return int(np.argmax(logits_of(state)))

    cells: Dict[tuple, List[Dict[str, Any]]] = {}
    order = rng.permutation(len(starts))
    for idx in order:
        st = starts[int(idx)]
        persp = ts.Player(int(st.ctx().decision_player))
        side = 0 if persp == ts.Player.US else 1
        obs = np.asarray(ts.extract_observation(st, persp))
        inf0 = influence(st)
        pol: Optional[Allocation] = None
        for name, (_, card, countries) in REGIONS.items():
            holds = bool(obs[CARD_OFFSET + (card - 1) * CARD_W + MY_HAND] > 0.5)
            key = (name, holds)
            if len(cells.get(key, [])) >= per_cell:
                continue
            if pol is None:
                pol = play_block(st, greedy)
            placed = np.clip(pol.inf_end[side] - inf0[side], 0, None)
            in_region = placed[list(countries)].sum() > 0
            if direction == "in":
                if in_region:
                    continue                          # the policy already places in this region
                forced = one_in(st, countries, logits_of, greedy)
            else:
                if not in_region:
                    continue                          # nothing of the policy's to take out
                forced = none_in(st, countries, logits_of)
            if forced is None:
                continue
            cells.setdefault(key, []).append({"index": int(idx), "region": name, "holds": holds, "side": side,
                                              "turn": int(st.turn), "one_in": forced, "policy": pol})
        if all(len(cells.get((n, h), [])) >= per_cell for n in REGIONS for h in (True, False)):
            break
    return [r for rows in cells.values() for r in rows]


def run(checkpoint: str, n_starts: int, per_cell: int, pairs: int, envs: int, seed: int, dev: torch.device,
        temperature: float, chunk_envs: int, direction: str = "in") -> Dict[str, Any]:
    agent = NeuralAgent.from_checkpoint(checkpoint, device=str(dev))
    model = agent.model.eval()
    t0 = time.time()
    starts = collect_starts(agent, n_starts, envs, seed, dev)
    rows = build(agent, starts, dev, per_cell, seed, direction)
    print(f"{len(rows)} (position, region) cells from {len(starts)} starts in {time.time() - t0:.0f}s", flush=True)
    per = 2 * pairs
    step = max(1, chunk_envs // per)
    for lo in range(0, len(rows), step):
        chunk = rows[lo:lo + step]
        states: List["ts.GameState"] = []
        seeds: List[int] = []
        for j, r in enumerate(chunk):
            for k in range(pairs):
                s = _pair_seed(seed, r["index"] * 16 + list(REGIONS).index(r["region"]), k)
                states += [r["one_in"].end, r["policy"].end]
                seeds += [s, s]
        util = playouts(model, dev, states, seeds, temperature)
        for j, r in enumerate(chunk):
            u = util[j * per:(j + 1) * per].reshape(pairs, 2)
            sign = 1.0 if r["side"] == 0 else -1.0
            score = (sign * u + 1.0) / 2.0
            r["win_one_in"] = float(score[:, 0].mean())
            r["win_policy"] = float(score[:, 1].mean())
    out = [{k: v for k, v in r.items() if k not in ("one_in", "policy")} for r in rows]
    return {"checkpoint": agent.name, "direction": direction, "starts": len(starts), "pairs": pairs,
            "temperature": temperature,
            "seconds": round(time.time() - t0, 1), "rows": out}


def summarise(res: Dict[str, Any]) -> Dict[str, Any]:
    rows = res["rows"]
    out: Dict[str, Any] = {}

    def stats(sel: List[Dict[str, Any]]) -> Dict[str, float]:
        d = np.array([r["win_one_in"] - r["win_policy"] for r in sel])
        if len(d) < 2:
            return {"n": len(d)}
        return {"n": len(d), "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}

    for name in list(REGIONS) + ["all"]:
        h = [r for r in rows if r["holds"] and (name == "all" or r["region"] == name)]
        n = [r for r in rows if not r["holds"] and (name == "all" or r["region"] == name)]
        sh, sn = stats(h), stats(n)
        cell: Dict[str, Any] = {"holds": sh, "not": sn}
        if "mean" in sh and "mean" in sn:
            cell["interaction"] = {"mean": sh["mean"] - sn["mean"], "se": float(np.hypot(sh["se"], sn["se"]))}
        out[name] = cell
    return out


def render(res: Dict[str, Any], summ: Dict[str, Any]) -> str:
    what = ("positions where the policy placed nothing in the region; the forced branch puts the first point "
            "there" if res["direction"] == "in" else
            "positions where the policy placed in the region; the forced branch places every point elsewhere")
    lab = "one in" if res["direction"] == "in" else "none in"
    L = [f"# Scoring-card region oracle ({res['direction']}): {res['checkpoint']}", "",
         f"{res['starts']} influence-play starts; {what}. "
         f"{res['pairs']} paired playouts per branch, temperature {res['temperature']}.", "",
         f"*{lab} − policy*: the mover's win-rate change in percentage points. *interaction*: holding minus not holding.", "",
         f"| region | holds: n | holds: {lab} − policy | not: n | not: {lab} − policy | **interaction** |",
         "|:---|---:|---:|---:|---:|---:|"]

    def f(s: Dict[str, Any]) -> str:
        return f"{100 * s['mean']:+.1f} ± {100 * s['se']:.1f}" if "mean" in s else "—"

    for name, c in summ.items():
        inter = c.get("interaction")
        L.append(f"| {name} | {c['holds']['n']} | {f(c['holds'])} | {c['not']['n']} | {f(c['not'])} | "
                 f"{f(inter) if inter else '—'} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--starts", type=int, default=3000)
    ap.add_argument("--per-cell", type=int, default=100, help="positions per (region, holds) cell")
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--chunk-envs", type=int, default=2048)
    ap.add_argument("--direction", choices=("in", "out"), default="in",
                    help="in: force a point into a region the policy skipped; out: keep every point out of a "
                         "region the policy placed in")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    _regions()
    res = run(a.checkpoint, a.starts, a.per_cell, a.pairs, a.envs, a.seed, torch.device(a.device),
              a.temperature, a.chunk_envs, a.direction)
    summ = summarise(res)
    res["summary"] = summ
    md = render(res, summ)
    print(md)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(res, f, indent=1)
    if a.output_md:
        with open(a.output_md, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
