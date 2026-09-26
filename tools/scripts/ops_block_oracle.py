#!/usr/bin/env python3
"""P27 stage 2, the rollout oracle: does taking the contested battleground actually win more?

Positions are the starts of influence ops plays from the checkpoint's own self-play (as in the
stage-1 probe) where a contested battleground is takeable and the checkpoint's greedy play takes
none. From each, two branches:

* **policy** -- the checkpoint's own greedy allocation;
* **take** -- points go to a contested battleground until the mover controls it, then the
  checkpoint places the rest (`ai.eval.ops_block.force_take`). Of the contested battlegrounds
  that can be taken, the one the checkpoint ranks highest at the first point: this measures
  "take one", not "take the best one".

Each branch is played to the end `--pairs` times by the checkpoint on both sides. Pair k of a
position re-seeds the engine RNG identically in both branches, so dice and draws start alike and
the difference is the placement's. Reported: the mover's paired win-rate difference, overall, per
seat and by what the policy did instead, beside the checkpoint's own critic on the two end boards
-- the question stage 1 left open, whether the critic or the rule is wrong.

    PYTHONPATH=.:build/release python tools/scripts/ops_block_oracle.py \\
        --checkpoint <snapshot.pt> --positions 600 --pairs 16 --output-json out.json --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from ai.eval.ops_block import (NODE_OFFSET, Allocation, analyse, contested_battlegrounds,  # noqa: E402
                               controlled, country_table, force_take, influence, play_block)
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.scripts.ops_block_probe import _obs_mask, collect_starts, critic_values  # noqa: E402

_SEED_MIX = 0x9E3779B97F4A7C15


def _pair_seed(base: int, position: int, k: int) -> int:
    return (base * 1_000_003 + position * 1_009 + k + 1) * _SEED_MIX % (1 << 64)


@torch.no_grad()
def playouts(model: torch.nn.Module, dev: torch.device, states: Sequence["ts.GameState"],
             seeds: Sequence[int], temperature: float, max_steps: int = 6000) -> np.ndarray:
    """Play every state to the end, the checkpoint on both sides; the result from the US side
    (+1 win, -1 loss, 0 draw). Each state's engine RNG is replaced by its seed first."""
    n = len(states)
    runner = ts.VectorizedBatchRunner(n, 0)
    for i, (s, seed) in enumerate(zip(states, seeds)):
        st = s.clone()
        st.rng_state = int(seed)
        runner.set_state(i, st)
    runner.refresh_all()
    util = np.zeros(n, dtype=np.float32)
    active = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        terms = np.asarray(runner.get_terminals(), dtype=bool)
        done = active & terms
        for i in np.flatnonzero(done):
            util[i] = float(ts.Engine.get_terminal_utility(runner.get_state(int(i))))
        active &= ~terms
        if not active.any():
            return util
        obs = torch.from_numpy(np.asarray(runner.get_observations())).float().to(dev)
        masks_np = np.asarray(runner.get_action_masks())
        masks = torch.from_numpy(masks_np).to(dev)
        logits = model(obs, masks)[0].float()
        if temperature <= 0.0:
            acts = logits.argmax(-1)
        else:
            acts = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(-1)
        acts_np = acts.cpu().numpy().astype(np.int64)
        # finished games still need a legal index to be stepped past
        for i in np.flatnonzero(~active):
            acts_np[i] = int(np.flatnonzero(masks_np[i])[0]) if masks_np[i].any() else 0
        res = runner.step_flat_all(acts_np.tolist(), auto_advance=True)
        if 0 in res:
            bad = [i for i, r in enumerate(res) if r == 0 and active[i]]
            if bad:
                raise RuntimeError(f"engine refused an action in {len(bad)} active playouts")
    raise RuntimeError(f"{int(active.sum())} playouts did not finish in {max_steps} steps")


def build_branches(agent: NeuralAgent, starts: Sequence["ts.GameState"], dev: torch.device
                   ) -> List[Dict[str, Any]]:
    """The missed positions and their two branches."""
    model = agent.model
    _, _, names = country_table()

    @torch.no_grad()
    def logits_of(state: "ts.GameState") -> np.ndarray:
        persp = ts.Player(int(state.ctx().decision_player))
        o, m = _obs_mask([state], persp, dev)
        return model(o, m)[0].float()[0].cpu().numpy()

    def greedy(state: "ts.GameState") -> int:
        return int(np.argmax(logits_of(state)))

    out: List[Dict[str, Any]] = []
    for idx, st in enumerate(starts):
        side = 0 if int(st.ctx().decision_player) == int(ts.Player.US) else 1
        cbg = contested_battlegrounds(st)
        if not cbg.any():
            continue
        pol = play_block(st, greedy)
        gained = controlled(pol.inf_end, side) & ~controlled(influence(st), side)
        if (gained & cbg).any():
            continue                                  # the policy takes one: not a miss
        lg = logits_of(st)
        order = sorted(np.flatnonzero(cbg), key=lambda c: -lg[NODE_OFFSET + int(c)])
        forced: Optional[Allocation] = None
        country = -1
        for c in order:
            forced = force_take(st, int(c), greedy)
            if forced is not None:
                country = int(c)
                break
        if forced is None:
            continue                                  # no contested battleground is takeable
        rep = analyse(st, [forced], pol)
        out.append({"index": idx, "start": st, "side": side, "ops": rep.ops, "country": country,
                    "country_name": names[country], "take": forced, "policy": pol,
                    "outcomes": rep.outcomes, "points_by_class": rep.points_by_class})
    return out


def run(checkpoint: str, positions: int, pairs: int, envs: int, seed: int, dev: torch.device,
        temperature: float, chunk_envs: int) -> Dict[str, Any]:
    agent = NeuralAgent.from_checkpoint(checkpoint, device=str(dev))
    if getattr(agent, "merged_influence", False):
        raise ValueError(f"{checkpoint} decides in the merged (E4.1) view; the oracle is E4 only")
    model = agent.model.eval()
    t0 = time.time()
    starts = collect_starts(agent, positions, envs, seed, dev)
    branches = build_branches(agent, starts, dev)
    t_build = time.time() - t0

    # critic on the two end boards, from the mover's side
    for b in branches:
        persp = ts.Player.US if b["side"] == 0 else ts.Player.USSR
        v = critic_values(model, [b["take"].end, b["policy"].end], persp, dev)
        b["v_take"], b["v_policy"] = float(v[0]), float(v[1])

    # playouts, chunked: every position contributes 2 x pairs games
    per = 2 * pairs
    step = max(1, chunk_envs // per)
    for lo in range(0, len(branches), step):
        chunk = branches[lo:lo + step]
        states: List["ts.GameState"] = []
        seeds: List[int] = []
        for b in chunk:
            for k in range(pairs):
                s = _pair_seed(seed, b["index"], k)
                states += [b["take"].end, b["policy"].end]
                seeds += [s, s]
        util = playouts(model, dev, states, seeds, temperature)
        for j, b in enumerate(chunk):
            u = util[j * per:(j + 1) * per].reshape(pairs, 2)
            sign = 1.0 if b["side"] == 0 else -1.0
            score = (sign * u + 1.0) / 2.0               # mover: 1 win, 0.5 draw, 0 loss
            b["win_take"] = float(score[:, 0].mean())
            b["win_policy"] = float(score[:, 1].mean())
    rows = [{k: v for k, v in b.items() if k not in ("start", "take", "policy")} for b in branches]
    return {"checkpoint": agent.name, "starts": len(starts), "positions": len(rows),
            "pairs": pairs, "temperature": temperature, "seconds": round(time.time() - t0, 1),
            "seconds_building": round(t_build, 1), "rows": rows}


def summarise(res: Dict[str, Any]) -> List[str]:
    rows = res["rows"]
    if not rows:
        return ["no missed positions"]
    d = np.array([r["win_take"] - r["win_policy"] for r in rows])
    dv = np.array([r["v_take"] - r["v_policy"] for r in rows])

    def line(label: str, sel: np.ndarray) -> str:
        x = d[sel]
        if len(x) < 2:
            return f"| {label} | {len(x)} | — | — | — | — |"
        wt = np.mean([rows[i]["win_take"] for i in np.flatnonzero(sel)])
        wp = np.mean([rows[i]["win_policy"] for i in np.flatnonzero(sel)])
        se = x.std(ddof=1) / np.sqrt(len(x))
        return (f"| {label} | {len(x)} | {100 * wt:.1f}% | {100 * wp:.1f}% | "
                f"**{100 * x.mean():+.1f}** ± {100 * se:.1f} | {100 * np.mean(dv[sel] > 0):.0f}% |")

    out = [f"{res['checkpoint']}: {res['positions']} missed positions of {res['starts']} influence-play "
           f"starts, {res['pairs']} paired playouts each, temperature {res['temperature']}",
           "",
           "| positions | n | take | policy | take − policy (pp) | critic prefers take |",
           "|:---|---:|---:|---:|---:|---:|"]
    everything = np.ones(len(rows), dtype=bool)
    out.append(line("all", everything))
    side = np.array([r["side"] for r in rows])
    out.append(line("mover US", side == 0))
    out.append(line("mover USSR", side == 1))
    for k in rows[0]["outcomes"]:
        if k == "takes a contested battleground":
            continue
        sel = np.array([bool(r["outcomes"][k]) for r in rows])
        out.append(line(f"policy {k}", sel))
    ops = np.array([r["ops"] for r in rows])
    for o in sorted(set(ops.tolist())):
        out.append(line(f"{o} Ops", ops == o))
    # does the critic's sign agree with the oracle's, where the oracle is clear?
    clear = np.abs(d) >= 0.25
    if clear.any():
        agree = np.mean(np.sign(d[clear]) == np.sign(dv[clear]))
        out += ["", f"Where the oracle's difference is at least 25 points ({int(clear.sum())} positions), "
                    f"the critic's preference has the same sign in {100 * agree:.0f}%."]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=1500,
                    help="Influence-play starts to collect; about 40%% are missed-and-takeable.")
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Playout temperature for both sides; 0 is greedy, as tournaments rate.")
    ap.add_argument("--chunk-envs", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    res = run(a.checkpoint, a.positions, a.pairs, a.envs, a.seed, torch.device(a.device),
              a.temperature, a.chunk_envs)
    lines = summarise(res)
    print("\n".join(lines))
    if a.output_md:
        with open(a.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
