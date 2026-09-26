#!/usr/bin/env python3
"""P27: how a checkpoint spends an influence ops play, against the exact set of alternatives.

For each checkpoint: positions where an influence play starts are collected from its own
self-play (temperature 0.1, both seats), each play is solved exactly (`ai.eval.ops_block`), and
the policy's greedy allocation is compared with every alternative -- by the rules (contested
battleground taken or missed, uncontested instead, reinforcing points) and by the network's own
critic (the value of its allocation against the critic's favourite).

    PYTHONPATH=.:build/release python tools/scripts/ops_block_probe.py \\
        --checkpoints a.pt b.pt --positions 300 --output-json out.json --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from ai.eval.ops_block import (BlockReport, analyse, enumerate_block, is_block_start,  # noqa: E402
                               play_block)
from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402


def _obs_mask(states: Sequence["ts.GameState"], persp: "ts.Player", dev: torch.device):
    obs = np.stack([np.asarray(ts.extract_observation(s, persp)) for s in states])
    masks = np.stack([np.asarray(ts.Engine.get_flat_action_mask(s, False)) for s in states])
    masks[masks.sum(1) == 0] = 1             # terminal states: any mask; only the value is read
    return (torch.from_numpy(obs).float().to(dev), torch.from_numpy(masks).to(dev))


@torch.no_grad()
def critic_values(model: torch.nn.Module, states: Sequence["ts.GameState"], persp: "ts.Player",
                  dev: torch.device, chunk: int = 2048) -> np.ndarray:
    """v_win of each state from `persp`'s side (the observation's perspective)."""
    out: List[np.ndarray] = []
    for i in range(0, len(states), chunk):
        o, m = _obs_mask(states[i:i + chunk], persp, dev)
        _, v, _ = model(o, m)
        out.append(v.float().squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def collect_starts(agent: NeuralAgent, n: int, envs: int, seed: int, dev: torch.device,
                   max_steps: int = 4000) -> List["ts.GameState"]:
    """Positions where an influence play starts, from the checkpoint's own self-play at 0.1."""
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    obs, masks, info = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(seed)
    out: List["ts.GameState"] = []
    for _ in range(max_steps):
        for i in range(envs):
            st = env.runner.get_state(i)
            if is_block_start(st):
                out.append(st.clone())
        if len(out) >= n:
            break
        with torch.no_grad():
            o = torch.from_numpy(np.asarray(obs)).float().to(dev)
            m = torch.from_numpy(np.asarray(masks)).to(dev)
            logits = agent.model(o, m)[0].float()
            probs = torch.softmax(logits / 0.1, dim=-1)
            acts = torch.multinomial(probs, 1, generator=gen).squeeze(-1).cpu().numpy()
        obs, masks, _r, _d, info = env.step(acts)
    return out[:n]


def probe_checkpoint(path: str, positions: int, envs: int, seed: int, dev: torch.device,
                     max_allocations: int) -> Dict[str, Any]:
    agent = NeuralAgent.from_checkpoint(path, device=str(dev))
    if getattr(agent, "merged_influence", False):
        raise ValueError(f"{path} decides in the merged (E4.1) view; the probe is E4 only")
    model = agent.model.eval()
    starts = collect_starts(agent, positions, envs, seed, dev)

    @torch.no_grad()
    def greedy(state: "ts.GameState") -> int:
        persp = ts.Player(int(state.ctx().decision_player))
        o, m = _obs_mask([state], persp, dev)
        return int(model(o, m)[0].float().argmax(-1).item())

    reports: List[BlockReport] = []
    skipped = 0
    t0 = time.time()
    for st in starts:
        try:
            allocs = enumerate_block(st, max_allocations=max_allocations)
        except RuntimeError:
            skipped += 1
            continue
        pol = play_block(st, greedy)
        persp = ts.Player(int(st.ctx().decision_player))
        vals = critic_values(model, [a.end for a in allocs] + [pol.end], persp, dev)
        reports.append(analyse(st, allocs, pol, values=vals[:-1], policy_value=float(vals[-1])))
    return {"checkpoint": agent.name, "positions": len(reports), "skipped_too_large": skipped,
            "seconds": round(time.time() - t0, 1), "reports": [asdict(r) for r in reports]}


def summarise(res: Dict[str, Any]) -> Dict[str, float]:
    rs = res["reports"]
    if not rs:
        return {}
    take = [r for r in rs if r["takeable_contested"]]
    missed = [r for r in take if not r["policy_takes_contested"]]
    pts = sum(r["policy_points"] for r in rs)
    s: Dict[str, float] = {
        "positions": float(len(rs)),
        "mean ops": float(np.mean([r["ops"] for r in rs])),
        "contested battleground takeable": len(take) / len(rs),
        "policy takes it | takeable": (len(take) - len(missed)) / max(1, len(take)),
        "missed | takeable": len(missed) / max(1, len(take)),
        "gains control of something else | missed": (
            sum(r["policy_gains_uncontested"] for r in missed) / max(1, len(missed))),
        "reinforcing points / points placed": sum(r["policy_reinforce_points"] for r in rs) / max(1, pts),
        "battleground shortfall (max - policy), mean": float(np.mean(
            [r["max_bg_gain"] - r["policy_bg_gain"] for r in rs])),
        "policy is the critic's favourite": float(np.mean([r["critic_rank_of_policy"] == 0 for r in rs])),
        "critic regret (best - policy), mean": float(np.mean(
            [r["critic_best"] - r["critic_policy"] for r in rs])),
        "critic-best takes it | takeable": sum(r["critic_best_takes_contested"] for r in take) / max(1, len(take)),
    }
    # what the policy does instead, in the missed positions: outcomes (multi-label) and where
    # the points went (share of the points it placed)
    if missed:
        for k in missed[0]["outcomes"]:
            s[f"missed -> {k}"] = sum(r["outcomes"][k] for r in missed) / len(missed)
        mp = sum(sum(r["points_by_class"].values()) for r in missed)
        for k in missed[0]["points_by_class"]:
            s[f"missed: points to {k}"] = sum(r["points_by_class"][k] for r in missed) / max(1, mp)
    # the diagnosis: in takeable positions, who takes it
    for cb in (True, False):
        for pt in (True, False):
            k = f"takeable: critic-best {'takes' if cb else 'skips'}, policy {'takes' if pt else 'skips'}"
            s[k] = sum(1 for r in take if r["critic_best_takes_contested"] == cb
                       and r["policy_takes_contested"] == pt) / max(1, len(take))
    return s


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-allocations", type=int, default=60_000)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    results = [probe_checkpoint(p, a.positions, a.envs, a.seed, dev, a.max_allocations)
               for p in a.checkpoints]
    sums = {r["checkpoint"]: summarise(r) for r in results}
    names = list(sums)
    keys = list(next(iter(sums.values())).keys()) if sums else []
    lines = ["| measure | " + " | ".join(names) + " |", "|:---|" + "---:|" * len(names)]
    for k in keys:
        cells = []
        for n in names:
            v = sums[n].get(k, float("nan"))
            cells.append(f"{v:.0f}" if k == "positions" else
                         f"{v:.2f}" if (k == "mean ops" or k.endswith(", mean")) else f"{100 * v:.1f}%")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    if a.output_md:
        with open(a.output_md, "w", encoding="utf-8") as f:
            f.write(table + "\n")
    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": sums, "results": results}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
