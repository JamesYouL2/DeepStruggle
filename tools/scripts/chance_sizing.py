#!/usr/bin/env python3
"""How much of the critic's step-to-step error is chance that P2 could average away? (P2 sizing.)

The critic's target and the policy's advantage are built from one-step errors
delta_t = V(s_{t+1}) - V(s_t) (plus the reward at the end), chained with lambda. When chance sits
between a decision and the next one -- a die roll, or the deal at a turn boundary -- s_{t+1} is one
draw of it, and delta_t carries that draw's luck. P2 replaces the drawn outcome's value with the
average over outcomes, which removes exactly Var_chance[V(s_{t+1})] from delta_t's variance
(`research/plans/P2_chance_aware_targets.md`).

From a checkpoint's own self-play at temperature 1 (the training distribution), decisions are
sampled. For each: the chosen action is applied to a copy, and the chance that follows is resolved
`--draws` times with fresh engine seeds, settling exactly as the training environment does
(`SettleMode.FORCED`). V is read from the next position's mover and sign-converted to the deciding
player's side, as the training bootstrap does. Reported:

* how often a decision is followed by chance at all, split into dice (same turn) and deal (the
  turn advances);
* E[Var_chance V(s_{t+1})], the luck P2 removes, against Var(delta_t), the whole one-step error;
  their ratio is the share of one-step variance P2 could remove, overall and split.

    PYTHONPATH=.:build/release python tools/scripts/chance_sizing.py --checkpoint <snap.pt> \\
        --games 2000 --record-prob 0.05 --draws 16 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from bindings.settle import SettleMode, settle  # noqa: E402
from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402


@torch.no_grad()
def values_for(model: torch.nn.Module, dev: torch.device, states: Sequence["ts.GameState"],
               persp: Sequence[int]) -> np.ndarray:
    """V of each state from `persp`'s side (+1 US / -1 USSR): read from the state's own mover and
    negated when that is the other player; a terminal state is its utility."""
    out = np.zeros(len(states), dtype=np.float64)
    live: List[int] = []
    for i, st in enumerate(states):
        if ts.Engine.is_terminal(st):
            out[i] = float(ts.Engine.get_terminal_utility(st)) * persp[i]
        else:
            live.append(i)
    for s in range(0, len(live), 512):
        idx = live[s:s + 512]
        movers = [int(states[i].ctx().decision_player) for i in idx]
        obs = np.stack([np.asarray(ts.extract_observation(states[i], ts.Player(mv))) for i, mv in zip(idx, movers)])
        masks = np.stack([np.asarray(ts.Engine.get_flat_action_mask(states[i], False)) for i in idx])
        masks[masks.sum(1) == 0] = 1
        v = model(torch.from_numpy(obs).float().to(dev), torch.from_numpy(masks).to(dev))[1]
        v = v.float().reshape(-1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = v[j] if movers[j] == persp[i] else -v[j]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--record-prob", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--output-json", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    agent = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev))
    model = agent.model.eval()
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    env = TsVectorizedEnv(num_envs=a.envs, base_seed=a.seed)
    obs, masks, _ = env.reset_all()
    recs: List[Dict[str, Any]] = []
    done = 0
    with torch.no_grad():
        while done < a.games:
            o = torch.from_numpy(np.asarray(obs)).float().to(dev)
            m = torch.from_numpy(np.asarray(masks)).to(dev)
            logits, v_now, _ = model(o, m)
            acts = torch.multinomial(torch.softmax(logits.float(), -1), 1, generator=gen).squeeze(-1).cpu().numpy()
            vn = v_now.float().reshape(-1).cpu().numpy()
            dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
            for i in np.flatnonzero((rng.random(a.envs) < a.record_prob) & (dp != 0)):
                st = env.runner.get_state(int(i)).clone()
                base = st.clone()
                ts.Engine.step_flat(base, int(acts[i]), False)
                draws = []
                for _ in range(a.draws):
                    c = base.clone()
                    c.rng_state = int(rng.integers(0, 2**63 - 1))
                    settle(c, SettleMode.FORCED)
                    draws.append(c)
                p = int(dp[i])
                vals = values_for(model, dev, draws, [p] * len(draws))
                recs.append({"v_now": float(vn[i]), "vals": vals, "turn": int(st.turn),
                             "deal": any(int(c.turn) > int(st.turn) for c in draws),
                             "chance": bool(np.ptp(vals) > 1e-6)})
            obs, masks, _r, d, _info = env.step(acts)
            done += int(np.asarray(d).sum())
    v_now = np.array([r["v_now"] for r in recs])
    vals = np.stack([r["vals"] for r in recs])                      # (n, draws)
    var_chance = vals.var(axis=1, ddof=1)                           # the luck P2 removes, per step
    delta = vals - v_now[:, None]                                   # one-step error, per draw
    deal = np.array([r["deal"] for r in recs])
    chance = np.array([r["chance"] for r in recs])
    dice = chance & ~deal
    turn = np.array([r["turn"] for r in recs])
    tot = float(delta.var())                                        # Var(delta) over steps and draws
    lines = [f"{agent.name}: {len(recs)} sampled decisions from {done} self-play games at temperature 1, "
             f"{a.draws} chance draws each.",
             "",
             f"* followed by chance at all: {100 * chance.mean():.1f}% of decisions "
             f"(dice within the turn {100 * dice.mean():.1f}%, a deal {100 * deal.mean():.1f}%)",
             f"* Var(delta), the whole one-step error: {tot:.4f}",
             "",
             "| removed by P2 | E[Var_chance V(s')] over all decisions | share of Var(delta) |",
             "|:---|---:|---:|"]
    for label, sel in (("dice", dice), ("deal", deal), ("both", chance)):
        part = float((var_chance * sel).mean())
        lines.append(f"| {label} | {part:.4f} | **{100 * part / tot:.1f}%** |")
    lines += ["", "| turn | decisions | chance share | Var(delta) | P2 share |", "|:---|---:|---:|---:|---:|"]
    for lo, hi in ((1, 1), (2, 3), (4, 5), (6, 7), (8, 10)):
        sel = (turn >= lo) & (turn <= hi)
        if sel.sum() < 100:
            continue
        t = float(delta[sel].var())
        lines.append(f"| {lo}–{hi} | {int(sel.sum())} | {100 * chance[sel].mean():.1f}% | {t:.4f} | "
                     f"{100 * float((var_chance * chance)[sel].mean()) / t:.1f}% |")
    print("\n".join(lines))
    if a.output_md:
        open(a.output_md, "w").write("\n".join(lines) + "\n")
    if a.output_json:
        json.dump({"n": len(recs), "var_delta": tot,
                   "share": {k: float((var_chance * s).mean()) / tot for k, s in (("dice", dice), ("deal", deal), ("both", chance))},
                   "frac": {"chance": float(chance.mean()), "dice": float(dice.mean()), "deal": float(deal.mean())}},
                  open(a.output_json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
