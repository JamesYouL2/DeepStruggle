#!/usr/bin/env python3
"""The realignment oracle: where the checkpoint could realign but does not, would realigning win more?

The play-mode census (`play_mode_census.py`) found the policy realigns at the human rate at DEFCON 2
and about 30x less than humans at DEFCON 3+. A rate does not say who is right. This asks the game.

Positions are the checkpoint's own play-mode nodes (the choice between event, space, influence,
coup and realignment), from its self-play at temperature 0.1, where realignment is legal and the
greedy policy picks something else. From each, two branches:

* **policy** -- the node as it stands; the playout's greedy policy makes its own choice;
* **realign** -- realignment is forced at this node, and from there the checkpoint plays on as it
  would, choosing the targets of every roll itself. This measures "realign here, the way you would
  realign", not "realign optimally", so a gain is a lower bound on what realignment is worth.

Each branch is played to the end `--pairs` times by the checkpoint on both sides, greedily. Pair k
of a position re-seeds the engine RNG identically in both branches, so the difference is the mode's.
Positions are stratified by DEFCON (2 against 3+), and results are reported by DEFCON, side, the
policy's own P(realign) and the mode it chose instead.

    PYTHONPATH=.:build/release python tools/scripts/realign_oracle.py \\
        --checkpoint <snapshot.pt> --positions-per-class 400 --pairs 16 --output-md out.md
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

from bindings.settle import SettleMode, settle  # noqa: E402
from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.scripts.ops_block_oracle import _pair_seed, playouts  # noqa: E402

MODE_BASE = 110
MODES = ("event", "space", "influence", "coup", "realign")
REALIGN = MODE_BASE + 4


def collect(agent: NeuralAgent, per_class: int, envs: int, seed: int, dev: torch.device,
            accept: float = 0.08, max_steps: int = 200000) -> List[Dict[str, Any]]:
    """Play-mode nodes where realignment is legal and the greedy policy declines it, up to
    `per_class` at DEFCON 2 and as many at DEFCON 3+, from self-play at temperature 0.1.

    Each candidate is kept with probability `accept`, so the sample spreads over many games and
    every stage of them instead of filling up from the opening of the first batch."""
    rng = np.random.default_rng(seed)
    model = agent.model
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    obs, masks, _info = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(seed)
    got: Dict[str, List[Dict[str, Any]]] = {"defcon2": [], "defcon3+": []}
    for _ in range(max_steps):
        m_np = np.asarray(masks)
        with torch.no_grad():
            o = torch.from_numpy(np.asarray(obs)).float().to(dev)
            m = torch.from_numpy(m_np).to(dev)
            logits = model(o, m)[0].float()
            probs = torch.softmax(logits, -1)
            greedy = logits.argmax(-1).cpu().numpy()
            acts = torch.multinomial(torch.softmax(logits / 0.1, -1), 1, generator=gen).squeeze(-1).cpu().numpy()
        legal = m_np[:, MODE_BASE:MODE_BASE + len(MODES)].astype(bool)
        cand = np.flatnonzero(legal[:, 4] & (legal.sum(1) >= 2) & (greedy != REALIGN)
                              & (greedy >= MODE_BASE) & (greedy < MODE_BASE + len(MODES)))
        if len(cand):
            p_re = probs[torch.from_numpy(cand).to(dev), REALIGN].cpu().numpy()
            for j, i in enumerate(cand):
                st = env.runner.get_state(int(i))
                cls = "defcon2" if int(st.defcon) == 2 else "defcon3+"
                if len(got[cls]) >= per_class or rng.random() >= accept:
                    continue
                got[cls].append({"state": st.clone(), "defcon": int(st.defcon), "turn": int(st.turn),
                                 "side": 0 if int(st.ctx().decision_player) == int(ts.Player.US) else 1,
                                 "p_realign": float(p_re[j]), "chosen": MODES[int(greedy[i]) - MODE_BASE]})
        if all(len(v) >= per_class for v in got.values()):
            break
        obs, masks, _r, _d, _info = env.step(acts)
    return got["defcon2"] + got["defcon3+"]


def run(checkpoint: str, per_class: int, pairs: int, envs: int, seed: int, dev: torch.device,
        chunk_envs: int, accept: float = 0.08) -> Dict[str, Any]:
    agent = NeuralAgent.from_checkpoint(checkpoint, device=str(dev))
    model = agent.model.eval()
    t0 = time.time()
    nodes = collect(agent, per_class, envs, seed, dev, accept)
    t_collect = time.time() - t0
    for n in nodes:
        forced = n["state"].clone()
        ts.Engine.step_flat(forced, REALIGN, False)
        settle(forced, SettleMode.CHANCE)
        n["forced"] = forced
    per = 2 * pairs
    step = max(1, chunk_envs // per)
    for lo in range(0, len(nodes), step):
        chunk = nodes[lo:lo + step]
        states: List["ts.GameState"] = []
        seeds: List[int] = []
        for idx, n in enumerate(chunk):
            for k in range(pairs):
                s = _pair_seed(seed, lo + idx, k)
                states += [n["forced"], n["state"]]
                seeds += [s, s]
        util = playouts(model, dev, states, seeds, 0.0)
        for j, n in enumerate(chunk):
            u = util[j * per:(j + 1) * per].reshape(pairs, 2)
            score = ((1.0 if n["side"] == 0 else -1.0) * u + 1.0) / 2.0
            n["win_realign"] = float(score[:, 0].mean())
            n["win_policy"] = float(score[:, 1].mean())
    rows = [{k: v for k, v in n.items() if k not in ("state", "forced")} for n in nodes]
    return {"checkpoint": agent.name, "positions": len(rows), "pairs": pairs,
            "seconds": round(time.time() - t0, 1), "seconds_collecting": round(t_collect, 1), "rows": rows}


def summarise(res: Dict[str, Any]) -> List[str]:
    rows = res["rows"]
    if not rows:
        return ["no positions"]
    d = np.array([r["win_realign"] - r["win_policy"] for r in rows])

    def line(label: str, sel: np.ndarray) -> str:
        x = d[sel]
        if len(x) < 2:
            return f"| {label} | {len(x)} | — | — | — | — | — |"
        idx = np.flatnonzero(sel)
        wr = np.mean([rows[i]["win_realign"] for i in idx])
        wp = np.mean([rows[i]["win_policy"] for i in idx])
        se = x.std(ddof=1) / np.sqrt(len(x))
        return (f"| {label} | {len(x)} | {100 * wr:.1f}% | {100 * wp:.1f}% | **{100 * x.mean():+.1f}** ± {100 * se:.1f} "
                f"| {100 * np.mean(x >= 0.25):.1f}% | {100 * np.mean(x <= -0.25):.1f}% |")

    dc = np.array([r["defcon"] for r in rows])
    side = np.array([r["side"] for r in rows])
    p = np.array([r["p_realign"] for r in rows])
    chosen = np.array([r["chosen"] for r in rows])
    out = [f"{res['checkpoint']}: {res['positions']} play-mode nodes where realignment is legal and the greedy "
           f"policy declines it; {res['pairs']} paired greedy playouts per branch",
           "",
           "| positions | n | realign | policy | realign − policy (pp) | realign ≥ 25 pp better | ≥ 25 pp worse |",
           "|:---|---:|---:|---:|---:|---:|---:|"]
    out.append(line("all", np.ones(len(rows), dtype=bool)))
    for label, sel in (("DEFCON 2", dc == 2), ("DEFCON 3+", dc >= 3)):
        out.append(line(label, sel))
        out.append(line(f"{label}, mover US", sel & (side == 0)))
        out.append(line(f"{label}, mover USSR", sel & (side == 1)))
    turn = np.array([r["turn"] for r in rows])
    for lo, hi in ((1, 3), (4, 7), (8, 10)):
        out.append(line(f"turns {lo}–{hi}", (turn >= lo) & (turn <= hi)))
    for lo, hi in ((0.0, 0.01), (0.01, 0.05), (0.05, 1.0)):
        out.append(line(f"P(realign) in [{lo:.0%}, {hi:.0%})", (p >= lo) & (p < hi)))
    for m in MODES[:4]:
        out.append(line(f"policy chose {m}", chosen == m))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions-per-class", type=int, default=400, help="per DEFCON class (2, 3+)")
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--accept", type=float, default=0.08,
                    help="Probability of keeping each candidate node, to spread the sample over games.")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-envs", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    res = run(a.checkpoint, a.positions_per_class, a.pairs, a.envs, a.seed, dev, a.chunk_envs, a.accept)
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
