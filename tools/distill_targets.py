#!/usr/bin/env python3
"""P36c's targets: the keyed rollout root's improved policy on the network's own entry decisions.

Greedy self-play of the network; every card choice (`SELECT_CARD`, the entry decisions: action-round and
headline cards) with two or more legal moves is a position, and a uniform subset of `--positions` of them
is kept, each tagged with its game so the held-out set is whole games (`--holdout`: one game in ten by
default). At each world count of `--worlds` the root (k 4, stop=earlier, leaf=own, keyed chance -- the
P36b teacher) searches every position once, and the target is R32's improved policy over the legal moves,
softmax(logit + sigma(completed Q)) (`ai.search.rollout_root.RolloutTeacher`), with the root's own z2 pick.
The held-out positions are searched a second time with another seed: the teacher's self-agreement on
exactly those positions, the ceiling any student's held-out agreement is read against (P36c's reading 3).

Output: one `.npz` (`--output`): `states` (N x state_size raw GameStates), `game`, `heldout`, `net_argmax`,
and per world count W: `acts_W`, `probs_W` (N x 16, padded with -1 / 0), `pick_W`, and for the held-out
rows `acts2_W`, `probs2_W`, `pick2_W` (the second seed; -1 elsewhere). A sidecar `.meta.json` records the
checkpoint, the configuration and the commit.

    tools/distill_targets.py --checkpoint swa.pt --positions 200000 --worlds 16 64 128 \\
        --output /workspace/data/datasets/p36c/targets.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ts_engine as ts  # noqa: E402
from ai.search.rollout_root import RolloutConfig, RolloutTeacher  # noqa: E402
from bindings.ts_env import model_obs_features  # noqa: E402
from tools.lib.engine_fingerprint import fingerprint  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402

WIDTH = 16


def collect(model: Any, dev: torch.device, games: int, seed: int) -> Tuple[List[ts.GameState], List[int], List[int]]:
    """Every card choice with two or more legal moves in `games` greedy games: states, games, argmaxes."""
    feats = model_obs_features(model)
    runner = ts.VectorizedBatchRunner(games, seed)
    if feats:
        runner.set_obs_features([feats] * games, [feats] * games)
    done = np.zeros(games, dtype=bool)
    states: List[ts.GameState] = []
    game: List[int] = []
    argmax: List[int] = []
    for _ in range(6000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32)).to(dev)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            lg = model(obs, torch.from_numpy(masks).to(dev))[0].float().cpu().numpy()
        acts = np.where(masks > 0, lg, -np.inf).argmax(axis=1)
        acts[done] = 0
        for i in np.flatnonzero(~done & (masks.sum(axis=1) >= 2)):
            st = runner.get_state(int(i))
            if int(st.ctx().decision_type) == int(ts.DecisionType.SELECT_CARD):
                states.append(st)
                game.append(int(i))
                argmax.append(int(acts[i]))
        runner.step_flat_all(acts.tolist(), True)
    return states, game, argmax


def z2_pick(stats: Dict[str, Any]) -> int:
    """The root's own decision from its statistics: the best mean on a lead of 2 SE over the network's move."""
    cands = stats["candidates"]
    if not cands:
        return -1
    base = int(cands[0])
    if stats["worlds"] <= 0:
        return base
    best = max(cands, key=lambda a: stats["q"][a])
    m, s = stats["lead"][best]
    return int(best) if s > 0 and m / s >= 2.0 and best != base else base


def search(teacher: RolloutTeacher, states: List[ts.GameState], chunk: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(states)
    acts = np.full((n, WIDTH), -1, dtype=np.int16)
    probs = np.zeros((n, WIDTH), dtype=np.float32)
    pick = np.full(n, -1, dtype=np.int16)
    for lo in range(0, n, chunk):
        part = states[lo:lo + chunk]
        for k, ((a, p), stats) in enumerate(zip(teacher.improved_policies(part), teacher.root.last_stats)):
            order = np.argsort(-np.asarray(p))[:WIDTH]
            acts[lo + k, :len(order)] = np.asarray(a)[order]
            probs[lo + k, :len(order)] = np.asarray(p)[order]
            pick[lo + k] = z2_pick(stats)
    return acts, probs, pick


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=200000)
    ap.add_argument("--games", type=int, default=2000, help="greedy games to draw the positions from")
    ap.add_argument("--worlds", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--holdout", type=int, default=10, help="one game in this many is held out")
    ap.add_argument("--seed", type=int, default=3637)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    t0 = time.time()
    states, game, argmax = collect(model, dev, a.games, a.seed)
    print(f"{len(states):,} entry decisions in {a.games} games", flush=True)
    rng = np.random.default_rng(a.seed)
    keep = np.sort(rng.choice(len(states), size=min(a.positions, len(states)), replace=False))
    states = [states[i] for i in keep]
    game_a = np.asarray(game, dtype=np.int32)[keep]
    heldout = (game_a % a.holdout) == 0
    out: Dict[str, np.ndarray] = {
        "states": np.stack([np.frombuffer(s.raw_bytes(), dtype=np.uint8) for s in states]),
        "game": game_a, "heldout": heldout, "net_argmax": np.asarray(argmax, dtype=np.int16)[keep]}
    print(f"{len(states):,} entry decisions from {a.games} games ({heldout.sum():,} held out) in {time.time() - t0:.0f}s",
          flush=True)
    held = [s for s, h in zip(states, heldout) if h]
    hidx = np.flatnonzero(heldout)
    for w in a.worlds:
        cfg = RolloutConfig(k=4, worlds=w, horizon=4, rule="z2", stop="earlier", leaf_side="own", keyed=True)
        chunk = max(32, 16384 // w)
        teacher = RolloutTeacher(model, cfg, device=dev, seed=a.seed + w)
        t1 = time.time()
        out[f"acts_{w}"], out[f"probs_{w}"], out[f"pick_{w}"] = search(teacher, states, chunk)
        second = RolloutTeacher(model, cfg, device=dev, seed=a.seed + 7919 * w)
        a2, p2, k2 = search(second, held, chunk)
        out[f"acts2_{w}"] = np.full_like(out[f"acts_{w}"], -1)
        out[f"probs2_{w}"] = np.zeros_like(out[f"probs_{w}"])
        out[f"pick2_{w}"] = np.full_like(out[f"pick_{w}"], -1)
        out[f"acts2_{w}"][hidx], out[f"probs2_{w}"][hidx], out[f"pick2_{w}"][hidx] = a2, p2, k2
        dep = out[f"pick_{w}"] != out["net_argmax"]
        print(f"{w} worlds: {time.time() - t1:.0f}s; the root departs at {100 * dep.mean():.1f}%; held-out pick agreement "
              f"between seeds {100 * (out[f'pick_{w}'][hidx] == k2).mean():.1f}%", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    cast(Any, np.savez)(a.output, **out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    with open(a.output + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": a.checkpoint, "positions": len(states), "games": a.games, "worlds": a.worlds,
                   "holdout": a.holdout, "seed": a.seed, "teacher": "rollout k4 stop=earlier leaf=own keyed, R32 improved policy",
                   "commit": commit, "compiler": ts.BUILD_COMPILER,
                   # raw states are read only by the build that wrote them (ai/training/policy_fit checks)
                   "engine_fingerprint": fingerprint()}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
