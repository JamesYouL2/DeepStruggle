#!/usr/bin/env python3
"""P30: engine-labelled card x board targets, for testing whether an architecture can represent
card <-> country interactions and whether a trained net has learned them.

Self-play positions of one or more checkpoints are sampled, and every card in the mover's hand is
labelled (`ai.training.card_event_targets`, shared with the `--aux-card-events` training target)
with quantities that depend on that card *and* the board, never on either alone:

* **T2, ops arithmetic** (from the observation's own slots, i.e. the engine's definitions): with
  this card's printed Ops, the countries the mover could bring under control (it can place there,
  does not control it yet, and needs no more influence points than the card has Ops;
  `board[24]` is the engine's deficit), the same for battlegrounds, and the best coup success
  chance over countries it can coup, overall and among battlegrounds -- a die d succeeds when
  d + Ops > 2 x stability. Ops modifiers (Red Scare, Containment, ...) are not applied. Scoring
  cards (0 Ops) are left out.
* **T3, the event can fire** -- `CardHandlers.can_trigger_event` for the player it fires for: the
  card's owner for a side card (so an opponent's card played for Ops fires for the opponent), the
  mover for a neutral one. Prerequisites (NATO needs Marshall or Warsaw, Solidarity needs John
  Paul) and board conditions both enter.
* **T4, what the event does** -- on a clone, the event is fired for that player, and the change is
  read from the mover's observation before and after: VP, DEFCON, the six regional scoring
  margins, battlegrounds controlled by each side, total influence of each side -- 12 numbers, all
  from the mover's side. Averaged over two engine seeds (dice). Only for events that resolve
  without a further decision; an event that stops for a choice is left out (T5, not built).
  **T1** is T4's VP change for the seven scoring cards.

Output: one .npz with the positions (observation, legal mask, mover, turn, game, decision type)
and per position up to `MAX_HAND` held cards with their targets and masks.

    PYTHONPATH=.:build/release python tools/scripts/card_board_targets.py \\
        --policies a.pt b.pt --games 3000 --output data/datasets/card_board/e6.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402

from ai.training.card_event_targets import (BOARD_W, CARD_OFFSET, CARD_W, DECISION_TYPE,  # noqa: E402,F401
                                            GLOBAL_OFFSET, MAX_HAND, N_CARDS, N_COUNTRIES, SCORING,
                                            T2_NAMES, T4_NAMES, label)


@torch.no_grad()
def collect(agent: NeuralAgent, games: int, envs: int, seed: int, sample_frac: float, temperature: float,
            dev: torch.device, game_offset: int) -> Dict[str, np.ndarray]:
    model: Any = agent.model
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    game_of = np.arange(envs) + game_offset
    next_game = game_offset + envs
    rows: List[Dict[str, Any]] = []
    done = 0
    t0 = time.time()
    while done < games:
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        ob = np.asarray(obs)
        mk = np.asarray(masks)
        for i in np.flatnonzero((dp != 0) & (rng.random(envs) < sample_frac)):
            st = env.runner.get_state(int(i))
            mover = ts.Player(int(dp[i]))
            o = ob[i].astype(np.float32)
            sd = int(rng.integers(1 << 30))
            lab = label(st, mover, o, (sd * 2 + 1, sd * 2 + 2))
            if not lab["m2"].any() and not lab["m3"].any():
                continue
            rows.append({"obs": ob[i].astype(np.float16), "mask": mk[i].astype(bool), "mover": int(dp[i]),
                         "turn": int(st.turn), "game": int(game_of[i]),
                         "decision": int(np.argmax(o[DECISION_TYPE:DECISION_TYPE + 8])), **lab})
        o_t = torch.from_numpy(ob).float().to(dev)
        m_t = torch.from_numpy(mk).to(dev)
        logits = model(o_t, m_t)[0].float()
        acts = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1).cpu().numpy()
        obs, masks, _r, _d, info = env.step(acts)
        for ep in info.get("completed_episodes", []):
            i = int(ep["env_idx"])
            game_of[i] = next_game
            next_game += 1
            done += 1
        if done and done % 500 == 0:
            print(f"  {agent.name}: {done} games, {len(rows)} positions, {time.time() - t0:.0f}s", flush=True)
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=3000, help="games per policy")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.output):
        raise SystemExit(f"{a.output} exists; refusing to overwrite")
    dev = torch.device(a.device)
    parts: List[Dict[str, np.ndarray]] = []
    for k, path in enumerate(a.policies):
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        parts.append(collect(ag, a.games, a.envs, a.seed + 1000 * k, a.sample_frac, a.temperature, dev,
                             game_offset=k * 10_000_000))
        print(f"{ag.name}: {parts[-1]['obs'].shape[0]} positions", flush=True)
    d = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    arrays: Dict[str, Any] = {"policies": np.array([str(p) for p in a.policies]), **d}
    np.savez(a.output, **arrays)
    print(f"wrote {a.output}: {d['obs'].shape[0]} positions; held-card labels: T2 {int(d['m2'].sum())}, "
          f"T3 {int(d['m3'].sum())} ({100 * d['y3'][d['m3']].mean():.1f}% can fire), T4 {int(d['m4'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
