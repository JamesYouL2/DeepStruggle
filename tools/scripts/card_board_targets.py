#!/usr/bin/env python3
"""P30: engine-labelled card x board targets, for testing whether an architecture can represent
card <-> country interactions and whether a trained net has learned them.

Self-play positions of one or more checkpoints are sampled, and every card in the mover's hand is
labelled with quantities that depend on that card *and* the board, never on either alone:

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

N_COUNTRIES, BOARD_W, N_CARDS, CARD_W = 84, 26, 110, 14
CARD_OFFSET = N_COUNTRIES * BOARD_W
GLOBAL_OFFSET = CARD_OFFSET + N_CARDS * CARD_W
MY_HAND = 1
DECISION_TYPE = GLOBAL_OFFSET + 72
CHINA = 6
MAX_HAND = 12
T2_NAMES = ("countries to control", "battlegrounds to control", "best coup chance", "best battleground coup chance")
T4_NAMES = ("VP", "DEFCON", "margin Europe", "margin Asia", "margin Middle East", "margin Africa",
            "margin Central America", "margin South America", "my battlegrounds", "their battlegrounds",
            "my influence", "their influence")

_INFO = {c: ts.CardData.get_card_info(c) for c in range(1, N_CARDS + 1)}
OPS = np.array([0] + [int(_INFO[c]["ops"]) for c in range(1, N_CARDS + 1)], dtype=np.int64)
SIDE = {c: str(_INFO[c]["side"]) for c in range(1, N_CARDS + 1)}
SCORING = sorted(c for c in range(1, N_CARDS + 1) if _INFO[c]["is_scoring"])


def _summary(obs: np.ndarray) -> np.ndarray:
    """The quantities T4 differences, from one observation (mover's side)."""
    b = obs[:CARD_OFFSET].reshape(N_COUNTRIES, BOARD_W)
    bg = b[:, 4] > 0.5
    return np.concatenate([
        (obs[GLOBAL_OFFSET + 64:GLOBAL_OFFSET + 70] * 20.0),
        [float(((b[:, 5] > 0.5) & bg).sum()), float(((b[:, 6] > 0.5) & bg).sum()),
         float((b[:, 0] * 10.0).sum()), float((b[:, 1] * 10.0).sum())]])


def t2(obs: np.ndarray, card: int) -> Optional[np.ndarray]:
    ops = int(OPS[card])
    if ops <= 0:
        return None
    b = obs[:CARD_OFFSET].reshape(N_COUNTRIES, BOARD_W)
    can_place = b[:, 19] > 0.5
    mine = b[:, 5] > 0.5
    bg = b[:, 4] > 0.5
    deficit = np.rint(b[:, 24] * 5.0)
    stab = np.rint(b[:, 3] * 5.0)
    can_coup = b[:, 21] > 0.5
    reach = can_place & ~mine & (deficit > 0) & (deficit <= ops)
    p = np.clip(6.0 - 2.0 * stab + ops, 0.0, 6.0) / 6.0
    best = float(p[can_coup].max()) if can_coup.any() else 0.0
    best_bg = float(p[can_coup & bg].max()) if (can_coup & bg).any() else 0.0
    return np.array([float(reach.sum()), float((reach & bg).sum()), best, best_bg], dtype=np.float32)


def _ctx(st: "ts.GameState") -> Tuple[int, int, int, int]:
    c = st.ctx()
    return int(c.decision_type), int(c.decision_player), int(c.pending_op_card), int(c.remaining_steps)


def t3_t4(st: "ts.GameState", mover: "ts.Player", card: int, base_obs: np.ndarray,
          seeds: Sequence[int]) -> Tuple[Optional[bool], Optional[np.ndarray]]:
    if card == CHINA:
        return None, None
    side = SIDE[card]
    fire = ts.Player.US if side == "US" else ts.Player.USSR if side == "USSR" else mover
    can = bool(ts.CardHandlers.can_trigger_event(st, card, fire))
    if not can:
        return False, None
    sign = 1.0 if mover == ts.Player.US else -1.0
    base = _summary(base_obs)
    c0 = _ctx(st)
    deltas: List[np.ndarray] = []
    for s in seeds:
        x = st.clone()
        x.rng_state = int(s)
        vp0, dc0 = int(x.victory_points), int(x.defcon)
        ts.CardHandlers.trigger_event(x, card, fire, 0)
        c1 = _ctx(x)
        if c1 != c0 and c1[0] != 0:
            return True, None                         # stops for a choice: T5, not labelled
        after = _summary(np.asarray(ts.extract_observation(x, mover)))
        deltas.append(np.concatenate([[sign * (int(x.victory_points) - vp0), float(int(x.defcon) - dc0)],
                                      after - base]).astype(np.float32))
    return True, np.mean(deltas, 0)


def label(st: "ts.GameState", mover: "ts.Player", obs: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    rows = obs[CARD_OFFSET:GLOBAL_OFFSET].reshape(N_CARDS, CARD_W)
    held = [c for c in range(1, N_CARDS + 1) if rows[c - 1, MY_HAND] > 0.5][:MAX_HAND]
    cards = np.zeros(MAX_HAND, dtype=np.int16)
    y2 = np.zeros((MAX_HAND, 4), dtype=np.float32)
    y3 = np.zeros(MAX_HAND, dtype=np.float32)
    y4 = np.zeros((MAX_HAND, 12), dtype=np.float32)
    m2 = np.zeros(MAX_HAND, dtype=bool)
    m3 = np.zeros(MAX_HAND, dtype=bool)
    m4 = np.zeros(MAX_HAND, dtype=bool)
    seeds = (seed * 2 + 1, seed * 2 + 2)
    for j, c in enumerate(held):
        cards[j] = c
        v2 = t2(obs, c)
        if v2 is not None:
            y2[j], m2[j] = v2, True
        can, v4 = t3_t4(st, mover, c, obs, seeds)
        if can is not None:
            y3[j], m3[j] = float(can), True
        if v4 is not None:
            y4[j], m4[j] = v4, True
    return {"cards": cards, "y2": y2, "y3": y3, "y4": y4, "m2": m2, "m3": m3, "m4": m4}


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
            lab = label(st, mover, o, int(rng.integers(1 << 30)))
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
