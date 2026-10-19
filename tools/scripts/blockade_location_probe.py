#!/usr/bin/env python3
"""Does the US policy hold its only 3-Ops card back against Blockade, and does that depend on
where Blockade is?

Blockade (USSR, 1 Ops, starred) makes the US discard a card worth 3+ Ops or lose all its
influence in West Germany. The US's only 3-Ops card is its protection, so a policy that has
learned the threat should be readier to play it when Blockade cannot come (removed) and more
reluctant when the USSR is known to hold it.

The probe self-plays the policy greedily to the end of turn 2 and keeps every US action-round
card choice of turn 2 (not one inside an event) where the US controls West Germany, holds exactly
one 3-Ops card (the China card is not a hand card and is ignored), does not hold Blockade itself,
and has neither Red Scare/Purge on it nor Containment in force. Each kept position is rebuilt
with Blockade in each location, and the policy's probability of choosing the 3-Ops card is read
in each:

    draw          the draw deck
    ussr_unknown  the USSR's hand, unseen by the US (the same observation as `draw`)
    discard       the discard pile
    removed       removed from the game
    ussr_known    the USSR's hand, known to the US

The USSR's hand size is in the observation (global 70), so a move into or out of its hand swaps
an unseen card the other way between the USSR's hand and the draw deck; the US sees nothing
else change. Nothing is played on from a rebuilt position.

    PYTHONPATH=.:build/release python tools/scripts/blockade_location_probe.py \\
        --checkpoint <ckpt.pt> --games 4000 --output-json o.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.corpus_driver import LogitsFn, load_policy  # noqa: E402

BLOCKADE = ts.CardData.get_card_by_name("Blockade")
WEST_GERMANY = ts.MapData.get_country_by_name("West Germany")
TURN = 2
L = ts.CardLocation
VARIANTS: Dict[str, ts.CardLocation] = {
    "draw": L.DRAW_DECK, "ussr_unknown": L.HAND_USSR_UNKNOWN, "discard": L.DISCARD_PILE,
    "removed": L.REMOVED_FROM_GAME, "ussr_known": L.HAND_USSR_KNOWN}
USSR_HAND = (L.HAND_USSR_UNKNOWN, L.HAND_USSR_KNOWN)
US_HAND = (L.HAND_US_UNKNOWN, L.HAND_US_KNOWN)
OPS = {c: int(ts.CardData.get_card_info(c)["ops"]) for c in range(1, 111)}


@dataclass
class Position:
    game: int
    action_round: int
    card: int                      # the US's only 3-Ops card
    natural: str                   # where Blockade really was
    holds_four: bool               # the US also holds a 4-Ops card
    hand_size: int
    prob: Dict[str, float] = field(default_factory=dict)
    top: Dict[str, bool] = field(default_factory=dict)


def us_hand(st: ts.GameState) -> List[int]:
    return [c for c in range(1, 111) if st.get_card_location(c) in US_HAND]


def qualifies(st: ts.GameState) -> Optional[int]:
    """The single 3-Ops card if `st` is a position the probe keeps, else None."""
    if int(st.turn) != TURN or st.current_phase != ts.Phase.ACTION_ROUND:
        return None
    ctx = st.ctx()
    if ctx.decision_type != ts.DecisionType.SELECT_CARD or int(ctx.resolving_card) != 0:
        return None
    mover = ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player
    if mover != ts.Player.US:
        return None
    if not ts.Scoring.is_controlled_by(st, WEST_GERMANY, ts.Player.US):
        return None
    eff = int(st.persistent_effects)
    if eff & (ts.EffectBits.PURGE_US_ACTIVE | ts.EffectBits.CONTAINMENT_ACTIVE):
        return None
    hand = us_hand(st)
    if BLOCKADE in hand:
        return None
    threes = [c for c in hand if OPS[c] == 3]
    if len(threes) != 1:
        return None
    if not ts.get_flat_action_mask(st)[threes[0] - 1]:
        return None
    return threes[0]


def place_blockade(st: ts.GameState, loc: ts.CardLocation, rng: np.random.Generator) -> Optional[ts.GameState]:
    """`st` with Blockade at `loc`, the USSR's hand size kept by swapping an unseen card; None
    when there is no unseen card to swap (the USSR's hand empty or fully known, or no deck)."""
    s = st.clone()
    was_in, now_in = s.get_card_location(BLOCKADE) in USSR_HAND, loc in USSR_HAND
    if was_in != now_in:
        src, dst = (L.DRAW_DECK, L.HAND_USSR_UNKNOWN) if was_in else (L.HAND_USSR_UNKNOWN, L.DRAW_DECK)
        pool = [c for c in range(1, 111) if c != BLOCKADE and s.get_card_location(c) == src]
        if not pool:
            return None
        s.set_card_location(int(rng.choice(pool)), dst)
    s.set_card_location(BLOCKADE, loc)
    return s


def natural_name(loc: ts.CardLocation) -> str:
    for k, v in VARIANTS.items():
        if v == loc:
            return k
    return str(loc).split(".")[-1].lower()


def probabilities(logits_fn: LogitsFn, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    z = np.where(masks > 0, logits_fn(obs, masks).astype(np.float64), -np.inf)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def evaluate(logits_fn: LogitsFn, features: int, kept: Sequence[Tuple[ts.GameState, Position]],
             rng: np.random.Generator, chunk: int = 400) -> Tuple[List[Position], List[Position]]:
    """Fills each position's probabilities; returns the positions every variant could be built
    for, and those dropped."""
    built: List[Tuple[Position, np.ndarray, np.ndarray]] = []
    dropped: List[Position] = []
    for st, pos in kept:
        variants = [place_blockade(st, loc, rng) for loc in VARIANTS.values()]
        if any(v is None for v in variants):
            dropped.append(pos)
            continue
        obs = np.stack([ts.extract_observation_features(v, ts.Player.US, features) for v in variants if v is not None])
        if not np.array_equal(obs[0], obs[1]):
            raise AssertionError("Blockade in the draw deck and unseen in the USSR's hand must look the same")
        built.append((pos, obs, np.asarray(ts.get_flat_action_mask(st))))
    k = len(VARIANTS)
    for c0 in range(0, len(built), chunk):
        part = built[c0:c0 + chunk]
        obs = np.concatenate([o for _, o, _ in part])
        masks = np.concatenate([np.repeat(m[None], k, axis=0) for _, _, m in part])
        p = probabilities(logits_fn, obs, masks)
        for j, (pos, _, _) in enumerate(part):
            for i, name in enumerate(VARIANTS):
                row = p[j * k + i]
                pos.prob[name] = float(row[pos.card - 1])
                pos.top[name] = bool(row.argmax() == pos.card - 1)
    return [pos for pos, _, _ in built], dropped


def collect(logits_fn: LogitsFn, features: int, games: int, seed: int,
            batch: int) -> List[Tuple[ts.GameState, Position]]:
    kept: List[Tuple[ts.GameState, Position]] = []
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(features, features)
        obs, masks, _ = env.reset_all()
        # Latched: a finished game's slot is reset by the runner into a new game, which must
        # not be read as more of the first.
        live = [True] * n
        for _ in range(20_000):
            states = [env.runner.get_state(i) for i in range(n)]
            live = [live[i] and not ts.Engine.is_terminal(s) and int(s.turn) <= TURN for i, s in enumerate(states)]
            if not any(live):
                break
            obs_np, masks_np = np.asarray(obs), np.asarray(masks)
            for i, s in enumerate(states):
                card = qualifies(s) if live[i] else None
                if card is None:
                    continue
                own = ts.extract_observation_features(s, ts.Player.US, features)
                if not np.array_equal(own, obs_np[i, :own.shape[0]]):
                    raise AssertionError("the runner's observation and extract_observation_features differ")
                hand = us_hand(s)
                kept.append((s, Position(
                    game=b0 + i, action_round=int(s.action_round), card=card,
                    natural=natural_name(s.get_card_location(BLOCKADE)),
                    holds_four=any(OPS[c] >= 4 for c in hand), hand_size=len(hand))))
            actions = logits_fn(obs_np, masks_np).argmax(axis=1)
            obs, masks, _, _, _ = env.step(actions)
    return kept


def summarise(positions: Sequence[Position], label: str) -> List[str]:
    names = list(VARIANTS)
    games = len({p.game for p in positions})
    out = [f"### {label}: {len(positions)} positions from {games} games", "",
           "| Blockade | P(play the 3-Ops card) | argmax share | vs draw (paired) |", "|:---|---:|---:|---:|"]
    if not positions:
        return out + [""]
    for nm in names:
        pr = np.array([p.prob[nm] for p in positions])
        top = np.mean([p.top[nm] for p in positions])
        by_game: Dict[int, List[float]] = collections.defaultdict(list)
        for p in positions:
            by_game[p.game].append(p.prob[nm] - p.prob["draw"])
        g = np.array([np.mean(v) for v in by_game.values()])
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float("nan")
        diff = "" if nm == "draw" else f"{(pr - np.array([p.prob['draw'] for p in positions])).mean():+.4f} ± {se:.4f}"
        out.append(f"| {nm} | {pr.mean():.4f} | {top:.1%} | {diff} |")
    return out + [""]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)

    logits_fn, features = load_policy(a.checkpoint)
    kept = collect(logits_fn, features, a.games, a.seed, a.batch)
    positions, dropped = evaluate(logits_fn, features, kept, np.random.default_rng(a.seed))

    lines = [f"# Blockade's location and the US's only 3-Ops card, turn {TURN}", "",
             f"`{a.checkpoint}`, {a.games} greedy self-play games, seed {a.seed}. "
             "± is one standard error of the paired difference, games as the unit. "
             f"{len(dropped)} of {len(kept)} kept positions dropped, no unseen card to swap (by action round: "
             + ", ".join(f"AR{k} {v}" for k, v in sorted(collections.Counter(p.action_round for p in dropped).items()))
             + ").", ""]
    lines += summarise(positions, "all")
    lines += summarise([p for p in positions if not p.holds_four], "no 4-Ops card in hand")
    lines += summarise([p for p in positions if p.holds_four], "also holds a 4-Ops card")
    for ar in sorted({p.action_round for p in positions}):
        lines += summarise([p for p in positions if p.action_round == ar], f"action round {ar}")
    nat = collections.Counter(p.natural for p in positions)
    cards = collections.Counter(ts.CardData.get_card_name(p.card) for p in positions)
    lines += ["Blockade's real location: " + ", ".join(f"{k} {v}" for k, v in nat.most_common()), "",
              "The 3-Ops card: " + ", ".join(f"{k} {v}" for k, v in cards.most_common()), ""]
    text = "\n".join(lines)
    print(text)
    if a.output_md:
        with open(a.output_md, "w") as f:
            f.write(text)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump({"checkpoint": a.checkpoint, "games": a.games, "seed": a.seed,
                       "positions": [asdict(p) for p in positions]}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
