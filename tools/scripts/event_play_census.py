#!/usr/bin/env python3
"""How often a card in its owner's hand is used for its event, in a checkpoint's self-play.

One record per *holding*: the card entering a player's hand until it leaves. A holding counts as
evented when the holder headlines the card, or plays it in an action round and chooses the event,
the decision being the holder's own (not a choice inside another event's resolution -- Grain
Sales' card, Star Wars' retrieval). For a US or USSR card only its owner's holdings count (playing
the opponent's card, which fires the opponent's event, is ignored); a neutral card counts for
whoever holds it, split by side.

    PYTHONPATH=.:build/release python tools/scripts/event_play_census.py --checkpoint <pt> --games 4096
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder

EVENT = ActionEncoder.PLAY_MODE_OFFSET
US, USSR = int(ts.Player.US), int(ts.Player.USSR)
#: Scoring cards (0 Ops): played in a round they go straight to their event, with no play-mode
#: decision, so selecting one for an action round is its event.
SCORING = {int(c["id"]) for c in json.load(open("rules/cards.json")) if int(c.get("ops", 0)) == 0}


@dataclass
class Holding:
    card: int
    side: int
    outcome: str = "kept"          # "headline", "event", "ops/space", or "kept" (left the hand otherwise)


def _holdings_now(st: ts.GameState) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for c in range(1, 111):
        loc = st.get_card_location(c)
        if ts.in_hand_of(loc, ts.Player.US):
            out[c] = US
        elif ts.in_hand_of(loc, ts.Player.USSR):
            out[c] = USSR
    return out


def play(model: Any, games: int, seed: int, batch: int, temperature: float) -> List[Holding]:
    from bindings.ts_env import TsVectorizedEnv, model_obs_features
    device = next(model.parameters()).device
    model.eval()
    holdings: List[Holding] = []
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(model_obs_features(model), model_obs_features(model))
        obs, masks, _ = env.reset_all()
        done = [False] * n
        held: List[Dict[int, int]] = [{} for _ in range(n)]           # card -> side, as of last step
        latest: List[Dict[int, Holding]] = [{} for _ in range(n)]     # card -> its latest holding
        selected: List[Optional[Tuple[int, int]]] = [None] * n        # (card, side) chosen for an AR
        for _ in range(20_000):
            if all(done):
                break
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
                               torch.from_numpy(np.asarray(masks)).to(device))[0].float()
                acts = (torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(-1)
                        if temperature > 0 else logits.argmax(-1))
            actions = acts.cpu().numpy()
            for i in range(n):
                if done[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                now = _holdings_now(st)
                for c, side in now.items():                            # a card entering a hand
                    if held[i].get(c) != side:
                        h = Holding(c, side)
                        holdings.append(h)
                        latest[i][c] = h
                held[i] = now
                ctx = st.ctx()
                if int(ctx.resolving_card) != 0:
                    continue                                           # inside an event's resolution
                mover = int(ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player)
                a = int(actions[i])
                if ctx.decision_type == ts.DecisionType.SELECT_CARD and a < ActionEncoder.PLAY_MODE_OFFSET:
                    # The flat index is not the card id: decode it (flat 102 is card 103).
                    card = int(ts.decode_flat_action(st, a).primary_id)
                    if now.get(card) == mover:
                        if st.current_phase == ts.Phase.HEADLINE:
                            latest[i][card].outcome = "headline"
                        elif st.current_phase == ts.Phase.ACTION_ROUND:
                            if card in SCORING:
                                latest[i][card].outcome = "event"
                            else:
                                selected[i] = (card, mover)
                elif ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and st.current_phase == ts.Phase.ACTION_ROUND:
                    card = int(ctx.pending_op_card)
                    sel = selected[i]
                    if sel is not None and sel == (card, mover) and card in latest[i]:
                        latest[i][card].outcome = "event" if a == EVENT else "ops/space"
                    selected[i] = None
            obs, masks, _, dones, _ = env.step(actions)
            for i, d in enumerate(dones):
                if d:
                    done[i] = True
    return holdings


def table(holdings: Sequence[Holding], games: int) -> str:
    cards = {int(c["id"]): c for c in json.load(open("rules/cards.json"))}
    by: Dict[Tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for h in holdings:
        by[(h.card, h.side)][h.outcome] += 1

    def stats(keys: Sequence[Tuple[int, int]]) -> Tuple[int, int, int]:
        c = collections.Counter()
        for k in keys:
            c.update(by.get(k, collections.Counter()))
        n = sum(c.values())
        return n, c["headline"], c["event"]

    def cell(n: int, hd: int, ev: int) -> str:
        return "—" if n == 0 else f"{100 * (hd + ev) / n:.0f}% ({n:,})"

    rows = []
    for cid, c in cards.items():
        side = str(c.get("side", "neutral")).lower()
        if cid == 6:                                   # The China Card: no event
            continue
        if side in ("us", "ussr"):
            n, hd, ev = stats([(cid, US if side == "us" else USSR)])
            us_cell = ussr_cell = ""
        else:
            n, hd, ev = stats([(cid, US), (cid, USSR)])
            us_cell, ussr_cell = cell(*stats([(cid, US)])), cell(*stats([(cid, USSR)]))
        if n == 0:
            continue
        rows.append(((hd + ev) / n, n, c["name"], side, hd, ev, us_cell, ussr_cell))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    out = [f"How often a card in its owner's hand is used for its event -- headlined, or played in an action "
           f"round as the event -- {games:,} greedy self-play games. One count per holding (the card entering "
           f"the hand until it leaves). US/USSR cards: the owner's holdings only. Neutral cards: whoever holds "
           f"it, with the split by side.", "",
           "| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |",
           "|:---|:---|---:|---:|---:|---:|---:|"]
    for frac, n, name, side, hd, ev, uc, sc in rows:
        out.append(f"| {name} | {side} | **{100 * frac:.0f}%** ({n:,}) | {100 * hd / n:.0f}% | {100 * ev / n:.0f}% | {uc} | {sc} |")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=55_000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--dump", default=None, help="write the holdings here (JSON), for --merge")
    ap.add_argument("--merge", nargs="+", default=None, help="tabulate these dumps instead of playing")
    a = ap.parse_args(argv)
    if a.merge:
        recs: List[Holding] = []
        games = 0
        for f in a.merge:
            d = json.load(open(f))
            games += int(d["games"])
            recs += [Holding(int(c), int(s), str(o)) for c, s, o in d["holdings"]]
        md = table(recs, games)
        print(md)
        if a.output_md:
            open(a.output_md, "w").write(md + "\n")
        return 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model
    hs = play(model, a.games, a.seed, a.batch, a.temperature)
    if a.dump:
        json.dump({"games": a.games, "holdings": [[h.card, h.side, h.outcome] for h in hs]}, open(a.dump, "w"))
    md = table(hs, a.games)
    print(md)
    if a.output_md:
        open(a.output_md, "w").write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
