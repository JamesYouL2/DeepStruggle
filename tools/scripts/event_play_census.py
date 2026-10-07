#!/usr/bin/env python3
"""How often a card in its owner's hand is used for its event, in a checkpoint's self-play.

One record per *holding*: the card entering a player's hand until it leaves. A holding counts as
evented when the holder headlines the card, or plays it in an action round and chooses the event,
the decision being the holder's own (not a choice inside another event's resolution -- Grain
Sales' card, Star Wars' retrieval). For a US or USSR card only its owner's holdings count (playing
the opponent's card, which fires the opponent's event, is ignored); a neutral card counts for
whoever holds it, split by side.

Every holding also records how it ended -- headline, event, Ops, space, or kept -- and the turn and
action round it was spent in. `HoldingTracker` is fed one decision at a time, so the same counting
runs on any decision stream, not only a model's self-play.

    PYTHONPATH=.:build/release python tools/scripts/event_play_census.py --checkpoint <pt|onnx> --games 4096
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import (LogitsFn, corpus_map, feed_corpus_game, load_policy, position_token,
                                     read_corpus_game, require_e4_view, rules_json, selfplay)
from tools.lib.player_agent import load_agent
from tools.scripts.placement_census import PlacementTracker

EVENT = ActionEncoder.PLAY_MODE_OFFSET
SPACE = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.SPACE)
US, USSR = int(ts.Player.US), int(ts.Player.USSR)
#: Scoring cards (0 Ops): played in a round they go straight to their event, with no play-mode
#: decision, so selecting one for an action round is its event.
_CARDS = rules_json("cards.json")
SCORING = {int(c["id"]) for c in _CARDS if int(c.get("ops", 0)) == 0}
_SIDES = {int(c["id"]): str(c.get("side", "neutral")).lower() for c in _CARDS}


def _owner(card: int) -> Optional[int]:
    """The side whose event the card is, or None for a neutral card."""
    return {"us": US, "ussr": USSR}.get(_SIDES.get(card, "neutral"))


@dataclass
class Holding:
    card: int
    side: int
    outcome: str = "kept"          # "headline", "event", "ops", "space", or "kept" (left the hand otherwise)
    legal: bool = False            # the owner could have played the event at some point while holding it
    turn: int = 0                  # when it was spent (0 while kept)
    ar: int = 0                    # action round; 0 for a headline
    game: int = 0                  # the corpus replay id (human holdings only)
    action: int = -1               # the flat action that spent it, at the position below
    pos: str = ""                  # that decision's position as a workbench token (when kept)


def _holdings_now(st: ts.GameState) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for c in range(1, 111):
        loc = st.get_card_location(c)
        if ts.in_hand_of(loc, ts.Player.US):
            out[c] = US
        elif ts.in_hand_of(loc, ts.Player.USSR):
            out[c] = USSR
    return out


def _mark_event_legal(st: ts.GameState, mover: int, now: Dict[int, int], latest: Dict[int, Holding]) -> None:
    """At the owner's action-round card choice: for each card in their hand not yet known to be
    eventable this holding, select it on a copy and see whether EVENT is legal at the play-mode
    decision that follows. A scoring card always is."""
    mask = np.asarray(ts.get_flat_action_mask(st))
    for idx in np.flatnonzero(mask[:ActionEncoder.PLAY_MODE_OFFSET]):
        card = int(ts.decode_flat_action(st, int(idx)).primary_id)
        h = latest.get(card)
        if h is None or h.legal or now.get(card) != mover:
            continue
        if card in SCORING:
            h.legal = True
            continue
        c = st.clone()
        if not ts.Engine.try_step_flat(c, int(idx), False, False):
            continue
        if c.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE and bool(np.asarray(ts.get_flat_action_mask(c))[EVENT]):
            h.legal = True


def _mark_headline_legal(st: ts.GameState, mover: int, now: Dict[int, int], latest: Dict[int, Holding]) -> None:
    """At the owner's headline choice a card they may headline is an event they could play when its
    event can trigger now -- the engine's own test, `CardHandlers::can_trigger_event`, the one the
    action-round mask uses. Any card may be headlined, but NATO before its prerequisite only
    fizzles. (Defectors, for one, is only ever playable as a headline.)"""
    mask = np.asarray(ts.get_flat_action_mask(st))
    who = ts.Player.US if mover == US else ts.Player.USSR
    for idx in np.flatnonzero(mask[:ActionEncoder.PLAY_MODE_OFFSET]):
        card = int(ts.decode_flat_action(st, int(idx)).primary_id)
        h = latest.get(card)
        if h is None or h.legal or now.get(card) != mover:
            continue
        if card in SCORING or ts.CardHandlers.can_trigger_event(st, card, who):
            h.legal = True


class HoldingTracker:
    """Follows one game's decisions and records its holdings. `observe(state, action)` takes each
    decision in order, before its action is applied, from whichever source drives the game: a
    model's self-play or a human game replayed by the ts-replayer converter."""

    def __init__(self, keep_positions: bool = False, game: int = 0) -> None:
        self.keep_positions = keep_positions
        self.game = game
        self.holdings: List[Holding] = []
        self.held: Dict[int, int] = {}                     # card -> side, as of the last decision
        self.latest: Dict[int, Holding] = {}               # card -> its latest holding
        self.selected: Optional[Tuple[int, int]] = None    # (card, side) chosen for an action round
        self.turn = 0                                      # the turn of the last decision seen

    def _spent(self, st: ts.GameState, card: int, outcome: str, a: int) -> None:
        h = self.latest[card]
        h.outcome, h.turn, h.ar = outcome, int(st.turn), int(st.action_round)
        h.game, h.action = self.game, a
        if self.keep_positions:
            h.pos = position_token(st)

    def observe(self, st: ts.GameState, a: int) -> None:
        self.turn = int(st.turn)
        now = _holdings_now(st)
        for c, side in now.items():                        # a card entering a hand
            if self.held.get(c) != side:
                h = Holding(c, side)
                self.holdings.append(h)
                self.latest[c] = h
        self.held = now
        ctx = st.ctx()
        if int(ctx.resolving_card) != 0:
            return                                         # inside an event's resolution
        mover = int(ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player)
        if ctx.decision_type == ts.DecisionType.SELECT_CARD and st.current_phase == ts.Phase.ACTION_ROUND:
            _mark_event_legal(st, mover, now, self.latest)
        elif ctx.decision_type == ts.DecisionType.SELECT_CARD and st.current_phase == ts.Phase.HEADLINE:
            _mark_headline_legal(st, mover, now, self.latest)
        if ctx.decision_type == ts.DecisionType.SELECT_CARD and a < ActionEncoder.PLAY_MODE_OFFSET:
            # The flat index is not the card id: decode it (flat 102 is card 103).
            card = int(ts.decode_flat_action(st, a).primary_id)
            if now.get(card) == mover:
                if st.current_phase == ts.Phase.HEADLINE:
                    self._spent(st, card, "headline", a)
                    self.latest[card].legal = True
                elif st.current_phase == ts.Phase.ACTION_ROUND:
                    if card in SCORING:
                        self._spent(st, card, "event", a)
                        self.latest[card].legal = True
                    else:
                        self.selected = (card, mover)
        elif ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and st.current_phase == ts.Phase.ACTION_ROUND:
            card = int(ctx.pending_op_card)
            if self.selected is not None and self.selected == (card, mover) and card in self.latest:
                # On the opponent's card EVENT is the event-first branch of an Ops play.
                own = _owner(card) in (None, mover)
                self._spent(st, card, "event" if a == EVENT and own else "space" if a == SPACE else "ops", a)
            self.selected = None


class _Both:
    """A holdings tracker and, optionally, a placement tracker fed the same decisions."""

    def __init__(self, placement: bool) -> None:
        self.holding = HoldingTracker()
        self.placement: Optional[PlacementTracker] = PlacementTracker() if placement else None

    def observe(self, st: ts.GameState, a: int) -> None:
        self.holding.observe(st, a)
        if self.placement is not None:
            self.placement.observe(st, a)


def play(logits_fn: LogitsFn, obs_features: int, games: int, seed: int, batch: int,
         temperature: float) -> List[Holding]:
    return [h for t in selfplay(logits_fn, obs_features, games, seed, batch, temperature, HoldingTracker)
            for h in t.holdings]


def _human_game(path: str) -> Tuple[List[List[Any]], str]:
    """One corpus game's holdings, dumped, and its status. A game whose record stops early loses
    the holdings still open at its last trusted decision: how they ended is unknown, and counting
    them as kept would invent an outcome the log never shows. The turn the record stops in never
    reaches the tracker (`untrusted_turn`), so the turns that do are complete and their timing is
    not tilted toward early rounds."""
    replay_id = int(read_corpus_game(path).get("replay_id", 0))
    tracker = HoldingTracker(keep_positions=True, game=replay_id)
    status = feed_corpus_game(path, tracker)
    hs = tracker.holdings
    if status == "partial":
        open_ = {id(tracker.latest[c]) for c in tracker.held if c in tracker.latest}
        hs = [h for h in hs if not (h.outcome == "kept" and id(h) in open_)]
    return ([dump_holding(h) for h in hs] if status != "skipped" else []), status


def human_holdings(workers: int, limit: int = 0) -> Tuple[List[Holding], Dict[str, int]]:
    """The holdings of every distinct game in the ts-replayer corpus, as the humans played them."""
    status: Dict[str, int] = collections.Counter()
    out: List[Holding] = []
    for recs, st in corpus_map(_human_game, workers, limit):
        status[st] += 1
        out += [load_holding(r) for r in recs]
    return out, dict(status)


def dump_holding(h: Holding) -> List[Any]:
    out: List[Any] = [h.card, h.side, h.outcome, h.legal, h.turn, h.ar]
    return out + [h.game, h.action, h.pos] if h.pos else out


def load_holding(r: Sequence[Any]) -> Holding:
    """A dumped holding; older dumps lack the legality flag (all legal) and the timing, and only
    human dumps carry the position."""
    h = Holding(int(r[0]), int(r[1]), str(r[2]), bool(r[3]) if len(r) > 3 else True,
                int(r[4]) if len(r) > 4 else 0, int(r[5]) if len(r) > 5 else 0)
    if len(r) > 8:
        h.game, h.action, h.pos = int(r[6]), int(r[7]), str(r[8])
    return h


def table(holdings: Sequence[Holding], games: int, source: str = "self-play") -> str:
    cards = {int(c["id"]): c for c in _CARDS}
    by: Dict[Tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for h in holdings:
        if h.legal:
            by[(h.card, h.side)][h.outcome] += 1

    def counts(keys: Sequence[Tuple[int, int]]) -> collections.Counter:
        c: collections.Counter = collections.Counter()
        for k in keys:
            c.update(by.get(k, collections.Counter()))
        return c

    def stats(keys: Sequence[Tuple[int, int]]) -> Tuple[int, int, int]:
        c = counts(keys)
        return sum(c.values()), c["headline"], c["event"]

    def cell(n: int, hd: int, ev: int) -> str:
        return "—" if n == 0 else f"{100 * (hd + ev) / n:.0f}% ({n:,})"

    def pct(k: int, n: int) -> str:
        return f"{100 * k / n:.0f}%"

    rows = []
    for cid, c in cards.items():
        side = str(c.get("side", "neutral")).lower()
        if cid == 6:                                   # The China Card: no event
            continue
        keys = [(cid, US if side == "us" else USSR)] if side in ("us", "ussr") else [(cid, US), (cid, USSR)]
        n, hd, ev = stats(keys)
        if side in ("us", "ussr"):
            us_cell = ussr_cell = ""
        else:
            us_cell, ussr_cell = cell(*stats([(cid, US)])), cell(*stats([(cid, USSR)]))
        if n == 0:
            continue
        cc = counts(keys)
        # Dumps written before the split hold "ops/space": shown under Ops, with space blank.
        legacy = cc["ops/space"] > 0
        ops = pct(cc["ops"] + cc["ops/space"], n)
        space = "—" if legacy else pct(cc["space"], n)
        rows.append(((hd + ev) / n, n, c["name"], side, hd, ev, ops, space, pct(cc["kept"], n), us_cell, ussr_cell))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    what = "games of the human ts-replayer corpus" if source == "human" else "greedy self-play games"
    out = [f"How often a card in its owner's hand is used for its event -- headlined, or played in an action "
           f"round as the event -- {games:,} {what}. One count per holding (the card entering "
           f"the hand until it leaves), counting only holdings in which the owner could have played the event at "
           f"one of their decisions: a headline choice where the card may be headlined and its event can trigger, "
           f"or an action-round card choice where selecting it offers the event. US/USSR cards: the owner's holdings only. Neutral "
           f"cards: whoever holds it, with the split by side.", "",
           "The rest of each holding: played for Ops (on an opponent's card that includes the event-first "
           "branch -- it is the opponent's event), sent to the space race, or kept -- still in hand at the "
           "end, or gone some other way (discarded, taken by an event, held when the game ended).", "",
           "| card | side | evented (holdings) | headlined | event in a round | Ops | space | kept | held by US | held by USSR |",
           "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for frac, n, name, side, hd, ev, ops, space, kept, uc, sc in rows:
        out.append(f"| {name} | {side} | **{100 * frac:.0f}%** ({n:,}) | {100 * hd / n:.0f}% | {100 * ev / n:.0f}% | "
                   f"{ops} | {space} | {kept} | {uc} | {sc} |")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="a .pt checkpoint or an .onnx export")
    ap.add_argument("--agent", default=None,
                    help="instead of --checkpoint, any tools.lib.player_agent.load_agent spec that decides "
                         "positions in a batch, e.g. gumbel:<ckpt.pt>:32:4 (search)")
    ap.add_argument("--placement-dump", default=None,
                    help="also count Ops influence placement in the same games (placement_census.py's dump)")
    ap.add_argument("--human-corpus", action="store_true",
                    help="count the humans' holdings in the ts-replayer corpus instead of self-play")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes for --human-corpus")
    ap.add_argument("--corpus-limit", type=int, default=0, help="at most this many corpus games (0 = all)")
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
        sources = set()
        for f in a.merge:
            with open(f) as fh:
                d = json.load(fh)
            games += int(d["games"])
            sources.add(str(d.get("source", "self-play")))
            recs += [load_holding(r) for r in d["holdings"]]
        if len(sources) > 1:
            ap.error(f"--merge mixes human and self-play dumps ({sorted(sources)}); tabulate them apart")
        md = table(recs, games, source=sources.pop())
        print(md)
        if a.output_md:
            open(a.output_md, "w").write(md + "\n")
        return 0
    if a.human_corpus:
        hs, status = human_holdings(a.workers, a.corpus_limit)
        games = status.get("complete", 0) + status.get("partial", 0)
        print(f"corpus: {status}", file=sys.stderr)
        if a.dump:
            json.dump({"games": games, "source": "human", "status": status,
                       "holdings": [dump_holding(h) for h in hs]}, open(a.dump, "w"))
        md = table(hs, games, source="human")
        print(md)
        if a.output_md:
            open(a.output_md, "w").write(md + "\n")
        return 0
    if not a.checkpoint and not a.agent:
        ap.error("--checkpoint or --agent is required unless --merge or --human-corpus is given")
    if a.agent:
        agent: Any = load_agent(a.agent, device="cuda" if torch.cuda.is_available() else "cpu")
        if not hasattr(agent, "select_actions_batch"):
            ap.error(f"--agent {a.agent!r} cannot decide positions in a batch")
        require_e4_view(a.agent, agent)
        if hasattr(agent, "reseed"):
            agent.reseed(a.seed)
        trackers = selfplay(None, 0, a.games, a.seed, a.batch, 0.0, lambda: _Both(bool(a.placement_dump)),
                            states_fn=agent.select_actions_batch)
    else:
        fn, features = load_policy(a.checkpoint)
        trackers = selfplay(fn, features, a.games, a.seed, a.batch, a.temperature,
                            lambda: _Both(bool(a.placement_dump)))
    hs = [h for t in trackers for h in t.holding.holdings]
    if a.placement_dump:
        json.dump({"games": a.games, "records": [r for t in trackers for r in t.placement.records]},
                  open(a.placement_dump, "w"))
    if a.dump:
        json.dump({"games": a.games, "holdings": [dump_holding(h) for h in hs]}, open(a.dump, "w"))
    md = table(hs, a.games)
    print(md)
    if a.output_md:
        open(a.output_md, "w").write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
