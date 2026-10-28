#!/usr/bin/env python3
"""How often a card in its owner's hand is used for its event, in a checkpoint's self-play.

One record per *holding*: the card entering a player's hand until it leaves, by whoever holds it.
A holding counts as evented when the holder headlines the card, or plays it in an action round and
chooses the event, the decision being the holder's own (not a choice inside another event's
resolution -- Grain Sales' card, Star Wars' retrieval). The table splits each side's card into its
owner's holdings and the opponent's; a neutral card counts for whoever holds it, split by side.

Every holding also records how it ended and the turn and action round it was spent in. The holder's
own or a neutral card: headline, event, Ops, space, or kept. The opponent's card: its event fired
(headlined, or played for Ops in either order, and the event occurred), no event (played that way,
but the event could not occur), space, UN Intervention's companion, or kept. `HoldingTracker` is fed
one decision at a time, so the same counting runs on any decision stream, not only a model's
self-play.

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
from tools.lib.engine_fingerprint import require_fresh
from tools.scripts.placement_census import PlacementTracker

EVENT = ActionEncoder.PLAY_MODE_OFFSET
SPACE = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.SPACE)
US, USSR = int(ts.Player.US), int(ts.Player.USSR)
#: Scoring cards (0 Ops): played in a round they go straight to their event, with no play-mode
#: decision, so selecting one for an action round is its event.
_CARDS = rules_json("cards.json")
SCORING = {int(c["id"]) for c in _CARDS if int(c.get("ops", 0)) == 0}
_SIDES = {int(c["id"]): str(c.get("side", "neutral")).lower() for c in _CARDS}
_STARRED = {int(c["id"]) for c in _CARDS if c.get("one_time")}
UN_INTERVENTION = 32
THE_CHINA_CARD = 6
#: Holdings from version 2 on record the opponent's card's own outcomes ("fired", "no event", "un");
#: before, they were filed as the holder's own ("headline", "ops"), and the table shows them n/a.
HOLDINGS_VERSION = 2


def _owner(card: int) -> Optional[int]:
    """The side whose event the card is, or None for a neutral card."""
    return {"us": US, "ussr": USSR}.get(_SIDES.get(card, "neutral"))


@dataclass
class Holding:
    card: int
    side: int                      # the holder
    # Own or neutral card: "headline", "event", "ops", "space"; the opponent's card: "fired", "no event",
    # "space", "un"; either: "kept" (left the hand otherwise)
    outcome: str = "kept"
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
        self.pending: List[Holding] = []                   # starred opponent cards, their events not yet settled

    def _spent(self, st: ts.GameState, card: int, outcome: str, a: int) -> None:
        h = self.latest[card]
        h.outcome, h.turn, h.ar = outcome, int(st.turn), int(st.action_round)
        h.game, h.action = self.game, a
        if self.keep_positions:
            h.pos = position_token(st)

    def _opponent_play(self, st: ts.GameState, card: int, mover: int, a: int) -> None:
        """The holder headlining the opponent's card or playing it for Ops, either order: the event
        fires when it can trigger (`trigger_event`'s own test, on flags an Ops-first play cannot
        change before it resolves). A headline resolves after the other side's, which can change them
        (Marshall Plan before NATO): for a card that is not starred -- Socialist Governments, OPEC,
        Arab-Israeli War, Muslim Revolution -- the test at the choice misses that. Our Man in Tehran
        and Kitchen Debates also test the board as they resolve (`event_has_effect`). So a starred
        card waits for the engine's own verdict -- removed from the game means its event occurred --
        at the next card choice after it has left the hand. A headlined card stays in the hand through the other side's headline choice, and is
        resolved by the first action round; a card discarded rather than removed comes back to a hand
        only at a later turn's deal, so it is still in flight only while it is in the holder's hand in
        the same turn. A game that ends first keeps the test at the choice."""
        owner = ts.Player.US if mover == USSR else ts.Player.USSR
        fires = ts.CardHandlers.can_trigger_event(st, card, owner)
        self._spent(st, card, "fired" if fires else "no event", a)
        if card in _STARRED:
            self.pending.append(self.latest[card])

    def _settle(self, st: ts.GameState) -> None:
        ctx = st.ctx()
        if not self.pending or ctx.decision_type != ts.DecisionType.SELECT_CARD or int(ctx.resolving_card) != 0:
            return
        waiting = []
        for h in self.pending:
            loc = st.get_card_location(h.card)
            holder = ts.Player.US if h.side == US else ts.Player.USSR
            if int(st.turn) == h.turn and (ts.in_hand_of(loc, holder) or loc == ts.CardLocation.HEADLINE_COMMITTED):
                waiting.append(h)                          # still played: the other side's headline choice
            else:
                h.outcome = "fired" if loc == ts.CardLocation.REMOVED_FROM_GAME else "no event"
        self.pending = waiting

    def observe(self, st: ts.GameState, a: int) -> None:
        self.turn = int(st.turn)
        self._settle(st)
        now = _holdings_now(st)
        for c, side in now.items():                        # a card entering a hand
            if self.held.get(c) != side:
                h = Holding(c, side)
                self.holdings.append(h)
                self.latest[c] = h
        self.held = now
        ctx = st.ctx()
        mover = int(ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player)
        if (int(ctx.resolving_card) == UN_INTERVENTION and ctx.decision_type == ts.DecisionType.SELECT_CARD
                and a < ActionEncoder.PLAY_MODE_OFFSET):
            # UN Intervention's companion: its Ops, without its event (SELECT_OP_MODE, no play mode).
            card = int(ts.decode_flat_action(st, a).primary_id)
            if now.get(card) == mover and card in self.latest:
                self._spent(st, card, "un", a)
            return
        if int(ctx.resolving_card) != 0:
            return                                         # inside an event's resolution
        if ctx.decision_type == ts.DecisionType.SELECT_CARD and st.current_phase == ts.Phase.ACTION_ROUND:
            _mark_event_legal(st, mover, now, self.latest)
        elif ctx.decision_type == ts.DecisionType.SELECT_CARD and st.current_phase == ts.Phase.HEADLINE:
            _mark_headline_legal(st, mover, now, self.latest)
        if ctx.decision_type == ts.DecisionType.SELECT_CARD and a < ActionEncoder.PLAY_MODE_OFFSET:
            # The flat index is not the card id: decode it (flat 102 is card 103).
            card = int(ts.decode_flat_action(st, a).primary_id)
            if now.get(card) == mover:
                if st.current_phase == ts.Phase.HEADLINE and _owner(card) not in (None, mover):
                    self._opponent_play(st, card, mover, a)
                elif st.current_phase == ts.Phase.HEADLINE:
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
                if a == SPACE:
                    self._spent(st, card, "space", a)
                elif _owner(card) in (None, mover):
                    self._spent(st, card, "event" if a == EVENT else "ops", a)
                else:                                      # EVENT here is the event-first branch of an Ops play
                    self._opponent_play(st, card, mover, a)
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


#: How a holding ended that counts as a play in an action round, for the space-race share (records
#: before HOLDINGS_VERSION 2 file the opponent's Ops plays as "ops").
_ROUND_PLAYS = ("event", "ops", "space", "fired", "no event")


def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%" if n else "—"


def table(holdings: Sequence[Holding], games: int, source: str = "self-play", opponent: bool = True) -> str:
    """Every card, what each side does with it: for a side's card its owner's holdings and the
    opponent's, for a neutral card each side's and both. `opponent` False for records before
    HOLDINGS_VERSION 2, whose opponent outcomes were filed as the holder's own: those columns n/a."""
    own: Dict[Tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
    opp: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for h in holdings:
        if _owner(h.card) in (None, h.side):
            if h.legal:
                own[(h.card, h.side)][h.outcome] += 1
        else:
            opp[h.card][h.outcome] += 1

    def own_cells(c: collections.Counter) -> List[str]:
        n = sum(c.values())
        if n == 0:
            return ["0", "—", "—", "—", "—"]
        return [f"{n:,}", f"{_pct(c['headline'] + c['event'], n)} (H {_pct(c['headline'], n)})",
                _pct(c["ops"], n), _pct(c["space"], n), _pct(c["kept"], n)]

    def opp_cells(c: collections.Counter) -> List[str]:
        n = sum(c.values())
        if not opponent:
            return [f"{n:,}"] + ["n/a"] * 5
        return [f"{n:,}"] + [_pct(c[k], n) for k in ("fired", "no event", "space", "un", "kept")]

    what = "games of the human ts-replayer corpus" if source == "human" else "greedy self-play games"
    played = {s: [h for h in holdings if h.side == s and h.ar > 0 and h.outcome in _ROUND_PLAYS] for s in (US, USSR)}
    space = ", ".join(f"{name} {_pct(sum(h.outcome == 'space' for h in played[s]), len(played[s]))} of its "
                      f"{len(played[s]):,}" for name, s in (("US", US), ("USSR", USSR)))
    out = [f"What each side does with every card in its hand, {games:,} {what}. One count per *holding* -- the "
           "card entering a hand until it leaves -- so each group of columns adds up to 100% of its holdings.", "",
           "**Owner**, and each side holding a neutral card: the holdings in which the holder could have played the "
           "event at one of its decisions (a headline choice where the event can trigger, or an action-round card "
           "choice where selecting it offers the event). \"Event\": headlined or played as the event in a round, "
           "(H …) the headlines among them. \"Not played\": still in hand at the end, or gone some other way "
           "(discarded, taken by an event).", "",
           "**Opponent**: every holding of the other side's card. \"Fires event\": headlined, or played for Ops in "
           "either order, and the event occurred (a starred card: removed from the game by the play). \"No event\": "
           "played that way when the event could not occur -- Iron Lady on Socialist Governments, North Sea Oil on "
           "OPEC, Our Man in Tehran with no US-controlled Middle East country. \"UN\": UN Intervention's companion, "
           "its Ops without its event.", "",
           f"Space race, of the cards each side played in its action rounds: {space}. Scoring cards (always their "
           "event) and the China Card (no event) are left out."]
    if not opponent:
        out += ["", "These records predate the opponent's outcomes (they filed its plays as the holder's own); "
                    "rerun without `--load` to fill those columns."]
    cards = [c for c in _CARDS if int(c["id"]) not in SCORING and int(c["id"]) != THE_CHINA_CARD]
    for side, owner in (("us", US), ("ussr", USSR)):
        out += ["", f"### {side.upper()} cards", "",
                "| card | Ops | owner: holdings | event (H) | Ops | space | not played "
                "| opponent: holdings | fires event | no event | space | UN | not played |",
                "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in cards:
            cid = int(c["id"])
            if _SIDES[cid] != side or not (own.get((cid, owner)) or opp.get(cid)):
                continue
            out.append(f"| {c['name']} | {c['ops']} | " + " | ".join(own_cells(own[(cid, owner)]) + opp_cells(opp[cid]))
                       + " |")
    out += ["", "### Neutral cards", "",
            "| card | Ops | US: holdings | event (H) | Ops | space | not played "
            "| USSR: holdings | event (H) | Ops | space | not played | both: event | Ops | space |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cards:
        cid = int(c["id"])
        if _SIDES[cid] != "neutral" or not (own.get((cid, US)) or own.get((cid, USSR))):
            continue
        both = own[(cid, US)] + own[(cid, USSR)]
        n = sum(both.values())
        out.append(f"| {c['name']} | {c['ops']} | " + " | ".join(
            own_cells(own[(cid, US)]) + own_cells(own[(cid, USSR)])
            + [_pct(both["headline"] + both["event"], n), _pct(both["ops"], n), _pct(both["space"], n)]) + " |")
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
    if not a.merge:
        require_fresh()
    if a.merge:
        recs: List[Holding] = []
        games = 0
        sources = set()
        current = True
        for f in a.merge:
            with open(f) as fh:
                d = json.load(fh)
            games += int(d["games"])
            sources.add(str(d.get("source", "self-play")))
            current = current and int(d.get("version", 1)) >= HOLDINGS_VERSION
            recs += [load_holding(r) for r in d["holdings"]]
        if len(sources) > 1:
            ap.error(f"--merge mixes human and self-play dumps ({sorted(sources)}); tabulate them apart")
        md = table(recs, games, source=sources.pop(), opponent=current)
        print(md)
        if a.output_md:
            open(a.output_md, "w").write(md + "\n")
        return 0
    if a.human_corpus:
        hs, status = human_holdings(a.workers, a.corpus_limit)
        games = status.get("complete", 0) + status.get("partial", 0)
        print(f"corpus: {status}", file=sys.stderr)
        if a.dump:
            json.dump({"games": games, "source": "human", "status": status, "version": HOLDINGS_VERSION,
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
        json.dump({"games": a.games, "version": HOLDINGS_VERSION, "holdings": [dump_holding(h) for h in hs]},
                  open(a.dump, "w"))
    md = table(hs, a.games)
    print(md)
    if a.output_md:
        open(a.output_md, "w").write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
