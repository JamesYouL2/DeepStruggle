#!/usr/bin/env python3
"""Per-checkpoint behaviour report: individual cards first, then events vs Ops for every card.

One pass of greedy self-play per checkpoint feeds every section:

1. **Star Wars** -- the US's plays of it while ahead in space (headline / Ops / event / space race
   in a round) and the card its event takes from the discard pile, split by who played Star Wars
   (`tools/scripts/star_wars_play.py`).
2. **Five Year Plan played by the USSR** -- the USSR playing the US card from its own hand so the
   US event fires (headline, or a round with the event first or the Ops first; not the space
   race): the action round, split Early War / Mid+Late War, and the USSR hand at that moment.
3. **Aldrich Ames Remix played by the US** -- the same for the US playing the USSR card.
4. **OPEC and Alliance for Progress** -- how often the owner plays the event by the VP it would
   score at that moment (0-2 / 3-4 / 5+), per card choice and per holding.
5. **Soviets Shoot Down KAL-007** -- the US's choices holding it, split by South Korea's control.
6. **Chernobyl** -- when each side plays it (headline / action round / mode), the region the US
   designates, whether the US then places its Ops influence in that region more than usual, and
   US Europe Control wins in games where Chernobyl closed Europe.
7. **UN Intervention** -- the opponent card each side plays with it.
8. **Space race** -- the cards each side sends there most.
9. **Events vs Ops, every card** -- the event census of `tools/scripts/event_play_census.py`, the
   same logic (one count per holding where the owner could have played the event).

    PYTHONPATH=.:build/release python tools/scripts/checkpoint_report.py \\
        --checkpoints <pt> [<pt> ...] --games 16384

Each checkpoint gets `research/log/per_checkpoint/<run>_<step>M.md` (and the raw records in
`data/eval/per_checkpoint/`); the directory's `README.md` index is rewritten to list every report.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.scripts.event_play_census import (EVENT, SCORING, Holding, HoldingTracker, dump_holding,
                                             load_holding, table as census_table)
from tools.scripts.star_wars_play import _tag_retrievals, retrieval_table

US, USSR = int(ts.Player.US), int(ts.Player.USSR)
SPACE = ActionEncoder.PLAY_MODE_OFFSET + 1
CARDS = {int(c["id"]): c for c in json.load(open("rules/cards.json"))}
ID = {c["name"]: i for i, c in CARDS.items()}
STAR_WARS = ID["Star Wars"]
FIVE_YEAR_PLAN = ID["Five Year Plan"]
ALDRICH_AMES = ID["Aldrich Ames Remix"]
KAL_007 = ID["Soviets Shoot Down KAL-007"]
OPEC = ID["OPEC"]
ALLIANCE = ID["Alliance for Progress"]
UN_INTERVENTION = ID["UN Intervention"]
ORTEGA = ID["Ortega Elected in Nicaragua"]
WARSAW_PACT = ID["Warsaw Pact Formed"]
RED_SCARE = ID["Red Scare/Purge"]
CHERNOBYL = ID["Chernobyl"]
#: The engine's `Region` order, which is also Chernobyl's designation (`primary_id`).
REGIONS = ["Europe", "Asia", "Middle East", "Africa", "Central America", "South America"]
_MAP = {c["name"]: c for c in json.load(open("rules/map.json"))["countries"]}
COUNTRY_REGION = {int(c["id"]): REGIONS.index(c["region"]) for c in _MAP.values()}
LATE_WAR = 8
OP_INFLUENCE = int(ts.OpMode.INFLUENCE)
SOUTH_KOREA = int(_MAP["South Korea"]["id"])
CUBA = int(_MAP["Cuba"]["id"])
#: The engine's lists (`trigger_opec`, `trigger_alliance_for_progress`): 1 VP per country controlled.
OPEC_COUNTRIES = [int(_MAP[n]["id"]) for n in
                  ("Egypt", "Iran", "Libya", "Saudi Arabia", "Iraq", "Gulf States", "Venezuela")]
ALLIANCE_COUNTRIES = [int(c["id"]) for c in _MAP.values()
                      if c["battleground"] and c["region"] in ("Central America", "South America")]
REPORT_DIR = "research/log/per_checkpoint"
DATA_DIR = "/workspace/data/eval/per_checkpoint"


def _side(card: int) -> str:
    return str(CARDS[card].get("side", "neutral")).lower()


def _hand(st: ts.GameState, player: Any) -> List[int]:
    return [c for c in range(1, 111) if ts.in_hand_of(st.get_card_location(c), player)]


def _mode(a: int) -> str:
    return "event first" if a == EVENT else "space race" if a == SPACE else "Ops first"


# --------------------------------------------------------------------------------------------
# Observers. Each sees every live env at every step, before the step is taken, with the action
# about to be played; game ids are global across batches.
# --------------------------------------------------------------------------------------------

class Census:
    """The event census: `event_play_census.HoldingTracker`, one per game, so the counting is the
    census tool's own -- each holding's legality and how it ended (headline, event, Ops, space race
    or kept)."""

    def __init__(self) -> None:
        self.holdings: List[Holding] = []
        self.trackers: Dict[int, HoldingTracker] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        self.trackers.setdefault(g, HoldingTracker()).observe(st, a)

    def done(self, g: int) -> None:
        t = self.trackers.pop(g, None)
        if t is not None:
            self.holdings += t.holdings


class StarWars:
    """`star_wars_play.play`'s plays and retrievals."""

    def __init__(self) -> None:
        self.plays: List[Dict[str, Any]] = []
        self.retrievals: List[Dict[str, Any]] = []
        self.pending: Dict[int, Dict[str, Any]] = {}

    def see(self, g: int, st: ts.GameState, a: int, mask: np.ndarray) -> None:
        ctx = st.ctx()
        sel = self.pending.get(g)
        if sel is not None and not (ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
                                    and int(ctx.pending_op_card) == STAR_WARS):
            del self.pending[g]                      # selected at a choice that was not a play
            sel = None
        if (int(ctx.resolving_card) == STAR_WARS and ctx.decision_player == ts.Player.US
                and ctx.decision_type == ts.DecisionType.SELECT_CARD):
            headline = st.current_phase == ts.Phase.HEADLINE
            by_us = (int(st.headline_us_card) == STAR_WARS if headline else st.phasing_player == ts.Player.US)
            self.retrievals.append({
                "game": g, "turn": int(st.turn), "by": "US" if by_us else "USSR",
                "phase": "headline" if headline else "ar", "ar": 0 if headline else int(st.action_round),
                "card": int(ts.decode_flat_action(st, a).primary_id),
                "options": [int(ts.decode_flat_action(st, int(k)).primary_id)
                            for k in np.flatnonzero(mask[:ActionEncoder.PLAY_MODE_OFFSET])]})
        if int(ctx.resolving_card) != 0 or ctx.decision_player != ts.Player.US:
            return
        if (ctx.decision_type == ts.DecisionType.SELECT_CARD and a < ActionEncoder.PLAY_MODE_OFFSET
                and int(ts.decode_flat_action(st, a).primary_id) == STAR_WARS
                and ts.in_hand_of(st.get_card_location(STAR_WARS), ts.Player.US)):
            headline = st.current_phase == ts.Phase.HEADLINE
            rec = {"game": g, "turn": int(st.turn), "ahead": int(st.us_space_track) > int(st.ussr_space_track),
                   "ar": 0 if headline else int(st.action_round), "phase": "headline" if headline else "ar",
                   "space": [int(st.us_space_track), int(st.ussr_space_track)]}
            if headline:
                rec["outcome"] = "headline"
                self.plays.append(rec)
            else:
                self.pending[g] = rec
        elif ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and sel is not None:
            sel["outcome"] = "event" if a == EVENT else "space" if a == SPACE else "ops"
            self.plays.append(sel)
            del self.pending[g]

    def done(self, g: int) -> None:
        self.pending.pop(g, None)


class OpponentCardPlay:
    """`player` playing `card` -- the opponent's card -- from its own hand: a headline, or an
    action-round play followed by the play mode. The event fires unless the mode is the space race.
    Records the turn, the action round (0 = headline) and the player's hand without the card."""

    def __init__(self, card: int, player: Any, features: Optional[Dict[str, Any]] = None) -> None:
        self.card, self.player = card, player
        #: name -> fn(state) -> int, read at the card choice and kept with the play
        self.features: Dict[str, Any] = features or {}
        self.plays: List[Dict[str, Any]] = []
        self.pending: Dict[int, Dict[str, Any]] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        sel = self.pending.get(g)
        if sel is not None:
            del self.pending[g]
            if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and int(ctx.pending_op_card) == self.card:
                sel["mode"] = _mode(a)
                self.plays.append(sel)
            # otherwise it was selected at a choice that was not a play (a forced discard)
        if (int(ctx.resolving_card) != 0 or ctx.decision_player != self.player
                or ctx.decision_type != ts.DecisionType.SELECT_CARD or a >= ActionEncoder.PLAY_MODE_OFFSET):
            return
        if int(ts.decode_flat_action(st, a).primary_id) != self.card:
            return
        if not ts.in_hand_of(st.get_card_location(self.card), self.player):
            return
        headline = st.current_phase == ts.Phase.HEADLINE
        rec: Dict[str, Any] = {"game": g, "turn": int(st.turn), "ar": 0 if headline else int(st.action_round),
                               "hand": [c for c in _hand(st, self.player) if c != self.card],
                               "vp": int(st.victory_points), "defcon": int(st.defcon),
                               "feat": {k: int(fn(st)) for k, fn in self.features.items()}}
        if headline:
            rec["mode"] = "headline"
            self.plays.append(rec)
        else:
            self.pending[g] = rec

    def done(self, g: int) -> None:
        self.pending.pop(g, None)


def _played_by(st: ts.GameState, card: int) -> str:
    """Who played `card`, whose event is resolving now: the headline's owner, or the phasing side."""
    if st.current_phase == ts.Phase.HEADLINE:
        if int(st.headline_us_card) == card:
            return "US"
        if int(st.headline_ussr_card) == card:
            return "USSR"
    return "US" if st.phasing_player == ts.Player.US else "USSR"


class OrtegaResponse:
    """Ortega Elected in Nicaragua's event: the USSR's free coup with the card's Ops in a country
    adjacent to Nicaragua (Cuba, Honduras, Costa Rica), or none. Records who played the card, the
    target, and what followed -- the target's influence and DEFCON at the next decision, or the game
    ending first."""

    def __init__(self) -> None:
        self.responses: List[Dict[str, Any]] = []
        self.waiting: Dict[int, Dict[str, Any]] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        w = self.waiting.pop(g, None)
        if w is not None and w["target"] is not None:
            c = st.get_country(w["target"])
            w.update(us_after=int(c.us_influence), ussr_after=int(c.ussr_influence), defcon_after=int(st.defcon))
        ctx = st.ctx()
        if (int(ctx.resolving_card) != ORTEGA or ctx.decision_player != ts.Player.USSR
                or ctx.decision_type != ts.DecisionType.POINT_NODE):
            return
        cid = int(ts.decode_flat_action(st, a).primary_id)
        target = cid if 0 <= cid < 84 and a != ActionEncoder.CONFIRM_DONE_INDEX else None
        rec: Dict[str, Any] = {"game": g, "turn": int(st.turn), "by": _played_by(st, ORTEGA), "target": target,
                               "defcon": int(st.defcon), "cuba_us": int(st.get_country(CUBA).us_influence),
                               "ended": False}
        if target is not None:
            c = st.get_country(target)
            rec.update(us_before=int(c.us_influence), ussr_before=int(c.ussr_influence))
        self.responses.append(rec)
        self.waiting[g] = rec

    def done(self, g: int) -> None:
        w = self.waiting.pop(g, None)
        if w is not None:
            w["ended"] = True


class WarsawPact:
    """Warsaw Pact Formed's event: the USSR's branch -- 0 removes all US influence from 4 Eastern
    European countries, 1 adds 5 USSR influence there (at most 2 per country) -- with the turn and
    who played the card."""

    def __init__(self) -> None:
        self.choices: List[Dict[str, Any]] = []

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        if int(ctx.resolving_card) != WARSAW_PACT or ctx.decision_type != ts.DecisionType.CHOOSE_BRANCH:
            return
        self.choices.append({"game": g, "turn": int(st.turn), "by": _played_by(st, WARSAW_PACT),
                             "branch": int(ts.decode_flat_action(st, a).primary_id)})

    def done(self, g: int) -> None:
        pass


class KalAftermath:
    """After the USSR plays KAL-007 so that the US event fires (headline, event first or Ops first):
    whether the US got the event's Ops (an Ops choice the event grants only when the US controls
    South Korea as it resolves), South Korea's control at the choice and once the play is over, DEFCON after,
    and whether the game ended before the next card choice. With Ops first the USSR's Ops come
    before the event, so they can break US control first."""

    def __init__(self) -> None:
        self.plays: List[Dict[str, Any]] = []
        self.watch: Dict[int, Dict[str, Any]] = {}
        self.selected: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _sk(st: ts.GameState) -> int:
        return int(ts.Scoring.is_controlled_by(st, SOUTH_KOREA, ts.Player.US))

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        w = self.watch.get(g)
        if w is not None:
            # The event hands the US its Ops as an ordinary Ops choice (SELECT_OP_MODE with KAL-007
            # pending, event_granted_ops set, resolving_card reset to 0) -- only when it controls
            # South Korea as the event resolves.
            if (ctx.decision_player == ts.Player.US and ctx.decision_type == ts.DecisionType.SELECT_OP_MODE
                    and int(ctx.pending_op_card) == KAL_007):
                w["us_ops"] = True
            elif (ctx.decision_type == ts.DecisionType.SELECT_CARD and int(ctx.resolving_card) == 0
                  and not (st.current_phase == ts.Phase.HEADLINE and w["mode"] == "headline")):
                w.update(sk_after=self._sk(st), defcon_after=int(st.defcon), vp_after=int(st.victory_points))
                del self.watch[g]
        sel = self.selected.pop(g, None)
        if sel is not None and ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and int(ctx.pending_op_card) == KAL_007:
            if a != SPACE:
                sel["mode"] = _mode(a)
                self._start(g, sel)
        if (int(ctx.resolving_card) != 0 or ctx.decision_player != ts.Player.USSR
                or ctx.decision_type != ts.DecisionType.SELECT_CARD or a >= ActionEncoder.PLAY_MODE_OFFSET
                or int(ts.decode_flat_action(st, a).primary_id) != KAL_007
                or not ts.in_hand_of(st.get_card_location(KAL_007), ts.Player.USSR)):
            return
        rec: Dict[str, Any] = {"game": g, "turn": int(st.turn), "defcon": int(st.defcon), "sk_us": self._sk(st),
                               "vp": int(st.victory_points), "us_ops": False, "ended": False}
        if st.current_phase == ts.Phase.HEADLINE:
            rec["mode"] = "headline"
            self._start(g, rec)
        else:
            self.selected[g] = rec

    def _start(self, g: int, rec: Dict[str, Any]) -> None:
        self.plays.append(rec)
        self.watch[g] = rec

    def done(self, g: int) -> None:
        w = self.watch.pop(g, None)
        if w is not None:
            w["ended"] = True
        self.selected.pop(g, None)


class OwnCardChoices:
    """Every card choice of `owner` (headline or action round, not inside an event) while `card`
    is in its hand: whether the event could trigger, `feature(st)` (a number that conditions the
    event's worth), and whether the card was played and how -- "headline", or in a round "event",
    "Ops" or "space race". `holding` numbers the spells of the card in the owner's hand."""

    def __init__(self, card: int, owner: Any, feature: Any) -> None:
        self.card, self.owner, self.feature = card, owner, feature
        self.choices: List[Dict[str, Any]] = []
        self.pending: Dict[int, Dict[str, Any]] = {}
        self.held: Dict[int, bool] = {}
        self.spell: Dict[int, int] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        sel = self.pending.pop(g, None)
        if sel is not None:
            if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and int(ctx.pending_op_card) == self.card:
                sel["played"] = "event" if a == EVENT else "space race" if a == SPACE else "Ops"
            else:
                sel["played"] = "not a play"
        holding = ts.in_hand_of(st.get_card_location(self.card), self.owner)
        if holding and not self.held.get(g, False):
            self.spell[g] = self.spell.get(g, 0) + 1
        self.held[g] = holding
        if (not holding or int(ctx.resolving_card) != 0 or ctx.decision_player != self.owner
                or ctx.decision_type != ts.DecisionType.SELECT_CARD
                or st.current_phase not in (ts.Phase.HEADLINE, ts.Phase.ACTION_ROUND)):
            return
        headline = st.current_phase == ts.Phase.HEADLINE
        chosen = a < ActionEncoder.PLAY_MODE_OFFSET and int(ts.decode_flat_action(st, a).primary_id) == self.card
        rec: Dict[str, Any] = {"game": g, "holding": self.spell[g], "turn": int(st.turn),
                               "phase": "headline" if headline else "ar",
                               "legal": bool(ts.CardHandlers.can_trigger_event(st, self.card, self.owner)),
                               "feature": self.feature(st), "played": "headline" if chosen and headline else None}
        self.choices.append(rec)
        if chosen and not headline:
            self.pending[g] = rec

    def done(self, g: int) -> None:
        for d in (self.pending, self.held, self.spell):
            d.pop(g, None)


class SetupTracker:
    """Each side's opening placement: the influence it places at the setup decisions (USSR 6 in
    Eastern Europe, then the US 7 in Western Europe and 2 more where it already has influence),
    per country, recorded once the game leaves the setup phase."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Dict[str, int]]] = []
        self.cur: Dict[int, Dict[str, Dict[str, int]]] = {}
        self.closed: set = set()

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        if g in self.closed:
            return
        if st.current_phase != ts.Phase.SETUP:
            self.done(g)
            return
        ctx = st.ctx()
        if ctx.decision_type != ts.DecisionType.POINT_NODE or int(ctx.resolving_card) != 0:
            return
        cid = int(ts.decode_flat_action(st, a).primary_id)
        if not 0 <= cid < 84:
            return
        side = "US" if ctx.decision_player == ts.Player.US else "USSR"
        rec = self.cur.setdefault(g, {"US": {}, "USSR": {}})
        rec[side][str(cid)] = rec[side].get(str(cid), 0) + 1

    def done(self, g: int) -> None:
        if g in self.closed:
            return
        self.closed.add(g)
        rec = self.cur.pop(g, None)
        if rec is not None:
            self.records.append(rec)


class UNIntervention:
    """The card a side plays with UN Intervention: the opponent card whose Ops it uses."""

    def __init__(self) -> None:
        self.picks: List[Dict[str, Any]] = []

    def see(self, g: int, st: ts.GameState, a: int, mask: np.ndarray) -> None:
        ctx = st.ctx()
        if int(ctx.resolving_card) != UN_INTERVENTION or ctx.decision_type != ts.DecisionType.SELECT_CARD:
            return
        self.picks.append({"game": g, "turn": int(st.turn), "side": "US" if ctx.decision_player == ts.Player.US else "USSR",
                           "card": int(ts.decode_flat_action(st, a).primary_id),
                           "options": [int(ts.decode_flat_action(st, int(k)).primary_id)
                                       for k in np.flatnonzero(mask[:ActionEncoder.PLAY_MODE_OFFSET])]})

    def done(self, g: int) -> None:
        pass


class PlayModes:
    """Every play-mode choice (a card played in an action round, not inside an event): which card,
    which side, and whether it went to the space race -- the denominator for the space-race list."""

    def __init__(self) -> None:
        self.counts: Dict[str, Dict[str, int]] = {"US": {}, "USSR": {}}
        self.space: Dict[str, Dict[str, int]] = {"US": {}, "USSR": {}}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        if (ctx.decision_type != ts.DecisionType.SELECT_PLAY_MODE or int(ctx.resolving_card) != 0
                or st.current_phase != ts.Phase.ACTION_ROUND):
            return
        side = "US" if ctx.decision_player == ts.Player.US else "USSR"
        card = str(int(ctx.pending_op_card))
        self.counts[side][card] = self.counts[side].get(card, 0) + 1
        if a == SPACE:
            self.space[side][card] = self.space[side].get(card, 0) + 1

    def done(self, g: int) -> None:
        pass


def _chernobyl_region(st: ts.GameState) -> int:
    """The region under Chernobyl now (`REGIONS` index), or -1."""
    if not st.has_flag(ts.EffectBits.CHERNOBYL_ACTIVE):
        return -1
    return int((int(st.persistent_effects) & int(ts.EffectBits.CHERNOBYL_REGION_MASK))
               >> int(ts.EffectBits.CHERNOBYL_REGION_SHIFT))


class ChernobylEffect:
    """Chernobyl's event -- the US designates a region, where the USSR may not place influence with
    Ops for the rest of the turn: the region, who played the card (whichever way the event fired)
    and when. And every US influence placement with Ops in the Late War, counted by turn, the
    region under Chernobyl at that moment (-1: none) and the region placed in -- the question being
    whether the US favours the region it has closed to the USSR."""

    def __init__(self, merged: bool) -> None:
        self.merged = merged
        self.regions: List[Dict[str, Any]] = []
        #: game -> "turn|chernobyl region|placed region" -> US influence points placed with Ops
        self.placements: Dict[int, Dict[str, int]] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        if int(ctx.resolving_card) == CHERNOBYL and ctx.decision_type == ts.DecisionType.CHOOSE_BRANCH:
            headline = st.current_phase == ts.Phase.HEADLINE
            self.regions.append({"game": g, "turn": int(st.turn), "ar": 0 if headline else int(st.action_round),
                                 "by": _played_by(st, CHERNOBYL),
                                 "region": int(ts.decode_flat_action(st, a).primary_id)})
            return
        if (ctx.decision_player != ts.Player.US or int(ctx.resolving_card) != 0 or int(st.turn) < LATE_WAR
                or st.current_phase != ts.Phase.ACTION_ROUND):
            return
        if ctx.decision_type == ts.DecisionType.POINT_NODE and int(ctx.op_mode) == OP_INFLUENCE:
            cid = int(ts.decode_flat_action(st, a).primary_id)
        elif self.merged and a < 84 and ts.ActionMask.is_merged_influence_action(st, a):
            cid = a          # E4.1: the op choice "Ops for influence, first point at <a>"
        else:
            return
        if not 0 <= cid < 84:
            return
        key = f"{int(st.turn)}|{_chernobyl_region(st)}|{COUNTRY_REGION[cid]}"
        game = self.placements.setdefault(g, {})
        game[key] = game.get(key, 0) + 1

    def done(self, g: int) -> None:
        pass


def _controlled(st: ts.GameState, countries: Sequence[int], side: Any) -> int:
    return sum(1 for c in countries if ts.Scoring.is_controlled_by(st, c, side))


def play(model: Any, merged: bool, games: int, seed: int, batch: int) -> Dict[str, Any]:
    from bindings.ts_env import TsVectorizedEnv, check_obs_width, model_obs_features
    check_obs_width(model)
    feats = model_obs_features(model)
    width = int(ts.obs_size_for(feats))
    device = next(model.parameters()).device
    model.eval()
    census, sw = Census(), StarWars()
    fyp, ames = OpponentCardPlay(FIVE_YEAR_PLAN, ts.Player.USSR), OpponentCardPlay(ALDRICH_AMES, ts.Player.US)
    kal = OwnCardChoices(KAL_007, ts.Player.US,
                         lambda st: int(ts.Scoring.is_controlled_by(st, SOUTH_KOREA, ts.Player.US)))
    opec = OwnCardChoices(OPEC, ts.Player.USSR, lambda st: _controlled(st, OPEC_COUNTRIES, ts.Player.USSR))
    alliance = OwnCardChoices(ALLIANCE, ts.Player.US, lambda st: _controlled(st, ALLIANCE_COUNTRIES, ts.Player.US))
    un, modes, setup = UNIntervention(), PlayModes(), SetupTracker()
    ortega_us = OpponentCardPlay(ORTEGA, ts.Player.US, {"cuba_us": lambda st: st.get_country(CUBA).us_influence})
    kal_ussr = OpponentCardPlay(KAL_007, ts.Player.USSR,
                                {"sk_us": lambda st: ts.Scoring.is_controlled_by(st, SOUTH_KOREA, ts.Player.US)})
    ortega, warsaw, kal_after = OrtegaResponse(), WarsawPact(), KalAftermath()
    chern_us, chern_ussr = OpponentCardPlay(CHERNOBYL, ts.Player.US), OpponentCardPlay(CHERNOBYL, ts.Player.USSR)
    chern = ChernobylEffect(merged)
    endings: List[Dict[str, Any]] = []
    observers = (census, sw, fyp, ames, kal, opec, alliance, un, modes, setup, ortega_us, kal_ussr, ortega, warsaw,
                 kal_after, chern_us, chern_ussr, chern)
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(feats, feats)
        if merged:
            env.set_merged_influence(True, True)
        obs, masks, _ = env.reset_all()
        done = [False] * n
        for _ in range(20_000):
            if all(done):
                break
            masks_np = np.asarray(masks)
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)[:, :width]).to(device),
                               torch.from_numpy(masks_np).to(device))[0].float()
            actions = logits.argmax(-1).cpu().numpy()
            for i in range(n):
                if done[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                g, a = b0 + i, int(actions[i])
                census.see(g, st, a)
                sw.see(g, st, a, masks_np[i])
                fyp.see(g, st, a)
                ames.see(g, st, a)
                kal.see(g, st, a)
                opec.see(g, st, a)
                alliance.see(g, st, a)
                un.see(g, st, a, masks_np[i])
                modes.see(g, st, a)
                setup.see(g, st, a)
                ortega_us.see(g, st, a)
                kal_ussr.see(g, st, a)
                ortega.see(g, st, a)
                warsaw.see(g, st, a)
                kal_after.see(g, st, a)
                chern_us.see(g, st, a)
                chern_ussr.see(g, st, a)
                chern.see(g, st, a)
            obs, masks, _, dones, info = env.step(actions)
            _collect_endings(info, done, b0, endings)
            for i, d in enumerate(dones):
                if d and not done[i]:
                    done[i] = True
                    for o in observers:
                        o.done(b0 + i)
        for i in range(n):                      # a game still running at the step cap
            if not done[i]:
                for o in observers:
                    o.done(b0 + i)
    return {"games": games, "seed": seed, "batch": batch,
            "holdings": [dump_holding(h) for h in census.holdings],
            "star_wars": {"plays": sw.plays, "retrievals": sw.retrievals},
            "five_year_plan": fyp.plays, "aldrich_ames": ames.plays,
            "kal_007": kal.choices, "opec": opec.choices, "alliance_for_progress": alliance.choices,
            "un_intervention": un.picks, "play_modes": {"all": modes.counts, "space": modes.space},
            "setups": setup.records, "endings": endings,
            "ortega_us": ortega_us.plays, "ortega_response": ortega.responses, "kal_ussr": kal_ussr.plays,
            "warsaw_pact": warsaw.choices, "kal_aftermath": kal_after.plays,
            "chernobyl": {"us_plays": chern_us.plays, "ussr_plays": chern_ussr.plays, "regions": chern.regions,
                          "us_placements": chern.placements}}


def _collect_endings(info: Dict[str, Any], done: List[bool], b0: int, out: List[Dict[str, Any]]) -> None:
    """The env's own record of each game that just finished (`completed_episodes`: the winner, the
    canonical ending reason, the final turn and VP), once per game -- after the first finish the
    env has reset to a new deal, which is not part of the run."""
    for ep in info.get("completed_episodes", []):
        i = int(ep["env_idx"])
        if not done[i]:
            out.append({"game": b0 + i, "winner": str(ep["winner"]), "reason": str(ep["ending_reason"]),
                        "turn": int(ep["turn"]), "vp": int(ep["victory_points"])})


def play_endings(model: Any, merged: bool, games: int, seed: int, batch: int) -> List[Dict[str, Any]]:
    """How the same games as `play` end (same seeds, batches and greedy policy, so the same games),
    without the per-decision bookkeeping -- for adding the endings section to a report built from
    records that predate it."""
    from bindings.ts_env import TsVectorizedEnv, model_obs_features
    feats = model_obs_features(model)
    width = int(ts.obs_size_for(feats))
    device = next(model.parameters()).device
    model.eval()
    endings: List[Dict[str, Any]] = []
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(feats, feats)
        if merged:
            env.set_merged_influence(True, True)
        obs, masks, _ = env.reset_all()
        done = [False] * n
        for _ in range(20_000):
            if all(done):
                break
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)[:, :width]).to(device),
                               torch.from_numpy(np.asarray(masks)).to(device))[0].float()
            actions = logits.argmax(-1).cpu().numpy()
            obs, masks, _, dones, info = env.step(actions)
            _collect_endings(info, done, b0, endings)
            for i, d in enumerate(dones):
                if d:
                    done[i] = True
    return endings


def play_setups(model: Any, merged: bool, games: int, seed: int, batch: int) -> List[Dict[str, Dict[str, int]]]:
    """Only the setup of the same games as `play` (same seeds, batches and greedy policy, so the
    same placements), stopping each batch once every game has left the setup phase -- for adding
    the setup section to a report built from records that predate it."""
    from bindings.ts_env import TsVectorizedEnv, model_obs_features
    feats = model_obs_features(model)
    width = int(ts.obs_size_for(feats))
    device = next(model.parameters()).device
    model.eval()
    setup = SetupTracker()
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(feats, feats)
        if merged:
            env.set_merged_influence(True, True)
        obs, masks, _ = env.reset_all()
        for _ in range(200):
            states = [env.runner.get_state(i) for i in range(n)]
            if all(st.current_phase != ts.Phase.SETUP for st in states):
                break
            masks_np = np.asarray(masks)
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)[:, :width]).to(device),
                               torch.from_numpy(masks_np).to(device))[0].float()
            actions = logits.argmax(-1).cpu().numpy()
            for i, st in enumerate(states):
                setup.see(b0 + i, st, int(actions[i]))
            obs, masks, _, _, _ = env.step(actions)
        for i in range(n):
            setup.see(b0 + i, env.runner.get_state(i), 0)
            setup.done(b0 + i)
    return setup.records


# --------------------------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------------------------

def _pct(k: int, n: int) -> str:
    return "—" if n == 0 else f"{100 * k / n:.1f}%"


def star_wars_section(d: Dict[str, Any]) -> str:
    plays = d["plays"]
    ahead = [p for p in plays if p["ahead"]]
    n = len(ahead)
    k = collections.Counter(p["outcome"] for p in ahead)
    behind = [p for p in plays if not p["ahead"]]
    dud = sum(p["outcome"] == "headline" for p in behind)
    tagged = _tag_retrievals({"plays": plays, "retrievals": d["retrievals"]})
    inside = sum(r["via"] == "inside another event" for r in tagged)
    out = ["## Star Wars", "",
           "Star Wars (US, 2 Ops): if the US is ahead on the Space Race, the US takes a non-scoring card from the "
           "discard pile and plays it as an event. A *play* is the US selecting Star Wars at its own card choice; "
           "\"ahead\" is the US space marker strictly above the USSR's at that choice.", "",
           "| plays while ahead | headline | Ops in a round | event in a round | space race |",
           "|---:|---:|---:|---:|---:|",
           f"| {n:,} | {k['headline']} ({_pct(k['headline'], n)}) | {k['ops']} ({_pct(k['ops'], n)}) | "
           f"{k['event']} ({_pct(k['event'], n)}) | {k['space']} ({_pct(k['space'], n)}) |", "",
           f"Not ahead: {len(behind):,} plays, {dud} of them headlines (the event does nothing).", "",
           "### What Star Wars takes from the discard pile", "",
           "\"Offered in\": the share of retrievals where the card was a legal option; \"picked when offered\": how "
           "often it was taken then."]
    for via, what in (("US play", "the US played Star Wars (headline or event in a round)"),
                      ("USSR play", "the USSR played Star Wars, firing the US event")):
        rs = [r for r in tagged if r["via"] == via]
        out += ["", f"#### {what[0].upper()}{what[1:]}", "", retrieval_table(rs) if rs else "No retrievals."]
    out += ["", f"Not counted: {inside} retrievals where the US event fired inside another event "
                f"(Star Wars handed to the US by an event such as Missile Envy)."]
    return "\n".join(out)


def _era(turn: int) -> str:
    return "Early War" if turn <= 3 else "Mid+Late War"


def opponent_card_section(title: str, intro: str, plays: Sequence[Dict[str, Any]], holder: str,
                          show_mode: bool = True) -> str:
    fired = [p for p in plays if p["mode"] != "space race"]
    eras = ["Early War", "Mid+Late War"]
    by = {e: [p for p in fired if _era(p["turn"]) == e] for e in eras}
    out = [f"## {title}", "", intro, "",
           f"{len(plays):,} plays from hand; {len(fired):,} fire the event "
           f"({sum(p['mode'] == 'space race' for p in plays)} went to the space race and are left out below).", ""]
    if not fired:
        return "\n".join(out + ["No plays."])
    # When: the action round
    ars = sorted({p["ar"] for p in fired})
    out += ["### When (action round; H = headline)", "",
            "| | " + " | ".join(f"{e} (n={len(by[e]):,})" for e in eras) + " |", "|:---|" + "---:|" * len(eras)]
    for ar in ars:
        cells = []
        for e in eras:
            k = sum(p["ar"] == ar for p in by[e])
            cells.append(f"{k} ({_pct(k, len(by[e]))})" if by[e] else "—")
        out.append(f"| {'H' if ar == 0 else f'AR{ar}'} | " + " | ".join(cells) + " |")
    if show_mode:
        out += ["", "| how it was played | " + " | ".join(eras) + " |", "|:---|" + "---:|" * len(eras)]
        for m in ("headline", "event first", "Ops first"):
            out.append(f"| {m} | " + " | ".join(
                f"{_pct(sum(p['mode'] == m for p in by[e]), len(by[e]))}" for e in eras) + " |")
    by_turn = collections.Counter(p["turn"] for p in fired)
    out += ["", "By turn: " + ", ".join(f"T{t} {by_turn[t]}" for t in sorted(by_turn)) + "."]
    # What is in the hand
    opp = "ussr" if holder == "us" else "us"
    out += ["", f"### The {holder.upper()} hand at that moment (without the card played)", "",
            "| | " + " | ".join(eras) + " |", "|:---|" + "---:|" * len(eras)]

    def stat(fn: Any, fmt: str = "{:.2f}") -> str:
        cells = []
        for e in eras:
            xs = [fn(p) for p in by[e]]
            cells.append(fmt.format(float(np.mean(xs))) if xs else "—")
        return " | ".join(cells)

    def frac(p: Dict[str, Any], pred: Any) -> float:
        return sum(1 for c in p["hand"] if pred(c)) / max(1, len(p["hand"]))
    out += [f"| cards in hand (mean) | {stat(lambda p: len(p['hand']))} |",
            f"| last card (hand empty after it) | {stat(lambda p: 100.0 * (len(p['hand']) == 0), '{:.1f}%')} |",
            f"| scoring cards (mean) | {stat(lambda p: sum(c in SCORING for c in p['hand']))} |",
            f"| plays with a scoring card in hand | {stat(lambda p: 100.0 * any(c in SCORING for c in p['hand']), '{:.1f}%')} |",
            f"| {holder.upper()} cards, own events (mean) | {stat(lambda p: sum(_side(c) == holder and c not in SCORING for c in p['hand']))} |",
            f"| {opp.upper()} cards, opponent events (mean) | {stat(lambda p: sum(_side(c) == opp for c in p['hand']))} |",
            f"| neutral non-scoring cards (mean) | {stat(lambda p: sum(_side(c) == 'neutral' and c not in SCORING for c in p['hand']))} |",
            f"| share of hand: scoring | {stat(lambda p: 100 * frac(p, lambda c: c in SCORING), '{:.1f}%')} |",
            f"| share of hand: {opp.upper()} cards | {stat(lambda p: 100 * frac(p, lambda c: _side(c) == opp), '{:.1f}%')} |"]
    for e in eras:
        if not by[e]:
            continue
        cnt = collections.Counter(c for p in by[e] for c in p["hand"])
        out += ["", f"Most frequent cards in hand, {e} (share of plays holding it): " + ", ".join(
            f"{CARDS[c]['name']} {100 * k / len(by[e]):.0f}%" for c, k in cnt.most_common(15)) + "."]
    return "\n".join(out)


VP_BUCKETS = (("0-2 VP", 0, 2), ("3-4 VP", 3, 4), ("5+ VP", 5, 99))


def _bucket(v: int) -> str:
    return next(name for name, lo, hi in VP_BUCKETS if lo <= v <= hi)


def _holding_outcomes(choices: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Per holding (game, spell): its choices where the event could trigger, the best `feature`
    among them, and how the holding ended -- the card's play, if it was played at one of them."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for c in choices:
        h = out.setdefault((c["game"], c["holding"]), {"max": -1, "played": None, "at": None, "legal": False})
        if c["legal"]:
            h["legal"] = True
            h["max"] = max(h["max"], int(c["feature"]))
        if c["played"] in ("headline", "event", "Ops", "space race"):
            h["played"], h["at"] = c["played"], int(c["feature"])
    return out


def vp_card_section(name: str, intro: str, choices: Sequence[Dict[str, Any]]) -> str:
    legal = [c for c in choices if c["legal"]]
    out = [f"### {name}", "", intro, "",
           "At each of the owner's card choices holding the card where the event can trigger, by the VP the event "
           "would score at that moment:", "",
           "| VP the event would score | headline choices | headlined | action-round choices | event in the round "
           "| Ops / space race in the round |", "|:---|---:|---:|---:|---:|---:|"]
    for b, _, _ in VP_BUCKETS:
        hl = [c for c in legal if c["phase"] == "headline" and _bucket(int(c["feature"])) == b]
        ar = [c for c in legal if c["phase"] == "ar" and _bucket(int(c["feature"])) == b]
        out.append(f"| {b} | {len(hl):,} | {_pct(sum(c['played'] == 'headline' for c in hl), len(hl))} | {len(ar):,} | "
                   f"{_pct(sum(c['played'] == 'event' for c in ar), len(ar))} | "
                   f"{_pct(sum(c['played'] in ('Ops', 'space race') for c in ar), len(ar))} |")
    hs = [h for h in _holding_outcomes(choices).values() if h["legal"]]
    out += ["", "Per holding (the card in the owner's hand until it leaves), by the most the event would have scored "
                "at any of those choices -- how the holding ended:", "",
            "| best VP while held | holdings | evented (headline or round) | Ops / space race | not played by the owner |",
            "|:---|---:|---:|---:|---:|"]
    for b, _, _ in VP_BUCKETS:
        xs = [h for h in hs if _bucket(h["max"]) == b]
        out.append(f"| {b} | {len(xs):,} | {_pct(sum(h['played'] in ('headline', 'event') for h in xs), len(xs))} | "
                   f"{_pct(sum(h['played'] in ('Ops', 'space race') for h in xs), len(xs))} | "
                   f"{_pct(sum(h['played'] is None for h in xs), len(xs))} |")
    ev = [h["at"] for h in hs if h["played"] in ("headline", "event")]
    if ev:
        out += ["", f"When evented, the VP it would score at that choice: mean {float(np.mean(ev)):.2f}, "
                    f"5+ in {_pct(sum(v >= 5 for v in ev), len(ev))} of {len(ev):,} plays (a headline resolves after "
                    f"the other side's may have changed the board)."]
    return "\n".join(out)


def kal_section(choices: Sequence[Dict[str, Any]]) -> str:
    out = ["## Soviets Shoot Down KAL-007", "",
           "KAL-007 (US, 4 Ops): DEFCON −1 and the US +2 VP; *if the US controls South Korea* the US may also place "
           "influence or realign with the card's Ops. Each of the US's card choices while holding it, by South "
           "Korea's control at that moment:", "",
           "| | US controls South Korea | does not |", "|:---|---:|---:|"]
    by = {k: [c for c in choices if int(c["feature"]) == k] for k in (1, 0)}
    hl = {k: [c for c in by[k] if c["phase"] == "headline"] for k in by}
    ar = {k: [c for c in by[k] if c["phase"] == "ar"] for k in by}
    out += [f"| headline choices | {len(hl[1]):,} | {len(hl[0]):,} |",
            f"| headlined | {_pct(sum(c['played'] == 'headline' for c in hl[1]), len(hl[1]))} | "
            f"{_pct(sum(c['played'] == 'headline' for c in hl[0]), len(hl[0]))} |",
            f"| action-round choices | {len(ar[1]):,} | {len(ar[0]):,} |"]
    for m, label in (("event", "event in the round"), ("Ops", "Ops in the round"), ("space race", "space race")):
        out.append(f"| {label} | {_pct(sum(c['played'] == m for c in ar[1]), len(ar[1]))} | "
                   f"{_pct(sum(c['played'] == m for c in ar[0]), len(ar[0]))} |")
    return "\n".join(out)


def kal_ussr_section(plays: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]] = (),
                     endings: Sequence[Dict[str, Any]] = ()) -> str:
    """The USSR playing the US card: everything but the space race fires the US event (DEFCON -1,
    US +2 VP, and with South Korea the US's Ops)."""
    out = ["", "### The USSR playing KAL-007", "",
           "The USSR playing the US card from its hand fires the US event unless it goes to the space race. By DEFCON "
           "and South Korea's control when the USSR chose it:", "",
           "| DEFCON | US controls South Korea | plays | headline | event first | Ops first | space race |",
           "|---:|:---|---:|---:|---:|---:|---:|"]
    if not plays:
        return "\n".join(out[:2] + ["No plays."])
    for dc in sorted({int(p["defcon"]) for p in plays}):
        for sk in (1, 0):
            xs = [p for p in plays if int(p["defcon"]) == dc and int(p["feat"].get("sk_us", 0)) == sk]
            if not xs:
                continue
            cells = [_pct(sum(p["mode"] == m for p in xs), len(xs)) for m in ("headline", "event first", "Ops first", "space race")]
            out.append(f"| {dc} | {'yes' if sk else 'no'} | {len(xs):,} | " + " | ".join(cells) + " |")
    if after:
        winner = {int(e["game"]): str(e["winner"]) for e in endings}
        out += ["", "What followed when the event fired -- whether the US got the event's Ops (South Korea still US "
                    "controlled when it resolved), whether the USSR's Ops first broke that control, DEFCON, and the "
                    "game ending before the next card choice:", "",
                "| DEFCON | US controlled South Korea at the choice | how | plays | the US got the Ops | "
                "USSR broke control first | DEFCON fell | the game ended | of those, the USSR won |",
                "|---:|:---|:---|---:|---:|---:|---:|---:|---:|"]
        for dc in sorted({int(p["defcon"]) for p in after}):
            for sk in (1, 0):
                for how in ("headline", "event first", "Ops first"):
                    xs = [p for p in after if int(p["defcon"]) == dc and int(p["sk_us"]) == sk and p["mode"] == how]
                    if not xs:
                        continue
                    ended = [p for p in xs if p["ended"]]
                    broke = (_pct(sum(not p["us_ops"] and p.get("sk_after", 1) == 0 for p in xs), len(xs))
                             if sk and how == "Ops first" else "—")
                    fell = [p for p in xs if "defcon_after" in p]
                    out.append(f"| {dc} | {'yes' if sk else 'no'} | {how} | {len(xs):,} | "
                               f"{_pct(sum(p['us_ops'] for p in xs), len(xs))} | {broke} | "
                               f"{_pct(sum(p['defcon_after'] < p['defcon'] for p in fell), len(fell)) if fell else '—'} | "
                               f"{_pct(len(ended), len(xs))} | "
                               f"{_pct(sum(winner.get(int(p['game'])) == 'USSR' for p in ended), len(ended)) if ended and winner else '—'} |")
    return "\n".join(out)


def ortega_section(plays: Sequence[Dict[str, Any]], responses: Sequence[Dict[str, Any]],
                   endings: Sequence[Dict[str, Any]] = (), retrievals: Sequence[Dict[str, Any]] = ()) -> str:
    out = ["## Ortega Elected in Nicaragua played by the US", "",
           "Ortega Elected in Nicaragua (USSR, 2 Ops): all US influence leaves Nicaragua and the USSR may make a free "
           "coup with the card's Ops in Cuba (a battleground, so a coup there degrades DEFCON), Honduras or Costa "
           "Rica. The US playing it from its hand fires that event unless it goes to the space race. By DEFCON and "
           "whether the US had influence in Cuba when it chose the card:", "",
           "| DEFCON | US influence in Cuba | plays | headline | event first | Ops first | space race |",
           "|---:|:---|---:|---:|---:|---:|---:|"]
    risky = [p for p in plays if int(p["defcon"]) == 2 and int(p["feat"].get("cuba_us", 0)) > 0]
    if risky:
        fired = [p for p in risky if p["mode"] != "space race"]
        out[-2:-2] = [f"**The suicide case** -- DEFCON 2 with US influence in Cuba, where the USSR's free coup in Cuba "
                      f"takes DEFCON to 1 on the US's play: {len(risky):,} US plays, of which {len(fired):,} "
                      f"({_pct(len(fired), len(risky))}) fired the event rather than going to the space race.", ""]
    if not plays:
        out.append("| — | — | 0 | | | | |")
    for dc in sorted({int(p["defcon"]) for p in plays}):
        for cu in (1, 0):
            xs = [p for p in plays if int(p["defcon"]) == dc and (int(p["feat"].get("cuba_us", 0)) > 0) == bool(cu)]
            if not xs:
                continue
            cells = [_pct(sum(p["mode"] == m for p in xs), len(xs)) for m in ("headline", "event first", "Ops first", "space race")]
            out.append(f"| {dc} | {'yes' if cu else 'no'} | {len(xs):,} | " + " | ".join(cells) + " |")
    rs = [r for r in responses if r["by"] == "US"]
    out += ["", "### What the USSR does with it", "",
            f"{len(rs):,} times the event fired from the US's play. The USSR's free coup, by DEFCON and US influence "
            f"in Cuba at that moment:", "",
            "| DEFCON | US influence in Cuba | events | coup Cuba | coup Honduras | coup Costa Rica | no coup |",
            "|---:|:---|---:|---:|---:|---:|---:|"]
    names = {CUBA: "Cuba", int(_MAP["Honduras"]["id"]): "Honduras", int(_MAP["Costa Rica"]["id"]): "Costa Rica"}
    for dc in sorted({int(r["defcon"]) for r in rs}):
        for cu in (1, 0):
            xs = [r for r in rs if int(r["defcon"]) == dc and (int(r["cuba_us"]) > 0) == bool(cu)]
            if not xs:
                continue
            cells = [_pct(sum(r["target"] == c for r in xs), len(xs)) for c in names] + \
                    [_pct(sum(r["target"] is None for r in xs), len(xs))]
            out.append(f"| {dc} | {'yes' if cu else 'no'} | {len(xs):,} | " + " | ".join(cells) + " |")
    coups = [r for r in rs if r["target"] is not None]
    if coups:
        out += ["", "The coups' results (influence and DEFCON at the USSR's next decision):", "",
                "| target | coups | US influence removed | USSR influence gained | DEFCON fell | the game ended | "
                "of those, the USSR won |",
                "|:---|---:|---:|---:|---:|---:|---:|"]
        winner = {int(e["game"]): str(e["winner"]) for e in endings}
        for c, name in names.items():
            xs = [r for r in coups if r["target"] == c]
            if not xs:
                continue
            seen = [r for r in xs if "us_after" in r]
            out.append(f"| {name} | {len(xs):,} | "
                       f"{_pct(sum(r['us_after'] < r['us_before'] for r in seen), len(seen))} | "
                       f"{_pct(sum(r['ussr_after'] > r['ussr_before'] for r in seen), len(seen))} | "
                       f"{_pct(sum(r['defcon_after'] < r['defcon'] for r in seen), len(seen))} | "
                       f"{_pct(sum(r['ended'] for r in xs), len(xs))} | "
                       f"{_pct(sum(winner.get(int(r['game'])) == 'USSR' for r in xs if r['ended']), sum(r['ended'] for r in xs)) if winner else '—'} |")
        ended = [r for r in coups if r["target"] == CUBA and r["ended"] and int(r["defcon"]) == 2]
        if ended:
            by_play: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for p in plays:
                by_play[(int(p["game"]), int(p["turn"]))] = p
            sw = {(int(r["game"]), int(r["turn"])) for r in retrievals if int(r["card"]) == ORTEGA}
            routes: collections.Counter = collections.Counter()
            for r in ended:
                p = by_play.get((int(r["game"]), int(r["turn"])))
                if p is not None:
                    routes[f"from the US hand at DEFCON {p['defcon']}, {p['mode']}"
                           + (", US influence in Cuba" if int(p["feat"].get("cuba_us", 0)) > 0 else ", no US influence in Cuba")] += 1
                elif (int(r["game"]), int(r["turn"])) in sw:
                    routes["taken with Star Wars"] += 1
                else:
                    routes["not from the US hand nor Star Wars (played inside another event, such as Grain Sales)"] += 1
            out += ["", f"A coup in Cuba at DEFCON 2 takes DEFCON to 1 on the US's play of the card, which almost always loses the "
                        f"game for the US (the last column): {len(ended):,} games here ({_pct(len(ended), len(winner) or 1)} of all "
                        f"games). How the US came to play the card (DEFCON when it chose it -- at DEFCON 3 a headline can still "
                        f"resolve at 2, after the other side's headline):", "",
                    "| how the US played Ortega | games lost |", "|:---|---:|"]
            out += [f"| {k} | {v:,} ({_pct(v, len(ended))}) |" for k, v in routes.most_common()]
    own = len(responses) - len(rs)
    if own:
        out += ["", f"(Not counted above: {own:,} events of the USSR's own Ortega plays.)"]
    return "\n".join(out)


def red_scare_section(holdings: Sequence[Holding], retrievals: Sequence[Dict[str, Any]] = ()) -> str:
    """Red Scare/Purge evented, by action round: each side's own plays (the census's holdings that
    ended as a headline or an event) and the US taking it from the discard pile with Star Wars."""
    own = [h for h in holdings if h.card == RED_SCARE and h.outcome in ("headline", "event") and h.turn > 0]
    sw = [r for r in retrievals if r["card"] == RED_SCARE and "ar" in r]
    out = ["## Red Scare/Purge: when it is evented", "",
           "Red Scare/Purge (neutral, 4 Ops): the opponent's cards get −1 Ops for the rest of the turn, so its worth "
           "falls with every action round already gone -- late in the turn it reaches one or two of the opponent's "
           "plays, or none. The action round it is evented in (H = headline), from each side's own plays and from the "
           "US taking it with Star Wars:", ""]
    if not own and not sw:
        return "\n".join(out + ["No timing recorded (records from before the census kept the turn and round)."])
    cols = [("US, own play", [0 if h.outcome == "headline" else h.ar for h in own if h.side == US]),
            ("USSR, own play", [0 if h.outcome == "headline" else h.ar for h in own if h.side == USSR]),
            ("US, with Star Wars", [int(r["ar"]) for r in sw])]
    out += ["| | " + " | ".join(c for c, _ in cols) + " |", "|:---|" + "---:|" * len(cols)]
    for ar in sorted({x for _, xs in cols for x in xs}):
        out.append(f"| {'H' if ar == 0 else f'AR{ar}'} | " + " | ".join(
            f"{sum(x == ar for x in xs):,} ({_pct(sum(x == ar for x in xs), len(xs))})" if xs else "—" for _, xs in cols) + " |")
    out.append("| evented | " + " | ".join(f"{len(xs):,}" for _, xs in cols) + " |")
    out.append("| **AR5 or later** | " + " | ".join(
        f"**{_pct(sum(x >= 5 for x in xs), len(xs))}**" if xs else "—" for _, xs in cols) + " |")
    if not sw and retrievals:
        out += ["", "(The Star Wars column needs records that keep each retrieval's action round.)"]
    return "\n".join(out)


def warsaw_section(choices: Sequence[Dict[str, Any]]) -> str:
    out = ["## Warsaw Pact Formed: the USSR's choice", "",
           "Warsaw Pact Formed (USSR, 3 Ops): the USSR either removes all US influence from 4 Eastern European "
           "countries or adds 5 USSR influence there (at most 2 per country). The branch it takes, by turn and by "
           "who played the card (the US playing it for Ops fires the USSR's event too):", ""]
    if not choices:
        return "\n".join(out + ["No events."])
    out += ["| turn | played by | events | remove US influence | add USSR influence |", "|---:|:---|---:|---:|---:|"]
    for t in sorted({int(c["turn"]) for c in choices}):
        for by in ("USSR", "US"):
            xs = [c for c in choices if int(c["turn"]) == t and c["by"] == by]
            if xs:
                out.append(f"| {t} | {by} | {len(xs):,} | {_pct(sum(c['branch'] == 0 for c in xs), len(xs))} | "
                           f"{_pct(sum(c['branch'] == 1 for c in xs), len(xs))} |")
    return "\n".join(out)


def _ar_name(ar: int) -> str:
    return "headline" if ar == 0 else f"AR{ar}"


def _ar_table(plays: Sequence[Dict[str, Any]]) -> List[str]:
    n = len(plays)
    counts = collections.Counter(int(p["ar"]) for p in plays)
    return ["| when | plays | share |", "|:---|---:|---:|"] + [
        f"| {_ar_name(ar)} | {k:,} | {_pct(k, n)} |" for ar, k in sorted(counts.items())]


def chernobyl_section(c: Dict[str, Any], endings: Sequence[Dict[str, Any]]) -> str:
    out = ["## Chernobyl", "",
           "Chernobyl (US, 3 Ops, Late War): the US designates a region, and for the rest of the turn the USSR "
           "may not place influence there with Ops (coups and realignments stay allowed)."]
    if not c:
        return "\n".join(out + ["", "Not recorded for this report (records from before the section)."])
    us, ussr, regions = c["us_plays"], c["ussr_plays"], c["regions"]
    # 1. The US playing it from its own hand.
    mode_us = {"headline": "headline", "event first": "event in a round", "Ops first": "Ops", "space race": "space race"}
    n = len(us)
    out += ["", f"### Played by the US ({n:,} plays)", ""]
    if n:
        cnt = collections.Counter(p["mode"] for p in us)
        out += ["| how | plays | share |", "|:---|---:|---:|"] + [
            f"| {name} | {cnt[m]:,} | {_pct(cnt[m], n)} |" for m, name in mode_us.items() if cnt[m]]
        ev = [p for p in us if p["mode"] in ("headline", "event first")]
        out += ["", f"The event -- {_pct(len(ev), n)} of the US's plays ({len(ev):,}) -- by when it was played:", ""]
        out += _ar_table(ev) if ev else ["(none)"]
    # 2. The USSR playing it so that the US event fires.
    n = len(ussr)
    out += ["", f"### Played by the USSR ({n:,} plays)", "",
            "The USSR's plays of the card fire the US event unless they go to the space race."]
    if n:
        cnt = collections.Counter(p["mode"] for p in ussr)
        out += ["", "| how | plays | share |", "|:---|---:|---:|"] + [
            f"| {m} | {cnt[m]:,} | {_pct(cnt[m], n)} |" for m in ("headline", "event first", "Ops first", "space race")
            if cnt[m]]
        fired = [p for p in ussr if p["mode"] != "space race"]
        out += ["", f"The event fired ({len(fired):,} plays), by when:", ""]
        out += _ar_table(fired) if fired else ["(none)"]
    # 3. The region the US designates, by whose play fired the event.
    out += ["", f"### The region designated ({len(regions):,} events)", ""]
    if regions:
        by = {w: collections.Counter(r["region"] for r in regions if r["by"] == w) for w in ("US", "USSR")}
        tot = {w: sum(by[w].values()) for w in by}
        allc = collections.Counter(r["region"] for r in regions)
        out += ["Split by whose play fired the event: the card played by that side, or another card that fired it "
                "(a US Star Wars taking it from the discard pile, a USSR Five Year Plan discarding it).", "",
                "| region | fired during a US play | fired during a USSR play | all |", "|:---|---:|---:|---:|"]
        for i in sorted(allc, key=lambda i: -allc[i]):
            out.append(f"| {REGIONS[i]} | {by['US'][i]:,} ({_pct(by['US'][i], tot['US'])}) | "
                       f"{by['USSR'][i]:,} ({_pct(by['USSR'][i], tot['USSR'])}) | {allc[i]:,} ({_pct(allc[i], len(regions))}) |")
    # 4. Where the US places influence with Ops while a region is closed to the USSR.
    # (chernobyl region, placed region) -> points: every game, and the games with a Chernobyl on region r
    pl: Dict[Tuple[int, int], int] = collections.Counter()
    same: Dict[int, Dict[Tuple[int, int], int]] = {r: collections.Counter() for r in range(6)}
    designated: Dict[int, set] = collections.defaultdict(set)
    for rec in regions:
        designated[int(rec["game"])].add(int(rec["region"]))
    for g, keys in c["us_placements"].items():
        for key, k in keys.items():
            _, ch, placed = (int(x) for x in key.split("|"))
            pl[(ch, placed)] += k
            for r in designated.get(int(g), ()):
                same[r][(ch, placed)] += k
    n_free = sum(pl[(-1, x)] for x in range(6))
    out += ["", "### Does the US place influence in the region it closed?", "",
            "Every influence point the US places with Ops in the Late War (turns 8-10, any card; an event's own "
            "placements are not counted). For each region: the share of the US's points that go into it while it "
            "is under Chernobyl, against two baselines with no Chernobyl in play -- every game's Late War, and, "
            "closer, the Late War turns of the same games (those where Chernobyl closed that region at some point), "
            "since the US may close a region because it is already contesting it.", "",
            "| region | US points while it is under Chernobyl | of them into it | no Chernobyl, all games | "
            "no Chernobyl, the same games | difference from the same games |",
            "|:---|---:|---:|---:|---:|---:|"]
    for r in range(6):
        under = sum(pl[(r, x)] for x in range(6))
        if under == 0:
            continue
        sm = sum(same[r][(-1, x)] for x in range(6))
        a, b = pl[(r, r)] / under, pl[(-1, r)] / max(n_free, 1)
        cs = same[r][(-1, r)] / sm if sm else float("nan")
        out.append(f"| {REGIONS[r]} | {under:,} | {pl[(r, r)]:,} ({100 * a:.1f}%) | {100 * b:.1f}% of {n_free:,} | "
                   + (f"{100 * cs:.1f}% of {sm:,} | {100 * (a - cs):+.1f} pt |" if sm else "— | — |"))
    # 5. Europe Control wins against a Chernobyl on Europe.
    end = {int(e["game"]): e for e in endings}
    first: Dict[int, Dict[str, Any]] = {}
    for r in regions:
        g = int(r["game"])
        prev = first.get(g)
        rank = 0 if r["region"] == 0 and r["by"] == "US" else 1 if r["region"] == 0 else 2
        if prev is None or rank < prev["rank"]:
            first[g] = {**r, "rank": rank}
    groups = [("Chernobyl on Europe, from the US's own play", lambda g: g in first and first[g]["rank"] == 0),
              ("Chernobyl on Europe, from the USSR's play", lambda g: g in first and first[g]["rank"] == 1),
              ("Chernobyl on another region", lambda g: g in first and first[g]["rank"] == 2),
              ("no Chernobyl event, the game reached the Late War",
               lambda g: g not in first and end[g]["turn"] >= LATE_WAR)]
    out += ["", "### Europe Control wins and Chernobyl on Europe", "",
            "Games by what Chernobyl did in them (a game with several events counts under the first group it "
            "fits), and how often the US won by controlling Europe -- in that game, and in the turn of the event, "
            "while the USSR could not place influence in Europe.", "",
            "| games | count | US wins | US wins by Europe Control | of them in the event's turn |",
            "|:---|---:|---:|---:|---:|"]
    for name, pred in groups:
        gs = [g for g in end if pred(g)]
        if not gs:
            continue
        wins = [g for g in gs if end[g]["winner"] == "US"]
        euro = [g for g in wins if end[g]["reason"] == "europe_control"]
        in_turn = [g for g in euro if g in first and int(end[g]["turn"]) == int(first[g]["turn"])]
        out.append(f"| {name} | {len(gs):,} | {_pct(len(wins), len(gs))} | {len(euro):,} ({_pct(len(euro), len(gs))}) | "
                   + (f"{len(in_turn):,}" if gs and gs[0] in first else "—") + " |")
    return "\n".join(out)


def _pick_table(recs: Sequence[Dict[str, Any]], top: int = 15) -> str:
    n = len(recs)
    if n == 0:
        return "None."
    picked = collections.Counter(r["card"] for r in recs)
    offered = collections.Counter(c for r in recs for c in r["options"])
    out = [f"{n:,} plays, {sum(len(r['options']) for r in recs) / n:.1f} eligible cards in hand on average.", "",
           "| card | played with it | share | eligible in | played when eligible |", "|:---|---:|---:|---:|---:|"]
    for c, k in picked.most_common(top):
        out.append(f"| {CARDS[c]['name']} | {k:,} | {100 * k / n:.1f}% | {100 * offered[c] / n:.0f}% | "
                   f"{100 * k / offered[c]:.0f}% |")
    return "\n".join(out)


def un_section(picks: Sequence[Dict[str, Any]]) -> str:
    out = ["## UN Intervention: the card played with it", "",
           "UN Intervention is played with an opponent card from the same hand, whose Ops are used without its event. "
           "\"Eligible\": the opponent's non-scoring cards in hand at that choice (the engine's options)."]
    for side in ("US", "USSR"):
        out += ["", f"### {side}", "", _pick_table([p for p in picks if p["side"] == side])]
    return "\n".join(out)


def space_section(modes: Dict[str, Dict[str, Dict[str, int]]], top: int = 15) -> str:
    out = ["## Space race: the cards sent", "",
           "Every card played in an action round goes through the play-mode choice; this counts those sent to the "
           "space race. \"Share of its plays\": of that side's plays of the card in a round, the share spent on the "
           "space race."]
    for side in ("US", "USSR"):
        allp, sp = modes["all"][side], modes["space"][side]
        n, tot = sum(sp.values()), sum(allp.values())
        out += ["", f"### {side}", "", f"{n:,} space race plays, {_pct(n, tot)} of the {tot:,} cards it played in a round.", "",
                "| card | sent to space | share of space plays | share of its plays |", "|:---|---:|---:|---:|"]
        for c, k in sorted(sp.items(), key=lambda kv: -kv[1])[:top]:
            out.append(f"| {CARDS[int(c)]['name']} | {k:,} | {_pct(k, n)} | {_pct(k, allp.get(c, 0))} |")
    return "\n".join(out)


#: Starting influence in the setup regions the placements add to (the engine's initial board).
SETUP_NOTE = ("East Germany starts with 3 USSR influence and the United Kingdom with 5 US, so \"East Germany +1\" "
              "is East Germany at 4.")


def _setup_name(placed: Dict[str, int]) -> str:
    items = sorted(placed.items(), key=lambda kv: (-kv[1], ts.MapData.get_country_name(int(kv[0]))))
    return ", ".join(f"{ts.MapData.get_country_name(int(c))} +{n}" for c, n in items)


def setup_section(records: Sequence[Dict[str, Dict[str, int]]], top: int = 5) -> str:
    out = ["## Opening setups", "",
           "Each side's opening placement -- the influence it places at the setup decisions, by country (USSR 6 "
           "in Eastern Europe, then the US 7 in Western Europe and 2 more where it already has influence). "
           + SETUP_NOTE + " The placement can vary from game to game with the hand dealt and, for the US, with "
           "the USSR's placement."]
    n = len(records)
    for side in ("USSR", "US"):
        cnt = collections.Counter(_setup_name(r[side]) for r in records if r.get(side))
        tot = sum(cnt.values())
        out += ["", f"### {side}", "",
                f"{len(cnt):,} distinct placement{'' if len(cnt) == 1 else 's'} in {tot:,} games; the top {top} cover "
                f"{_pct(sum(k for _, k in cnt.most_common(top)), tot)}.", "",
                "| # | placement | games | share |", "|---:|:---|---:|---:|"]
        for i, (name, k) in enumerate(cnt.most_common(top), 1):
            out.append(f"| {i} | {name} | {k:,} | {_pct(k, tot)} |")
    if n == 0:
        out += ["", "No setups recorded."]
    return "\n".join(out)


#: The env's ending keys (bindings.ts_env.ENDING_REASON_KEYS), named as the tournament reports name them.
#: The DEFCON-1 pair is from the loser's side: its own move took DEFCON to 1 (or it couped under the
#: Cuban Missile Crisis), or the winner's play forced it.
ENDING_NAMES = [("20vp", "20 VP"), ("europe_control", "Europe Control"), ("final_scoring", "final scoring"),
                ("held_scoring", "a scoring card held at turn end"), ("wargames", "Wargames"),
                ("defcon1_self", "DEFCON 1, the loser's own move"),
                ("defcon1_provoked", "DEFCON 1, forced by the winner's play")]


def endings_section(endings: Sequence[Dict[str, Any]]) -> str:
    n = len(endings)
    out = ["## How games end", ""]
    if n == 0:
        return "\n".join(out + ["No endings recorded."])
    by = {w: [e for e in endings if e["winner"] == w] for w in ("US", "USSR", "DRAW")}
    out += [f"{n:,} games: the US wins {_pct(len(by['US']), n)}, the USSR {_pct(len(by['USSR']), n)}, "
            f"drawn {_pct(len(by['DRAW']), n)}. The ending as the engine classifies it (the tournament's "
            f"categories), as a share of each side's wins:", "",
            "| how the game ended | US wins | USSR wins | draws | all games |", "|:---|---:|---:|---:|---:|"]
    known = {k for k, _ in ENDING_NAMES}
    for key, name in ENDING_NAMES + [("other", "other")]:
        def cnt(es: Sequence[Dict[str, Any]]) -> int:
            return sum((e["reason"] == key) if key != "other" else (e["reason"] not in known) for e in es)
        if cnt(endings) == 0:
            continue
        cells = [f"{cnt(by[w]):,} ({_pct(cnt(by[w]), len(by[w]))})" if by[w] else "—" for w in ("US", "USSR", "DRAW")]
        out.append(f"| {name} | " + " | ".join(cells) + f" | {cnt(endings):,} ({_pct(cnt(endings), n)}) |")
    out += ["", "| | US wins | USSR wins |", "|:---|---:|---:|",
            "| mean final turn | " + " | ".join(
                f"{float(np.mean([e['turn'] for e in by[w]])):.1f}" if by[w] else "—" for w in ("US", "USSR")) + " |",
            "| won before final scoring | " + " | ".join(
                _pct(sum(e["reason"] != "final_scoring" for e in by[w]), len(by[w])) if by[w] else "—"
                for w in ("US", "USSR")) + " |"]
    return "\n".join(out)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label(path: str) -> Tuple[str, str, str]:
    """(run short name, step label, file stem) from a checkpoint path."""
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(path)))
    short = re.sub(r"_\d{8}_\d{6}.*$", "", run_dir)
    m = re.search(r"(\d+)steps\.pt$", path)
    step = f"{int(m.group(1)) / 1e6:,.0f}M" if m else os.path.splitext(os.path.basename(path))[0]
    stem = f"{short}_{step.replace(',', '')}" if m else f"{short}_{step}"
    return short, step, stem


def report(path: str, d: Dict[str, Any], merged: bool, feats: int,
           label: Optional[str] = None, note: Optional[str] = None) -> str:
    short, step, _ = _label(path)
    title = label or f"{short} @ {step}"
    meta: Dict[str, Any] = {}
    try:
        meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(path)), "metadata.json")))
    except (OSError, ValueError):
        pass
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    holdings = [load_holding(r) for r in d["holdings"]]
    rel = os.path.relpath(path, "/workspace")
    out = [f"# {title}: behaviour report", "",
           f"Generated {datetime.date.today().isoformat()} by `tools/scripts/checkpoint_report.py` (commit {commit}).", "",
           "## The checkpoint", "",
           f"* **File:** `{rel}` (sha256 `{_sha(path)[:12]}…`)",
           f"* **Run:** {note or meta.get('description', '(no metadata.json description)')}"]
    if meta.get("resumed_from"):
        out.append(f"* **Resumed from:** `{os.path.relpath(str(meta['resumed_from']), '/workspace')}`")
    out += [f"* **Action view:** {'merged influence (E4.1)' if merged else 'E4'}; observation feature bits: {feats}",
            f"* **Games:** {d['games']:,} greedy self-play games of the checkpoint against itself, seed "
            f"{d['seed']:,}, batches of {d['batch']}. Every section reads the same games.", "",
            "Sections: How games end · Opening setups · Star Wars · Five Year Plan played by the USSR · Aldrich Ames "
            "Remix played by the US · Ortega Elected in Nicaragua played by the US · OPEC and Alliance for Progress · "
            "Soviets Shoot Down KAL-007 (both sides) · Red Scare/Purge · Warsaw Pact Formed · Chernobyl · UN Intervention · Space "
            "race · and, last, Events vs Ops for every card.", "",
            endings_section(d.get("endings", [])), "",
            setup_section(d.get("setups", [])), "",
            star_wars_section(d["star_wars"]), "",
            opponent_card_section(
                "Five Year Plan played by the USSR",
                "Five Year Plan (US, 3 Ops): the USSR discards a random card from its hand, and if that card's event "
                "is a US event it fires. Here: the USSR playing it from its own hand so that the event fires -- "
                "headlined, or played in a round with the event first or the Ops first. Early War is turns 1-3.",
                d["five_year_plan"], "ussr"), "",
            opponent_card_section(
                "Aldrich Ames Remix played by the US",
                "Aldrich Ames Remix (USSR, 3 Ops): the US reveals its hand for the rest of the turn and the USSR "
                "discards a card of its choice from it. Here: the US playing it from its own hand so that the event "
                "fires (a Late War card, so all plays fall in Mid+Late War).",
                d["aldrich_ames"], "us", show_mode=False), "",
            ortega_section(d.get("ortega_us", []), d.get("ortega_response", []), d.get("endings", []),
                           d["star_wars"]["retrievals"]), "",
            "## OPEC and Alliance for Progress", "",
            vp_card_section("OPEC (USSR)", "OPEC (USSR, 3 Ops): the USSR gains 1 VP for each of Egypt, Iran, Libya, "
                            "Saudi Arabia, Iraq, the Gulf States and Venezuela it controls (cancelled by North Sea Oil, "
                            "when the event cannot trigger).", d["opec"]), "",
            vp_card_section("Alliance for Progress (US)", "Alliance for Progress (US, 3 Ops): the US gains 1 VP for "
                            "each battleground it controls in Central and South America (Mexico, Panama, Cuba, "
                            "Venezuela, Brazil, Chile, Argentina).", d["alliance_for_progress"]), "",
            kal_section(d["kal_007"]) + kal_ussr_section(d.get("kal_ussr", []), d.get("kal_aftermath", []),
                                                         d.get("endings", [])), "",
            red_scare_section(holdings, d["star_wars"]["retrievals"]), "",
            warsaw_section(d.get("warsaw_pact", [])), "",
            chernobyl_section(d.get("chernobyl", {}), d.get("endings", [])), "",
            un_section(d["un_intervention"]), "",
            space_section(d["play_modes"]), "",
            "## Events vs Ops, every card", "", census_table(holdings, d["games"])]
    return "\n".join(out) + "\n"


def write_index() -> None:
    rows = []
    for f in sorted(os.listdir(REPORT_DIR)):
        if f.endswith(".md") and f != "README.md":
            first = open(os.path.join(REPORT_DIR, f)).readline().lstrip("# ").strip()
            rows.append(f"* [{first}]({f})")
    open(os.path.join(REPORT_DIR, "README.md"), "w").write(
        "# Per-checkpoint behaviour reports\n\nGenerated by `tools/scripts/checkpoint_report.py`: for one "
        "checkpoint, greedy self-play and from it the event census (events vs Ops), Star Wars plays and "
        "retrievals, and when and with what hand the USSR plays Five Year Plan and the US plays Aldrich Ames "
        "Remix.\n\n" + "\n".join(rows) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from bindings.ts_env import model_obs_features
    from tools.lib.action_view import checkpoint_merged_influence
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=55_000)
    ap.add_argument("--load", action="store_true", help="rebuild the reports from the saved records")
    ap.add_argument("--output-dir", default=REPORT_DIR)
    ap.add_argument("--data-dir", default=DATA_DIR, help="where the raw records go (and --load reads them)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="a name per checkpoint, for the title and the file name (default: run @ step); "
                         "needed for a checkpoint outside a run directory, such as an SWA or a soup")
    ap.add_argument("--notes", nargs="+", default=None,
                    help="a description per checkpoint, in place of its run's metadata.json description")
    a = ap.parse_args(argv)
    if a.labels and len(a.labels) != len(a.checkpoints) or a.notes and len(a.notes) != len(a.checkpoints):
        ap.error("--labels and --notes need one entry per checkpoint")
    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(a.data_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k, path in enumerate(a.checkpoints):
        label = a.labels[k] if a.labels else None
        note = a.notes[k] if a.notes else None
        stem = re.sub(r"[^A-Za-z0-9.+-]+", "_", label).strip("_") if label else _label(path)[2]
        dump = os.path.join(a.data_dir, f"{stem}.json")
        merged = checkpoint_merged_influence(path)
        model = NeuralAgent.from_checkpoint(path, device=str(dev)).model
        feats = model_obs_features(model)
        if a.load:
            d = json.load(open(dump))
            if "setups" not in d:            # records from before the setup section: replay the setups
                d["setups"] = play_setups(model, merged, int(d["games"]), int(d["seed"]), int(d["batch"]))
                json.dump(d, open(dump, "w"))
            if "endings" not in d:           # ... and before the endings section: replay the games
                d["endings"] = play_endings(model, merged, int(d["games"]), int(d["seed"]), int(d["batch"]))
                json.dump(d, open(dump, "w"))
        else:
            d = play(model, merged, a.games, a.seed, a.batch)
            json.dump(d, open(dump, "w"))
        md = report(path, d, merged, feats, label, note)
        out = os.path.join(a.output_dir, f"{stem}.md")
        open(out, "w").write(md)
        print(f"{path} -> {out}", flush=True)
    if os.path.abspath(a.output_dir) == os.path.abspath(REPORT_DIR):
        write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
