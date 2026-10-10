"""The census trackers on a real human game: what they record is what the decisions say.

`event_play_census.HoldingTracker`, `placement_census.PlacementTracker` and `ops_census.OpsTracker` are fed the converter's
decisions one at a time. Every holding spent at a decision carries that decision's position and
action, so the record can be checked against the position itself: the position decodes, the action
is legal there, and it concerns the holding's card.
"""

from __future__ import annotations

import os

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import feed_corpus_game, state_from_token
from tools.lib.corpus_paths import corpus_dir
from tools.scripts.event_play_census import HoldingTracker, _human_game, load_holding
from tools.scripts.ops_census import OpsTracker
from tools.scripts.placement_census import COUNTRY, PlacementTracker

#: a complete game (it converts to the end), used throughout
GAME = os.path.join(str(corpus_dir()), "100.json.gz")


def test_spent_holdings_point_at_their_own_decision() -> None:
    recs, status = _human_game(GAME)
    assert status == "complete"
    hs = [load_holding(r) for r in recs]
    spent = [h for h in hs if h.outcome != "kept"]
    assert len(spent) > 60
    for h in spent:
        st = state_from_token(h.pos)
        assert int(st.turn) == h.turn and int(st.action_round) == h.ar
        assert np.asarray(ActionEncoder.get_legal_mask(st))[h.action]
        ctx = st.ctx()
        if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE:
            assert int(ctx.pending_op_card) == h.card
            assert h.outcome in ("event", "ops", "space")
        else:
            assert ctx.decision_type == ts.DecisionType.SELECT_CARD
            assert int(ts.decode_flat_action(st, h.action).primary_id) == h.card
            assert h.outcome in ("headline", "event")


def test_every_holding_of_a_complete_game_is_accounted_for() -> None:
    t = HoldingTracker()
    assert feed_corpus_game(GAME, t) == "complete"
    assert {h.outcome for h in t.holdings} >= {"headline", "event", "ops"}
    for h in t.holdings:
        # A spent holding knows when; a kept one does not.
        assert (h.turn > 0) == (h.outcome != "kept")
    # A card still in a hand at the end is held by its latest holding, which nothing spent.
    latest = {h.card: h for h in t.holdings}
    for card in t.held:
        assert latest[card].outcome == "kept"


def test_placement_records_only_action_round_ops_into_countries() -> None:
    t = PlacementTracker()
    assert feed_corpus_game(GAME, t) == "complete"
    assert len(t.records) > 50
    for side, turn, cid, held in t.records:
        assert side in (1, -1) and 1 <= turn <= 10 and cid in COUNTRY and held >= 0


class _Hands:
    def __init__(self) -> None:
        self.turns: set = set()
        self.largest = 0

    def observe(self, st: ts.GameState, a: int) -> None:
        self.turns.add(int(st.turn))
        for side in (ts.Player.US, ts.Player.USSR):
            n = sum(1 for c in range(1, 111) if ts.in_hand_of(st.get_card_location(c), side))
            self.largest = max(self.largest, n)


def test_the_turn_a_record_stops_in_never_reaches_a_tracker() -> None:
    # Replay 147's record stops in turn 9, where the converter has only a fragment of the hands:
    # fed that turn, a tracker saw the USSR holding 14 cards at AR1.
    t = _Hands()
    assert feed_corpus_game(os.path.join(str(corpus_dir()), "147.json.gz"), t) == "partial"
    assert max(t.turns) == 8
    assert t.largest <= 10                          # nine cards and the China Card


def test_ops_targets_are_the_placements_plus_coups_and_realignments() -> None:
    """`ops_census.OpsTracker` records every Ops target: its influence records are exactly the
    placement census's, and each record's position decodes to a decision whose legal targets include
    the country chosen."""
    t, p = OpsTracker(positions=True), PlacementTracker()
    assert feed_corpus_game(GAME, t) == "complete"
    feed_corpus_game(GAME, p)
    inf = [(r["side"], r["turn"], r["c"], r["held"]) for r in t.records if r["mode"] == "influence"]
    assert inf == p.records
    assert {r["mode"] for r in t.records} >= {"influence", "coup"}
    for r in t.records:
        st = state_from_token(r["pos"])
        ctx = st.ctx()
        assert ctx.decision_type == ts.DecisionType.POINT_NODE and int(ctx.resolving_card) == 0
        assert int(st.turn) == r["turn"] and int(st.action_round) == r["ar"]
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
        assert r["c"] in {int(ts.decode_flat_action(st, int(a)).primary_id) for a in legal}
