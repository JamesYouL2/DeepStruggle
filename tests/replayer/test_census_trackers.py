"""The census trackers on a real human game: what they record is what the decisions say.

`event_play_census.HoldingTracker` and `placement_census.PlacementTracker` are fed the converter's
decisions one at a time. Every holding spent at a decision carries that decision's position and
action, so the record can be checked against the position itself: the position decodes, the action
is legal there, and it concerns the holding's card.
"""

from __future__ import annotations

import os

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_paths import corpus_dir
from tools.scripts.event_play_census import (HoldingTracker, _human_game, feed_corpus_game, load_holding,
                                             state_from_token)
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
