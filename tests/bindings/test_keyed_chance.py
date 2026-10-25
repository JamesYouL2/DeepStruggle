"""Keyed chance's two packings of a situation (bindings/keyed_chance.hpp).

`keyed` is kept bit for bit, so every keyed player measured with it stays one searcher: there a USSR
player field (-1, widened as a signed byte) hides the turn, action round and staged roll. `keyed2`
narrows each player field to its byte first, so every packed field separates situations, and it is
`keyed` wherever no player field is the USSR.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import ts_engine as ts

FIELDS = [("turn", 5, False), ("action_round", 3, False),
          ("decision_type", int(ts.DecisionType.POINT_NODE), True),
          ("pending_roll", int(ts.RollType.COUP), True), ("roll_target", 24, True)]


def _pair(player: int, field: str, value: int, context: bool) -> Tuple[ts.GameState, ts.GameState]:
    """Two positions that differ in `field` alone, every player field `player`."""
    state = ts.GameState()
    ts.Engine.init_game(state, 1)
    saved: Dict[str, Any] = state.to_save_dict()
    saved.update(turn=4, action_round=1, phasing_player=player)
    saved["ctx_stack"][0].update(decision_player=player, decision_type=int(ts.DecisionType.ROLL_DIE),
                                pending_roll=0, roll_actor=player, roll_target=25)
    changed = copy.deepcopy(saved)
    (changed["ctx_stack"][0] if context else changed)[field] = value
    return ts.state_from_save_dict(saved), ts.state_from_save_dict(changed)


@pytest.mark.parametrize("player", [-1, 1])
@pytest.mark.parametrize("field,value,context", FIELDS)
def test_keyed2_separates_every_packed_field(player: int, field: str, value: int, context: bool) -> None:
    a, b = _pair(player, field, value, context)
    assert ts.keyed_key(a, 123, True).rng_state != ts.keyed_key(b, 123, True).rng_state
    assert ts.keyed_key(a, 123, True).rng_state != ts.keyed_key(a, 124, True).rng_state


@pytest.mark.parametrize("field,value,context", FIELDS)
def test_keyed_keeps_its_original_packing(field: str, value: int, context: bool) -> None:
    """The original packing: on the US's side every field separates situations and keyed2 keys alike;
    on the USSR's they collide -- the defect keyed2 removes, kept so keyed results stay comparable."""
    a, b = _pair(1, field, value, context)
    assert ts.keyed_key(a, 123).rng_state != ts.keyed_key(b, 123).rng_state
    assert ts.keyed_key(a, 123).rng_state == ts.keyed_key(a, 123, True).rng_state
    a, b = _pair(-1, field, value, context)
    assert ts.keyed_key(a, 123).rng_state == ts.keyed_key(b, 123).rng_state
    assert ts.keyed_key(a, 123).rng_state != ts.keyed_key(a, 123, True).rng_state


def test_keyed2_che_coups_do_not_reuse_one_die_for_different_targets() -> None:
    ids = {ts.MapData.get_country_info(i)["name"]: i for i in range(84)}
    pairs = []
    for world in range(10):
        state = ts.GameState()
        ts.Engine.init_game(state, 1)
        state.current_phase = ts.Phase.ACTION_ROUND
        state.phasing_player = ts.Player.USSR
        state.turn, state.action_round = 4, 1
        state.ctx().pending_op_card = 107
        for name in ("Colombia", "Sudan"):
            state.set_country(ids[name], 2, 0)
        ts.CardHandlers.trigger_event(state, 107, ts.Player.USSR)
        dice = []
        for name in ("Colombia", "Sudan"):
            ts.CardHandlers.handle_event_step(state, ts.MicroAction(ts.DecisionType.POINT_NODE, ids[name], 0, 0))
            state = ts.keyed_key(state, world, True)
            ts.CardHandlers.handle_event_step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            dice.append(state.to_save_dict()["last_die_roll"])
        pairs.append(dice)
    assert any(a != b for a, b in pairs), pairs


def test_the_rollout_batch_keys_with_the_packing_it_was_given() -> None:
    """keyed2=True reaches the batch (and implies keyed): at a USSR decision its rows are keyed apart
    from keyed's, from the same worlds."""
    state = ts.GameState()
    ts.Engine.init_game(state, 3)
    while ts.Engine.auto_advance_step(state, 1):
        pass
    saved: Dict[str, Any] = state.to_save_dict()
    assert saved["ctx_stack"][saved["ctx_stack_depth"]]["decision_player"] == -1
    cands = [int(a) for a in np.flatnonzero(np.asarray(ts.get_flat_action_mask(state, False)))[:2]]
    rows = []
    for keyed, keyed2 in ((True, False), (False, True), (True, True)):
        b = ts.RolloutBatch(horizon=0, seed=5, keyed=keyed, keyed2=keyed2)
        n = b.add(state, cands, 2)
        rows.append([b.row_state(r).rng_state for r in range(n)])
    assert rows[0] != rows[1] and rows[1] == rows[2]
