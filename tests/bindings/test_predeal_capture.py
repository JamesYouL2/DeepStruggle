"""P33's pre-deal critic: the batch runner records each env's turn-end park -- after a turn's last action,
before the deal (`StateMachine::begin_turn_cleanup`) -- from both sides, without changing the decision
stream, and the observations it records are that position's."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

import ts_engine as ts

_STEPS = 900


def _actions(masks: np.ndarray, rng: np.random.Generator) -> List[int]:
    return [int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks]


def _at_park(s: ts.GameState) -> bool:
    c = s.ctx()
    return (int(c.decision_player) == 0 and c.decision_type == ts.DecisionType.ROLL_DIE
            and c.pending_roll_type == ts.RollType.TURN_CLEANUP)


def test_capture_leaves_the_decision_stream_unchanged_and_records_the_park() -> None:
    plain, capt = ts.VectorizedBatchRunner(8, 31), ts.VectorizedBatchRunner(8, 31)
    capt.set_capture_predeal(True)
    rng = np.random.default_rng(5)
    parks = 0
    for _ in range(_STEPS):
        masks = np.asarray(plain.get_action_masks())
        assert np.array_equal(masks, np.asarray(capt.get_action_masks()))
        assert np.array_equal(np.asarray(plain.get_observations()), np.asarray(capt.get_observations()))
        acts = _actions(masks, rng)
        before = [capt.get_state(i) for i in range(8)]
        plain.step_flat_all(acts, True)
        capt.step_flat_all(acts, True)
        flags, obs_us, obs_ussr = (np.array(x) for x in capt.get_predeal())
        for i in np.flatnonzero(flags):
            # Rebuild the park independently: the same action, then one chance node at a time.
            s = before[i]
            ts.Engine.step_flat(s, acts[i], False)
            for _k in range(256):
                if _at_park(s):
                    break
                assert ts.Engine.auto_advance_step(s, 1) == 1
            assert _at_park(s)
            assert np.array_equal(obs_us[i, :ts.OBS_SIZE], np.asarray(ts.extract_observation(s, ts.Player.US)))
            assert np.array_equal(obs_ussr[i, :ts.OBS_SIZE], np.asarray(ts.extract_observation(s, ts.Player.USSR)))
            parks += 1
        for i, t in enumerate(np.asarray(capt.get_terminals())):
            if t:
                plain.reset_game(i, 1000 + i)
                capt.reset_game(i, 1000 + i)
    assert parks >= 8                                  # every env crossed at least one turn end


def test_get_predeal_needs_capture_on() -> None:
    with pytest.raises(RuntimeError):
        ts.VectorizedBatchRunner(2, 3).get_predeal()
