"""The rule oracle: generic probes read the engine correctly, and spots play out."""
import numpy as np
import ts_engine as ts

from ai.eval import rule_oracle as O


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def _uniform(st: ts.GameState) -> np.ndarray:
    from bindings.action_encoder import ActionEncoder
    m = np.asarray(ActionEncoder.get_legal_mask(st)).astype(float)
    return m / m.sum()


def _value(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.zeros(len(obs))


def test_score_value_is_the_engines_scoring() -> None:
    st = ts.GameState()
    ts.Engine.init_game(st, 2)
    us = O.score_value(st, 2, ts.Player.US)
    ussr = O.score_value(st, 2, ts.Player.USSR)
    assert us == -ussr == int(ts.Scoring.evaluate_region(st, ts.Region.EUROPE).net_delta)
    assert O.score_value(st, 38, ts.Player.US) is None


def test_collect_and_play_spots() -> None:
    act = _random_policy(0)
    spots, games = O.collect(act, _uniform, _value, 6, seed=3, rate=1.0, cap=5, envs=4)
    assert games >= 6 and spots
    assert {s["rule"] for s in spots} <= set(O.RULES)
    rows = O.play(act, spots, pairs=2, seed=1)
    assert len(rows) == len(spots) and all(-1 <= r["diff"] <= 1 for r in rows)
    md, summary = O.report(rows, {"model": "x", "pairs": 2, "games": games})
    assert "Rule oracle" in md
