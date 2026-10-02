"""The playout audit: branches are scored from the deciding player's side, and regret is unbiased."""
import numpy as np
import ts_engine as ts

from ai.eval import playout_audit as A
from ai.eval.reply_probe import _ar_key


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def _uniform(st: ts.GameState) -> np.ndarray:
    return np.full(220, 1.0 / 220)


def test_split_half_regret_does_not_reward_noise() -> None:
    rng = np.random.default_rng(0)
    # Four identical branches: picking the best on the same pairs would always show a gain.
    vals = [A.split_half_regret(rng.integers(0, 2, size=(64, 4)).astype(float), 0) for _ in range(400)]
    assert abs(float(np.mean(vals))) < 0.02
    sc = np.zeros((8, 2))
    sc[:, 1] = 1.0
    assert A.split_half_regret(sc, 0) == 1.0


def test_score_is_from_the_side_given_not_the_side_to_move() -> None:
    act = _random_policy(1)
    pos = A.collect(act, 4, seed=3, envs=4, accept=0.3)
    st = pos[0][0]
    key = _ar_key(st)
    # The same playout (same dice, same random stream) labelled for each side mirrors exactly.
    one = A.play_safe([st.clone()], [ts.Player.US], [key], _random_policy(9), seed=5)[0]
    two = A.play_safe([st.clone()], [ts.Player.USSR], [key], _random_policy(9), seed=5)[0]
    assert one + two == 1.0


def test_audit_rows() -> None:
    act = _random_policy(2)
    pos = A.collect(act, 6, seed=4, envs=4, accept=0.3)
    assert pos and {k for _, k in pos} <= set(A.KINDS)
    rows = A.audit(act, _uniform, pos, pairs=4, seed=1)
    for r in rows:
        assert r["chosen"] in r["scores"] and len(r["scores"]) >= 2
        assert all(0.0 <= v <= 1.0 for v in r["scores"].values())
    md, summary = A.report(rows, {"model": "x", "pairs": 4})
    assert summary["n"] == len(rows) and "Playout audit" in md
