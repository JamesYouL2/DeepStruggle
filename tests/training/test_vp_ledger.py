"""The VP ledger: VP are credited to the right source, and the books balance."""
import numpy as np
import ts_engine as ts

from ai.eval import vp_ledger as L


def _random(seed: int):
    rng = np.random.default_rng(seed)

    def policy(obs: np.ndarray, masks: np.ndarray):
        acts = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int64)
        return acts, rng.uniform(-1, 1, size=len(masks))
    return policy


def test_credit_splits_the_shortfall_off_a_turn_ending_play() -> None:
    assert L.credit("other", 0, -2, 3, 4, False, -2) == [("turn end: Military Ops", -2)]
    assert L.credit("scoring: Europe", 0, 5, 3, 4, False, -2) == [("turn end: Military Ops", -2), ("scoring: Europe", 7)]
    assert L.credit("scoring: Europe", 0, 5, 3, 3, False, -2) == [("scoring: Europe", 5)]
    assert L.credit("other", 1, 4, 10, 11, True, 0) == [("final scoring", 3)]
    assert L.credit("space race", 1, 4, 10, 11, True, 0) == [("mixed", 3)]


def test_shortfall_matches_the_engine() -> None:
    st = ts.GameState()
    ts.Engine.init_game(st, 1)
    st.defcon, st.us_mil_ops, st.ussr_mil_ops, st.victory_points = 4, 1, 4, 0
    want = L.shortfall(st)
    ts.Scoring.evaluate_military_ops(st)
    assert want == -3 and int(st.victory_points) == want


def test_the_books_balance_in_self_play() -> None:
    rows, cal = L.selfplay(_random(0), 6, seed=2, envs=4)
    assert len(rows) >= 6
    for r in rows:
        net = sum(us - ussr for us, ussr in r["gains"].values())
        assert net == r["vp"]                    # every VP change is credited somewhere, once
    assert sum(b[0] for bins in cal["bins"].values() for b in bins) > 0
    md, summary = L.report(rows, [], L.merge_cal([cal]), {"model": "x"})
    assert "VP ledger" in md and summary["bot_games"] == len(rows)
