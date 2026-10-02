"""Live scoring battlegrounds: regions are classed by their scoring cards, and spots play out."""
import numpy as np
import ts_engine as ts

from ai.eval import scoring_bg as S
from ai.eval.ops_block import country_table


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def test_scoring_cards_per_country() -> None:
    _, _, names = country_table()
    by = dict(zip(names, S.scoring_cards_for()))
    assert by["Thailand"] == (1, 38) and by["Pakistan"] == (1,) and by["Iran"] == (3,)
    assert by["West Germany"] == (2,) and by["Panama"] == (37,) and by["Brazil"] == (81,) and by["Angola"] == (79,)


def test_status_follows_the_card() -> None:
    st = ts.GameState()
    ts.Engine.init_game(st, 3)
    _, _, names = country_table()
    iran = names.index("Iran")
    mover = ts.Player.US
    st.set_card_location(3, ts.CardLocation.DISCARD_PILE)
    assert S.status(st, mover, iran) == S.GROUPS[2]
    st.set_card_location(3, ts.CardLocation.HAND_US_UNKNOWN)
    assert S.status(st, mover, iran) == S.GROUPS[0]
    st.set_card_location(3, ts.CardLocation.HAND_USSR_UNKNOWN)
    assert S.status(st, mover, iran) == S.GROUPS[1]


def test_collect_and_play() -> None:
    act = _random_policy(0)
    sens, spots = S.collect(act, 6, seed=2, envs=4, max_spots=4)
    assert sens
    for ex in spots:
        assert ex["live"] and ex["live_points"] == 0
    rows = S.play(act, spots, pairs=2, seed=1)
    for r in rows:
        assert "policy" in r["scores"] and len(r["scores"]) >= 2
    md, summary = S.report(sens, rows, {"model": "x"})
    assert summary["plays"] == len(sens) and "Live scoring" in md
