"""The reply probe: both branches play exactly the mover's round, and the measures read it right."""
import numpy as np
import ts_engine as ts

from ai.eval import reply_probe as R
from bindings.action_encoder import ActionEncoder


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


class _LastLegal:
    """A stand-in searcher: the highest legal action, so it disagrees with the policy often."""

    def select_actions_batch(self, states):
        return [int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))[-1]) for s in states]


def test_rounds_stop_at_the_end_of_the_movers_round() -> None:
    act = _random_policy(0)
    starts = R.collect(act, 6, seed=2, envs=4, accept=0.5)
    assert len(starts) == 6 and all(R.is_round_start(s) for s in starts)
    rounds = R.play_rounds(starts, act, _LastLegal(), seed=1)
    for r in rounds:
        key = R._ar_key(r["start"])
        for b in R.BRANCHES:
            end = r[b]["end"]
            assert ts.Engine.is_terminal(end) or R._ar_key(end) != key
            assert r[b]["moves"] and r[b]["moves"][0][0] < 110        # it starts with the card
    differ = [r for r in rounds if [m for m, _ in r["net"]["moves"]] != [m for m, _ in r["search"]["moves"]]]
    assert differ
    rows = R.summarise(differ, R.play_replies(differ, act, pairs=2, seed=1))
    for row in rows:
        for b in R.BRANCHES:
            assert 0.0 <= row[b]["score"] <= 1.0 and row[b]["placed"] >= row[b]["into_own"]
        assert row["link"].startswith(R.WORKBENCH)
    md, summary = R.report({"rounds": len(rounds), "rows": rows, "all": [[0, True]] * len(rounds)}, {"model": "x"})
    assert summary["disagreements"] == len(rows) and "Reply probe" in md


def test_coup_break_and_rounds_left() -> None:
    # 3 Ops against a 2-stability country held 2-0: x = d - 1, so d >= 2 removes 1+ and breaks.
    assert R._coup_break(3, 2, 2, 0) == 5 / 6
    assert R._coup_break(3, 3, 5, 0) == 1 / 6                   # x = d - 3: only a 6 takes 5 down to 2
    assert R._coup_break(3, 3, 6, 0) == 0.0
    st = ts.GameState()
    ts.Engine.init_game(st, 1)
    st.turn, st.action_round = 3, 6
    st.phasing_player = ts.Player.US
    assert R.rounds_left(st) == 0 and not R.opponent_replies(st)
    st.phasing_player = ts.Player.USSR
    assert R.opponent_replies(st)
    st.turn = 4
    assert R.rounds_left(st) == 1
