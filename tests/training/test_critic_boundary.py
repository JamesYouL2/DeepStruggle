"""ai/eval/critic_boundary: hand-over sums and clustered errors on hand-made games."""

from __future__ import annotations

from ai.eval.critic_boundary import Decision, Game, _clustered, handovers


def _game(values: list[tuple[int, float]], us_util: float = 1.0) -> Game:
    return Game(decisions=[Decision(side, v, 1, "ACTION_ROUND") for side, v in values], us_util=us_util)


def test_a_hand_over_is_a_change_of_mover_and_sums_both_views() -> None:
    g = _game([(1, 0.2), (1, 0.3), (-1, -0.25), (-1, -0.2), (1, 0.1)])
    rows = handovers([g])
    assert [r["dir"] for r in rows] == ["US->USSR", "USSR->US"]
    assert [round(r["sum"], 6) for r in rows] == [0.05, -0.1]
    # each side's error against the result, in its own frame (US won: +1 for US, -1 for USSR)
    assert round(rows[0]["err_a"], 6) == -0.7 and round(rows[0]["err_b"], 6) == 0.75


def test_the_standard_error_is_clustered_by_game() -> None:
    # two games, each contributing identical values: the SE reflects 2 clusters, not 6 samples
    m, se, n = _clustered([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], [0, 0, 0, 1, 1, 1])
    assert m == 0.0 and n == 6
    assert abs(se - (3 * 2 ** 0.5) / 6) < 1e-9
