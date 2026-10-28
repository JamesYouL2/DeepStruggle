"""Engine helpers shared by several test files (import them; they are not fixtures)."""

from __future__ import annotations

import ts_engine as ts


def resolve_rolls(state: ts.GameState) -> None:
    """Resolve the pending die rolls, as every reader of the replay format and the batch runner
    do: a ROLL_DIE node nobody decides is stepped with a random roll until a decision or the end.

    Written out rather than borrowed from `ai.search.pimcts.drain_chance_nodes`, because the tests
    that use it pin that contract -- they must not inherit a change to the code they check.
    """
    while (not ts.Engine.is_terminal(state)
           and state.ctx().decision_player == ts.Player.NONE
           and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
        ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
