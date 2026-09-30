"""tools/play_match.py plays any tournament agent (`agent:<spec>`, `roundsearch:`, `safe:`,
`ensemble:`), so a tournament entrant's games can be replayed move by move."""
from __future__ import annotations

from tools.play_match import resolve_agent, run_match


def test_an_agent_spec_resolves_to_a_state_reading_bot() -> None:
    bot, label = resolve_agent("agent:heuristic_v2", "US")
    assert getattr(bot, "wants_game_state", False)
    assert label == bot.name


def test_an_agent_spec_plays_a_whole_recorded_game(tmp_path) -> None:
    log, _ = run_match("agent:heuristic_v2", "safe:random", seed=3,
                       output_path=str(tmp_path / "g.tslog.json"), verbose=False)
    result = log["metadata"]["result"]
    assert result is not None and result["winner"] in ("US", "USSR", "DRAW")
    assert log["steps"], "the replay records every move"
