"""Which dice the map's roll tray shows, and how it shows them.

A roll stays on the map for the half action round it was made in and the next one (the other
side's reply) -- across a turn's end too: the turn's last action round and the next turn's
headline are as adjacent as any two half rounds. The live session's steps carry every die their
click rolled (`ReplayStep.rolls`): one click can roll more than once, and the turn's cleanup clears
the roll record from the state the click ends on. Checked here on the page's own TypeScript
(game/rolls.ts, roll_tray.ts, and the log's dice lines in game/describe.ts) under node, on steps built for each case.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from tests.web.js_runner import run_ts


def _state(turn: int, phase: str, ar: int, phasing: str) -> Dict[str, Any]:
    return {"turn": turn, "current_phase_name": phase, "action_round": ar, "phasing_player": phasing}


def _roll(kind: str, roller: str, roll1: int = 3) -> Dict[str, Any]:
    return {"type": kind, "roller": roller, "country_id": 20, "country_name": "Iran", "roll1": roll1}


def _step(i: int, turn: int, phase: str, ar: int, player: str, after: Dict[str, Any],
          rolls: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    step: Dict[str, Any] = {"step_index": i, "turn": turn, "ar": ar, "phase": phase, "player": player,
                            "action": {}, "description": f"step {i}", "state_snapshot": after}
    if rolls is not None:
        step["rolls"] = rolls
    return step


def _run(tmp_path: Any, start: Dict[str, Any], steps: List[Dict[str, Any]], now: Dict[str, Any],
         tray: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"start": start, "steps": steps, "now": now, "tray": tray or []}))
    return run_ts("recent_rolls.ts", str(tmp_path), str(case))


#: The US's coup in turn 1's last action round, whose click carried into turn 2's headline: the
#: state it ends on has no roll left (the cleanup cleared it), the step's own rolls have the coup.
_AR7_US = _state(1, "ACTION_ROUND", 7, "US")
_HEADLINE_2 = _state(2, "HEADLINE", 0, "NONE")
_LAST_COUP = _step(1, 1, "ACTION_ROUND", 7, "US", {**_HEADLINE_2, "die_roll": {"type": "NONE"}},
                   [_roll("COUP", "US")])


def test_the_turns_last_coup_is_shown_in_the_next_headline(tmp_path: Any) -> None:
    out = _run(tmp_path, _AR7_US, [_LAST_COUP], _HEADLINE_2)
    assert [(r["type"], r["when"]) for r in out["recent"]] == [("COUP", "T1 AR7")]


def test_it_is_gone_a_half_round_later(tmp_path: Any) -> None:
    ar1 = _state(2, "ACTION_ROUND", 1, "USSR")
    headline = _step(2, 2, "HEADLINE", 0, "USSR", ar1, [])
    assert _run(tmp_path, _AR7_US, [_LAST_COUP, headline], ar1)["recent"] == []


def test_every_die_of_one_click_is_shown(tmp_path: Any) -> None:
    """A realignment whose card is the opponent's war: the realignment rolls, then the war."""
    ar = _state(3, "ACTION_ROUND", 2, "US")
    after = {**ar, "die_roll": _roll("WAR_EVENT", "USSR")}
    step = _step(1, 3, "ACTION_ROUND", 2, "US", after, [_roll("REALIGNMENT", "US"), _roll("WAR_EVENT", "USSR")])
    assert [r["type"] for r in _run(tmp_path, ar, [step], ar)["recent"]] == ["REALIGNMENT", "WAR_EVENT"]


def test_a_replay_files_step_shows_its_snapshots_roll(tmp_path: Any) -> None:
    ar = _state(3, "ACTION_ROUND", 2, "US")
    step = _step(1, 3, "ACTION_ROUND", 2, "US", {**ar, "die_roll": _roll("COUP", "US")}, None)
    assert [r["type"] for r in _run(tmp_path, ar, [step], ar)["recent"]] == ["COUP"]


def test_the_olympic_sponsors_die_is_the_die(tmp_path: Any) -> None:
    """The engine records the sponsor's roll with its +2 in (card_dispatcher.cpp): a 5 is roll1 7."""
    olympics = {"type": "OLYMPIC_GAMES", "roller": "US", "country_id": 255, "roll1": 7, "mod1": 2,
                "total1": 9, "roll2": 4, "mod2": 0, "success": True}
    ar = _state(1, "ACTION_ROUND", 1, "US")
    out = _run(tmp_path, ar, [], ar, [olympics])
    view = out["tray"][0]
    assert 'aria-label="US rolls 5"' in view["dice"]
    assert "<b>7</b>" in view["dice"] and "<b>9</b>" not in view["dice"]
    assert out["log"][0] == ["  🎲 Olympic Games Competition: Sponsor (US) rolls 5 (+2 = 7), "
                             "Opponent (USSR) rolls 4 -> Sponsor US wins by +3 (+2 VP)"]
