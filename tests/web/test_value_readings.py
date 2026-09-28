"""Which of the critic's two readings the workbench believes.

The value head is trained only on the observation of the side to move, so of the two readings a
replay records for each position -- `v_win_us` from the US observation, `v_win_ussr` from the
USSR one -- only the decider's is calibrated. The page must show each side's own P(win), light
the decider's and dim the other, and plot the decider's alone (as P(US wins)). In a replay the
critic is read *after* the step's action, so the decider is the next step's player; a terminal
position has none. Checked here on the page's own TypeScript (trace_view.ts) under node, on a
replay built to exercise every case -- the browser tests check the same thing on screen.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, cast

from tests.web.js_runner import run_ts


def _critic(v_us: float, v_ussr: float, vp_us: float = 0.0, vp_ussr: float = 0.0) -> Dict[str, Any]:
    return {"v_win_us": v_us, "v_win_ussr": v_ussr, "v_vp_us": vp_us, "v_vp_ussr": vp_ussr,
            "win_residual": v_us + v_ussr, "vp_residual": vp_us + vp_ussr, "at": "after"}


def _step(i: int, player: str, critic: Dict[str, Any] | None, decision_player: str = "US",
          terminal: bool = False, vp: int = 0, utility: float = 0.0) -> Dict[str, Any]:
    step: Dict[str, Any] = {
        "step_index": i, "turn": 1, "ar": 1, "phase": "ACTION_ROUND", "player": player,
        "action": {}, "description": f"step {i}",
        "state_snapshot": {"is_terminal": terminal, "victory_points": vp,
                           "terminal_utility": utility,
                           "decision_context": {"decision_player": decision_player}},
    }
    if critic is not None:
        step["critic"] = critic
    return step


def _run(tmp_path: Any, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"steps": steps}))
    # The entry prints a list (one row per step); run_ts is typed for the usual object.
    return cast(List[Dict[str, Any]], run_ts("value_readings.ts", str(tmp_path), str(case)))


def test_the_calibrated_reading_is_the_next_players(tmp_path: Any) -> None:
    steps = [
        # USSR moves; the US decides next, so the US reading (-0.2) is the calibrated one, and
        # the USSR's +0.99 is the kind of untrained extrapolation that must not be believed.
        _step(0, "USSR", _critic(-0.2, 0.99, vp_us=-0.1, vp_ussr=0.05)),
        # US moves; the USSR decides next.
        _step(1, "US", _critic(0.9, 0.3, vp_us=0.2, vp_ussr=-0.15)),
        # USSR moves again; the last step: no next one, the snapshot's decider stands in.
        _step(2, "USSR", _critic(0.1, -0.4), decision_player="US"),
    ]
    out = _run(tmp_path, steps)

    assert [o["decider"] for o in out] == ["US", "USSR", "US"]
    assert out[0]["calibrated"] == ["US"] and out[0]["uncalibrated"] == ["USSR"]
    assert out[1]["calibrated"] == ["USSR"] and out[1]["uncalibrated"] == ["US"]
    assert out[0]["table_calibrated"] == ["US"] and out[1]["table_calibrated"] == ["USSR"]

    # The calibrated reading as P(US wins): (1 + v_us)/2 when the US decides, 1 - (1 + v_ussr)/2
    # when the USSR does. The bar is filled from it; the untrained reading is only the faint line.
    assert abs(out[0]["p_us"] - 0.4) < 1e-9 and abs(out[0]["bar_p_us"] - 0.4) < 1e-6
    assert abs(out[1]["p_us"] - 0.35) < 1e-9 and abs(out[1]["bar_p_us"] - 0.35) < 1e-6
    assert abs(out[0]["p_us_other"] - (1 - 0.995)) < 1e-9
    assert abs(out[1]["p_us_other"] - 0.95) < 1e-9

    # v_vp is the margin / 20: each side's own margin, and the bar's from the calibrated side.
    assert out[0]["side_vp"] == ["−2.0", "+1.0"]
    assert out[0]["bar_vp"] == "VP USSR +2.0"
    assert out[1]["bar_vp"] == "VP US +3.0"   # the USSR expects to finish 3 VP behind

    # The move chip compares calibrated values: 40% -> 35% -> 55%, not raw v_win_us.
    assert out[0]["chip"] is None
    assert out[1]["chip"] == "ΔP −5%"
    assert out[2]["chip"] == "ΔP +20%"


def test_a_terminal_position_has_no_decider(tmp_path: Any) -> None:
    steps = [
        _step(0, "US", _critic(0.1, -0.1)),
        # The engine leaves a stale decision context on a finished game; it must not count.
        _step(1, "USSR", _critic(-0.9, 0.8), decision_player="USSR", terminal=True, vp=-20,
              utility=-1.0),
    ]
    out = _run(tmp_path, steps)
    assert out[1]["decider"] is None and out[1]["p_us"] is None
    assert out[1]["calibrated"] == [] and sorted(out[1]["uncalibrated"]) == ["US", "USSR"]
    assert out[1]["table_calibrated"] == []
    assert out[1]["bar_p_us"] is None, "a finished game has no calibrated reading to fill a bar"
    assert out[1]["result"] is not None and "USSR won" in out[1]["result"]
    assert out[1]["chip"] is None
