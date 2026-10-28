"""A saved replay must reproduce its game when its actions are re-applied.

test_replay_matches_training.py pins the *stepping contract* -- that the generator drains
chance nodes the way the vectorized runner does. It never re-applies a stored log, so the
property the log exists for ("we can generate the same games for replays as in training")
had no coverage: an engine change could silently make every archived replay unreplayable
and nothing would fail.

The ordering matters and is easy to get wrong. The generator records its snapshot *after*
draining the ROLL_DIE nodes that follow an action, so a reader that compares before
draining sees spurious mismatches -- a coup's die roll has not been applied yet. Drain
after stepping, then compare.

Both tests generate their own replay, so they work in a fresh checkout where data/replays is
empty (it is gitignored): the first retraces the log object the generator returns, the second
the file it saved, read back from disk (the `generated_replay_dir` fixture). They used to read
whatever was in data/replays and skip when it held nothing -- that is, everywhere but one
machine.
"""

import glob
import json
import os
import pathlib
from typing import Any, Dict, List

import ts_engine as ts

from ai.models.coldwar_net import create_coldwar_net
from tools.lib.self_play import generate_self_play_replay
from tests.engine_helpers import resolve_rolls


def _hand(cards: List[Any]) -> List[int]:
    return sorted(c["id"] if isinstance(c, dict) else c for c in cards)


def _digest(d: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of a state a divergence would show up in first."""
    return {
        "vp": d["victory_points"], "defcon": d["defcon"], "turn": d["turn"],
        "ar": d["action_round"], "space": d["space"],
        "us_hand": _hand(d["hands"]["US_cards"]),
        "ussr_hand": _hand(d["hands"]["USSR_cards"]),
    }


def _assert_reproduces(log: Any, label: str) -> None:
    steps = log.get("steps") or []
    assert steps, f"{label}: replay has no steps"
    assert "flat_action_idx" in steps[0].get("action", {}), (
        f"{label}: the log does not record flat_action_idx, so its actions cannot be re-applied"
    )

    state = ts.GameState()
    ts.Engine.init_game(state, log["metadata"]["seed"])

    for step in steps:
        resolve_rolls(state)
        assert not ts.Engine.is_terminal(state), (
            f"{label}: game ended at step {step['step_index']} but the log continues"
        )
        idx = step["action"]["flat_action_idx"]
        assert ts.Engine.try_step_flat(state, idx), (
            f"{label}: engine rejected logged action {idx} ({step['description']}) "
            f"at step {step['step_index']}"
        )
        resolve_rolls(state)

        got, want = _digest(ts.state_to_dict(state)), _digest(step["state_snapshot"])
        assert got == want, (
            f"{label}: diverged at step {step['step_index']} ({step['description']}); "
            f"logged {want}, replayed {got}"
        )


def test_a_freshly_generated_replay_reproduces(tmp_path: pathlib.Path) -> None:
    """Generate through the canonical path, then retrace it from the seed alone."""
    out = str(tmp_path / "roundtrip.tslog.json")
    log, _ = generate_self_play_replay(
        model=create_coldwar_net("cpu"), seed=31337, temperature=1.0,
        game_id="roundtrip", output_path=out, device="cpu", verbose=False,
    )
    assert log["steps"], "generator produced an empty replay"
    _assert_reproduces(log, "freshly generated replay")


def test_saved_replay_reproduces_exactly(generated_replay_dir: str) -> None:
    """A replay as saved to disk -- written, then read back -- retraces under the current engine."""
    paths = sorted(glob.glob(os.path.join(generated_replay_dir, "*.tslog.json")))
    assert paths, f"the replay fixture wrote no .tslog.json into {generated_replay_dir}"
    for path in paths:
        with open(path) as fh:
            log = json.load(fh)
        _assert_reproduces(log, os.path.basename(path))
