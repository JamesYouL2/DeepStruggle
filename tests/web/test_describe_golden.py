"""The page's action log says what the Python session's said.

The workbench's game session moved from web/server/session.py into the browser
(web/ui/src/game/). `describe_action_and_deltas` -- the text of every log line -- was ported line
for line, and fixtures/describe_golden.json.gz holds what the Python original wrote for 12 whole
games (1,103 steps, every kind of line: plays, headlines, coups, realignments, space race, event
targets, influence, DEFCON/VP/effect/card deltas, dice). The TypeScript port replays those games
through the WebAssembly engine and must write the same lines.

The games are the current engine's: a rule change that makes a recorded action illegal fails
this test with "engine refused", and the fix is to re-record them with
fixtures/make_describe_golden.py, which runs the Python original from the repository's history.

The one intended difference -- the port logs MilOps changes, which the original never did -- is
filtered out in the runner.
"""
from __future__ import annotations

import os

from tests.web.js_runner import PUBLIC, run_ts

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "describe_golden.json.gz")


def test_the_typescript_action_log_matches_the_python_session(tmp_path) -> None:
    result = run_ts("describe_golden.ts", str(tmp_path), PUBLIC, GOLDEN)
    assert result["steps"] == 1103
    assert result["mismatches"] == 0, result["examples"]
