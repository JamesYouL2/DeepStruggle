"""Re-record fixtures/describe_golden.json.gz: what the Python session's action log said.

The golden is the Python original of `describe_action_and_deltas` -- deleted from the tree when
the session moved into the browser -- run over whole games of the *current* engine, for the
TypeScript port to reproduce (tests/web/test_describe_golden.py). An engine change that alters
the decision stream leaves the recorded actions illegal; re-record then:

    PYTHONPATH=.:build/release .venv/bin/python tests/web/fixtures/make_describe_golden.py

The function is taken from git at SOURCE_COMMIT, the last commit that has the Python session,
and executed on its own: it needs only the engine's card and map data. Games are seeded random
legal play, stepped exactly as the page's engine steps them (tools.lib.game_step.drain_chance
resolves the dice from the state's RNG, as bindings/wasm/ts_engine_wasm.cpp does).
"""
from __future__ import annotations

import gzip
import json
import os
import random
import subprocess
from typing import Any, Callable, Dict, List

import numpy as np

import ts_engine as ts
from tools.lib.game_step import drain_chance, step_checked

SOURCE_COMMIT = "649c21aacf204d37ddf0432c8dbb70dce5c25ab0"
SEEDS = range(101, 113)
MAX_STEPS = 170
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "describe_golden.json.gz")


def _python_describe() -> Callable[[Dict[str, Any], Dict[str, Any], Any], List[str]]:
    src = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:web/server/session.py"],
                         capture_output=True, text=True, check=True).stdout
    start = src.index("def describe_action_and_deltas(")
    end = src.index("\nclass GameSession")
    namespace: Dict[str, Any] = {"ts_engine": ts, "List": List, "Any": Any, "Dict": Dict}
    exec(compile(src[start:end], f"{SOURCE_COMMIT}:web/server/session.py", "exec"), namespace)
    return namespace["describe_action_and_deltas"]


def main() -> None:
    describe = _python_describe()
    games = []
    for seed in SEEDS:
        s = ts.GameState()
        ts.Engine.init_game(s, seed)
        drain_chance(s)
        rng = random.Random(seed)
        actions, lines = [], []
        while not ts.Engine.is_terminal(s) and len(actions) < MAX_STEPS:
            legal = np.nonzero(ts.get_flat_action_mask(s, False))[0]
            a = ts.decode_flat_action(s, int(rng.choice(legal)))
            before = s.to_dict()
            step_checked(s, a, context=f"seed {seed} step {len(actions)}")
            drain_chance(s)
            try:
                described = describe(before, s.to_dict(), a)
            except Exception as e:
                # The original cannot describe every step -- during the headline phase it reads
                # any card choice, an event's included, as a headline commit, and throws on
                # card 0. There is nothing to compare such a step against: the game ends here.
                print(f"seed {seed}: the original stops at step {len(actions)} ({type(e).__name__}: {e})")
                break
            actions.append([int(a.decision_type), int(a.primary_id), int(a.secondary_id), int(a.flags)])
            lines.append(described)
        games.append({"seed": seed, "actions": actions, "lines": lines})
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump({"source": f"web/server/session.py describe_action_and_deltas @ {SOURCE_COMMIT[:7]}",
                   "games": games}, f)
    print(f"{OUT}: {len(games)} games, {sum(len(g['actions']) for g in games)} steps")


if __name__ == "__main__":
    main()
