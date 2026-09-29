"""The page's searcher plays the same player `search:<onnx>:64:determinize` does.

web/ui/src/search/mcts.ts is the port of ai/search/batched_mcts.py's tree plus
ai/search/dmcts.py's determinize, and the properties pinned here are the ones a port gets wrong
(the Python side is tests/training/test_search_tiebreak.py):

* one simulation plays the network's argmax -- not whichever move the engine lists first;
* a visit tie breaks by mover-mean value, then prior -- again not by list position;
* the sampled world preserves every count and every card the mover has seen, and the worlds
  actually differ from one another;
* the search hands the engine singleton back exactly as it found it;
* the pick is legal in the REAL state, and a search off the opening still works.

Needs the built WebAssembly engine and node, like the rest of tests/web: a missing build fails
here, it is not skipped.
"""
from __future__ import annotations

from tests.web.js_runner import PUBLIC, run_ts

EXPECTED = (
    "fixtureDiscriminates",
    "oneSimArgmax",
    "twoSimTieBreak",
    "visitsSum",
    "worldCounts",
    "seenUntouched",
    "worldsDiffer",
    "engineRestored",
    "pickLegal",
    "secondPosition",
)


def test_the_page_searcher_keeps_the_search64_player(tmp_path) -> None:
    result = run_ts("search_tiebreak.ts", str(tmp_path), PUBLIC)
    for key in EXPECTED:
        assert result.get(key) is True, f"{key}: {result}"
