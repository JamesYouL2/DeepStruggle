"""The page's Gumbel root plays the player `search:<model>:64:determinize:all:gumbel_k=8:
gumbel_scale=0:fpu_reduction=0.2` does.

web/ui/src/search/mcts.ts ports ai/search/gumbel_root.py on top of the page's PUCT tree (itself
the port tests/web/test_page_search.py holds). Pinned here, with a stub network
(tests/web/js/gumbel_root.ts):

* sigma(completed Q) gives the Python's numbers -- the cases are computed here by
  `gumbel_root.sigma_completed` and handed to the page's function;
* the candidates are the k largest logits; k = 1 plays the argmax without searching;
* sequential halving keeps 8 -> 4 -> 2 -> 1 over three rounds, each candidate getting
  floor(64 / (3 x survivors)) simulations per round it takes part in, within the budget;
* with every value equal the logits decide; otherwise the move played maximises logit + sigma
  among the last round's pair;
* first-play urgency keeps a 2-simulation tree on its first child when the reduction is large;
* the engine singleton is handed back byte-identical and the pick is legal in the real state.

Needs the built WebAssembly engine and node, like the rest of tests/web: a missing build fails
here, it is not skipped.
"""
from __future__ import annotations

import json
import random

from tests.web.js_runner import PUBLIC, run_ts

EXPECTED = (
    "sigmaMatchesPython",
    "fixtureHasEight",
    "kOnePlaysArgmax",
    "candidatesAreTopK",
    "halving",
    "withinBudget",
    "spentRight",
    "pickMaximisesScore",
    "equalValuesPlayArgmax",
    "fpuKeepsFirstChild",
    "engineRestored",
    "pickLegal",
    "secondPosition",
)


def sigma_cases() -> list:
    """sigma_completed on random inputs, including the unvisited, all-unvisited and flat-Q cases."""
    from ai.search.gumbel_root import sigma_completed

    rng = random.Random(7)
    cases = []
    for t in range(40):
        legal = rng.sample(range(220), rng.randint(1, 12))
        logits = {a: rng.uniform(-4, 4) for a in legal}
        if t % 10 == 0:
            n: dict = {}                                      # nothing visited: the network's value
        else:
            n = {a: float(rng.choice([0, 0, 1, 2, 5, 10])) for a in legal}
        flat = t % 10 == 5
        q = {a: (0.25 if flat else rng.uniform(-1, 1)) for a in legal if n.get(a, 0) > 0}
        value = rng.uniform(-1, 1)
        sig = sigma_completed(logits, value, n, q)
        cases.append({"logits": {str(a): v for a, v in logits.items()}, "value": value,
                      "n": {str(a): v for a, v in n.items()}, "q": {str(a): v for a, v in q.items()},
                      "expected": {str(a): v for a, v in sig.items()}})
    return cases


def test_the_page_gumbel_root_keeps_the_gumbel64_player(tmp_path) -> None:
    result = run_ts("gumbel_root.ts", str(tmp_path), PUBLIC, json.dumps(sigma_cases()))
    for key in EXPECTED:
        assert result.get(key) is True, f"{key}: {result}"
