"""The census comparison tools without the corpus: the robust-gap rule, and the trackers on self-play.

`event_census_compare.gap_verdict` calls a card's difference robust only when every bot sits on
the same side of the humans by the minimum gap and more than 3 standard errors; one bot on the
other side, or one too close, must void it. The self-play half drives the trackers with a uniform
random legal policy, which needs no checkpoint, and checks what they may record.
"""

from __future__ import annotations

import collections
from typing import Tuple

import numpy as np

from tools.scripts.event_census_compare import gap_verdict, last_round, rate
from tools.scripts.event_play_census import HoldingTracker, SCORING, _owner, selfplay
from tools.scripts.placement_census import PlacementTracker


def _r(p: float, n: int) -> Tuple[float, float, int]:
    return rate(collections.Counter({"event": round(p * n), "ops": n - round(p * n)}), ("event",))


def test_gap_needs_every_bot_on_one_side() -> None:
    h = _r(0.7, 400)
    assert gap_verdict(h, [_r(0.2, 4000), _r(0.3, 4000)], 0.10, 20) == 1
    assert gap_verdict(_r(0.2, 400), [_r(0.7, 4000), _r(0.6, 4000)], 0.10, 20) == -1
    assert gap_verdict(h, [_r(0.2, 4000), _r(0.9, 4000)], 0.10, 20) is None     # bots disagree
    assert gap_verdict(h, [_r(0.2, 4000), _r(0.65, 4000)], 0.10, 20) is None    # one bot too close
    assert gap_verdict(_r(0.7, 10), [_r(0.2, 4000)], 0.10, 20) is None          # too few holdings
    assert gap_verdict(_r(0.6, 30), [_r(0.45, 4000)], 0.10, 20) is None         # inside 3 SE


def test_last_round() -> None:
    assert [last_round(t) for t in (1, 3, 4, 10)] == [6, 6, 7, 7]


def _random(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(obs.sum() * 1000) % (2 ** 32))
    return np.where(masks.astype(bool), rng.random(masks.shape), -np.inf)


def test_trackers_on_random_self_play() -> None:
    holdings = [h for t in selfplay(_random, 0, 4, 7, 4, 0.0, HoldingTracker) for h in t.holdings]
    assert holdings
    for h in holdings:
        if h.outcome == "event":
            assert _owner(h.card) in (None, h.side)            # the opponent's card is never "evented"
        if h.outcome in ("event", "ops", "space"):
            assert h.ar > 0 and h.turn > 0
        if h.outcome == "headline":
            assert h.ar == 0
        if h.outcome == "event" and h.card in SCORING:
            assert h.legal
    recs = [r for t in selfplay(_random, 0, 2, 7, 2, 0.0, PlacementTracker) for r in t.records]
    assert recs and all(1 <= r[1] <= 10 for r in recs)
