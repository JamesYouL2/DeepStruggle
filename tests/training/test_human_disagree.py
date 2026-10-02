"""The human-disagreement scan: decisions are classed, reordered points are not disagreements.

Reads the real ts-replayer corpus, like tests/replayer; a missing corpus fails rather than skips."""
import numpy as np

from ai.eval import human_disagree as H
from tools.lib.corpus_paths import distinct_corpus_files


def _uniform_logits(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.zeros(masks.shape, dtype=np.float32)


def test_scan_classes_every_decision_and_spots_play() -> None:
    paths, _ = distinct_corpus_files()
    rows, states = H.scan_game(H.load_game(sorted(paths, key=str)[0]), _uniform_logits, 0)
    assert rows and {r["kind"] for r in rows} <= set(H.KINDS)
    assert {"headline", "card", "mode", "influence"} <= {r["kind"] for r in rows}
    assert sum(not r["agree"] for r in rows) == len(states)
    for r in rows:
        assert 0.0 <= r["p_human"] <= 1.0 and (r["agree"] == (r["human"] == r["bot"]))

    rng = np.random.default_rng(0)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) for m in masks], dtype=np.int32)

    spots = H.choose_spots(rows, per_kind=1, max_p=1.1, seed=0)
    played = H.play(spots, states, act, pairs=2, seed=0)
    assert len(played) == len(spots) and all(-1.0 <= r["diff"] <= 1.0 for r in played)
    md, summary = H.report(rows, played, {"model": "x", "max_p": 1.1})
    assert summary["decisions"] == len(rows) and "strong human play" in md


def test_another_order_of_the_same_play_agrees() -> None:
    # Two influence points on one card, then a mode choice, then two points on another card.
    d = [("influence", 1, 5, True, 120), ("influence", 1, 5, True, 130), ("mode", 1, 7, False, 112),
         ("influence", 1, 7, True, 140), ("influence", 1, 7, True, 150)]
    used = H.same_play_sets(d)
    assert used[0] == used[1] == {120, 130} and used[2] == set() and used[3] == used[4] == {140, 150}
    # The other side's points on the same card are a different play.
    used = H.same_play_sets([("influence", 1, 5, True, 120), ("influence", 2, 5, True, 130)])
    assert used[0] == {120} and used[1] == {130}
