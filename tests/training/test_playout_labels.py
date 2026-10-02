"""Playout labels: candidates start with the model's choice, labels pack and pool losslessly."""
import numpy as np
import ts_engine as ts

from ai.eval import playout_labels as P
from bindings.action_encoder import ActionEncoder


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def _probs(st: ts.GameState) -> np.ndarray:
    m = np.asarray(ActionEncoder.get_legal_mask(st)).astype(float)
    return m / m.sum()


def test_label_pack_and_soft_target() -> None:
    act = _random_policy(0)
    pos = P.collect(act, 8, seed=3, envs=4, base=0.05)
    assert pos
    labels = P.label(act, _probs, pos, pairs=2, seed=1, k=3)
    for lb in labels:
        mask = lb["mask"].astype(bool)
        assert len(lb["cands"]) >= 2 and all(mask[a] for a in lb["cands"])
        assert lb["cands"][0] == int(np.flatnonzero(mask)[0])   # uniform probs: argmax is the first legal move
        assert np.all((0 <= lb["q"]) & (lb["q"] <= 1))
    arr = P.pack(labels)
    assert arr["obs"].shape == (len(labels), ts.OBS_SIZE) and arr["obs"].dtype == np.float16
    assert np.array_equal(np.unpackbits(arr["mask"], axis=1)[:, :len(labels[0]["mask"])][0], labels[0]["mask"])
    md, summary = P.report(arr, {"model": "x", "pairs": 2})
    assert summary["n"] == len(labels) and "Playout labels" in md


def test_soft_target_moves_toward_the_better_move() -> None:
    t = P.soft_target(np.array([[0.4, 0.6]]), np.array([[0.5, 0.5]]), tau=0.1)
    assert t[0, 1] > 0.85 and abs(t.sum() - 1) < 1e-9
    same = P.soft_target(np.array([[0.5, 0.5]]), np.array([[0.9, 0.1]]), tau=0.1)
    assert np.allclose(same, [[0.9, 0.1]])                  # no evidence, no change
