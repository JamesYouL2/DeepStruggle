"""The choice oracle: positions match their scenario, and every branch is scored per position."""
import numpy as np
import ts_engine as ts

from ai.eval import choice_oracle as C
from bindings.action_encoder import ActionEncoder


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def test_positions_match_and_branches_are_scored() -> None:
    scs = [s for s in C.scenarios() if "Grain Sales" in s.name or "Asia Scoring vs Nasser" in s.name]
    act = _random_policy(0)
    found = C.collect(act, scs, per=2, seed=2, envs=16, headline_games=4000, full_games=400, accept=1.0)
    for s in scs:
        assert found[s.name], s.name
        for st, _ in found[s.name]:
            assert C.matches(s, st, np.asarray(ActionEncoder.get_legal_mask(st)))
            if s.kind == "headline":
                assert int(st.turn) == 1 and st.ctx().decision_player == ts.Player.USSR
        rows = C.play(act, s, found[s.name], pairs=2, seed=1)
        assert len(rows) == len(found[s.name])
        for r in rows:
            assert all(0.0 <= r[b] <= 1.0 for b in s.branches())
    md, js = C.report([r for s in scs for r in C.play(act, s, found[s.name], 2, 1)], {"model": "random"})
    assert "Asia Scoring − Nasser" in md
