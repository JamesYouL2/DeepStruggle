"""The position bank: records hold every candidate's pairs, and queries read them back."""
import numpy as np
import ts_engine as ts

from ai.eval import bank_query as Q
from ai.eval import position_bank as B
from bindings.action_encoder import ActionEncoder


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def _probs(st: ts.GameState) -> np.ndarray:
    m = np.asarray(ActionEncoder.get_legal_mask(st)).astype(float)
    return m / m.sum()


def _value(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.zeros(len(obs))


def test_timing_cards_resolve() -> None:
    assert len(B.timing_ids()) == len(B.HAND_TIMING_CARDS)


def test_build_and_query() -> None:
    act = _random_policy(1)
    pos = B.collect(act, 10, seed=2, envs=4, base=0.05)
    recs = B.build(act, _probs, _value, pos, pairs=3, seed=1)
    assert recs
    for r in recs:
        assert len(r["candidates"]) == len(r["results"]) >= 2
        assert all(len(x) == 3 and set(x) <= {"0", "1", "2"} for x in r["results"])
        assert ts.state_from_save_json(r["save"]) is not None
        f = r["features"]
        assert f["side"] in ("US", "USSR") and f["weight"] > 0 and len(f["region_status"]) == 6
        if r["kind"] == "card":                               # every legal card is a candidate
            legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(ts.state_from_save_json(r["save"])))[:110])
            assert {int(c["prefix"][0]) for c in r["candidates"]} >= set(int(a) for a in legal)
    # A rule that always picks candidate 1 has, per record, the effect of candidate 1 minus 0.
    rule = Q.Rule("second", lambda r: 1)
    hits = Q.ask(recs, rule)
    assert len(hits) == len(recs)
    r0 = recs[0]
    assert hits[0][2] == float((B.scores(r0, 1) - B.scores(r0, 0)).mean())
    md, summary = Q.report(recs, (rule,) + Q.RULES, {"model": "x"})
    assert "Bank query" in md and summary["rules"]["second"]["n"] == len(recs)


def test_a_line_stops_where_it_stops_being_legal() -> None:
    st = ts.GameState()
    ts.Engine.init_game(st, 4)
    legal = [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]
    illegal = next(a for a in range(220) if a not in legal)
    out = B.apply_lenient(st, [legal[0], illegal, illegal])
    ref = st.clone()
    ts.Engine.step_flat(ref, legal[0])
    from tools.lib.game_step import drain_chance
    drain_chance(ref, context="test")
    assert out.to_save_json() == ref.to_save_json()     # the first step taken, the illegal rest not
