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


def test_targeted_bank_and_card_rules() -> None:
    from ai.eval import card_rules as C
    from ai.eval.doctrine_census import cards

    names = {k: str(v["name"]) for k, v in cards().items()}
    by_name = {v: k for k, v in names.items()}
    targets = C.target_ids(by_name)
    assert len(targets) == sum(len(v) for v in C.TARGETS.values())
    act = _random_policy(3)
    pos, counts = B.collect_targets(act, targets, per_card=2, games=6, seed=4, envs=4)
    assert counts and all(sum(row.values()) > 0 for row in counts.values())
    for st, kind, _ in pos:
        ctx = st.ctx()
        side = "US" if ctx.decision_player == ts.Player.US else "USSR"
        assert kind == "mode" and ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
        assert (side, int(ctx.pending_op_card)) in targets
    recs = B.build(act, _probs, _value, pos, pairs=2, seed=1)
    md, summary = C.report(recs, names, counts, {"model": "x"})
    assert "Card event rules" in md


def test_a_condition_is_scored_out_of_sample() -> None:
    from ai.eval import card_rules as C

    # The event gains only at DEFCON 2: the fitted rule says so, and holds on the held-out half.
    rows = [({"defcon": d, "turn": 1, "ar": 1, "rounds_left": 3, "lead": 0, "hand": 5, "surplus": 2, "space": 0,
              "space_opp": 0, "short": 0, "short_opp": 0}, 0.2 if d == 2 else -0.1) for d in (2, 3, 4, 5) for _ in range(20)]
    fit = C.best_condition(rows)
    assert fit is not None
    cond, gain = fit
    assert cond == ("DEFCON", "<=", 2.0) and abs(gain - 0.05) < 1e-9
    held, conds = C.cross_validated(rows)
    assert conds == [cond, cond] and abs(float(np.mean(held)) - 0.05) < 1e-9


def test_forced_conditions_keep_only_where_they_hold() -> None:
    from ai.eval import card_rules as C
    from ai.eval.doctrine_census import cards

    names = {k: str(v["name"]) for k, v in cards().items()}
    by_name = {v: k for k, v in names.items()}
    act = _random_policy(5)
    for which in C.FORCED_SETS:
        keep = C.forced_keep(which, by_name, act)
        assert len(keep) == len(C.FORCED_SETS[which])
        # A test nothing passes keeps nothing; the counts still come back.
        pos, counts = B.collect_targets(act, set(keep), per_card=5, games=4, seed=6, envs=4,
                                        keep={k: (lambda st: False) for k in keep})
        assert pos == [] and counts
    keep = C.forced_keep("oss12", by_name, act)
    pos, _ = B.collect_targets(act, set(keep), per_card=5, games=60, seed=6, envs=8, keep=keep)
    for st, _, _ in pos:
        us = st.ctx().decision_player == ts.Player.US
        me, opp = (int(st.us_space_track), int(st.ussr_space_track))[:: 1 if us else -1]
        assert (me, opp) == (1, 2)
    md, _ = C.forced_report(B.build(act, _probs, _value, pos, pairs=2, seed=1), names, "oss12", {"model": "x"})
    assert "Forced event conditions (oss12)" in md
