"""The takeover pricing (tools/scripts/takeover_pricing.py) and the CI batcher that runs it: a macro's
base steps reach the engine's own macro position, playouts can apply a macro slot, a run is counted once,
a macro already in raw's line is not priced, and the gain is read only on fresh pairs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.paired_playouts import apply_move
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import TsVectorizedEnv
from tools.lib.corpus_driver import position_token, state_from_token
from tools.scripts.ci_batches import batches, pending_jobs
from tools.scripts.takeover_pricing import (MACRO, NODE, RunStarts, analyse, build_row, clustered, expansion, macro_slots,
                                            opportunity_kind, priced_macros, render, same_run, sparse_mean,
                                            takeover_macros)

COUNTRIES = {0: {"name": "A", "region": "Europe", "battleground": True},
             1: {"name": "B", "region": "Asia", "battleground": False}}


def _point_positions(games: int, seed: int) -> List[ts.GameState]:
    """Positions of random games where a macro slot is legal."""
    out: List[ts.GameState] = []
    env = TsVectorizedEnv(num_envs=games, base_seed=seed)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    for _ in range(1200):
        for i in range(games):
            st = env.runner.get_state(i)
            if not ts.Engine.is_terminal(st) and macro_slots(st):
                out.append(st.clone())
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks], dtype=np.int32)
        _, masks, _, _, _ = env.step(actions)
    assert len(out) > 100
    return out[::7]


def test_expansion_reaches_the_engines_macro_position() -> None:
    for st in _point_positions(4, 11):
        for slot in macro_slots(st):
            engine = st.clone()
            ts.Engine.step_flat(engine, slot, False, False, True)
            chain = expansion(st, slot - MACRO)
            assert len(chain) >= 3, "a legal macro places at least two points"
            assert chain[-1].to_save_json() == engine.to_save_json()


def test_apply_move_takes_a_macro_slot_and_refuses_an_illegal_one() -> None:
    st = _point_positions(2, 5)[0]
    slot = macro_slots(st)[0]
    before = st.to_save_json()
    moved = apply_move(st, slot)
    assert st.to_save_json() == before, "the position itself is left alone"
    assert moved.to_save_json() != before
    illegal = next(a for a in range(MACRO, ActionEncoder.FLAT_ACTION_SIZE)
                   if not np.asarray(ts.Engine.get_flat_action_mask(st, False, True))[a])
    with pytest.raises(ValueError):
        apply_move(st, illegal)
    # A base action is what it always was.
    base = int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))[0])
    plain = st.clone()
    ts.Engine.step_flat(plain, base)
    assert apply_move(st, base).to_save_json() == plain.to_save_json()


def test_a_position_round_trips_with_its_macros() -> None:
    for st in _point_positions(2, 7):
        assert macro_slots(state_from_token(position_token(st))) == macro_slots(st)


def test_the_setups_are_three_runs_counted_once_each() -> None:
    env = TsVectorizedEnv(num_envs=1, base_seed=3)
    _, masks, _ = env.reset_all()
    tracker = RunStarts()
    rng = np.random.default_rng(0)
    st = env.runner.get_state(0)
    points = 0
    while st.current_phase == ts.Phase.SETUP:
        a = int(rng.choice(np.flatnonzero(masks[0])))
        tracker.observe(st, a)
        points += st.ctx().decision_type == ts.DecisionType.POINT_NODE
        _, masks, _, _, _ = env.step(np.array([a], dtype=np.int32))
        st = env.runner.get_state(0)
    assert points == 15  # USSR 6, US 7, US 2
    assert tracker.run_starts == 3


def test_same_run_ends_when_the_counter_stops_falling() -> None:
    assert not same_run(None, (1, "S", 0, 0, 0, 7))
    assert same_run((1, "S", 0, 0, 0, 7), (1, "S", 0, 0, 0, 6))
    assert not same_run((1, "S", 0, 0, 0, 1), (1, "S", 0, 0, 0, 2))
    assert not same_run((1, "S", 0, 0, 0, 7), (0, "S", 0, 0, 0, 6))
    assert not same_run((1, "A", 2, 3, 9, 3), (1, "A", 2, 3, 8, 2))


def test_takeover_macros_raise_the_movers_influence() -> None:
    for st in _point_positions(3, 13):
        side = st.ctx().decision_player
        for m in takeover_macros(st):
            after = st.clone()
            ts.Engine.step_flat(after, int(m["slot"]), False, False, True)
            c = after.get_country(int(m["country"]))
            was = st.get_country(int(m["country"]))
            gained = (c.us_influence - was.us_influence) if side == ts.Player.US else (c.ussr_influence - was.ussr_influence)
            assert gained == m["placed"] > 0


def test_priced_macros_are_the_top_lines_and_a_stable_draw_from_the_rest() -> None:
    macros = [{"slot": MACRO + i, "country": i, "p_line": 1.0 / (i + 1)} for i in range(20)]
    a = priced_macros(macros, 3, 2, "00ab00ab00ab0000")
    assert [m["country"] for m in a[:3]] == [0, 1, 2]
    assert all(m["picked"] == "top" for m in a[:3]) and all(m["picked"] == "spread" for m in a[3:])
    assert len({m["country"] for m in a}) == 5 and all(m["country"] >= 3 for m in a[3:])
    assert a == priced_macros(list(reversed(macros)), 3, 2, "00ab00ab00ab0000")
    assert len(priced_macros(macros[:2], 3, 2, "00ab00ab00ab0000")) == 2


def _prefers(country: int) -> Any:
    def fn(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        logits = np.where(masks > 0, 0.0, -np.inf)
        logits[:, NODE + country] = np.where(masks[:, NODE + country] > 0, 5.0, -np.inf)
        return logits
    return fn


def test_a_macro_in_raws_own_line_is_not_priced() -> None:
    st = next(s for s in _point_positions(4, 17) if opportunity_kind(s) is not None and takeover_macros(s))
    target = int(takeover_macros(st)[0]["country"])
    token = position_token(st)
    row, counts = build_row(st, token, "ops", "1:0", _prefers(target), 0, 8, 4)
    assert counts["in_line"] >= 1
    assert counts["macros"] == len(takeover_macros(st))
    if row is not None:
        assert row["raw"] == NODE + target
        assert target not in [m["country"] for m in row["macros"]]
        assert row["pmoves"][0] == row["raw"] and row["pmoves"][1:] == [m["slot"] for m in row["macros"]]
        assert all(0.0 <= m["p_line"] <= 1.0 for m in row["macros"])


def test_sparse_mean_is_the_zero_filled_estimate() -> None:
    listed = [("a", 0.5), ("a", 0.25), ("b", -0.5)]
    n, clusters = 10, 4
    m, se = sparse_mean(listed, n, clusters)
    assert m == pytest.approx(0.025)
    totals = np.array([0.75, -0.5, 0.0, 0.0])
    want = np.sqrt(((totals - totals.mean()) ** 2).sum() * 4 / 3) / n
    assert se == pytest.approx(want)
    assert clustered([("a", 1.0), ("b", 3.0)]) == pytest.approx((2.0, 1.0))


def _row(rid: str, game: str, turn: int = 2) -> Dict[str, Any]:
    return {"id": rid, "pos": "x", "raw": NODE, "game": game, "kind": "ops", "side": "US", "turn": turn, "ar": 1, "card": 9,
            "macros": [{"slot": MACRO, "country": 0, "placed": 2, "control": True, "p_line": 0.002, "picked": "top"},
                       {"slot": MACRO + 1, "country": 1, "placed": 3, "control": False, "p_line": 0.2, "picked": "spread"}]}


def _analysis() -> Tuple[Dict[str, Any], float]:
    rows = [_row("a", "1:0"), _row("b", "1:1"), _row("c", "1:2")]
    pairs = 4
    screen = {
        # Row a's screen lead is a big selected maximum; the fresh pairs below are what count.
        "a": {"scores": {str(NODE): [0.0] * pairs, str(MACRO): [1.0] * pairs, str(MACRO + 1): [0.0] * pairs}},
        "b": {"scores": {str(NODE): [1.0] * pairs, str(MACRO): [0.0] * pairs, str(MACRO + 1): [1.0] * pairs}},
        "c": {"scores": {str(NODE): [0.5] * pairs, str(MACRO): [0.5] * pairs, str(MACRO + 1): [0.5] * pairs}},
    }
    selected = {"a": {"id": "a", "escalated": True, "best": MACRO}, "b": {"id": "b", "escalated": False, "best": NODE},
                "c": {"id": "c", "escalated": False, "best": NODE}}
    confirm = {"a": {"scores": {str(NODE): [0.0, 0.0, 1.0, 1.0], str(MACRO): [0.5, 0.5, 1.0, 1.0]}}}
    summaries = [{"games": 3, "opportunities": 30, "run_starts": 60, "sampled": 4, "macros": 8, "in_line": 1, "all_in_line": 1,
                  "unpriced": 0, "unloadable": 0, "opportunities_by_kind": {"ops": 30}}]
    return analyse(rows, summaries, screen, selected, confirm, COUNTRIES), 0.25


def test_the_gain_reads_only_fresh_pairs_and_counts_the_rest_zero() -> None:
    x, lead = _analysis()
    assert x["escalated"] == 1
    # One row's fresh lead over all four sampled opportunities (the all-in-line one included), not the screen's 1.0.
    assert x["gain"] == pytest.approx(lead / 4)
    assert x["opportunities_a_game"] == pytest.approx(10.0)
    assert x["gain_points_a_game"] == pytest.approx(100 * 10.0 * lead / 4)
    groups = {g["group"]: g for g in x["gain_by_group"]["side"]}
    assert groups["US"]["points_a_game"] == pytest.approx(x["gain_points_a_game"])
    assert x["gain_by_country"][0]["country"] == "A"


def test_the_unselected_leads_are_every_macros_screen_lead() -> None:
    x, _ = _analysis()
    by_picked = {g["group"]: g for g in x["unselected"]["how picked"]}
    assert by_picked["top"]["n"] == 3 and by_picked["spread"]["n"] == 3
    assert by_picked["top"]["lead"] == pytest.approx(0.0)
    assert by_picked["spread"]["lead"] == pytest.approx(0.0)
    assert {g["group"] for g in x["unselected"]["raw's probability of the line"]} == {"0.1-1%", ">= 10%"}
    text = render(x, 5)
    assert "points of win probability a game" in text and "| A |" in text
    json.dumps(x)


def test_analyse_refuses_a_row_with_no_playouts() -> None:
    x = [_row("a", "1:0")]
    with pytest.raises(ValueError):
        analyse(x, [{"games": 1, "opportunities": 1, "sampled": 1}], {}, {}, {}, COUNTRIES)


def test_ci_batches_cover_every_part_once() -> None:
    assert batches(14, 6) == [(1, 6), (7, 12), (13, 14)]
    assert batches(14, 6, start=7) == [(7, 12), (13, 14)]
    assert batches(6, 6) == [(1, 6)]


def test_pending_jobs_counts_an_unplanned_run_at_its_batch() -> None:
    assert pending_jobs(["completed", "in_progress", "queued", "completed"]) == 2
    # A run that has only its plan job listed has not expanded its matrix yet.
    assert pending_jobs(["in_progress"]) >= 6
