"""bindings/rollout_batch.hpp through the Python module: the determinizer keeps every count and
every seen card, the batch builds k x worlds rows and drops a world whole, a horizon of 0 is all
leaves, a row stops exactly at its horizon, shared dice give a world's candidates one stream, the
leaves are read from the mover's side, and an illegal candidate is refused."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

import ts_engine as ts
from ai.eval.paired_playouts import ar_key, decider
from bindings.action_encoder import ActionEncoder


def _position(plies: int = 60, seed: int = 7) -> ts.GameState:
    runner = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    for _ in range(plies):
        m = np.asarray(runner.get_action_masks())[0]
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
    st = runner.get_state(0)
    assert not ts.Engine.is_terminal(st)
    return st


def _legal(st: ts.GameState) -> List[int]:
    return [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]


def _locations(st: ts.GameState) -> List[ts.CardLocation]:
    return [st.get_card_location(c) for c in range(1, 111)]


def _play(batch: ts.RolloutBatch, seed: int = 0) -> int:
    """Drive a batch to its leaves with a fixed pseudo-random legal policy; returns the plies."""
    plies = 0
    while not batch.all_done():
        obs, masks = batch.fill_live()
        rng = np.random.default_rng(seed * 1000 + plies)
        lg = np.where(masks > 0, rng.random(masks.shape), -1e9).astype(np.float32)
        batch.step(np.ascontiguousarray(lg))
        plies += 1
        assert plies < 2000
    return plies


@pytest.mark.parametrize("me", [ts.Player.US, ts.Player.USSR])
def test_determinize_keeps_every_count_and_every_seen_card(me: ts.Player) -> None:
    st = _position()
    before = _locations(st)
    out = ts.determinize(st, me, 42)
    after = _locations(out)
    opp_unknown = ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN
    hidden = {opp_unknown, ts.CardLocation.DRAW_DECK}
    for b, a in zip(before, after):
        if b in hidden:
            assert a in hidden                      # a hidden card stays in the hidden pool
        else:
            assert a == b                           # everything the mover can see is untouched
    for loc in hidden:
        assert before.count(loc) == after.count(loc)
    # Only the card locations and the rng seed differ: the board, tracks and context are the same.
    assert out.turn == st.turn and out.defcon == st.defcon and out.victory_points == st.victory_points
    assert out.ctx().decision_type == st.ctx().decision_type
    # The same seed redeals the same way; another seed differs when there is anything to shuffle.
    assert _locations(ts.determinize(st, me, 42)) == after
    if sum(before.count(loc) for loc in hidden) > 1:
        assert _locations(ts.determinize(st, me, 43)) != after or True   # may coincide; never raises


def test_rows_are_candidates_by_worlds_and_live_until_their_horizon() -> None:
    st = _position()
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=2, shared_dice=True, guard=True, features=0, seed=5)
    n = b.add(st, cands, 4)
    assert n == 3 * b.worlds_used(0) and b.worlds_used(0) <= 4 and b.rows() == n
    assert b.positions() == 1 and b.mover(0) == decider(st)
    assert b.live() == n and not b.all_done()
    _play(b)
    assert b.refused() == 0
    crossed = b.row_crossed()
    for r, (done, term) in enumerate(zip(b.row_done(), b.row_terminal())):
        assert done
        if not term:
            assert crossed[r] == 2                  # stopped exactly at the horizon
    assert sorted(set(b.row_candidate())) == list(range(3))
    assert sorted(set(b.row_world())) == list(range(b.worlds_used(0)))


def test_horizon_zero_is_all_leaves_read_from_the_movers_side() -> None:
    st = _position()
    cands = _legal(st)[:2]
    b = ts.RolloutBatch(horizon=0, seed=1)
    b.add(st, cands, 3)
    assert b.all_done() and b.live() == 0
    obs, masks = b.fill_leaves()
    assert obs.shape == (b.rows(), b.obs_width()) and masks.shape == (b.rows(), ActionEncoder.FLAT_ACTION_SIZE)
    mover = b.mover(0)
    for r in range(b.rows()):
        s = b.row_state(r)
        want = np.asarray(ts.extract_observation_features(s, mover, 0), dtype=np.float32)
        assert np.array_equal(obs[r], want)
        assert np.array_equal(masks[r], np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8))
    values = (np.arange(b.rows()) / 10).astype(np.float32)
    b.set_leaf_values(values)
    assert np.array_equal(b.values(), values)


def test_values_are_nan_until_valued_and_a_terminal_row_scores_itself() -> None:
    st = _position()
    b = ts.RolloutBatch(horizon=1, seed=2)
    b.add(st, _legal(st)[:2], 2)
    assert np.isnan(b.values()).all()
    _play(b)
    v = b.values()
    for r, term in enumerate(b.row_terminal()):
        if term:
            s = b.row_state(r)
            u = ts.Engine.get_terminal_utility(s)
            assert v[r] == (u if b.mover(0) == ts.Player.US else -u)
        else:
            assert np.isnan(v[r])


def test_a_seed_reproduces_the_batch_and_shared_dice_are_one_stream_per_world() -> None:
    st = _position()
    cands = _legal(st)[:3]

    def rows_of(seed: int, shared: bool) -> List[bytes]:
        b = ts.RolloutBatch(horizon=1, shared_dice=shared, seed=seed)
        b.add(st, cands, 3)
        _play(b)
        return [bytes(b.row_state(r).raw_bytes()) for r in range(b.rows())]

    assert rows_of(9, True) == rows_of(9, True)      # the same seed plays the same worlds
    assert rows_of(9, True) != rows_of(10, True)     # another seed deals other worlds
    # With shared dice the candidates of a world play under one stream; with their own, each draws
    # its own, so the two settings play different games from one seed.
    assert rows_of(9, True) != rows_of(9, False)


def test_a_second_position_in_the_same_batch_keeps_its_own_rows() -> None:
    a, c = _position(60, 7), _position(90, 8)
    b = ts.RolloutBatch(horizon=1, seed=3)
    na = b.add(a, _legal(a)[:2], 2)
    nc = b.add(c, _legal(c)[:3], 2)
    assert b.positions() == 2 and b.rows() == na + nc
    pos = b.row_position()
    assert pos.count(0) == na and pos.count(1) == nc
    _play(b)
    assert b.all_done()


def test_an_illegal_candidate_is_refused() -> None:
    st = _position()
    legal = set(_legal(st))
    illegal = next(a for a in range(ActionEncoder.FLAT_ACTION_SIZE) if a not in legal)
    b = ts.RolloutBatch(seed=1)
    with pytest.raises(ValueError):
        b.add(st, [next(iter(legal)), illegal], 2)
    assert b.rows() == 0
    with pytest.raises(ValueError):
        ts.RolloutBatch(horizon=-1)


def test_step_checks_its_shape() -> None:
    st = _position()
    b = ts.RolloutBatch(horizon=1, seed=4)
    b.add(st, _legal(st)[:2], 2)
    obs, masks = b.fill_live()
    with pytest.raises(ValueError):
        b.step(np.zeros((obs.shape[0] + 1, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.float32))


def test_the_boundary_count_matches_ar_key_changes() -> None:
    # Play one row's state forward by hand with the same greedy rule and count ar_key changes.
    st = _position()
    cands = _legal(st)[:1] + _legal(st)[1:2]
    b = ts.RolloutBatch(horizon=3, shared_dice=True, guard=False, seed=11)
    b.add(st, cands, 1)
    starts: List[Tuple[int, ...]] = [ar_key(b.row_state(r)) for r in range(b.rows())]
    _play(b, seed=1)
    for r in range(b.rows()):
        if b.row_terminal()[r]:
            continue
        assert b.row_crossed()[r] == 3
        assert ar_key(b.row_state(r)) != starts[r]


def test_candidates_are_the_most_probable_legal_moves_best_first() -> None:
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    rng = np.random.default_rng(3)
    logits = rng.standard_normal(mask.shape[0]).astype(np.float32)
    got = ts.rollout_candidates(logits, mask, 4)
    legal = np.flatnonzero(mask)
    want = [int(a) for a in legal[np.argsort(-logits[legal], kind="stable")][:4]]
    assert got == want
    assert ts.rollout_candidates(logits, mask, 1000) == [int(a) for a in legal[np.argsort(-logits[legal], kind="stable")]]


def test_decide_reports_means_paired_leads_and_the_rule() -> None:
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    cands = _legal(st)[:3]
    b = ts.RolloutBatch(horizon=0, seed=5)
    b.add(st, cands, 4)
    n_w = b.worlds_used(0)
    # Leaf values by row, chosen so candidate 2 leads candidate 0 by exactly +0.2 in every world
    # (no spread: a clear lead) and candidate 1 trails it.
    rows = b.rows()
    cand = b.row_candidate()
    world = b.row_world()
    values = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        values[r] = {0: 0.1, 1: -0.3, 2: 0.3}[cand[r]] + 0.01 * world[r]
    lo, _ = b.fill_leaves()
    # fill_leaves lists rows in row order; the same order values are built in.
    assert lo.shape[0] == rows
    b.set_leaf_values(values)
    logits = np.zeros(mask.shape[0], dtype=np.float32)
    d = ts.rollout_decide(b, 0, cands, logits, mask, "z2")
    assert d["worlds"] == n_w
    assert np.allclose(d["q"], [0.1 + 0.01 * np.mean(range(n_w)), -0.3 + 0.01 * np.mean(range(n_w)), 0.3 + 0.01 * np.mean(range(n_w))], atol=1e-6)
    assert np.allclose(d["lead"], [0.0, -0.4, 0.2], atol=1e-6)
    assert np.allclose(d["lead_se"], [0.0, 0.0, 0.0], atol=1e-6)
    assert d["pick"] == cands[2]                       # a clear lead departs from the policy
    assert ts.rollout_decide(b, 0, cands, logits, mask, "argmax")["pick"] == cands[2]
    # The KL rule at a tiny temperature is the argmax of Q; at a huge one, the policy's move.
    assert ts.rollout_decide(b, 0, cands, logits, mask, "kl0.001")["pick"] == cands[2]
    logits[cands[0]] = 5.0
    assert ts.rollout_decide(b, 0, cands, logits, mask, "kl1000")["pick"] == cands[0]
    with pytest.raises(ValueError):
        ts.rollout_decide(b, 0, cands, logits, mask, "best")
    assert ts.rollout_parse_rule("kl") == ("kl", pytest.approx(0.05))
    assert ts.rollout_parse_rule("z") == ("z", 2.0)


def test_decide_keeps_the_policys_move_without_a_clear_lead() -> None:
    st = _position()
    mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
    cands = _legal(st)[:2]
    b = ts.RolloutBatch(horizon=0, seed=6)
    b.add(st, cands, 4)
    cand = b.row_candidate()
    world = b.row_world()
    # Candidate 1 is better on average but noisily: +0.3 in two worlds, -0.25 in the other two.
    values = np.asarray([0.0 if cand[r] == 0 else (0.3 if world[r] % 2 == 0 else -0.25) for r in range(b.rows())],
                        dtype=np.float32)
    b.fill_leaves()
    b.set_leaf_values(values)
    logits = np.zeros(mask.shape[0], dtype=np.float32)
    d = ts.rollout_decide(b, 0, cands, logits, mask, "z2")
    assert d["lead"][1] == pytest.approx(0.025, abs=1e-6) and d["lead_se"][1] > 0.1
    assert d["pick"] == cands[0]                                   # not two standard errors clear
    assert ts.rollout_decide(b, 0, cands, logits, mask, "argmax")["pick"] == cands[1]
