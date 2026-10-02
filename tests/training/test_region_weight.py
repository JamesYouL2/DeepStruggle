"""Region weight: placements are tallied by region and status, and a restricted play stays in its region."""
import numpy as np
import ts_engine as ts

from ai.eval import region_weight as R
from ai.eval.branch_oracle import _decider
from ai.eval.ops_block import NODE_OFFSET, influence
from ai.eval.playout_audit import play_safe
from ai.eval.reply_probe import _ar_key


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def test_region_masks_cover_the_map_once() -> None:
    total = np.sum([R.region_mask(r).astype(int) for r in R.REGIONS], axis=0)
    assert total[NODE_OFFSET:NODE_OFFSET + 84].tolist() == [1] * 84 and total.sum() == 84


def test_collect_tally_and_restricted_play() -> None:
    act = _random_policy(0)
    starts, counts, games = R.collect(act, 8, seed=2, envs=4, accept=0.5)
    assert games >= 1 and counts and starts
    for k in counts:
        side, reg, era, code = k.split("|")
        assert side in ("US", "USSR") and reg in R.REGIONS and len(code) == len(R.REGIONS)
    st = starts[0]
    mover, card = _decider(st), int(st.ctx().pending_op_card)
    region = next(r for r in R.REGIONS if (R.region_mask(r) & np.asarray(ts_mask(st)).astype(bool)).any())
    # One step of a restricted play: the point lands in the region.
    s = st.clone()
    play_safe([s], [mover], [_ar_key(st)], act, seed=1, max_steps=1, restrict=[(card, int(mover), R.region_mask(region))])
    rows = R.play(act, [st], pairs=2, seed=1)
    assert rows and region in rows[0]["regions"]
    md, summary = R.report(counts, counts, rows, {"model": "x", "pairs": 2})
    assert "Region weight" in md


def ts_mask(st: ts.GameState) -> np.ndarray:
    from bindings.action_encoder import ActionEncoder
    return np.asarray(ActionEncoder.get_legal_mask(st))


def test_restriction_keeps_the_play_in_the_region() -> None:
    """play_safe's restriction: one step of a restricted influence play places in the region."""
    act = _random_policy(3)
    starts, _, _ = R.collect(act, 6, seed=4, envs=4, accept=0.5)
    checked = 0
    for st in starts:
        mover, card = _decider(st), int(st.ctx().pending_op_card)
        side = 0 if mover == ts.Player.US else 1
        for region in R.REGIONS:
            if not (R.region_mask(region) & ts_mask(st).astype(bool)).any():
                continue
            runner = ts.VectorizedBatchRunner(1, 1)
            runner.set_state(0, st)
            runner.refresh_all()
            before = influence(st)
            m = np.asarray(runner.get_action_masks())[0].astype(bool) & R.region_mask(region)
            runner.step_flat_all([int(np.flatnonzero(m)[0])], auto_advance=True)
            after = influence(runner.get_state(0))
            added = np.flatnonzero(after[side] > before[side])
            assert len(added) == 1 and R.region_of()[int(added[0])] == region
            checked += 1
    assert checked
