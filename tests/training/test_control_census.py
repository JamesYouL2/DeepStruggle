"""The control-break census (tools/scripts/control_census.py): a point's class agrees with the engine's
control, every point into the opponent's control is accounted for by a break play, the no-break
steer removes exactly the opponent's countries for the rest of the mover's play, and the paired
playouts score both lines of every pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import ts_engine as ts
from ai.eval.paired_playouts import apply_move, ar_key, decider
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import load_policy, selfplay
from tools.scripts.control_census import (BREAK, CHIP, CUBA, NODE, OPEN, BreakTracker, NoBreak, avoid_of,
                                          nobreak_compare, opponent, ops_influence, point_class, ratio_se)


def _checkpoint(tmp_path: Path) -> str:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    path = tmp_path / "m.pt"
    torch.save(m.state_dict(), str(path))
    return str(path)


def _us_ops_influence_with_a_us_target_held_by_the_ussr() -> ts.GameState:
    """A US Ops influence decision with North Korea (USSR 3, stability 3) among its targets."""
    runner = ts.VectorizedBatchRunner(32, 777)
    rng = np.random.default_rng(0)
    for _ in range(400):
        for i in range(32):
            st = runner.get_state(i)
            if ops_influence(st) and decider(st) == ts.Player.US:
                mask = np.asarray(ActionEncoder.get_legal_mask(st))
                if any(mask[NODE + c] and ts.Scoring.get_country_control(st, c) == ts.Player.USSR for c in range(84)):
                    return st
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks], True)
    raise AssertionError("no US Ops influence decision with a USSR-controlled target")


def test_point_class_follows_the_engines_control_and_the_cost_of_one_more_point() -> None:
    st = _us_ops_influence_with_a_us_target_held_by_the_ussr()
    mask = np.asarray(ActionEncoder.get_legal_mask(st))
    seen = set()
    for a in np.flatnonzero(mask):
        cid = int(ts.decode_flat_action(st, int(a)).primary_id)
        assert int(a) == NODE + cid
        k = point_class(st, ts.Player.US, cid)
        held = ts.Scoring.get_country_control(st, cid) == ts.Player.USSR
        assert (k != OPEN) == held
        if held:
            after = apply_move(st, int(a))
            assert (ts.Scoring.get_country_control(after, cid) == ts.Player.USSR) == (k == CHIP)
        seen.add(k)
    assert BREAK in seen or CHIP in seen


def test_nobreak_steer_removes_the_opponents_countries_for_the_rest_of_the_play_only() -> None:
    st = _us_ops_influence_with_a_us_target_held_by_the_ussr()
    mask = np.asarray(ActionEncoder.get_legal_mask(st)).copy()
    w = NoBreak({0: (ts.Player.US, ar_key(st), None)})
    steered = w.steer(0, st, mask)
    removed = set(np.flatnonzero(mask)) - set(np.flatnonzero(steered))
    assert removed and all(ts.Scoring.get_country_control(st, int(a) - NODE) == opponent(ts.Player.US)
                           for a in removed)
    assert all(ts.Scoring.get_country_control(st, int(a) - NODE) != ts.Player.USSR for a in np.flatnonzero(steered))
    assert w.changed[0] == 1
    later = apply_move(st, int(np.flatnonzero(steered)[0]))
    while ar_key(later) == ar_key(st) and not ts.Engine.is_terminal(later):
        later = apply_move(later, int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(later)))[0]))
    m2 = np.asarray(ActionEncoder.get_legal_mask(later)).copy()
    assert (w.steer(0, later, m2) == m2).all() and not w.wants(0)   # the next round is the policy's own


def test_a_cuba_row_steers_out_of_cuba_only() -> None:
    st = _us_ops_influence_with_a_us_target_held_by_the_ussr()
    mask = np.asarray(ActionEncoder.get_legal_mask(st)).copy()
    assert mask[NODE + CUBA]                                   # the US can always reach Cuba
    avoid = avoid_of({"kind": "cuba"})
    assert avoid == frozenset({CUBA}) and avoid_of({"kind": "break"}) is None and avoid_of({}) is None
    w = NoBreak({0: (ts.Player.US, ar_key(st), avoid)})
    steered = w.steer(0, st, mask)
    assert set(np.flatnonzero(mask)) - set(np.flatnonzero(steered)) == {NODE + CUBA}


def _census(tmp_path: Path, games: int) -> List[BreakTracker]:
    fn, features = load_policy(_checkpoint(tmp_path))
    return selfplay(fn, features, games, 5, games, 1.0, lambda: BreakTracker(bank=True))   # varied play


def test_every_point_into_opponent_control_belongs_to_a_break_play_followed_to_its_end(tmp_path: Path) -> None:
    trackers = _census(tmp_path, 8)
    plays = [p for t in trackers for p in t.result()["plays"]]
    assert plays
    for t in trackers:
        res = t.result()
        into = sum(1 for r in res["points"] if r["k"] != OPEN)
        # each point into control costs 2 and every later point of the play into that country 1
        assert into == sum(e["ops"] - e["pts"] for p in res["plays"] for e in p["countries"])
    for t in trackers:
        for cp in t.result()["cuba"]:
            assert cp["fidel_at"] in ("unseen", "discard", "US hand") and cp["pts"] >= 1 and cp["raw"] == NODE + CUBA
            assert (cp["fired"] is None) == (cp["us_when_fired"] is None)
    for p in plays:
        assert p["countries"][0]["ops"] >= 2 and isinstance(p["pos"], str) and p["raw"] >= NODE
        for e in p["countries"]:
            assert e["end"] in ("flipped", "broken", "held", None)
            if e["end"] == "held":
                assert e["retaken"] is None and e["turn_end"] is None


def test_paired_playouts_score_the_models_play_and_the_no_break_line_on_every_pair(tmp_path: Path) -> None:
    trackers = _census(tmp_path, 8)
    p = next(p for t in trackers for p in t.result()["plays"])
    row: Dict[str, Any] = {"id": "0123456789abcdef", "pos": p["pos"], "raw": p["raw"]}
    fn, _ = load_policy(str(tmp_path / "m.pt"))

    def act(obs: Any, masks: Any) -> Any:
        return np.asarray(fn(obs, masks)).argmax(axis=1).astype(np.int32)

    (got,) = nobreak_compare([row], act, 3, 11)
    assert len(got["scores"]["raw"]) == len(got["scores"]["steered"]) == 3
    assert all(s in (0.0, 0.5, 1.0) for line in got["scores"].values() for s in line)
    assert got["changed"] + got["forced"] >= 1          # the first decision is steered in every pair


def test_ratio_standard_error_is_clustered_by_game() -> None:
    r, se = ratio_se({"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 4.0})
    assert abs(r - 4 / 6) < 1e-12
    # residuals 1 - 2r and 3 - 4r, scaled by g / (g - 1)
    assert abs(se - np.sqrt(2 * ((1 - 2 * r) ** 2 + (3 - 4 * r) ** 2)) / 6) < 1e-12
