"""The ops-efficient realignment probe: spot detection, the rule's targeting, and the branches."""
import numpy as np
import ts_engine as ts

from ai.eval import realign_spots as R
from ai.eval.ops_block import country_table, influence


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) for m in masks], dtype=np.int32)
    return act


def _probs(st: ts.GameState) -> np.ndarray:
    m = np.asarray(R.ActionEncoder.get_legal_mask(st), dtype=np.float64)
    return m / m.sum()


def _spots(n: int = 2):
    act = _random_policy(0)
    col = R.collect(act, _probs, n, seed=5, envs=8, accept=1.0)
    assert len(col["spots"]) == n
    return col, act


def test_spots_meet_the_rule() -> None:
    col, _ = _spots()
    _, bg, _ = country_table()
    for sp in col["spots"]:
        st = sp["state"]
        assert st.defcon == 2 and st.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
        assert sp["n_plus1"] >= 2 or sp["n_bg2"] >= 1
        assert sp["kind"] == ("bg+2" if sp["n_bg2"] else "multi+1")
    assert R.spot(ts.GameState()) is None


def test_expected_swing() -> None:
    assert R.expected_swing(0, 3, 3) == 0.0                     # symmetric dice, symmetric stakes
    assert R.expected_swing(0, 0, 2) > 0                        # nothing of mine to lose
    assert R.expected_swing(2, 1, 1) > R.expected_swing(1, 1, 1) > 0 > R.expected_swing(-1, 1, 1)
    assert R.expected_swing(-3, 0, 1) > 0                       # a long shot is still free


def test_the_rule_takes_battlegrounds_first_and_stops_when_nothing_is_worth_a_roll() -> None:
    col, _ = _spots(2)
    _, bg, _ = country_table()
    for sp in col["spots"]:
        st = sp["state"]
        mover = st.ctx().decision_player
        side = 0 if mover == ts.Player.US else 1
        probe = st.clone()
        ts.Engine.step_flat(probe, R.REALIGN)
        R.drain_chance(probe)
        legal = np.flatnonzero(R._realign_targets(probe))
        net = R.net_modifiers(probe, mover)
        inf = influence(probe)
        swing = {int(c): R.expected_swing(int(net[c]), int(inf[side, c]), int(inf[1 - side, c])) for c in legal}
        worth = [c for c, v in swing.items() if v > 0]
        got = R.rule_target(probe, mover)
        if not worth:
            assert got is None
            continue
        assert got == max(worth, key=lambda c: (bool(bg[c]), swing[c]))
        if any(bg[c] for c in worth):
            assert bg[got]
        end, mods = R.drive_rule(st)
        assert len(mods) <= sp["ops"]
        assert not R._in_realign(end, mover, (st.turn, st.action_round))


def test_branches_are_scored_per_spot() -> None:
    col, act = _spots(2)
    rows = R.play_spots(col["spots"], pairs=2, act=act, seed=1)
    assert len(rows) == 2
    for r in rows:
        assert "state" not in r and r["pairs"] == 2
        assert all(0.0 <= r[b] <= 1.0 for b in R.BRANCHES)
    md, js = R.report(rows, {"model": "random", **{k: col[k] for k in ("seen", "greedy_realign", "games")}})
    assert "rule − policy" in md and js["groups"]["all"]["n"] == 2
