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


def test_the_rule_rolls_on_its_best_target_each_time() -> None:
    col, _ = _spots(1)
    st = col["spots"][0]["state"]
    mover = st.ctx().decision_player
    end, mods = R.drive_rule(st)
    assert mods and mods[0] == col["spots"][0]["best_net"]       # the first roll takes the best modifier
    assert not R._in_realign(end, mover, (st.turn, st.action_round))
    # rule_target agrees with a brute-force reading of its own key on the entered node
    probe = st.clone()
    ts.Engine.step_flat(probe, R.REALIGN)
    R.drain_chance(probe)
    legal = np.flatnonzero(R._realign_targets(probe))
    net = R.net_modifiers(probe, mover)
    opp = influence(probe)[1 if mover == ts.Player.US else 0]
    _, bg, _ = country_table()
    best = max(legal, key=lambda c: (opp[c] > 0, net[c], bool(bg[c]), opp[c]))
    assert R.rule_target(probe, mover) == best


def test_branches_are_scored_per_spot() -> None:
    col, act = _spots(2)
    rows = R.play_spots(col["spots"], pairs=2, act=act, seed=1)
    assert len(rows) == 2
    for r in rows:
        assert "state" not in r and r["pairs"] == 2
        assert all(0.0 <= r[b] <= 1.0 for b in R.BRANCHES)
    md, js = R.report(rows, {"model": "random", **{k: col[k] for k in ("seen", "greedy_realign", "games")}})
    assert "rule − policy" in md and js["groups"]["all"]["n"] == 2
