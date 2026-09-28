"""--setup-mc-credit: setup placements are credited with their game's result, not the lambda-return."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ai.models import create_coldwar_net
from ai.training import NashPGTrainer
from ai.training.rollout_buffer import setup_phase_slot


def _trainer(**kw: Any) -> NashPGTrainer:
    from bindings.ts_env import TsVectorizedEnv
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_coldwar_net(dev)
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    # 16 steps from a fresh game: almost all of them are setup placements.
    return NashPGTrainer(active_net=model, env=env, num_envs=8, buffer_size=16, lr=3e-4, eta=0.1,
                         ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


def test_off_is_bitwise_the_control() -> None:
    torch.manual_seed(1)
    a = _trainer()
    a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(setup_mc_credit=False)
    b.train_iteration()
    for pa, pb in zip(a.active_net.parameters(), b.active_net.parameters()):
        assert torch.equal(pa, pb)


def test_setup_placements_wait_for_their_game() -> None:
    t = _trainer(setup_mc_credit=True, setup_mc_min_batch=10_000)
    m = t.train_iteration()
    pending = sum(len(p) for p in t._setup_pending)
    # 8 fresh games, 15 setup placements each, all within the first 16 steps
    assert pending == 8 * 15, pending
    assert m["setup_mc_ready"] == 0.0 and "setup_mc_n" not in m
    assert all(int(r[5]) in (1, -1) for p in t._setup_pending for r in p)


def test_a_finished_game_gives_each_placement_its_movers_result() -> None:
    t = _trainer(setup_mc_credit=True, setup_mc_min_batch=10_000)
    t.train_iteration()
    n0 = len(t._setup_pending[0])
    players = [int(r[5]) for r in t._setup_pending[0]]
    t._setup_mc_resolve([{"env_idx": 0, "terminal_utility": 1.0}])       # the US won
    assert t._setup_pending[0] == [] and len(t._setup_ready) == n0
    assert [r[6] for r in t._setup_ready] == [1.0 if p == 1 else -1.0 for p in players]


def _mc_shift(result: float) -> np.ndarray:
    """Change in log-prob of six recorded setup placements after one MC step crediting them all
    with `result` against a zero baseline."""
    t = _trainer(setup_mc_credit=True, setup_mc_min_batch=4)
    t.train_iteration()
    recs = t._setup_pending[0][:6]
    obs = torch.stack([r[0] for r in recs])
    masks = torch.stack([r[1] for r in recs])
    act = torch.stack([r[2] for r in recs]).long()
    with torch.no_grad():
        before = F.log_softmax(t.active_net(obs, masks)[0].float(), -1).gather(1, act[:, None]).squeeze(1)
    t._setup_ready = [r[:4] + (torch.tensor(0.0),) + r[5:] + (result,) for r in recs]
    m = t._setup_mc_update()
    assert m["setup_mc_n"] == 6.0 and t._setup_ready == []
    with torch.no_grad():
        after = F.log_softmax(t.active_net(obs, masks)[0].float(), -1).gather(1, act[:, None]).squeeze(1)
    return (after - before).numpy()


def test_the_mc_step_moves_toward_won_placements_and_away_from_lost_ones() -> None:
    # On average: one Adam step on six related positions is large and shared across them, so an
    # individual placement can move against its own credit.
    assert _mc_shift(1.0).mean() > 0
    assert _mc_shift(-1.0).mean() < 0


def test_setup_rows_leave_the_ordinary_surrogate() -> None:
    # With every buffered row a setup placement, the ordinary surrogate has nothing to train on,
    # so turning MC credit on must change the update (the surrogate's share is gone).
    torch.manual_seed(1)
    a = _trainer(setup_mc_min_batch=10_000)
    a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(setup_mc_credit=True, setup_mc_min_batch=10_000)
    b.train_iteration()
    obs = np.asarray(b.buffer.obs.view(-1, b.buffer.obs_dim).cpu())
    assert (obs[:, setup_phase_slot()] < 0.5 / 6.0).mean() > 0.9
    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.active_net.parameters(), b.active_net.parameters()))


def test_the_cli_has_it_off_by_default() -> None:
    from ai.training.train import build_parser
    a = build_parser().parse_args([])
    assert (a.setup_mc_credit, a.setup_mc_coef, a.setup_mc_min_batch) == (False, 1.0, 512)
