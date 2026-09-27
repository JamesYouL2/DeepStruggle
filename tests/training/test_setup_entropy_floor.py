"""--setup-entropy-floor: an adaptive entropy floor on the learner's setup placements.

Off at 0, and off must leave the update exactly what it was. On, its coefficient rises only while
the rollout's setup entropy is under the floor, and the bonus reaches the parameters."""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from ai.models import create_coldwar_net
from ai.training import NashPGTrainer
from ai.training.nash_pg import BaseNashPGTrainer


def _floor(floor: float = 0.4, lr: float = 0.01, max_coef: float = 1.0, coef: float = 0.0) -> Any:
    return types.SimpleNamespace(setup_entropy_floor=floor, setup_entropy_lr=lr,
                                 setup_entropy_max_coef=max_coef, setup_ent_coef=coef)


def test_the_coefficient_stays_zero_while_the_setup_is_above_the_floor() -> None:
    t = _floor()
    for _ in range(50):
        BaseNashPGTrainer._update_setup_entropy(t, 1.5)
    assert t.setup_ent_coef == 0.0


def test_the_coefficient_rises_under_the_floor_and_is_capped() -> None:
    t = _floor(lr=0.1, max_coef=0.3)
    BaseNashPGTrainer._update_setup_entropy(t, 0.0)
    assert t.setup_ent_coef == pytest.approx(0.04)
    for _ in range(100):
        BaseNashPGTrainer._update_setup_entropy(t, 0.0)
    assert t.setup_ent_coef == pytest.approx(0.3)


def test_the_coefficient_decays_back_once_the_setup_recovers() -> None:
    t = _floor(coef=0.05)
    for _ in range(100):
        BaseNashPGTrainer._update_setup_entropy(t, 1.0)
    assert t.setup_ent_coef == 0.0


def _trainer(**kw: Any) -> NashPGTrainer:
    from bindings.ts_env import TsVectorizedEnv
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_coldwar_net(dev)
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    # 16 steps from a fresh game: almost all of them are setup placements.
    return NashPGTrainer(active_net=model, env=env, num_envs=8, buffer_size=16, lr=3e-4, eta=0.1,
                         ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


def test_off_is_bitwise_the_control_and_setup_entropy_is_logged_either_way() -> None:
    torch.manual_seed(1)
    a = _trainer()
    ma = a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(setup_entropy_floor=0.0)
    b.train_iteration()
    for pa, pb in zip(a.active_net.parameters(), b.active_net.parameters()):
        assert torch.equal(pa, pb)
    assert ma["entropy_setup"] > 0.0 and "setup_ent_coef" not in ma


def test_a_binding_floor_moves_the_coefficient_and_the_parameters() -> None:
    torch.manual_seed(1)
    a = _trainer()
    a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(setup_entropy_floor=50.0, setup_entropy_lr=0.01)   # far above any entropy
    b.setup_ent_coef = 0.5                                            # as if already risen
    mb = b.train_iteration()
    assert mb["setup_ent_coef"] > 0.5
    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.active_net.parameters(),
                                                       b.active_net.parameters()))


def test_negative_settings_are_refused() -> None:
    with pytest.raises(ValueError):
        _trainer(setup_entropy_floor=-0.1)


def test_the_cli_has_the_floor_off_by_default() -> None:
    from ai.training.train import build_parser
    a = build_parser().parse_args([])
    assert (a.setup_entropy_floor, a.setup_entropy_lr, a.setup_entropy_max_coef) == (0.0, 0.01, 1.0)
