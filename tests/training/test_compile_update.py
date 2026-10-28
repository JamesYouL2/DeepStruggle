"""--compile-update (P26): off is the eager update exactly; on, only the update's forwards go through
torch.compile, the rollout does not, and training runs.

Only the test that trains through the compiled nets needs a GPU (inductor's Triton backend). The
rest -- which net the update uses, lazy wrapping, refusing an unknown mode -- never runs a compiled
forward, so they run on the CPU; the module used to be GPU-only as a whole, which checked none of
it on a machine without one."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from ai.models import create_coldwar_net
from ai.training import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _trainer(device: str = "cpu", **kw: Any) -> NashPGTrainer:
    torch.manual_seed(0)
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    return NashPGTrainer(active_net=create_coldwar_net(torch.device(device)), env=env, num_envs=8,
                         buffer_size=16, batch_size=64, device=device, **kw)


def test_off_returns_the_network_itself() -> None:
    t = _trainer()
    assert t._update_net(t.active_net) is t.active_net


def test_on_compiles_the_update_nets_lazily_and_follows_a_swapped_net() -> None:
    t = _trainer(compile_update="default")
    c = t._update_net(t.active_net)
    assert c is not t.active_net and getattr(c, "_orig_mod") is t.active_net
    assert t._update_net(t.active_net) is c
    other = create_coldwar_net(torch.device("cpu"))
    assert getattr(t._update_net(other), "_orig_mod") is other


@_needs_gpu
def test_on_trains_and_the_rollout_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    t = _trainer("cuda", compile_update="default")
    before = [p.detach().clone() for p in t.active_net.parameters()]
    calls: list[str] = []
    orig = t._update_net
    monkeypatch.setattr(t, "_update_net", lambda net: calls.append("u") or orig(net))
    t.collect_rollouts()
    assert calls == [], "the rollout must not go through the compiled nets"
    m = t.train_step()
    assert calls, "the update must"
    assert all(v == v for v in m.values() if isinstance(v, float))
    assert any(not torch.equal(a, b) for a, b in zip(before, t.active_net.parameters()))


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        _trainer(compile_update="reduce-overhead")
