"""--phase-lock: two arms take turns on one machine (rollout under one lock, update under another).
The lock must exclude a second holder, and an arm under it must train exactly as without it -- only
the wall clock may change."""

from __future__ import annotations

import fcntl
import os
import pathlib

import numpy as np
import pytest
import torch

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.training.nash_pg import NashPGTrainer, phase_lock
from bindings.ts_env import TsVectorizedEnv


def _try_lock(path: str) -> bool:
    """Whether a second open of `path` can take the lock right now -- non-blocking, so the answer
    does not depend on how long anything waits. flock is per open file, so this second open in
    the same process is exactly the second holder the lock has to exclude."""
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def test_the_lock_excludes_a_second_holder(tmp_path: pathlib.Path) -> None:
    """Probed directly rather than timed: this used to hold the lock for 0.3 s of sleep and check
    that a second thread had waited at least 0.25 s, which a loaded machine can break either way."""
    path = str(tmp_path / "arms.update")
    with phase_lock(path) as waited:
        assert waited >= 0.0
        assert not _try_lock(path), "a second holder took the lock while the first held it"
    assert _try_lock(path), "the lock was not released when the block ended"


def test_the_lock_is_released_when_the_block_raises(tmp_path: pathlib.Path) -> None:
    path = str(tmp_path / "arms.update")
    with pytest.raises(RuntimeError):
        with phase_lock(path):
            raise RuntimeError("the update failed")
    assert _try_lock(path)


def _losses(lock: str | None) -> list[float]:
    torch.manual_seed(0)
    np.random.seed(0)
    env = TsVectorizedEnv(num_envs=8, base_seed=99)
    tr = NashPGTrainer(active_net=create_coldwar_net_v2(), env=env, num_envs=8, buffer_size=8, batch_size=32,
                       device="cpu", cuda_graphs=False, phase_lock_path=lock)
    out = []
    for _ in range(2):
        m = tr.train_iteration()
        out.append(float(m["loss"]))
        assert ("phase_lock_wait_update" in m) == (lock is not None)
    return out


def test_an_arm_under_the_lock_trains_exactly_as_without_it(tmp_path: pathlib.Path) -> None:
    assert _losses(str(tmp_path / "arms")) == _losses(None)
