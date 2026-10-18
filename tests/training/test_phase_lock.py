"""--phase-lock: two arms take turns on one machine (rollout under one lock, update under another).
The lock must exclude a second holder, and an arm under it must train exactly as without it -- only
the wall clock may change."""

from __future__ import annotations

import pathlib
import threading
import time

import numpy as np
import torch

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.training.nash_pg import NashPGTrainer, phase_lock
from bindings.ts_env import TsVectorizedEnv


def test_the_lock_excludes_a_second_holder(tmp_path: pathlib.Path) -> None:
    path = str(tmp_path / "arms.update")
    held = threading.Event()
    waited: list[float] = []

    def second() -> None:
        held.wait()
        with phase_lock(path) as w:
            waited.append(w)
    t = threading.Thread(target=second)
    t.start()
    with phase_lock(path):
        held.set()
        time.sleep(0.3)
    t.join()
    assert waited and waited[0] >= 0.25


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
