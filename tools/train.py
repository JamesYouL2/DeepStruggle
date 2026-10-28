#!/usr/bin/env python3
"""Unified Training Tool for Twilight Struggle Neural Policies (BC / NashPG / Blunder-Aware RL)."""

import os
import sys

# Idle OpenMP workers sleep instead of spinning. Both PyTorch and the engine's batch runner use
# OpenMP, and with the default policy their idle workers spun between parallel regions: ~8.5 cores
# per training process for the same throughput that passive waiting reaches on ~1.6
# (research/log/training_throughput_cpu.md). Must be set before torch or ts_engine loads libgomp;
# an explicit OMP_WAIT_POLICY in the environment still wins.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine  # noqa: E402, F401 -- the build require_fresh() checks before training starts
from ai.training.train import main  # noqa: E402
from tools.lib.engine_fingerprint import require_fresh  # noqa: E402

if __name__ == "__main__":
    # Key invariant 13: a checkpoint trained on a stale engine keeps loading and running later,
    # so nothing announces it. check_engine_fresh.sh guards a shell launch; this guards every one.
    require_fresh()
    main()
