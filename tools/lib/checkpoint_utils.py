"""Checkpoint discovery: every .pt file under a directory, with its size and modification time.

Discovery does not load anything. A run directory also holds `resume_*.pt` files, which are not
policies, so a loader here had to swallow their errors -- and swallowed every other load error
with them. The architecture of a checkpoint is read by `tools/inspect_checkpoints.py`, through the
same loader every evaluation uses (`tools.lib.player_agent.NeuralAgent`).
"""

from __future__ import annotations

import os
import time
from typing import List, TypedDict


class CheckpointEntry(TypedDict):
    path: str
    filename: str
    dir: str
    size_mb: float
    mtime: str


def checkpoint_entry(path: str) -> CheckpointEntry:
    """A checkpoint file's location, size and modification time."""
    stat = os.stat(path)
    return {
        "path": path,
        "filename": os.path.basename(path),
        "dir": os.path.dirname(path),
        "size_mb": stat.st_size / (1024 * 1024),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }


def discover_checkpoints(directory: str) -> List[CheckpointEntry]:
    """Every .pt file under `directory` (recursively), sorted by path.

    Raises FileNotFoundError for a directory that does not exist, which is a wrong path rather
    than a directory with no checkpoints in it.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")
    checkpoints: List[CheckpointEntry] = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".pt"):
                checkpoints.append(checkpoint_entry(os.path.join(root, f)))
    checkpoints.sort(key=lambda x: x["path"])
    return checkpoints
