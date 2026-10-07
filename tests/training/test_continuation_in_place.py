"""A run is one directory: an unchanged continuation writes into it, and its metadata keeps every leg.

Before 2026-10-07 every continuation opened a `<name>_<timestamp>` directory, so one run spread over
several -- E7-A4-R1-S44 over seven -- with its metrics and TensorBoard split along the way. Writing
into the run's own directory is only right for the run itself, continued from where it stopped; a
branch (another name) or a re-run from an earlier state would collide with what is there.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from ai.training.generic_trainer import (RESUME_FILENAME, continuation_dir, record_start_steps,
                                         with_leg)

NAME = "E9-A1-R1-S01"


def _run_dir(root: Any, name: str = NAME, ts: str = "20261007_120000", **meta: Any) -> str:
    d = os.path.join(str(root), f"{name}_{ts}")
    os.makedirs(d)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump({"run_name": name, "train_steps": 100, **meta}, f)
    for fname in (RESUME_FILENAME, "resume_50steps.pt", "resume_100steps.pt"):
        open(os.path.join(d, fname), "wb").close()
    return d


def test_the_run_continued_under_its_own_name_stays_in_its_directory(tmp_path: Any) -> None:
    d = _run_dir(tmp_path)
    assert continuation_dir(d, NAME) == d
    assert continuation_dir(os.path.join(d, RESUME_FILENAME), NAME) == d
    assert continuation_dir(f"{d}:100", NAME) == d            # its newest per-snapshot state


def test_a_branch_or_an_earlier_state_gets_its_own_directory(tmp_path: Any) -> None:
    d = _run_dir(tmp_path)
    assert continuation_dir(d, NAME + "@50M+S02") is None      # another run
    assert continuation_dir(f"{d}:50", NAME) is None           # later snapshots would collide
    assert continuation_dir(None, NAME) is None
    assert continuation_dir(d, None) is None


def test_an_old_runs_link_directory_is_never_written_into(tmp_path: Any) -> None:
    """run_codes.py --link: every file a symlink into the original. Writing there would append to
    the original's metrics through the link."""
    real = _run_dir(tmp_path, name="E7-08-43")
    link = os.path.join(str(tmp_path), f"{NAME}_20261001_164848")
    os.makedirs(link)
    for f in os.listdir(real):
        os.symlink(os.path.join(real, f), os.path.join(link, f))
    assert continuation_dir(link, NAME) is None


def test_a_run_that_is_still_writing_is_refused(tmp_path: Any) -> None:
    d = _run_dir(tmp_path)
    with open(os.path.join(d, "run.pid"), "w") as f:
        f.write(f"{os.getppid()}\n")                           # alive, and not this process
    with pytest.raises(RuntimeError, match="still being written"):
        continuation_dir(d, NAME)
    with open(os.path.join(d, "run.pid"), "w") as f:
        f.write("999999999\n")                                 # gone
    assert continuation_dir(d, NAME) == d


def test_metadata_keeps_every_leg() -> None:
    first = {"run_name": NAME, "train_steps": 100, "base_commit": "aaa", "description": "first"}
    second = {"run_name": NAME, "train_steps": 200, "base_commit": "bbb", "description": "second"}
    third = {"run_name": NAME, "train_steps": 300, "base_commit": "ccc", "description": "third"}
    m = with_leg(first, second)
    assert m["train_steps"] == 200 and m["base_commit"] == "bbb"   # the top level is the current leg
    assert [leg["base_commit"] for leg in m["legs"]] == ["aaa", "bbb"]
    m = with_leg(m, third)
    assert [leg["description"] for leg in m["legs"]] == ["first", "second", "third"]
    assert all("legs" not in leg for leg in m["legs"])


def test_the_start_step_is_recorded_on_the_current_leg(tmp_path: Any) -> None:
    path = os.path.join(str(tmp_path), "metadata.json")
    with open(path, "w") as f:
        json.dump(with_leg({"train_steps": 100}, {"train_steps": 200}), f)
    record_start_steps(path, 100)
    with open(path) as f:
        m = json.load(f)
    assert m["start_steps"] == 100 and m["legs"][-1]["start_steps"] == 100
    assert "start_steps" not in m["legs"][0]
