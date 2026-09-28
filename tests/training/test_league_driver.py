"""tools/scripts/league.py: which resume state a main-latest exploiter generation starts from."""

from __future__ import annotations

import os

from tools.scripts.league import latest_resume


def _touch(d: str, name: str) -> None:
    with open(os.path.join(d, name), "w") as f:
        f.write("x")


def test_latest_resume_takes_the_newest_tagged_state_and_ignores_the_rolling_one(tmp_path) -> None:
    d = str(tmp_path)
    for n in ("resume_10027008steps.pt", "resume_40042496steps.pt", "resume_state.pt",
              "snapshot_50000000steps.pt", "resume_9000000steps.pt"):
        _touch(d, n)
    assert latest_resume(d) == os.path.join(d, "resume_40042496steps.pt")


def test_latest_resume_waits_for_the_start_threshold(tmp_path) -> None:
    d = str(tmp_path)
    _touch(d, "resume_10027008steps.pt")
    assert latest_resume(d, 40_000_000) is None
    _touch(d, "resume_40042496steps.pt")
    assert latest_resume(d, 40_000_000) == os.path.join(d, "resume_40042496steps.pt")
