"""P24: league members in the opponent pool, and the directory scan that finds them.

A league member is another run's snapshot -- a main exploiter's for the main agent, the main
agent's newest for the exploiter. It must not disturb the run's own history (spacing eviction, the
resume record), and a pool without any must behave exactly as before.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import torch
import torch.nn as nn

from ai.training.generic_trainer import LEAGUE_SNAPSHOT_RE, scan_league_snapshots
from ai.training.opponent_pool import OpponentPool


class _Tiny(nn.Module):
    def __init__(self, k: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.full((2,), k))

    def forward(self, x: Any) -> Any:  # pragma: no cover - never called
        return x


def _main_pool(league_capacity: int = 2, league_frac: Any = None) -> OpponentPool:
    """A run's own pool of capacity 4, seeded at step 0, ready to take league members."""
    return OpponentPool([_Tiny(0.0)], num_envs=8, frac=0.5, seed=3, capacity=4,
                        league_capacity=league_capacity, league_frac=league_frac)


def test_league_members_do_not_count_against_the_runs_own_capacity() -> None:
    p = _main_pool()
    for n in (5, 10, 15, 20, 25):
        p.add(_Tiny(float(n)), n * 1_000_000)
    p.add(_Tiny(-1.0), 60_000_000, group="league")
    p.add(_Tiny(-2.0), 70_000_000, group="league")
    own = [p.steps[i] for i in p._members("self")]
    assert len(own) == 4 and own[0] == 0 and own[-1] == 25_000_000   # spacing kept the endpoints
    assert [p.steps[i] for i in p._members("league")] == [60_000_000, 70_000_000]


def test_the_league_evicts_its_oldest_added_member_whatever_its_steps() -> None:
    """Exploiter generations restart their step counter, so arrival order is what ages them."""
    p = _main_pool(league_capacity=2)
    p.add(_Tiny(1.0), 90_000_000, group="league")
    p.add(_Tiny(2.0), 70_000_000, group="league")
    p.add(_Tiny(3.0), 80_000_000, group="league")
    assert [p.steps[i] for i in p._members("league")] == [70_000_000, 80_000_000]


def test_league_frac_sets_the_leagues_share_of_draws() -> None:
    p = _main_pool(league_capacity=1, league_frac=0.25)
    for n in (5, 10, 15):
        p.add(_Tiny(float(n)), n * 1_000_000)
    p.add(_Tiny(-1.0), 1, group="league")
    league_id = p.ids[p._members("league")[0]]
    hits = 0
    for _ in range(4000):
        p.start_iteration()
        hits += p.current_id == league_id
    assert 0.21 < hits / 4000 < 0.29


def test_without_league_frac_every_member_is_drawn_alike() -> None:
    p = _main_pool(league_capacity=1)
    for n in (5, 10, 15):
        p.add(_Tiny(float(n)), n * 1_000_000)
    p.add(_Tiny(-1.0), 1, group="league")
    league_id = p.ids[p._members("league")[0]]
    hits = sum(1 for _ in range(4000) if (p.start_iteration() or p.current_id == league_id))
    assert 0.16 < hits / 4000 < 0.24   # 1 of 5 members


def test_the_resume_record_holds_the_runs_own_history_only() -> None:
    p = _main_pool()
    p.add(_Tiny(1.0), 5_000_000)
    p.add(_Tiny(-1.0), 60_000_000, group="league")
    blob = p.state_dict()
    assert blob["steps"] == [0, 5_000_000]
    assert len(blob["ids"]) == 2 and len(blob["paths"]) == 2


def test_league_stats_are_per_seat() -> None:
    p = OpponentPool([_Tiny(0.0)], num_envs=2, frac=1.0, seed=1, groups=["league"],
                     league_capacity=2)
    p.start_iteration()
    p.learner_side[:] = [1, -1]          # env 0: learner is US; env 1: learner is USSR
    p.on_episode_end(0, victory_points=5.0)    # US won: the learner (US) won
    p.on_episode_end(1, victory_points=5.0)    # US won: the learner (USSR) lost
    st = p.stats()
    assert st["opp_league_size"] == 1.0
    assert st["opp_league_win_us"] == 1.0 and st["opp_league_win_ussr"] == 0.0
    assert st["opp_league_games_us"] == 1.0 and st["opp_league_games_ussr"] == 1.0


def test_a_pool_without_league_members_reports_no_league_stats() -> None:
    st = _main_pool().stats()
    assert not any(k.startswith("opp_league") for k in st)


def test_the_scan_takes_snapshots_only_and_waits_for_a_file_to_settle() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = time.time() - 120
        for k, f in enumerate(("snapshot_10000000steps.pt", "E5-03-43-1_snapshot_20000000steps.pt",
                               "pool_15000000steps.pt", "resume_10000000steps.pt",
                               "snapshot_final.pt", "snapshot_0s.pt")):
            open(os.path.join(d, f), "wb").close()
            os.utime(os.path.join(d, f), (old + k, old + k))   # oldest first by mtime
        fresh = os.path.join(d, "snapshot_30000000steps.pt")
        open(fresh, "wb").close()             # just written: left for the next scan
        seen: set = set()
        got = [os.path.basename(q) for q in scan_league_snapshots([d], seen)]
        assert got == ["snapshot_10000000steps.pt", "E5-03-43-1_snapshot_20000000steps.pt"]
        assert scan_league_snapshots([d], seen, settle_seconds=0.0)[-1] == os.path.realpath(fresh)


def test_the_scan_takes_a_snapshot_published_under_two_names_once() -> None:
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as league:
        real = os.path.join(src, "snapshot_40000000steps.pt")
        open(real, "wb").close()
        os.symlink(real, os.path.join(league, "E5-03-43-1_snapshot_40000000steps.pt"))
        seen = {os.path.realpath(real)}
        assert scan_league_snapshots([league], seen, settle_seconds=0.0) == []
        assert LEAGUE_SNAPSHOT_RE.search("E5-03-43-1_snapshot_40000000steps.pt")
