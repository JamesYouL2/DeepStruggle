"""Split a tournament's games across processes.

`BatchMatchRunner` batches a matchup's games inside one process, which is all a network needs:
its cost is one forward pass per step for every game at once. A bot that decides in Python
(Doctrine: ~20 s a game) gains nothing from that batching -- its games run one decision at a
time on one core. This module cuts every matchup into shards of game pairs and plays the shards
in a pool of worker processes, one core each.

**The split does not change the games.** A shard plays the same pairs, with the same deals, as the
unsplit matchup (`play_parallel_matchup(pairs=...)`), and the shard results merge into the result
one call would have produced (`merge_matchup_results`). What a shard cannot inherit is the state
an agent carries between games -- Doctrine's determinization stream, a sampling generator -- so
every shard starts those from a seed of its own (`shard_seed`), and every agent offering
`reseed(seed)` is reseeded. The outcome of a game then depends on its shard, never on the worker
that ran it or on what that worker ran before: at a fixed `shard_pairs`, any number of workers
gives the same results. A torch policy sampling above the greedy threshold draws from torch's
generator, which is seeded per shard as well.

Workers are started with `spawn`, not `fork`: the parent has torch and the engine's OpenMP pool
loaded, and a forked child of a process with live OpenMP threads can deadlock in libgomp. Each
worker is pinned to one thread (OMP_NUM_THREADS=1 in its environment, torch.set_num_threads(1)),
so N workers use N cores rather than N times the machine.
"""
from __future__ import annotations

import multiprocessing
import os
import random
import shutil
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tools.lib.batch_tournament import BatchMatchRunner, merge_matchup_results
from tools.lib.player_agent import PlayerAgent, load_agent

Pair = Tuple[int, int]


@dataclass(frozen=True)
class Shard:
    """A run of consecutive game pairs of one matchup."""

    pair: Pair                  # (i, j): indices of the two entrants in the tournament's list
    index: int                  # position of this shard within its matchup
    pairs: Tuple[int, ...]      # game pair indices, into range(games_per_side)


def plan_shards(matchups: Sequence[Pair], games_per_side: int, shard_pairs: int) -> List[Shard]:
    """Every matchup cut into shards of `shard_pairs` game pairs (the last may be shorter)."""
    if shard_pairs <= 0:
        raise ValueError(f"shard_pairs must be positive, got {shard_pairs}")
    shards: List[Shard] = []
    for pair in matchups:
        for n, lo in enumerate(range(0, games_per_side, shard_pairs)):
            ks = tuple(range(lo, min(lo + shard_pairs, games_per_side)))
            shards.append(Shard(pair=pair, index=n, pairs=ks))
    return shards


def shard_seed(base_seed: int, shard: Shard) -> int:
    """The seed a shard's agents and samplers start from. A function of the shard alone."""
    i, j = shard.pair
    return (base_seed * 1_000_003 + i * 65_537 + j * 257 + shard.pairs[0]) % (2 ** 31 - 1)


def seed_shard(seed: int, agents: Sequence[PlayerAgent]) -> None:
    """Start every generator a shard's games can draw from at `seed`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for n, agent in enumerate(agents):
        reseed = getattr(agent, "reseed", None)
        if callable(reseed):
            reseed(seed + n)


# -- worker side ------------------------------------------------------------------------------

_AGENTS: Dict[str, PlayerAgent] = {}
_DEVICE: str = "cpu"


def _init_worker(device: str) -> None:
    global _DEVICE
    _DEVICE = device
    torch.set_num_threads(1)


def _agent(spec: str) -> PlayerAgent:
    # Loaded once per worker: a checkpoint is read from disk once, not once per shard. Nothing an
    # agent carries between games survives into a shard's results, because seed_shard restarts it.
    if spec not in _AGENTS:
        _AGENTS[spec] = load_agent(spec, device=_DEVICE)
    return _AGENTS[spec]


def _run_shard(spec_a: str, spec_b: str, shard: Shard, games_per_side: int, base_seed: int,
               options: Dict[str, Any], log_path: Optional[str]) -> Dict[str, Any]:
    a, b = _agent(spec_a), _agent(spec_b)
    seed_shard(shard_seed(base_seed, shard), (a, b))
    return BatchMatchRunner.play_parallel_matchup(
        a, b, games_per_side=games_per_side, base_seed=base_seed, device=_DEVICE,
        pairs=shard.pairs, log_games_file=log_path, **options)


# -- driver side ------------------------------------------------------------------------------

def _shard_log(log_games: str, shard: Shard) -> str:
    i, j = shard.pair
    return f"{log_games}.part-{i}-{j}-{shard.index}"


def run_matchups_parallel(
    specs: Sequence[str],
    matchups: Sequence[Pair],
    games_per_side: int,
    workers: int,
    shard_pairs: int,
    *,
    device: str = "cpu",
    base_seed: int = 10000,
    temperature: float = 0.1,
    batch_chunk_size: int = 1000,
    track_choices: bool = False,
    log_games: Optional[str] = None,
    auto_advance: bool = True,
    on_matchup_done: Optional[Callable[[Pair, Dict[str, Any]], None]] = None,
) -> Dict[Pair, Dict[str, Any]]:
    """Play every matchup's games over `workers` processes; one merged result per matchup.

    `specs` are `load_agent` specs, loaded afresh in each worker. `on_matchup_done` is called
    in the parent as each matchup's last shard lands, so progress is reported as it happens.
    With `log_games`, the per-game log holds every matchup's games, matchup by matchup and
    shard by shard.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    shards = plan_shards(matchups, games_per_side, shard_pairs)
    options: Dict[str, Any] = dict(temperature=temperature, batch_chunk_size=batch_chunk_size,
                                   track_choices=track_choices, auto_advance=auto_advance)
    remaining: Dict[Pair, int] = {}
    for s in shards:
        remaining[s.pair] = remaining.get(s.pair, 0) + 1
    parts: Dict[Pair, List[Tuple[int, Dict[str, Any]]]] = {p: [] for p in remaining}
    results: Dict[Pair, Dict[str, Any]] = {}

    # The workers inherit this environment when they are spawned, and the engine and torch read
    # it when their OpenMP runtime starts -- which in the parent has already happened.
    saved = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker, initargs=(device,)) as pool:
            futures: Dict[Future[Dict[str, Any]], Shard] = {}
            for s in shards:
                i, j = s.pair
                log_path = _shard_log(log_games, s) if log_games else None
                futures[pool.submit(_run_shard, specs[i], specs[j], s, games_per_side,
                                    base_seed, options, log_path)] = s
            for fut in as_completed(futures):
                s = futures[fut]
                parts[s.pair].append((s.index, fut.result()))
                remaining[s.pair] -= 1
                if remaining[s.pair] == 0:
                    ordered = [r for _, r in sorted(parts[s.pair], key=lambda t: t[0])]
                    results[s.pair] = merge_matchup_results(ordered)
                    if on_matchup_done is not None:
                        on_matchup_done(s.pair, results[s.pair])
    finally:
        if saved is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = saved

    if log_games:
        os.makedirs(os.path.dirname(os.path.abspath(log_games)), exist_ok=True)
        with open(log_games, "w", encoding="utf-8") as out:
            for s in shards:
                part = _shard_log(log_games, s)
                with open(part, "r", encoding="utf-8") as f:
                    shutil.copyfileobj(f, out)
                os.remove(part)
    return {p: results[p] for p in matchups}
