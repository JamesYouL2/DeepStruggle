"""Split a tournament's games across processes.

`BatchMatchRunner` batches a matchup's games inside one process, which is all a network needs:
its cost is one forward pass per step for every game at once. A bot that decides in Python
(`heuristic_mcts`, for one) gains nothing from that batching -- its games run one decision at a
time on one core. This module cuts every matchup into shards of game pairs and plays the shards
in a pool of worker processes, one core each.

**The split does not change the games.** A shard plays the same pairs, with the same deals, as the
unsplit matchup (`play_parallel_matchup(pairs=...)`), and the shard results merge into the result
one call would have produced (`merge_matchup_results`). What a shard cannot inherit is the state
an agent carries between games -- a search's determinization stream, a sampling generator -- so
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

import json
import multiprocessing
import os
import random
import shutil
import tempfile
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

@dataclass
class ShardResult:
    shard: Shard
    result: Dict[str, Any]
    games: List[Dict[str, Any]]      # the shard's per-game log records, when logs were asked for


def select_part(shards: Sequence[Shard], index: int, count: int) -> List[Shard]:
    """Part `index` of `count` (0-based): every count-th shard. Striding rather than cutting into
    blocks spreads every matchup, and each matchup's slow and fast pairs, over all the parts."""
    if not 0 <= index < count:
        raise ValueError(f"part index must be in range({count}), got {index}")
    return list(shards[index::count])


def run_shards(
    specs: Sequence[str],
    shards: Sequence[Shard],
    games_per_side: int,
    workers: int,
    *,
    device: str = "cpu",
    base_seed: int = 10000,
    temperature: float = 0.1,
    batch_chunk_size: int = 1000,
    track_choices: bool = False,
    capture_games: bool = False,
    auto_advance: bool = True,
    on_shard_done: Optional[Callable[[ShardResult], None]] = None,
) -> List[ShardResult]:
    """Play `shards` over `workers` processes. Results come back in `shards` order.

    `specs` are `load_agent` specs, loaded afresh in each worker; a shard's `pair` indexes them.
    `on_shard_done` is called in the parent as each shard lands.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    options: Dict[str, Any] = dict(temperature=temperature, batch_chunk_size=batch_chunk_size,
                                   track_choices=track_choices, auto_advance=auto_advance)
    done: Dict[Shard, ShardResult] = {}
    log_dir = tempfile.mkdtemp(prefix="tournament_shards_") if capture_games else None

    # The workers inherit this environment when they are spawned, and the engine and torch read
    # it when their OpenMP runtime starts -- which in the parent has already happened.
    saved = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker, initargs=(device,)) as pool:
            futures: Dict[Future[Dict[str, Any]], Tuple[Shard, Optional[str]]] = {}
            for s in shards:
                i, j = s.pair
                log_path = (os.path.join(log_dir, f"{i}-{j}-{s.index}.jsonl")
                            if log_dir else None)
                futures[pool.submit(_run_shard, specs[i], specs[j], s, games_per_side,
                                    base_seed, options, log_path)] = (s, log_path)
            for fut in as_completed(futures):
                s, log_path = futures[fut]
                games: List[Dict[str, Any]] = []
                if log_path:
                    with open(log_path, "r", encoding="utf-8") as f:
                        games = [json.loads(line) for line in f if line.strip()]
                done[s] = ShardResult(s, fut.result(), games)
                if on_shard_done is not None:
                    on_shard_done(done[s])
    finally:
        if saved is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = saved
        if log_dir:
            shutil.rmtree(log_dir, ignore_errors=True)
    return [done[s] for s in shards]


def merge_shards(results: Sequence[ShardResult]) -> Dict[Pair, Dict[str, Any]]:
    """One merged result per matchup, from its shards in shard order."""
    by_pair: Dict[Pair, List[ShardResult]] = {}
    for r in results:
        by_pair.setdefault(r.shard.pair, []).append(r)
    return {p: merge_matchup_results([r.result for r in sorted(rs, key=lambda r: r.shard.index)])
            for p, rs in by_pair.items()}


def write_games_log(path: str, results: Sequence[ShardResult]) -> None:
    """Every shard's per-game records, matchup by matchup and shard by shard."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for r in sorted(results, key=lambda r: (r.shard.pair, r.shard.index)):
            for g in r.games:
                out.write(json.dumps(g) + "\n")


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

    `on_matchup_done` is called in the parent as each matchup's last shard lands, so progress is
    reported as it happens. With `log_games`, the per-game log holds every matchup's games,
    matchup by matchup and shard by shard.
    """
    shards = plan_shards(matchups, games_per_side, shard_pairs)
    remaining: Dict[Pair, int] = {}
    for s in shards:
        remaining[s.pair] = remaining.get(s.pair, 0) + 1
    landed: Dict[Pair, List[ShardResult]] = {p: [] for p in remaining}

    def shard_done(r: ShardResult) -> None:
        landed[r.shard.pair].append(r)
        remaining[r.shard.pair] -= 1
        if remaining[r.shard.pair] == 0 and on_matchup_done is not None:
            on_matchup_done(r.shard.pair, merge_shards(landed[r.shard.pair])[r.shard.pair])

    results = run_shards(specs, shards, games_per_side, workers, device=device,
                         base_seed=base_seed, temperature=temperature,
                         batch_chunk_size=batch_chunk_size, track_choices=track_choices,
                         capture_games=bool(log_games), auto_advance=auto_advance,
                         on_shard_done=shard_done)
    if log_games:
        write_games_log(log_games, results)
    merged = merge_shards(results)
    return {p: merged[p] for p in matchups}


# -- parts: one tournament over several machines --------------------------------------------
#
# A part file is what one machine played: the tournament's configuration, and the raw result of
# every shard it was given. `pool_parts` checks that a set of part files describe one tournament
# and cover each of its shards exactly once, and merges them. Because a shard's games depend only
# on the shard (module docstring), the pooled tournament equals one machine playing every shard.

PART_FORMAT = "ts-tournament-part-v1"


def write_part(path: str, config: Dict[str, Any], part: Tuple[int, int],
               results: Sequence[ShardResult], wall_seconds: float) -> None:
    doc = {
        "format": PART_FORMAT,
        "config": config,
        "part": list(part),
        "wall_seconds": wall_seconds,
        "shards": [{"pair": list(r.shard.pair), "index": r.shard.index,
                    "pairs": list(r.shard.pairs), "result": r.result, "games": r.games}
                   for r in results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def pool_parts(paths: Sequence[str]) -> Tuple[Dict[str, Any], List[ShardResult], float]:
    """(config, every shard's result in plan order, the slowest part's wall time).

    Refuses parts of different tournaments, and a set that misses a shard or holds one twice --
    a pooled rating over part of the games would read as a rating over all of them.
    """
    if not paths:
        raise ValueError("no part files given")
    docs: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("format") != PART_FORMAT:
            raise ValueError(f"{p} is not a {PART_FORMAT} file")
        docs.append(doc)
    config = docs[0]["config"]
    for p, d in zip(paths, docs):
        if d["config"] != config:
            diff = sorted(k for k in set(config) | set(d["config"])
                          if config.get(k) != d["config"].get(k))
            raise ValueError(f"{p} is from a different tournament than {paths[0]} "
                             f"(differs in {', '.join(diff)})")

    matchups = [tuple(m) for m in config["matchups"]]
    plan = plan_shards([(int(a), int(b)) for a, b in matchups], int(config["games_per_side"]),
                       int(config["shard_pairs"]))
    got: Dict[Shard, ShardResult] = {}
    for p, d in zip(paths, docs):
        for e in d["shards"]:
            s = Shard(pair=(int(e["pair"][0]), int(e["pair"][1])), index=int(e["index"]),
                      pairs=tuple(int(k) for k in e["pairs"]))
            if s in got:
                raise ValueError(f"shard {s.pair}#{s.index} appears in more than one part ({p})")
            got[s] = ShardResult(s, e["result"], e["games"])
    missing = [s for s in plan if s not in got]
    extra = [s for s in got if s not in set(plan)]
    if missing or extra:
        raise ValueError(
            f"the parts do not cover the tournament: {len(missing)} of {len(plan)} shards "
            f"missing (first: {[f'{s.pair}#{s.index}' for s in missing[:5]]}), "
            f"{len(extra)} not in its plan")
    return config, [got[s] for s in plan], max(float(d["wall_seconds"]) for d in docs)
