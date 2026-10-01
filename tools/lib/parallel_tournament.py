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
`reseed(seed)` (`Reseedable`) is reseeded. The outcome of a game then depends on its shard, never
on the worker that ran it or on what that worker ran before: at a fixed `shard_pairs`, any number
of workers gives the same results. A torch policy sampling above the greedy threshold draws from
torch's generator, which is seeded per shard as well.

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
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple,
                    TypedDict, cast)

import numpy as np
import torch

from tools.lib.batch_tournament import BatchMatchRunner, MatchupResult, merge_matchup_results
from tools.lib.player_agent import PlayerAgent, Reseedable, load_agent

Pair = Tuple[int, int]
#: (index, count): one machine's share of a tournament spread over `count` machines, 0-based.
Part = Tuple[int, int]
#: One line of a `--log-games` file, as `play_parallel_matchup` writes it.
GameRecord = Dict[str, Any]


@dataclass(frozen=True)
class MatchOptions:
    """How every matchup of a tournament is played; the same for every shard."""

    device: str = "cpu"
    base_seed: int = 10000
    temperature: float = 0.1
    batch_chunk_size: int = 1000
    track_choices: bool = False
    auto_advance: bool = True


class ShardDoc(TypedDict):
    pair: List[int]
    index: int
    pairs: List[int]
    result: MatchupResult
    games: List[GameRecord]


@dataclass(frozen=True)
class Shard:
    """A run of consecutive game pairs of one matchup."""

    pair: Pair                  # (i, j): indices of the two entrants in the tournament's list
    index: int                  # position of this shard within its matchup
    pairs: Tuple[int, ...]      # game pair indices, into range(games_per_side)

    @property
    def label(self) -> str:
        return f"{self.pair}#{self.index}"

    @classmethod
    def from_doc(cls, doc: ShardDoc) -> "Shard":
        return cls(pair=(int(doc["pair"][0]), int(doc["pair"][1])), index=int(doc["index"]),
                   pairs=tuple(int(k) for k in doc["pairs"]))


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
        if isinstance(agent, Reseedable):
            agent.reseed(seed + n)


# -- worker side ------------------------------------------------------------------------------

_AGENTS: Dict[str, PlayerAgent] = {}


def _init_worker() -> None:
    torch.set_num_threads(1)


def _agent(spec: str, device: str) -> PlayerAgent:
    # Loaded once per worker: a checkpoint is read from disk once, not once per shard. Nothing an
    # agent carries between games survives into a shard's results, because seed_shard restarts it.
    if spec not in _AGENTS:
        _AGENTS[spec] = load_agent(spec, device=device)
    return _AGENTS[spec]


def _run_shard(spec_a: str, spec_b: str, shard: Shard, games_per_side: int,
               options: MatchOptions, log_path: Optional[str]) -> MatchupResult:
    a, b = _agent(spec_a, options.device), _agent(spec_b, options.device)
    seed_shard(shard_seed(options.base_seed, shard), (a, b))
    return BatchMatchRunner.play_parallel_matchup(
        a, b, games_per_side=games_per_side, base_seed=options.base_seed,
        device=options.device, temperature=options.temperature,
        batch_chunk_size=options.batch_chunk_size, track_choices=options.track_choices,
        auto_advance=options.auto_advance, pairs=shard.pairs, log_games_file=log_path)


# -- driver side ------------------------------------------------------------------------------

@dataclass
class ShardResult:
    shard: Shard
    result: MatchupResult
    games: List[GameRecord]      # the shard's per-game log records, when logs were asked for

    def to_doc(self) -> ShardDoc:
        return {"pair": list(self.shard.pair), "index": self.shard.index,
                "pairs": list(self.shard.pairs), "result": self.result, "games": self.games}

    @classmethod
    def from_doc(cls, doc: ShardDoc) -> "ShardResult":
        return cls(Shard.from_doc(doc), doc["result"], doc["games"])


def select_part(shards: Sequence[Shard], index: int, count: int) -> List[Shard]:
    """Part `index` of `count` (0-based): every count-th shard. Striding rather than cutting into
    blocks spreads every matchup, and each matchup's slow and fast pairs, over all the parts."""
    if not 0 <= index < count:
        raise ValueError(f"part index must be in range({count}), got {index}")
    return list(shards[index::count])


@contextmanager
def _env_var(name: str, value: str) -> Iterator[None]:
    saved = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def run_shards(
    specs: Sequence[str],
    shards: Sequence[Shard],
    games_per_side: int,
    workers: int,
    options: MatchOptions = MatchOptions(),
    *,
    capture_games: bool = False,
    on_shard_done: Optional[Callable[[ShardResult], None]] = None,
) -> List[ShardResult]:
    """Play `shards` over `workers` processes. Results come back in `shards` order.

    `specs` are `load_agent` specs, loaded afresh in each worker; a shard's `pair` indexes them.
    `on_shard_done` is called in the parent as each shard lands.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    done: Dict[Shard, ShardResult] = {}
    log_dir = tempfile.mkdtemp(prefix="tournament_shards_") if capture_games else None
    ctx = multiprocessing.get_context("spawn")
    try:
        # The workers inherit this environment when they are spawned, and the engine and torch
        # read it when their OpenMP runtime starts -- which in the parent has already happened.
        with _env_var("OMP_NUM_THREADS", "1"), ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx, initializer=_init_worker) as pool:
            futures: Dict[Future[MatchupResult], Tuple[Shard, Optional[str]]] = {}
            for s in shards:
                i, j = s.pair
                log_path = (os.path.join(log_dir, f"{i}-{j}-{s.index}.jsonl")
                            if log_dir else None)
                futures[pool.submit(_run_shard, specs[i], specs[j], s, games_per_side,
                                    options, log_path)] = (s, log_path)
            for fut in as_completed(futures):
                s, log_path = futures[fut]
                # The result first: a worker that raised wrote no log, and the shard's error is
                # what to report, not the missing file.
                result = fut.result()
                games: List[GameRecord] = []
                if log_path:
                    with open(log_path, "r", encoding="utf-8") as f:
                        games = [json.loads(line) for line in f if line.strip()]
                done[s] = ShardResult(s, result, games)
                if on_shard_done is not None:
                    on_shard_done(done[s])
    finally:
        if log_dir:
            shutil.rmtree(log_dir, ignore_errors=True)
    return [done[s] for s in shards]


def merge_shard_group(results: Sequence[ShardResult]) -> MatchupResult:
    """One matchup's result from its shards, in shard order."""
    return merge_matchup_results([r.result for r in sorted(results, key=lambda r: r.shard.index)])


def merge_shards(results: Sequence[ShardResult]) -> Dict[Pair, MatchupResult]:
    """One merged result per matchup, from its shards in shard order."""
    by_pair: Dict[Pair, List[ShardResult]] = defaultdict(list)
    for r in results:
        by_pair[r.shard.pair].append(r)
    return {p: merge_shard_group(rs) for p, rs in by_pair.items()}


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
    options: MatchOptions = MatchOptions(),
    *,
    log_games: Optional[str] = None,
    on_matchup_done: Optional[Callable[[Pair, MatchupResult], None]] = None,
) -> Dict[Pair, MatchupResult]:
    """Play every matchup's games over `workers` processes; one merged result per matchup.

    `on_matchup_done` is called in the parent as each matchup's last shard lands, so progress is
    reported as it happens. With `log_games`, the per-game log holds every matchup's games,
    matchup by matchup and shard by shard.
    """
    shards = plan_shards(matchups, games_per_side, shard_pairs)
    remaining = Counter(s.pair for s in shards)
    landed: Dict[Pair, List[ShardResult]] = defaultdict(list)

    def shard_done(r: ShardResult) -> None:
        landed[r.shard.pair].append(r)
        remaining[r.shard.pair] -= 1
        if remaining[r.shard.pair] == 0 and on_matchup_done is not None:
            on_matchup_done(r.shard.pair, merge_shard_group(landed[r.shard.pair]))

    results = run_shards(specs, shards, games_per_side, workers, options,
                         capture_games=bool(log_games), on_shard_done=shard_done)
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


class PartConfig(TypedDict):
    """What every part of one tournament must agree on. `models` are the entrants' names as the
    report shows them, so two parts given different entrants -- or the same entrants under
    different `--opening`s -- are refused by name."""

    models: List[str]
    matchups: List[List[int]]
    games_per_side: int
    shard_pairs: int
    temperature: float
    batch_chunk_size: int
    track_choices: bool
    auto_advance: bool
    anchor_model: str
    anchor_elo: float


class PartDoc(TypedDict):
    format: str
    config: PartConfig
    part: List[int]
    wall_seconds: float
    shards: List[ShardDoc]


class PooledParts(NamedTuple):
    config: PartConfig
    shards: List[ShardResult]        # every shard's result, in plan order
    wall_seconds: float              # the slowest part's wall time


def write_part(path: str, config: PartConfig, part: Part, results: Sequence[ShardResult],
               wall_seconds: float) -> None:
    doc: PartDoc = {
        "format": PART_FORMAT,
        "config": config,
        "part": list(part),
        "wall_seconds": wall_seconds,
        "shards": [r.to_doc() for r in results],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def _read_part(path: str) -> PartDoc:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("format") != PART_FORMAT:
        raise ValueError(f"{path} is not a {PART_FORMAT} file")
    return cast(PartDoc, doc)


def pool_parts(paths: Sequence[str]) -> PooledParts:
    """The tournament's configuration, every shard's result in plan order, and the slowest
    part's wall time.

    Refuses parts of different tournaments, and a set that misses a shard or holds one twice --
    a pooled rating over part of the games would read as a rating over all of them.
    """
    if not paths:
        raise ValueError("no part files given")
    docs = [_read_part(p) for p in paths]
    config = docs[0]["config"]
    for p, d in zip(paths, docs):
        if d["config"] != config:
            diff = sorted(k for k in set(config) | set(d["config"])
                          if config.get(k) != d["config"].get(k))
            raise ValueError(f"{p} is from a different tournament than {paths[0]} "
                             f"(differs in {', '.join(diff)})")

    plan = plan_shards([(m[0], m[1]) for m in config["matchups"]], config["games_per_side"],
                       config["shard_pairs"])
    planned = set(plan)
    got: Dict[Shard, ShardResult] = {}
    for p, d in zip(paths, docs):
        for e in d["shards"]:
            r = ShardResult.from_doc(e)
            if r.shard in got:
                raise ValueError(f"shard {r.shard.label} appears in more than one part ({p})")
            got[r.shard] = r
    missing = [s for s in plan if s not in got]
    extra = [s for s in got if s not in planned]
    if missing or extra:
        raise ValueError(
            f"the parts do not cover the tournament: {len(missing)} of {len(plan)} shards "
            f"missing (first: {[s.label for s in missing[:5]]}), {len(extra)} not in its plan")
    return PooledParts(config, [got[s] for s in plan], max(d["wall_seconds"] for d in docs))
