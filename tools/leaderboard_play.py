#!/usr/bin/env python3
"""Play leaderboard pairings and record them in `leaderboard/matches/<epoch>.jsonl`.

    tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/leaderboard_play.py \\
        --epoch E7 --players <player-id> ... [--opponents <player-id> ... | --round-robin] \\
        --games-per-side 1000

Every player plays every opponent (the epoch's main players unless --opponents is given), each
pairing `--games-per-side` deals from both seats, through tools/tournament.py's batched path.
`--round-robin` plays every pairing among --players instead. A pairing already on record is
skipped; to add games to it, pass a `--base-seed` it has not been played at (the same seed would
replay the same deals). Each pairing is appended as it finishes, so a crash keeps what was done.

Refuses to play on an engine whose fingerprint the epoch does not list, on a stale build, on
network files whose sha256 is not the registered one, or from a checkout with uncommitted changes
to the code that plays (each record names the commit it was played at).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import ts_engine as ts

from tools.lib.batch_tournament import MatchupResult
from tools.lib.data_root import data_path
from tools.lib.engine_fingerprint import fingerprint, staleness_reason
from tools.lib.leaderboard import (Board, LeaderboardError, ROOT, append_record, fit_epoch, load,
                                   make_record, markdown_table, player_spec, validate, view)
from tools.lib.player_spec import PlayerSpecError, load_spec
from tools.tournament import run_massive_tournament

#: Code whose changes can change a game; a record's commit must describe it.
_CODE_PATHS = ("ai", "bot", "bindings", "engine", "tools")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_pairs(board: Board, epoch: str, players: Sequence[str], opponents: Optional[Sequence[str]],
               round_robin: bool, base_seed: Optional[int]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """(pairs to play, pairs skipped as already recorded). Unordered pairs, each once."""
    wanted: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def add(x: str, y: str) -> None:
        key = (min(x, y), max(x, y))
        if x != y and key not in seen:
            seen.add(key)
            wanted.append((x, y))

    if round_robin:
        for i, x in enumerate(players):
            for y in players[i + 1:]:
                add(x, y)
    else:
        for x in players:
            for y in (opponents if opponents is not None else board.epochs[epoch]["main"]):
                add(x, y)
    recorded: Dict[Tuple[str, str], Set[int]] = {}
    for r in board.matches.get(epoch, []):
        recorded.setdefault((min(r["a"], r["b"]), max(r["a"], r["b"])), set()).add(int(r["base_seed"]))
    play, skipped = [], []
    for x, y in wanted:
        seeds = recorded.get((min(x, y), max(x, y)), set())
        if base_seed is None and seeds:
            skipped.append((x, y))
        elif base_seed is not None and base_seed in seeds:
            raise LeaderboardError(f"{x} vs {y} is already recorded at base seed {base_seed}")
        else:
            play.append((x, y))
    return play, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch", required=True)
    ap.add_argument("--players", nargs="+", required=True, help="player ids (leaderboard/players.json)")
    ap.add_argument("--opponents", nargs="+", default=None,
                    help="whom --players play (default: the epoch's main players)")
    ap.add_argument("--round-robin", action="store_true", help="every pairing among --players instead")
    ap.add_argument("--games-per-side", type=int, default=1000)
    ap.add_argument("--base-seed", type=int, default=None,
                    help="first deal seed (default 10000); give an unused one to add games to a recorded pairing")
    ap.add_argument("--batch-chunk-size", type=int, default=1000)
    ap.add_argument("--pack-pairs", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-report", default=None, help="also write tournament.py's markdown report here")
    ap.add_argument("--dry-run", action="store_true", help="print the pairings and stop")
    a = ap.parse_args(argv)
    if a.round_robin and a.opponents is not None:
        ap.error("--round-robin and --opponents exclude each other")
    try:
        return _run(a)
    except (LeaderboardError, PlayerSpecError) as e:
        print(f"leaderboard_play: {e}", file=sys.stderr)
        return 1


def _run(a: argparse.Namespace) -> int:
    board = load()
    problems = validate(board)
    if problems:
        raise LeaderboardError("the leaderboard records are inconsistent:\n  " + "\n  ".join(problems))
    if a.epoch not in board.epochs:
        raise LeaderboardError(f"no epoch {a.epoch!r}; epochs: {', '.join(board.epochs)}")
    for pid in [*a.players, *(a.opponents or [])]:
        if pid not in board.players:
            raise LeaderboardError(f"player {pid!r} is not registered (tools/leaderboard.py add-player)")

    pairs, skipped = plan_pairs(board, a.epoch, a.players, a.opponents, a.round_robin, a.base_seed)
    for x, y in skipped:
        print(f" already recorded, skipped: {x} vs {y}")
    if not pairs:
        print("leaderboard_play: nothing to play")
        return 0
    for x, y in pairs:
        print(f" to play: {x} vs {y}")
    if a.dry_run:
        return 0

    reason = staleness_reason(ts.__file__)
    if reason:
        raise LeaderboardError(reason)
    engine = fingerprint()
    if engine not in board.epochs[a.epoch]["engines"]:
        raise LeaderboardError(
            f"engine {engine} is not one of epoch {a.epoch}'s. If it plays the same games as the "
            f"epoch's engine, add it to epochs.json after checking that a recorded pairing replays "
            f"identically; if it does not, it is a new epoch.")
    dirty = _git("status", "--porcelain", "--", *_CODE_PATHS)
    if dirty:
        raise LeaderboardError(f"uncommitted changes to the code that plays; commit them first:\n{dirty}")
    commit = _git("rev-parse", "HEAD")

    entrants = sorted({p for pr in pairs for p in pr})
    specs = []
    for pid in entrants:
        net = board.players[pid]["network"]
        path = None
        if net is not None:
            entry = board.networks[net]
            path = data_path(entry["file"])
            got = _sha256(path)
            if got != entry["sha256"]:
                raise LeaderboardError(f"{path}: sha256 {got[:12]}… is not the registered "
                                       f"{entry['sha256'][:12]}… for {net}")
        specs.append(load_spec(player_spec(board, pid), path))
    index = {pid: i for i, pid in enumerate(entrants)}
    matchups = [(min(index[x], index[y]), max(index[x], index[y])) for x, y in pairs]
    base_seed = 10000 if a.base_seed is None else a.base_seed
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def on_matchup(name_a: str, name_b: str, m: MatchupResult) -> None:
        rec = make_record(
            name_a, name_b,
            {"w": m["a_wins_as_us"], "l": m["a_losses_as_us"], "d": m["a_draws_as_us"]},
            {"w": m["a_wins_as_ussr"], "l": m["a_losses_as_ussr"], "d": m["a_draws_as_ussr"]},
            base_seed, engine, commit, date)
        append_record(board, a.epoch, rec)

    anchor = board.epochs[a.epoch]["anchor"]
    run_massive_tournament(
        model_specs=specs, games_per_side=a.games_per_side, temperature=0.0,
        batch_chunk_size=a.batch_chunk_size, pack_pairs=a.pack_pairs,
        anchor_model=anchor if anchor in index else entrants[0],
        anchor_elo=float(board.epochs[a.epoch]["anchor_elo"]),
        output_report=a.output_report, device=a.device,
        matchups=matchups, entrant_names=entrants, base_seed=base_seed, on_matchup=on_matchup)

    fit = fit_epoch(board, a.epoch)
    print(f"\n## {a.epoch} after this session\n")
    print(markdown_table(board, view(board, fit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
