#!/usr/bin/env python3
"""The Elo leaderboard: register networks and players, check the records, fit and show ratings.

    PYTHONPATH=. python tools/leaderboard.py validate
    PYTHONPATH=. python tools/leaderboard.py show [--epoch E7] [--main-only] [--lineage E7-A8-R1-S44 ...]
    PYTHONPATH=. python tools/leaderboard.py fit --output web/ui/public/leaderboard.json
    PYTHONPATH=. python tools/leaderboard.py add-network <name> <file> --description ... [--old-name ...] [--report ...] [--hf ...]
    PYTHONPATH=. python tools/leaderboard.py add-player <player-id> --description ...

Games are played, and recorded, by `tools/leaderboard_play.py`. The records are in `leaderboard/`
and the fit is `tools/lib/leaderboard.py`; this CLI needs neither torch nor the engine, so CI runs
`fit` to build the page's data. See tools/README.md, "The Elo leaderboard".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import List, Optional, Sequence

from tools.lib.data_root import data_root
from tools.lib.leaderboard import (LeaderboardError, ROOT, fit_epoch, load, markdown_table,
                                   save_registry, site_data, validate, view)
from tools.lib.player_spec import PlayerSpecError, parse_player_id, player_id, spec_to_dict


def _head_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _checked_board():
    board = load()
    problems = validate(board)
    if problems:
        raise LeaderboardError("the leaderboard records are inconsistent:\n  " + "\n  ".join(problems))
    return board


def cmd_validate(_: argparse.Namespace) -> int:
    board = _checked_board()
    n = sum(len(v) for v in board.matches.values())
    print(f"leaderboard: {len(board.networks)} networks, {len(board.players)} players, "
          f"{len(board.epochs)} epoch(s), {n} pairing records -- consistent")
    return 0


def cmd_fit(a: argparse.Namespace) -> int:
    board = _checked_board()
    data = site_data(board, commit=a.commit or _head_commit())
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    if a.output:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"leaderboard: wrote {a.output}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    board = _checked_board()
    epochs = [a.epoch] if a.epoch else list(board.epochs)
    for name in epochs:
        if name not in board.epochs:
            raise LeaderboardError(f"no epoch {name!r}; epochs: {', '.join(board.epochs)}")
        fit = fit_epoch(board, name)
        rows = view(board, fit, main_only=a.main_only, lineages=a.lineage or ())
        ep = board.epochs[name]
        print(f"## {name}: {ep['description']}\n")
        print(f"Anchor `{fit.anchor}` = {fit.anchor_elo:.0f}; US seat advantage "
              f"{fit.seat_us:+.0f} ± {fit.seat_us_se:.0f} Elo.\n")
        print(markdown_table(board, rows))
        if fit.unrated:
            print(f"\nNot connected to the main players, so unrated: {', '.join(fit.unrated)}")
        print()
    return 0


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_add_network(a: argparse.Namespace) -> int:
    board = _checked_board()
    if a.name in board.networks and not a.replace:
        raise LeaderboardError(f"network {a.name!r} is already registered (--replace to overwrite)")
    path = os.path.realpath(a.file)
    root = os.path.realpath(data_root())
    if os.path.commonpath([path, root]) != root:
        raise LeaderboardError(f"{a.file} is not inside the shared data tree {root}")
    if a.report and not (ROOT / a.report).is_file():
        raise LeaderboardError(f"report {a.report} does not exist")
    # The path as given, not resolved: a name under data/checkpoints/_models/ is the readable
    # one, and the symlink it may be stays part of the record.
    rel = os.path.relpath(os.path.abspath(a.file), os.path.abspath(data_root()))
    board.networks[a.name] = {"file": rel, "sha256": _sha256(path), "description": a.description,
                              "old_names": list(a.old_name or []), "hf": a.hf, "report": a.report}
    problems = validate(board)
    if problems:
        raise LeaderboardError("\n".join(problems))
    save_registry(board)
    print(f"leaderboard: network {a.name} -> data/{rel}")
    return 0


def cmd_add_player(a: argparse.Namespace) -> int:
    board = _checked_board()
    network, spec = parse_player_id(a.player)
    pid = player_id(network, spec)
    if pid in board.players:
        raise LeaderboardError(f"player {pid!r} is already registered")
    if network is not None and network not in board.networks:
        raise LeaderboardError(f"network {network!r} is not registered (add-network first)")
    board.players[pid] = {"network": network, "inference": spec_to_dict(spec), "description": a.description}
    save_registry(board)
    print(f"leaderboard: player {pid}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check the records").set_defaults(fn=cmd_validate)
    p = sub.add_parser("fit", help="fit every epoch and write the page's data")
    p.add_argument("--output", default=None, help="where to write leaderboard.json (default: stdout)")
    p.add_argument("--commit", default=None, help="the commit to stamp (default: HEAD)")
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("show", help="print the ratings as a markdown table")
    p.add_argument("--epoch", default=None)
    p.add_argument("--main-only", action="store_true", help="only the main players (plus any --lineage)")
    p.add_argument("--lineage", nargs="+", default=None,
                   help="the main players plus these lineages (a network's run name before '@')")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("add-network", help="register a network file")
    p.add_argument("name", help="the network's name in the run nomenclature, e.g. E7-A8-R1-S44@6800M")
    p.add_argument("file", help="the .pt file, inside the shared data tree")
    p.add_argument("--description", required=True)
    p.add_argument("--old-name", nargs="+", default=None, help="names it went by before")
    p.add_argument("--report", default=None, help="its behaviour report, a path in the repository")
    p.add_argument("--hf", default=None, help="its path in the Hugging Face repo, once published")
    p.add_argument("--replace", action="store_true")
    p.set_defaults(fn=cmd_add_network)
    p = sub.add_parser("add-player", help="register a player: a network and how it is run")
    p.add_argument("player", help="a player id, e.g. E7-A8-R1-S44@6800M~gumbel(sims=256,k=8,fpu=0.2)")
    p.add_argument("--description", default="")
    p.set_defaults(fn=cmd_add_player)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.fn(a))
    except (LeaderboardError, PlayerSpecError) as e:
        print(f"leaderboard: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
