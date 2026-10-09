#!/usr/bin/env python3
"""Publish checkpoints to the Hugging Face model repo, each with its ONNX export beside it.

    PYTHONPATH=.:build/release python tools/publish_hf.py run data/checkpoints/<run-dir> [--dry-run]
    PYTHONPATH=.:build/release python tools/publish_hf.py files data/checkpoints/_models/<name>.pt ... [--dry-run]
    PYTHONPATH=.:build/release python tools/publish_hf.py default <path-in-repo>.onnx [--dry-run]

The repo mirrors `data/checkpoints/`: a run directory is published whole as `<run-dir>/`, a model
outside one (an SWA or a soup under `_models/`) as `_models/<name>.pt`. Next to every `.pt` goes
its `.onnx` (tools/export_onnx.py, verified against torch before it is written), which is what
the workbench runs; the page groups the repo's files by directory, one group per run.

* `run` exports every snapshot and SWA of the run that has no up-to-date `.onnx`, then uploads the
  directory without its resume and opponent-pool states (`resume_*.pt`, `pool_*.pt` -- optimizer
  state, about two thirds of a run's size) or `run.pid`.
* `files` does the same for single files.
* `default` writes the repo's `default.json`, the model a workbench link that names none opens.

After an upload, every network in leaderboard/networks.json whose file was published gets its
`hf` path, so the leaderboard page links the weights and opens the network in the workbench.
Uploading needs a token with write access (`hf auth login`, or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from huggingface_hub import HfApi

from tools.export_onnx import ExportRefused, export
from tools.lib.data_root import data_path
from tools.lib.leaderboard import HF_REPO, LeaderboardError, load, save_registry, validate

#: Never published: optimizer / opponent-pool state for resuming, and the live process id.
EXCLUDE = ["resume_*.pt", "resume_state.pt", "pool_*.pt", "run.pid", "*.tmp"]
#: The weights a run directory holds, each of which gets an .onnx.
WEIGHTS = ["snapshot_*.pt", "swa_*.pt"]


def _checkpoints_root() -> str:
    return os.path.realpath(data_path("checkpoints"))


def repo_path(local: str) -> str:
    """Where a file under data/checkpoints/ lives in the repo: the same relative path."""
    root = _checkpoints_root()
    full = os.path.abspath(local)
    # Not realpath: _models/ holds readable symlinks into _soups/ etc., and the readable name is
    # the one to publish under.
    rel = os.path.relpath(full, os.path.abspath(data_path("checkpoints")))
    if rel.startswith(".."):
        rel = os.path.relpath(os.path.realpath(full), root)
    if rel.startswith(".."):
        raise LeaderboardError(f"{local} is not under data/checkpoints/")
    return rel.replace(os.sep, "/")


def _onnx_of(pt: str) -> str:
    return os.path.splitext(pt)[0] + ".onnx"


def ensure_onnx(pt: str, dry_run: bool) -> str:
    """The .onnx beside `pt`, exported (and verified) unless an up-to-date one is there."""
    out = _onnx_of(pt)
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(os.path.realpath(pt)):
        return out
    print(f" export {os.path.basename(pt)} -> {os.path.basename(out)}{' (dry run)' if dry_run else ''}", flush=True)
    if not dry_run:
        export(pt, out)
    return out


def _local_paths_in(metadata: str) -> List[str]:
    """metadata.json values that are absolute local paths -- published as they are; listed so
    the owner sees them before they go out."""
    try:
        with open(metadata, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return sorted(f"{k}={v}" for k, v in meta.items() if isinstance(v, str) and v.startswith("/"))


def register(published: Sequence[str]) -> None:
    """Point every registered network whose file was just published at its path in the repo."""
    board = load()
    by_path = {repo_path(data_path(n["file"])): name for name, n in board.networks.items()}
    changed = []
    for rel in published:
        name = by_path.get(rel)
        if name and board.networks[name]["hf"] != rel:
            board.networks[name]["hf"] = rel
            changed.append(name)
    if changed:
        problems = validate(board)
        if problems:
            raise LeaderboardError("\n".join(problems))
        save_registry(board)
        print(f" leaderboard/networks.json: hf set for {', '.join(changed)}")


def _api(repo: str, dry_run: bool) -> Optional[HfApi]:
    if dry_run:
        return None
    api = HfApi()
    user = api.whoami()          # fails here, before any export, without a token
    print(f" uploading to {repo} as {user.get('name')}")
    return api


def cmd_run(a: argparse.Namespace) -> int:
    run_dir = os.path.abspath(a.run_dir.rstrip("/"))
    if not os.path.isfile(os.path.join(run_dir, "metadata.json")):
        raise LeaderboardError(f"{run_dir} is not a run directory (no metadata.json)")
    rel_dir = repo_path(run_dir)
    api = _api(a.repo, a.dry_run)
    for note in _local_paths_in(os.path.join(run_dir, "metadata.json")):
        print(f" note: metadata.json carries a local path, published as is: {note}")
    weights = sorted(f for f in os.listdir(run_dir)
                     if any(fnmatch.fnmatch(f, w) for w in WEIGHTS)
                     and not any(fnmatch.fnmatch(f, x) for x in EXCLUDE))
    for f in weights:
        ensure_onnx(os.path.join(run_dir, f), a.dry_run)
    published = [f"{rel_dir}/{f}" for f in weights]
    print(f" {rel_dir}/: {len(weights)} weight files with their .onnx, plus metadata, metrics and reports")
    if api is not None:
        api.upload_folder(repo_id=a.repo, folder_path=run_dir, path_in_repo=rel_dir,
                          ignore_patterns=EXCLUDE, commit_message=f"Publish {rel_dir}")
        register(published)
    return 0


def cmd_files(a: argparse.Namespace) -> int:
    api = _api(a.repo, a.dry_run)
    published = []
    for pt in a.files:
        if not pt.endswith(".pt") or not os.path.isfile(pt):
            raise LeaderboardError(f"{pt}: not a .pt file")
        rel = repo_path(pt)
        onnx = ensure_onnx(pt, a.dry_run)
        print(f" {rel} (+ .onnx)")
        if api is not None:
            for local, dest in ((pt, rel), (onnx, _onnx_of(rel))):
                api.upload_file(path_or_fileobj=os.path.realpath(local), path_in_repo=dest,
                                repo_id=a.repo, commit_message=f"Publish {rel}")
        published.append(rel)
    if api is not None:
        register(published)
    return 0


def cmd_default(a: argparse.Namespace) -> int:
    if not a.model.endswith(".onnx"):
        raise LeaderboardError("the default model is an .onnx path in the repo")
    api = _api(a.repo, a.dry_run)
    if api is not None:
        if not api.file_exists(a.repo, a.model):
            raise LeaderboardError(f"{a.repo} holds no {a.model}")
        body: Dict[str, str] = {"model": a.model}
        api.upload_file(path_or_fileobj=(json.dumps(body, indent=2) + "\n").encode(),
                        path_in_repo="default.json", repo_id=a.repo,
                        commit_message=f"Default model: {a.model}")
    print(f" default.json -> {a.model}{' (dry run)' if a.dry_run else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=HF_REPO)
    common.add_argument("--dry-run", action="store_true", help="say what would be exported and uploaded, do neither")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", parents=[common], help="publish a whole run directory")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("files", parents=[common], help="publish single .pt files (SWAs, soups)")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_files)
    p = sub.add_parser("default", parents=[common], help="set the model a workbench link that names none opens")
    p.add_argument("model", help="an .onnx path in the repo")
    p.set_defaults(fn=cmd_default)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.fn(a))
    except (LeaderboardError, ExportRefused) as e:
        print(f"publish_hf: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
