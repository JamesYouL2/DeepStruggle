#!/usr/bin/env python3
"""Publish checkpoints to the Hugging Face model repo, with ONNX exports of the ones worth opening.

    PYTHONPATH=.:build/release python tools/publish_hf.py run data/checkpoints/<run-dir> [--dry-run]
    PYTHONPATH=.:build/release python tools/publish_hf.py files data/checkpoints/_models/<name>.pt ... [--dry-run]
    PYTHONPATH=.:build/release python tools/publish_hf.py default <path-in-repo>.onnx [--dry-run]
    PYTHONPATH=.:build/release python tools/publish_hf.py index [--dry-run]

The repo mirrors `data/checkpoints/`: a run directory is published whole as `<run-dir>/`, a model
outside one (an SWA or a soup under `_models/`) as `_models/<name>.pt`. The workbench runs `.onnx`
files (tools/export_onnx.py, verified against torch before one is written), and a run saves a
snapshot every 10M steps, so only these get one, beside their `.pt`:

* in a run directory: the final plain snapshot (the highest `snapshot_<N>steps.pt`), the
  final SWA (the `swa_<lo>-<hi>M.pt` with the highest `hi`), and every file of it registered in
  leaderboard/networks.json;
* every file given to `files` -- the soups and SWAs under `_models/`.

Every other `.pt` goes up without one; `files` adds one later. `run` leaves out the resume and
opponent-pool states (`resume_*.pt`, `pool_*.pt` -- optimizer state, about two thirds of a run's
size), `run.pid`, `snapshot_final.pt` (a copy named by no step: snapshots are always referred to by
step), and any `.onnx` it did not select.

Every publish also rewrites `models.json` at the repo root, in the same commit: each `.onnx` the
repo holds with its commit date, sha256 and size, and the default model -- the one a workbench
link that names none opens (`default` sets it). The page reads that one file instead of listing
the whole tree, which with `expand` (the dates) is one request per 50 files. `index` rebuilds it
from the repo as it is, for files uploaded another way; a legacy `default.json`'s model carries
over.

After an upload, every network in leaderboard/networks.json whose file was published gets its
`hf` path, so the leaderboard page links the weights and opens the network in the workbench.
Uploading needs a token with write access (`hf auth login`, or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Set, TypedDict

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.hf_api import RepoFile

from tools.export_onnx import ExportRefused, export
from tools.lib.data_root import data_path
from tools.lib.leaderboard import HF_REPO, LeaderboardError, load, save_registry, validate

#: Never published: optimizer / opponent-pool state for resuming, the live process id, and the
#: final-weights copy, which names no step (snapshots are referred to by step).
EXCLUDE = ["resume_*.pt", "resume_state.pt", "pool_*.pt", "run.pid", "*.tmp",
           "snapshot_final.pt", "snapshot_final.onnx"]
#: The repo's catalogue for the workbench: one small file instead of listing the whole tree.
MANIFEST = "models.json"
#: Where the default model was named before the manifest; read once, to carry it over.
LEGACY_DEFAULT = "default.json"
_SNAPSHOT_RE = re.compile(r"^snapshot_(\d+)steps\.pt$")


class ManifestEntry(TypedDict):
    path: str
    #: The commit date, as the Hugging Face API writes it ("2026-10-09T15:48:30.000Z").
    date: str
    sha256: str
    size: int


class Manifest(TypedDict):
    schema: int
    default: Optional[str]
    #: Newest first.
    models: List[ManifestEntry]
_SWA_RE = re.compile(r"^swa_(\d+)-(\d+)M\.pt$")


def repo_path(local: str) -> str:
    """Where a file under data/checkpoints/ lives in the repo: the same relative path."""
    # Not realpath first: _models/ holds readable symlinks into _soups/ etc., and the readable
    # name is the one to publish under.
    rel = os.path.relpath(os.path.abspath(local), os.path.abspath(data_path("checkpoints")))
    if rel.startswith(".."):
        rel = os.path.relpath(os.path.realpath(local), os.path.realpath(data_path("checkpoints")))
    if rel.startswith(".."):
        raise LeaderboardError(f"{local} is not under data/checkpoints/")
    return rel.replace(os.sep, "/")


def _onnx_of(pt: str) -> str:
    return os.path.splitext(pt)[0] + ".onnx"


def _excluded(name: str) -> bool:
    return any(fnmatch.fnmatch(name, x) for x in EXCLUDE)


def leaderboard_files(run_dir: str) -> Set[str]:
    """The files of `run_dir` registered as networks in leaderboard/networks.json."""
    here = os.path.realpath(run_dir)
    out = set()
    for net in load().networks.values():
        path = os.path.realpath(data_path(net["file"]))
        if os.path.dirname(path) == here:
            out.add(os.path.basename(path))
    return out


def onnx_selection(files: Sequence[str], registered: Set[str]) -> List[str]:
    """Which of a run directory's .pt files get an .onnx: the final plain snapshot, the final SWA,
    and the ones on the leaderboard."""
    chosen = set(f for f in registered if f in files)
    steps = [(int(m.group(1)), f) for f in files if (m := _SNAPSHOT_RE.match(f))]
    if steps:
        chosen.add(max(steps)[1])
    swas = [((int(m.group(2)), int(m.group(1))), f) for f in files if (m := _SWA_RE.match(f))]
    if swas:
        chosen.add(max(swas)[1])
    return sorted(chosen)


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


def _api(dry_run: bool) -> Optional[HfApi]:
    if dry_run:
        return None
    api = HfApi()
    user = api.whoami()          # fails here, before any export, without a token
    print(f" uploading as {user.get('name')}")
    return api


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_date(d: datetime.datetime) -> str:
    """The date format the Hugging Face API writes, so the page orders both kinds as strings."""
    return d.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def manifest_entry(repo_file: str, local: str, date: datetime.datetime) -> ManifestEntry:
    return {"path": repo_file, "date": _hf_date(date), "sha256": _sha256_file(os.path.realpath(local)),
            "size": os.path.getsize(os.path.realpath(local))}


def _remote_json(api: HfApi, repo: str, name: str) -> Optional[Dict[str, object]]:
    if not api.file_exists(repo, name):
        return None
    with open(api.hf_hub_download(repo, name), encoding="utf-8") as f:
        doc = json.load(f)
    return doc if isinstance(doc, dict) else None


def build_manifest(api: HfApi, repo: str, added: Sequence[ManifestEntry],
                   default: Optional[str] = None) -> Manifest:
    """`models.json` after a commit that adds `added`: every .onnx the repo holds (from one
    listing of its tree, and one paths-info request per 50 files for their dates) plus the new
    ones. The default is `default`, else the current manifest's, else a legacy default.json's;
    one the repo will not hold is refused."""
    new = {e["path"]: e for e in added}
    existing = {e.path: e for e in api.list_repo_tree(repo, recursive=True)
                if isinstance(e, RepoFile) and e.path.endswith(".onnx") and e.path not in new}
    dates: Dict[str, str] = {}
    paths = sorted(existing)
    for k in range(0, len(paths), 50):
        for info in api.get_paths_info(repo, paths[k:k + 50], expand=True):
            if isinstance(info, RepoFile) and info.last_commit is not None:
                dates[info.path] = _hf_date(info.last_commit.date)
    entries: List[ManifestEntry] = [
        {"path": p, "date": dates.get(p, ""), "sha256": e.lfs.sha256 if e.lfs else "", "size": e.size}
        for p, e in existing.items()] + list(new.values())
    entries.sort(key=lambda e: (e["date"], e["path"]), reverse=True)
    if default is None:
        for name, key in ((MANIFEST, "default"), (LEGACY_DEFAULT, "model")):
            doc = _remote_json(api, repo, name)
            if doc and isinstance(doc.get(key), str):
                default = str(doc[key])
                break
    if default is not None and default not in {e["path"] for e in entries}:
        raise LeaderboardError(f"the default model {default} is not an .onnx the repo holds")
    return {"schema": 1, "default": default, "models": entries}


def manifest_op(manifest: Manifest) -> CommitOperationAdd:
    body = (json.dumps(manifest, indent=1) + "\n").encode()
    return CommitOperationAdd(path_in_repo=MANIFEST, path_or_fileobj=body)


def cmd_run(a: argparse.Namespace) -> int:
    run_dir = os.path.abspath(a.run_dir.rstrip("/"))
    if not os.path.isfile(os.path.join(run_dir, "metadata.json")):
        raise LeaderboardError(f"{run_dir} is not a run directory (no metadata.json)")
    rel_dir = repo_path(run_dir)
    api = _api(a.dry_run)
    for note in _local_paths_in(os.path.join(run_dir, "metadata.json")):
        print(f" note: metadata.json carries a local path, published as is: {note}")
    top = sorted(os.listdir(run_dir))
    weights = [f for f in top if f.endswith(".pt") and not _excluded(f)]
    with_onnx = onnx_selection(weights, leaderboard_files(run_dir))
    for f in with_onnx:
        ensure_onnx(os.path.join(run_dir, f), a.dry_run)
    keep_onnx = {_onnx_of(f) for f in with_onnx}
    uploads: List[str] = []
    for root, dirs, names in os.walk(run_dir):
        dirs.sort()
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(root, n), run_dir)
            if _excluded(n) or (n.endswith(".onnx") and rel not in keep_onnx):
                continue
            uploads.append(rel)
    print(f" {rel_dir}/: {len(uploads)} files -- {len(weights)} weight files, ONNX for "
          f"{', '.join(with_onnx) or 'none'}")
    if api is not None:
        now = datetime.datetime.now(datetime.timezone.utc)
        ops = [CommitOperationAdd(path_in_repo=f"{rel_dir}/{r}", path_or_fileobj=os.path.join(run_dir, r))
               for r in uploads]
        added = [manifest_entry(f"{rel_dir}/{r}", os.path.join(run_dir, r), now)
                 for r in uploads if r.endswith(".onnx")]
        ops.append(manifest_op(build_manifest(api, a.repo, added)))
        api.create_commit(repo_id=a.repo, operations=ops, commit_message=f"Publish {rel_dir}")
        register([f"{rel_dir}/{f}" for f in weights])
    return 0


def cmd_files(a: argparse.Namespace) -> int:
    api = _api(a.dry_run)
    published: List[str] = []
    ops: List[CommitOperationAdd] = []
    added: List[ManifestEntry] = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for pt in a.files:
        if not pt.endswith(".pt") or not os.path.isfile(pt):
            raise LeaderboardError(f"{pt}: not a .pt file")
        rel = repo_path(pt)
        onnx = ensure_onnx(pt, a.dry_run)
        print(f" {rel} (+ .onnx)")
        if api is not None:
            ops += [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=os.path.realpath(local))
                    for local, dest in ((pt, rel), (onnx, _onnx_of(rel)))]
            added.append(manifest_entry(_onnx_of(rel), onnx, now))
        published.append(rel)
    if api is not None:
        ops.append(manifest_op(build_manifest(api, a.repo, added)))
        api.create_commit(repo_id=a.repo, operations=ops,
                          commit_message=f"Publish {', '.join(published)}")
        register(published)
    return 0


def cmd_default(a: argparse.Namespace) -> int:
    if not a.model.endswith(".onnx"):
        raise LeaderboardError("the default model is an .onnx path in the repo")
    api = _api(a.dry_run)
    if api is not None:
        api.create_commit(repo_id=a.repo, operations=[manifest_op(build_manifest(api, a.repo, [], a.model))],
                          commit_message=f"Default model: {a.model}")
    print(f" {MANIFEST}: default -> {a.model}{' (dry run)' if a.dry_run else ''}")
    return 0


def cmd_index(a: argparse.Namespace) -> int:
    """Rebuild models.json from the repo as it is -- for files uploaded some other way, and once
    for a repo published before the manifest existed."""
    api = HfApi() if a.dry_run else _api(False)
    assert api is not None
    manifest = build_manifest(api, a.repo, [])
    print(f" {MANIFEST}: {len(manifest['models'])} models, default {manifest['default']}"
          f"{' (dry run)' if a.dry_run else ''}")
    if not a.dry_run:
        api.create_commit(repo_id=a.repo, operations=[manifest_op(manifest)],
                          commit_message="Rebuild models.json")
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
    p = sub.add_parser("files", parents=[common], help="publish single .pt files (SWAs, soups), each with its .onnx")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_files)
    p = sub.add_parser("default", parents=[common], help="set the model a workbench link that names none opens")
    p.add_argument("model", help="an .onnx path in the repo")
    p.set_defaults(fn=cmd_default)
    p = sub.add_parser("index", parents=[common], help="rebuild models.json from what the repo holds")
    p.set_defaults(fn=cmd_index)
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
