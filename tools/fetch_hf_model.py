#!/usr/bin/env python3
"""Download a published ONNX model from Hugging Face -- by default the newest one.

    PYTHONPATH=. .venv/bin/python tools/fetch_hf_model.py                     # newest upload
    PYTHONPATH=. .venv/bin/python tools/fetch_hf_model.py --path E4-08-36_240M.onnx

"Newest" is the workbench's rule (web/ui/src/analysis/model.ts, `listHfModels`): the `.onnx` file
whose last commit is latest, ties broken by path. So a tournament run by CI against "the newest
model" plays the model a visitor to the page gets by default.

`pt:<path>.pt` downloads a PyTorch checkpoint as it is, for an entrant that needs the torch network
itself (`search:`). A checkpoint that was never exported can be named as `<revision>:<path>.pt` -- a run directory's
`snapshot_*.pt` or `swa_*.pt`, at any revision of the repo, including one whose files were since
deleted. It is downloaded and exported to ONNX here with `tools/export_onnx.py` (which needs the
built engine and torch, so this form only works after setup), as `<run>@<steps>M.onnx`:
`83120a1a3e:E7-04-44_20261001_090354/snapshot_1200029696steps.pt` -> `E7-04-44@1200M.onnx`.

A model that only exists as a workflow artifact (a distilled checkpoint, say) is named as
`gh:<run id>/<artifact name>/<file.onnx>` and fetched with the `gh` CLI, which needs `GH_TOKEN`
and the repository in `GITHUB_REPOSITORY` (both set in a workflow).

Prints the downloaded file's path as the last line of output, for a shell to capture;
`--resolve-only` prints the file's path in the repo instead and downloads nothing. Needs no
token: the default repo is public. Standard library only, so it runs before anything is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_REPO = "mihaild/deepstruggle"   # web/ui/src/analysis/model.ts DEFAULT_HF_REPO
DEFAULT_REVISION = "main"

_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def order_newest_first(entries: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    """(path, date) of every .onnx file in a tree listing, newest first, as the page orders it."""
    files: List[Tuple[str, str]] = []
    for e in entries:
        path = e.get("path")
        if e.get("type") != "file" or not isinstance(path, str) or not path.endswith(".onnx"):
            continue
        commit = e.get("lastCommit")
        date = commit.get("date") if isinstance(commit, dict) else None
        files.append((path, date if isinstance(date, str) else ""))
    # ISO-8601 times order as strings. Newest first; within one date, by path ascending.
    files.sort(key=lambda f: f[0])
    files.sort(key=lambda f: f[1], reverse=True)
    return files


def list_models(repo: str, revision: str) -> List[Tuple[str, str]]:
    if not _REPO_RE.match(repo):
        raise ValueError("a Hugging Face repo must look like owner/name")
    entries: List[Dict[str, object]] = []
    url: Optional[str] = (f"https://huggingface.co/api/models/{repo}/tree/"
                          f"{urllib.parse.quote(revision, safe='')}?recursive=true&expand=true")
    while url:
        with urllib.request.urlopen(url, timeout=60) as res:
            entries.extend(json.load(res))
            m = _NEXT_RE.search(res.headers.get("Link") or "")
            url = m.group(1) if m else None
    return order_newest_first(entries)


def download(repo: str, revision: str, path: str, out_dir: str) -> str:
    dest = os.path.join(out_dir, os.path.basename(path))
    os.makedirs(out_dir, exist_ok=True)
    url = (f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/"
           f"{urllib.parse.quote(path)}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as res, open(tmp, "wb") as f:
        while chunk := res.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)   # never leave a truncated file under the real name
    return dest


def onnx_name(path: str) -> str:
    """`<run>@<steps>M.onnx` for a run directory's checkpoint, so reports name the model."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if "@" in stem:   # already named <run>@<tag> (a release asset)
        return f"{stem}.onnx"
    run = os.path.basename(os.path.dirname(path)).split("_")[0] or "model"
    m = re.fullmatch(r"snapshot_(\d+)steps", stem)
    tag = f"{round(int(m.group(1)) / 1e6)}M" if m else stem.replace("snapshot_", "")
    return f"{run}@{tag}.onnx"


def fetch_checkpoint_as_onnx(repo: str, spec: str, out_dir: str) -> str:
    """Download `<revision>:<path>.pt` and export it next to where an .onnx would have landed."""
    revision, path = spec.split(":", 1) if ":" in spec else (DEFAULT_REVISION, spec)
    ckpt = download(repo, revision, path, os.path.join(out_dir, "_pt", revision, os.path.dirname(path)))
    out = os.path.join(out_dir, onnx_name(path))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([".", "build/release", os.environ.get("PYTHONPATH", "")]))
    python = os.environ.get("TS_PYTHON", ".venv/bin/python")
    subprocess.run([python, "tools/export_onnx.py", "--checkpoint", ckpt, "--out", out],
                   check=True, env=env, stdout=sys.stderr)
    return out


def fetch_artifact(spec: str, out_dir: str) -> str:
    """`gh:<run id>/<artifact>/<file>`: download that run's artifact and return the file in it."""
    run_id, artifact, name = spec[len("gh:"):].split("/", 2)
    dest = os.path.join(out_dir, "_gh", run_id, artifact)
    repo = os.environ.get("GITHUB_REPOSITORY")
    subprocess.run(["gh", "run", "download", run_id, "-n", artifact, "-D", dest]
                   + (["-R", repo] if repo else []), check=True, stdout=sys.stderr)
    out = os.path.join(out_dir, os.path.basename(name))
    os.replace(os.path.join(dest, name), out)
    return out


def fetch_release(spec: str, out_dir: str, keep_pt: bool = False) -> str:
    """`rel:<tag>/<asset>`: download a release asset of `$GITHUB_REPOSITORY` (gh's default
    repository when unset); a `.pt` is exported to `<asset stem>.onnx`, so name it `<run>@<steps>M.pt`.
    `relpt:<tag>/<asset>.pt` (keep_pt) keeps the torch checkpoint: the export reads only the base
    observation, so a model with observation features (`obs_features`) must play as itself."""
    tag, name = spec.split(":", 1)[1].split("/", 1)
    dest = os.path.join(out_dir, "_rel", tag)
    repo = os.environ.get("GITHUB_REPOSITORY")
    # GitHub's asset server answers an occasional HTTP 500 (one runner of sixteen, 2026-10-05);
    # a retry costs seconds, a failed runner a whole rerun.
    for attempt in range(4):
        try:
            subprocess.run(["gh", "release", "download", tag, "-p", name, "-D", dest, "--clobber"]
                           + (["-R", repo] if repo else []), check=True, stdout=sys.stderr)
            break
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    src = os.path.join(dest, name)
    if keep_pt or not name.endswith(".pt"):
        out = os.path.join(out_dir, name)
        os.replace(src, out)
        return out
    out = os.path.join(out_dir, onnx_name(name))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([".", "build/release", os.environ.get("PYTHONPATH", "")]))
    python = os.environ.get("TS_PYTHON", ".venv/bin/python")
    subprocess.run([python, "tools/export_onnx.py", "--checkpoint", src, "--out", out],
                   check=True, env=env, stdout=sys.stderr)
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--path", default=None, help="a file in the repo; default the newest .onnx")
    ap.add_argument("--out-dir", default="data/checkpoints")
    ap.add_argument("--resolve-only", action="store_true",
                    help="print the path the download would take, and download nothing -- to pin "
                         "one file before several machines each fetch it")
    args = ap.parse_args(argv)

    path = args.path
    if path is None:
        models = list_models(args.repo, args.revision)
        if not models:
            print(f"fetch_hf_model: {args.repo}@{args.revision} has no .onnx files", file=sys.stderr)
            return 1
        for p, d in models:
            print(f"  {d or '(no date)':<25s} {p}")
        path = models[0][0]
        print(f"fetch_hf_model: newest is {path}")
    if args.resolve_only:
        print(path)
        return 0
    if path.startswith("pt:"):
        # pt:<path>.pt -- the PyTorch checkpoint itself, not exported: a search: entrant needs
        # the torch network (the searcher runs it), which an .onnx export cannot give it.
        print(download(args.repo, args.revision, path[len("pt:"):], args.out_dir))
    elif path.startswith("gh:"):
        print(fetch_artifact(path, args.out_dir))
    elif path.startswith("rel:"):
        print(fetch_release(path, args.out_dir))
    elif path.startswith("relpt:"):
        print(fetch_release(path, args.out_dir, keep_pt=True))
    elif path.endswith(".pt"):
        print(fetch_checkpoint_as_onnx(args.repo, path, args.out_dir))
    else:
        print(download(args.repo, args.revision, path, args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
