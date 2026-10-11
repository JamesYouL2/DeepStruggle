"""JSON-lines files, plain or gzipped by their suffix, and a file's SHA-256.

The bank tools (`tools/scripts/bank_playouts.py`, `blunder_census.py`, `blunder_bank.py`) read
and write rows the same way: one JSON object per line, gzipped where the path ends in `.gz` and
plain otherwise, so the committed bank (`ai/eval/banks/*.jsonl`) and a CI part (`*.jsonl.gz`) go
through one reader. `sha256_file` names the checkpoint a row was made with.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from typing import IO, Any, Dict, Iterator, Sequence, cast


def _open(path: str, mode: str) -> IO[str]:
    if path.endswith(".gz"):
        return cast(IO[str], gzip.open(path, mode + "t"))
    return open(path, mode)


def read_jsonl(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    """The rows of each file in turn."""
    for p in paths:
        with _open(p, "r") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with _open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with _open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
