"""Imports live at the top of the module, not inside functions or classes.

A function-level import hides a dependency, defers its failure to the first call, and in this
repository has twice stood in for an import cycle that injection or `TYPE_CHECKING` should have
broken. This check is a ratchet: `nested_imports_baseline.json` records, per file, how many such
imports were there when the check was introduced, and a file may only go down. A new file, or
a file not listed, must have none. When you remove some, regenerate the baseline so it cannot
creep back:

    .venv/bin/python tests/lint/test_imports_at_top.py

Imports directly under a module-level `if` or `try` (an optional dependency, `TYPE_CHECKING`)
are top-level and allowed.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
BASELINE = os.path.join(_HERE, "nested_imports_baseline.json")
TREES = ("ai", "tools", "bindings", "web", "tests")
SKIP = ("tests/differential",)


def nested_imports(path: str) -> List[int]:
    """The lines of the imports inside a function or class body."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    lines: List[int] = []

    def walk(node: ast.AST, inside: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if inside:
                    lines.append(child.lineno)
                continue
            scoped = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
            walk(child, inside or scoped)

    walk(tree, False)
    return lines


def python_files() -> List[str]:
    out: List[str] = []
    for tree in TREES:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, tree)):
            rel = os.path.relpath(dirpath, ROOT)
            if any(rel == s or rel.startswith(s + os.sep) for s in SKIP):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "node_modules", ".venv")]
            out.extend(os.path.join(dirpath, f) for f in filenames if f.endswith(".py"))
    return sorted(out)


def census() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in python_files():
        n = len(nested_imports(path))
        if n:
            counts[os.path.relpath(path, ROOT).replace(os.sep, "/")] = n
    return counts


def test_no_new_nested_imports() -> None:
    with open(BASELINE, encoding="utf-8") as f:
        baseline: Dict[str, int] = json.load(f)
    now = census()
    worse: List[Tuple[str, int, int]] = []
    better: List[Tuple[str, int, int]] = []
    for path, n in now.items():
        was = baseline.get(path, 0)
        if n > was:
            worse.append((path, was, n))
        elif n < was:
            better.append((path, was, n))
    better.extend((path, was, 0) for path, was in baseline.items() if path not in now)
    msg: List[str] = []
    if worse:
        msg.append("imports inside a function or class (hoist them to the module top; "
                   "break a cycle with TYPE_CHECKING or injection, never a lazy import):")
        for path, was, n in worse:
            lines = nested_imports(os.path.join(ROOT, path))
            msg.append(f"  {path}: {n} (baseline {was}) at lines {lines}")
    if better:
        msg.append("fewer nested imports than the baseline -- good; regenerate it so they cannot "
                   "return: .venv/bin/python tests/lint/test_imports_at_top.py")
        msg.extend(f"  {path}: {n} (baseline {was})" for path, was, n in better)
    assert not msg, "\n".join(msg)


if __name__ == "__main__":
    counts = census()
    with open(BASELINE, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{sum(counts.values())} nested imports in {len(counts)} files written to {BASELINE}",
          file=sys.stderr)
