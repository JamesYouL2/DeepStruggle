#!/usr/bin/env python3
"""Check every markdown cross-reference, the way Obsidian resolves them.

    .venv/bin/python tools/scripts/check_doc_links.py

Exits non-zero if anything is broken, so it can gate a commit that touches documentation. The
repository root is found from this file, so it runs from any directory.

Obsidian differs from GitHub in ways that matter here:
  * a relative link resolves from the FILE's directory, so `../x.md` must be right per file;
  * a link to a directory does not open anything;
  * an anchor `#Some Heading` matches the heading TEXT, not a GitHub-style slug, so
    `#3-what-search-costs` resolves on GitHub and fails in Obsidian;
  * a bare `file.md` with no path is resolved by vault-wide filename search, which works only if
    the name is unique.

Reports broken targets, ambiguous bare names, and anchors that do not match a heading.
`tests/docs/test_research_links.py` overlaps only partly: it checks that the file part of links
under `research/` resolves, not anchors, bare names or the rest of the tree.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Union

ROOT: str = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCAN: Tuple[str, ...] = ("research", "engine", "bindings", "bot", "web", "tools", "ai", "tests", ".")
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)

#: Directory names skipped wherever they appear, matched as whole path components (a substring
#: match on ".git" also skipped ".github").
SKIP_ANYWHERE = frozenset({".git", "node_modules", "external", ".venv", "__pycache__"})


def _skipped(rel_dir: str) -> bool:
    """Whether a directory (relative to ROOT) is outside the documentation.

    Build trees are skipped only at the repository root (`build/`, `build_san/`, ...), so a
    source directory that merely has "build" in its name is still scanned.
    """
    parts = [p for p in rel_dir.split(os.sep) if p not in ("", ".")]
    if any(p in SKIP_ANYWHERE for p in parts):
        return True
    return bool(parts) and (parts[0] == "build" or parts[0].startswith("build_"))


def markdown_files() -> List[str]:
    """Every .md file in the scanned trees, relative to ROOT, sorted."""
    found: set[str] = set()
    for base in SCAN:
        top = os.path.join(ROOT, base)
        for dirpath, dirnames, filenames in os.walk(top):
            rel = os.path.relpath(dirpath, ROOT)
            if _skipped(rel):
                dirnames[:] = []
                continue
            for fn in filenames:
                if fn.endswith(".md"):
                    found.add(os.path.normpath(os.path.join(rel, fn)))
    return sorted(found)


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def obsidian_anchor_ok(headings: Dict[str, List[str]], target_file: str, anchor: str) -> Union[bool, str]:
    """Obsidian matches the heading text, case-insensitively, ignoring markdown emphasis."""
    want = anchor.replace("-", " ").strip().lower()
    for h in headings.get(target_file, []):
        clean = re.sub(r"[`*_]", "", h).strip().lower()
        if clean == anchor.strip().lower() or clean == want:
            return True
        # GitHub-style slug of the heading
        slug = re.sub(r"[^a-z0-9\s-]", "", clean).replace(" ", "-")
        if slug == anchor.strip().lower():
            return "github-only"
    return False


def main() -> int:
    md_files = markdown_files()
    texts: Dict[str, str] = {f: _read(f) for f in md_files}
    by_name: DefaultDict[str, List[str]] = defaultdict(list)
    headings: Dict[str, List[str]] = {}
    for f in md_files:
        by_name[os.path.basename(f)].append(f)
        headings[f] = [m.group(2) for m in HEADING.finditer(texts[f])]

    broken: List[Tuple[str, ...]] = []
    ambiguous: List[Tuple[str, ...]] = []
    bad_anchor: List[Tuple[str, ...]] = []
    dir_links: List[Tuple[str, ...]] = []
    total = 0
    for f in md_files:
        for m in LINK.finditer(texts[f]):
            target = m.group(2).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            total += 1
            path, _, anchor = target.partition("#")
            path = path.strip()
            if not path:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(f), path))
            absolute = os.path.join(ROOT, resolved)
            if os.path.isdir(absolute):
                dir_links.append((f, target))
                continue
            if not os.path.exists(absolute):
                if "/" not in path and path in by_name:
                    if len(by_name[path]) > 1:
                        ambiguous.append((f, target, ", ".join(by_name[path])))
                else:
                    broken.append((f, target))
                continue
            if anchor and resolved.endswith(".md"):
                ok = obsidian_anchor_ok(headings, resolved, anchor)
                if ok is not True:
                    bad_anchor.append((f, target, "github-slug only" if ok else "no such heading"))

    print(f"{len(md_files)} markdown files, {total} internal links\n")
    for name, rows in (("BROKEN TARGET", broken), ("LINK TO A DIRECTORY", dir_links),
                       ("AMBIGUOUS BARE NAME", ambiguous), ("ANCHOR NOT OBSIDIAN-RESOLVABLE", bad_anchor)):
        print(f"{name}: {len(rows)}")
        for r in rows[:25]:
            print("   ", " -> ".join(r))
        if len(rows) > 25:
            print(f"    ... and {len(rows) - 25} more")
        print()
    return 1 if (broken or dir_links or ambiguous or bad_anchor) else 0


if __name__ == "__main__":
    sys.exit(main())
