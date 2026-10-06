#!/usr/bin/env python3
"""Paired playouts for disagreement-bank positions: each of the human's, the network's and the
search's moves played out by the model (`ai/eval/paired_playouts.py`).

Three steps:

* `select` -- the positions to play, from `bank_clarity.py`'s output: every position whose value lead
  is at least `--min-gap` win-probability points (or else the `--top` largest margins in SEs),
  written with their positions and move indices as one input file;
* `run` -- this part's share of an input file (`--part k/N`, for CI runners): every distinct move
  of every position, `--pairs` pairs each, the network playing both sides greedily;
* `pool` -- the parts merged into one file.

A row of `run`'s output: each move's mean score for the mover (1 win, 0.5 draw, 0 loss), and the
paired differences between the three moves with their standard errors.

    PYTHONPATH=.:build/release python tools/scripts/bank_playouts.py select --clarity clarity*.jsonl.gz \\
        --bank part*.jsonl.gz --top 1000 --out playout_input.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/bank_playouts.py run --input playout_input.jsonl.gz \\
        --onnx newest.onnx --pairs 32 --part 1/20 --out playouts-1.jsonl.gz
    PYTHONPATH=. python tools/scripts/bank_playouts.py pool --parts playouts-*.jsonl.gz --out playouts.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

WHO = ("human", "network", "search")


def _read(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                yield json.loads(line)


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def value_gap(c: Dict[str, Any]) -> Optional[float]:
    """The value-best move's lead over the next distinct move, in win-probability points (values run
    from -1 to +1, so half the difference). A lead matters only if it is large: in a game that is
    won or lost whatever happens, every move is worth about the same."""
    vals = sorted({(v[0], v[1]) for v in c["q"].values()}, key=lambda v: -v[0])
    return (vals[0][0] - vals[1][0]) / 2 * 100 if len(vals) > 1 else None


def select(clarity: Sequence[str], banks: Sequence[str], top: int, out: str, min_gap: float = 0.0) -> int:
    from tools.scripts.disagreement_bank import row_id

    if min_gap > 0:
        cl = [c for c in _read(clarity) if (value_gap(c) or 0.0) >= min_gap]
        cl.sort(key=lambda c: -(value_gap(c) or 0.0))
        cl = cl[:top]
    else:
        cl = sorted((c for c in _read(clarity) if c.get("z") is not None), key=lambda c: -c["z"])[:top]
    want = {c["id"]: c for c in cl}
    pos: Dict[str, str] = {}
    for r in _read(banks):
        rid = row_id(r)
        if rid in want and rid not in pos:
            pos[rid] = r["pos"]
    rows = [{"id": c["id"], "pos": pos[c["id"]], "a": c["a"]} for c in cl if c["id"] in pos]
    _write(out, rows)
    print(f"{len(rows)} positions selected", file=sys.stderr)
    return 0


def run(inp: str, onnx: str, pairs: int, part: str, seed: int, out: str) -> int:
    from ai.eval.paired_playouts import compare, paired_diff
    from tools.lib.player_agent import OnnxAgent
    from tools.scripts.event_play_census import state_from_token

    k, n = (int(x) for x in part.split("/"))
    rows = [r for i, r in enumerate(_read([inp])) if i % n == k - 1]
    agent = OnnxAgent(onnx)

    def act(obs: Any, masks: Any) -> Any:
        return agent.act_batch(obs, masks, 0.0, True)

    t0 = time.time()
    positions = [(state_from_token(r["pos"]), sorted(set(int(v) for v in r["a"].values()))) for r in rows]
    scores = compare(positions, act, pairs, seed)
    res: List[Dict[str, Any]] = []
    for r, sc in zip(rows, scores):
        a = {w: int(v) for w, v in r["a"].items()}
        mean = {w: round(sum(sc[a[w]]) / len(sc[a[w]]), 4) for w in WHO}
        diffs = {}
        for x, y in (("human", "network"), ("human", "search"), ("network", "search")):
            if a[x] != a[y]:
                m, se = paired_diff(sc[a[x]], sc[a[y]])
                diffs[f"{x}-{y}"] = [round(m, 4), round(se, 4)]
        best = max(set(a.values()), key=lambda m: sum(sc[m]))
        res.append({"id": r["id"], "pairs": pairs, "score": mean, "diff": diffs,
                    "best": [w for w in WHO if a[w] == best]})
    _write(out, res)
    print(f"{len(res)} positions, {pairs} pairs, in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--clarity", nargs="+", required=True)
    s.add_argument("--bank", nargs="+", required=True)
    s.add_argument("--top", type=int, default=1000)
    s.add_argument("--min-gap", type=float, default=0.0,
                   help="take every position whose value lead is at least this many win-probability points "
                        "(largest first, up to --top) instead of the largest margins in SEs")
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", required=True)
    r.add_argument("--onnx", required=True)
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--part", default="1/1")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        return select(a.clarity, a.bank, a.top, a.out, a.min_gap)
    if a.cmd == "run":
        return run(a.input, a.onnx, a.pairs, a.part, a.seed, a.out)
    _write(a.out, list(_read(a.parts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
