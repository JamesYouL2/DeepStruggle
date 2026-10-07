#!/usr/bin/env python3
"""The human-marked disagreement bank: positions from the human corpus where a strong player marked
which moves are good and which are bad, and a model scored against those marks.

The marks are made on the review page (`tools/scripts/disagreement_review.html`), which keeps one
document per position in its artifact's `verdicts` collection. `collect` turns an export of that
collection (one `<id>.json` per document, as the artifact database tool saves it) into the
committed bank, `ai/eval/banks/disagreement_verdicts.jsonl`:

* one row per position, keyed by `row_id` (the position and the decision kind);
* a document the page carried to a new id after the CONV-1 converter fix (`moved_from`) replaces
  the one it came from, so a position reviewed before and after the fix appears once;
* `marks` maps a move, named as `disagreement_bank._name` names it, to "good" or "bad". A position
  with no marks is kept (it may be flagged `unclear`), but scores nothing. A mark on a move that is
  not legal at the position -- a card the fix took out of the hand the reviewer saw -- is moved to
  `stale_marks`.

`score` asks an agent for its move at every marked position and counts how often it lands on a
move marked good, one marked bad, or one the reviewer did not mark:

    PYTHONPATH=.:build/release python tools/scripts/bank_verdicts.py score --agent newest.onnx
    PYTHONPATH=.:build/release python tools/scripts/bank_verdicts.py score --agent gumbel:newest.pt:256:8
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import require_e4_view, state_from_token
from tools.lib.player_agent import BatchSelector, load_agent
from tools.scripts.disagreement_bank import _name, mover_of, row_id

BANK = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "ai", "eval", "banks",
                                    "disagreement_verdicts.jsonl"))

#: the fields kept from a review-page document, in this order
FIELDS = ("id", "game", "kind", "pattern", "side", "turn", "ar", "card", "human", "network", "search",
          "strong", "z", "pl_best", "marks", "stale_marks", "unclear", "confidence", "bank", "note", "pos")


def collect(src: str, out: str) -> int:
    docs: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(src, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        docs[os.path.splitext(os.path.basename(path))[0]] = d.get("data", d)
    superseded = {d["moved_from"] for d in docs.values() if d.get("moved_from")}
    rows: List[Dict[str, Any]] = []
    for doc_id, d in sorted(docs.items(), key=lambda kv: (kv[1]["game"], kv[1]["turn"], kv[1]["ar"], kv[0])):
        if doc_id in superseded:
            continue
        r: Dict[str, Any] = {"id": doc_id, **{f: d.get(f) for f in FIELDS if f not in ("id", "stale_marks")}}
        if row_id(r) != doc_id:
            raise SystemExit(f"{doc_id}: the id is not the hash of its position ({row_id(r)})")
        legal = legal_names(state_from_token(r["pos"]))
        r["stale_marks"] = {m: v for m, v in r["marks"].items() if m not in legal}
        r["marks"] = {m: v for m, v in r["marks"].items() if m in legal}
        r = {f: r[f] for f in FIELDS}
        rows.append(r)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    marked = sum(1 for r in rows if r["marks"])
    stale = sum(len(r["stale_marks"]) for r in rows)
    print(f"{len(rows)} positions ({marked} marked, {stale} stale marks set aside, "
          f"{len(superseded)} superseded documents dropped) -> {out}", file=sys.stderr)
    return 0


def load(path: str = BANK) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def legal_names(st: ts.GameState) -> Dict[str, int]:
    """Every legal move at the position by its reader's name."""
    mask = np.asarray(ActionEncoder.get_legal_mask(st))
    return {_name(st, int(a)): int(a) for a in np.flatnonzero(mask)}


def score(agent_spec: str, path: str, by: Sequence[str], out: Optional[str]) -> int:
    rows = [r for r in load(path) if r["marks"]]
    agent: Any = load_agent(agent_spec, device="cuda" if torch.cuda.is_available() else "cpu")
    # The marks name moves as decoded in the E4 view (legal_names), so the agent must decide in it.
    require_e4_view(agent_spec, agent)
    if hasattr(agent, "reseed"):
        agent.reseed(0)
    states = [state_from_token(r["pos"]) for r in rows]
    if isinstance(agent, BatchSelector):
        picks = [int(a) for a in agent.select_actions_batch(states)]
    else:
        picks = [int(agent.select_action(st, mover_of(st), temperature=0.0)) for st in states]
    results: List[Dict[str, Any]] = []
    for r, st, a in zip(rows, states, picks):
        move = _name(st, a)
        results.append({"id": r["id"], "move": move, "mark": r["marks"].get(move, "unmarked"),
                        **{k: r[k] for k in ("kind", "pattern", "side", "confidence")}})
    if out:
        with open(out, "w") as f:
            for x in results:
                f.write(json.dumps(x) + "\n")
    print(f"{agent_spec}: {len(results)} marked positions")
    for key in ("all", *by):
        groups: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for x in results:
            groups["all" if key == "all" else str(x[key] or "-")][x["mark"]] += 1
        for g, c in sorted(groups.items()):
            n = sum(c.values())
            print(f"  {key:>10} {g:<14} n={n:>3}  good {c['good'] / n:5.1%}  bad {c['bad'] / n:5.1%}"
                  f"  unmarked {c['unmarked'] / n:5.1%}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="the review page's exported verdicts -> the bank")
    c.add_argument("src", help="a directory of <id>.json documents from the page's `verdicts` collection")
    c.add_argument("--out", default=BANK)
    s = sub.add_parser("score", help="an agent's moves at the marked positions, against the marks")
    s.add_argument("--agent", required=True, help="a load_agent spec: .pt, .onnx, gumbel:<pt>:sims:k, ...")
    s.add_argument("--bank", default=BANK)
    s.add_argument("--by", nargs="*", default=["kind", "confidence"],
                   choices=["kind", "pattern", "side", "confidence"], help="also break the score down by these")
    s.add_argument("--out", default=None, help="JSONL of the agent's move and its mark, per position")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        return collect(a.src, a.out)
    return score(a.agent, a.bank, a.by, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
