"""Check a named decision in a replay by paired playouts of its alternatives (`ai/eval/branch_oracle.py`).

    # what was legal at step 120, and what the model thought of each option
    PYTHONPATH=.:build/release .venv/bin/python tools/branch_oracle.py options \\
        --replay game.tslog.json --step 120 --model model.onnx
    # ...and after a partial alternative (a card, then the node its play leads to)
    ... options --replay game.tslog.json --step 120 --model model.onnx --prefix 37

    # play the policy's choice, the recorded move and two alternatives out, 256 pairs each
    ... run --replay game.tslog.json --step 120 --model model.onnx \\
        --alt cambridge=37,111 --alt nasser_ops=14,112 --pairs 256 --out part-1.jsonl
    ... report --parts part-*.jsonl --out-md report.md --out-json report.json

An alternative is `name=a,b,c`: flat actions forced in order from the decision, each of which must
be legal where it lands (`options --prefix` shows what is). On CI:
`.github/workflows/branch_oracle.yml`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import ts_engine as ts  # noqa: E402

from ai.eval.branch_oracle import (apply_prefix, onnx_policy, options, play_branches,  # noqa: E402
                                   position_before, report)
from bindings.action_encoder import ActionEncoder  # noqa: E402


def _ints(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x != ""]


def parse_alts(specs: List[str]) -> Dict[str, List[int]]:
    """`name=a,b` or bare `a,b` (named alt1, alt2, ...)."""
    out: Dict[str, List[int]] = {}
    for i, spec in enumerate(specs, 1):
        name, _, acts = spec.rpartition("=") if "=" in spec else (f"alt{i}", "", spec)
        out[name] = _ints(acts)
    return out


def describe(state: ts.GameState) -> str:
    mover = "US" if state.ctx().decision_player == ts.Player.US else "USSR"
    return (f"turn {state.turn} AR {state.action_round}, {mover} to decide "
            f"({ts.DecisionType(int(state.ctx().decision_type)).name}); VP {state.victory_points:+d} "
            f"(+ = US), DEFCON {state.defcon}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("options")
    o.add_argument("--replay", required=True)
    o.add_argument("--step", type=int, required=True)
    o.add_argument("--model", required=True)
    o.add_argument("--prefix", default="", help="Flat actions applied first, comma-separated")

    r = sub.add_parser("run")
    r.add_argument("--replay", required=True)
    r.add_argument("--step", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--alt", action="append", default=[], help="name=a,b,c (repeatable)")
    r.add_argument("--pairs", type=int, default=256)
    r.add_argument("--hidden", choices=["resample", "true"], default="resample")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--part", default="1/1", help="k/N: pairs k-1, k-1+N, ...")
    r.add_argument("--out", required=True)

    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)

    a = ap.parse_args()
    if a.cmd == "options":
        state, played, rec = position_before(a.replay, a.step)
        _, probs = onnx_policy(a.model)
        st = apply_prefix(state, _ints(a.prefix))
        print(f"step {a.step}: {describe(state)}")
        print(f"  recorded: {played} {rec['description']}")
        if a.prefix:
            print(f"  after prefix {a.prefix}: {describe(st)}")
        for row in options(st, probs):
            print(f"  {row['idx']:4d}  p={row.get('p', 0):.3f}  {row['name']}")
    elif a.cmd == "run":
        state, played, rec = position_before(a.replay, a.step)
        act, probs = onnx_policy(a.model)
        greedy = int(np.argmax(probs(state)))
        branches: Dict[str, List[int]] = {"policy": []}
        if played != greedy:
            branches["played"] = [played]
        branches.update(parse_alts(a.alt))
        k, n = (int(x) for x in a.part.split("/"))
        pairs = list(range(a.pairs))[k - 1::n]
        rows = play_branches(state, branches, pairs, act, seed=a.seed, hidden=a.hidden)
        meta = {"replay": os.path.basename(a.replay), "step": a.step, "position": describe(state),
                "recorded": [played, rec["description"]],
                "policy_greedy": [greedy, ActionEncoder.get_action_name(state, greedy)],
                "branches": branches,
                "hidden": a.hidden}
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": meta}) + "\n")
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"part {k}/{n}: {len(pairs)} pairs x {len(branches)} branches -> {a.out}")
    else:
        rows, meta = [], None
        for path in a.parts:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if "meta" in d:
                        meta = d["meta"]
                    else:
                        rows.append(d)
        md, js = report(rows)
        head = ""
        if meta:
            head = (f"# Branch oracle: {meta['replay']}, step {meta['step']}\n\n{meta['position']}. "
                    f"Recorded: `{meta['recorded'][1]}`; the model's greedy choice: "
                    f"`{meta['policy_greedy'][1]}`. Hidden cards: {meta['hidden']}.\n\n"
                    + "".join(f"* `{b}`: forced {v or 'nothing (the policy)'}\n"
                              for b, v in meta["branches"].items()) + "\n")
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(head + md)
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "branches": js}, f, indent=1)
        print(head + md)


if __name__ == "__main__":
    main()
