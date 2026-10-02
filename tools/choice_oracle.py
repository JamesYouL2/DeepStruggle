"""The choice oracle: named alternatives at a named kind of decision, by paired playouts (`ai/eval/choice_oracle.py`).

    # one part: up to 24 positions per scenario, 64 pairs each
    PYTHONPATH=.:build/release .venv/bin/python tools/choice_oracle.py run \\
        --model data/checkpoints/<model>.onnx --per 24 --pairs 64 --seed 1 --out parts/part-1.json
    ... report --parts parts/part-*.json --out-md report.md --out-json report.json

On CI: `.github/workflows/choice_oracle.yml`. Scenarios are `choice_oracle.SCENARIOS`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.branch_oracle import onnx_policy  # noqa: E402
from ai.eval.choice_oracle import collect, play, report, scenarios  # noqa: E402
from ai.eval.doctrine_census import MODES, MODE_BASE, N_CARDS, cards  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--per", type=int, default=24, help="positions per scenario")
    r.add_argument("--pairs", type=int, default=64)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--only", default=None, help="run only scenarios whose name contains one of these |-separated substrings")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        info = cards()

        def label(action: int) -> str:
            if action < N_CARDS:
                return str(info[action + 1]["name"])
            if MODE_BASE <= action < MODE_BASE + 5:
                return MODES[action - MODE_BASE]
            return str(action)

        act, _ = onnx_policy(a.model)
        scs = [s for s in scenarios() if a.only is None or any(o in s.name for o in a.only.split("|"))]
        found = collect(act, scs, a.per, a.seed)
        rows = []
        for s in scs:
            rows += play(act, s, found[s.name], a.pairs, a.seed, label=label)
            print(f"{s.name}: {len(found[s.name])} positions ({round(time.time() - t0)}s)", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"meta": {"model": os.path.basename(a.model), "seed": a.seed,
                                "seconds": round(time.time() - t0, 1)}, "rows": rows}, f)
    else:
        rows, model = [], None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with open(fn) as f:
                part = json.load(f)
            if model not in (None, part["meta"]["model"]):
                raise SystemExit(f"{fn} was played by {part['meta']['model']}, not {model}")
            model = part["meta"]["model"]
            rows += part["rows"]
        md, summary = report(rows, {"model": model})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump({**summary, "rows": rows}, f)
        print(md)


if __name__ == "__main__":
    main()
