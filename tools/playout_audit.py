"""The playout audit: where the model's own playouts disagree with what it plays (`ai/eval/playout_audit.py`).

    # screen: 200 decisions, every alternative 32 paired playouts
    PYTHONPATH=.:build/release .venv/bin/python tools/playout_audit.py run \\
        --model data/checkpoints/<model>.onnx --positions 200 --pairs 32 --seed 1 --out parts/part-1.json.gz
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json.gz
    # confirm the largest regrets with fresh dice
    ... confirm --model ... --report report.json.gz --top 64 --pairs 512 --out-md confirm.md

On CI: `.github/workflows/playout_audit.yml`.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.branch_oracle import onnx_policy  # noqa: E402
from ai.eval.playout_audit import audit, collect, confirm, report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--positions", type=int, default=200)
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--card-branches", type=int, default=4, help="cards tried at a card choice (the most likely)")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    c = sub.add_parser("confirm")
    c.add_argument("--model", required=True)
    c.add_argument("--report", required=True, help="the report's --out-json")
    c.add_argument("--top", type=int, default=64)
    c.add_argument("--pairs", type=int, default=512)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out-md", required=True)
    c.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        act, probs = onnx_policy(a.model)
        positions = collect(act, a.positions, a.seed)
        print(f"{len(positions)} decisions ({round(time.time() - t0)}s)", flush=True)
        rows = []
        step = 25
        for lo in range(0, len(positions), step):
            rows += audit(act, probs, positions[lo:lo + step], a.pairs, a.seed * 7_919 + lo, a.card_branches)
            print(f"  {min(lo + step, len(positions))}/{len(positions)} ({round(time.time() - t0)}s)", flush=True)
        meta = {"model": os.path.basename(a.model), "pairs": a.pairs, "card_branches": a.card_branches,
                "seed": a.seed, "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump({"meta": meta, "rows": rows}, f)
        print(f"{len(rows)} decisions, {meta['seconds']}s")
    elif a.cmd == "report":
        rows, meta = [], None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            key = {k: part["meta"][k] for k in ("model", "pairs", "card_branches")}
            if meta not in (None, key):
                raise SystemExit(f"{fn} was run as {key}, not {meta}")
            meta = key
            rows += part["rows"]
        md, summary = report(rows, meta or {})
        with open(a.out_md, "w") as f:
            f.write(md)
        with gzip.open(a.out_json, "wt") as f:
            json.dump({**summary, "rows": rows}, f)
        print(md)
    else:
        from ai.eval.playout_audit import position_link
        import ts_engine as ts

        with gzip.open(a.report, "rt") as f:
            rep = json.load(f)
        worst = sorted(rep["rows"], key=lambda r: -r["regret"])[:a.top]
        act, probs = onnx_policy(a.model)
        res = confirm(act, probs, worst, a.pairs, a.seed)
        lines = [f"# Playout audit, confirmed — {rep['meta'].get('model', '?')}", "",
                 f"The {len(res)} largest screening regrets, re-played with fresh dice and redeals at {a.pairs} "
                 f"pairs. Confirmed gain = the screen's best branch minus the model's choice, on the new pairs.", "",
                 "| kind | side | turn | DEFCON | card | model's choice | screen's best | screen regret | confirmed gain | |",
                 "|:---|:---|---:|---:|:---|:---|:---|---:|---:|:---|"]
        for r in sorted(res, key=lambda r: -r["confirmed_gain"]):
            se = r["se"].get(r["screen_best"], float("nan"))
            link = position_link(ts.state_from_save_json(r["save"]))
            lines.append(f"| {r['kind']} | {r['side']} | {r['turn']}.{r['ar']} | {r['defcon']} | {r.get('card', '—')} | "
                         f"{r['chosen']} | {r['screen_best']} | {100 * r['screen_regret']:+.0f} | "
                         f"{100 * r['confirmed_gain']:+.1f} ± {100 * se:.1f} | [position]({link}) |")
        held = [r for r in res if r["screen_best"] != r["chosen"]]
        if held:
            lines += ["", f"Mean confirmed gain over the {len(held)} where the screen's best was not the model's "
                      f"choice: {100 * sum(r['confirmed_gain'] for r in held) / len(held):+.1f} pts."]
        md = "\n".join(lines) + "\n"
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "save"} for r in res], f)
        print(md)


if __name__ == "__main__":
    main()
