"""Deep search on the USSR opening, paired by deal (`ai/eval/opening_search.py`).

    # search part k of N of the deals
    PYTHONPATH=.:build/release .venv/bin/python tools/opening_search.py search \\
        --model data/checkpoints/E6-06-44@soup_680-760.onnx --deals 400 --part 1/16 \\
        --budgets 64 1024 8192 --worlds 4 --us-setup fixed --out part-1.jsonl
    # pool the parts
    PYTHONPATH=.:build/release .venv/bin/python tools/opening_search.py report \\
        --parts part-*.jsonl --out-md report.md --out-json report.json

On CI: `.github/workflows/opening_search.yml`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.opening_search import report, search_deals  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("--model", required=True)
    s.add_argument("--deals", type=int, default=400)
    s.add_argument("--seed-base", type=int, default=7_000_000,
                   help="Deal k is seed seed-base + k, the same for every opening")
    s.add_argument("--part", default="1/1", help="k/N: every N-th deal from the k-th")
    s.add_argument("--budgets", type=int, nargs="+", default=[64, 1024, 8192])
    s.add_argument("--worlds", type=int, default=4, help="Determinizations per (deal, opening, budget)")
    s.add_argument("--us-setup", choices=["fixed", "net"], default="fixed",
                   help="fixed: WG 3 / France 3 / Italy 2 / Iran 2; net: the network's own US setup")
    s.add_argument("--out", required=True)

    r = sub.add_parser("report")
    r.add_argument("--parts", nargs="+", required=True)
    r.add_argument("--out-md", required=True)
    r.add_argument("--out-json", required=True)

    a = ap.parse_args()
    if a.cmd == "search":
        k, n = (int(x) for x in a.part.split("/"))
        seeds = [a.seed_base + i for i in range(a.deals)][k - 1::n]
        rows = search_deals(a.model, seeds, a.budgets, a.worlds, us_setup=a.us_setup,
                            search_seed=1000 * k)
        with open(a.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"part {k}/{n}: {len(seeds)} deals, {len(rows)} rows -> {a.out}")
    else:
        rows = []
        for path in a.parts:
            with open(path, "r", encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
        md, js = report(rows)
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(js, f, indent=1)
        print(md)


if __name__ == "__main__":
    main()
