"""One world against eight: how noisy a determinized search target is, and where it depends on
the sampled world (`ai/eval/determinization_targets.py`).

Three stages, so CI can spread the search over runners
(`.github/workflows/determinization_probe.yml`):

    # 1. positions from the model's own self-play at the rollout temperature
    PYTHONPATH=.:build/release .venv/bin/python tools/determinization_probe.py positions \\
        --model data/checkpoints/E6-06-44@soup_680-760.onnx --n 2000 --out positions.jsonl
    # 2. search part k of N of them
    PYTHONPATH=.:build/release .venv/bin/python tools/determinization_probe.py search \\
        --model <same> --positions positions.jsonl --part 1/16 --out part-1.jsonl
    # 3. pool the parts into a report
    PYTHONPATH=.:build/release .venv/bin/python tools/determinization_probe.py report \\
        --parts part-*.jsonl --out-md report.md --out-json report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.determinization_targets import (blind_spots, collect_positions,  # noqa: E402
                                             dump_positions, load_positions, report,
                                             search_positions, verify_blind_spots)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("positions")
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Rollout temperature; training samples at 1.0")
    p.add_argument("--out", required=True)

    s = sub.add_parser("search")
    s.add_argument("--model", required=True)
    s.add_argument("--positions", required=True)
    s.add_argument("--part", default="1/1", help="k/N: search every N-th position from the k-th")
    s.add_argument("--worlds", type=int, default=8)
    s.add_argument("--sims", type=int, default=64)
    s.add_argument("--small-sims", type=int, default=8,
                   help="Simulations per world in the equal-budget split (worlds x small = sims)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dirichlet-frac", type=float, default=0.0,
                   help="Root noise mixed into every determinized search (0 = the training searcher)")
    s.add_argument("--dirichlet-alpha", type=float, default=1.0)
    s.add_argument("--out", required=True)

    v = sub.add_parser("verify", help="Paired playouts at every blind-spot candidate in the parts")
    v.add_argument("--model", required=True)
    v.add_argument("--positions", required=True)
    v.add_argument("--parts", nargs="+", required=True)
    v.add_argument("--pairs", type=int, default=256)
    v.add_argument("--out-md", required=True)
    v.add_argument("--out-json", required=True)

    r = sub.add_parser("report")
    r.add_argument("--parts", nargs="+", required=True)
    r.add_argument("--out-md", required=True)
    r.add_argument("--out-json", required=True)

    a = ap.parse_args()
    if a.cmd == "positions":
        pos = collect_positions(a.model, a.n, a.seed, temperature=a.temperature)
        dump_positions(a.out, pos)
        print(f"wrote {len(pos)} positions to {a.out}")
    elif a.cmd == "search":
        k, n = (int(x) for x in a.part.split("/"))
        allpos = load_positions(a.positions)
        idx = list(range(len(allpos)))[k - 1::n]
        # The part index goes into the seed so two parts never draw the same worlds.
        rows = search_positions(a.model, [allpos[i] for i in idx], worlds=a.worlds, sims=a.sims,
                                small_sims=a.small_sims, seed=a.seed + 1000 * k,
                                dirichlet_frac=a.dirichlet_frac, dirichlet_alpha=a.dirichlet_alpha,
                                indices=idx)
        with open(a.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"part {k}/{n}: searched {len(rows)} positions -> {a.out}")
    elif a.cmd == "verify":
        rows = []
        for path in a.parts:
            with open(path, "r", encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
        cands = blind_spots(rows)
        md, out = verify_blind_spots(a.model, load_positions(a.positions), cands, pairs=a.pairs)
        md = (f"# Blind-spot candidates, checked by paired playouts\n\n{len(cands)} candidates of "
              f"{len(rows)} positions ({a.pairs} pairs each).\n\n" + md)
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(md)
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
