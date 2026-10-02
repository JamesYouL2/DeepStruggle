"""Training labels from the model's own paired playouts (`ai/eval/playout_labels.py`).

    # one part: 1,500 decisions from greedy self-play, top-4 candidates, 24 paired playouts each
    PYTHONPATH=.:build/release .venv/bin/python tools/playout_labels.py run \\
        --model data/checkpoints/<model>.onnx --decisions 1500 --pairs 24 --seed 1 --out parts/labels-1.npz
    # pool the parts into one file and summarise what the labels would teach
    ... pool --parts parts/labels-*.npz --out labels.npz --out-md report.md --out-json report.json

On CI: `.github/workflows/playout_labels.yml`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.branch_oracle import onnx_policy  # noqa: E402
from ai.eval.playout_labels import collect, label, pack, report  # noqa: E402


def _save(path: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kw: Dict[str, Any] = {"meta": np.array(json.dumps(meta)), **arrays}
    np.savez_compressed(path, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--decisions", type=int, default=1500)
    r.add_argument("--pairs", type=int, default=24)
    r.add_argument("--k", type=int, default=4, help="candidates per decision (all legal modes at a play-mode decision)")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        act, probs = onnx_policy(a.model)
        positions = collect(act, a.decisions, a.seed)
        print(f"{len(positions)} decisions ({round(time.time() - t0)}s)", flush=True)
        labels = []
        step = 50
        for lo in range(0, len(positions), step):
            labels += label(act, probs, positions[lo:lo + step], a.pairs, a.seed * 7_919 + lo, a.k)
            print(f"  {min(lo + step, len(positions))}/{len(positions)} ({round(time.time() - t0)}s)", flush=True)
        arrays = pack(labels)
        meta = {"model": os.path.basename(a.model), "pairs": a.pairs, "k": a.k, "seed": a.seed,
                "seconds": round(time.time() - t0, 1)}
        _save(a.out, meta, arrays)
        print(f"{len(labels)} labels, {meta['seconds']}s")
    else:
        parts, meta = [], None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with np.load(fn) as z:
                m = json.loads(str(z["meta"]))
                key = {k: m[k] for k in ("model", "pairs", "k")}
                if meta not in (None, key):
                    raise SystemExit(f"{fn} was labelled as {key}, not {meta}")
                meta = key
                parts.append({k: z[k] for k in z.files if k != "meta"})
        if not parts:
            raise SystemExit("no parts")
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        _save(a.out, meta or {}, arrays)
        md, summary = report(arrays, meta or {})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump(summary, f)
        print(md)


if __name__ == "__main__":
    main()
