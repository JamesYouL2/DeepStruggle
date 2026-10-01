"""Greedy self-play games of one model, written as traced .tslog.json replays for a human to review.

The review method of `research/log/E5_11_selfplay_review.md`: a reader goes through the games in
the workbench, names the decisions that look wrong, and each one is checked by paired playouts
(`tools/branch_oracle.py`). Every replay goes through `generate_self_play_replay`, the one replay
writer, with the policy/critic trace on, so the workbench shows what the model believed at every
step.

    PYTHONPATH=.:build/release .venv/bin/python tools/review_games.py \\
        --model data/checkpoints/E6-06-44@soup_680-760.onnx --games 8 --out-dir data/replays/review

On CI: `.github/workflows/review_games.yml`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.lib.self_play import generate_self_play_replay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="A .pt checkpoint or a published .onnx export")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=9_100_001,
                    help="Game k is played on seed seed-base + k")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 plays the argmax (greedy), as the E5-11 review did")
    ap.add_argument("--opening", default=None, help="A named opening from tools/lib/openings.py")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    label = os.path.splitext(os.path.basename(a.model))[0]
    index = []
    for k in range(a.games):
        seed = a.seed_base + k
        gid = f"review_{label}_seed{seed}".replace("@", "_")
        path = os.path.join(a.out_dir, f"{gid}.tslog.json")
        doc, _ = generate_self_play_replay(
            a.model, model_name=label, seed=seed, temperature=a.temperature, game_id=gid,
            output_path=path, device="cpu", verbose=False, opening=a.opening)
        meta = doc.get("metadata", {})
        result = meta.get("result") or meta.get("winner") or ""
        steps = len(doc.get("steps", []))
        index.append({"file": os.path.basename(path), "seed": seed, "steps": steps, "result": result})
        print(f"{path}: {steps} steps, {result}")
    with open(os.path.join(a.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"model": label, "temperature": a.temperature, "opening": a.opening,
                   "games": index}, f, indent=1)


if __name__ == "__main__":
    main()
