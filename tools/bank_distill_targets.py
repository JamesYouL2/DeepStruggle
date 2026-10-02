#!/usr/bin/env python3
"""Distillation targets from position-bank spots: the event, where the bank says the event is right.

The expert review measured, with the model's own paired playouts, spots where it never events a card
whose event is worth points (One Small Step one box behind, KAL-007 with South Korea US-controlled,
Glasnost with The Reformer, Special Relationship with NATO). A rule confirmed over a whole spot is
distilled, not a per-position argmax: one position's 64 pairs carry a standard error of several
points, the spot's mean a fraction of one.

    # targets: every position of each named spot, event as the target, mixed into anchor games
    PYTHONPATH=.:build/release .venv/bin/python tools/bank_distill_targets.py export \\
        --spot banks/oss34/bank.json.gz:US:80 --spot banks/nextvp/bank.json.gz:US:89 \\
        --anchors anchors/part-1.jsonl.gz --repeat 4 --out distill.jsonl.gz

    # how far a checkpoint moved at the same spots
    ... tools/bank_distill_targets.py check --checkpoint <pt> --spot ... --spot ...

**Anchors.** Spot positions alone would pull the policy toward events with nothing holding the rest
of it in place. The anchors are the model's own distribution at every decision of its own games
(`tools/generate_policy_targets.py --us X --ussr X`), so the cross-entropy there is a pull toward
where the policy already is. The spot records are interleaved among the anchor games at random, so
the loader's shuffle buffer sees both throughout an epoch.

A spot record is one line, `{"save": <save json>, "search_pi": {"a": [EVENT], "v": [1]}}`, which
`WarmupDataset.stream_policy_transitions` reads as a single position.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

EVENT = 110   # the play-mode decision's "event" (ai/eval/rule_oracle.py EVENT)

Spot = Tuple[str, str, int]


def parse_spot(text: str) -> Spot:
    """`<bank.json.gz>:<side>:<card id>`."""
    path, side, card = text.rsplit(":", 2)
    if side not in ("US", "USSR"):
        raise argparse.ArgumentTypeError(f"side must be US or USSR, not {side!r}")
    return path, side, int(card)


def spot_records(spot: Spot) -> List[Dict[str, Any]]:
    path, side, card = spot
    with gzip.open(path, "rt", encoding="utf-8") as f:
        bank = json.load(f)
    recs = [r for r in bank["records"]
            if r["kind"] == "mode" and r["features"]["side"] == side and r["features"].get("card") == card
            and any(c["prefix"][0] == EVENT for c in r["candidates"])]
    if not recs:
        raise SystemExit(f"{path}: no play-mode records for {side} card {card} with the event legal")
    return recs


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export(a: argparse.Namespace) -> int:
    rng = random.Random(a.seed)
    spot_lines: List[str] = []
    counts: Dict[str, int] = {}
    for spot in a.spot:
        recs = spot_records(spot)
        counts[f"{os.path.basename(os.path.dirname(spot[0])) or spot[0]}:{spot[1]}:{spot[2]}"] = len(recs)
        for r in recs:
            line = json.dumps({"save": r["save"], "search_pi": {"a": [EVENT], "v": [1.0]}})
            spot_lines.extend([line] * a.repeat)
    anchor_lines: List[str] = []
    for path in a.anchors:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            anchor_lines.extend(line.rstrip("\n") for line in f if line.strip())
    # Spread the spot lines evenly at random among the anchor games.
    rng.shuffle(spot_lines)
    slots: List[List[str]] = [[] for _ in range(len(anchor_lines) + 1)]
    for line in spot_lines:
        slots[rng.randrange(len(slots))].append(line)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as out:
        for i, slot in enumerate(slots):
            for line in slot:
                out.write(line + "\n")
            if i < len(anchor_lines):
                out.write(anchor_lines[i] + "\n")
    meta = {"searcher": "bank_distill_targets (event at bank-confirmed spots + own-policy anchors)",
            "commit": _commit(), "spots": counts, "repeat": a.repeat,
            "spot_targets": len(spot_lines), "anchor_games": len(anchor_lines), "merged_influence": False}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return 0


def check(a: argparse.Namespace) -> int:
    import torch
    import ts_engine as ts
    from tools.lib.player_agent import NeuralAgent

    model = NeuralAgent.from_checkpoint(a.checkpoint, device="cpu").model.eval()
    rows = ["| spot | positions | event is the argmax | mean P(event) |", "|:---|---:|---:|---:|"]
    for spot in a.spot:
        recs = spot_records(spot)
        top = 0
        p_sum = 0.0
        for r in recs:
            st = ts.state_from_save_json(r["save"])
            obs = torch.from_numpy(np.asarray(ts.extract_observation(st, st.ctx().decision_player), np.float32))
            mask = torch.from_numpy(np.asarray(ts.Engine.get_flat_action_mask(st, False)))
            with torch.no_grad():
                logits, _, _ = model(obs[None], mask[None])
            p = torch.softmax(logits[0].float().masked_fill(mask <= 0, float("-inf")), -1)
            top += int(int(p.argmax()) == EVENT)
            p_sum += float(p[EVENT])
        name = f"{os.path.basename(os.path.dirname(spot[0])) or spot[0]} {spot[1]} card {spot[2]}"
        rows.append(f"| {name} | {len(recs)} | {100 * top / len(recs):.0f}% | {p_sum / len(recs):.3f} |")
    print(f"# {os.path.basename(a.checkpoint)} at the distillation spots (training positions)\n")
    print("\n".join(rows))
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the distillation dataset")
    e.add_argument("--spot", type=parse_spot, action="append", required=True,
                   help="<bank.json.gz>:<US|USSR>:<card id>, repeatable")
    e.add_argument("--anchors", action="append", default=[], help="own-policy target games (jsonl.gz), repeatable")
    e.add_argument("--repeat", type=int, default=4, help="copies of each spot position")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    c = sub.add_parser("check", help="event rate of a checkpoint at the spots")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--spot", type=parse_spot, action="append", required=True)
    a = ap.parse_args(argv)
    return export(a) if a.cmd == "export" else check(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
