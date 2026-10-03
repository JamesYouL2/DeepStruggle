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
import math
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _q(result: str) -> float:
    """A candidate's mean result for the mover, from its pairs ('0' loss, '1' draw, '2' win)."""
    return sum(int(ch) for ch in result) / (2.0 * len(result))


def _scores(result: str) -> List[float]:
    return [int(ch) / 2.0 for ch in result]


def contrast_target(rec: Dict[str, Any]) -> Optional[Dict[str, List[Any]]]:
    """The event-or-not decision from the playouts, the rest from the model.

    P(event) = Phi(gap / se): the probability, given this record's paired playouts, that the event
    beats the best non-event mode. `gap` is the mean of the per-pair differences (same dice and deal in
    both branches), `se` its standard error, so a position whose playouts cannot tell the two apart
    gets a target near 1/2 -- where the choice costs nothing -- rather than its noise. The non-event
    mass is split over the non-event modes in proportion to the model's own probabilities (each
    candidate's `prior`), so the target corrects whether to event and leaves the choice among Ops
    modes -- the board play the review found strong -- where the model had it. Records without the
    event among the candidates, or with no other mode, give None."""
    res: Dict[int, List[float]] = {}
    prior: Dict[int, float] = {}
    for c, r in zip(rec["candidates"], rec["results"]):
        a = int(c["prefix"][0])
        sc = _scores(r)
        # A two-step line (Wargames' event, then end the game) is the event too: keep its better line.
        if a not in res or sum(sc) > sum(res[a]):
            res[a] = sc
        prior[a] = max(prior.get(a, 0.0), float(c.get("prior", 0.0)))
    if EVENT not in res or len(res) < 2:
        return None
    others = [a for a in res if a != EVENT]
    best = max(others, key=lambda a: sum(res[a]))
    d = [e - o for e, o in zip(res[EVENT], res[best])]
    n = len(d)
    gap = sum(d) / n
    var = sum((x - gap) ** 2 for x in d) / max(1, n - 1)
    se = math.sqrt(var / n)
    p_event = (0.5 * (1.0 + math.erf(gap / (se * math.sqrt(2.0)))) if se > 0
               else (1.0 if gap > 0 else 0.0 if gap < 0 else 0.5))
    tot = sum(prior[a] for a in others)
    split = {a: (prior[a] / tot if tot > 0 else 1.0 / len(others)) for a in others}
    return {"a": [EVENT] + others, "v": [p_event] + [(1.0 - p_event) * split[a] for a in others]}


def _labelled_keys(paths: List[str]) -> Set[Tuple[str, int]]:
    keys: Set[Tuple[str, int]] = set()
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for r in json.load(f)["records"]:
                if r["kind"] == "mode" and r["features"].get("card") is not None:
                    keys.add((r["features"]["side"], int(r["features"]["card"])))
    return keys


def strip_labelled(game: Dict[str, Any], keys: Set[Tuple[str, int]]) -> int:
    """Drop the own-policy target at play-mode decisions on a labelled (side, card): there the
    playout label is the target, and the model's own choice would argue against it. Returns how many."""
    import ts_engine as ts

    st = ts.GameState()
    ts.Engine.init_game(st, game["seed"])
    dropped = 0
    for a in game["actions"]:
        ctx = st.ctx()
        if (a.get("search_pi") and ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
                and int(ctx.pending_op_card)):
            side = "US" if ctx.decision_player == ts.Player.US else "USSR"
            if (side, int(ctx.pending_op_card)) in keys:
                del a["search_pi"]
                dropped += 1
        mask = np.asarray(ts.get_flat_action_mask(st, False))
        if not (0 <= a["flat_action"] < mask.shape[0] and mask[a["flat_action"]]):
            break
        ts.Engine.step(st, ts.decode_flat_action(st, a["flat_action"]))
        while (not ts.Engine.is_terminal(st) and st.ctx().decision_player == ts.Player.NONE
               and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
    return dropped


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export(a: argparse.Namespace) -> int:
    rng = random.Random(a.seed)
    spot_lines: List[str] = []
    counts: Dict[str, int] = {}
    # Contrastive labels: every position of the labelled banks. --denoise labels each from a
    # cross-fitted regression over all of them (tools/lib/gap_labels.py); otherwise from its own playouts.
    all_recs: List[Dict[str, Any]] = []
    for path in a.labelled:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            all_recs += json.load(f)["records"]
    denoised: List[Optional[float]] = []
    if a.denoise and all_recs:
        from tools.lib.gap_labels import label as gap_label, report as gap_report
        from ai.eval.doctrine_census import cards

        denoised, fit = gap_label(all_recs, seed=a.seed)
        with open(a.out + ".labels.md", "w", encoding="utf-8") as f:
            f.write(gap_report(fit, {k: str(v["name"]) for k, v in cards().items()}) + "\n")
    for ri, r in enumerate(all_recs):
        if r["kind"] != "mode":
            continue
        t = contrast_target(r)
        if t is None:
            continue
        if denoised:
            pe = denoised[ri]
            if pe is None:
                continue
            rest = sum(t["v"][1:])
            t = {"a": t["a"], "v": [pe] + [(1.0 - pe) * v / rest if rest > 0 else (1.0 - pe) / (len(t["v"]) - 1)
                                          for v in t["v"][1:]]}
        key = f"{r['features']['side']}:{r['features'].get('card')}"
        counts[key] = counts.get(key, 0) + 1
        spot_lines.extend([json.dumps({"save": r["save"], "search_pi": t})] * a.repeat)
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
    stripped = 0
    if a.labelled:
        keys = _labelled_keys(a.labelled)
        games = [json.loads(line) for line in anchor_lines]
        stripped = sum(strip_labelled(g, keys) for g in games)
        anchor_lines = [json.dumps(g) for g in games]
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
            "spot_targets": len(spot_lines), "anchor_games": len(anchor_lines),
            "labelled_banks": a.labelled, "label": ("cross-fitted regression (tools/lib/gap_labels.py)" if a.denoise
                      else "P(event) = Phi(paired gap / se)"), "anchor_targets_dropped_at_labelled_cards": stripped,
            "merged_influence": False}
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
    e.add_argument("--spot", type=parse_spot, action="append", default=[],
                   help="<bank.json.gz>:<US|USSR>:<card id>, repeatable")
    e.add_argument("--labelled", action="append", default=[],
                   help="a bank whose every play-mode record becomes a contrastive target (repeatable)")
    e.add_argument("--denoise", action="store_true",
                   help="label --labelled positions from a cross-fitted regression of the playout gap on the "
                        "position's features (tools/lib/gap_labels.py) instead of each one's own playouts")
    e.add_argument("--anchors", action="append", default=[], help="own-policy target games (jsonl.gz), repeatable")
    e.add_argument("--repeat", type=int, default=4, help="copies of each spot or labelled position")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    c = sub.add_parser("check", help="event rate of a checkpoint at the spots")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--spot", type=parse_spot, action="append", required=True)
    a = ap.parse_args(argv)
    return export(a) if a.cmd == "export" else check(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
