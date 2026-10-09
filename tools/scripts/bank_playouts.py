#!/usr/bin/env python3
"""Paired playouts for disagreement-bank positions: each of the human's, the network's and the
search's moves played out by the model (`ai/eval/paired_playouts.py`).

Four steps:

* `select` -- the positions to play, from `bank_clarity.py`'s output: every position whose value lead
  is at least `--min-gap` win-probability points (or else the `--top` largest margins in SEs),
  written with their positions and move indices as one input file;
* `run` -- this part's share of an input file (`--part k/N`, for CI runners): every distinct move
  of every position, `--pairs` pairs each, the network playing both sides greedily;
* `validate` -- stage 2 of the catastrophic-blunder census (`tools/scripts/blunder_census.py`):
  each screened position searched with Gumbel k=`--search-k` at `--search-sims` simulations over
  `--search-seeds` independent runs, and the moves that puts in play paired out by the model's
  play (or `--continue-with SIMS:K`, a Gumbel root of the model playing both sides), with exact
  terminal mistakes (the engine's labels) classified apart from the estimated strategic regret
  and the verdict recorded (`verdict_of`). `--escalate-from` a previous output takes only its
  most serious unresolved rows, for a rerun at a larger budget;
* `pool` -- the parts merged into one file; with `--expect N` the parts that did not arrive (a
  runner that failed) are named in `<out>.parts.json` rather than silently missing.

Each position's pairs are seeded from its row id and `--seed` alone, so a position plays the same
redeals and dice whichever part it falls in: every part of one run must share one `--seed`. In
`validate` the searches and the continuation searcher are reseeded per position from the same
material, so a row is position-local too.

A row of `run`'s output: each move's mean score for the mover (1 win, 0.5 draw, 0 loss), and the
paired differences between the three moves with their standard errors.

    PYTHONPATH=.:build/release python tools/scripts/bank_playouts.py select --clarity clarity*.jsonl.gz \\
        --bank part*.jsonl.gz --top 1000 --out playout_input.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/bank_playouts.py run --input playout_input.jsonl.gz \\
        --onnx newest.onnx --pairs 32 --part 1/20 --out playouts-1.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/bank_playouts.py validate --input candidates.jsonl.gz \\
        control.jsonl.gz --model newest.pt --pairs 64 --part 1/20 --out validation-1.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/bank_playouts.py pool --parts playouts-*.jsonl.gz \\
        --expect 20 --out playouts.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ai.eval.paired_playouts import compare, paired_diff
from ai.eval.safety import classify_legal_actions
from tools.lib.corpus_driver import load_policy, require_e4_view, state_from_token
from tools.lib.player_agent import OnnxAgent, load_agent, resolve_device
from tools.scripts.disagreement_bank import _name, row_id

WHO = ("human", "network", "search")
#: A validation row's verdicts. The two `exact-` ones are the engine's own labels, not an
#: estimate; the rest are what the paired continuations could settle.
VERDICTS = ("exact-missed-win", "exact-forced-loss", "confirmed-regret", "refuted", "unresolved")


def _read(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                yield json.loads(line)


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def value_gap(c: Dict[str, Any]) -> Optional[float]:
    """The value-best move's lead over the next distinct move, in win-probability points (values run
    from -1 to +1, so half the difference). A lead matters only if it is large: in a game that is
    won or lost whatever happens, every move is worth about the same."""
    vals = sorted({(v[0], v[1]) for v in c["q"].values()}, key=lambda v: -v[0])
    return (vals[0][0] - vals[1][0]) / 2 * 100 if len(vals) > 1 else None


def select(clarity: Sequence[str], banks: Sequence[str], top: int, out: str, min_gap: float = 0.0) -> int:
    if min_gap > 0:
        cl = [c for c in _read(clarity) if (value_gap(c) or 0.0) >= min_gap]
        cl.sort(key=lambda c: -(value_gap(c) or 0.0))
        cl = cl[:top]
    else:
        cl = sorted((c for c in _read(clarity) if c.get("z") is not None), key=lambda c: -c["z"])[:top]
    want = {c["id"]: c for c in cl}
    pos: Dict[str, str] = {}
    for r in _read(banks):
        rid = row_id(r)
        if rid in want and rid not in pos:
            pos[rid] = r["pos"]
    rows = [{"id": c["id"], "pos": pos[c["id"]], "a": c["a"]} for c in cl if c["id"] in pos]
    _write(out, rows)
    print(f"{len(rows)} positions selected", file=sys.stderr)
    return 0


def position_seed(rid: str, seed: int) -> int:
    """A position's seed: its row id (16 hex digits) mixed with the run's seed, kept below 2**63."""
    return (int(rid, 16) ^ (seed * 0x9E3779B97F4A7C15)) & 0x7FFF_FFFF_FFFF_FFFF


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def searched_continuation(model: str, sims_k: str) -> Tuple[Any, str]:
    """`--continue-with SIMS:K`: a Gumbel root of the model playing both sides of the playouts,
    and the spec as a row records it. The caller reseeds the agent per position, so one
    position's play does not depend on which positions preceded it."""
    sims, k = (int(x) for x in sims_k.split(":"))
    spec = f"gumbel:{model}:{sims}:{k}"
    agent = load_agent(spec, device=resolve_device("cuda"))
    require_e4_view(spec, agent)
    return agent, f"gumbel:{sims}:{k}"


def run(inp: Sequence[str], onnx: str, pairs: int, part: str, seed: int, out: str) -> int:
    k, n = (int(x) for x in part.split("/"))
    rows = [r for i, r in enumerate(_read(inp)) if i % n == k - 1]
    agent = OnnxAgent(onnx)
    require_e4_view(onnx, agent)

    def act(obs: Any, masks: Any) -> Any:
        return agent.act_batch(obs, masks, 0.0, True)

    t0 = time.time()
    positions = [(state_from_token(r["pos"]), sorted(set(int(v) for v in r["a"].values()))) for r in rows]
    scores = compare(positions, act, pairs, [position_seed(r["id"], seed) for r in rows])
    res: List[Dict[str, Any]] = []
    for r, sc in zip(rows, scores):
        a = {w: int(v) for w, v in r["a"].items()}
        mean = {w: round(sum(sc[a[w]]) / len(sc[a[w]]), 4) for w in WHO}
        diffs = {}
        for x, y in (("human", "network"), ("human", "search"), ("network", "search")):
            if a[x] != a[y]:
                m, se = paired_diff(sc[a[x]], sc[a[y]])
                diffs[f"{x}-{y}"] = [round(m, 4), round(se, 4)]
        best = max(set(a.values()), key=lambda m: sum(sc[m]))
        res.append({"id": r["id"], "pairs": pairs, "score": mean, "diff": diffs,
                    "best": [w for w in WHO if a[w] == best]})
    _write(out, res)
    print(f"{len(res)} positions, {pairs} pairs, in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def verdict_of(greedy: int, diff: Tuple[float, float], labels: Dict[int, str],
               min_z: float, min_gap: float) -> str:
    """One row's verdict. Exact terminal mistakes first -- the engine's labels, not an estimate:
    a forced win on the board that greedy did not take, or a forced loss greedy took while some
    legal move avoids it. The rest is the paired continuations' estimated strategic regret over
    the best alternative's lead on greedy: confirmed at `min_gap` score points and `min_z`
    standard errors, refuted when the continuations rank greedy `min_z` standard errors ahead,
    and unresolved where they cannot tell."""
    if any(l == "win" for l in labels.values()) and labels.get(greedy) != "win":
        return "exact-missed-win"
    if labels.get(greedy) == "loss" and any(l != "loss" for l in labels.values()):
        return "exact-forced-loss"
    m, se = diff
    if se and se > 0:
        if m >= min_gap and m / se >= min_z:
            return "confirmed-regret"
        if m / se <= -min_z:
            return "refuted"
    return "unresolved"


def escalation_shortlist(rows: Sequence[Dict[str, Any]], prev: Sequence[Dict[str, Any]],
                         cap: int) -> List[Dict[str, Any]]:
    """The rows a previous validation left unresolved, most serious first (largest estimated
    regret on greedy), at most `cap`. An exact label or a settled comparison resolves a row and
    never escalates."""
    regret: Dict[str, float] = {}
    for p in prev:
        if p.get("verdict") == "unresolved":
            d = p.get("diff_vs_greedy", {}).get(str(p.get("best")), [0.0, 0.0])
            regret[str(p["id"])] = abs(float(d[0] or 0.0))
    want = sorted(regret, key=lambda i: -regret[i])[:cap or None]
    keep = set(want)
    return [r for r in rows if r["id"] in keep]


def validate(inputs: Sequence[str], model: str, out: str, part: str, pairs: int, seed: int,
             search_sims: int, search_k: int, search_seeds: int,
             continue_with: Optional[str], min_z: float, min_gap: float,
             escalate_from: Optional[str], escalate_cap: int) -> int:
    """Stage 2 of the blunder census over the rows `tools/scripts/blunder_census.py screen`
    saved (`--input` takes the candidates and the control sample alike -- the control's verdicts
    are what measures the screening's misses). Every seed of a row comes from its id and `--seed`,
    so the row's verdict does not depend on the part it fell in."""
    k, n = (int(x) for x in part.split("/"))
    rows = sorted(_read(inputs), key=lambda r: str(r["id"]))
    if escalate_from:
        rows = escalation_shortlist(rows, list(_read([escalate_from])), escalate_cap)
    rows = [r for i, r in enumerate(rows) if i % n == k - 1]
    searcher, search_label = searched_continuation(model, f"{search_sims}:{search_k}")
    cont: Optional[Any] = None
    cont_label = "greedy"
    if continue_with:
        cont, cont_label = searched_continuation(model, continue_with)
    logits_fn, _ = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    res: List[Dict[str, Any]] = []
    t0 = time.time()
    for ri, r in enumerate(rows):
        st = state_from_token(r["pos"])
        runs: List[Dict[str, Any]] = []
        for j in range(search_seeds):
            # the searcher's reseed takes a 32-bit seed; the row records what it was given
            s_j = position_seed(str(r["id"]), seed + 104_729 * (j + 1)) % (1 << 31)
            searcher.reseed(s_j)
            runs.append({"seed": s_j, "pick": int(searcher.select_actions_batch([st])[0])})
        picks = [x["pick"] for x in runs]
        choice = max(set(picks), key=lambda a: (picks.count(a), -picks.index(a)))
        moves: Dict[str, int] = {"greedy": int(r["greedy"]), "g32": int(r["g32"]),
                                 "search": choice}
        safe = r.get("safety", {}).get("safe")
        if safe is not None and int(safe) not in moves.values():
            moves["safe"] = int(safe)
        extra = 0
        for a in picks:
            if int(a) not in moves.values():
                extra += 1
                moves[f"search-{extra}"] = int(a)
        cont_seed = position_seed(str(r["id"]), seed + 2_000_003) % (1 << 31) if cont is not None else 0
        if cont is not None:
            cont.reseed(cont_seed)
        sc = compare([(st, sorted(set(moves.values())))], act, pairs,
                     [position_seed(str(r["id"]), seed)],
                     select=cont.select_actions_batch if cont is not None else None)[0]
        greedy = moves["greedy"]
        alternatives = [a for a in sc if a != greedy]
        diffs = {a: paired_diff(sc[a], sc[greedy]) for a in alternatives}
        if alternatives:
            best = max(alternatives, key=lambda a: sum(sc[a]))
            m, se = diffs[best]
        else:
            best, (m, se) = greedy, (0.0, 0.0)
        labels = classify_legal_actions(st)
        res.append({
            "schema": 1, "id": r["id"], "control": bool(r.get("control", False)),
            "model": os.path.basename(model), "model_sha256": _sha256(model),
            "game": r["game"], "decision": r["decision"], "side": r["side"],
            "turn": r["turn"], "ar": r["ar"], "phase": r["phase"],
            "decision_type": r["decision_type"], "defcon": r["defcon"], "vp": r["vp"],
            "card": r["card"], "moves": moves,
            "names": {str(a): _name(st, a) for a in sorted(set(moves.values()))},
            "search": {"spec": search_label, "runs": runs, "choice": choice},
            "pairs": pairs, "cont": cont_label,
            "seeds": {"continuation": position_seed(str(r["id"]), seed),
                      "continuation_play": cont_seed},
            "score": {str(a): round(sum(v) / len(v), 4) for a, v in sc.items()},
            "diff_vs_greedy": {str(a): [round(mm, 4), round(sse, 4)]
                               for a, (mm, sse) in diffs.items()},
            "best": best,
            "exact": {"labels": {str(a): l for a, l in labels.items()},
                      "greedy": labels.get(greedy),
                      "missed_win": any(l == "win" for l in labels.values())
                      and labels.get(greedy) != "win",
                      "forced_loss": labels.get(greedy) == "loss"
                      and any(l != "loss" for l in labels.values())},
            "verdict": verdict_of(greedy, (m, se), labels, min_z, min_gap),
            "pos": r["pos"],
        })
        print(f"[{ri + 1}/{len(rows)}] {r['id']}: {res[-1]['verdict']} in {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
    _write(out, res)
    json.dump({"schema": 1, "stage": "validate",
               "budget": "escalation" if escalate_from else "validation",
               "model": os.path.basename(model), "model_sha256": _sha256(model),
               "search_sims": search_sims, "search_k": search_k, "search_seeds": search_seeds,
               "pairs": pairs, "cont": cont_label, "seed": seed, "part": part,
               "min_z": min_z, "min_gap": min_gap, "rows": len(res),
               "wall_s": round(time.time() - t0, 1)}, open(out + ".meta.json", "w"), indent=1)
    print(f"{len(res)} rows validated, {pairs} pairs, in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def pool(parts: Sequence[str], out: str, expect: int = 0) -> int:
    """Merge the parts (`playouts-<k>.jsonl.gz`); with `expect`, name the parts that are missing in
    `<out>.parts.json`, so a pool over a run with a failed runner says which share it lacks."""
    _write(out, list(_read(parts)))
    found = sorted(int(m.group(1)) for m in (re.search(r"playouts-(\d+)\.jsonl\.gz$", os.path.basename(p))
                                             for p in parts) if m)
    missing = [k for k in range(1, expect + 1) if k not in found]
    with open(out + ".parts.json", "w") as f:
        json.dump({"expected": expect, "found": found, "missing": missing}, f)
    if missing:
        print(f"parts missing: {missing} -- their positions are not in {out}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--clarity", nargs="+", required=True)
    s.add_argument("--bank", nargs="+", required=True)
    s.add_argument("--top", type=int, default=1000)
    s.add_argument("--min-gap", type=float, default=0.0,
                   help="take every position whose value lead is at least this many win-probability points "
                        "(largest first, up to --top) instead of the largest margins in SEs")
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", nargs="+", required=True)
    r.add_argument("--onnx", required=True)
    r.add_argument("--pairs", type=int, default=32)
    r.add_argument("--part", default="1/1")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    v = sub.add_parser("validate", help="stage 2 of the blunder census (tools/scripts/blunder_census.py)")
    v.add_argument("--input", nargs="+", required=True,
                   help="screen rows: candidates.jsonl.gz and control.jsonl.gz")
    v.add_argument("--model", required=True, help="the checkpoint (.pt) whose search validates the rows")
    v.add_argument("--pairs", type=int, default=32)
    v.add_argument("--part", default="1/1")
    v.add_argument("--seed", type=int, default=0,
                   help="the run's seed; give the escalation a different one from the 256 run")
    v.add_argument("--search-sims", type=int, default=256)
    v.add_argument("--search-k", type=int, default=8)
    v.add_argument("--search-seeds", type=int, default=3,
                   help="independent search runs per position; their agreement is reported")
    v.add_argument("--continue-with", default=None, metavar="SIMS:K",
                   help="play the continuations with a Gumbel root of the model (default: its greedy play)")
    v.add_argument("--min-z", type=float, default=2.0)
    v.add_argument("--min-gap", type=float, default=0.05,
                   help="score points a move must lead greedy by to count as a confirmed blunder")
    v.add_argument("--escalate-from", default=None,
                   help="a previous validation output: take only its most serious unresolved rows")
    v.add_argument("--escalate-cap", type=int, default=0, help="how many of them (0: all)")
    v.add_argument("--out", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expect", type=int, default=0, help="the number of parts the run was split into")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        return select(a.clarity, a.bank, a.top, a.out, a.min_gap)
    if a.cmd == "run":
        return run(a.input, a.onnx, a.pairs, a.part, a.seed, a.out)
    if a.cmd == "validate":
        return validate(a.input, a.model, a.out, a.part, a.pairs, a.seed,
                        a.search_sims, a.search_k, a.search_seeds, a.continue_with,
                        a.min_z, a.min_gap, a.escalate_from, a.escalate_cap)
    return pool(a.parts, a.out, a.expect)


if __name__ == "__main__":
    raise SystemExit(main())
