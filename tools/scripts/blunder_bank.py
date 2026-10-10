#!/usr/bin/env python3
"""The blunder bank: positions where a strong network's own move costs a large share of a game,
confirmed on fresh playouts, and a score of how often any player repeats them.

A census row's cost (`bank_playouts.py validate`) is chosen and measured on the same continuations,
and at 32 pairs most "confirmed" strategic regret is chance: the census's random control clears the
same bar almost as often as its candidates (`research/log/E7_blunder_census.md`). The bank keeps
only what survives a second, independent measurement:

* `select` -- the shortlist: every census row whose best alternative leads the network's move by
  `--min-gap` of a game at `--min-z` standard errors. The leading alternative is fixed here, before
  the confirmation, so the confirmed cost is not a maximum over noisy alternatives;
* `confirm` -- this part's share of the shortlist (`--part k/N`, for CI runners): every measured move
  paired out again by a judge network's greedy play, `--pairs` pairs from a seed the census never
  used. Run it once per judge;
* `build` -- the rows whose confirmed cost clears `--min-gap` and `--min-z` under every judge,
  written as the bank (one JSON row per position, with its workbench `pos=` token) and a
  readable table;
* `score` -- each player's move on every bank position (a checkpoint's greedy move, or any
  `load_agent` searcher): the blunder repeated, the confirmed better move found, or another move,
  which `--pairs` pays out against both with a judge network.

A bank row's cost is a paired difference in the mover's score (1 win, 0.5 draw, 0 loss), so 0.2 is
twenty points of win probability.

    export PYTHONPATH=.:build/release
    python tools/scripts/blunder_bank.py select --validation validation.jsonl.gz \\
        --min-gap 0.1 --min-z 2 --out shortlist.jsonl.gz
    python tools/scripts/blunder_bank.py confirm --input shortlist.jsonl.gz --model soup.pt \\
        --pairs 512 --seed 7 --part 1/20 --out confirm-soup-1.jsonl.gz
    python tools/scripts/blunder_bank.py build --confirm soup=confirm-soup-*.jsonl.gz \\
        --confirm r32=confirm-r32-*.jsonl.gz --min-gap 0.1 --min-z 3 \\
        --out ai/eval/banks/blunders_E7.jsonl --md blunders_E7.md
    python tools/scripts/blunder_bank.py score --bank ai/eval/banks/blunders_E7.jsonl \\
        --player soup=soup.pt --player g256=gumbel:soup.pt:256:8 --judge soup.pt --pairs 128 --out score.md
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.paired_playouts import compare, paired_diff
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import load_policy, require_e4_view, rules_json, state_from_token
from tools.lib.player_agent import BatchSelector, Reseedable, load_agent, resolve_device
from tools.scripts.bank_playouts import _sha256, position_seed
from tools.scripts.blunder_census import mover_of
from tools.scripts.disagreement_bank import _name

SCHEMA = 1
WORKBENCH = "https://mihaild.github.io/DeepStruggle"
#: positions per paired-playout batch in `confirm` and `score`
BATCH = 8
CARD_NAMES = {int(c["id"]): str(c["name"]) for c in rules_json("cards.json")}


def _read(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def leading_alternative(row: Dict[str, Any]) -> Optional[Tuple[int, float, float]]:
    """A census validation row's best alternative to the network's move: (action, lead, SE), or
    None when nothing else was measured."""
    diffs = row.get("diff_vs_greedy") or {}
    if not diffs:
        return None
    a, (m, se) = max(diffs.items(), key=lambda kv: (kv[1][0], -int(kv[0])))
    return int(a), float(m), float(se)


def shortlisted(row: Dict[str, Any], min_gap: float, min_z: float) -> bool:
    lead = leading_alternative(row)
    if lead is None:
        return False
    _, m, se = lead
    return m >= min_gap and se > 0 and m / se >= min_z


def select(validation: Sequence[str], min_gap: float, min_z: float, out: str) -> int:
    """The census rows to confirm, each with its network move and the alternative fixed now."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for r in _read(validation):
        if r["id"] in seen or not shortlisted(r, min_gap, min_z):
            continue
        seen.add(r["id"])
        lead = leading_alternative(r)
        assert lead is not None
        better, m, se = lead
        network = int(r["moves"]["greedy"])
        rows.append({
            "schema": SCHEMA, "id": r["id"], "pos": r["pos"],
            "source": {"kind": "blunder-census", "model": r["model"], "model_sha256": r["model_sha256"],
                       "game": r["game"], "decision": r["decision"], "control": bool(r.get("control"))},
            "side": r["side"], "turn": r["turn"], "ar": r["ar"], "phase": r["phase"],
            "decision_type": r["decision_type"], "defcon": r["defcon"], "vp": r["vp"], "card": r["card"],
            "network": network, "better": better,
            "moves": sorted({int(a) for a in r["moves"].values()}),
            "names": r["names"],
            "first": {"cost": round(m, 4), "se": round(se, 4), "pairs": r["pairs"], "verdict": r["verdict"],
                      "search_choice": r["search"]["choice"]},
        })
    rows.sort(key=lambda x: -x["first"]["cost"])
    _write(out, rows)
    print(f"{len(rows)} rows shortlisted at {min_gap} points and z >= {min_z}", file=sys.stderr)
    return 0


def greedy_fn(model: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    logits_fn, _ = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)
    return act


def confirm(inputs: Sequence[str], model: str, pairs: int, seed: int, part: str, out: str) -> int:
    """Every measured move of this part's rows, paired out by `model`'s greedy play. Each row's
    pairs are seeded from its id and `--seed` alone, so a row's result does not depend on its part."""
    k, n = (int(x) for x in part.split("/"))
    rows = [r for i, r in enumerate(sorted(_read(inputs), key=lambda r: str(r["id"]))) if i % n == k - 1]
    act = greedy_fn(model)
    sha = _sha256(model)
    res: List[Dict[str, Any]] = []
    t0 = time.time()
    for lo in range(0, len(rows), BATCH):
        chunk = rows[lo:lo + BATCH]
        positions = [(state_from_token(r["pos"]), [int(a) for a in r["moves"]]) for r in chunk]
        scores = compare(positions, act, pairs, [position_seed(str(r["id"]), seed) for r in chunk])
        for r, sc in zip(chunk, scores):
            net = int(r["network"])
            diffs = {a: paired_diff(sc[a], sc[net]) for a in sc if a != net}
            m, se = diffs[int(r["better"])]
            res.append(dict(r, judge={"model": os.path.basename(model), "model_sha256": sha},
                            pairs=pairs, seed=seed,
                            score={str(a): round(sum(v) / len(v), 4) for a, v in sc.items()},
                            diff_vs_network={str(a): [round(x, 4), round(y, 4)] for a, (x, y) in diffs.items()},
                            cost=[round(m, 4), round(se, 4)]))
        print(f"  {len(res)}/{len(rows)} rows, {time.time() - t0:.0f}s", file=sys.stderr)
    _write(out, res)
    json.dump({"part": part, "rows": len(res), "seconds": round(time.time() - t0, 1), "pairs": pairs,
               "seed": seed, "model": os.path.basename(model), "model_sha256": sha},
              open(out + ".meta.json", "w"))
    return 0


def confirmed(rows: Dict[str, Dict[str, Dict[str, Any]]], min_gap: float, min_z: float) -> List[str]:
    """The ids confirmed under every judge: cost at least `min_gap` and `min_z` standard errors."""
    judges = list(rows)
    ids = set.intersection(*(set(v) for v in rows.values())) if rows else set()
    keep = []
    for rid in ids:
        ok = True
        for j in judges:
            m, se = rows[j][rid]["cost"]
            ok = ok and m >= min_gap and se > 0 and m / se >= min_z
        if ok:
            keep.append(rid)
    return sorted(keep, key=lambda i: -min(rows[j][i]["cost"][0] for j in judges))


def bank_row(rid: str, by_judge: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    first = next(iter(by_judge.values()))
    st = state_from_token(first["pos"])
    net, better = int(first["network"]), int(first["better"])
    return {
        "schema": SCHEMA, "id": rid, "pos": first["pos"],
        "side": first["side"], "turn": first["turn"], "ar": first["ar"], "phase": first["phase"],
        "decision_type": first["decision_type"], "defcon": first["defcon"], "vp": first["vp"],
        "card": first["card"], "card_name": CARD_NAMES.get(int(first["card"]), ""),
        "network": {"action": net, "name": _name(st, net)},
        "better": {"action": better, "name": _name(st, better)},
        "cost": {j: {"cost": r["cost"][0], "se": r["cost"][1], "pairs": r["pairs"], "seed": r["seed"],
                     "model": r["judge"]["model"], "model_sha256": r["judge"]["model_sha256"]}
                 for j, r in by_judge.items()},
        "score": {j: r["score"] for j, r in by_judge.items()},
        "moves": first["moves"], "names": {str(a): _name(st, int(a)) for a in first["moves"]},
        "source": first["source"], "first": first["first"],
    }


def _link(pos: str) -> str:
    return f"{WORKBENCH}/?pos={pos}&model=off"


def render(bank: Sequence[Dict[str, Any]], judges: Sequence[str], header: str) -> str:
    out = [header, "",
           "| # | side | turn/AR | decision | network's move | better move | "
           + " | ".join(f"cost ({j})" for j in judges) + " | position |",
           "|---:|:---|:---|:---|:---|:---|" + "---:|" * len(judges) + ":---|"]
    for i, r in enumerate(bank, 1):
        costs = " | ".join(f"{100 * r['cost'][j]['cost']:.0f} ± {100 * r['cost'][j]['se']:.0f}" for j in judges)
        card = f" ({r['card_name']})" if r.get("card_name") else ""
        out.append(f"| {i} | {r['side']} | T{r['turn']} AR{r['ar']} | {r['decision_type']}{card} | "
                   f"{r['network']['name']} | {r['better']['name']} | {costs} | [open]({_link(r['pos'])}) |")
    return "\n".join(out)


def build(sets: Sequence[str], min_gap: float, min_z: float, out: str, md: Optional[str]) -> int:
    """`sets` are NAME=GLOB, one per judge."""
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in sets:
        name, _, pattern = spec.partition("=")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"no files match {pattern}")
        rows[name] = {r["id"]: r for r in _read(paths)}
    judges = list(rows)
    keep = confirmed(rows, min_gap, min_z)
    bank = [bank_row(rid, {j: rows[j][rid] for j in judges}) for rid in keep]
    _write(out, bank)
    shortlist = len(set.intersection(*(set(v) for v in rows.values())))
    print(f"{len(bank)} of {shortlist} shortlisted rows confirmed under {', '.join(judges)}", file=sys.stderr)
    if md:
        header = (f"# The blunder bank: {len(bank)} positions\n\nConfirmed at {100 * min_gap:.0f}+ points of "
                  f"a game and z >= {min_z:g} under every judge ({', '.join(judges)}), out of {shortlist} "
                  f"shortlisted. Cost in points of the mover's score (paired, ± 1 SE).")
        open(md, "w").write(render(bank, judges, header) + "\n")
    return 0


class Picker:
    """One player's move on a position: a checkpoint's greedy move, or a searcher reseeded per
    position so its pick does not depend on the order the bank is read in."""

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.logits: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
        self.features = 0
        self.agent: Optional[BatchSelector] = None
        if spec.endswith((".pt", ".onnx")) and ":" not in spec:
            self.logits, self.features = load_policy(spec)
        else:
            agent = load_agent(spec, device=resolve_device("cpu"))
            require_e4_view(spec, agent)
            if not isinstance(agent, BatchSelector):
                raise SystemExit(f"{spec}: not a batch searcher or a checkpoint")
            self.agent = agent

    def pick(self, st: ts.GameState, seed: int) -> int:
        if self.logits is not None:
            obs = np.asarray(ts.extract_observation_features(st, mover_of(st), self.features), dtype=np.float32)
            mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
            return int(np.asarray(self.logits(obs[None], mask[None]))[0].argmax())
        assert self.agent is not None
        if isinstance(self.agent, Reseedable):
            self.agent.reseed(seed % (1 << 31))
        return int(self.agent.select_actions_batch([st])[0])


def score(bank_path: str, players: Sequence[str], judge: Optional[str], pairs: int, seed: int,
          out: str) -> int:
    bank = list(_read([bank_path]))
    act = greedy_fn(judge) if judge and pairs > 0 else None
    judge_name = os.path.basename(judge) if judge else ""
    result: Dict[str, Any] = {"bank": bank_path, "rows": len(bank), "judge": judge, "pairs": pairs, "players": {}}
    lines = [f"# The blunder bank, scored: {len(bank)} positions", "",
             "Repeated: the player plays the network's confirmed blunder. Better: the confirmed better move. "
             "Other: anything else" + (f", paid out by {judge_name} over {pairs} pairs against "
                                       "the blunder" if act else "") + ". Points: the blunders' confirmed "
             "cost summed over the rows the player repeats (the first judge's figure).", "",
             "| player | repeated | better | other | points repeated |" + (" other vs blunder |" if act else ""),
             "|:---|---:|---:|---:|---:|" + ("---:|" if act else "")]
    for spec in players:
        name, s = spec, spec
        if "=" in spec.split(":")[0]:
            name, _, s = spec.partition("=")
        picker = Picker(s)
        per: List[Dict[str, Any]] = []
        t0 = time.time()
        for r in bank:
            st = state_from_token(r["pos"])
            a = picker.pick(st, position_seed(str(r["id"]), seed))
            kind = ("repeated" if a == r["network"]["action"] else
                    "better" if a == r["better"]["action"] else "other")
            per.append({"id": r["id"], "pick": a, "name": _name(st, a), "kind": kind})
        others = [(p, r) for p, r in zip(per, bank) if p["kind"] == "other"]
        if act is not None and others:
            for lo in range(0, len(others), BATCH):
                chunk = others[lo:lo + BATCH]
                positions = [(state_from_token(r["pos"]), sorted({p["pick"], r["network"]["action"]}))
                             for p, r in chunk]
                sc = compare(positions, act, pairs, [position_seed(str(r["id"]), seed + 1) for _, r in chunk])
                for (p, r), s_ in zip(chunk, sc):
                    m, se = paired_diff(s_[p["pick"]], s_[r["network"]["action"]])
                    p["vs_blunder"] = [round(m, 4), round(se, 4)]
        first = (lambda r: next(iter(r["cost"].values()))["cost"])
        rep = [r for p, r in zip(per, bank) if p["kind"] == "repeated"]
        counts = {k: sum(p["kind"] == k for p in per) for k in ("repeated", "better", "other")}
        pts = sum(first(r) for r in rep)
        row = (f"| {name} | {counts['repeated']} | {counts['better']} | {counts['other']} | {pts:.2f} |")
        if act is not None:
            vs = [p["vs_blunder"][0] for p in per if "vs_blunder" in p]
            row += f" {np.mean(vs):+.2f} ({len(vs)}) |" if vs else " — |"
        lines.append(row)
        result["players"][name] = {"spec": s, "counts": counts, "points_repeated": round(pts, 4),
                                   "seconds": round(time.time() - t0, 1), "rows": per}
        print(f"{name}: {counts} in {time.time() - t0:.0f}s", file=sys.stderr)
    open(out, "w").write("\n".join(lines) + "\n")
    json.dump(result, open(os.path.splitext(out)[0] + ".json", "w"), indent=1)
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="the shortlist from census validation rows")
    s.add_argument("--validation", nargs="+", required=True)
    s.add_argument("--min-gap", type=float, default=0.1)
    s.add_argument("--min-z", type=float, default=2.0)
    s.add_argument("--out", required=True)
    c = sub.add_parser("confirm", help="fresh paired playouts of a shortlist, one judge")
    c.add_argument("--input", nargs="+", required=True)
    c.add_argument("--model", required=True, help="the judge: its greedy play continues every pair")
    c.add_argument("--pairs", type=int, default=512)
    c.add_argument("--seed", type=int, default=7, help="a seed the census did not use (it used 0 and 1)")
    c.add_argument("--part", default="1/1")
    c.add_argument("--out", required=True)
    b = sub.add_parser("build", help="the bank: rows confirmed under every judge")
    b.add_argument("--confirm", nargs="+", required=True, metavar="NAME=GLOB")
    b.add_argument("--min-gap", type=float, default=0.1)
    b.add_argument("--min-z", type=float, default=3.0)
    b.add_argument("--out", required=True)
    b.add_argument("--md", default=None)
    g = sub.add_parser("score", help="players on the bank")
    g.add_argument("--bank", required=True)
    g.add_argument("--player", nargs="+", required=True, metavar="NAME=SPEC")
    g.add_argument("--judge", default=None, help="a checkpoint whose greedy play pays out 'other' picks")
    g.add_argument("--pairs", type=int, default=0)
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        return select(a.validation, a.min_gap, a.min_z, a.out)
    if a.cmd == "confirm":
        return confirm(a.input, a.model, a.pairs, a.seed, a.part, a.out)
    if a.cmd == "build":
        return build(a.confirm, a.min_gap, a.min_z, a.out, a.md)
    return score(a.bank, a.player, a.judge, a.pairs, a.seed, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
