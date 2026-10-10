#!/usr/bin/env python3
"""Did a search-trained network learn its teacher's corrections? Positions where a teacher's search
overrules the teacher's own network, and what a student network plays there.

    select  a weighted sample of the teacher's self-play positions from a distillation round's
            targets (tools/generate_search_targets.py --target gchoice: each record carries the
            network's argmax and the Gumbel root's pick), stratified into "the search departs" and
            "the search agrees", written as a bank for tools/search_reliability.py (`raw` is the
            teacher network's move; `weight` is the inverse inclusion probability).
    report  from that bank, tools/search_reliability.py's search rows (the student raw and searched,
            the teacher searched, both rollout roots) and its paired playouts (one judge per file
            set): where the teacher's search departs, does the student adopt the correction, keep
            the teacher's move, or do something else -- and is what it adopts good; where the search
            agrees, how often and how badly does the student change the move; where the rollout root
            and Gumbel disagree, whom does the student follow; and each searcher's offline gain.

The sample is of the teacher's own positions, population-weighted, so rates are per decision of the
teacher's play -- unlike the search bank, which is enriched for disagreements and is a floor, not a
rate. Gains are paired playout differences against the teacher network's move, in win probability.

    tools/search_transfer.py select --targets targets-*.jsonl.gz --budget 256 --departures 3000 \\
        --agreements 1500 --out transfer_bank.jsonl.gz
    tools/search_transfer.py report --bank transfer_bank.jsonl.gz --search search-*.jsonl.gz \\
        --playouts playouts-*.jsonl.gz --out transfer.md
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def _read(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _keep(key: str, seed: int, p: float) -> bool:
    """A record's inclusion, a fixed function of its key and the seed."""
    h = int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "little")
    return h / float(1 << 64) < p


def select(targets: Sequence[str], budget: str, departures: int, agreements: int, seed: int,
           out: str) -> int:
    import ts_engine as ts
    from tools.lib.corpus_driver import position_token

    counts = {"dep": 0, "agree": 0}
    games = 0
    for g in _read(targets):
        games += 1
        for a in g["actions"]:
            rec = a.get("gchoice")
            if rec is not None:
                counts["dep" if int(rec["g"][budget]["c"]) != int(rec["raw"]) else "agree"] += 1
    p = {"dep": min(1.0, departures / max(1, counts["dep"])),
         "agree": min(1.0, agreements / max(1, counts["agree"]))}
    rows: List[Dict[str, Any]] = []
    for g in _read(targets):
        st = ts.GameState()
        ts.Engine.init_game(st, g["seed"])
        for idx, a in enumerate(g["actions"]):
            rec = a.get("gchoice")
            mask = np.asarray(ts.get_flat_action_mask(st, False))
            fa = int(a["flat_action"])
            if mask.sum() == 0 or mask[fa] == 0:
                raise RuntimeError(f"game {g.get('game_id')} desynchronised on replay")
            if rec is not None:
                stratum = "dep" if int(rec["g"][budget]["c"]) != int(rec["raw"]) else "agree"
                if _keep(f"{g['game_id']}|{idx}", seed, p[stratum]):
                    tok = position_token(st)
                    c = st.ctx()
                    mover = c.decision_player if c.decision_player != ts.Player.NONE else st.phasing_player
                    rows.append({"id": hashlib.sha256(tok.encode()).hexdigest()[:16], "pos": tok,
                                 "raw": int(rec["raw"]), "recorded_search": int(rec["g"][budget]["c"]),
                                 "stratum": stratum, "weight": round(1.0 / p[stratum], 4), "pl_incl": 1.0,
                                 "decision_type": str(c.decision_type).split(".")[-1],
                                 "side": "US" if mover == ts.Player.US else "USSR", "turn": int(st.turn),
                                 "game": g.get("game_id"), "decision": idx})
            ts.Engine.step(st, ts.decode_flat_action(st, fa))
            while (not ts.Engine.is_terminal(st) and st.ctx().decision_player == ts.Player.NONE
                   and st.ctx().decision_type == ts.DecisionType.ROLL_DIE):
                ts.Engine.step(st, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
    _write(out, rows)
    srcs = [json.load(open(t + ".meta.json")) for t in targets if os.path.exists(t + ".meta.json")]
    searched = counts["dep"] + counts["agree"]
    subsample = float(srcs[0]["searcher"].get("subsample", 1.0)) if srcs else 1.0
    meta = {"stage": "select", "targets": [os.path.basename(t) for t in targets], "budget": budget,
            "games": games, "searched_records": searched, "population": counts,
            "inclusion": p, "seed": seed, "rows": len(rows),
            "decisions_per_game": searched / subsample / max(1, games),
            "teacher_sha256": srcs[0].get("checkpoint_sha256") if srcs else None}
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return 0


def _wmean(xs: Sequence[float], ws: Sequence[float]) -> Tuple[float, float]:
    sw = float(sum(ws))
    if not xs or sw <= 0:
        return float("nan"), float("nan")
    m = sum(x * w for x, w in zip(xs, ws)) / sw
    return m, math.sqrt(sum((w * (x - m)) ** 2 for x, w in zip(xs, ws))) / sw


def analyse(bank: Dict[str, Dict[str, Any]], picks: Dict[str, Dict[str, int]],
            diffs: Dict[str, Dict[int, Tuple[float, float]]], names: Dict[str, str]) -> Dict[str, Any]:
    """`names` maps the roles teacher_search, student_raw, student_search, teacher_rollout,
    student_rollout to search-row spec names (a role may be absent)."""
    ids = [i for i in bank if i in picks and i in diffs]

    def d(i: str, a: Optional[int]) -> Optional[float]:
        if a is None:
            return None
        if a == int(bank[i]["raw"]):
            return 0.0
        v = diffs[i].get(a)
        return None if v is None else v[0]

    def pick(i: str, role: str) -> Optional[int]:
        n = names.get(role)
        return None if n is None or n not in picks[i] else int(picks[i][n])

    w = {i: float(bank[i]["weight"]) for i in ids}
    total = sum(w.values())
    out: Dict[str, Any] = {"positions": len(ids)}

    # Each searcher's offline gain over the teacher network's move, and the student's own gaps.
    gains: Dict[str, Tuple[float, float]] = {}
    for role in ("student_raw", "teacher_search", "student_search", "teacher_rollout", "student_rollout"):
        xs, ws = [], []
        for i in ids:
            v = d(i, pick(i, role))
            if v is not None:
                xs.append(v); ws.append(w[i])
        gains[role] = _wmean(xs, ws)
    for role, base in (("student_search", "student_raw"), ("student_rollout", "student_raw")):
        xs, ws = [], []
        for i in ids:
            a, b = d(i, pick(i, role)), d(i, pick(i, base))
            if a is not None and b is not None:
                xs.append(a - b); ws.append(w[i])
        gains[f"{role} over student_raw"] = _wmean(xs, ws)
    out["gain_per_decision"] = gains

    # Where the teacher's search departs from the teacher's network.
    dep = [i for i in ids if pick(i, "teacher_search") is not None
           and pick(i, "teacher_search") != int(bank[i]["raw"])]
    agr = [i for i in ids if pick(i, "teacher_search") == int(bank[i]["raw"])]
    out["teacher_search_departs"] = sum(w[i] for i in dep) / total if total else float("nan")
    classes: Dict[str, List[str]] = {"adopt": [], "reject": [], "other": []}
    for i in dep:
        s = pick(i, "student_raw")
        classes["adopt" if s == pick(i, "teacher_search") else "reject" if s == int(bank[i]["raw"]) else "other"].append(i)
    wd = sum(w[i] for i in dep)
    out["departures"] = {}
    for k, members in classes.items():
        corr = _wmean([d(i, pick(i, "teacher_search")) or 0.0 for i in members], [w[i] for i in members])
        stu = _wmean([d(i, pick(i, "student_raw")) or 0.0 for i in members], [w[i] for i in members])
        out["departures"][k] = {"n": len(members), "share": sum(w[i] for i in members) / wd if wd else float("nan"),
                                "correction_gain": corr, "student_gain": stu}

    # Where the teacher's search agrees: does the student move off it, and at what cost?
    changed = [i for i in agr if pick(i, "student_raw") not in (None, int(bank[i]["raw"]))]
    wa = sum(w[i] for i in agr)
    def lead(i: str, a: Optional[int]) -> Optional[Tuple[float, float]]:
        return None if a is None else diffs[i].get(a)

    worse = [i for i in changed
             if (lv := lead(i, pick(i, "student_raw"))) is not None and lv[0] < -2 * lv[1]]
    bad = [i for i in worse if (d(i, pick(i, "student_raw")) or 0.0) < -0.10]
    out["agreements"] = {"n": len(agr), "changed_share": sum(w[i] for i in changed) / wa if wa else float("nan"),
                         "changed_gain": _wmean([d(i, pick(i, "student_raw")) or 0.0 for i in changed],
                                                [w[i] for i in changed]),
                         "confirmed_worse_share": sum(w[i] for i in worse) / wa if wa else float("nan"),
                         "worse_by_10pts_share": sum(w[i] for i in bad) / wa if wa else float("nan"),
                         "n_changed": len(changed), "n_confirmed_worse": len(worse), "n_worse_10": len(bad)}

    # Where the teacher's rollout root and its Gumbel root disagree: whom does the student follow?
    if names.get("teacher_rollout"):
        x = [i for i in ids if pick(i, "teacher_rollout") is not None and pick(i, "teacher_search") is not None
             and pick(i, "teacher_rollout") != pick(i, "teacher_search")]
        wx = sum(w[i] for i in x)
        follows = {"rollout": [i for i in x if pick(i, "student_raw") == pick(i, "teacher_rollout")],
                   "gumbel": [i for i in x if pick(i, "student_raw") == pick(i, "teacher_search")]}
        follows["neither"] = [i for i in x if i not in follows["rollout"] and i not in follows["gumbel"]]
        out["rollout_vs_gumbel"] = {
            "n": len(x), "share_of_decisions": wx / total if total else float("nan"),
            "rollout_minus_gumbel": _wmean([(d(i, pick(i, "teacher_rollout")) or 0.0) - (d(i, pick(i, "teacher_search")) or 0.0)
                                            for i in x], [w[i] for i in x]),
            "student_follows": {k: sum(w[i] for i in v) / wx if wx else float("nan") for k, v in follows.items()}}

    # By decision type: the departures' adoption and the agreements' changes.
    kinds: Dict[str, Dict[str, Any]] = {}
    for kind in sorted({str(bank[i]["decision_type"]) for i in ids}):
        dk = [i for i in dep if bank[i]["decision_type"] == kind]
        ak = [i for i in agr if bank[i]["decision_type"] == kind]
        wdk = sum(w[i] for i in dk)
        wak = sum(w[i] for i in ak)
        kinds[kind] = {
            "departures": len(dk),
            "adopt": sum(w[i] for i in dk if i in classes["adopt"]) / wdk if wdk else float("nan"),
            "reject": sum(w[i] for i in dk if i in classes["reject"]) / wdk if wdk else float("nan"),
            "correction_gain": _wmean([d(i, pick(i, "teacher_search")) or 0.0 for i in dk], [w[i] for i in dk])[0],
            "agreements": len(ak),
            "changed": sum(w[i] for i in ak if i in changed) / wak if wak else float("nan")}
    out["by_kind"] = kinds
    return out


def markdown(res: Dict[str, Any], judge: str) -> str:
    g = res["gain_per_decision"]
    pt = lambda m: f"{100 * m[0]:+.2f} ± {100 * m[1]:.2f}"
    lines = [f"### Judge: {judge} ({res['positions']} positions)", "",
             "| offline gain over the teacher network's move (points per decision) | |", "|:---|---:|"]
    labels = {"student_raw": "student raw", "teacher_search": "teacher Gumbel", "student_search": "student Gumbel",
              "teacher_rollout": "teacher rollout root", "student_rollout": "student rollout root",
              "student_search over student_raw": "student Gumbel over student raw",
              "student_rollout over student_raw": "student rollout over student raw"}
    for k, v in g.items():
        if not math.isnan(v[0]):
            lines.append(f"| {labels.get(k, k)} | {pt(v)} |")
    lines += ["", f"The teacher's Gumbel departs from its network at {res['teacher_search_departs']:.1%} of decisions. "
              "There, the student network:", "",
              "| | n | share | gain of the teacher's correction | gain of the student's move |", "|:---|---:|---:|---:|---:|"]
    for k, v in res["departures"].items():
        lines.append(f"| {k} | {v['n']} | {v['share']:.1%} | {pt(v['correction_gain'])} | {pt(v['student_gain'])} |")
    a = res["agreements"]
    lines += ["", f"Where the teacher's Gumbel agrees ({a['n']} positions): the student plays another move at "
              f"{a['changed_share']:.1%} of them (gain {pt(a['changed_gain'])}); confirmed worse (2 SE) at "
              f"{a['confirmed_worse_share']:.2%}, worse by 10+ points at {a['worse_by_10pts_share']:.2%} "
              f"({a['n_confirmed_worse']} / {a['n_worse_10']} positions).", ""]
    if "rollout_vs_gumbel" in res:
        r = res["rollout_vs_gumbel"]
        f = r["student_follows"]
        lines += [f"Where the teacher's rollout root and Gumbel disagree ({r['n']} positions, {r['share_of_decisions']:.1%} "
                  f"of decisions; rollout minus Gumbel {pt(r['rollout_minus_gumbel'])}): the student plays the rollout "
                  f"root's move {f['rollout']:.1%}, Gumbel's {f['gumbel']:.1%}, neither {f['neither']:.1%}.", ""]
    lines += ["| decision type | departures | adopt | reject | correction gain | agreements | changed |",
              "|:---|---:|---:|---:|---:|---:|---:|"]
    for k, v in res["by_kind"].items():
        lines.append(f"| {k} | {v['departures']} | {v['adopt']:.0%} | {v['reject']:.0%} | "
                     f"{100 * v['correction_gain']:+.2f} | {v['agreements']} | {v['changed']:.1%} |")
    return "\n".join(lines) + "\n"


def report(bank_path: str, search_paths: Sequence[str], play_sets: Sequence[Tuple[str, Sequence[str]]],
           names: Dict[str, str], out: str) -> int:
    bank = {str(r["id"]): r for r in _read([bank_path])}
    picks: Dict[str, Dict[str, int]] = {}
    for r in _read(search_paths):
        cur = picks.setdefault(str(r["id"]), {})
        for n, ps in r["picks"].items():
            if ps:
                cur[n] = int(ps[0])
    results, md = {}, ["# Search-correction transfer", ""]
    for judge, paths in play_sets:
        diffs: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for r in _read(paths):
            diffs[str(r["id"])] = {int(a): (float(v[0]), float(v[1])) for a, v in r["diff_vs_raw"].items()}
        results[judge] = analyse(bank, picks, diffs, names)
        md.append(markdown(results[judge], judge))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("\n".join(md))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--targets", nargs="+", required=True)
    s.add_argument("--budget", default="256")
    s.add_argument("--departures", type=int, default=3000)
    s.add_argument("--agreements", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--bank", required=True)
    r.add_argument("--search", nargs="+", required=True)
    r.add_argument("--playouts", nargs="+", action="append", required=True,
                   help="one judge's playout files (repeat the flag per judge)")
    r.add_argument("--judges", nargs="+", default=None, help="a label per --playouts set")
    r.add_argument("--role", nargs="+", default=[], help="role=spec_name, e.g. teacher_search=soup_g256")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        return select(a.targets, a.budget, a.departures, a.agreements, a.seed, a.out)
    judges = a.judges or [f"judge {k + 1}" for k in range(len(a.playouts))]
    names = dict(x.split("=", 1) for x in a.role)
    return report(a.bank, a.search, list(zip(judges, a.playouts)), names, a.out)


if __name__ == "__main__":
    sys.exit(main())
