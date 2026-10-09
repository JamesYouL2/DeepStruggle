#!/usr/bin/env python3
"""The two-stage catastrophic-blunder census: where does a checkpoint throw a game away?

Stage 1 (`screen`, cheap) generates positions from self-play games of one checkpoint and screens
every decision two ways: the engine's own proof of an immediate catastrophe
(`ai/eval/safety.classify_legal_actions` -- a forced loss taken, or a forced win not taken), and
disagreement between the greedy policy and a low-budget Gumbel root (32 simulations). A position
that triggers either is saved with its exact state (`pos=` token), the moves and their labels,
every seed and the model's identity; a random control sample of the positions that triggered
nothing is kept beside them, so the screening's misses are measured rather than assumed.

Stage 2 (expensive) is `tools/scripts/bank_playouts.py validate`: each shortlisted position is
searched with Gumbel k=8 at 256 simulations over independent runs, and the moves it puts in play
-- greedy's, the cheap search's, the strong search's and a deterministic safe alternative -- are
played out in paired continuations (`ai/eval/paired_playouts.py`), on CI runners through
`.github/workflows/bank_playouts.yml`. Exact terminal mistakes (the engine's labels) are kept
apart from the strategic regret the continuations only estimate, with paired standard errors; the
unresolved remainder can be escalated to 1,024 simulations (`validate --escalate-from`).

`report` merges the two stages into the census report: candidate count and confirmation rate,
confirmed catastrophic blunders per 1,000 decisions, the share of confirmed mistakes the cheap
screening found, the share rescued by 256-simulation search, the share unresolved even under the
stronger search, the worst failures grouped by root cause, and the runtime of each stage.

    # 1. benchmark 20 games before choosing the full experiment size
    PYTHONPATH=.:build/release python tools/scripts/blunder_census.py benchmark \\
        --model data/checkpoints/hf/E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt \\
        --games 20 --project 2000 --out-dir data/reports/blunder_census/bench
    # 2. stage 1 -- the positions (resumable, in parts)
    PYTHONPATH=.:build/release python tools/scripts/blunder_census.py screen \\
        --model <ckpt.pt> --games 2000 --out-dir data/reports/blunder_census/screen
    # 3. stage 2 -- bank_playouts.py validate, locally or through bank_playouts.yml
    PYTHONPATH=.:build/release python tools/scripts/bank_playouts.py validate \\
        --input screen/candidates.jsonl.gz screen/control.jsonl.gz --model <ckpt.pt> \\
        --pairs 64 --part 1/20 --out validation.jsonl.gz
    # 4. the report
    PYTHONPATH=.:build/release python tools/scripts/blunder_census.py report \\
        --screen screen/candidates.jsonl.gz --control screen/control.jsonl.gz \\
        --summary screen/summary.json --validation validation.jsonl.gz --out report.md

The screening output and the validation output are kept apart on purpose: `validate` re-runs on
saved positions (a different budget, a different continuation) without ever regenerating them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.safety import (DEFCON_DEGRADING_CARDS, OLYMPIC_GAMES, OPPONENT_COUP_CARDS,
                            classify_legal_actions)
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import (load_policy, position_token, require_e4_view, selfplay,
                                     state_from_token)
from tools.lib.player_agent import load_agent
from tools.scripts.disagreement_bank import _name

SCHEMA = 1
#: the cheap search of stage 1 (research/log/E7_gumbel_headroom.md calls k=4 at 16-64 evaluations
#: the budget that keeps most of the search's strength)
SCREEN_SEARCH = (32, 4)
TRIGGERS = ("forced-loss", "missed-forced-win", "g32-disagrees")


def mover_of(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def decision_card(st: ts.GameState) -> int:
    """The card a decision is about: the one resolving, else the one being played for Ops, else 0."""
    ctx = st.ctx()
    return int(ctx.resolving_card) or int(ctx.pending_op_card)


def legal_actions(st: ts.GameState) -> List[int]:
    return [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]


def row_id(pos: str) -> str:
    """A stable id: the position itself, so a rebuilt census keeps the verdicts already given."""
    return hashlib.sha1(pos.encode()).hexdigest()[:16]


def triggers(greedy: int, g32: int, labels: Dict[int, str]) -> List[str]:
    """Which cheap checks fire at one decision. `forced-loss`: the greedy move ends the game
    against the mover whatever they do next, while some other legal move does not.
    `missed-forced-win`: a legal move wins on the spot and greedy's does not. `g32-disagrees`:
    a 32-simulation Gumbel root plays something else."""
    out: List[str] = []
    if labels.get(greedy) == "loss" and any(l != "loss" for l in labels.values()):
        out.append("forced-loss")
    if any(l == "win" for l in labels.values()) and labels.get(greedy) != "win":
        out.append("missed-forced-win")
    if g32 != greedy:
        out.append("g32-disagrees")
    return out


def safe_alternative(labels: Dict[int, str], greedy: int, probs: Sequence[Tuple[int, float]]) -> Optional[int]:
    """The deterministic alternative to a greedy move that triggers: a forced win where one
    exists, else the most probable legal move that is not itself a forced loss."""
    wins = [a for a, l in labels.items() if l == "win"]
    if wins:
        return sorted(wins)[0]
    for a, _ in sorted(probs, key=lambda t: -t[1]):
        if a != greedy and labels.get(a) != "loss":
            return int(a)
    return None


def decision_group(r: Dict[str, Any]) -> str:
    """Coarse decision class for the report's grouping."""
    dt = str(r["decision_type"])
    if dt == "SELECT_CARD":
        return "headline" if r["phase"] == "HEADLINE" else "card"
    if dt in ("SELECT_PLAY_MODE", "SELECT_OP_MODE", "POINT_NODE"):
        return dt.lower()
    return "event-choice"


def root_cause(r: Dict[str, Any]) -> str:
    """What made the failure, as the engine proves it (exact labels) or as the continuations
    estimate it (strategic regret). Used to group the report's worst failures."""
    v = str(r.get("verdict", ""))
    if v == "exact-missed-win":
        return "missed a forced win"
    if v == "exact-forced-loss":
        card, defcon = int(r.get("card") or 0), int(r.get("defcon") or 0)
        if str(r.get("decision_type")) == "SELECT_PLAY_MODE" and defcon <= 2:
            if card in DEFCON_DEGRADING_CARDS:
                return (f"played {DEFCON_DEGRADING_CARDS[card]} at DEFCON 2 "
                        "(its event degrades DEFCON)")
            if card == OLYMPIC_GAMES:
                return "played Olympic Games at DEFCON 2 (the boycott degrades DEFCON)"
            if card in OPPONENT_COUP_CARDS:
                return f"handed the opponent a free coup at DEFCON 2 ({OPPONENT_COUP_CARDS[card]})"
        return "took a forced loss (VP threshold / scoring card / coup line)"
    if v == "confirmed-regret":
        return f"strategic regret, confirmed by searched continuations ({decision_group(r)})"
    if v == "refuted":
        return "refuted -- the screen's suspicion did not hold up"
    return "unresolved"


def rescued_by_search(r: Dict[str, Any], label: str = "search") -> bool:
    """Whether the search's move (the run's `label`) avoids a confirmed mistake: it is not
    greedy's move, and it escapes the exact label where there is one, or scores strictly better
    than greedy in the paired continuations where there is not."""
    moves, choice = r["moves"], r["moves"].get(label)
    if choice is None or choice == moves["greedy"]:
        return False
    exact = r.get("exact", {})
    labels = {int(a): l for a, l in exact.get("labels", {}).items()}
    if exact.get("missed_win") and labels.get(int(choice)) != "win":
        return False
    if exact.get("forced_loss") and labels.get(int(choice)) == "loss":
        return False
    if exact.get("missed_win") or exact.get("forced_loss"):
        return True
    score = {int(a): v for a, v in r.get("score", {}).items()}
    return score.get(int(choice), 0.0) > score.get(int(moves["greedy"]), 0.0)


class DecisionLog:
    """Every decision of one self-play game with two or more legal moves, as the model met it."""

    def __init__(self) -> None:
        self.decision = 0
        self.kept: List[Tuple[int, ts.GameState, int]] = []

    def observe(self, st: ts.GameState, a: int) -> None:
        self.decision += 1
        if len(legal_actions(st)) >= 2:
            self.kept.append((self.decision, st, int(a)))


def annotate(kept: Sequence[Tuple[int, ts.GameState, int]], logits_fn: Any, features: int,
             search: Any, control_rate: float, rng: random.Random, chunk: int,
             model: str, model_sha256: str, game: Dict[str, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One game's decisions screened: (candidates, control sample). The searcher is reseeded with
    the game's seed by the caller, so a game's rows do not depend on which part it fell in."""
    if not kept:
        return [], []
    states = [s for _, s, _ in kept]
    obs = np.stack([np.asarray(ts.extract_observation_features(s, mover_of(s), features),
                               dtype=np.float32) for s in states])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
    logits = np.asarray(logits_fn(obs, masks), dtype=np.float64)
    picks: List[int] = []
    for lo in range(0, len(states), chunk):
        picks += [int(a) for a in search.select_actions_batch(states[lo:lo + chunk])]
    candidates: List[Dict[str, Any]] = []
    control: List[Dict[str, Any]] = []
    for i, ((dec, st, played), g32) in enumerate(zip(kept, picks)):
        legal = legal_actions(st)
        probs = [(a, float(np.exp(logits[i, a] - logits[i, legal].max()))) for a in legal]
        greedy = max(legal, key=lambda a: logits[i, a])
        labels = classify_legal_actions(st, mover_of(st))
        trig = triggers(greedy, g32, labels)
        ctx = st.ctx()
        pos = position_token(st)
        row: Dict[str, Any] = {
            "schema": SCHEMA, "id": row_id(pos),
            "model": model, "model_sha256": model_sha256, "game": dict(game), "decision": dec,
            "side": "US" if mover_of(st) == ts.Player.US else "USSR",
            "turn": int(st.turn), "ar": int(st.action_round),
            "phase": str(st.current_phase).split(".")[-1],
            "decision_type": str(ctx.decision_type).split(".")[-1],
            "defcon": int(st.defcon), "vp": int(st.victory_points), "card": decision_card(st),
            "legal": legal, "greedy": greedy, "played": played, "g32": g32,
            "safety": {"labels": {str(a): labels.get(a, "normal") for a in legal},
                       "safe": safe_alternative(labels, greedy, probs)},
            "triggers": trig,
            "names": {str(a): _name(st, a) for a in sorted({greedy, g32} | {a for a, l in labels.items()
                                                                        if l in ("win", "loss")})},
            "pos": pos,
        }
        if trig:
            candidates.append(row)
        elif rng.random() < control_rate:
            control.append(dict(row, control=True))
    return candidates, control


def screen(model: str, out_dir: str, games: int, part: str, seed: int, temperature: float,
           control_rate: float, chunk: int, resume: bool,
           search_sims: int = SCREEN_SEARCH[0], search_k: int = SCREEN_SEARCH[1]) -> int:
    """Stage 1 over this part's share of `--games` self-play games of `model` (game g is played
    from seed `--seed` + g, so the games do not depend on the split or on a resume)."""
    k, n = (int(x) for x in part.split("/"))
    mine = [g for g in range(games) if g % n == k - 1]
    os.makedirs(out_dir, exist_ok=True)
    cand_path = os.path.join(out_dir, "candidates.jsonl.gz")
    ctrl_path = os.path.join(out_dir, "control.jsonl.gz")
    ledger = os.path.join(out_dir, "games.jsonl")
    done: Dict[int, Dict[str, Any]] = {}
    if resume and os.path.exists(ledger):
        for line in open(ledger):
            g = json.loads(line)
            done[int(g["game"])] = g["counts"]
    elif os.path.exists(ledger):
        raise SystemExit(f"{ledger} exists: pass --resume to continue it, or remove it")
    todo = [g for g in mine if g not in done]
    t0 = time.time()
    logits_fn, features = load_policy(model)
    spec = f"gumbel:{model}:{search_sims}:{search_k}"
    search = load_agent(spec, device="cuda" if _cuda() else "cpu")
    require_e4_view(spec, search)
    meta = {"schema": SCHEMA, "stage": "screen", "model": os.path.basename(model),
            "model_sha256": _sha256(model), "seed": seed, "games": games, "part": part,
            "temperature": temperature, "search": f"gumbel:{search_sims}:{search_k}",
            "control_rate": control_rate}
    for gi, g in enumerate(todo):
        tracker = selfplay(logits_fn, features, 1, seed + g, 1, temperature,
                           DecisionLog)[0]
        if hasattr(search, "reseed"):
            search.reseed(seed + g)
        cands, ctrl = annotate(tracker.kept, logits_fn, features, search, control_rate,
                               random.Random(seed * 1_000_003 + g), chunk,
                               meta["model"], meta["model_sha256"], {"seed": seed + g, "index": g})
        _append(cand_path, cands)
        _append(ctrl_path, ctrl)
        counts: Dict[str, Any] = {"decisions": len(tracker.kept), "candidates": len(cands),
                                  "control": len(ctrl),
                                  "triggers": {t: sum(1 for r in cands if t in r["triggers"])
                                               for t in TRIGGERS}}
        with open(ledger, "a") as f:
            f.write(json.dumps({"game": g, "counts": counts}) + "\n")
        done[g] = counts
        print(f"[{gi + 1}/{len(todo)}] game {g}: {counts} ({time.time() - t0:.0f}s this run)",
              file=sys.stderr, flush=True)
    summary: Dict[str, Any] = dict(meta, part=part, games=len(done), wall_s=round(time.time() - t0, 1))
    for key in ("decisions", "candidates", "control"):
        summary[key] = sum(int(c[key]) for c in done.values())
    summary["triggers"] = {t: sum(int(c.get("triggers", {}).get(t, 0)) for c in done.values())
                           for t in TRIGGERS}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"{len(todo)} games screened in {time.time() - t0:.0f}s: {summary}", file=sys.stderr)
    return 0


def benchmark(model: str, out_dir: str, games: int, project: int, seed: int,
              control_rate: float, runners: int) -> int:
    """Screen `--games` (20 by design) and print what a full run of `--project` games would cost,
    so the experiment size is chosen from a measurement rather than a guess."""
    t0 = time.time()
    rc = screen(model, out_dir, games, "1/1", seed, 0.0, control_rate, 64, False)
    if rc != 0:
        return rc
    summary = json.load(open(os.path.join(out_dir, "summary.json")))
    wall = time.time() - t0
    per_game = wall / max(1, games)
    print(f"\nbenchmark: {games} games in {wall:.0f}s ({per_game:.1f}s/game), "
          f"{summary['decisions']} decisions ({summary['decisions'] / wall:.1f}/s), "
          f"{summary['candidates']} candidates, {summary['control']} control")
    print(f"projected for {project} games: {per_game * project / 3600:.1f} h alone, "
          f"{per_game * project / max(1, runners) / 60:.0f} min per runner over {runners} runners")
    rows = summary["candidates"] + summary["control"]
    print(f"stage 2 is then {rows * project / max(1, games):,.0f} positions to validate "
          f"({rows} in this benchmark)")
    return 0


# ---------------------------------------------------------------------------------------------
# report: the two stages merged
# ---------------------------------------------------------------------------------------------


def _read(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def _append(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with gzip.open(path, "at") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cuda() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def metrics(screened: Sequence[Dict[str, Any]], control: Sequence[Dict[str, Any]],
            validation: Sequence[Dict[str, Any]], rows_256: Sequence[Dict[str, Any]],
            decisions: Optional[int] = None) -> Dict[str, Any]:
    """The report's figures from the two stages' rows (`validation` with the escalated rows in
    place of their first verdicts; `rows_256` the first pass, what "rescued by 256-simulation
    search" is read from). `decisions` is stage 1's decision count (from its summary.json);
    without it the per-1,000 figure is None. The control sample's weight is the non-triggering
    decisions over the control rows kept, so the control's verdicts stand for all the positions
    the screening never looked at."""
    cand_ids = {r["id"] for r in screened}
    ctrl_ids = {r["id"] for r in control}
    n_cand, n_ctrl = len(cand_ids), len(ctrl_ids)
    if decisions is not None:
        non_trigger = max(0, decisions - n_cand)
        weight = non_trigger / n_ctrl if n_ctrl else 0.0
    else:
        non_trigger, weight = 0, 1.0

    def confirmed(r: Dict[str, Any]) -> bool:
        return str(r.get("verdict", "")).startswith("exact") or r.get("verdict") == "confirmed-regret"

    confirmed_cand = [r for r in validation if r["id"] in cand_ids and confirmed(r)]
    confirmed_ctrl = [r for r in validation if r["id"] in ctrl_ids and confirmed(r)]
    confirmed = confirmed_cand + confirmed_ctrl
    first = {r["id"]: r for r in rows_256}
    rescued = [r for r in confirmed if rescued_by_search(first.get(r["id"], r))]
    unresolved = [r for r in validation if r["id"] in cand_ids and r["verdict"] == "unresolved"]
    refuted = [r for r in validation if r["id"] in cand_ids and r["verdict"] == "refuted"]
    miss_w = len(confirmed_ctrl) * weight
    out: Dict[str, Any] = {
        "decisions": decisions, "candidates": n_cand, "control": n_ctrl,
        "control_weight": round(weight, 2),
        "validated": len(validation), "confirmed_candidates": len(confirmed_cand),
        "confirmed_control": len(confirmed_ctrl), "confirmation_rate":
            (len(confirmed_cand) / len([r for r in validation if r["id"] in cand_ids])
             if any(r["id"] in cand_ids for r in validation) else None),
        "refuted": len(refuted), "unresolved": len(unresolved),
        "per_1000_decisions": ((len(confirmed_cand) + miss_w) / decisions * 1000
                               if decisions else None),
        "detected_by_screen": (len(confirmed_cand) / (len(confirmed_cand) + miss_w)
                               if (confirmed_cand or miss_w) else None),
        "rescued_by_256": (len(rescued) / len(confirmed) if confirmed else None),
        "unresolved_share": (len(unresolved) / max(1, len([r for r in validation
                                                           if r["id"] in cand_ids]))),
    }
    return out


def render(census: Dict[str, Any], worst: Sequence[Tuple[str, List[Dict[str, Any]]]],
           meta: Dict[str, Any]) -> str:
    """The report's markdown. `worst` is (root cause, rows) in severity order."""

    def f(x: Optional[float], digits: int = 1) -> str:
        return "n/a" if x is None else f"{x:.{digits}f}"

    esc = meta.get("escalation_sims")
    header = (f"* model `{meta.get('model', '?')}` (sha256 `{str(meta.get('model_sha256', ''))[:12]}...`), "
              f"screen search `{meta.get('search', 'gumbel:32:4')}`, "
              f"validation Gumbel k={meta.get('search_k', 8)} at {meta.get('search_sims', 256)} "
              f"simulations" + (f" (escalation at {esc} simulations)" if esc else "") +
              f", continuations `{meta.get('cont', 'greedy')}`, {meta.get('pairs', '?')} pairs, "
              f"screen seed {meta.get('seed', '?')}, validation seed {meta.get('val_seed', '?')}.")
    lines = [
        f"# Catastrophic-blunder census: {meta.get('model', '?')}",
        "",
        header,
        "",
        "## Stage 1 -- screening",
        "",
        f"* {census['decisions']} decisions with two or more legal moves in "
        f"{meta.get('games', '?')} self-play games; {census['candidates']} triggered the cheap "
        f"filter and {census['control']} control positions (rate {meta.get('control_rate', '?')}, "
        f"weight {census['control_weight']}) did not.",
        f"* triggers: " + ", ".join(f"{t} {c}" for t, c in (meta.get("triggers") or {}).items()) + ".",
        "",
        "## Stage 2 -- validation",
        "",
        f"* {census['validated']} positions validated: "
        f"{census['confirmed_candidates']} candidates and {census['confirmed_control']} control "
        f"positions confirmed as catastrophic, {census['refuted']} candidates refuted, "
        f"{census['unresolved']} unresolved.",
        "",
        "## The figures",
        "",
        f"| figure | value |",
        f"|:---|---:|",
        f"| candidate count | {census['candidates']} |",
        f"| confirmation rate (candidates) | {f(census['confirmation_rate'] * 100 if census['confirmation_rate'] is not None else None)}% |",
        f"| confirmed catastrophic blunders per 1,000 decisions | {f(census['per_1000_decisions'], 2)} |",
        f"| share of confirmed mistakes the cheap screening found | {f(census['detected_by_screen'] * 100 if census['detected_by_screen'] is not None else None)}% |",
        f"| share rescued by 256-simulation search | {f(census['rescued_by_256'] * 100 if census['rescued_by_256'] is not None else None)}% |",
        f"| share unresolved even under the stronger search | {f(census['unresolved_share'] * 100)}% |",
        "",
        "Exact terminal mistakes (the engine's own labels) are counted separately from the "
        "strategic regret the paired continuations estimate; the per-1,000 figure sums both and "
        "adds the control sample's confirmed mistakes at their weight, standing in for the "
        "positions the screening never looked at.",
        "",
    ]
    for cause, rows in worst:
        lines.append(f"### {cause} ({len(rows)})")
        lines.append("")
        lines.append("| id | side | turn/ar | decision | greedy | search | verdict | regret (SE) | replay |")
        lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|:---|")
        for r in rows:
            d = r.get("diff_vs_greedy", {}).get(str(r.get("best")), [None, None])
            regret = ("--" if d[0] is None else f"{d[0]:+.3f} ({d[1]:.3f})")
            names = r.get("names", {})
            replay = f"game seed {r['game']['seed']}, decision {r['decision']}, pos `{r['id']}`"
            lines.append(
                f"| `{r['id']}` | {r['side']} | {r['turn']}/{r['ar']} | {decision_group(r)} "
                f"| {names.get(str(r['moves'].get('greedy')), r['moves'].get('greedy'))} "
                f"| {names.get(str(r['moves'].get('search')), r['moves'].get('search', '--'))} "
                f"| {r['verdict']} | {regret} | {replay} |")
        lines.append("")
    lines += [
        "Every failure's exact board is its `pos=` token in the screening JSONL (`pos`), openable "
        "in the workbench (`?pos=<token>`) and regenerable from its `game.seed` and `decision`.",
        "",
        "## Runtime",
        "",
    ]
    for k, v in meta.get("runtime", {}).items():
        lines.append(f"* {k}: {v}")
    return "\n".join(lines) + "\n"


def report(screen_paths: Sequence[str], control_paths: Sequence[str], summary_paths: Sequence[str],
           validation_paths: Sequence[str], escalation_paths: Sequence[str], out: str,
           top: int) -> int:
    screened = _read(screen_paths)
    control = _read(control_paths)
    validation = _read(validation_paths)
    escalation = _read(escalation_paths)
    decisions: Optional[int] = None
    runtime: Dict[str, str] = {}
    trigger_counts: Dict[str, int] = {}
    stage2_wall: Dict[str, float] = {"validation": 0.0, "escalation": 0.0}
    stage2_parts: Dict[str, int] = {"validation": 0, "escalation": 0}
    for p in summary_paths:
        s = json.load(open(p))
        decisions = (decisions or 0) + int(s.get("decisions", 0))
        for t, c in (s.get("triggers") or {}).items():
            trigger_counts[t] = trigger_counts.get(t, 0) + int(c)
        runtime.setdefault("stage 1 (screening)", "")
        runtime["stage 1 (screening)"] += (f"{s.get('games', '?')} games in {s.get('wall_s', '?')}s "
                                           f"local wall ({p}); ")
    for p in list(validation_paths) + list(escalation_paths):
        mp = p + ".meta.json"
        if os.path.exists(mp):
            m = json.load(open(mp))
            budget = str(m.get("budget", "validation"))
            stage2_wall[budget] = stage2_wall.get(budget, 0.0) + float(m.get("wall_s", 0.0))
            stage2_parts[budget] = stage2_parts.get(budget, 0) + 1
            key = f"stage 2 ({budget}, part {m.get('part', '?')})"
            runtime[key] = f"{m.get('wall_s', '?')}s wall on one runner ({mp})"
    for budget, parts in stage2_parts.items():
        if parts:
            runtime[f"stage 2 ({budget}) total"] = (
                f"{parts} parts, {stage2_wall[budget]:.0f} runner-seconds = "
                f"{stage2_wall[budget] / 60:.0f} runner-minutes")
    # the escalated rows replace their 256-simulation verdicts where the stronger search ran
    final = {r["id"]: r for r in validation}
    for r in escalation:
        final[r["id"]] = r
    rows = list(final.values())
    census = metrics(screened, control, rows, validation, decisions)
    serious = {"exact-missed-win": 0, "exact-forced-loss": 1, "confirmed-regret": 2,
               "unresolved": 3, "refuted": 4}

    def rank(r: Dict[str, Any]) -> Tuple[int, float]:
        d = r.get("diff_vs_greedy", {}).get(str(r.get("best")), [0.0, 0.0])
        return (serious.get(str(r.get("verdict")), 9), -abs(float(d[0] or 0.0)))

    confirmed = [r for r in rows if str(r.get("verdict", "")).startswith("exact")
                 or r.get("verdict") == "confirmed-regret"]
    worst_rows = sorted(confirmed, key=rank)[:top]
    grouped: List[Tuple[str, List[Dict[str, Any]]]] = []
    for cause in dict.fromkeys(root_cause(r) for r in worst_rows):
        grouped.append((cause, [r for r in worst_rows if root_cause(r) == cause]))
    meta: Dict[str, Any] = {"model": (screened + control or [{"model": "?"}])[0].get("model"),
                            "model_sha256": (screened + control or [{"model_sha256": ""}])[0].get("model_sha256"),
                            "games": 0, "runtime": runtime}
    for p in summary_paths:
        s = json.load(open(p))
        meta["games"] = int(meta["games"]) + int(s.get("games", 0))
        meta.setdefault("search", s.get("search"))
        meta.setdefault("control_rate", s.get("control_rate"))
        meta.setdefault("seed", s.get("seed"))
    meta["triggers"] = trigger_counts
    for p in list(validation_paths) + list(escalation_paths):
        mp = p + ".meta.json"
        if os.path.exists(mp):
            m = json.load(open(mp))
            meta.setdefault("val_seed", m.get("seed"))
            if m.get("budget") == "escalation":
                meta["escalation_sims"] = m.get("search_sims")
            else:
                meta.setdefault("search_sims", m.get("search_sims"))
                meta.setdefault("search_k", m.get("search_k"))
            meta.setdefault("pairs", m.get("pairs"))
            meta.setdefault("cont", m.get("cont"))
    text = render(census, grouped, meta)
    with open(out, "w") as f:
        f.write(text)
    print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark", help="screen 20 games and project the full run's cost")
    b.add_argument("--model", required=True, help="the checkpoint the census is about (.pt)")
    b.add_argument("--games", type=int, default=20)
    b.add_argument("--project", type=int, default=2000, help="the full run's game count to project")
    b.add_argument("--runners", type=int, default=1, help="runners the projection divides by")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--control-rate", type=float, default=0.05)
    b.add_argument("--out-dir", required=True)
    s = sub.add_parser("screen", help="stage 1: generate and screen self-play positions")
    s.add_argument("--model", required=True)
    s.add_argument("--games", type=int, default=100)
    s.add_argument("--part", default="1/1", help="k/N: this part's share of the games")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--temperature", type=float, default=0.0,
                   help="self-play sampling temperature (0: the model's own greedy games)")
    s.add_argument("--search-sims", type=int, default=SCREEN_SEARCH[0])
    s.add_argument("--search-k", type=int, default=SCREEN_SEARCH[1])
    s.add_argument("--control-rate", type=float, default=0.05,
                   help="share of the positions that trigger nothing to keep as control")
    s.add_argument("--chunk", type=int, default=64, help="positions per search call")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--out-dir", required=True)
    r = sub.add_parser("report", help="merge screening and validation into the census report")
    r.add_argument("--screen", nargs="+", required=True, help="stage 1 candidates.jsonl.gz")
    r.add_argument("--control", nargs="+", default=[], help="stage 1 control.jsonl.gz")
    r.add_argument("--summary", nargs="*", default=[], help="stage 1 summary.json files")
    r.add_argument("--validation", nargs="+", required=True, help="stage 2 output(s) at 256 simulations")
    r.add_argument("--escalation", nargs="*", default=[], help="stage 2 escalation output(s) at 1,024")
    r.add_argument("--top", type=int, default=20, help="failures to show, grouped by root cause")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "benchmark":
        return benchmark(a.model, a.out_dir, a.games, a.project, a.seed, a.control_rate, a.runners)
    if a.cmd == "screen":
        return screen(a.model, a.out_dir, a.games, a.part, a.seed, a.temperature,
                      a.control_rate, a.chunk, a.resume, a.search_sims, a.search_k)
    return report(a.screen, a.control, a.summary, a.validation, a.escalation, a.out, a.top)


if __name__ == "__main__":
    raise SystemExit(main())
