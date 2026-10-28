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
  which `--pairs` pays out against both with a judge network;
* `prove` -- where the game ends in the current turn, the rest of it enumerated exactly: whether the
  blunder is proved, and whether the row is decided whatever is played (the move wins whatever
  follows, or the position is already lost). Playouts charge such a move for a later mistake, or
  for the opponent failing to finish a game it has won;
* `settle` -- the bank without the rows `prove` finds decided.

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
    python tools/scripts/blunder_bank.py prove --input ai/eval/banks/blunders_E7.jsonl \\
        --budget 2000000 --part 1/37 --out prove-1.jsonl.gz
    python tools/scripts/blunder_bank.py settle --bank ai/eval/banks/blunders_E7.jsonl \\
        --prove prove-*.jsonl.gz --out ai/eval/banks/blunders_E7.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.paired_playouts import compare, decider, pair_start, paired_diff
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import load_policy, require_e4_view, state_from_token
from tools.lib.hidden_draws import ManyDraws, step_outcomes
from tools.lib.jsonl import read_jsonl, sha256_file, write_jsonl
from tools.lib.player_agent import BatchSelector, Reseedable, load_agent, resolve_device
from tools.lib.position_bank import CARD_NAMES, move_name, position_seed, searched_continuation
from tools.lib.engine_fingerprint import require_fresh

SCHEMA = 1
WORKBENCH = "https://mihaild.github.io/DeepStruggle"
#: positions per paired-playout batch in `confirm` and `score`
BATCH = 8


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


def search_sims(row: Dict[str, Any]) -> int:
    """The simulations of the search that validated a row (`search.spec`, "gumbel:SIMS:K")."""
    parts = str(row.get("search", {}).get("spec", "")).split(":")
    return int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0


def strongest(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One row per id: the one validated at the most simulations (an escalation replaces the
    validation it re-measured), the first given between equals."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = str(r["id"])
        if rid not in best or search_sims(r) > search_sims(best[rid]):
            best[rid] = r
    return best


def select(validation: Sequence[str], min_gap: float, min_z: float, out: str) -> int:
    """The census rows to confirm, each with its network move and the alternative fixed now. Given
    the census's validation and its escalation together, a row is judged on its escalated
    measurement (`strongest`)."""
    rows: List[Dict[str, Any]] = []
    for r in strongest(list(read_jsonl(validation))).values():
        if not shortlisted(r, min_gap, min_z):
            continue
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
    write_jsonl(out, rows)
    print(f"{len(rows)} rows shortlisted at {min_gap} points and z >= {min_z}", file=sys.stderr)
    return 0


def greedy_fn(model: str) -> Tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], int]:
    """`model`'s greedy policy over observations, and the observation feature set it reads."""
    logits_fn, feats = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)
    return act, feats


def _move(x: Any) -> int:
    """A move as a shortlist row holds it (an action) or a bank row does ({"action", "name"})."""
    return int(x["action"]) if isinstance(x, dict) else int(x)


def confirm(inputs: Sequence[str], model: str, pairs: int, seed: int, part: str, out: str,
            continue_with: Optional[str] = None) -> int:
    """Every measured move of this part's rows, paired out by `model`'s greedy play -- or, with
    `continue_with` (SIMS:K), by a Gumbel root of `model` playing both sides. Each row's pairs (and
    the searcher's stream) are seeded from its id and `--seed` alone, so a row's result does not
    depend on its part. Takes shortlist rows or bank rows."""
    k, n = (int(x) for x in part.split("/"))
    rows = [r for i, r in enumerate(sorted(read_jsonl(inputs), key=lambda r: str(r["id"]))) if i % n == k - 1]
    act, feats = greedy_fn(model)
    sha = sha256_file(model)
    cont: Any = None
    cont_label = "greedy"
    if continue_with:
        cont, cont_label = searched_continuation(model, continue_with)
    batch = 1 if cont is not None else BATCH
    res: List[Dict[str, Any]] = []
    t0 = time.time()
    for lo in range(0, len(rows), batch):
        chunk = rows[lo:lo + batch]
        positions = [(state_from_token(r["pos"]), [int(a) for a in r["moves"]]) for r in chunk]
        if cont is not None:
            cont.reseed(position_seed(str(chunk[0]["id"]), seed + 2_000_003) % (1 << 31))
        scores = compare(positions, act, pairs, [position_seed(str(r["id"]), seed) for r in chunk],
                         select=cont.select_actions_batch if cont is not None else None, features=feats)
        for r, sc in zip(chunk, scores):
            net = _move(r["network"])
            diffs = {a: paired_diff(sc[a], sc[net]) for a in sc if a != net}
            m, se = diffs[_move(r["better"])]
            res.append(dict(r, judge={"model": os.path.basename(model), "model_sha256": sha},
                            pairs=pairs, seed=seed, cont=cont_label,
                            score={str(a): round(sum(v) / len(v), 4) for a, v in sc.items()},
                            diff_vs_network={str(a): [round(x, 4), round(y, 4)] for a, (x, y) in diffs.items()},
                            cost=[round(m, 4), round(se, 4)]))
        print(f"  {len(res)}/{len(rows)} rows, {time.time() - t0:.0f}s", file=sys.stderr)
    write_jsonl(out, res)
    json.dump({"part": part, "rows": len(res), "seconds": round(time.time() - t0, 1), "pairs": pairs,
               "seed": seed, "cont": cont_label, "model": os.path.basename(model), "model_sha256": sha},
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
    net, better = _move(first["network"]), _move(first["better"])
    return {
        "schema": SCHEMA, "id": rid, "pos": first["pos"],
        "side": first["side"], "turn": first["turn"], "ar": first["ar"], "phase": first["phase"],
        "decision_type": first["decision_type"], "defcon": first["defcon"], "vp": first["vp"],
        "card": first["card"], "card_name": CARD_NAMES.get(int(first["card"]), ""),
        "network": {"action": net, "name": move_name(st, net)},
        "better": {"action": better, "name": move_name(st, better)},
        "cost": {j: {"cost": r["cost"][0], "se": r["cost"][1], "pairs": r["pairs"], "seed": r["seed"],
                     "model": r["judge"]["model"], "model_sha256": r["judge"]["model_sha256"]}
                 for j, r in by_judge.items()},
        "score": {j: r["score"] for j, r in by_judge.items()},
        "moves": first["moves"], "names": {str(a): move_name(st, int(a)) for a in first["moves"]},
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
        rows[name] = {r["id"]: r for r in read_jsonl(paths)}
    judges = list(rows)
    keep = confirmed(rows, min_gap, min_z)
    bank = [bank_row(rid, {j: rows[j][rid] for j in judges}) for rid in keep]
    write_jsonl(out, bank)
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
            obs = np.asarray(ts.extract_observation_features(st, decider(st), self.features), dtype=np.float32)
            mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
            return int(np.asarray(self.logits(obs[None], mask[None]))[0].argmax())
        assert self.agent is not None
        if isinstance(self.agent, Reseedable):
            self.agent.reseed(seed % (1 << 31))
        return int(self.agent.select_actions_batch([st])[0])


def score(bank_path: str, players: Sequence[str], judge: Optional[str], pairs: int, seed: int,
          out: str) -> int:
    bank = list(read_jsonl([bank_path]))
    act, feats = greedy_fn(judge) if judge and pairs > 0 else (None, 0)
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
            per.append({"id": r["id"], "pick": a, "name": move_name(st, a), "kind": kind})
        others = [(p, r) for p, r in zip(per, bank) if p["kind"] == "other"]
        if act is not None and others:
            for lo in range(0, len(others), BATCH):
                chunk = others[lo:lo + BATCH]
                positions = [(state_from_token(r["pos"]), sorted({p["pick"], r["network"]["action"]}))
                             for p, r in chunk]
                sc = compare(positions, act, pairs, [position_seed(str(r["id"]), seed + 1) for _, r in chunk],
                             features=feats)
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


class Unprovable(Exception):
    """The rest of the game cannot be enumerated: a new turn (a new deal) or the node budget."""


def exact_value(st: ts.GameState, mover: ts.Player, turn: int, budget: List[int], opponent: str) -> float:
    """The mover's exact score (1 win, 0.5 draw, 0 loss) to the end of a game that ends in this turn:
    the maximum over the mover's moves, the average over the six faces of each die and over the
    cards a step draws (`tools/lib/hidden_draws.py`), and over the opponent's moves the minimum
    (`opponent="min"`, its best reply) or the maximum ("max", its most helpful one -- the bound a
    blunder is held to). A new turn, an exhausted node budget or a step that draws more than once
    raises `Unprovable`."""
    if ts.Engine.is_terminal(st):
        u = float(ts.Engine.get_terminal_utility(st))
        if u == 0:
            return 0.5
        return 1.0 if (u > 0) == (mover == ts.Player.US) else 0.0
    budget[0] -= 1
    if budget[0] < 0:
        raise Unprovable("node budget")
    if int(st.turn) != turn:
        raise Unprovable("a new turn")
    ctx = st.ctx()

    def value_of(step: Callable[[ts.GameState], None]) -> float:
        # a step's outcomes: one, or the equally likely results of a card it draws
        outs = outcomes(st, step)
        return sum(exact_value(c, mover, turn, budget, opponent) for c in outs) / len(outs)

    if ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE:
        return sum(value_of(_face(face)) for face in range(1, 7)) / 6
    legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
    if legal.size == 0:
        return value_of(_advance(st, turn))
    maximise = decider(st) == mover or opponent == "max"
    best = 0.0 if maximise else 1.0
    for a in legal:
        v = value_of(_flat(int(a)))
        best = max(best, v) if maximise else min(best, v)
        if best == (1.0 if maximise else 0.0):
            break
    return best


def outcomes(st: ts.GameState, step: Callable[[ts.GameState], None]) -> List[ts.GameState]:
    """`hidden_draws.step_outcomes`, with a step that draws more than once reported as
    unprovable rather than taken at the draw the world's generator happens to make."""
    try:
        return step_outcomes(st, step)
    except ManyDraws as e:
        raise Unprovable(str(e)) from e


def _flat(a: int) -> Callable[[ts.GameState], None]:
    return lambda c: ts.Engine.step_flat(c, a)


def _face(face: int) -> Callable[[ts.GameState], None]:
    return lambda c: ts.Engine.step(c, ts.MicroAction(ts.DecisionType.ROLL_DIE, face, 0, 0))


def _advance(st: ts.GameState, turn: int) -> Callable[[ts.GameState], None]:
    """Between decisions (a phase or round transition): the step that advances to the next
    node, after checking it is one -- a new turn deals cards, which no search enumerates."""
    nxt = st.clone()
    if ts.Engine.auto_advance_step(nxt) == 0:
        raise Unprovable("no legal move and nothing to advance")
    if int(nxt.turn) != turn:
        raise Unprovable("a new turn")

    def step(c: ts.GameState) -> None:
        ts.Engine.auto_advance_step(c)
    return step


def reaches(st: ts.GameState, mover: ts.Player, turn: int, want: str, budget: List[int],
            memo: Dict[bytes, bool]) -> bool:
    """Whether the mover, playing its best, reaches `want` against the opponent's best reply in a game
    that ends in this turn: "win" -- a win on every face of every die; "any" -- a draw or a win on
    some face. A yes/no question cuts off at the first move that settles it, and `memo` (keyed by
    the save, which leaves out the move history) merges the orders of one set of placements, so it
    reaches rows `exact_value` cannot. A new turn or an exhausted node budget raises `Unprovable`."""
    if ts.Engine.is_terminal(st):
        u = float(ts.Engine.get_terminal_utility(st))
        won = u != 0 and (u > 0) == (mover == ts.Player.US)
        return won if want == "win" else (u == 0 or won)
    if int(st.turn) != turn:
        raise Unprovable("a new turn")
    key = hashlib.blake2b(st.to_save_json().encode(), digest_size=16).digest()
    if key in memo:
        return memo[key]
    budget[0] -= 1
    if budget[0] < 0:
        raise Unprovable("node budget")
    ctx = st.ctx()
    # chance -- a die, or a card a step draws: "win" needs every outcome, "any" one
    chance = want == "win"
    steps: List[Callable[[ts.GameState], None]]
    if ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE:
        steps = [_face(face) for face in range(1, 7)]
        every = chance
    else:
        legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
        if legal.size == 0:
            steps = [_advance(st, turn)]
            every = True
        else:
            # the mover needs one move that gets there; against the opponent, every reply must
            steps = [_flat(int(a)) for a in legal]
            every = decider(st) != mover

    def holds(step: Callable[[ts.GameState], None]) -> bool:
        results = (reaches(c, mover, turn, want, budget, memo) for c in outcomes(st, step))
        return all(results) if chance else any(results)

    res = every
    for step in steps:
        if holds(step) != every:
            res = not every
            break
    memo[key] = res
    return res


def outcome(st: ts.GameState, mover: ts.Player, turn: int, budget: int, after: bool = True) -> Optional[str]:
    """"win" where the mover wins whatever the opponent and the dice do, "loss" where it loses
    whatever it does against the opponent's best reply, "open" otherwise; None where the rest of
    the game cannot be enumerated. With `after=False` the question is only whether the position
    is lost (so "win" is never returned): the mover to move, every move it has tried."""
    try:
        if after and reaches(st, mover, turn, "win", [budget], {}):
            return "win"
        return "open" if reaches(st, mover, turn, "any", [budget], {}) else "loss"
    except Unprovable:
        return None


def outcome_after(st: ts.GameState, action: int, mover: ts.Player, turn: int, budget: int) -> Optional[str]:
    """`outcome` after `action` is played -- over every card the move itself may draw: "win" or
    "loss" where every draw is, "open" otherwise, None where a draw was not enumerated."""
    try:
        kinds = {outcome(c, mover, turn, budget) for c in outcomes(st, _flat(action))}
    except Unprovable:
        return None
    if None in kinds:
        return None
    return str(kinds.pop()) if len(kinds) == 1 else "open"


def value_after(st: ts.GameState, action: int, mover: ts.Player, turn: int, budget: int,
                opponent: str) -> float:
    """`exact_value` after `action` is played, averaged over the cards the move itself may draw."""
    outs = outcomes(st, _flat(action))
    return sum(exact_value(c, mover, turn, [budget], opponent) for c in outs) / len(outs)


def settled(network: Optional[str], position: Optional[str]) -> Optional[str]:
    """One world's verdict on whether the move matters: "win" where the move played wins whatever
    happens after it (a sure win is the best result there is, so whatever the playouts charge it is
    a later mistake), "loss" where the position before it is lost whatever the mover plays (so the
    better move's lead is the opponent failing to finish the game); None where the move can matter."""
    if network == "win":
        return "win"
    return "loss" if position == "loss" else None


def decided(worlds: Sequence[Optional[str]]) -> str:
    """"win" or "loss" where every world is settled that way (`settled`) -- the row is then no
    blunder, whatever its cost in playouts; "either" where the worlds split between the two; ""
    where the move can matter in some world or a world was not enumerated."""
    if not worlds or any(w is None for w in worlds):
        return ""
    kinds = set(worlds)
    return str(kinds.pop()) if len(kinds) == 1 else "either"


def prove(inputs: Sequence[str], deals: int, budget: int, seed: int, part: str, out: str) -> int:
    """Each bank row solved to the end of the game where the game ends in this turn, in the position
    as recorded and in `deals` worlds with the mover's unseen cards redealt
    (`paired_playouts.pair_start`). The blunder is held to its best case (every later choice, the
    opponent's too, made for the mover) and the better move to its worst (the opponent's replies
    against the mover). A row is proved where, in every world, the better move's worst case is at
    least the blunder's best case, and above it in at least one; proved against the best reply where
    the same holds with the blunder too answered by the opponent's best reply in that world.

    Separately, whether the move matters at all (`outcome`, a yes/no search, so it settles rows the
    values cannot): a row is decided where, in every world, the move played wins whatever follows,
    or the position is lost whatever the mover plays (`settled`, `decided`).
    This part's share of the rows (`--part k/N`, for CI runners), one JSON row each."""
    k, n = (int(x) for x in part.split("/"))
    bank = [r for i, r in enumerate(sorted(read_jsonl(inputs), key=lambda r: str(r["id"]))) if i % n == k - 1]
    res: List[Dict[str, Any]] = []
    t0 = time.time()
    for r in bank:
        st0 = state_from_token(r["pos"])
        who = decider(st0)
        worlds = [st0] + [pair_start(st0, k, position_seed(str(r["id"]), seed)) for k in range(deals)]
        outcomes: Dict[str, List[Optional[str]]] = {"network": [], "position": [], "settled": []}
        for w in worlds:
            net = outcome_after(w, int(r["network"]["action"]), who, int(w.turn), budget)
            # the position is searched only where the move loses: otherwise it is not lost
            pos = outcome(w, who, int(w.turn), budget, after=False) if net == "loss" else None
            outcomes["network"].append(net)
            outcomes["position"].append(pos)
            outcomes["settled"].append(settled(net, pos))
        verdict_decided = decided(outcomes["settled"])
        vals: Dict[str, List[Optional[float]]] = {"network_best_case": [], "network_vs_best_reply": [],
                                                   "better_worst_case": []}
        why = ""
        for w in worlds:
            for role, key, opp in (("network", "network_best_case", "max"), ("network", "network_vs_best_reply", "min"),
                                   ("better", "better_worst_case", "min")):
                try:
                    vals[key].append(value_after(w, int(r[role]["action"]), who, int(w.turn), budget, opp))
                except Unprovable as e:
                    vals[key].append(None)
                    why = why or str(e)
        nets = [v for v in vals["network_best_case"] if v is not None]
        replies = [v for v in vals["network_vs_best_reply"] if v is not None]
        bets = [v for v in vals["better_worst_case"] if v is not None]
        ok = len(nets) == len(replies) == len(bets) == len(worlds)

        def dominates(xs: List[float]) -> bool:
            return all(b >= x for x, b in zip(xs, bets)) and any(b > x for x, b in zip(xs, bets))
        proved = ok and dominates(nets)
        # the weaker claim: the blunder loses to the opponent's correct reply in that world
        proved_vs_reply = ok and dominates(replies)
        gap = round(sum(b - x for x, b in zip(replies, bets)) / len(worlds), 4) if ok else None
        res.append({"id": r["id"], "side": r["side"], "turn": r["turn"], "ar": r["ar"],
                    "decision_type": r["decision_type"], "card_name": r.get("card_name", ""),
                    "network": r["network"]["name"], "better": r["better"]["name"], "exact": vals,
                    "proved": proved, "proved_vs_best_reply": proved_vs_reply, "mean_exact_gap": gap,
                    "unprovable": "" if ok else why, "outcomes": outcomes, "decided": verdict_decided})
        verdict = ("PROVED" if proved else "PROVED against the best reply" if proved_vs_reply else "solved, not proved")
        print(f"{r['side']} T{r['turn']} AR{r['ar']} {r['decision_type']}: "
              + (f"{verdict} gap {gap}" if ok else f"unprovable ({why})")
              + (f"; decided ({verdict_decided} whatever is played)" if verdict_decided else ""), file=sys.stderr)
    write_jsonl(out, res)
    json.dump({"part": part, "rows": len(res), "seconds": round(time.time() - t0, 1), "deals": deals,
               "budget": budget, "seed": seed}, open(out + ".meta.json", "w"))
    print(f"{sum(x['proved'] for x in res)} of {len(res)} proved whatever the opponent does; "
          f"{sum(x['proved_vs_best_reply'] for x in res)} against its best reply; "
          f"{sum(1 for x in res if not x['unprovable'] and not x['proved_vs_best_reply'])} solved and not proved; "
          f"{sum(1 for x in res if x['decided'])} decided whatever is played", file=sys.stderr)
    return 0


def settle(bank_path: str, proofs: Sequence[str], out: str) -> int:
    """The bank without its decided rows (`prove`): where the move played wins whatever follows,
    or the position is lost whatever the mover plays, the move decides nothing, and the cost the
    playouts give it belongs to a later move or to the opponent's failure to finish the game. Every
    remaining row keeps its `prove` outcome in `exact.decided` ("")."""
    bank = list(read_jsonl([bank_path]))
    proof = {str(p["id"]): p for p in read_jsonl(sorted({f for g in proofs for f in glob.glob(g)}))}
    missing = [str(r["id"]) for r in bank if str(r["id"]) not in proof]
    if missing:
        raise SystemExit(f"no prove row for {len(missing)} bank rows (a part missing?): {missing[:5]}")
    kept: List[Dict[str, Any]] = []
    for r in bank:
        p = proof[str(r["id"])]
        if p["decided"]:
            print(f"dropped {r['id']} {r['side']} T{r['turn']} AR{r['ar']} {r['network']['name']}: "
                  f"{p['decided']} whatever is played", file=sys.stderr)
            continue
        kept.append(dict(r, exact=dict(r.get("exact", {}), decided="")))
    write_jsonl(out, kept)
    print(f"{len(kept)} of {len(bank)} rows kept", file=sys.stderr)
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
    c.add_argument("--continue-with", default=None, metavar="SIMS:K",
                   help="continuations played by a Gumbel root of the judge on both sides (default: its greedy play)")
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
    v = sub.add_parser("prove", help="exact values where the rest of the game is the mover's alone")
    v.add_argument("--input", "--bank", nargs="+", required=True, dest="input")
    v.add_argument("--deals", type=int, default=8, help="worlds with the mover's unseen cards redealt")
    v.add_argument("--budget", type=int, default=200_000, help="nodes per move per world")
    v.add_argument("--seed", type=int, default=13)
    v.add_argument("--part", default="1/1")
    v.add_argument("--out", required=True, help="one JSON row per bank row (.jsonl or .jsonl.gz)")
    t = sub.add_parser("settle", help="the bank without the rows prove finds decided whatever is played")
    t.add_argument("--bank", required=True)
    t.add_argument("--prove", nargs="+", required=True, help="prove's rows (every part)")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd in ("prove", "confirm", "score"):
        require_fresh()
    if a.cmd == "settle":
        return settle(a.bank, a.prove, a.out)
    if a.cmd == "prove":
        return prove(a.input, a.deals, a.budget, a.seed, a.part, a.out)
    if a.cmd == "select":
        return select(a.validation, a.min_gap, a.min_z, a.out)
    if a.cmd == "confirm":
        return confirm(a.input, a.model, a.pairs, a.seed, a.part, a.out, a.continue_with)
    if a.cmd == "build":
        return build(a.confirm, a.min_gap, a.min_z, a.out, a.md)
    return score(a.bank, a.player, a.judge, a.pairs, a.seed, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
