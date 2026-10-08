#!/usr/bin/env python3
"""The search disagreement bank: where Gumbel search departs from the raw network, and why.

An offline diagnostic. Nothing here changes how a player or a training run behaves; the Gumbel
root is driven through its ordinary agent (`gumbel:` spec), and only reads its per-candidate
statistics (`GumbelRoot.last_stats`).

Stages (each resumable, each writing JSONL.gz rows of schema `SCHEMA`):

* `select` -- (local) the bank: a stratified random sample of the annotated positions, most of it
  where some searcher departs from the raw network and the rest an agreement control. Strata are
  (agreement, decision group, era, departure size), and every row carries `weight` = positions in
  its stratum / positions kept from it, so population figures (regret per game, the share of regret
  in the tail) reweight the bank back to the decisions the raw network meets.
* `reference` -- each bank position solved by a larger Gumbel root (`--ref-k`, default 16, so the
  reference can choose a move outside every deployed searcher's candidates) at `--ref-sims`, with
  `--ref-seeds` independent runs; then the moves that matter (raw, each searcher's, every reference
  run's, the raw network's top `--top`) each valued by an equal-budget search (`--value-sims`) in
  the same `--worlds` redealt worlds, and by the bare critic (one evaluation) in those worlds, so
  the difference between two moves is taken world by world.
* `oracle` -- for bank rows carrying a `target` move (the reference best) that a deployed searcher
  never considered, that searcher run again with the target forced into its candidates in place of
  its least probable one (the set's size unchanged), `--seeds` times: does search pick the move once
  it is shown it? Diagnostic only (`GumbelRoot.choose(candidates=...)`).
* `playouts` -- for bank rows carrying `pmoves`, paired playouts of each move by the raw network
  playing both sides (ai/eval/paired_playouts.py: the mover's unseen cards redealt per pair, the dice
  shared): an outcome-based check on the search-valued regret, independent of the value head.
* `report` -- (local) the tables: reference recall against raw top-k, the regret distribution and its
  concentration, each method's agreement with the reference, breakdowns, the worst raw decisions,
  the candidate/ranking decomposition, the oracle and the playout check (`analysis()`).
* `event-select` -- (local) the event-decision bank, a second bank from the same annotated positions:
  every action-round play-mode decision where the event is legal beside another play, at most
  `--cap` per (card, side), weighted back to its group's population. Each row's `pmoves` are the
  event and the raw network's most probable alternative; on an opponent's card the event is the
  event-first play, the alternative the most probable Ops-first mode, and Space is added, the one
  play that keeps the opponent's event from firing. The `playouts` stage plays them.
* `event-report` -- (local) per card and side: how often the decision comes up, how often raw
  events, the event's playout lead over the alternative, raw's split-sample regret and its loss a
  game split into missed events, wrong events and other departures, beside the model's and the
  human corpus's action-round event rates (event_play_census dumps).
* `annotate` -- the raw network plays itself (`--temperature`, default 0.1, the tournaments'
  setting); one decision in `--sample` with two or more legal moves is kept, and each is put to the
  raw network and to every Gumbel configuration in `--budgets` (default the three measured in
  research/log/E7_gumbel_headroom.md: k=4 @16, k=4 @64, k=8 @256), recording each one's choice,
  its candidates, their evaluations and mean values, and when each was dropped. `--part k/N` takes
  a disjoint share of the games, so CI runners split the work.

A row (`SCHEMA` 1):

    id          sha1 of the position token, 16 hex
    model       the checkpoint's file name; the run's sidecar <out>.meta.json holds its sha256
    game        {"seed", "index"}: TsVectorizedEnv base seed and the game's slot; "decision" its
                decision count in the game
    side, turn, ar, phase, decision_type, card   (card: the card the decision is about, or 0)
    legal       legal flat actions, ascending
    logits      the raw network's logit of each legal action, same order
    raw         the raw network's argmax
    played      the action self-play took (sampled at the temperature)
    methods     {label: {"spec", "choice", "candidates", "n", "q", "dropped", "value"}}
                (n / q / dropped keyed by action, as strings in JSON; q is the mover's mean value
                of the candidate's position, value the network's value of the position for the mover)
    names       a readable name for every action that appears above
    pos         the workbench's pos= token (tools/scripts/event_play_census.position_token)

    PYTHONPATH=.:build/release python tools/search_bank.py annotate --model swa.pt \\
        --games 1100 --part 1/20 --out part1.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder

SCHEMA = 1
#: label -> (simulations, k): the configurations of research/log/E7_gumbel_headroom.md
DEFAULT_BUDGETS = ("g16=16:4", "g64=64:4", "g256=256:8")


def parse_budgets(specs: Sequence[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for s in specs:
        label, rest = s.split("=", 1)
        sims, k = (int(x) for x in rest.split(":"))
        out.append((label, sims, k))
    return out


def row_id(pos: str) -> str:
    return hashlib.sha1(pos.encode()).hexdigest()[:16]


def mover_of(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def decision_card(st: ts.GameState) -> int:
    """The card a decision is about: the one resolving, else the one being played for Ops, else 0
    (a card choice is about the card chosen, which the action names)."""
    ctx = st.ctx()
    return int(ctx.resolving_card) or int(ctx.pending_op_card)


def legal_actions(st: ts.GameState) -> List[int]:
    return [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]


def raw_rank(logits: Sequence[float], legal: Sequence[int], action: int) -> int:
    """1-based rank of `action` among the legal moves by raw logit (ties to the lower action id,
    as a stable sort of the legal list gives)."""
    order = sorted(range(len(legal)), key=lambda i: -logits[i])
    return 1 + [legal[i] for i in order].index(action)


def softmax(logits: Sequence[float]) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = np.exp(z - z.max())
    return z / z.sum()


class Sampler:
    """Keeps one decision in `every` with two or more legal moves, from one game."""

    def __init__(self, every: int, seed: int) -> None:
        self.every = every
        self.rng = random.Random(seed)
        self.decision = 0
        self.kept: List[Tuple[int, ts.GameState, int]] = []

    def observe(self, st: ts.GameState, a: int) -> None:
        self.decision += 1
        if len(legal_actions(st)) >= 2 and self.rng.randrange(self.every) == 0:
            self.kept.append((self.decision, st, a))


def _key(d: Dict[int, Any]) -> Dict[str, Any]:
    return {str(a): (round(v, 5) if isinstance(v, float) else v) for a, v in d.items()}


def annotate_states(states: Sequence[ts.GameState], logits_fn: Any, features: int,
                    searchers: Sequence[Tuple[str, str, Any]], seed: int) -> List[Dict[str, Any]]:
    """The raw network's and every searcher's answer at each position (positions of one game)."""
    from tools.scripts.event_play_census import position_token

    obs = np.stack([np.asarray(ts.extract_observation_features(s, mover_of(s), features), dtype=np.float32)
                    for s in states])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
    lg = np.asarray(logits_fn(obs, masks), dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    for i, st in enumerate(states):
        legal = legal_actions(st)
        ctx = st.ctx()
        rows.append({
            "schema": SCHEMA, "side": "US" if mover_of(st) == ts.Player.US else "USSR",
            "turn": int(st.turn), "ar": int(st.action_round), "phase": str(st.current_phase).split(".")[-1],
            "decision_type": str(ctx.decision_type).split(".")[-1], "card": decision_card(st),
            "legal": legal, "logits": [round(float(lg[i, a]), 4) for a in legal],
            "raw": legal[int(np.argmax([lg[i, a] for a in legal]))],
            "methods": {}, "pos": position_token(st),
        })
    for label, spec, agent in searchers:
        agent.reseed(seed)
        picks = agent.select_actions_batch(list(states))
        stats = agent.mcts._gumbel.last_stats
        if len(stats) != len(states):
            raise RuntimeError(f"{label}: {len(stats)} search records for {len(states)} positions "
                               "(a position was not searched)")
        for r, a, s in zip(rows, picks, stats):
            r["methods"][label] = {"spec": spec, "choice": int(a), "candidates": list(s["candidates"]),
                                   "n": _key(s["n"]), "q": _key(s["q"]), "dropped": _key(s["dropped"]),
                                   "value": round(float(s["value"]), 5)}
    for r, st in zip(rows, states):
        seen = {r["raw"]} | {m["choice"] for m in r["methods"].values()}
        for m in r["methods"].values():
            seen |= set(m["candidates"])
        r["names"] = {str(a): ActionEncoder.get_action_name(st, a) for a in sorted(seen)}
        r["id"] = row_id(r["pos"])
    return rows


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def annotate(a: argparse.Namespace) -> int:
    import torch

    from tools.lib.player_agent import load_agent
    from tools.scripts.event_play_census import load_policy, selfplay

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))))
    k, n = (int(x) for x in a.part.split("/"))
    # Game g (0-based, over all parts) is played from seed base + g; this part takes g = k-1 mod n.
    mine = [g for g in range(a.games) if g % n == k - 1]
    ledger = a.out + ".games.jsonl"
    done = set()
    if os.path.exists(ledger):
        if not a.resume:
            raise SystemExit(f"{a.out} exists: pass --resume to continue it, or remove it")
        done = {json.loads(line)["game"] for line in open(ledger)}
    todo = [g for g in mine if g not in done]
    budgets = parse_budgets(a.budgets)
    json.dump({"schema": SCHEMA, "model": os.path.basename(a.model), "model_sha256": _sha256(a.model),
               "temperature": a.temperature, "sample": a.sample, "seed": a.seed, "games": a.games,
               "part": a.part, "budgets": {lb: {"simulations": s, "k": kk} for lb, s, kk in budgets}},
              open(a.out + ".meta.json", "w"), indent=1)
    logits_fn, features = load_policy(a.model)
    searchers = []
    for label, sims, kk in budgets:
        spec = f"gumbel:{a.model}:{sims}:{kk}"
        searchers.append((label, f"gumbel:{sims}:{kk}", load_agent(spec, device="cpu")))
    t0 = time.time()
    total = 0
    for gi, g in enumerate(todo):
        trackers = selfplay(logits_fn, features, 1, a.seed + g, 1, a.temperature,
                            lambda: Sampler(a.sample, a.seed + g))
        kept = trackers[0].kept
        rows: List[Dict[str, Any]] = []
        for lo in range(0, len(kept), a.chunk):
            part = kept[lo:lo + a.chunk]
            got = annotate_states([s for _, s, _ in part], logits_fn, features, searchers, seed=a.seed + g)
            for (dec, _s, played), r in zip(part, got):
                r.update({"model": os.path.basename(a.model), "game": {"seed": a.seed + g, "index": g},
                          "decision": dec, "played": played})
                rows.append(r)
        with gzip.open(a.out, "at") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        with open(ledger, "a") as f:
            f.write(json.dumps({"game": g, "rows": len(rows)}) + "\n")
        total += len(rows)
        print(f"[{gi + 1}/{len(todo)}] game {g}: {len(rows)} positions ({total} this run, "
              f"{time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return 0


def decision_group(r: Dict[str, Any]) -> str:
    """Coarse decision class for stratification: the card decisions apart, then the rest by type."""
    dt = r["decision_type"]
    if dt == "SELECT_CARD":
        return "headline" if r["phase"] == "HEADLINE" else "card"
    if dt in ("SELECT_PLAY_MODE", "SELECT_OP_MODE", "POINT_NODE"):
        return dt.lower()
    return "event_choice"


def era(turn: int) -> str:
    return "early" if turn <= 3 else ("mid" if turn <= 7 else "late")


def departure(r: Dict[str, Any], label: Optional[str] = None) -> float:
    """How much better a searcher (default the last, i.e. the largest, budget) thinks its move is than
    the raw argmax: q(choice) - q(raw), both from the searcher's own record (0 where it played the
    raw move or lacks a value)."""
    m = r["methods"][label or list(r["methods"])[-1]]
    qc, qr = m["q"].get(str(m["choice"])), m["q"].get(str(r["raw"]))
    if m["choice"] == r["raw"] or qc is None or qr is None:
        return 0.0
    return float(qc) - float(qr)


def stratum(r: Dict[str, Any]) -> Tuple[str, str, str, str]:
    choices = {m["choice"] for m in r["methods"].values()}
    agree = choices == {r["raw"]}
    d = departure(r)
    size = "-" if agree else ("small" if d < 0.02 else ("medium" if d < 0.08 else "large"))
    return ("agree" if agree else "disagree", decision_group(r), era(int(r["turn"])), size)


def select(rows: Sequence[Dict[str, Any]], size: int, control: int, seed: int) -> List[Dict[str, Any]]:
    """`size - control` disagreement rows spread as evenly as availability allows over their strata
    (water-filling), and `control` agreement rows the same way; each row weighted by its stratum's
    population over its sample."""
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        groups.setdefault(stratum(r), []).append(r)
    out: List[Dict[str, Any]] = []
    for kind, budget in (("disagree", size - control), ("agree", control)):
        keys = sorted(k for k in groups if k[0] == kind)
        alloc = {k: 0 for k in keys}
        left = budget
        open_ = [k for k in keys if groups[k]]
        while left > 0 and open_:
            share = max(1, left // len(open_))
            for k in list(open_):
                take = min(share, len(groups[k]) - alloc[k], left)
                alloc[k] += take
                left -= take
                if alloc[k] >= len(groups[k]):
                    open_.remove(k)
                if left == 0:
                    break
        for k in keys:
            if alloc[k] == 0:
                continue
            pick = rng.sample(groups[k], alloc[k])
            w = len(groups[k]) / alloc[k]
            for r in pick:
                out.append(dict(r, stratum="/".join(k), weight=round(w, 4)))
    return out


def moves_to_value(r: Dict[str, Any], ref_choices: Sequence[int], top: int) -> List[int]:
    """The moves a reference values at a position: the raw argmax, every searcher's choice, every
    reference run's, and the raw network's `top` most probable, most probable first."""
    order = sorted(range(len(r["legal"])), key=lambda i: -r["logits"][i])
    want = [r["legal"][i] for i in order[:top]] + [r["raw"]]
    want += [m["choice"] for m in r["methods"].values()] + list(ref_choices)
    rank = {r["legal"][i]: k for k, i in enumerate(order)}
    return sorted(dict.fromkeys(int(a) for a in want), key=lambda a: rank[a])


def reference(a: argparse.Namespace) -> int:
    import torch

    from tools.lib.player_agent import load_agent
    from tools.scripts.bank_clarity import value_moves_per_world
    from tools.scripts.event_play_census import state_from_token

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))))
    k, n = (int(x) for x in a.part.split("/"))
    rows = sorted(read_rows([a.bank]), key=lambda r: r["id"])
    rows = [r for i, r in enumerate(rows) if i % n == k - 1]
    done = set()
    if os.path.exists(a.out):
        if not a.resume:
            raise SystemExit(f"{a.out} exists: pass --resume to continue it, or remove it")
        with gzip.open(a.out, "rt") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [r for r in rows if r["id"] not in done]
    json.dump({"schema": SCHEMA, "stage": "reference", "model": os.path.basename(a.model),
               "model_sha256": _sha256(a.model), "ref": {"simulations": a.ref_sims, "k": a.ref_k,
                                                          "seeds": a.ref_seeds},
               "values": {"simulations": a.value_sims, "worlds": a.worlds, "top": a.top},
               "critic": {"simulations": a.critic_sims}}, open(a.out + ".meta.json", "w"), indent=1)
    agent: Any = load_agent(f"gumbel:{a.model}:{a.ref_sims}:{a.ref_k}", device="cpu")
    t0 = time.time()
    for lo in range(0, len(todo), a.chunk):
        batch = todo[lo:lo + a.chunk]
        states = [state_from_token(r["pos"]) for r in batch]
        runs: List[List[Dict[str, Any]]] = [[] for _ in batch]
        for j in range(a.ref_seeds):
            seed = (int(batch[0]["id"], 16) + 7919 * j) % (1 << 31)
            agent.reseed(seed)
            picks = agent.select_actions_batch(states)
            for i, (c, st) in enumerate(zip(picks, agent.mcts._gumbel.last_stats)):
                runs[i].append({"seed": seed, "choice": int(c), "candidates": list(st["candidates"]),
                                "n": _key(st["n"]), "q": _key(st["q"])})
        moves = [moves_to_value(r, [x["choice"] for x in rr], a.top) for r, rr in zip(batch, runs)]
        seed = int(batch[0]["id"], 16) % (1 << 31)
        vals = value_moves_per_world(agent.mcts, states, moves, a.value_sims, a.worlds, seed)
        crit = value_moves_per_world(agent.mcts, states, moves, a.critic_sims, a.worlds, seed)
        with gzip.open(a.out, "at") as f:
            for r, st, rr, ms, v, c in zip(batch, states, runs, moves, vals, crit):
                choices = [x["choice"] for x in rr]
                best = max(set(choices), key=lambda x: (choices.count(x), -choices.index(x)))
                out = {"schema": SCHEMA, "id": r["id"], "moves": ms,
                       "names": {str(m): ActionEncoder.get_action_name(st, m) for m in ms},
                       "reference": {"runs": rr, "action": best, "agreement": choices.count(best) / len(choices)},
                       "values": {str(m): [None if x is None else round(x, 5) for x in v[m]] for m in ms},
                       "critic": {str(m): [None if x is None else round(x, 5) for x in c[m]] for m in ms}}
                f.write(json.dumps(out, separators=(",", ":")) + "\n")
        print(f"{lo + len(batch)}/{len(todo)} positions, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return 0


def _my_rows(a: argparse.Namespace) -> List[Dict[str, Any]]:
    k, n = (int(x) for x in a.part.split("/"))
    rows = sorted(read_rows([a.bank]), key=lambda r: r["id"])
    return [r for i, r in enumerate(rows) if i % n == k - 1]


def oracle(a: argparse.Namespace) -> int:
    import torch

    from tools.lib.player_agent import load_agent
    from tools.scripts.event_play_census import state_from_token

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))))
    rows = _my_rows(a)
    budgets = parse_budgets(a.budgets)
    with gzip.open(a.out, "wt") as f:
        for label, sims, kk in budgets:
            agent: Any = load_agent(f"gumbel:{a.model}:{sims}:{kk}", device="cpu")
            todo = []
            for r in rows:
                order = sorted(range(len(r["legal"])), key=lambda i: -r["logits"][i])
                top = [r["legal"][i] for i in order[:kk]]
                if r["target"] in top or len(top) < 2:
                    continue
                todo.append((r, top[:-1] + [r["target"]]))
            for lo in range(0, len(todo), a.chunk):
                batch = todo[lo:lo + a.chunk]
                states = [state_from_token(r["pos"]) for r, _ in batch]
                picks: List[List[int]] = [[] for _ in batch]
                for j in range(a.seeds):
                    agent.reseed(1000 + j)
                    got = agent.mcts._gumbel.choose(states, candidates=[c for _, c in batch])
                    for i, c in enumerate(got):
                        picks[i].append(int(c))
                for (r, cands), pk in zip(batch, picks):
                    f.write(json.dumps({"schema": SCHEMA, "id": r["id"], "label": label, "target": r["target"],
                                        "candidates": cands, "choices": pk}, separators=(",", ":")) + "\n")
            print(f"{label}: {len(todo)} positions", file=sys.stderr, flush=True)
    return 0


def playouts(a: argparse.Namespace) -> int:
    from ai.eval.paired_playouts import compare
    from tools.lib.player_agent import OnnxAgent
    from tools.scripts.event_play_census import state_from_token

    rows = _my_rows(a)
    onnx = a.onnx or (os.path.splitext(a.model)[0] + ".onnx")
    agent = OnnxAgent(onnx)

    def act(obs: Any, masks: Any) -> Any:
        return agent.act_batch(obs, masks, 0.0, True)

    k = int(a.part.split("/")[0])
    t0 = time.time()
    with gzip.open(a.out, "wt") as f:
        for lo in range(0, len(rows), a.chunk):
            batch = rows[lo:lo + a.chunk]
            positions = [(state_from_token(r["pos"]), [int(m) for m in r["pmoves"]]) for r in batch]
            scores = compare(positions, act, a.pairs, a.seed + 100_003 * k + lo)
            for r, sc in zip(batch, scores):
                f.write(json.dumps({"schema": SCHEMA, "id": r["id"], "pairs": a.pairs,
                                    "scores": {str(m): v for m, v in sc.items()}}, separators=(",", ":")) + "\n")
            print(f"{lo + len(batch)}/{len(rows)} positions, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return 0


EVENT = ActionEncoder.PLAY_MODE_OFFSET
SPACE = ActionEncoder.PLAY_MODE_OFFSET + 1
OPS_MODES = (EVENT + 2, EVENT + 3, EVENT + 4)


def card_relation(card_side: str, mover: str) -> str:
    """Whose event a card is, seen from the side playing it: "own", "opponent" or "neutral"."""
    s = card_side.lower()
    return "neutral" if s == "neutral" else ("own" if s == mover.lower() else "opponent")


def event_moves(r: Dict[str, Any], relation: str) -> List[int]:
    """The moves an event-decision row plays out, the event first: the event (on an opponent's card,
    event-first), then the raw network's most probable alternative (on an opponent's card, its most
    probable Ops-first mode), and on an opponent's card Space as well when it is legal -- the only
    play that keeps the opponent's event from firing. The raw argmax is always among them."""
    lg = dict(zip(r["legal"], r["logits"]))
    alts = [a for a in (OPS_MODES if relation == "opponent" else (SPACE,) + OPS_MODES) if a in lg]
    out = [EVENT, max(alts, key=lambda a: lg[a])]
    if relation == "opponent" and SPACE in lg:
        out.append(SPACE)
    return out


def is_event_decision(r: Dict[str, Any]) -> bool:
    """An action-round play-mode decision whose event is legal beside some other play. Headlines are
    left out: they have no Ops alternative, the choice there is which event to fire."""
    if r["decision_type"] != "SELECT_PLAY_MODE" or r["phase"] != "ACTION_ROUND" or EVENT not in r["legal"]:
        return False
    return any(a in r["legal"] for a in (SPACE,) + OPS_MODES)


def event_select(rows: Iterable[Dict[str, Any]], cards: Dict[int, Dict[str, Any]], cap: int,
                 seed: int) -> List[Dict[str, Any]]:
    """The event-decision bank: every event decision in `rows` grouped by (card, side), at most `cap`
    drawn at random from each, each weighted by its group's population over its sample; each row
    carries its card's `relation` to the mover and the `pmoves` the playouts stage plays."""
    rng = random.Random(seed)
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if is_event_decision(r):
            groups.setdefault((int(r["card"]), str(r["side"])), []).append(r)
    out: List[Dict[str, Any]] = []
    for card, side in sorted(groups):
        g = sorted(groups[(card, side)], key=lambda r: r["id"])
        pick = rng.sample(g, min(cap, len(g)))
        rel = card_relation(str(cards[card]["side"]), side)
        for r in pick:
            out.append(dict(r, stratum=f"{card}/{side}", relation=rel, weight=round(len(g) / len(pick), 4),
                            pmoves=event_moves(r, rel)))
    return out


def iter_annotated(paths: Sequence[str], keep: Callable[[Dict[str, Any]], bool]) -> Iterator[Dict[str, Any]]:
    """Annotated rows one at a time, those `keep` accepts (the annotated files are too large to hold)."""
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("schema") != SCHEMA:
                    raise ValueError(f"{p}: schema {r.get('schema')} is not {SCHEMA}")
                if keep(r):
                    yield r


def event_verdict(b: Dict[str, Any], scores: Dict[str, List[float]]) -> Dict[str, Any]:
    """One event-decision row's playout verdict, in points of win probability for the mover.

    `gain` is the event's lead over the alternative on all pairs, `gain_se` its paired SE. The best
    move is picked on the even pairs and raw's `regret` measured on the odd ones, so choosing the
    largest of several noisy means does not inflate it; `kind` says whether that regret is a missed
    event, a wrong event, or another departure (on an opponent's card, Space against Ops)."""
    sc = {int(m): v for m, v in scores.items()}
    alt, raw = int(b["pmoves"][1]), int(b["raw"])
    d = [100 * (x - y) for x, y in zip(sc[EVENT], sc[alt])]
    n = len(d)
    gain = sum(d) / n
    se = math.sqrt(sum((x - gain) ** 2 for x in d) / (n - 1) / n) if n > 1 else math.nan
    pick = max(sc, key=lambda m: sum(sc[m][0::2]))
    odd = [100 * (x - y) for x, y in zip(sc[pick][1::2], sc[raw][1::2])] if raw in sc else []
    regret = 0.0 if pick == raw or not odd else sum(odd) / len(odd)
    kind = ("none" if pick == raw else "missed_event" if pick == EVENT else
            "wrong_event" if raw == EVENT else "other")
    return {"gain": gain, "gain_se": se, "regret": regret, "kind": kind, "raw_event": raw == EVENT,
            "pick": pick}


def ar_event_rates(holdings: Sequence[Any]) -> Dict[Tuple[int, str], Tuple[int, int]]:
    """(card, side) -> (holdings spent in an action round, of them evented), from event_play_census
    holdings (only the owner's for a side's card, either side's for a neutral one)."""
    out: Dict[Tuple[int, str], List[int]] = collections.defaultdict(lambda: [0, 0])
    for h in holdings:
        if h.legal and h.outcome in ("event", "ops", "space"):
            k = (int(h.card), "US" if h.side == int(ts.Player.US) else "USSR")
            out[k][0] += 1
            out[k][1] += h.outcome == "event"
    return {k: (v[0], v[1]) for k, v in out.items()}


Verdicts = Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]


def _wmean_se(rs: Verdicts, f: Callable[[Dict[str, Any]], float]) -> Tuple[float, float]:
    """Weighted mean over positions and its SE between positions (each one's playout noise is in it)."""
    w = [b["weight"] for b, _ in rs]
    x = [float(f(v)) for _, v in rs]
    tw = sum(w)
    mu = sum(wi * xi for wi, xi in zip(w, x)) / tw
    if len(x) < 2:
        return mu, math.nan
    var = sum((wi / tw) ** 2 * (xi - mu) ** 2 for wi, xi in zip(w, x)) * len(x) / (len(x) - 1)
    return mu, math.sqrt(var)


def event_summary(rs: Verdicts, per_game: float) -> Dict[str, float]:
    """A group's figures; `per_game` turns a row's weight into decisions a game."""
    pop = sum(b["weight"] for b, _ in rs)

    def share(f: Callable[[Dict[str, Any]], float]) -> float:
        return sum(b["weight"] * f(v) for b, v in rs) / pop

    g, gse = _wmean_se(rs, lambda v: v["gain"])
    rg, rse = _wmean_se(rs, lambda v: v["regret"])
    games = pop * per_game
    return {"n": len(rs), "per_game": games, "raw_event": share(lambda v: v["raw_event"]),
            "best_event": share(lambda v: v["pick"] == EVENT), "gain": g, "gain_se": gse,
            "regret": rg, "regret_se": rse, "loss": rg * games, "loss_se": rse * games,
            **{k: share(lambda v, k=k: v["regret"] * (v["kind"] == k)) * games
               for k in ("missed_event", "wrong_event", "other")}}


def event_analysis(bank_rows: Sequence[Dict[str, Any]], pl_rows: Sequence[Dict[str, Any]],
                   cards: Dict[int, Dict[str, Any]], per_game: float,
                   human: Optional[Dict[Tuple[int, str], Tuple[int, int]]] = None,
                   bot: Optional[Dict[Tuple[int, str], Tuple[int, int]]] = None) -> Tuple[str, Dict[str, Any]]:
    """The event-decision tables. `per_game` = annotate's --sample over its game count."""
    pl = {r["id"]: r["scores"] for r in pl_rows}
    rows = [(b, event_verdict(b, pl[b["id"]])) for b in bank_rows if b["id"] in pl]
    out = [f"# Event decisions: {len(rows)} positions", "",
           "Action-round play-mode decisions where the event is legal beside another play, from the raw network's "
           "self-play: up to a fixed number per (card, side), reweighted to how often each comes up. At each, the "
           "event and the raw network's most probable alternative (on an opponent's card: event-first against its "
           "most probable Ops-first mode, and Space) are played out in pairs by the raw network. Points are win "
           "probability for the side deciding. **Event lead**: the event over the alternative, all pairs. "
           "**Regret**: the best move's lead over raw's, the best picked on the even pairs and measured on the odd "
           "ones. **Loss a game**: regret times how often the decision comes up, split into missed events, wrong "
           "events and other departures. The playouts are the raw network's continuation, a lower bound on a move "
           "whose payoff needs a follow-up it does not find.", ""]

    def table(head: Sequence[str], body: Sequence[Sequence[Any]]) -> None:
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join([":---"] + ["---:"] * (len(head) - 1)) + "|")
        out.extend("| " + " | ".join(str(x) for x in r) + " |" for r in body)
        out.append("")

    def fmt(s: Dict[str, float]) -> List[str]:
        return [str(s["n"]), f"{s['per_game']:.2f}", f"{100 * s['raw_event']:.0f}%", f"{100 * s['best_event']:.0f}%",
                f"{s['gain']:+.2f} ± {s['gain_se']:.2f}", f"{s['regret']:+.2f} ± {s['regret_se']:.2f}",
                f"{s['loss']:+.2f} ± {s['loss_se']:.2f}", f"{s['missed_event']:+.2f}", f"{s['wrong_event']:+.2f}",
                f"{s['other']:+.2f}"]

    def grouped(key: Callable[[Dict[str, Any]], Any]) -> Dict[Any, List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
        g: Dict[Any, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = collections.defaultdict(list)
        for b, v in rows:
            g[key(b)].append((b, v))
        return dict(sorted(g.items()))

    head = ["n", "a game", "raw events", "best is event", "event lead (pts)", "regret (pts)", "loss a game (pts)",
            "missed event", "wrong event", "other"]
    tot = event_summary(rows, per_game)
    num: Dict[str, Any] = {"positions": len(rows), "all": tot}
    out += [f"Over all of them: {tot['per_game']:.1f} such decisions a game; raw events {100 * tot['raw_event']:.0f}% "
            f"of them, the playouts' best is the event in {100 * tot['best_event']:.0f}%. Raw's loss: "
            f"**{tot['loss']:+.1f} ± {tot['loss_se']:.1f} points a game** (missed events {tot['missed_event']:+.1f}, "
            f"wrong events {tot['wrong_event']:+.1f}, other {tot['other']:+.1f}).", ""]
    for title, key in (("By side and whose event the card is", lambda b: f"{b['side']} / {b['relation']}"),
                       ("By whose event and era", lambda b: f"{b['relation']} / {era(int(b['turn']))}")):
        g = grouped(key)
        num[title] = {k: event_summary(v, per_game) for k, v in g.items()}
        out += [f"## {title}", ""]
        table(["group"] + head, [[k] + fmt(num[title][k]) for k in g])

    by_card = grouped(lambda b: (int(b["card"]), str(b["side"])))
    per_card = {k: event_summary(v, per_game) for k, v in by_card.items()}
    num["by_card"] = {f"{cards[c]['name']}/{s}": x for (c, s), x in per_card.items()}

    def rate(src: Optional[Dict[Tuple[int, str], Tuple[int, int]]], k: Tuple[int, str]) -> str:
        if src is None or src.get(k, (0, 0))[0] == 0:
            return "—"
        n, e = src[k]
        return f"{100 * e / n:.0f}% ({n:,})"

    card_head = ["card", "side", "whose", "n", "a game", "raw events", "bot AR event rate", "human AR event rate",
                 "event lead (pts)", "regret (pts)", "loss a game (pts)", "missed", "wrong", "other"]

    def card_row(k: Tuple[int, str]) -> List[str]:
        f = fmt(per_card[k])
        return [str(cards[k[0]]["name"]), k[1], card_relation(str(cards[k[0]]["side"]), k[1]), f[0], f[1], f[2],
                rate(bot, k), rate(human, k)] + f[4:]

    out += ["## Cards by raw's loss a game", "",
            "The AR event rates are per holding spent in an action round (evented / evented + Ops + Space) from "
            "tools/scripts/event_play_census.py: the same model's greedy self-play, and the human corpus. Neither "
            "counts an opponent's card, whose event fires unless it goes to Space.", ""]
    table(card_head, [card_row(k) for k in sorted(per_card, key=lambda k: -per_card[k]["loss"])[:30]])
    for title, key in (("missed events", "missed_event"), ("wrong events", "wrong_event")):
        out += [f"## Largest {title} (loss a game)", ""]
        table(card_head, [card_row(k) for k in sorted(per_card, key=lambda k: -per_card[k][key])[:15]
                          if per_card[k][key] > 0])
    clear = []
    for k, rs in by_card.items():
        better = [v for _, v in rs if v["gain"] > 2 * v["gain_se"]]
        worse = [v for _, v in rs if v["gain"] < -2 * v["gain_se"]]
        miss = sum(not v["raw_event"] for v in better)
        wrong = sum(v["raw_event"] for v in worse)
        if miss + wrong:
            clear.append((k, len(rs), len(better), miss, len(worse), wrong))
    clear.sort(key=lambda t: (-(t[3] + t[5]), t[0]))
    out += ["## Positions where the playouts are clear (event lead beyond 2 SE either way)", "",
            "Unweighted counts of bank positions, cards with at least one clear raw error.", ""]
    table(["card", "side", "n", "event clearly better", "raw did not event", "event clearly worse", "raw evented"],
          [[cards[k[0]]["name"], k[1], n, b, m, w, wr] for k, n, b, m, w, wr in clear[:30]])
    out += ["## Every card", ""]
    table(card_head, [card_row(k) for k in sorted(per_card, key=lambda k: (str(cards[k[0]]["name"]), k[1]))])
    return "\n".join(out), num


def _paired(a: Sequence[Optional[float]], b: Sequence[Optional[float]], idx: Sequence[int]) -> Tuple[float, float, int]:
    """Mean and SE of a - b over the worlds in `idx` where both are defined."""
    d = [a[i] - b[i] for i in idx if a[i] is not None and b[i] is not None]   # type: ignore[operator]
    if not d:
        return math.nan, math.nan, 0
    mu = sum(d) / len(d)
    se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else math.nan
    return mu, se, len(d)


def _mean(v: Sequence[Optional[float]], idx: Sequence[int]) -> float:
    xs = [v[i] for i in idx if v[i] is not None]
    return sum(xs) / len(xs) if xs else -math.inf     # type: ignore[arg-type]


def solve(bank: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """One position's verdict from its equal-budget move values.

    The best move is chosen on the even worlds and every move's regret measured on the odd ones,
    so choosing the best of several noisy values does not inflate the regret (the winner's curse);
    `best` and its `lead` over the runner-up (paired, all worlds) say how sure the verdict is."""
    vals = {int(m): v for m, v in ref["values"].items()}
    nw = len(next(iter(vals.values())))
    even, odd, every = list(range(0, nw, 2)), list(range(1, nw, 2)), list(range(nw))
    pick = max(vals, key=lambda m: _mean(vals[m], even))
    best = max(vals, key=lambda m: _mean(vals[m], every))
    others = [m for m in vals if m != best]
    lead, lead_se = (math.inf, 0.0) if not others else min(
        (_paired(vals[best], vals[m], every)[:2] for m in others), key=lambda t: t[0])
    regret: Dict[str, Tuple[float, float]] = {}
    who = {"raw": bank["raw"], **{lb: m["choice"] for lb, m in bank["methods"].items()},
           "ref": ref["reference"]["action"]}
    for lb, mv in who.items():
        mu, se, _ = _paired(vals[pick], vals[mv], odd) if mv in vals else (math.nan, math.nan, 0)
        regret[lb] = (0.0 if mv == pick else mu, 0.0 if mv == pick else se)
    rank = raw_rank(bank["logits"], bank["legal"], best)
    sure = lead_se > 0 and lead >= 2 * lead_se if others else True
    agree = ref["reference"]["action"] == best
    conf = "high" if (sure and agree and ref["reference"]["agreement"] >= 2 / 3) else \
        ("medium" if (sure or agree) else "low")
    crit = {int(m): _mean(v, every) for m, v in ref["critic"].items()}
    return {"best": best, "pick": pick, "lead": lead, "lead_se": lead_se, "confidence": conf,
            "rank": rank, "regret": regret, "who": who,
            "critic_best": max(crit, key=lambda m: crit[m]) if crit else None,
            "p_best": float(softmax(bank["logits"])[bank["legal"].index(best)])}


def _wq(xs: Sequence[float], ws: Sequence[float], q: float) -> float:
    pairs = sorted(zip(xs, ws))
    tot = sum(ws)
    acc = 0.0
    for x, w in pairs:
        acc += w
        if acc >= q * tot:
            return x
    return pairs[-1][0] if pairs else math.nan


def analysis(bank_rows: Sequence[Dict[str, Any]], ref_rows: Sequence[Dict[str, Any]],
             oracle_rows: Sequence[Dict[str, Any]] = (), playout_rows: Sequence[Dict[str, Any]] = (),
             cards: Optional[Dict[int, str]] = None,
             decisions_per_game: Optional[float] = None,
             playout_incl: Optional[Dict[str, float]] = None) -> Tuple[str, Dict[str, Any]]:
    """The report (Markdown) and its numbers."""
    refs = {r["id"]: r for r in ref_rows}
    rows = [(b, refs[b["id"]], solve(b, refs[b["id"]])) for b in bank_rows if b["id"] in refs]
    methods = ["raw"] + list(bank_rows[0]["methods"]) if bank_rows else ["raw"]
    cards = cards or {}
    out: List[str] = []
    num: Dict[str, Any] = {"positions": len(rows)}
    W = sum(b["weight"] for b, _, _ in rows)

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    def table(head: Sequence[str], body: Sequence[Sequence[Any]]) -> None:
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join(":---" if i == 0 else "---:" for i in range(len(head))) + "|")
        for line in body:
            out.append("| " + " | ".join(str(x) for x in line) + " |")
        out.append("")

    conf = collections.Counter(v["confidence"] for _, _, v in rows)
    out += [f"# Search disagreement bank: {len(rows)} positions", "",
            "Population-weighted (each bank row stands for `weight` raw-play decisions). Values and regret "
            "are in the value head's units, the mover's [-1, 1]: 0.02 is one point of win probability.", "",
            f"Confidence of the reference: high {conf['high']}, medium {conf['medium']}, low {conf['low']}.", ""]
    hi = [(b, r, v) for b, r, v in rows if v["confidence"] == "high"]

    # 1. Recall of the reference move by the raw policy's top k.
    out += ["## Reference-best recall by raw top-k (high-confidence positions)", ""]
    ks = [1, 2, 4, 8, 16]
    body = []
    for sub, name in ((hi, "all"), ([x for x in hi if x[2]["best"] != x[0]["raw"]], "raw wrong")):
        w = sum(b["weight"] for b, _, _ in sub) or 1.0
        rw = sum(b["weight"] * max(0.0, v["regret"]["raw"][0]) for b, _, v in sub) or 1.0
        line = [name, len(sub)] + [pct(sum(b["weight"] for b, _, v in sub if v["rank"] <= k) / w) for k in ks]
        line += [pct(sum(b["weight"] * max(0.0, v["regret"]["raw"][0]) for b, _, v in sub if v["rank"] > 4) / rw),
                 pct(sum(b["weight"] * max(0.0, v["regret"]["raw"][0]) for b, _, v in sub if v["rank"] > 8) / rw)]
        body.append(line)
        num[f"recall_{name}"] = {k: sum(b["weight"] for b, _, v in sub if v["rank"] <= k) / w for k in ks}
    table(["positions", "n"] + [f"top-{k}" for k in ks] + ["raw regret outside top-4", "outside top-8"], body)

    # 2. Agreement and regret per method.
    out += ["## Each method against the reference", ""]
    body = []
    for m in methods:
        sel = [(b, v) for b, _, v in rows if not math.isnan(v["regret"][m][0])]
        w = sum(b["weight"] for b, _ in sel) or 1.0
        agree_ref = sum(b["weight"] for b, v in sel if v["who"][m] == v["best"]) / w
        agree_raw = sum(b["weight"] for b, v in sel if v["who"][m] == b["raw"]) / w
        mean_reg = sum(b["weight"] * v["regret"][m][0] for b, v in sel) / w
        regs = [v["regret"][m][0] for _, v in sel]
        ws = [b["weight"] for b, _ in sel]
        body.append([m, pct(agree_ref), pct(agree_raw), f"{mean_reg:.4f}"] +
                    [f"{_wq(regs, ws, q):.4f}" for q in (0.5, 0.9, 0.95, 0.99)])
        num[f"mean_regret_{m}"] = mean_reg
    table(["method", "= reference", "= raw", "mean regret", "median", "p90", "p95", "p99"], body)

    if decisions_per_game:
        per_game = {m: num[f"mean_regret_{m}"] * decisions_per_game / 2 for m in methods}
        num["regret_per_game_points"] = per_game
        out += ["**Calibration.** Over the decisions a raw game makes "
                f"({decisions_per_game:.0f} with two or more legal moves), the measured regret adds up to "
                + ", ".join(f"{m} {100 * per_game[m]:.1f}" for m in methods)
                + " points of win probability a game. A tournament measured Gumbel k=8 @256 at about +9 "
                "points over the raw network; a total far from that says the regret measures search's opinion "
                "of itself more than strength.", ""]

    # 3. Concentration: the share of raw regret held by the worst positions.
    sel = sorted(((max(0.0, v["regret"]["raw"][0]), b["weight"]) for b, _, v in rows), reverse=True)
    tot_r = sum(r * w for r, w in sel) or 1.0
    tot_w = sum(w for _, w in sel) or 1.0
    out += ["## Concentration of raw regret (positive part)", ""]
    body = []
    acc_r = acc_w = 0.0
    marks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    mi = 0
    for r, w in sel:
        acc_r += r * w
        acc_w += w
        while mi < len(marks) and acc_w / tot_w >= marks[mi]:
            body.append([pct(marks[mi]), pct(acc_r / tot_r)])
            num[f"regret_share_top_{marks[mi]}"] = acc_r / tot_r
            mi += 1
    table(["worst share of decisions", "share of raw regret"], body)

    # 4. Breakdowns of raw regret.
    def breakdown(title: str, key: Any) -> None:
        g: Dict[Any, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
        for b, _, v in rows:
            g.setdefault(key(b, v), []).append((b, v))
        body = []
        for k_, sub in sorted(g.items(), key=lambda kv: str(kv[0])):
            w = sum(b["weight"] for b, _ in sub) or 1.0
            share = sum(b["weight"] for b, _ in sub) / W if W else 0.0
            body.append([k_, len(sub), pct(share)] +
                        [f"{sum(b['weight'] * v['regret'][m][0] for b, v in sub) / w:.4f}" for m in methods] +
                        [pct(sum(b["weight"] for b, v in sub if v["best"] != b["raw"]) / w)])
        out.extend([f"## Raw regret by {title}", ""])
        table([title, "n", "of decisions"] + [f"regret {m}" for m in methods] + ["raw ≠ best"], body)

    breakdown("turn", lambda b, v: int(b["turn"]))
    breakdown("action round", lambda b, v: int(b["ar"]))
    breakdown("decision", lambda b, v: decision_group(b))
    breakdown("raw rank of the best move", lambda b, v: min(v["rank"], 9) if v["rank"] < 9 else "9+")
    breakdown("confidence", lambda b, v: v["confidence"])
    breakdown("agreement", lambda b, v: b.get("stratum", "/").split("/")[0])
    breakdown("card", lambda b, v: cards.get(int(b["card"]), "-") if b["card"] else "-")

    # 5. Decomposition of raw's errors.
    out += ["## Why raw is wrong (high-confidence positions where raw is not the best)", ""]
    err = [(b, v) for b, _, v in hi if v["best"] != b["raw"]]
    w = sum(b["weight"] for b, _ in err) or 1.0
    rw = sum(b["weight"] * max(0.0, v["regret"]["raw"][0]) for b, v in err) or 1.0
    body = []
    for m in (methods[1:] if err else []):
        inside = [(b, v) for b, v in err if v["best"] in b["methods"][m]["candidates"]]
        wi = sum(b["weight"] for b, _ in inside) or 1.0
        body.append([m, pct(1 - sum(b["weight"] for b, _ in inside) / w),
                     pct(1 - sum(b["weight"] * max(0.0, v["regret"]["raw"][0]) for b, v in inside) / rw),
                     pct(sum(b["weight"] for b, v in inside if b["methods"][m]["choice"] == v["best"]) / wi),
                     pct(sum(b["weight"] for b, v in inside if v["critic_best"] == v["best"]) / wi)])
    table(["searcher", "best outside its candidates (errors)", "(regret)", "picks best when inside",
           "bare critic ranks best first when inside"], body)

    if oracle_rows:
        out += ["## Oracle candidates: the best move forced into the candidate set", ""]
        body = []
        for lb in sorted({o["label"] for o in oracle_rows}):
            sub = [o for o in oracle_rows if o["label"] == lb]
            hits = sum(c == o["target"] for o in sub for c in o["choices"])
            runs = sum(len(o["choices"]) for o in sub) or 1
            body.append([lb, len(sub), pct(hits / runs)])
            num[f"oracle_{lb}"] = hits / runs
        table(["searcher", "positions", "picks the forced best move"], body)

    if playout_rows:
        # Each playout row stands for weight / inclusion raw-play decisions: the bank's weight, over
        # the chance the playout sample took the row (playout input's `pl_incl`, 1 if not given).
        out += ["## Playout check: outcomes, independent of the value head", ""]
        pl = {p["id"]: p for p in playout_rows}
        incl = playout_incl or {}

        def diff(p: Dict[str, Any], x: int, y: int) -> Optional[Tuple[float, float]]:
            if x == y:
                return 0.0, 0.0
            if str(x) not in p["scores"] or str(y) not in p["scores"]:
                return None
            d = [u - w for u, w in zip(p["scores"][str(x)], p["scores"][str(y)])]
            mu = sum(d) / len(d)
            return mu, math.sqrt(sum((t - mu) ** 2 for t in d) / max(1, len(d) - 1) / len(d))

        # (bank row, verdict, population weight, playout lead and SE, search-valued lead) per comparison
        Cmp = Tuple[Dict[str, Any], Dict[str, Any], float, Tuple[float, float], float]
        vs_best: List[Cmp] = []
        vs_g256: List[Cmp] = []
        for b, _, v in rows:
            p = pl.get(b["id"])
            if p is None:
                continue
            w = b["weight"] / incl.get(b["id"], 1.0)
            db = diff(p, v["best"], b["raw"])
            if db is not None:
                vs_best.append((b, v, w, db, v["regret"]["raw"][0]))
            if "g256" in b["methods"]:
                dg = diff(p, b["methods"]["g256"]["choice"], b["raw"])
                if dg is not None:
                    vs_g256.append((b, v, w, dg, v["regret"]["raw"][0] - v["regret"]["g256"][0]))
        for name, sel in (("the reference best", vs_best), ("Gumbel k=8 @256's move", vs_g256)):
            sw = sum(g[2] for g in sel) or 1.0
            pl_gain = sum(g[2] * g[3][0] for g in sel) / sw
            se = math.sqrt(sum((g[2] * g[3][1]) ** 2 for g in sel)) / sw
            sv_gain = sum(g[2] * g[4] for g in sel) / sw / 2
            line = (f"**{name} over the raw move**, per raw-play decision: playouts {100 * pl_gain:+.3f} "
                    f"± {100 * se:.3f} points of win probability; search-valued {100 * sv_gain:+.3f}.")
            if decisions_per_game:
                line += (f" Per game ({decisions_per_game:.0f} decisions): playouts "
                         f"{100 * pl_gain * decisions_per_game:+.1f} ± {100 * se * decisions_per_game:.1f}, "
                         f"search-valued {100 * sv_gain * decisions_per_game:+.1f}.")
                num[f"playout_gain_per_game[{name}]"] = pl_gain * decisions_per_game
                num[f"playout_gain_per_game_se[{name}]"] = se * decisions_per_game
                num[f"search_gain_per_game[{name}]"] = sv_gain * decisions_per_game
            out += [line, ""]
        body = []
        for lo, hi_, lab in ((-9.0, 0.01, "< 1 point"), (0.01, 0.04, "1-2 points"),
                             (0.04, 0.1, "2-5 points"), (0.1, 9.0, "5+ points")):
            bucket = [g for g in vs_best if g[1]["best"] != g[0]["raw"] and lo <= g[4] < hi_]
            if not bucket:
                continue
            m = sum(g[3][0] for g in bucket) / len(bucket)
            se = math.sqrt(sum(g[3][1] ** 2 for g in bucket)) / len(bucket)
            body.append([lab, len(bucket), f"{100 * sum(g[4] for g in bucket) / len(bucket) / 2:+.2f}",
                         f"{100 * m:+.2f} ± {100 * se:.2f}", pct(sum(g[3][0] > 0 for g in bucket) / len(bucket))])
        body2 = []
        for lab, keep in (("search-valued regret < 2 points", lambda g: g[4] < 0.04),
                          ("search-valued regret >= 2 points", lambda g: g[4] >= 0.04)):
            for name, sel in (("best", vs_best), ("g256", vs_g256)):
                part = [g for g in sel if keep((g[0], g[1], g[2], g[3], g[1]["regret"]["raw"][0]))]
                share = sum(g[2] for g in part) / W if W else 0.0
                gain = sum(g[2] * g[3][0] for g in part) / W * (decisions_per_game or 1.0) if W else 0.0
                se = math.sqrt(sum((g[2] * g[3][1]) ** 2 for g in part)) / W * (decisions_per_game or 1.0) if W else 0.0
                body2.append([lab, name, pct(share), f"{100 * gain:+.1f} ± {100 * se:.1f}"])
                num[f"playout_gain_split[{lab}][{name}]"] = gain
        out += ["Where the playout-measured gain sits (population-weighted; gain per game if "
                "--decisions-per-game is given, else per decision):", ""]
        table(["decisions", "move", "share of decisions", "playout gain over raw (points)"], body2)
        conf = [g for g in vs_best if g[4] >= 0.04 and g[1]["best"] != g[0]["raw"] and g[3][0] > 2 * g[3][1]]
        if conf:
            out += [f"Recall where the playouts confirm a search-valued regret of 2+ points ({len(conf)} positions): "
                    + ", ".join(f"top-{k} {pct(sum(g[1]['rank'] <= k for g in conf) / len(conf))}" for k in (2, 4, 8))
                    + ".", ""]
            num["recall_playout_confirmed"] = {k: sum(g[1]["rank"] <= k for g in conf) / len(conf) for k in (2, 4, 8)}
        out += ["Where the reference best differs from raw, by the search-valued regret (unweighted):", ""]
        table(["search-valued regret", "positions", "search says (points)", "playouts say (points)",
               "playouts favour best"], body)

    # 6. The worst raw decisions.
    out += ["## Highest-regret raw decisions", ""]
    worst = sorted(hi, key=lambda x: -x[2]["regret"]["raw"][0])[:25]
    table(["id", "turn/AR", "side", "decision", "card", "raw", "best (raw rank)", "regret ± SE"],
          [[b["id"], f"T{b['turn']} AR{b['ar']}", b["side"], b["decision_type"],
            cards.get(int(b["card"]), "-") if b["card"] else "-", r["names"].get(str(b["raw"]), b["raw"]),
            f"{r['names'].get(str(v['best']), v['best'])} ({v['rank']})",
            f"{v['regret']['raw'][0]:.3f} ± {v['regret']['raw'][1]:.3f}"] for b, r, v in worst])
    return "\n".join(out), num


def read_all(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Every row of the files, duplicates kept (oracle rows repeat an id once per searcher)."""
    out: List[Dict[str, Any]] = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            out += [json.loads(line) for line in f]
    return out


def read_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Rows of one or more bank files, deduplicated by id (first wins)."""
    seen: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("schema") != SCHEMA:
                    raise ValueError(f"{p}: schema {r.get('schema')} is not {SCHEMA}")
                seen.setdefault(r["id"], r)
    return list(seen.values())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("annotate", help="raw self-play positions, each put to the raw network and Gumbel")
    an.add_argument("--model", required=True, help="a .pt checkpoint (onnx_to_checkpoint.py for an export)")
    an.add_argument("--games", type=int, required=True, help="games over all parts")
    an.add_argument("--part", default="1/1")
    an.add_argument("--seed", type=int, default=700_000, help="game g is played from seed + g")
    an.add_argument("--sample", type=int, default=8, help="keep one decision in this many")
    an.add_argument("--temperature", type=float, default=0.1)
    an.add_argument("--budgets", nargs="+", default=list(DEFAULT_BUDGETS), help="label=sims:k")
    an.add_argument("--chunk", type=int, default=64, help="positions per search call")
    an.add_argument("--out", required=True)
    an.add_argument("--resume", action="store_true")
    se = sub.add_parser("select", help="the stratified bank from annotated positions")
    se.add_argument("--annotated", nargs="+", required=True)
    se.add_argument("--size", type=int, default=8000)
    se.add_argument("--control", type=int, default=2000, help="agreement positions among --size")
    se.add_argument("--seed", type=int, default=0)
    se.add_argument("--out", required=True)
    rf = sub.add_parser("reference", help="high-budget reference solves and equal-budget move values")
    rf.add_argument("--model", required=True)
    rf.add_argument("--bank", required=True)
    rf.add_argument("--part", default="1/1")
    rf.add_argument("--ref-sims", type=int, default=1024)
    rf.add_argument("--ref-k", type=int, default=16)
    rf.add_argument("--ref-seeds", type=int, default=3)
    rf.add_argument("--value-sims", type=int, default=256)
    rf.add_argument("--critic-sims", type=int, default=1)
    rf.add_argument("--worlds", type=int, default=64)
    rf.add_argument("--top", type=int, default=4, help="also value the raw network's top moves")
    rf.add_argument("--chunk", type=int, default=4, help="positions per search call (memory)")
    rf.add_argument("--out", required=True)
    rf.add_argument("--resume", action="store_true")
    orc = sub.add_parser("oracle", help="searchers rerun with the reference move forced into their candidates")
    orc.add_argument("--model", required=True)
    orc.add_argument("--bank", required=True, help="rows with a `target` move")
    orc.add_argument("--part", default="1/1")
    orc.add_argument("--budgets", nargs="+", default=["g64=64:4", "g256=256:8"])
    orc.add_argument("--seeds", type=int, default=3)
    orc.add_argument("--chunk", type=int, default=64)
    orc.add_argument("--out", required=True)
    pl = sub.add_parser("playouts", help="paired playouts of each row's `pmoves` by the raw network")
    pl.add_argument("--model", required=True, help="the .pt; its .onnx beside it plays (or --onnx)")
    pl.add_argument("--onnx", default=None)
    pl.add_argument("--bank", required=True, help="rows with `pmoves`")
    pl.add_argument("--part", default="1/1")
    pl.add_argument("--pairs", type=int, default=64)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--chunk", type=int, default=16)
    pl.add_argument("--out", required=True)
    es = sub.add_parser("event-select", help="the event-decision bank: play-mode decisions by card and side")
    es.add_argument("--annotated", nargs="+", required=True)
    es.add_argument("--cap", type=int, default=30, help="positions per (card, side) at most")
    es.add_argument("--seed", type=int, default=0)
    es.add_argument("--out", required=True)
    er = sub.add_parser("event-report", help="the event-decision tables, from its bank and playouts")
    er.add_argument("--bank", required=True)
    er.add_argument("--playouts", nargs="+", required=True)
    er.add_argument("--games", type=int, required=True, help="annotate's game count")
    er.add_argument("--sample", type=int, default=8, help="annotate's --sample")
    er.add_argument("--human-holdings", default=None, help="event_play_census --dump of the human corpus")
    er.add_argument("--bot-holdings", default=None, help="event_play_census --dump of the model's self-play")
    er.add_argument("--out", required=True, help="Markdown; the numbers go to <out>.json")
    rp = sub.add_parser("report", help="the tables, from the bank and the later stages' outputs")
    rp.add_argument("--bank", required=True)
    rp.add_argument("--reference", nargs="+", required=True)
    rp.add_argument("--oracle", nargs="*", default=[])
    rp.add_argument("--playouts", nargs="*", default=[])
    rp.add_argument("--playouts-input", default=None,
                    help="the playout stage's input bank, for each row's inclusion probability (pl_incl)")
    rp.add_argument("--decisions-per-game", type=float, default=None,
                    help="decisions with 2+ legal moves in a raw game (annotate keeps 1 in --sample)")
    rp.add_argument("--out", required=True, help="Markdown; the numbers go to <out>.json")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        cards = {int(c["id"]): str(c["name"]) for c in json.load(open("rules/cards.json"))}
        md, num = analysis(read_rows([a.bank]), read_rows(a.reference), read_all(a.oracle) if a.oracle else [],
                           read_rows(a.playouts) if a.playouts else [], cards, a.decisions_per_game,
                           {r["id"]: float(r.get("pl_incl", 1.0)) for r in read_rows([a.playouts_input])}
                           if a.playouts_input else None)
        open(a.out, "w").write(md + "\n")
        json.dump(num, open(a.out + ".json", "w"), indent=1, default=str)
        print(f"report -> {a.out}", file=sys.stderr)
        return 0
    if a.cmd == "event-select":
        cards = {int(c["id"]): c for c in json.load(open("rules/cards.json"))}
        bank = event_select(iter_annotated(a.annotated, is_event_decision), cards, a.cap, a.seed)
        with gzip.open(a.out, "wt") as f:
            for r in bank:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        groups = collections.Counter(r["stratum"] for r in bank)
        print(f"{len(bank)} positions in {len(groups)} (card, side) groups, "
              f"{sum(len(r['pmoves']) for r in bank)} moves to play out -> {a.out}", file=sys.stderr)
        return 0
    if a.cmd == "event-report":
        from tools.scripts.event_play_census import load_holding

        def rates(path: Optional[str]) -> Optional[Dict[Tuple[int, str], Tuple[int, int]]]:
            return ar_event_rates([load_holding(r) for r in json.load(open(path))["holdings"]]) if path else None

        cards = {int(c["id"]): c for c in json.load(open("rules/cards.json"))}
        md, num = event_analysis(read_rows([a.bank]), read_rows(a.playouts), cards, a.sample / a.games,
                                 rates(a.human_holdings), rates(a.bot_holdings))
        open(a.out, "w").write(md + "\n")
        json.dump(num, open(a.out + ".json", "w"), indent=1, default=str)
        print(f"event report -> {a.out}", file=sys.stderr)
        return 0
    if a.cmd == "oracle":
        return oracle(a)
    if a.cmd == "playouts":
        return playouts(a)
    if a.cmd == "annotate":
        return annotate(a)
    if a.cmd == "select":
        bank = select(read_rows(a.annotated), a.size, a.control, a.seed)
        with gzip.open(a.out, "wt") as f:
            for r in bank:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        strata = collections.Counter(r["stratum"] for r in bank)
        print(f"{len(bank)} positions in {len(strata)} strata -> {a.out}", file=sys.stderr)
        return 0
    if a.cmd == "reference":
        return reference(a)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
