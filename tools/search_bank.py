#!/usr/bin/env python3
"""The search disagreement bank: where Gumbel search departs from the raw network, and why.

An offline diagnostic. Nothing here changes how a player or a training run behaves; the Gumbel
root is driven through its ordinary agent (`gumbel:` spec), and only reads its per-candidate
statistics (`GumbelRoot.last_stats`).

Stages (each resumable, each writing JSONL.gz rows of schema `SCHEMA`):

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
    a = ap.parse_args(argv)
    if a.cmd == "annotate":
        return annotate(a)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
