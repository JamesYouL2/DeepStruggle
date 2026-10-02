"""Where does the model disagree with strong human play, and who is right by its own playouts?

The human ts-replayer corpus (`tools/lib/ts_replayer_convert.py`) rebuilds every decision of 266
games between strong players, with the hands solved exactly. At each one this asks the model what
it would play there, and where its greedy choice differs from the human's, plays both out:

* **scan:** every human decision, classed by kind (headline, the card to play, its play mode,
  influence points, coup and realignment targets, choices inside an event, setup), with the
  model's probability of the human's move and whether its own top move agrees;
* **playouts:** a sample of the disagreements -- where the model gave the human's move less than
  `max_p` -- each with two branches, the human's move and the model's, then the model plays on:
  the rest of the round without suicide (`playout_audit.play_safe`), then the game on both sides.
  Pair k redeals the cards the mover cannot see and uses the same dice in both branches.

The human's move is judged by what the model makes of it afterwards, not by the plan the human had
for it, so a human move whose value lies in a follow-up the model does not find scores as worse
than it is. A positive human − model is the stronger statement for that reason.
"""
from __future__ import annotations

import gzip
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import _decider, _pair_start, apply_prefix
from ai.eval.doctrine_census import cards
from ai.eval.ops_block import NODE_OFFSET
from ai.eval.playout_audit import play_safe
from ai.eval.reply_probe import _ar_key, is_round_start, position_link
from bindings.action_encoder import ActionEncoder

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
KINDS = ("headline", "card", "mode", "influence", "coup", "realign", "event choice", "setup", "other")
#: (observations, masks) -> logits
LogitsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def kind_of(st: ts.GameState) -> str:
    ctx = st.ctx()
    dt = ctx.decision_type
    if st.current_phase == ts.Phase.HEADLINE and dt == ts.DecisionType.SELECT_CARD:
        return "headline"
    if st.current_phase not in (ts.Phase.ACTION_ROUND, ts.Phase.HEADLINE):
        return "setup"
    if is_round_start(st):
        return "card"
    if dt == ts.DecisionType.SELECT_PLAY_MODE:
        return "mode"
    if int(ctx.resolving_card) != 0:
        return "event choice"
    if dt == ts.DecisionType.POINT_NODE:
        return {ts.OpMode.INFLUENCE: "influence", ts.OpMode.COUP: "coup",
                ts.OpMode.REALIGN: "realign"}.get(ctx.op_mode, "other")
    return "other"


def _card(st: ts.GameState) -> int:
    ctx = st.ctx()
    return int(ctx.pending_op_card or ctx.resolving_card or 0)


def _probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    z = np.where(m, logits, -np.inf)
    z = z - z[m].max()
    p = np.exp(z)
    return p / p.sum()


def same_play_sets(decisions: Sequence[Tuple[str, int, int, bool, int]]) -> List[set]:
    """Per decision (kind, mover, card, is a point, the human's action): the actions the human took
    in the same multi-point play. Influence, realignment rolls and an event's placements or
    removals are a block of consecutive point decisions by one player on one card, and a point
    placed in another order is the same play; any other decision gets an empty set."""
    used: List[set] = []
    block: List[int] = []
    for k, (kind, mover, card, point, human) in enumerate(decisions):
        multi = point and kind in ("influence", "realign", "event choice")
        prev = decisions[block[-1]] if block else None
        same = (multi and prev is not None and prev[0] == kind and prev[1] == mover and prev[2] == card
                and block[-1] == k - 1)
        if not same:
            block = []
        block.append(k)
        used.append(set())
        if multi:
            used[block[0]].add(human)
            for j in block:
                used[j] = used[block[0]]
    return used


def scan_game(game: Dict[str, Any], logits_fn: LogitsFn, game_id: int) -> Tuple[List[Dict[str, Any]], List[ts.GameState]]:
    """Every human decision with two or more legal moves: one row each, and the states of the
    disagreements (row["state_index"] points into the second list)."""
    from tools.lib.ts_replayer_convert import convert_game

    seen: List[Tuple[ts.GameState, ts.Player, int]] = []

    def on_decision(state: ts.GameState, mover: ts.Player, entry: Any, chosen: int) -> None:
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        if mask.sum() >= 2 and mask[chosen]:
            seen.append((state, mover, int(chosen)))

    conv = convert_game(game, on_decision=on_decision)
    if conv.skipped or not seen:
        return [], []
    obs = np.stack([np.asarray(ts.extract_observation(s, m), dtype=np.float32) for s, m, _ in seen])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s, _, _ in seen])
    lg = logits_fn(obs, masks)
    info = cards()
    used = same_play_sets([(kind_of(st), int(mover), _card(st), st.ctx().decision_type == ts.DecisionType.POINT_NODE,
                            human) for st, mover, human in seen])
    rows: List[Dict[str, Any]] = []
    states: List[ts.GameState] = []
    for k, ((st, mover, human), m, l) in enumerate(zip(seen, masks, lg)):
        p = _probs(l, m)
        bot = int(np.argmax(np.where(m.astype(bool), p, -1.0)))
        if bot != human and bot in used[k]:
            bot = human                       # another order of the same play: not a disagreement
        cid = _card(st)
        row: Dict[str, Any] = {
            "game": game_id, "k": k, "kind": kind_of(st), "side": "US" if mover == ts.Player.US else "USSR",
            "turn": int(st.turn), "ar": int(st.action_round), "defcon": int(st.defcon),
            "card": str(info[cid]["name"]) if cid in info else "", "human": human, "bot": bot,
            "p_human": float(p[human]), "p_bot": float(p[bot]), "agree": human == bot,
            "p_top": float(p.max()),
            "human_name": ActionEncoder.get_action_name(st, human), "bot_name": ActionEncoder.get_action_name(st, bot),
        }
        if not row["agree"]:
            row["state_index"] = len(states)
            states.append(st)
        rows.append(row)
    return rows, states


def choose_spots(rows: Sequence[Dict[str, Any]], per_kind: int, max_p: float, seed: int) -> List[Dict[str, Any]]:
    """Up to `per_kind` disagreements of each kind where the model gave the human's move less
    than `max_p`, drawn uniformly (so pooled means are not tilted toward the strangest moves)."""
    rng = np.random.default_rng(seed)
    out = []
    for kind in KINDS:
        pool = [r for r in rows if not r["agree"] and r["kind"] == kind and r["p_human"] < max_p]
        if len(pool) > per_kind:
            pool = [pool[j] for j in sorted(rng.choice(len(pool), per_kind, replace=False))]
        out += pool
    return out


def play(spots: Sequence[Dict[str, Any]], states: Sequence[ts.GameState], act: Any, pairs: int, seed: int,
         chunk: int = 1536) -> List[Dict[str, Any]]:
    """human − model per spot, over `pairs` paired playouts."""
    out: List[Dict[str, Any]] = []
    per = 2 * pairs
    step = max(1, chunk // per)
    for lo in range(0, len(spots), step):
        group = spots[lo:lo + step]
        starts, movers, keys = [], [], []
        for gi, r in enumerate(group):
            st = states[r["state_index"]]
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo + gi, "resample")
                for a in (r["human"], r["bot"]):
                    starts.append(apply_prefix(base, [a]))
                    movers.append(_decider(st))
                    keys.append(_ar_key(st))
        flat = np.array(play_safe(starts, movers, keys, act, seed + lo)).reshape(len(group), pairs, 2)
        for r, sc in zip(group, flat):
            d = sc[:, 0] - sc[:, 1]
            st = states[r["state_index"]]
            out.append({**{k: v for k, v in r.items() if k != "state_index"}, "pairs": pairs,
                        "human_score": float(sc[:, 0].mean()), "bot_score": float(sc[:, 1].mean()),
                        "diff": float(d.mean()), "diff_se": float(d.std(ddof=1) / math.sqrt(pairs)),
                        "link": position_link(st)})
    return out


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def _label(r: Dict[str, Any], which: str) -> str:
    """A move named for pooling: the mode name, the card, or the country -- without the
    engine's numbering, so the same choice pools across positions."""
    nm = r[f"{which}_name"]
    return nm.split("(")[-1].rstrip(")") if "(" in nm else nm.split("/")[-1].strip()


def report(scan: Sequence[Dict[str, Any]], played: Sequence[Dict[str, Any]], meta: Dict[str, Any],
           top: int = 25) -> Tuple[str, Dict[str, Any]]:
    games = len({r["game"] for r in scan})
    out = [f"# The model against strong human play — {meta.get('model', '?')}", "",
           f"{len(scan)} human decisions with a real choice, from {games} games of the ts-replayer corpus.", ""]
    summary: Dict[str, Any] = {"meta": meta, "decisions": len(scan), "games": games, "kinds": {}}
    out += ["## Where it agrees", "",
            "| decision | decisions | model's top move = human's | model's probability of the human's move |",
            "|:---|---:|---:|---:|"]
    for kind in KINDS:
        sel = [r for r in scan if r["kind"] == kind]
        if not sel:
            continue
        agree = float(np.mean([r["agree"] for r in sel]))
        ph = float(np.mean([r["p_human"] for r in sel]))
        out.append(f"| {kind} | {len(sel)} | {100 * agree:.0f}% | {100 * ph:.0f}% |")
        summary["kinds"][kind] = {"n": len(sel), "agree": agree, "p_human": ph}

    # The most common disagreements, by kind and card: what humans do that the model does not.
    pairs_: Dict[Tuple[str, str, str, str, str], int] = {}
    for r in scan:
        if r["agree"] or r["kind"] not in ("headline", "card", "mode", "event choice"):
            continue
        key = (r["kind"], r["side"], r["card"] if r["kind"] in ("mode", "event choice") else "",
               _label(r, "human"), _label(r, "bot"))
        pairs_[key] = pairs_.get(key, 0) + 1
    common = sorted(pairs_.items(), key=lambda x: -x[1])[:top]
    out += ["", "## The most common disagreements (cards and modes)", "",
            "| decision | side | card | human | model | times |", "|:---|:---|:---|:---|:---|---:|"]
    for (kind, side, card, h, b), n in common:
        out.append(f"| {kind} | {side} | {card or '—'} | {h} | {b} | {n} |")

    if played:
        out += ["", "## Who is right, by the model's own playouts", "",
                f"Disagreements where the model gave the human's move under {100 * meta.get('max_p', 0.25):.0f}%, "
                "sampled per kind. human − model: + means the human's move scores better, with the model "
                "playing every continuation.", "",
                "| decision | spots | human − model | human better in |", "|:---|---:|---:|---:|"]
        for kind in KINDS:
            sel = [r["diff"] for r in played if r["kind"] == kind]
            if len(sel) < 2:
                continue
            m, se = _mean_se(sel)
            out.append(f"| {kind} | {len(sel)} | {100 * m:+.1f} ± {100 * se:.1f} | "
                       f"{100 * np.mean([x > 0 for x in sel]):.0f}% |")
            summary["kinds"].setdefault(kind, {})["human_minus_model"] = (m, se)
        m, se = _mean_se([r["diff"] for r in played])
        out.append(f"| **all** | {len(played)} | {100 * m:+.1f} ± {100 * se:.1f} | "
                   f"{100 * np.mean([r['diff'] > 0 for r in played]):.0f}% |")
        summary["all"] = (m, se)

        pooled: Dict[Tuple[str, str, str, str, str], List[float]] = {}
        for r in played:
            key = (r["kind"], r["side"], r["card"] if r["kind"] in ("mode", "event choice") else "",
                   _label(r, "human"), _label(r, "bot"))
            pooled.setdefault(key, []).append(r["diff"])
        rows = [(k, *_mean_se(v), len(v)) for k, v in pooled.items() if len(v) >= 5]
        rows.sort(key=lambda x: -(x[1] / x[2]) if x[2] and x[2] > 0 else 0.0)
        out += ["", "### Recurring disagreements, pooled (5+ spots), the human's best first", "",
                "| decision | side | card | human | model | spots | human − model |",
                "|:---|:---|:---|:---|:---|---:|---:|"]
        for (kind, side, card, h, b), m, se, n in rows[:top]:
            out.append(f"| {kind} | {side} | {card or '—'} | {h} | {b} | {n} | {100 * m:+.1f} ± {100 * se:.1f} |")
        out += ["", "### …and the model's best", "", "| decision | side | card | human | model | spots | human − model |",
                "|:---|:---|:---|:---|:---|---:|---:|"]
        for (kind, side, card, h, b), m, se, n in rows[::-1][:10]:
            out.append(f"| {kind} | {side} | {card or '—'} | {h} | {b} | {n} | {100 * m:+.1f} ± {100 * se:.1f} |")

        best = sorted(played, key=lambda r: -(r["diff"] / max(r["diff_se"], 0.02)))[:top]
        out += ["", f"## The {top} spots where the human's move beats the model's most clearly", "",
                "| decision | side | turn | card | human | model (p of human's move) | human − model | |",
                "|:---|:---|---:|:---|:---|:---|---:|:---|"]
        for r in best:
            out.append(f"| {r['kind']} | {r['side']} | {r['turn']}.{r['ar']} | {r['card'] or '—'} | {r['human_name']} | "
                       f"{r['bot_name']} ({100 * r['p_human']:.0f}%) | {100 * r['diff']:+.0f} ± {100 * r['diff_se']:.0f} | "
                       f"[position]({r['link']}) |")
    return "\n".join(out) + "\n", summary


def load_game(path: Any) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)
