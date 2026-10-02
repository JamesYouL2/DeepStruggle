"""A reusable bank of positions with every candidate move played out, so a new rule is a query.

The playout probes so far each collected positions, ran their own branches and threw the rest away.
This keeps everything a later question could need, once:

* **the position** -- its save JSON, so any spot can be re-played or opened in the workbench;
* **generic features** -- side, turn, action round, rounds left for each side, both hand sizes and
  the surplus over rounds left, DEFCON, the VP lead, Military Ops and the shortfall for both sides,
  the space race, the China Card, each region's scoring-card status and what it would score now,
  the hand-timing cards the mover holds (`HAND_TIMING_CARDS`, both sides' cards), and for the card
  in play its owner, Ops, what its event gains (own or neutral card) or costs (opponent's card);
* **every candidate move, played out** -- at a card choice every legal card (so any timing rule can
  be asked offline), at a play-mode decision every legal mode that is not suicide, elsewhere the
  model's four likeliest moves; plus the critic's favourite option at a choice inside an event and,
  where one exists, the two-step line that wins on the spot (Wargames' event, then "end the game").
  Each candidate is played `pairs` times with paired redeals and dice (`branch_oracle._pair_start`),
  the rest of the round without suicide and the game by the model (`playout_audit.play_safe`), and
  every pair's result is kept -- '0' loss, '1' draw, '2' win, for the mover -- so pooled and
  paired statistics can be rebuilt for any subset.

Candidate 0 is always the model's own (greedy) move. A rule is then a function from a record to a
candidate index (`ai/eval/bank_query.py`): pooled over the records it applies to, its candidate's
pairs minus candidate 0's give the effect with no new playouts. A rule whose move is not among the
candidates cannot be asked of the bank and needs its own run.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _decider, _pair_start
from ai.eval.doctrine_census import _hand, cards
from ai.eval.human_disagree import kind_of
from ai.eval.playout_audit import play_safe, suicide
from ai.eval.reply_probe import _ar_key, opponent_replies, rounds_left
from ai.eval.rule_oracle import (EVENT, MODE_BASE, SCORING_REGION, ValueFn, _owner, card_cost, decisive,
                                 event_cost, event_gain, greedy_option, score_value)
from ai.eval.region_weight import REGIONS, STATUS, status as region_status
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

#: Cards whose effect turns on hands, discards, card counts or action rounds -- either side's.
HAND_TIMING_CARDS = ("Five Year Plan", "Aldrich Ames Remix", "SALT Negotiations", "Missile Envy",
                     "Grain Sales to Soviets", "Ussuri River Skirmish", "Nixon Plays the China Card",
                     "Cultural Revolution", "The China Card", "Blockade", "Terrorism", "Star Wars",
                     "“Ask Not What Your Country Can Do For You…”", "Our Man in Tehran", "“Lone Gunman”",
                     "CIA Created", "The Cambridge Five", "Quagmire", "Bear Trap", "UN Intervention", "North Sea Oil")
KIND_WEIGHT: Dict[str, float] = {"card": 4.0, "mode": 4.0, "event choice": 3.0, "headline": 2.0, "other": 2.0,
                                 "setup": 1.0, "coup": 1.0, "realign": 1.0, "influence": 0.4}
ProbsFn = Callable[[ts.GameState], np.ndarray]
CHINA = 6


def timing_ids() -> Dict[int, str]:
    by = {str(v["name"]): k for k, v in cards().items()}
    return {by[n]: n for n in HAND_TIMING_CARDS}


def features(st: ts.GameState, act: PolicyFn, probe_cards: bool = True) -> Dict[str, Any]:
    """The generic features of a decision, from the mover's side (see the module doc)."""
    mover = _decider(st)
    us = mover == ts.Player.US
    sign = 1 if us else -1
    ctx = st.ctx()
    rl = rounds_left(st)
    hand = _hand(st, mover)
    opp = ts.Player.USSR if us else ts.Player.US
    opp_hand = _hand(st, opp)
    mil_me = int(st.us_mil_ops) if us else int(st.ussr_mil_ops)
    mil_opp = int(st.ussr_mil_ops) if us else int(st.us_mil_ops)
    china = st.china_card_holder
    tids = timing_ids()
    f: Dict[str, Any] = {
        "side": "US" if us else "USSR", "turn": int(st.turn), "ar": int(st.action_round), "kind": kind_of(st),
        "phasing": st.phasing_player == mover, "rounds_left": rl,
        "opp_rounds_left": rl + (1 if (st.phasing_player == ts.Player.USSR and us) else 0),
        "opp_replies": bool(opponent_replies(st)),
        "hand": len([c for c in hand if c != CHINA]), "opp_hand": len([c for c in opp_hand if c != CHINA]),
        "defcon": int(st.defcon), "lead": sign * int(st.victory_points),
        "mil": mil_me, "mil_opp": mil_opp,
        "short": max(0, int(st.defcon) - mil_me), "short_opp": max(0, int(st.defcon) - mil_opp),
        "space": int(st.us_space_track if us else st.ussr_space_track),
        "space_opp": int(st.ussr_space_track if us else st.us_space_track),
        "china": "me" if china == mover else ("opp" if china == opp else "none"),
        "china_playable": bool(st.china_card_playable),
        "timing_held": sorted(c for c in hand if c in tids),
        "scoring_held": sorted(c for c in hand if c in SCORING_REGION or c == 38),
        "region_status": "".join(str(STATUS.index(region_status(st, mover, r))) for r in REGIONS),
        "region_value": [score_value(st, c, mover) for c in (2, 1, 3, 79, 37, 81)],
        "card": int(ctx.pending_op_card or ctx.resolving_card or 0),
    }
    f["surplus"] = f["hand"] - rl
    cid = int(ctx.pending_op_card)
    if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and cid:
        info = cards().get(cid, {})
        f["owner"] = _owner(cid, mover)
        f["ops"] = int(info.get("ops", 0))
        legal = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
        if legal[EVENT]:
            if f["owner"] == "opp":
                f["event_cost"] = event_cost(st, mover, act)
            else:
                f["event_gain"] = event_gain(st, mover, act)
    if probe_cards and f["kind"] == "card":
        per = {}
        legal_now = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
        for c in hand:
            if c == CHINA:
                continue
            row: Dict[str, Any] = {"owner": _owner(c, mover), "ops": int(cards().get(c, {}).get("ops", 0)),
                                   "playable": bool(legal_now[c - 1])}
            if c in SCORING_REGION:
                row["score"] = score_value(st, c, mover)
            elif row["owner"] == "opp" and row["playable"]:
                row["cost"] = card_cost(st, c, mover, act)
            per[str(c)] = row
        f["hand_cards"] = per
    return f


def candidates(st: ts.GameState, kind: str, probs: np.ndarray, value_fn: ValueFn, k: int = 4
               ) -> List[Tuple[str, List[int]]]:
    """(name, prefix) per candidate, the model's own move first (see the module doc)."""
    mask = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
    greedy = int(np.argmax(np.where(mask, probs, -1.0)))
    name = lambda a: ActionEncoder.get_action_name(st, int(a))  # noqa: E731
    out: List[Tuple[str, List[int]]] = [(name(greedy), [greedy])]
    seen = {greedy}
    legal = [int(a) for a in np.flatnonzero(mask)]
    if kind == "card":
        rest = legal                                    # every card, and passing where legal
    elif kind == "mode":
        rest = [a for a in legal if MODE_BASE <= a < MODE_BASE + 5 and not suicide(st, a)]
    else:
        rest = sorted(legal, key=lambda a: -probs[a])[:k]
    for a in rest:
        if a not in seen:
            out.append((name(a), [a]))
            seen.add(a)
    mover = _decider(st)
    if int(st.ctx().resolving_card) and len(legal) >= 2:
        g = greedy_option(st, mover, value_fn)
        if g is not None and g[0] not in seen:
            out.append(("critic's favourite: " + name(g[0]), [g[0]]))
            seen.add(g[0])
    sign = 1 if mover == ts.Player.US else -1
    if sign * int(st.victory_points) >= 6:
        d = decisive(st, mover)
        if d is not None and len(d) == 2:
            out.append(("decisive: " + " → ".join(name(a) for a in d[:1]) + " then win", d))
    return out


def boost(st: ts.GameState, kind: str) -> float:
    """Extra sampling weight for the rare situations the rules ask about: an action round's first
    decision in the last round or holding a hand-timing or scoring card, and a play-mode decision at a
    winning lead (Wargames territory)."""
    mover = _decider(st)
    if kind == "card":
        if rounds_left(st) == 0:
            return 4.0
        hand = set(_hand(st, mover))
        if hand & (set(timing_ids()) - {CHINA}) or hand & (set(SCORING_REGION) | {38}):
            return 2.0
    if kind == "mode":
        lead = int(st.victory_points) * (1 if mover == ts.Player.US else -1)
        if lead >= 6:
            return 5.0
    return 1.0


MAX_BOOST = 5.0


def collect(act: PolicyFn, n: int, seed: int, envs: int = 32, base: float = 0.004, per_game: int = 10,
            max_steps: int = 5_000_000) -> List[Tuple[ts.GameState, str, float]]:
    """`n` decisions with two or more legal moves, drawn uniformly from finished greedy self-play
    games, each kept with probability base × its kind's weight × `boost`. The weight is returned
    with each, so a pooled number can be reweighted to the self-play mix (divide by it)."""
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(envs, seed * 47 + 31)
    runner.refresh_all()
    pool: Dict[int, List[Tuple[ts.GameState, str, float]]] = {}
    game = list(range(envs))
    taken = [0] * envs
    done: List[int] = []
    started, finished = envs, 0
    top = max(KIND_WEIGHT.values())
    want_games = max(1, int(math.ceil(n / (per_game * 0.5))))
    for _ in range(max_steps):
        if finished >= want_games and sum(len(pool.get(g, [])) for g in done) >= n:
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        draws = rng.random(envs)
        for i in range(envs):
            if taken[i] >= per_game or masks[i].sum() < 2 or draws[i] >= base * top * MAX_BOOST:
                continue
            st = runner.get_state(i)
            if st.current_phase not in (ts.Phase.ACTION_ROUND, ts.Phase.HEADLINE):
                continue
            kd = kind_of(st)
            w = KIND_WEIGHT.get(kd, 1.0) * boost(st, kd)
            if draws[i] < base * w:
                pool.setdefault(game[i], []).append((st.clone(), kd, w))
                taken[i] += 1
        runner.step_flat_all([int(x) for x in act(obs, masks)], auto_advance=True)
        ends = np.flatnonzero(np.array(runner.get_terminals())).tolist()
        for i in ends:
            done.append(game[i])
            finished += 1
            runner.reset_game(int(i), seed * 1_000_003 + started)
            game[i] = started
            started += 1
            taken[i] = 0
        if ends:
            runner.refresh_all()
    kept = [x for g in done for x in pool.get(g, [])]
    if len(kept) > n:
        kept = [kept[j] for j in sorted(rng.choice(len(kept), n, replace=False))]
    return kept


def build(act: PolicyFn, probs_fn: ProbsFn, value_fn: ValueFn, positions: Sequence[Tuple[ts.GameState, str, float]],
          pairs: int, seed: int, chunk: int = 1536) -> List[Dict[str, Any]]:
    """One bank record per position: features, candidates and every pair's result per candidate."""
    plan = []
    for st, kind, w in positions:
        cands = candidates(st, kind, probs_fn(st), value_fn)
        if len(cands) >= 2:
            feats = features(st, act)
            feats["weight"] = w
            plan.append((st, kind, cands, feats))
    records: List[Dict[str, Any]] = []
    lo = 0
    while lo < len(plan):
        group, size = [], 0
        while lo < len(plan) and (not group or size + len(plan[lo][2]) * pairs <= chunk):
            group.append(plan[lo])
            size += len(plan[lo][2]) * pairs
            lo += 1
        starts, movers, keys = [], [], []
        for gi, (st, _, cands, _) in enumerate(group):
            mover = _decider(st)
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo * 7 + gi, "resample")
                for _, prefix in cands:
                    starts.append(apply_lenient(base, prefix))
                    movers.append(mover)
                    keys.append(_ar_key(st))
        res = play_safe(starts, movers, keys, act, seed + lo)
        at = 0
        for st, kind, cands, feats in group:
            nb = len(cands)
            sc = np.array(res[at:at + nb * pairs]).reshape(pairs, nb)
            at += nb * pairs
            p = probs_fn(st)
            records.append({
                "save": st.to_save_json(), "kind": kind, "features": feats, "pairs": pairs,
                "candidates": [{"name": nm, "prefix": pre, "prior": float(p[pre[0]])} for nm, pre in cands],
                "results": ["".join(str(int(round(2 * x))) for x in sc[:, j]) for j in range(nb)],
            })
    return records


def apply_lenient(state: ts.GameState, prefix: Sequence[int]) -> ts.GameState:
    """A copy of `state` with `prefix` applied while each step is legal; the model plays on from the
    first that is not. A later step can turn on hidden cards (the follow-up of an event that picks
    from the opponent's hand), so in a redealt world it may not exist: that pair then measures the
    line as far as it goes, rather than failing the run."""
    st = state.clone()
    for a in prefix:
        if ts.Engine.is_terminal(st) or not np.asarray(ActionEncoder.get_legal_mask(st))[int(a)]:
            break
        ts.Engine.step_flat(st, int(a))
        drain_chance(st, context="position_bank prefix")
    return st


def scores(record: Dict[str, Any], j: int) -> np.ndarray:
    """Candidate j's per-pair results as 0 / 0.5 / 1."""
    return np.array([int(ch) / 2.0 for ch in record["results"][j]])
