"""Pooled rules over generic features of the position: does following a rule beat the model's own move?

The fork owner's request (2026-10-02): rules a training step could distil must be **general** --
written over features any card can have, never over card names -- so that every card fitting a
pattern feeds one pooled test, and a single rule's verdict reaches cards the test never named. Each
feature is the engine's own answer, read by probing a copy of the position.

| rule | applies when (and the model departs from it) | prescribes | feature (dose) |
|:---|:---|:---|:---|
| `decisive` | some move wins on the spot or with the mover's own next decision (Wargames' "end the game") | that move | -- |
| `event_vp` | an own or neutral card's event is legal and not chosen | the event | VP the event gains by the end of the mover's round |
| `space_harmful` | an opponent's card can be spaced and is not | space it | what its event costs the mover: VP lost + battlegrounds lost |
| `delay_harmful` | an action round opens with an opponent's card, rounds remain, and another card could be played | the model's likeliest other card | the same event cost |
| `score_now` | a held scoring card would score for the mover now and is not played | play it | the VP it would score now |
| `hold_score` | a scoring card that would score against the mover is played with 2+ rounds left | the model's likeliest other card | the VP it scores (negative) |
| `greedy_choice` | a choice inside an event, where the critic's one-step reading prefers another option | that option | the critic's gain, in win-rate points |
| `milops_coup` | the last action round (or the one before), 2+ short of the Military Ops requirement, a coup is legal and not suicide, and the model does not coup | coup | the shortfall |

Every spot is a position from the model's greedy self-play where the rule applies and the model's
own move is not the rule's. Two branches -- the model's move and the rule's -- are played out with
paired redeals and dice (`branch_oracle._pair_start`), the rest of the round without suicide and the
game by the model (`playout_audit.play_safe`). Results are pooled by rule, binned by the feature (a
dose-response, so no cutoff is guessed), and checked for generality: side, era, and the effect with
the single most frequent card left out -- a "rule" that lives on one card is a card fact.

As everywhere here, a branch is valued by how the model follows it up.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _decider, _pair_start
from ai.eval.choice_oracle import _loses_now, resolve_safely
from ai.eval.doctrine_census import _hand, cards
from ai.eval.ops_block import country_table, controlled, influence
from ai.eval.playout_audit import play_safe, suicide
from ai.eval.reply_probe import _ar_key, is_round_start, position_link, rounds_left
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
EVENT, SPACE, INFLUENCE, COUP = MODE_BASE, MODE_BASE + 1, MODE_BASE + 2, MODE_BASE + 3
RULES = ("decisive", "event_vp", "space_harmful", "delay_harmful", "score_now", "hold_score",
         "greedy_choice", "milops_coup")
SCORING_REGION = {1: ts.Region.ASIA, 2: ts.Region.EUROPE, 3: ts.Region.MIDDLE_EAST, 37: ts.Region.CENTRAL_AMERICA,
                  79: ts.Region.AFRICA, 81: ts.Region.SOUTH_AMERICA}
#: (observations, masks) -> v_win from each row's decider's side.
ValueFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
ERAS = ((1, 3, "Early War"), (4, 7, "Mid War"), (8, 10, "Late War"))
BINS: Dict[str, Tuple[Tuple[float, float, str], ...]] = {
    "event_vp": ((-99, 0, "≤ 0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+")),
    "space_harmful": ((-99, 0, "≤ 0"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+")),
    "delay_harmful": ((-99, 0, "≤ 0"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+")),
    "score_now": ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+")),
    "hold_score": ((-1, -1, "−1"), (-2, -2, "−2"), (-4, -3, "−3 to −4"), (-99, -5, "≤ −5")),
    "greedy_choice": ((0, 2, "0-2 pts"), (2, 5, "2-5"), (5, 10, "5-10"), (10, 999, "10+")),
    "milops_coup": ((2, 2, "2"), (3, 3, "3"), (4, 99, "4+")),
    "decisive": ((-999, 999, "all"),),
}


def _side(p: ts.Player) -> int:
    return 0 if p == ts.Player.US else 1


def _card_side(cid: int) -> str:
    c = cards().get(cid)
    return str(c["side"]).lower() if c else "?"


def _owner(cid: int, mover: ts.Player) -> str:
    s = _card_side(cid)
    if s == "neutral":
        return "neutral"
    return "own" if s == ("us" if mover == ts.Player.US else "ussr") else "opp"


def _step(st: ts.GameState, a: int) -> None:
    ts.Engine.step_flat(st, int(a))
    drain_chance(st, context="rule_oracle probe")


def _wins_now(st: ts.GameState, a: int, mover: ts.Player) -> bool:
    probe = st.clone()
    _step(probe, a)
    if not ts.Engine.is_terminal(probe):
        return False
    u = float(ts.Engine.get_terminal_utility(probe))
    return (u > 0) if mover == ts.Player.US else (u < 0)


def decisive(st: ts.GameState, mover: ts.Player) -> Optional[List[int]]:
    """A move that wins on the spot, or whose follow-up decision by the mover can win on the spot
    (Wargames: the event, then "end the game"); the moves as a prefix, else None."""
    legal = [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]
    for a in legal:
        if _wins_now(st, a, mover):
            return [a]
    for a in legal:
        probe = st.clone()
        _step(probe, a)
        if ts.Engine.is_terminal(probe) or _decider(probe) != mover:
            continue
        for b in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(probe))):
            if _wins_now(probe, int(b), mover):
                return [a, int(b)]
    return None


def _advance_opponent(st: ts.GameState, mover: ts.Player, act: PolicyFn, limit: int = 40) -> None:
    """Let the opponent resolve what it decides (an event fired for it), the model choosing, until
    the mover decides again or the round ends."""
    key = _ar_key(st)
    for _ in range(limit):
        if ts.Engine.is_terminal(st) or _ar_key(st) != key or _decider(st) == mover:
            return
        obs = np.asarray(ts.extract_observation(st, _decider(st)), dtype=np.float32)[None]
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)[None]
        _step(st, int(act(obs, mask)[0]))


def event_cost(st: ts.GameState, mover: ts.Player, act: PolicyFn) -> Optional[float]:
    """At a play-mode decision on an opponent's card: what its event costs the mover, played
    event-first and resolved by the model for the opponent -- VP lost plus battlegrounds lost."""
    if not np.asarray(ActionEncoder.get_legal_mask(st))[EVENT]:
        return None
    _, bg, _ = country_table()
    s = _side(mover)
    probe = st.clone()
    vp0, bg0 = int(st.victory_points), int((controlled(influence(st), s) & bg).sum())
    _step(probe, EVENT)
    _advance_opponent(probe, mover, act)
    sign = 1 if mover == ts.Player.US else -1
    if ts.Engine.is_terminal(probe):
        u = float(ts.Engine.get_terminal_utility(probe)) * sign
        return 20.0 if u < 0 else -20.0
    dvp = sign * (int(probe.victory_points) - vp0)
    dbg = int((controlled(influence(probe), s) & bg).sum()) - bg0
    return float(-dvp - dbg)


def card_cost(st: ts.GameState, cid: int, mover: ts.Player, act: PolicyFn) -> Optional[float]:
    """At an action round's first decision: the event cost of playing opponent card `cid` now."""
    probe = st.clone()
    _step(probe, cid - 1)
    if ts.Engine.is_terminal(probe) or probe.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
        return None
    return event_cost(probe, mover, act)


def event_gain(st: ts.GameState, mover: ts.Player, act: PolicyFn) -> float:
    """At a play-mode decision on an own or neutral card: the VP its event gains the mover by the
    end of the mover's round (inner choices the model's, without suicide)."""
    probe = st.clone()
    vp0 = int(st.victory_points)
    _step(probe, EVENT)
    probe = resolve_safely(probe, act)
    sign = 1 if mover == ts.Player.US else -1
    if ts.Engine.is_terminal(probe):
        u = float(ts.Engine.get_terminal_utility(probe)) * sign
        return 20.0 if u > 0 else -20.0
    return float(sign * (int(probe.victory_points) - vp0))


def score_value(st: ts.GameState, cid: int, mover: ts.Player) -> Optional[int]:
    """The VP scoring card `cid` would score for the mover if played now (Southeast Asia: None)."""
    reg = SCORING_REGION.get(cid)
    if reg is None:
        return None
    net = int(ts.Scoring.evaluate_region(st, reg).net_delta)
    return net if mover == ts.Player.US else -net


def greedy_option(st: ts.GameState, mover: ts.Player, value_fn: ValueFn, cap: int = 16) -> Optional[Tuple[int, float, Dict[int, float]]]:
    """At a choice inside an event: each option's one-step critic reading for the mover; the best
    option, its gain over the model's (filled in by the caller), and the readings."""
    legal = [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))][:cap]
    if len(legal) < 2:
        return None
    obs, masks, signs, idx, vals = [], [], [], [], {}
    for a in legal:
        probe = st.clone()
        _step(probe, a)
        if ts.Engine.is_terminal(probe):
            u = float(ts.Engine.get_terminal_utility(probe)) * (1 if mover == ts.Player.US else -1)
            vals[a] = 1.0 if u > 0 else (0.0 if u < 0 else 0.5)
            continue
        d = _decider(probe)
        obs.append(np.asarray(ts.extract_observation(probe, d), dtype=np.float32))
        masks.append(np.asarray(ActionEncoder.get_legal_mask(probe), dtype=np.uint8))
        signs.append(1.0 if d == mover else -1.0)
        idx.append(a)
    if obs:
        v = value_fn(np.stack(obs), np.stack(masks))
        for a, sg, vv in zip(idx, signs, v):
            vals[a] = 0.5 + 0.5 * sg * float(vv)
    best = max(vals, key=lambda a: vals[a])
    return best, vals[best], vals


def _likeliest_other(probs: np.ndarray, mask: np.ndarray, exclude: set) -> Optional[int]:
    order = [int(a) for a in np.argsort(-probs) if mask[a] and int(a) < 110 and int(a) + 1 not in exclude]
    return order[0] if order else None


def spots_here(st: ts.GameState, mask: np.ndarray, greedy: int, probs: np.ndarray, act: PolicyFn,
               value_fn: ValueFn, want: Callable[[str], bool]) -> List[Dict[str, Any]]:
    """Every rule that applies here and that the model's move departs from (each rule asked only if
    `want(rule)`, so a caller can sample rules before paying for their probes)."""
    out: List[Dict[str, Any]] = []
    if ts.Engine.is_terminal(st) or st.current_phase != ts.Phase.ACTION_ROUND:
        return out
    ctx = st.ctx()
    mover = _decider(st)
    sign = 1 if mover == ts.Player.US else -1
    dt = ctx.decision_type
    cid = int(ctx.pending_op_card)

    def add(rule: str, prefix: List[int], feature: float, card: int = 0) -> None:
        if prefix and prefix[0] != greedy:
            out.append({"rule": rule, "prefix": prefix, "feature": float(feature), "card": card})

    lead = sign * int(st.victory_points)
    if want("decisive") and (lead >= 6 or (int(st.defcon) == 2 and lead > 6)):
        d = decisive(st, mover)
        if d is not None and not (len(d) == 1 and d[0] == greedy):
            # The model's own first move may already lead to the win; only a different first move counts.
            if d[0] != greedy:
                add("decisive", d, 0.0, cid)

    if dt == ts.DecisionType.SELECT_PLAY_MODE and cid:
        owner = _owner(cid, mover)
        if owner in ("own", "neutral") and mask[EVENT] and greedy != EVENT and want("event_vp"):
            add("event_vp", [EVENT], event_gain(st, mover, act), cid)
        if owner == "opp" and mask[SPACE] and greedy != SPACE and want("space_harmful"):
            c = event_cost(st, mover, act)
            if c is not None:
                add("space_harmful", [SPACE], c, cid)
        if (mask[COUP] and greedy != COUP and rounds_left(st) <= 1 and want("milops_coup")):
            mil = int(st.us_mil_ops) if mover == ts.Player.US else int(st.ussr_mil_ops)
            short = int(st.defcon) - mil
            if short >= 2 and not suicide(st, COUP):
                add("milops_coup", [COUP], short, cid)

    if is_round_start(st):
        hand = set(_hand(st, mover))
        if greedy < 110:
            g_card = greedy + 1
            if (_owner(g_card, mover) == "opp" and rounds_left(st) >= 1 and want("delay_harmful")):
                other = _likeliest_other(probs, mask, {g_card} | {c for c in hand if _owner(c, mover) == "opp"})
                if other is not None:
                    c = card_cost(st, g_card, mover, act)
                    if c is not None:
                        add("delay_harmful", [other], c, g_card)
            if g_card in SCORING_REGION and rounds_left(st) >= 2 and want("hold_score"):
                v = score_value(st, g_card, mover)
                if v is not None and v <= -1:
                    other = _likeliest_other(probs, mask, set(SCORING_REGION) | {38})
                    if other is not None:
                        add("hold_score", [other], v, g_card)
        if want("score_now"):
            for sc in sorted(hand & set(SCORING_REGION)):
                if mask[sc - 1] and greedy != sc - 1:
                    v = score_value(st, sc, mover)
                    if v is not None and v >= 1:
                        add("score_now", [sc - 1], v, sc)
                        break

    rc = int(ctx.resolving_card)
    if rc and mask.sum() >= 2 and want("greedy_choice"):
        g = greedy_option(st, mover, value_fn)
        if g is not None:
            best, vbest, vals = g
            if best != greedy and greedy in vals:
                add("greedy_choice", [best], 100.0 * (vbest - vals[greedy]), rc)
    return out


def collect(act: PolicyFn, probs_fn: Callable[[ts.GameState], np.ndarray], value_fn: ValueFn, n_games: int,
            seed: int, rate: float = 0.25, cap: int = 400, envs: int = 32,
            max_steps: int = 5_000_000) -> Tuple[List[Dict[str, Any]], int]:
    """Spots from `n_games` greedy self-play games. Each rule is asked at a matching decision with
    probability `rate` (so its probes are paid on a sample), and at most `cap` spots are kept per
    rule, drawn uniformly from the finished games' spots."""
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(envs, seed * 43 + 29)
    runner.refresh_all()
    pool: Dict[int, List[Dict[str, Any]]] = {}
    game = list(range(envs))
    done: List[int] = []
    started, finished = envs, 0
    for _ in range(max_steps):
        if finished >= n_games:
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        acts = act(obs, masks)
        for i in range(envs):
            m = masks[i]
            if m.sum() < 2:
                continue
            draws = {r: rng.random() for r in RULES}
            if min(draws.values()) >= rate:
                continue
            st = runner.get_state(i)
            want = lambda r: draws[r] < rate  # noqa: E731
            for sp in spots_here(st, m.astype(bool), int(acts[i]), probs_fn(st), act, value_fn, want):
                sp.update(state=st.clone(), mover=_decider(st), side="US" if _decider(st) == ts.Player.US else "USSR",
                          turn=int(st.turn))
                pool.setdefault(game[i], []).append(sp)
        runner.step_flat_all([int(x) for x in acts], auto_advance=True)
        ends = np.flatnonzero(np.array(runner.get_terminals())).tolist()
        for i in ends:
            done.append(game[i])
            finished += 1
            runner.reset_game(int(i), seed * 1_000_003 + started)
            game[i] = started
            started += 1
        if ends:
            runner.refresh_all()
    spots = [sp for g in done for sp in pool.get(g, [])]
    out: List[Dict[str, Any]] = []
    for r in RULES:
        sel = [sp for sp in spots if sp["rule"] == r]
        if len(sel) > cap:
            sel = [sel[j] for j in sorted(rng.choice(len(sel), cap, replace=False))]
        out += sel
    return out, len(done)


def _lenient(state: ts.GameState, prefix: Sequence[int]) -> ts.GameState:
    """`prefix` applied while each step is legal in this (redealt) world; see position_bank.apply_lenient."""
    st = state.clone()
    for a in prefix:
        if ts.Engine.is_terminal(st) or not np.asarray(ActionEncoder.get_legal_mask(st))[int(a)]:
            break
        _step(st, int(a))
    return st


def play(act: PolicyFn, spots: Sequence[Dict[str, Any]], pairs: int, seed: int, chunk: int = 1536
         ) -> List[Dict[str, Any]]:
    """Each spot's two branches, the model's move and the rule's, over `pairs` paired playouts."""
    rows: List[Dict[str, Any]] = []
    per = 2 * pairs
    step = max(1, chunk // per)
    for lo in range(0, len(spots), step):
        group = spots[lo:lo + step]
        starts, movers, keys = [], [], []
        for gi, sp in enumerate(group):
            st = sp["state"]
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo + gi, "resample")
                starts.append(base.clone())
                starts.append(_lenient(base, sp["prefix"]))
                movers += [sp["mover"], sp["mover"]]
                keys += [_ar_key(st), _ar_key(st)]
        res = np.array(play_safe(starts, movers, keys, act, seed + lo)).reshape(len(group), pairs, 2)
        for sp, sc in zip(group, res):
            d = sc[:, 1] - sc[:, 0]
            rows.append({k: v for k, v in sp.items() if k not in ("state", "mover")} | {
                "pairs": pairs, "model": float(sc[:, 0].mean()), "rule_score": float(sc[:, 1].mean()),
                "diff": float(d.mean()), "diff_se": float(d.std(ddof=1) / math.sqrt(pairs)),
                "era": next(nm for a, b, nm in ERAS if a <= max(1, min(sp["turn"], 10)) <= b),
                "link": position_link(sp["state"])})
    return rows


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def _fmt(xs: Sequence[float]) -> str:
    if len(xs) < 2:
        return f"— ({len(xs)})"
    m, se = _mean_se(xs)
    return f"{100 * m:+.1f} ± {100 * se:.1f} ({len(xs)})"


def report(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    info = cards()
    out = [f"# Rule oracle — {meta.get('model', '?')}", "",
           f"{len(rows)} spots from {meta.get('games', '?')} greedy self-play games, {meta.get('pairs', '?')} paired "
           "playouts per branch. Each spot: the model departs from the rule; rule − model is the rule's move minus the "
           "model's, in win-rate points (+ means the rule is better), ± one standard error over spots.", "",
           "| rule | spots | rule − model | US | USSR | Early | Mid | Late | without its top card |",
           "|:---|---:|---:|---:|---:|---:|---:|---:|:---|"]
    summary: Dict[str, Any] = {"meta": meta, "rules": {}}
    for r in RULES:
        sel = [x for x in rows if x["rule"] == r]
        if not sel:
            out.append(f"| {r} | 0 | — | — | — | — | — | — | — |")
            continue
        d = [x["diff"] for x in sel]
        cards_n: Dict[int, int] = {}
        for x in sel:
            cards_n[x["card"]] = cards_n.get(x["card"], 0) + 1
        top = max(cards_n, key=lambda c: cards_n[c])
        rest = [x["diff"] for x in sel if x["card"] != top]
        top_name = str(info[top]["name"]) if top in info else "—"
        m, se = _mean_se(d)
        out.append(f"| {r} | {len(sel)} | **{100 * m:+.1f} ± {100 * se:.1f}** | "
                   + " | ".join(_fmt([x["diff"] for x in sel if f(x)]) for f in (
                       lambda x: x["side"] == "US", lambda x: x["side"] == "USSR",
                       lambda x: x["era"] == "Early War", lambda x: x["era"] == "Mid War",
                       lambda x: x["era"] == "Late War"))
                   + f" | {_fmt(rest)} without {top_name} ({cards_n[top]} spots) |")
        summary["rules"][r] = {"n": len(sel), "diff": (m, se), "top_card": top_name, "without_top": _mean_se(rest),
                               "cards": len(cards_n)}
    out.append("")
    for r in RULES:
        sel = [x for x in rows if x["rule"] == r]
        if not sel or r == "decisive":
            continue
        out += [f"## `{r}` by dose", "", "| feature | spots | rule − model | cards |", "|:---|---:|---:|---:|"]
        for lo_, hi_, lab in BINS[r]:
            b = [x for x in sel if lo_ <= x["feature"] <= hi_ or (r == "greedy_choice" and lo_ <= x["feature"] < hi_)]
            out.append(f"| {lab} | {len(b)} | {_fmt([x['diff'] for x in b])} | {len({x['card'] for x in b})} |")
        out.append("")
    dec = [x for x in rows if x["rule"] == "decisive"]
    if dec:
        out += ["## `decisive`: the spots", "", "| side | turn | card | rule − model | |", "|:---|---:|:---|---:|:---|"]
        for x in sorted(dec, key=lambda x: -x["diff"])[:15]:
            out.append(f"| {x['side']} | {x['turn']} | {info.get(x['card'], {'name': '—'})['name']} | "
                       f"{100 * x['diff']:+.0f} | [position]({x['link']}) |")
        out.append("")
    return "\n".join(out) + "\n", summary
