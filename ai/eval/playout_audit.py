"""Where does the model's own playout judgment disagree with what it plays? A scan, not a probe.

The choice oracle tests named alternatives at named decisions. This tests *every* alternative at
a sample of ordinary decisions, so the disagreements find themselves:

* `mode` -- a play-mode decision with two or more legal modes (event / space / influence / coup /
  realign): every legal mode is a branch;
* `card` -- the first decision of an action round, the card to play: the model's own card and the
  next most likely ones (all of them when the hand is small) are branches.

Each branch forces its choice and the model plays on: the rest of the action round without
suicide (an option that loses on the spot is never taken while another exists -- the
untrained-follow-up guard of `choice_oracle.resolve_safely`), then the game on both sides. Pair k
redeals the cards the mover cannot see and uses the same dice in every branch
(`branch_oracle._pair_start`), so branches differ only in the decision.

Two outputs:

* **pooled**, by card and side: what each alternative scores against the model's choice, over
  every situation it was legal in. Cheap and tight -- the way OPEC's +2.4 showed up;
* **per situation**, a split-half regret: the best branch is picked on the even pairs and scored
  on the odd ones (and the reverse), minus the model's choice on the same pairs. Picking the best
  of several noisy branches and scoring it on the same pairs would always find a gain; this does
  not. The largest are re-tested with fresh dice by `confirm`.

Every verdict is the model's judgment of its own alternatives -- a choice whose value lies in a
follow-up the model does not find scores as worse than it is.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _decider, _pair_start, apply_prefix
from ai.eval.choice_oracle import _loses_now
from ai.eval.doctrine_census import MODES, N_CARDS, cards
from ai.eval.reply_probe import _ar_key, is_round_start, position_link
from bindings.action_encoder import ActionEncoder

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
KINDS = ("mode", "card")


def kind_of(st: ts.GameState, mask: np.ndarray) -> Optional[str]:
    """Which audited decision this is, if any."""
    if ts.Engine.is_terminal(st) or st.current_phase != ts.Phase.ACTION_ROUND:
        return None
    ctx = st.ctx()
    if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and mask[MODE_BASE:MODE_BASE + 5].sum() >= 2:
        return "mode"
    if is_round_start(st) and mask[:N_CARDS].sum() >= 2:
        return "card"
    return None


def collect(act: PolicyFn, n: int, seed: int, envs: int = 32, accept: float = 0.02, per_game: int = 4,
            max_steps: int = 3_000_000) -> List[Tuple[ts.GameState, str]]:
    """`n` audited decisions drawn uniformly from finished games of the model's greedy self-play:
    each decision kept with probability `accept` (at most `per_game` from one game) until `n // 2`
    games have finished, then `n` drawn from those. Only finished games count, so the sample is
    not tilted toward the early turns every unfinished game has reached."""
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(envs, seed * 31 + 7)
    runner.refresh_all()
    pool: Dict[int, List[Tuple[ts.GameState, str]]] = {}
    game = list(range(envs))
    taken = [0] * envs
    started, finished = envs, 0
    done: List[int] = []
    for _ in range(max_steps):
        if finished >= max(1, n // 2):
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        for i in range(envs):
            m = masks[i]
            if taken[i] >= per_game or not (m[:N_CARDS].any() or m[MODE_BASE:MODE_BASE + 5].any()):
                continue
            if rng.random() >= accept:
                continue
            st = runner.get_state(i)
            k = kind_of(st, m)
            if k is not None:
                pool.setdefault(game[i], []).append((st.clone(), k))
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


def branches(st: ts.GameState, kind: str, probs: np.ndarray, card_branches: int = 4
             ) -> Tuple[Dict[str, int], str]:
    """{branch name: forced action} and the name of the model's own (greedy) choice."""
    info = cards()
    mask = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
    if kind == "mode":
        legal = [MODE_BASE + k for k in range(5) if mask[MODE_BASE + k]]
        names = {MODES[a - MODE_BASE]: a for a in legal}
    else:
        legal = [a for a in np.flatnonzero(mask[:N_CARDS])]
        legal.sort(key=lambda a: -probs[a])
        names = {str(info[int(a) + 1]["name"]): int(a) for a in legal[:card_branches]}
    greedy = int(np.argmax(np.where(mask, probs, -1.0)))
    chosen = next(nm for nm, a in names.items() if a == greedy)
    return names, chosen


def play_safe(starts: Sequence[ts.GameState], movers: Sequence[ts.Player], keys: Sequence[Tuple[int, int, int, int]],
              act: PolicyFn, seed: int, max_steps: int = 4000) -> List[float]:
    """Play every state to the end with the model on both sides; through the rest of the mover's
    action round an option that loses on the spot is never taken while another exists. Returns the
    mover's score per state (1 win, 0.5 draw, 0 loss).

    `movers` and `keys` (the action round) come from the position *before* the forced choice: a
    choice that resolves on its own (a scoring card) hands the next decision to the opponent, and
    reading the mover off the started state scored those branches from the wrong side."""
    n = len(starts)
    runner = ts.VectorizedBatchRunner(n, seed)
    for i, st in enumerate(starts):
        runner.set_state(i, st)
    runner.refresh_all()
    guarding = [True] * n
    active = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        active &= ~np.array(runner.get_terminals())
        if not active.any():
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks()).copy()
        rows = np.flatnonzero(active)
        for i in (int(x) for x in rows):
            if not guarding[i]:
                continue
            st = runner.get_state(i)
            if _ar_key(st) != keys[i] or _decider(st) != movers[i]:
                guarding[i] = False
                continue
            if int(st.defcon) > 2:
                continue                 # nothing one decision can do loses on the spot above DEFCON 2
            legal = np.flatnonzero(masks[i])
            safe = [a for a in legal if not _loses_now(st, int(a), movers[i])]
            if safe and len(safe) < len(legal):
                masks[i] = 0
                masks[i, safe] = 1
        acts = np.zeros(n, dtype=np.int32)
        acts[rows] = act(obs[rows], masks[rows])
        res = runner.step_flat_all(acts.tolist(), auto_advance=True)
        refused = [i for i in rows if res[i] == 0]
        if refused:
            raise RuntimeError(f"the engine refused a playout action in game {refused[0]}")
    out = []
    for i in range(n):
        u = float(ts.Engine.get_terminal_utility(runner.get_state(i)))
        u = u if movers[i] == ts.Player.US else -u
        out.append(1.0 if u > 0 else (0.5 if u == 0 else 0.0))
    return out


def split_half_regret(scores: np.ndarray, chosen: int) -> float:
    """scores[pairs, branches]. The best branch picked on one half of the pairs, scored on the
    other, minus the model's choice on that half; both ways round, averaged."""
    even, odd = scores[0::2], scores[1::2]
    total = 0.0
    for pick, judge in ((even, odd), (odd, even)):
        b = int(np.argmax(pick.mean(axis=0)))
        total += float(judge[:, b].mean() - judge[:, chosen].mean())
    return total / 2.0


def audit(act: PolicyFn, probs_fn: Any, positions: Sequence[Tuple[ts.GameState, str]], pairs: int, seed: int,
          card_branches: int = 4, chunk: int = 1536) -> List[Dict[str, Any]]:
    """Every position's branches over `pairs` paired playouts; one row per position."""
    info = cards()
    rows: List[Dict[str, Any]] = []
    plan = []
    for st, kind in positions:
        names, chosen = branches(st, kind, probs_fn(st), card_branches)
        plan.append((st, kind, names, chosen))
    lo = 0
    while lo < len(plan):
        group, size = [], 0
        while lo < len(plan) and (not group or size + len(plan[lo][2]) * pairs <= chunk):
            group.append(plan[lo])
            size += len(plan[lo][2]) * pairs
            lo += 1
        starts: List[ts.GameState] = []
        movers: List[ts.Player] = []
        keys: List[Tuple[int, int, int, int]] = []
        for gi, (st, _, names, _) in enumerate(group):
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo * 7 + gi, "resample")
                for a in names.values():
                    starts.append(apply_prefix(base, [a]))
                    movers.append(_decider(st))
                    keys.append(_ar_key(st))
        flat = play_safe(starts, movers, keys, act, seed + lo)
        at = 0
        for st, kind, names, chosen in group:
            nb = len(names)
            sc = np.array(flat[at:at + nb * pairs]).reshape(pairs, nb)
            at += nb * pairs
            keys = list(names)
            ci = keys.index(chosen)
            mover = _decider(st)
            row: Dict[str, Any] = {
                "kind": kind, "side": "US" if mover == ts.Player.US else "USSR", "turn": int(st.turn),
                "ar": int(st.action_round), "defcon": int(st.defcon),
                "vp": int(st.victory_points) * (1 if mover == ts.Player.US else -1),
                "space": [int(st.us_space_track), int(st.ussr_space_track)][:: 1 if mover == ts.Player.US else -1],
                "chosen": chosen, "pairs": pairs, "scores": {nm: float(sc[:, j].mean()) for j, nm in enumerate(keys)},
                "se": {nm: _mean_se(sc[:, j] - sc[:, ci])[1] for j, nm in enumerate(keys) if j != ci},
                "regret": split_half_regret(sc, ci), "save": st.to_save_json(),
            }
            if kind == "mode":
                row["card"] = str(info.get(int(st.ctx().pending_op_card), {"name": "?"})["name"])
            rows.append(row)
    return rows


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def pooled(rows: Sequence[Dict[str, Any]], min_n: int = 15) -> List[Dict[str, Any]]:
    """Per (kind, card, side, alternative): the alternative's score minus the model's choice, over
    the situations where it was a branch and not the model's choice."""
    acc: Dict[Tuple[str, str, str, str], List[float]] = {}
    for r in rows:
        base = r["scores"][r["chosen"]]
        for nm, s in r["scores"].items():
            if nm == r["chosen"]:
                continue
            if r["kind"] == "mode":
                key = ("mode", r["card"], r["side"], nm)            # play this card for `nm`
            else:
                key = ("card", nm, r["side"], "play now")           # play card `nm` now
            acc.setdefault(key, []).append(s - base)
    out = []
    for (kind, card, side, alt), d in acc.items():
        if len(d) < min_n:
            continue
        m, se = _mean_se(d)
        out.append({"kind": kind, "card": card, "side": side, "alt": alt, "n": len(d), "gain": m, "se": se})
    out.sort(key=lambda x: -(x["gain"] / x["se"] if x["se"] and x["se"] > 0 else 0.0))
    return out


def report(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any], top: int = 30) -> Tuple[str, Dict[str, Any]]:
    out = [f"# Playout audit — {meta.get('model', '?')}", "",
           f"{len(rows)} decisions from the model's greedy self-play, every alternative played out "
           f"{meta.get('pairs', '?')} times with the unseen cards redealt per pair and the same dice in "
           f"every branch. Scores are the mover's win % (draw ½); ± one standard error.", ""]
    summary: Dict[str, Any] = {"meta": meta, "n": len(rows)}
    out += ["## How often its own playouts disagree", "",
            "Split-half regret: the playout-best branch picked on half the pairs and scored on the other "
            "half, against the model's choice. Unbiased by the choice of the best; its mean is the "
            "win % the model leaves on the table by its own judgment.", "",
            "| decision | situations | mean regret | regret > 5 pts | regret > 10 pts |", "|:---|---:|---:|---:|---:|"]
    for kind in KINDS:
        sel = [r["regret"] for r in rows if r["kind"] == kind]
        if not sel:
            continue
        m, se = _mean_se(sel)
        out.append(f"| {kind} | {len(sel)} | {100 * m:+.2f} ± {100 * se:.2f} | "
                   f"{100 * np.mean([x > 0.05 for x in sel]):.0f}% | {100 * np.mean([x > 0.10 for x in sel]):.0f}% |")
        summary[kind] = {"n": len(sel), "regret": (m, se)}
    pool = pooled(rows)
    summary["pooled"] = pool
    for kind, title, note in (
            ("mode", "Play-mode decisions: alternatives that beat the model's choice, pooled by card",
             "Each row: playing this card this way instead of the model's choice, over every situation the "
             "model chose otherwise. Sorted by gain over its standard error."),
            ("card", "Card choice: cards that should have been played now, pooled",
             "Each row: playing this card at the start of the action round instead of the model's card "
             "(the card's mode and targets then the model's). Timing shows up here.")):
        sel = [p for p in pool if p["kind"] == kind]
        out += [f"## {title}", "", note, "", "| card | side | alternative | situations | gain | z |",
                "|:---|:---|:---|---:|---:|---:|"]
        for p in sel[:top]:
            z = p["gain"] / p["se"] if p["se"] and p["se"] > 0 else float("nan")
            out.append(f"| {p['card']} | {p['side']} | {p['alt']} | {p['n']} | "
                       f"{100 * p['gain']:+.1f} ± {100 * p['se']:.1f} | {z:+.1f} |")
        out.append("")
    worst = sorted(rows, key=lambda r: -r["regret"])[:top]
    out += [f"## The {top} largest single-situation regrets (to confirm)", "",
            "| kind | side | turn | DEFCON | card | model's choice | playout-best | regret | |",
            "|:---|:---|---:|---:|:---|:---|:---|---:|:---|"]
    for r in worst:
        best = max(r["scores"], key=lambda k: r["scores"][k])
        st = ts.state_from_save_json(r["save"])
        out.append(f"| {r['kind']} | {r['side']} | {r['turn']}.{r['ar']} | {r['defcon']} | {r.get('card', '—')} | "
                   f"{r['chosen']} | {best} | {100 * r['regret']:+.0f} | [position]({position_link(st)}) |")
    out.append("")
    return "\n".join(out) + "\n", summary


def confirm(act: PolicyFn, probs_fn: Any, rows: Sequence[Dict[str, Any]], pairs: int, seed: int
            ) -> List[Dict[str, Any]]:
    """Re-test situations with fresh dice and redeals: each branch against the model's choice."""
    positions = [(ts.state_from_save_json(r["save"]), str(r["kind"])) for r in rows]
    again = audit(act, probs_fn, positions, pairs, seed + 1_000_000)
    out = []
    for r, a in zip(rows, again):
        best = max(r["scores"], key=lambda k: r["scores"][k])
        out.append({**a, "screen_best": best, "screen_regret": r["regret"],
                    "confirmed_gain": a["scores"].get(best, float("nan")) - a["scores"][a["chosen"]]})
    return out
