"""Training labels from the model's own paired playouts: for sampled decisions, each candidate move's
score, so a policy can be fine-tuned toward π′ ∝ π·exp(Q/τ) (P3's operator) and relabelled.

The playout audit (`playout_audit.py`) measures the same thing at play-mode and card decisions for a
report; this writes it for training, at every kind of decision, and oversamples the kinds a strong
player's review found weak (`research/log/expert_review_E7.md` upstream): choices inside events,
headlines, the play mode, the card to play, setup, and the Ops after an opponent's event.

**A decision** is drawn uniformly from finished games of the model's greedy self-play, kept with a
probability that depends on its kind (`KIND_WEIGHT`), and only when two or more moves are legal.

**Candidates** are the model's `k` most likely moves; at a play-mode decision every legal mode,
except one whose every follow-up loses on the spot (`playout_audit.suicide`), unless it is the
model's own choice; at an action round's first decision, passing when it is legal (the eighth round).

**Each candidate** is forced and played out `pairs` times: the rest of the action round without
suicide (`playout_audit.play_safe`), then the game, the model on both sides. Pair k redeals the cards
the mover cannot see and uses the same dice for every candidate, so the candidates differ only in
the move. The score is the mover's win rate (draw ½), with its standard error over pairs.

**Output**, per decision: the observation from the mover's side, the legal-move mask, the
candidates, their scores and standard errors, the model's probabilities for them, and the kind,
side and turn. The labels value a move given how *this* model plays afterwards; relabelling after
each round of training is what lets a later round value follow-ups an earlier round taught.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _decider, _pair_start, apply_prefix
from ai.eval.human_disagree import KINDS, kind_of
from ai.eval.ops_block import CONFIRM_DONE
from ai.eval.playout_audit import play_safe, suicide
from ai.eval.reply_probe import _ar_key, is_round_start
from bindings.action_encoder import ActionEncoder

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
#: Relative sampling weight per kind of decision (influence points are the bulk of decisions and the
#: model's strength, so they are thinned; the weak kinds are oversampled).
KIND_WEIGHT: Dict[str, float] = {"event choice": 5.0, "headline": 4.0, "mode": 3.0, "other": 3.0, "setup": 2.0,
                                 "card": 2.0, "realign": 2.0, "coup": 1.0, "influence": 0.4}
ProbsFn = Callable[[ts.GameState], np.ndarray]


#: The targeted pilot's extra weight on the kinds of decision the review found weak.
TARGETED_KIND_WEIGHT: Dict[str, float] = {"event choice": 8.0, "headline": 3.0, "mode": 4.0, "other": 3.0,
                                          "setup": 1.0, "card": 3.0, "realign": 1.0, "coup": 0.5, "influence": 0.2}
#: Events whose inside choices the review found weak (by name, resolved to ids at first use).
WEAK_EVENT_CHOICES = ("Aldrich Ames Remix", "De-Stalinization", "Marshall Plan", "Star Wars", "Che",
                      "Warsaw Pact Formed", "Comecon", "Suez Crisis", "Decolonization")


def _ids(names: Sequence[str]) -> set:
    from ai.eval.doctrine_census import cards
    by = {str(v["name"]): k for k, v in cards().items()}
    return {by[n] for n in names}


def leak_spot(st: ts.GameState) -> bool:
    """A decision of a kind where a focused test found the model leaking: Wargames at a winning
    lead, OPEC or Alliance for Progress worth 5+ VP, Star Wars while ahead in space, the timing of
    Five Year Plan (USSR) and Aldrich Ames Remix (US), and the choices inside the weak events."""
    from ai.eval.doctrine_census import (ALDRICH_AMES, ALLIANCE_FOR_PROGRESS, FIVE_YEAR_PLAN, OPEC, STAR_WARS,
                                         _hand, card_context)
    ctx = st.ctx()
    me = _decider(st)
    us = me == ts.Player.US
    lead = int(st.victory_points) * (1 if us else -1)
    cid = int(ctx.pending_op_card)
    if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and cid:
        if cid == 100 and int(st.defcon) == 2 and lead > 6:
            return True
        if cid in (OPEC, ALLIANCE_FOR_PROGRESS) and card_context(st, cid).get("event_vp", 0) >= 5 \
                and us == (cid == ALLIANCE_FOR_PROGRESS):
            return True
        if cid == STAR_WARS and us and int(st.us_space_track) > int(st.ussr_space_track):
            return True
        if (cid == FIVE_YEAR_PLAN and not us) or (cid == ALDRICH_AMES and us):
            return True
    if is_round_start(st):
        hand = _hand(st, me)
        if (not us and FIVE_YEAR_PLAN in hand) or (us and ALDRICH_AMES in hand):
            return True
    rc = int(ctx.resolving_card)
    return bool(rc) and rc in _ids(WEAK_EVENT_CHOICES)


def collect(act: PolicyFn, n: int, seed: int, envs: int = 32, base: float = 0.004, per_game: int = 8,
            targeted: bool = False, leak_p: float = 0.35, max_steps: int = 5_000_000) -> List[Tuple[ts.GameState, str]]:
    """`n` decisions from finished games, each kept with probability base × its kind's weight; under
    `targeted`, the review's weights, and a leak spot (`leak_spot`) kept with probability `leak_p`."""
    weights = TARGETED_KIND_WEIGHT if targeted else KIND_WEIGHT
    top = max(weights.values())
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(envs, seed * 37 + 19)
    runner.refresh_all()
    pool: Dict[int, List[Tuple[ts.GameState, str]]] = {}
    game = list(range(envs))
    taken = [0] * envs
    done: List[int] = []
    started, finished = envs, 0
    want_games = max(1, int(math.ceil(n / (per_game * 0.6))))
    for _ in range(max_steps):
        if finished >= want_games and sum(len(pool.get(g, [])) for g in done) >= n:
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        draws = rng.random(envs)
        for i in range(envs):
            if taken[i] >= per_game or masks[i].sum() < 2:
                continue
            if draws[i] >= max(base * top, leak_p if targeted else 0.0):
                continue
            st = runner.get_state(i)
            k = kind_of(st)
            keep = draws[i] < base * weights.get(k, 1.0)
            if targeted and not keep and draws[i] < leak_p and leak_spot(st):
                keep = True
            if keep:
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


def candidates(st: ts.GameState, kind: str, probs: np.ndarray, k: int) -> List[int]:
    """The moves to label at `st`: the model's choice first, then the rest (see the module doc)."""
    mask = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
    greedy = int(np.argmax(np.where(mask, probs, -1.0)))
    if kind == "mode":
        rest = [MODE_BASE + j for j in range(5) if mask[MODE_BASE + j] and MODE_BASE + j != greedy
                and not suicide(st, MODE_BASE + j)]
    else:
        legal = [int(a) for a in np.flatnonzero(mask) if int(a) != greedy]
        legal.sort(key=lambda a: -probs[a])
        rest = legal[:max(0, k - 1)]
        if kind == "card" and mask[CONFIRM_DONE] and CONFIRM_DONE != greedy and CONFIRM_DONE not in rest:
            rest.append(CONFIRM_DONE)
    return [greedy] + rest


def _paired(sc: np.ndarray, j: int) -> Tuple[float, float, int]:
    """Mean and standard error of candidate j minus the model's choice (column 0), over the pairs
    where both were played."""
    ok = ~np.isnan(sc[:, j]) & ~np.isnan(sc[:, 0])
    d = sc[ok, j] - sc[ok, 0]
    if len(d) < 2:
        return (float(d.mean()) if len(d) else 0.0), float("inf"), int(len(d))
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))), int(len(d))


def _play_round(act: PolicyFn, work: Sequence[Tuple[int, ts.GameState, Sequence[int], Sequence[int]]], seed: int,
                pair_seed: Callable[[int], int], chunk: int) -> Dict[Tuple[int, int, int], float]:
    """Play (entry, pair, candidate) branches; `work` holds (entry id, state, pair indices, candidate
    actions). Pair k of an entry uses the same redeal and dice for every candidate."""
    jobs = [(e, st, k, a) for e, st, ks, cands in work for k in ks for a in cands]
    out: Dict[Tuple[int, int, int], float] = {}
    for lo in range(0, len(jobs), chunk):
        part = jobs[lo:lo + chunk]
        starts, movers, keys = [], [], []
        bases: Dict[Tuple[int, int], ts.GameState] = {}
        for e, st, k, a in part:
            if (e, k) not in bases:
                bases[(e, k)] = _pair_start(st, k, pair_seed(e), "resample")
            starts.append(apply_prefix(bases[(e, k)], [a]))
            movers.append(_decider(st))
            keys.append(_ar_key(st))
        res = play_safe(starts, movers, keys, act, seed + lo)
        for (e, st, k, a), r in zip(part, res):
            out[(e, k, a)] = r
    return out


def label(act: PolicyFn, probs_fn: ProbsFn, positions: Sequence[Tuple[ts.GameState, str]], pairs: int, seed: int,
          k: int = 4, adaptive: bool = False, batch: int = 8, pairs_max: int = 64, verify_pairs: int = 0,
          chunk: int = 1536) -> List[Dict[str, Any]]:
    """One label per position (positions where only one candidate survives are dropped).

    Fixed: every candidate `pairs` times. Adaptive: rounds of `batch` pairs up to `pairs_max`; after
    each round a candidate clearly worse than the model's choice (mean + 2 se < 0) is dropped, and
    the position stops once its best candidate is clearly better (mean − 3 se > 0) or none can be
    (every mean + 2 se < 1 point). A label is **confirmed** when its best candidate beats the
    model's choice by more than two standard errors (three when adaptive, since its gaps are looked
    at after every round). With `verify_pairs`, each confirmed label is
    re-played -- the model's choice against that best candidate -- on fresh redeals and dice."""
    entries: List[Dict[str, Any]] = []
    for st, kind in positions:
        p = probs_fn(st)
        cands = candidates(st, kind, p, k)
        if len(cands) >= 2:
            entries.append({"st": st, "kind": kind, "cands": cands, "p": p, "active": list(range(len(cands))),
                            "done": False, "sc": np.full((pairs_max if adaptive else pairs, len(cands)), np.nan)})
    pair_seed = lambda e: seed * 100_003 + e  # noqa: E731
    # Looking at a gap after every round is a test repeated up to pairs_max / batch times, which
    # inflates false confirmations; the adaptive variant asks for three standard errors, not two.
    z = 3.0 if adaptive else 2.0
    played = 0
    rounds = [(0, pairs)] if not adaptive else [(r, min(batch, pairs_max - r)) for r in range(0, pairs_max, batch)]
    for k0, nk in rounds:
        work = [(e, en["st"], range(k0, k0 + nk), [en["cands"][j] for j in en["active"]])
                for e, en in enumerate(entries) if not en["done"]]
        if not work:
            break
        res = _play_round(act, work, seed + k0, pair_seed, chunk)
        played += len(res)
        for e, en in enumerate(entries):
            if en["done"]:
                continue
            for kk in range(k0, k0 + nk):
                for j in en["active"]:
                    en["sc"][kk, j] = res[(e, kk, en["cands"][j])]
            if not adaptive:
                continue
            stats = {j: _paired(en["sc"], j) for j in en["active"] if j}
            en["active"] = [0] + [j for j in en["active"] if j and stats[j][0] + 2 * stats[j][1] >= 0]
            live = {j: stats[j] for j in en["active"] if j}
            if not live or max(m + 2 * se for m, se, _ in live.values()) < 0.01:
                en["done"] = True
            elif max(m - z * se for m, se, _ in live.values()) > 0:
                en["done"] = True
    out: List[Dict[str, Any]] = []
    for e, en in enumerate(entries):
        sc = en["sc"]
        st, cands = en["st"], en["cands"]
        gains = [_paired(sc, j) if j else (0.0, 0.0, int(np.sum(~np.isnan(sc[:, 0])))) for j in range(len(cands))]
        best = max(range(1, len(cands)), key=lambda j: gains[j][0] - z * gains[j][1])
        confirmed = gains[best][0] - z * gains[best][1] > 0
        full = sc[~np.isnan(sc).any(axis=1)]
        mover = _decider(st)
        out.append({
            "st": st, "obs": np.asarray(ts.extract_observation(st, mover), dtype=np.float32),
            "mask": np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8),
            "cands": cands, "q": np.nanmean(sc, axis=0), "gain": np.array([g[0] for g in gains]),
            "gain_se": np.array([g[1] for g in gains]), "prior": np.array([en["p"][a] for a in cands], dtype=np.float32),
            "pairs_used": int(np.sum(~np.isnan(sc))), "confirmed": bool(confirmed), "best": int(best),
            "regret": _split_half(full) if len(full) >= 4 else float("nan"),
            "kind": en["kind"], "side": int(mover == ts.Player.US), "turn": int(st.turn),
            "vgain": float("nan"), "vgain_se": float("nan"),
        })
    if verify_pairs:
        conf = [(i, lb) for i, lb in enumerate(out) if lb["confirmed"]]
        work = [(i, lb["st"], range(1_000_000, 1_000_000 + verify_pairs), [lb["cands"][0], lb["cands"][lb["best"]]])
                for i, lb in conf]
        res = _play_round(act, work, seed + 7, pair_seed, chunk) if work else {}
        for i, lb in conf:
            d = np.array([res[(i, kk, lb["cands"][lb["best"]])] - res[(i, kk, lb["cands"][0])]
                          for kk in range(1_000_000, 1_000_000 + verify_pairs)])
            lb["vgain"] = float(d.mean())
            lb["vgain_se"] = float(d.std(ddof=1) / math.sqrt(len(d)))
        played += len(res)
    for lb in out:
        lb.pop("st")
    return out


def _split_half(sc: np.ndarray) -> float:
    """Best candidate picked on one half of the pairs, scored on the other, minus the model's choice
    (column 0); both ways round, averaged."""
    tot = 0.0
    for pick, judge in ((sc[0::2], sc[1::2]), (sc[1::2], sc[0::2])):
        b = int(np.argmax(pick.mean(axis=0)))
        tot += float(judge[:, b].mean() - judge[:, 0].mean())
    return tot / 2.0


def pack(labels: Sequence[Dict[str, Any]], k_max: int = 5) -> Dict[str, np.ndarray]:
    """Arrays for `np.savez_compressed`: observations as float16, masks bit-packed, candidates padded
    with -1 to `k_max` columns."""
    n = len(labels)
    cands = np.full((n, k_max), -1, dtype=np.int16)
    q = np.zeros((n, k_max), dtype=np.float32)
    gain = np.zeros((n, k_max), dtype=np.float32)
    gain_se = np.zeros((n, k_max), dtype=np.float32)
    prior = np.zeros((n, k_max), dtype=np.float32)
    for i, lb in enumerate(labels):
        m = min(k_max, len(lb["cands"]))
        cands[i, :m] = lb["cands"][:m]
        q[i, :m] = lb["q"][:m]
        gain[i, :m] = lb["gain"][:m]
        gain_se[i, :m] = np.minimum(lb["gain_se"][:m], 9.0)
        prior[i, :m] = lb["prior"][:m]
    return {
        "obs": np.stack([lb["obs"] for lb in labels]).astype(np.float16) if n else np.zeros((0, ts.OBS_SIZE), np.float16),
        "mask": np.packbits(np.stack([lb["mask"] for lb in labels]), axis=1) if n else np.zeros((0, 28), np.uint8),
        "cands": cands, "q": q, "gain": gain, "gain_se": gain_se, "prior": prior,
        "pairs_used": np.array([lb["pairs_used"] for lb in labels], dtype=np.int32),
        "confirmed": np.array([lb["confirmed"] for lb in labels], dtype=bool),
        "best": np.array([lb["best"] for lb in labels], dtype=np.int8),
        "vgain": np.array([lb["vgain"] for lb in labels], dtype=np.float32),
        "vgain_se": np.array([lb["vgain_se"] for lb in labels], dtype=np.float32),
        "regret": np.array([lb["regret"] for lb in labels], dtype=np.float32),
        "kind": np.array([KINDS.index(lb["kind"]) for lb in labels], dtype=np.int8),
        "side": np.array([lb["side"] for lb in labels], dtype=np.int8),
        "turn": np.array([lb["turn"] for lb in labels], dtype=np.int8),
    }


def soft_target(q: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """π′ ∝ π·exp(Q/τ) over one decision's candidates (rows are decisions; -inf-free inputs)."""
    z = np.log(np.maximum(prior, 1e-8)) + q / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def shrunk_gain(gain: np.ndarray, gain_se: np.ndarray) -> np.ndarray:
    """Each candidate's gain over the model's choice, kept only where it exceeds two standard errors
    (else 0): the shrinkage rule, so a soft target moves only on evidence."""
    return np.where(gain - 2 * gain_se > 0, gain, 0.0)


def report(arrays: Dict[str, np.ndarray], meta: Dict[str, Any], tau: float = 0.1) -> Tuple[str, Dict[str, Any]]:
    """What the labels would teach, and how much of it holds up on fresh dice."""
    n = len(arrays["kind"])
    verify_pairs = int(meta.get("verify_pairs", 0))
    conf = arrays["confirmed"]
    label_playouts = int(arrays["pairs_used"].sum())
    verify_playouts = int(conf.sum()) * 2 * verify_pairs
    total = label_playouts + verify_playouts
    v = arrays["vgain"][conf]
    vse = arrays["vgain_se"][conf]
    out = [f"# Playout labels — {meta.get('model', '?')} ({meta.get('variant', '?')})", "",
           f"{n} labelled decisions; {label_playouts} labelling playouts ({label_playouts / max(1, n):.0f} per decision) "
           f"and {verify_playouts} verification playouts.", "",
           "A label is **confirmed** when its best candidate beats the model's choice by more than two standard "
           f"errors (the shrinkage rule). Each confirmed label is re-played on fresh redeals and dice, {verify_pairs} "
           "pairs: the **verified gain** is what holds up.", ""]
    summary: Dict[str, Any] = {"meta": meta, "n": n, "playouts": total, "kinds": {}}
    vm = float(np.nanmean(v)) if len(v) else float("nan")
    vs = float(np.nanstd(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    per10k = float(np.nansum(v)) / max(1, total) * 10_000
    out += ["| | value |", "|:---|---:|",
            f"| confirmed labels | {int(conf.sum())} ({100 * conf.mean():.1f}%) |",
            f"| claimed gain of confirmed (mean) | {100 * float(np.mean(arrays['gain'][conf, :].max(axis=1))) if conf.any() else float('nan'):.1f} pts |",
            f"| **verified gain of confirmed** | **{100 * vm:+.1f} ± {100 * vs:.1f} pts** |",
            f"| confirmed labels whose verified gain is positive | {100 * float(np.mean(v > 0)) if len(v) else float('nan'):.0f}% |",
            f"| **verified gain per 10,000 playouts** (points × labels) | **{100 * per10k:.2f}** |", ""]
    summary.update(confirmed=int(conf.sum()), verified_gain=(vm, vs), per10k=per10k)
    out += ["## By kind", "", "| decision | labels | playouts per label | confirmed | verified gain of confirmed | "
            f"KL(π′‖π), shrunk, τ={tau} |", "|:---|---:|---:|---:|---:|---:|"]
    valid = arrays["cands"] >= 0
    for ki, kind in enumerate(KINDS):
        sel = arrays["kind"] == ki
        if not sel.any():
            continue
        c = conf & sel
        vv = arrays["vgain"][c]
        pr = np.where(valid[sel], arrays["prior"][sel], 0.0)
        pr = pr / np.maximum(pr.sum(axis=1, keepdims=True), 1e-8)
        g = np.where(valid[sel], shrunk_gain(arrays["gain"][sel], arrays["gain_se"][sel]), 0.0)
        tgt = soft_target(g, np.where(valid[sel], pr, 1e-12), tau)
        kl = float(np.mean(np.sum(np.where(valid[sel], tgt * (np.log(np.maximum(tgt, 1e-12)) -
                                                           np.log(np.maximum(pr, 1e-12))), 0.0), axis=1)))
        vtxt = (f"{100 * np.nanmean(vv):+.1f} ± {100 * np.nanstd(vv, ddof=1) / math.sqrt(len(vv)):.1f}"
                if len(vv) > 1 else "—")
        out.append(f"| {kind} | {int(sel.sum())} | {arrays['pairs_used'][sel].mean():.0f} | "
                   f"{int(c.sum())} ({100 * c.sum() / sel.sum():.0f}%) | {vtxt} | {kl:.4f} |")
        summary["kinds"][kind] = {"n": int(sel.sum()), "confirmed": int(c.sum()),
                                  "verified": float(np.nanmean(vv)) if len(vv) else None, "kl": kl}
    return "\n".join(out) + "\n", summary
