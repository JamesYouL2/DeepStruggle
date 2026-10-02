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
from ai.eval.reply_probe import _ar_key
from bindings.action_encoder import ActionEncoder

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
#: Relative sampling weight per kind of decision (influence points are the bulk of decisions and the
#: model's strength, so they are thinned; the weak kinds are oversampled).
KIND_WEIGHT: Dict[str, float] = {"event choice": 5.0, "headline": 4.0, "mode": 3.0, "other": 3.0, "setup": 2.0,
                                 "card": 2.0, "realign": 2.0, "coup": 1.0, "influence": 0.4}
ProbsFn = Callable[[ts.GameState], np.ndarray]


def collect(act: PolicyFn, n: int, seed: int, envs: int = 32, base: float = 0.004, per_game: int = 8,
            max_steps: int = 5_000_000) -> List[Tuple[ts.GameState, str]]:
    """`n` decisions from finished games, each kind kept with probability base × its weight."""
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
            if taken[i] >= per_game or masks[i].sum() < 2 or draws[i] >= base * 5.0:
                continue
            st = runner.get_state(i)
            k = kind_of(st)
            if draws[i] < base * KIND_WEIGHT.get(k, 1.0):
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


def label(act: PolicyFn, probs_fn: ProbsFn, positions: Sequence[Tuple[ts.GameState, str]], pairs: int, seed: int,
          k: int = 4, chunk: int = 1536) -> List[Dict[str, Any]]:
    """One label per position (positions where only one candidate survives are dropped)."""
    plan = []
    for st, kind in positions:
        p = probs_fn(st)
        cands = candidates(st, kind, p, k)
        if len(cands) >= 2:
            plan.append((st, kind, cands, p))
    out: List[Dict[str, Any]] = []
    lo = 0
    while lo < len(plan):
        group, size = [], 0
        while lo < len(plan) and (not group or size + len(plan[lo][2]) * pairs <= chunk):
            group.append(plan[lo])
            size += len(plan[lo][2]) * pairs
            lo += 1
        starts, movers, keys = [], [], []
        for gi, (st, _, cands, _) in enumerate(group):
            for j in range(pairs):
                base = _pair_start(st, j, seed * 100_003 + lo * 7 + gi, "resample")
                for a in cands:
                    starts.append(apply_prefix(base, [a]))
                    movers.append(_decider(st))
                    keys.append(_ar_key(st))
        flat = play_safe(starts, movers, keys, act, seed + lo)
        at = 0
        for st, kind, cands, p in group:
            nb = len(cands)
            sc = np.array(flat[at:at + nb * pairs]).reshape(pairs, nb)
            at += nb * pairs
            mover = _decider(st)
            out.append({
                "obs": np.asarray(ts.extract_observation(st, mover), dtype=np.float32),
                "mask": np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8),
                "cands": cands, "q": sc.mean(axis=0), "q_se": sc.std(axis=0, ddof=1) / math.sqrt(pairs),
                "prior": np.array([p[a] for a in cands], dtype=np.float32),
                "regret": _split_half(sc), "kind": kind, "side": int(mover == ts.Player.US), "turn": int(st.turn),
            })
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
    q_se = np.zeros((n, k_max), dtype=np.float32)
    prior = np.zeros((n, k_max), dtype=np.float32)
    for i, lb in enumerate(labels):
        m = min(k_max, len(lb["cands"]))
        cands[i, :m] = lb["cands"][:m]
        q[i, :m] = lb["q"][:m]
        q_se[i, :m] = lb["q_se"][:m]
        prior[i, :m] = lb["prior"][:m]
    return {
        "obs": np.stack([lb["obs"] for lb in labels]).astype(np.float16) if n else np.zeros((0, ts.OBS_SIZE), np.float16),
        "mask": np.packbits(np.stack([lb["mask"] for lb in labels]), axis=1) if n else np.zeros((0, 28), np.uint8),
        "cands": cands, "q": q, "q_se": q_se, "prior": prior,
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


def report(arrays: Dict[str, np.ndarray], meta: Dict[str, Any], taus: Sequence[float] = (0.02, 0.05, 0.1)
           ) -> Tuple[str, Dict[str, Any]]:
    """What the labels would teach: per kind, how often the playout-best candidate is not the
    model's, the split-half regret, and how far a soft target at each τ moves from the prior."""
    n = len(arrays["kind"])
    out = [f"# Playout labels — {meta.get('model', '?')}", "",
           f"{n} labelled decisions, {meta.get('pairs', '?')} paired playouts per candidate.", "",
           "| decision | labels | best ≠ model's | split-half regret | " +
           " | ".join(f"KL(π′‖π) at τ={t}" for t in taus) + " |",
           "|:---|---:|---:|---:|" + "---:|" * len(taus)]
    summary: Dict[str, Any] = {"meta": meta, "n": n, "kinds": {}}
    valid = arrays["cands"] >= 0
    for ki, kind in enumerate(KINDS):
        sel = arrays["kind"] == ki
        if not sel.any():
            continue
        q = np.where(valid[sel], arrays["q"][sel], -np.inf)
        best_not_model = float(np.mean(np.argmax(q, axis=1) != 0))
        reg = arrays["regret"][sel]
        kls = []
        for t in taus:
            pr = np.where(valid[sel], arrays["prior"][sel], 0.0)
            pr = pr / np.maximum(pr.sum(axis=1, keepdims=True), 1e-8)
            qq = np.where(valid[sel], arrays["q"][sel], 0.0)
            tgt = soft_target(qq, np.where(valid[sel], pr, 1e-12), t)
            kl = np.sum(np.where(valid[sel], tgt * (np.log(np.maximum(tgt, 1e-12)) - np.log(np.maximum(pr, 1e-12))), 0.0),
                        axis=1)
            kls.append(float(kl.mean()))
        se = float(reg.std(ddof=1) / math.sqrt(len(reg))) if len(reg) > 1 else float("nan")
        out.append(f"| {kind} | {int(sel.sum())} | {100 * best_not_model:.0f}% | {100 * reg.mean():+.2f} ± {100 * se:.2f} | " +
                   " | ".join(f"{x:.3f}" for x in kls) + " |")
        summary["kinds"][kind] = {"n": int(sel.sum()), "best_not_model": best_not_model,
                                  "regret": (float(reg.mean()), se), "kl": dict(zip(map(str, taus), kls))}
    reg = arrays["regret"]
    if len(reg) > 1:
        out += ["", f"All decisions: split-half regret {100 * reg.mean():+.2f} ± {100 * reg.std(ddof=1) / math.sqrt(len(reg)):.2f} "
                "points -- what the playout-best move gains over the model's own, unbiased by choosing the best."]
    return "\n".join(out) + "\n", summary
