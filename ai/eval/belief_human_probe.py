"""Does the policy-likelihood belief transfer to an opponent that is not our network?

The belief root (P35 3b, `ai.search.belief`) weighs hands by our network's probability of the
opponent's moves -- exact in self-play, a model of anyone else. Here the opponent is a human: on the
ts-replayer corpus (every decision rebuilt in the engine, the hands known), the posterior over the
opponent's unseen hand is scored against the truth, like P35a on self-play, at several likelihood
temperatures (a flatter likelihood trusts the model of the opponent less).

Only card and play-mode choices are conditioned on: the logs write a play's influence points in an
order of their own (`ai.eval.agreement.group_point_runs`), so point-by-point likelihoods would score
the log's order, not the human's choice. On self-play the same restriction (256 hands) gained 7.6%
over the prior (P35a), the reference for the number printed here.

    PYTHONPATH=.:build/release python -m ai.eval.belief_human_probe --checkpoint swa.pt
"""

from __future__ import annotations

import argparse
import gzip
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.belief import TRANSIENT, Decision, Now, card_locations, log_weights, posterior, record, sample_hands
from tools.lib.corpus_paths import distinct_corpus_files, missing_corpus_reason
from tools.lib.player_agent import NeuralAgent
from tools.lib.ts_replayer_convert import convert_game


def game_decisions(path: str) -> List[Decision]:
    """Every decision the converter emits for one replay, as the human faced it (empty on failure)."""
    with gzip.open(path, "rt") as fh:
        game = json.load(fh)
    if not game.get("all_turns"):
        return []
    out: List[Decision] = []

    def on_decision(state: ts.GameState, mover: ts.Player, _entry: object, chosen: int) -> None:
        out.append(Decision(state, 1 if mover == ts.Player.US else 2, int(chosen), card_locations(state)))

    conv = convert_game(game, on_decision=on_decision)
    if conv.skipped is not None or conv.failure is not None:
        return []
    return out


def positions(decisions: List[Decision], every: int, rng: np.random.Generator) -> List[Tuple[Now, List[Decision], np.ndarray]]:
    """One decision in `every` where the decider has an unseen opposing hand: its view, the turn's
    earlier decisions, and the true unseen hand (110 booleans)."""
    out: List[Tuple[Now, List[Decision], np.ndarray]] = []
    turn_start = 0
    for i, d in enumerate(decisions):
        if i and int(d.state.turn) != int(decisions[i - 1].state.turn):
            turn_start = i
        if rng.random() >= 1.0 / every or np.isin(d.locs, TRANSIENT).any():
            continue
        now = Now.of(d.state, d.decider)
        if now.hand == 0:
            continue
        opp_unknown = (d.locs == (int(ts.CardLocation.HAND_USSR_UNKNOWN) if d.decider == 1
                                  else int(ts.CardLocation.HAND_US_UNKNOWN)))
        out.append((now, decisions[turn_start:i], opp_unknown))
    return out


def log_loss(p: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    q = np.clip(p[m], 1e-6, 1 - 1e-6)
    yy = y[m]
    return float(-(yy * np.log(q) + (1 - yy) * np.log(1 - q)).mean())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--every", type=int, default=3, help="sample one decision in this many")
    ap.add_argument("--hands", type=int, default=256, help="hands per position (P35a's matched reference: 256)")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--limit", type=int, default=None, help="only the first N replays")
    ap.add_argument("--seed", type=int, default=9092)
    a = ap.parse_args(argv)
    reason = missing_corpus_reason()
    if reason is not None:
        raise SystemExit(reason)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    rng = np.random.default_rng(a.seed)
    paths, _ = distinct_corpus_files()
    if a.limit is not None:
        paths = paths[:a.limit]
    jobs: List[Tuple[Now, np.ndarray, list]] = []
    truth: List[np.ndarray] = []
    games = failed = 0
    for path in paths:
        decs = game_decisions(str(path))
        if not decs:
            failed += 1
            continue
        games += 1
        for now, log, true_hand in positions(decs, a.every, rng):
            jobs.append((now, sample_hands(now, a.hands, rng), record(log, now, False)))
            truth.append(true_hand)
    y = np.stack(truth).astype(np.float64)
    m = np.stack([now.pool for now, _, _ in jobs])
    prior = np.stack([np.where(now.pool, now.hand / max(int(now.pool.sum()), 1), 0.0) for now, _, _ in jobs])
    used = np.asarray([len(rec) > 0 for _, _, rec in jobs])
    base = log_loss(prior, y, m)
    print(f"checkpoint {a.checkpoint}: {games} human games ({failed} without a conversion), {len(jobs):,} positions "
          f"(one decision in {a.every}), {100 * used.mean():.0f}% with an opposing card or play-mode choice this turn; "
          f"{a.hands} hands; pool entries {int(m.sum()):,}, {100 * y[m].mean():.1f}% in the opponent's hand")
    print(f"  prior: hand size / pool size          log loss {base:.4f}  (with a choice {log_loss(prior[used], y[used], m[used]):.4f})")
    for t in a.temperatures:
        post = np.zeros_like(prior)
        ess: List[float] = []
        for lo in range(0, len(jobs), 256):
            chunk = jobs[lo:lo + 256]
            for k, (job, lw) in enumerate(zip(chunk, log_weights(model, dev, chunk, temperature=t))):
                p, e = posterior(job[1].astype(np.float64), lw)
                post[lo + k] = p
                ess.append(e)
        ll = log_loss(post, y, m)
        print(f"  posterior at T={t:<4g}                  log loss {ll:.4f} ({100 * (base - ll) / base:+5.1f}% vs the prior; "
              f"with a choice {log_loss(post[used], y[used], m[used]):.4f}); effective hands median "
              f"{np.median(np.asarray(ess)[used]):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
