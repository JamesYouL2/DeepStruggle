"""Paired-branch playout advantages: a low-noise policy-gradient signal at sampled decisions.

The trainer's credit for one decision is the game's result (through GAE), whose noise -- about
+-50 points of win rate -- dwarfs what a single card decision is worth (2-5 points at the spots the
expert review measured: One Small Step one box behind, KAL-007 with South Korea, Game-winning
Wargames). RL therefore leaves those decisions where the warm start put them, and the P30 card-event
target (C2) did not move them either (E7-72-45). A clean per-position signal does: distilling
paired-playout labels fixed the spots (research/log/expert_review_E7.md).

This module produces that signal inside RL, on the policy as it trains:

* **Sampling.** A small fraction of the learner's decisions of chosen kinds (by default choosing a
  card, choosing how to play it, and the choices inside events -- among them Wargames' "end the
  game") is recorded during the rollout, with the true GameState.
* **Candidates.** The policy's K likeliest legal actions there.
* **Paired playouts.** Every candidate is played from P copies of the true state; copy j of every
  candidate gets the same dice seed (`rng_state`), so the candidates differ only in the move --
  common random numbers, the noise of the comparison is far below that of independent games. By
  default (`hidden="true"`) the hidden cards stay as dealt, so for the policy, which only sees its
  own observation, Q(s, a) is an unbiased sample of the value of its information state -- but every
  pair shares that one hand, so its noise does not shrink with P. `hidden="redeal"` redeals them per
  pair instead (the same redeal for every candidate), as the expert review's oracles did: that
  noise then averages out, at the price of a uniform belief over the unseen cards.
* **Continuation and horizon.** The current network plays both sides greedily (its most likely
  move) to the end of the turn, where the critic values the position for the decider (`turn`), or
  to the end of the game (`game`). Twilight Struggle's planning is mostly inside a turn -- hands
  turn over -- and what carries across turns is the board, which is the critic's job.
* **The loss.** The all-actions policy gradient over the candidates,
  -sum_c pi(c|s) * (Q(c) - sum_c' pi~(c') Q(c')), pi~ the policy renormalised over the candidates.
  Summed over actions it needs no importance ratio, so a buffer of labelled positions can be reused
  across iterations, as the card-event target's is. Only the candidates' logits move.

Everything here is batched: one VectorizedBatchRunner holds every (decision, candidate, pair) game
of a labelling round, and the network is called on all of them at once each step.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import ts_engine as ts

HORIZONS = ("turn", "game")
#: Decision kinds a run may sample, by the engine's names.
DECISION_KINDS = ("SELECT_CARD", "SELECT_PLAY_MODE", "SELECT_OP_MODE", "CHOOSE_BRANCH",
                  "CHOOSE_TIMING_BRANCH", "POINT_NODE")
#: CHOOSE_BRANCH holds the choices inside events -- among them Wargames' "end the game" at DEFCON 2,
#: which the play-mode decision before it does not settle.
DEFAULT_DECISIONS = ("SELECT_CARD", "SELECT_PLAY_MODE", "CHOOSE_BRANCH")
#: What the playouts do with the cards the decider cannot see: keep them as dealt, or redeal them
#: per pair (`ai.search.dmcts.determinize`), pair j of every candidate getting the same redeal.
HIDDEN_MODES = ("true", "redeal")


def decision_codes(names: Sequence[str]) -> Tuple[int, ...]:
    """Engine DecisionType codes for the names; raises on a name the engine does not know."""
    out = []
    for n in names:
        if n not in DECISION_KINDS:
            raise ValueError(f"unknown playout decision kind {n!r}; choose from {DECISION_KINDS}")
        out.append(int(getattr(ts.DecisionType, n)))
    return tuple(out)


def select_candidates(logits_row: np.ndarray, mask_row: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """The k likeliest legal actions (fewer if fewer are legal) and their policy probabilities."""
    legal = np.flatnonzero(mask_row)
    z = logits_row[legal].astype(np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p, kind="stable")[:k]
    return legal[order].astype(np.int64), p[order].astype(np.float32)


@dataclass
class PlayoutRecord:
    """One sampled decision, waiting for its playouts."""
    state: Any                   # ts.GameState, a copy taken at the decision
    obs: torch.Tensor            # the decider's observation, as the rollout stored it (half)
    mask: torch.Tensor           # the legal mask in the decider's view
    decider: int                 # +1 US, -1 USSR
    cands: np.ndarray            # candidate flat actions, likeliest first
    prior: np.ndarray            # the policy's probability of each candidate at recording time


@dataclass
class PlayoutLabel:
    obs: torch.Tensor
    mask: torch.Tensor
    cands: np.ndarray
    q: np.ndarray                # mean decider score per candidate, in [-1, 1]


class PlayoutLabeller:
    """Plays every candidate of every record from paired copies of its state, batched."""

    def __init__(self, pairs: int, horizon: str, max_steps: int, merged: bool, obs_features: int,
                 chunk: int = 4096, hidden: str = "true") -> None:
        if hidden not in HIDDEN_MODES:
            raise ValueError(f"playout hidden must be one of {HIDDEN_MODES}, got {hidden!r}")
        self.hidden = hidden
        if horizon not in HORIZONS:
            raise ValueError(f"playout horizon must be one of {HORIZONS}, got {horizon!r}")
        if pairs < 1:
            raise ValueError("playout pairs must be at least 1")
        self.pairs = int(pairs)
        self.horizon = horizon
        self.max_steps = int(max_steps)
        self.merged = bool(merged)
        self.obs_features = int(obs_features)
        self.chunk = int(chunk)

    def _forward(self, net: Any, obs: np.ndarray, masks: Optional[np.ndarray], device: Any
                 ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Logits (None when no masks are given) and v_win, over `obs` in chunks."""
        logits, values = [], []
        for s in range(0, obs.shape[0], self.chunk):
            o = torch.from_numpy(obs[s:s + self.chunk]).to(device, torch.float32)
            m = None if masks is None else torch.from_numpy(masks[s:s + self.chunk]).to(device)
            lg, v, _vp = net(o, m)
            if masks is not None:
                logits.append(lg.float())
            values.append(v.float().reshape(-1))
        return (torch.cat(logits) if logits else None), torch.cat(values)

    def label(self, net: Any, records: Sequence[PlayoutRecord], device: Any, seed: int
              ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """Per record, the (pairs, n_candidates) matrix of decider scores; and run statistics."""
        if not records:
            return [], {}
        t0 = time.perf_counter()
        P = self.pairs
        layout: List[Tuple[int, int]] = []          # env -> (record, candidate)
        for r, rec in enumerate(records):
            for c in range(len(rec.cands)):
                layout.extend([(r, c)] * P)
        n = len(layout)
        runner = ts.VectorizedBatchRunner(n, int(seed) & 0x7FFFFFFFFFFFFFFF)
        runner.set_merged_influence([self.merged] * n, [self.merged] * n)
        runner.set_obs_features([self.obs_features] * n, [self.obs_features] * n)
        rng = np.random.default_rng(int(seed))
        dice = rng.integers(1, 1 << 62, size=(len(records), P), dtype=np.int64)
        first = np.zeros(n, dtype=np.uint16)
        decider = np.zeros(n, dtype=np.int8)
        start_turn = np.zeros(n, dtype=np.int16)
        worlds: List[List[Any]] = [[rec.state] * P for rec in records]
        if self.hidden == "redeal":
            import random

            from ai.search.dmcts import determinize
            shuffle = random.Random(int(seed))
            worlds = [[determinize(rec.state, ts.Player(rec.decider), shuffle) for _ in range(P)]
                      for rec in records]
        e = 0
        for r, rec in enumerate(records):
            for c, a in enumerate(rec.cands):
                for j in range(P):
                    runner.set_state(e, worlds[r][j])
                    runner.get_state(e).rng_state = int(dice[r, j])
                    first[e] = int(a)
                    decider[e] = rec.decider
                    start_turn[e] = int(rec.state.turn)
                    e += 1
        runner.refresh_all()
        ok = np.asarray(runner.step_flat_all(first.tolist(), True))
        if (ok == 0).any():
            bad = int(np.flatnonzero(ok == 0)[0])
            raise RuntimeError(f"playout candidate {int(first[bad])} was refused by the engine "
                               f"(record {layout[bad][0]}): the recorded mask and state disagree")

        score = np.full(n, np.nan, dtype=np.float64)
        steps = 0
        capped = 0

        def finish(idx: np.ndarray) -> None:
            """Score envs that just reached the horizon: the result if over, else the critic."""
            if idx.size == 0:
                return
            term = np.asarray(runner.get_terminals(), dtype=bool)[idx]
            util = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)[idx]
            score[idx[term]] = util[term] * decider[idx[term]]
            live = idx[~term]
            if live.size:
                obs = np.stack([np.asarray(ts.extract_observation_features(
                    runner.get_state(int(i)), ts.Player(int(decider[i])), self.obs_features),
                    dtype=np.float32) for i in live])
                with torch.no_grad():
                    _l, v = self._forward(net, obs, None, device)
                score[live] = v.cpu().numpy().astype(np.float64)

        open_ = np.ones(n, dtype=bool)
        while True:
            term = np.asarray(runner.get_terminals(), dtype=bool)
            done = term.copy()
            if self.horizon == "turn":
                done |= np.asarray(runner.get_turns(), dtype=np.int16) != start_turn
            newly = np.flatnonzero(open_ & done)
            finish(newly)
            open_ &= ~done
            act = np.flatnonzero(open_)
            if act.size == 0:
                break
            if steps >= self.max_steps:
                capped = int(act.size)
                finish(act)
                break
            masks = np.asarray(runner.get_action_masks())
            actions = masks.argmax(axis=1).astype(np.uint16)   # a legal move for finished envs
            obs = np.asarray(runner.get_observations())[act]
            with torch.no_grad():
                logits, _v = self._forward(net, obs, masks[act], device)
            assert logits is not None
            actions[act] = logits.argmax(dim=1).cpu().numpy().astype(np.uint16)
            runner.step_flat_all(actions.tolist(), True)
            steps += 1

        out: List[np.ndarray] = []
        e = 0
        for rec in records:
            k = len(rec.cands)
            out.append(score[e:e + k * P].reshape(k, P).T.copy())
            e += k * P
        return out, {"playout_games": float(n), "playout_steps": float(steps),
                     "playout_capped": float(capped), "playout_label_s": time.perf_counter() - t0}


def split_half_reliability(mats: Sequence[np.ndarray]) -> Optional[float]:
    """Correlation between the candidate advantages measured on the even and on the odd pairs.

    Advantages are each candidate's mean score minus the record's mean over its candidates; 1 is
    a perfectly repeatable measurement, 0 pure noise. None with too few pairs or candidates."""
    a, b = [], []
    for m in mats:
        if m.shape[0] < 2 or m.shape[1] < 2:
            continue
        ev, od = m[0::2].mean(axis=0), m[1::2].mean(axis=0)
        a.append(ev - ev.mean())
        b.append(od - od.mean())
    if not a:
        return None
    x, y = np.concatenate(a), np.concatenate(b)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def playout_pg_loss(logits: torch.Tensor, cands: torch.Tensor, valid: torch.Tensor, q: torch.Tensor,
                    min_scale: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The all-actions policy gradient over each row's candidates.

    `cands` (b, K) flat actions, `valid` (b, K) which entries are real (rows have up to K), `q`
    (b, K) the candidates' playout values. The advantage is q minus the policy-weighted mean over
    the row's candidates, divided by the batch's advantage standard deviation (at least
    `min_scale`), so the coefficient means the same thing as the playout noise changes."""
    logp = F.log_softmax(logits.float(), dim=-1).gather(1, cands)
    p = logp.exp() * valid
    with torch.no_grad():
        w = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        adv = (q - (w * q).sum(dim=1, keepdim=True)) * valid
        scale = adv[valid].std().clamp_min(min_scale) if bool(valid.sum() > 1) else torch.tensor(min_scale)
        best = torch.where(valid, q, torch.full_like(q, -9.0)).argmax(dim=1)
        top = torch.where(valid, p, torch.zeros_like(p)).argmax(dim=1)
        p_best = (p.gather(1, best.unsqueeze(1)).squeeze(1) / p.sum(dim=1).clamp_min(1e-12)).mean()
    loss = -(p * adv).sum(dim=1).mean() / scale
    return loss, {"playout_adv_scale": float(scale), "playout_agree": float((best == top).float().mean()),
                  "playout_p_best": float(p_best)}
