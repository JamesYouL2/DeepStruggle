"""The policy-likelihood belief over the opponent's unseen hand (P35 item 3b).

P35a (`ai.eval.belief_history_probe`) measured it: hands drawn uniformly from the decider's hidden
pool, each weighted by the probability the network gives the opponent's actual decisions this turn
with that hand put back into each decision's position, predict the hand at log loss 0.336 against the
uniform prior's 0.384 and the observation's 0.371. This module makes it a world sampler for the
rollout root: `BeliefTracker` logs every decision of every game it is shown (`observe`), and at a
root decision `worlds` deals the root's worlds by resampling the weighted hands.

The record and the hypothesis. Within a turn the opponent's unseen hand only loses cards, and every
card it loses leaves the decider's hidden pool for somewhere visible. So a hypothesis for the hand
now fixes the hand at each earlier decision of the turn: the hypothesis plus the cards that have left
the pool since, for anywhere but the decider's own hand (`left_since`). A decision whose record does
not fit (a card entered the pool since, or the hand changed some other way -- an event's draw) is left
out, not approximated. Earlier turns are not used: the deal mixes the held cards with new ones.

The weights are the network's own: exact in self-play, a model of the opponent otherwise. The hands
are importance-sampled, so their number (`particles`) sets the noise: P35a's log loss was 0.353 at
256 and 0.336 at 2,048 on the same positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from bindings.ts_env import model_obs_features

DECK = int(ts.CardLocation.DRAW_DECK)
UNKNOWN = {1: int(ts.CardLocation.HAND_US_UNKNOWN), 2: int(ts.CardLocation.HAND_USSR_UNKNOWN)}
KNOWN = {1: int(ts.CardLocation.HAND_US_KNOWN), 2: int(ts.CardLocation.HAND_USSR_KNOWN)}
#: Locations whose meaning to the decider is transient; at a position holding one there is no belief.
TRANSIENT = (int(ts.CardLocation.PEEKED_TEMP), int(ts.CardLocation.HEADLINE_COMMITTED))
#: The rows of one network call.
_CHUNK = 16384


def side(code: int) -> int:
    """The runner's decision-player code (US 1, USSR -1) as 1 (US) / 2 (USSR)."""
    return 1 if code == 1 else 2


def player(s: int) -> ts.Player:
    return ts.Player.US if s == 1 else ts.Player.USSR


def card_locations(st: ts.GameState) -> np.ndarray:
    return np.asarray([int(st.get_card_location(c)) for c in range(1, 111)], dtype=np.int64)


@dataclass
class Decision:
    """One decision of the current turn: the position before it, who decided (1 US, 2 USSR), what
    they chose, and every card's location then."""
    state: ts.GameState
    decider: int
    action: int
    locs: np.ndarray


@dataclass
class Now:
    """The decider's view of the position a belief is for."""
    me: int                  # 1 US, 2 USSR
    locs: np.ndarray         # every card's location
    pool: np.ndarray         # 110 booleans: the decider's hidden pool (the opponent's unknown hand, the deck)
    hand: int                # the opponent's unknown hand size

    @staticmethod
    def of(st: ts.GameState, me: int) -> "Now":
        locs = card_locations(st)
        opp = 3 - me
        return Now(me, locs, (locs == DECK) | (locs == UNKNOWN[opp]), int((locs == UNKNOWN[opp]).sum()))


def left_since(d: Decision, now: Now) -> Optional[np.ndarray]:
    """The cards that were in the decider's hidden pool at `d` and have left it since, for anywhere but
    the decider's own hand: with the opponent's unseen hand now, its unseen hand at `d`. None when `d`
    is not on this hand's record (a card entered the pool since, or the hand changed by other means)."""
    me, opp = now.me, 3 - now.me
    pool_tau = (d.locs == DECK) | (d.locs == UNKNOWN[opp])
    if not pool_tau[now.pool].all():
        return None
    in_my_hand = (now.locs == UNKNOWN[me]) | (now.locs == KNOWN[me])
    left = pool_tau & ~now.pool & ~in_my_hand
    if int((d.locs == UNKNOWN[opp]).sum()) != int(left.sum()) + now.hand:
        return None
    return left


def record(log: Sequence[Decision], now: Now, all_decisions: bool) -> List[Tuple[Decision, np.ndarray]]:
    """The opponent's decisions this turn that the belief conditions on, each with `left_since`."""
    opp = 3 - now.me
    out: List[Tuple[Decision, np.ndarray]] = []
    for d in log:
        if d.decider != opp:
            continue
        if not all_decisions and int(d.state.ctx().decision_type) not in (
                int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)):
            continue
        left = left_since(d, now)
        if left is not None:
            out.append((d, left))
    return out


def sample_hands(now: Now, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` hands of the opponent's unknown size drawn uniformly from the decider's hidden pool, as an
    n x 110 uint8 membership array."""
    pool = np.flatnonzero(now.pool)
    member = np.zeros((n, 110), dtype=np.uint8)
    if now.hand:
        idx = np.argsort(rng.random((n, len(pool))), axis=1)[:, :now.hand]
        np.put_along_axis(member, pool[idx], 1, axis=1)
    return member


def log_weights(model: Any, device: torch.device, jobs: Sequence[Tuple[Now, np.ndarray, List[Tuple[Decision, np.ndarray]]]],
                temperature: float = 1.0) -> List[np.ndarray]:
    """For each job (the decider's view, its hands, the record), every hand's log-likelihood of the
    opponent's recorded decisions under the network: one batch of rows over all jobs, the rows built in
    C++ (`ts.hypothesis_rows`)."""
    feats = model_obs_features(model)
    out = [np.zeros(len(h), dtype=np.float64) for _, h, _ in jobs]
    obs_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    owner: List[Tuple[int, int, int]] = []       # (job, first row, action) per decision block
    rows = 0
    for j, (now, hands, rec) in enumerate(jobs):
        for d, left in rec:
            held = np.ascontiguousarray(hands | left.astype(np.uint8)[None, :])
            obs, masks = ts.hypothesis_rows(d.state, player(now.me), held, feats)
            obs_parts.append(obs)
            mask_parts.append(masks)
            owner.append((j, rows, d.action))
            rows += len(hands)
    if not rows:
        return out
    obs_all = np.concatenate(obs_parts)
    mask_all = np.concatenate(mask_parts)
    lp = np.empty(rows, dtype=np.float64)
    acts = np.empty(rows, dtype=np.int64)
    for j, first, a in owner:
        acts[first:first + len(jobs[j][1])] = a
    for lo in range(0, rows, _CHUNK):
        with torch.no_grad():
            m = torch.from_numpy(mask_all[lo:lo + _CHUNK]).to(device)
            lg = model(torch.from_numpy(obs_all[lo:lo + _CHUNK]).to(device), m)[0].float()
            lg = (lg / temperature).masked_fill(m == 0, float("-inf"))
            a_t = torch.from_numpy(acts[lo:lo + _CHUNK]).to(device)
            lp[lo:lo + _CHUNK] = torch.log_softmax(lg, -1).gather(1, a_t[:, None]).squeeze(1).double().cpu().numpy()
    lp = np.where(np.isfinite(lp), lp, -1e9)
    for j, first, _ in owner:
        out[j] += lp[first:first + len(jobs[j][1])]
    return out


def posterior(hands: np.ndarray, logw: np.ndarray) -> Tuple[np.ndarray, float]:
    """The per-card posterior (110) and the effective number of hands."""
    w = np.exp(logw - logw.max())
    return (w[:, None] * hands).sum(0) / w.sum(), float(w.sum() ** 2 / (w ** 2).sum())


class BeliefTracker:
    """Every game's decisions this turn, and on request the root's worlds dealt from the belief.

    `observe` is called with every decision of every game the agent plays (both sides), before it is
    applied; `worlds` for the decisions the agent is about to search. A game's weighted hands are kept
    until its opponent decides again (or the turn changes), so consecutive decisions of the agent's own
    reuse them."""

    def __init__(self, model: Any, device: torch.device, particles: int = 2048, all_decisions: bool = True,
                 seed: int = 0) -> None:
        self.model = model
        self.device = device
        self.particles = int(particles)
        self.all_decisions = bool(all_decisions)
        self.rng = np.random.default_rng(seed)
        self.logs: Dict[int, List[Decision]] = {}
        self.turns: Dict[int, int] = {}
        self._cache: Dict[int, Tuple[int, bytes, np.ndarray, np.ndarray]] = {}
        #: Per `worlds` call: positions with a belief, mean effective hands, decisions conditioned on.
        self.stats: Dict[str, float] = {"positions": 0, "ess_sum": 0.0, "decisions_sum": 0}

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def new_games(self) -> None:
        self.logs.clear()
        self.turns.clear()
        self._cache.clear()

    def observe(self, games: Sequence[int], states: Sequence[ts.GameState], actions: Sequence[int],
                players: Sequence[int]) -> None:
        for g, st, a, p in zip(games, states, actions, players):
            if int(p) == 0:
                continue
            turn = int(st.turn)
            if self.turns.get(g) != turn:
                self.logs[g], self.turns[g] = [], turn
            self.logs[g].append(Decision(st, side(int(p)), int(a), card_locations(st)))

    def worlds(self, games: Sequence[int], states: Sequence[ts.GameState], n_worlds: int) -> List[Optional[np.ndarray]]:
        """For each game's position (the agent to move): `n_worlds` opposing hands (n_worlds x 110 uint8)
        resampled from the belief, or None where there is nothing to believe (no hidden hand, a
        transient location) -- the root then redeals at random as before."""
        out: List[Optional[np.ndarray]] = [None] * len(games)
        jobs: List[Tuple[Now, np.ndarray, List[Tuple[Decision, np.ndarray]]]] = []
        where: List[Tuple[int, int, bytes, int]] = []
        for i, (g, st) in enumerate(zip(games, states)):
            me = side(1 if st.ctx().decision_player == ts.Player.US else -1)
            now = Now.of(st, me)
            if now.hand == 0 or np.isin(now.locs, TRANSIENT).any():
                continue
            rec = record(self.logs.get(g, []), now, self.all_decisions)
            key = now.locs.tobytes()
            hit = self._cache.get(g)
            if hit is not None and hit[0] == len(rec) and hit[1] == key:
                out[i] = self._resample(hit[2], hit[3], n_worlds)
                continue
            hands = sample_hands(now, self.particles, self.rng)
            jobs.append((now, hands, rec))
            where.append((i, g, key, len(rec)))
        if jobs:
            for (i, g, key, n_rec), (now, hands, rec), logw in zip(where, jobs, log_weights(self.model, self.device, jobs)):
                self._cache[g] = (n_rec, key, hands, logw)
                out[i] = self._resample(hands, logw, n_worlds)
                self.stats["positions"] += 1
                self.stats["ess_sum"] += posterior(hands, logw)[1]
                self.stats["decisions_sum"] += len(rec)
        return out

    def _resample(self, hands: np.ndarray, logw: np.ndarray, n: int) -> np.ndarray:
        w = np.exp(logw - logw.max())
        pick = self.rng.choice(len(hands), size=n, replace=True, p=w / w.sum())
        return np.ascontiguousarray(hands[pick])
