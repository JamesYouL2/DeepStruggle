"""A Gumbel root for play: choose the move by sequential halving over sampled candidates.

Danihelka et al., "Policy improvement by planning with Gumbel" (ICLR 2022), the root half. PUCT
first visits a move of prior p after about 1/p simulations, so at 16-32 simulations it rarely tries
anything the network is unsure of. Here, at a searched decision (`BatchedMCTSConfig.gumbel_k = k`):

1. **Candidates.** k moves by Gumbel-top-k over the network's logits: g(a) + logit(a), g ~
   Gumbel(0, gumbel_scale). gumbel_scale 0 takes the k most probable moves, deterministically.
2. **Sequential halving.** The `simulations` budget is split over ceil(log2 k) phases; in each, every
   surviving candidate's position is searched by the ordinary batched search with an equal share,
   and the better half by g + logit + sigma(completed Q) survives. sigma and the completed Q follow
   mctx (c_visit 50, c_scale 0.1, Q rescaled to [0, 1] over the legal moves), as the paper does for
   the root's choice. The last survivor is played.

Honest search samples one world from the decider's side for each phase and searches the candidates'
positions in it undeterminized -- after the move the opponent usually decides, and resampling from
there would show the decider the opponent's real hand (as ai/search/placement_search.py).

Unlike Gumbel MuZero, each phase searches a candidate's position afresh rather than growing one
tree (visits and values pooled across phases), so the batched C++ search does the work unchanged.
The target half of the method is not used: as a training target it measured worse than the prior
(research/log/E7_search_target_forms.md); this is only how a move is chosen in play.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Sequence

import numpy as np
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, settle
from ai.search.dmcts import determinize
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder

_UINT64 = 1 << 64
_US = int(ts.Player.US)
C_VISIT, C_SCALE = 50.0, 0.1


def sigma_completed(logits: Dict[int, float], value_mover: float, n: Dict[int, float],
                    q: Dict[int, float]) -> Dict[int, float]:
    """sigma(completed Q) per legal move: unvisited moves take the mixed value, Q is rescaled to
    [0, 1] over the legal moves, then scaled by (c_visit + max visits) * c_scale (mctx)."""
    acts = list(logits)
    z = {a: math.exp(logits[a] - max(logits.values())) for a in acts}
    zs = sum(z.values())
    prior = {a: z[a] / zs for a in acts}
    total = sum(n.get(a, 0.0) for a in acts)
    visited = [a for a in acts if n.get(a, 0.0) > 0]
    if visited:
        pz = sum(prior[a] for a in visited)
        mean_q = sum(prior[a] * q[a] for a in visited) / pz if pz > 0 else \
            sum(q[a] for a in visited) / len(visited)
        v_mix = (value_mover + total * mean_q) / (1.0 + total)
    else:
        v_mix = value_mover
    cq = {a: (q[a] if n.get(a, 0.0) > 0 else v_mix) for a in acts}
    lo, hi = min(cq.values()), max(cq.values())
    scale = (hi - lo) if hi - lo > 1e-8 else 1.0
    max_n = max((n.get(a, 0.0) for a in acts), default=0.0)
    return {a: (C_VISIT + max_n) * C_SCALE * (cq[a] - lo) / scale for a in acts}


class GumbelRoot:
    """`BatchedMCTS.best_actions` when `gumbel_k > 0`."""

    def __init__(self, mcts: BatchedMCTS) -> None:
        self.mcts = mcts
        self._subs: Dict[int, BatchedMCTS] = {}

    def _sub(self, sims: int) -> BatchedMCTS:
        sub = self._subs.get(sims)
        if sub is None:
            cfg = replace(self.mcts.cfg, simulations=sims, determinize=False, node_filter="all",
                          subsample=1.0, gumbel_k=0, placement_k=0, reuse_subtree=False,
                          dirichlet_frac=0.0, root_prior_temp=1.0)
            sub = BatchedMCTS(self.mcts.model, device=self.mcts.device, config=cfg,
                              featurise_capacity=self.mcts._featurise_capacity)
            self._subs[sims] = sub
        sub._rng.seed(self.mcts._rng.getrandbits(64))
        return sub

    def _network(self, states: Sequence[ts.GameState]):
        obs = np.stack([np.asarray(ts.extract_observation(s, acting_player(s)), dtype=np.float32)
                        for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        dev = self.mcts.device
        with torch.no_grad():
            lg, v, _ = self.mcts.model(torch.from_numpy(obs).to(dev), torch.from_numpy(masks).to(dev))
        return lg.float().cpu().numpy() / self.mcts.cfg.prior_temp, masks, v.float().reshape(-1).cpu().numpy()

    def choose(self, states: Sequence[ts.GameState]) -> List[int]:
        mcts, cfg = self.mcts, self.mcts.cfg
        states = list(states)
        lg, masks, v = self._network(states)
        movers = [int(acting_player(s)) for s in states]
        logits: List[Dict[int, float]] = []
        g: List[Dict[int, float]] = []
        alive: List[List[int]] = []
        for i in range(len(states)):
            legal = [int(a) for a in np.flatnonzero(masks[i])]
            logits.append({a: float(lg[i, a]) for a in legal})
            noise = mcts._np_rng.gumbel(size=len(legal)) * cfg.gumbel_scale
            g.append({a: float(x) for a, x in zip(legal, noise)})
            alive.append(sorted(legal, key=lambda a: -(g[i][a] + logits[i][a]))[:max(1, cfg.gumbel_k)])
        n = [{a: 0.0 for a in lo} for lo in logits]
        w = [{a: 0.0 for a in lo} for lo in logits]
        phases = [max(1, math.ceil(math.log2(len(c)))) if len(c) > 1 else 0 for c in alive]
        for ph in range(max(phases, default=0)):
            jobs = []
            for i, st in enumerate(states):
                if ph >= phases[i] or len(alive[i]) < 2:
                    continue
                world = st.clone()
                if cfg.determinize and not ts.Engine.is_terminal(world):
                    world = determinize(world, ts.Player(movers[i]), mcts._rng)
                world.rng_state = mcts._rng.getrandbits(64) % _UINT64
                per = max(1, cfg.simulations // (phases[i] * len(alive[i])))
                for a in alive[i]:
                    s = world.clone()
                    if not np.asarray(ActionEncoder.get_legal_mask(s))[a]:
                        continue                      # illegal in this world (Cambridge Five)
                    ts.Engine.step_flat(s, a)
                    settle(s, cfg.auto_advance)
                    jobs.append((i, a, s, per))
            for per in sorted({j[3] for j in jobs}):
                group = [j for j in jobs if j[3] == per]
                roots = self._sub(per)._search([j[2] for j in group])
                for (i, a, _s, k), r in zip(group, roots):
                    if r is None:
                        continue
                    if r.terminal or not r.actions:
                        v_us = float(r.value_us)
                    else:
                        v_us = (float(r.value_us) + float(sum(r.w))) / (1.0 + float(sum(r.n)))
                    n[i][a] += k
                    w[i][a] += (v_us if movers[i] == _US else -v_us) * k
            for i in range(len(states)):
                if ph >= phases[i] or len(alive[i]) < 2:
                    continue
                q = {a: w[i][a] / n[i][a] for a in n[i] if n[i][a] > 0}
                sig = sigma_completed(logits[i], float(v[i]), n[i], q)
                ranked = sorted(alive[i], key=lambda a: -(g[i][a] + logits[i][a] + sig[a]))
                alive[i] = ranked[:1] if ph == phases[i] - 1 else ranked[:max(1, (len(ranked) + 1) // 2)]
        return [int(c[0]) if c else 0 for c in alive]
