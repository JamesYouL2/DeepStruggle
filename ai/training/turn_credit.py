"""P32 B4': paired counterfactual credit to the end of the turn.

During each rollout a sample of the learner's small decisions (2 to `cap` options, setup and single
Influence points excluded; `ai.search.turn_pricing.priceable`) is cloned. When the rollout ends, up
to `budget` of them are priced: every option over `worlds` worlds -- a shared redeal of the hidden
cards per world, independent dice per option -- with the current network playing both sides
greedily to the end of the turn and its critic reading the decider's value there (or the result).
Half the budget (`narrow_share`) goes to narrow spots -- the policy confident (max p > 0.9) while
some option sits below p = 0.02 -- the rest is drawn uniformly. With `nested_p` > 0 the decider's
small decisions met in the branches of options the policy does not prefer -- positions its own play
does not reach -- are priced too (up to `nested_share` of the budget), so a pair of neglected
decisions is learned from its end: the second first, then the first.

Each priced decision becomes a training row: the improved target

    pi'(a) ∝ max(pi(a), floor) * exp(beta * Q(a))

over its options, by cross-entropy. The floor keeps a saturated option reachable: without it an
option at p = 1e-13 stays at ~0 in the target whatever its value, the same dead end the logit caps
were built for (research/log/E7_saturated_policy_and_caps.md). Each row is weighted by its
evidence, w = d^2 / (d^2 + k * se^2), with d the best option's margin over the policy's own
expected value at the decision and se its paired standard error -- no hard significance gate,
which would push only toward options that got lucky. Rows wait in a FIFO buffer and are trained on
in `steps` minibatches after each PPO update; rollouts are untouched, so the state distribution is
the control's.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ts_engine as ts
from ai.eval.paired_playouts import decider
from ai.search.turn_pricing import Candidate, Nested, forward, price, priceable
from bindings.action_encoder import ActionEncoder


def turn_credit_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """mean over rows of w * CE(target, softmax(logits)); the logits come masked (-inf / -1e9 off
    the legal set) and the target is zero there."""
    log_p = F.log_softmax(logits.float(), dim=-1).clamp_min(-1e4)
    ce = -(target.float() * log_p).sum(dim=-1)
    return (weight.float() * ce).mean()


class TurnCredit:
    """Queues sampled learner decisions during a rollout, prices them at its end, and trains on the
    improved targets."""

    def __init__(self, coef: float, budget: int = 64, worlds: int = 16, beta: float = 10.0,
                 k_se: float = 9.0, floor: float = 1e-3, nested_p: float = 0.0, nested_share: float = 0.33,
                 narrow_share: float = 0.5, cap: int = 12, buffer: int = 8192, steps: int = 4,
                 batch: int = 512, obs_features: int = 0, seed: int = 24680) -> None:
        if budget < 1 or worlds < 4:
            raise ValueError("--turn-credit-budget must be >= 1 and --turn-credit-worlds >= 4")
        self.coef, self.budget, self.worlds, self.beta = float(coef), int(budget), int(worlds), float(beta)
        self.k_se, self.floor, self.nested_p = float(k_se), float(floor), float(nested_p)
        self.nested_share, self.narrow_share, self.cap = float(nested_share), float(narrow_share), int(cap)
        self.buffer, self.steps, self.batch = int(buffer), int(steps), int(batch)
        self.obs_features = int(obs_features)
        self.rng = random.Random(seed)
        self.queue: List[Any] = []
        self.seen = 0
        self.ready: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        self.stats: Dict[str, float] = {}

    # -- the rollout ---------------------------------------------------------------------------

    def record(self, runner: Any, masks_np: np.ndarray, learner_np: np.ndarray) -> None:
        """Reservoir-sample this step's learner decisions with 2..cap options into the queue (at
        most 4 x budget per rollout), cloning only the ones kept."""
        counts = masks_np.sum(axis=1)
        rows = np.flatnonzero(np.asarray(learner_np, dtype=bool) & (counts >= 2) & (counts <= self.cap))
        cap_q = 4 * self.budget
        for i in rows:
            self.seen += 1
            if len(self.queue) < cap_q:
                slot = len(self.queue)
            else:
                slot = self.rng.randrange(self.seen)
                if slot >= cap_q:
                    continue
            st = runner.get_state(int(i))
            if not priceable(st, self.cap):
                continue
            if slot == len(self.queue):
                self.queue.append(st.clone())
            else:
                self.queue[slot] = st.clone()

    def _policy(self, net: Any, device: torch.device, states: List[Any]) -> List[np.ndarray]:
        obs = np.stack([np.asarray(ts.extract_observation_features(s, decider(s), self.obs_features),
                                   dtype=np.float32) for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        lg, _ = forward(net, obs, masks, device)
        out = []
        for i in range(len(states)):
            legal = np.flatnonzero(masks[i])
            z = lg[i][legal] - lg[i][legal].max()
            out.append(np.exp(z) / np.exp(z).sum())
        return out

    def flush(self, net: Any, device: torch.device) -> None:
        """Price up to `budget` queued decisions (and nested ones) and file their targets."""
        t0 = time.perf_counter()
        queue, self.queue, self.seen = self.queue, [], 0
        self.stats = {"tc_queue": float(len(queue))}
        if not queue:
            return
        probs = self._policy(net, device, queue)
        narrow = [i for i, p in enumerate(probs) if p.max() > 0.9 and p.min() < 0.02]
        n_main = self.budget - int(self.nested_share * self.budget) if self.nested_p > 0 else self.budget
        k_narrow = min(len(narrow), int(self.narrow_share * n_main))
        pick = self.rng.sample(narrow, k_narrow)
        rest = [i for i in range(len(queue)) if i not in set(pick)]
        pick += self.rng.sample(rest, min(len(rest), n_main - len(pick)))
        cands = []
        for i in pick:
            legal = [int(a) for a in np.flatnonzero(ActionEncoder.get_legal_mask(queue[i]))]
            cands.append(Candidate(game=-1, choice=-1, state=queue[i], options=legal,
                                   probs=[float(x) for x in probs[i]], taken=legal[int(np.argmax(probs[i]))]))
        nested: List[Nested] = []
        price(net, device, cands, self.worlds, self.rng.getrandbits(30),
              nested=nested if self.nested_p > 0 else None, nested_p=self.nested_p)
        n_nested = min(len(nested), self.budget - len(cands))
        nc = [n.cand for n in self.rng.sample(nested, n_nested)] if n_nested > 0 else []
        if nc:
            price(net, device, nc, self.worlds, self.rng.getrandbits(30))
        rows = [self._row(c, device) for c in cands + nc]
        kept = [r for r in rows if r is not None]
        self.ready.extend(r[0] for r in kept)
        if len(self.ready) > self.buffer:
            del self.ready[:len(self.ready) - self.buffer]
        gains = [r[1] for r in kept]
        zs = [r[2] for r in kept]
        self.stats.update({
            "tc_priced": float(len(cands)), "tc_nested_found": float(len(nested)),
            "tc_nested_priced": float(len(nc)), "tc_narrow": float(k_narrow),
            "tc_gain_mean": float(np.mean(gains)) if gains else 0.0,
            "tc_clear_frac": float(np.mean([z >= 3.0 for z in zs])) if zs else 0.0,
            "tc_weight_mean": float(np.mean([r[0][3] for r in kept])) if kept else 0.0,
            "tc_ready": float(len(self.ready)), "tc_seconds": time.perf_counter() - t0})

    def _row(self, c: Candidate, device: torch.device
             ) -> "Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], float, float] | None":
        v = c.values
        if v is None or v.shape[1] < 4:
            return None
        p = np.asarray(c.probs, dtype=np.float64)
        means = v.mean(axis=1)
        best = int(np.argmax(means))
        d_w = v[best] - p @ v                       # the best option over the policy's expectation, per world
        d = float(d_w.mean())
        se = float(d_w.std(ddof=1) / math.sqrt(len(d_w)))
        w = d * d / (d * d + self.k_se * se * se) if d > 0 else 0.0
        logit = np.log(np.maximum(p, self.floor)) + self.beta * means
        tgt = np.exp(logit - logit.max())
        tgt /= tgt.sum()
        obs = np.asarray(ts.extract_observation_features(c.state, decider(c.state), self.obs_features),
                         dtype=np.float32)
        mask = np.asarray(ActionEncoder.get_legal_mask(c.state), dtype=bool)
        full = np.zeros(mask.shape[0], dtype=np.float32)
        full[c.options] = tgt
        row = (torch.from_numpy(obs).to(device, torch.float16), torch.from_numpy(mask).to(device),
               torch.from_numpy(full).to(device, torch.float16), float(w))
        return row, d, (d / se if se > 0 else 0.0)

    # -- the update ----------------------------------------------------------------------------

    def update(self, net: nn.Module, optimizer: torch.optim.Optimizer, max_grad_norm: float) -> Dict[str, float]:
        out = dict(self.stats)
        n = len(self.ready)
        if n == 0 or self.steps <= 0:
            return out
        was_training = net.training
        net.train()
        loss_sum = 0.0
        for _ in range(self.steps):
            pick = np.random.randint(0, n, size=min(self.batch, n))
            recs = [self.ready[int(k)] for k in pick]
            obs = torch.stack([r[0] for r in recs]).float()
            mask = torch.stack([r[1] for r in recs])
            tgt = torch.stack([r[2] for r in recs])
            w = torch.tensor([r[3] for r in recs], device=obs.device)
            logits = net(obs, mask)[0]
            loss = turn_credit_loss(logits, tgt, w)
            optimizer.zero_grad(set_to_none=True)
            (self.coef * loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            loss_sum += float(loss.detach())
        net.train(was_training)
        out["tc_loss"] = loss_sum / self.steps
        return out
