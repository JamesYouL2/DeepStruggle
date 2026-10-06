"""P31 1c: counterfactual mode credit from paired playouts.

At 1 in k of the learner's floor decisions (play mode and the non-country choices inside events, as
1a), every legal option is played out to the end of the game -- the current network on both sides,
sampling at temperature 1 -- from a clone of the decision's state. All options of one decision start
from the same state, RNG included, so their dice and redeals start in common. Each option's value is
the game result from the deciding side (+1 / 0 / -1), averaged over `playouts` games, and centred on
the option the rollout actually took. The policy loss then gets an all-options term,

    L_cf = -c * sum_a pi_theta(a | s) * (Q(a) - Q(a_taken)),

the all-actions policy gradient (Mean Actor-Critic): it moves probability toward the options whose
playouts beat the one taken, including options the policy almost never samples. Centring changes
nothing in expectation (sum_a grad pi(a) = 0); it keeps the term at zero for a decision where every
option did equally well.

Nothing is acted on: the rollout still samples from the policy, and the playouts only price options.
A playout values an option *given how this model plays afterwards*, which is why P31 applies the
term at single decisions rather than at the first link of a chain whose follow-up the model lacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import ts_engine as ts


def mode_cf_loss(logits: torch.Tensor, adv: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
    """The all-options term on the priced rows: mean over them of -sum_a pi(a) * adv(a), where
    adv(a) = Q(a) - Q(taken). Zero (and no host sync) when the minibatch holds no priced row."""
    has_f = has.to(torch.float32)
    term = -(F.softmax(logits.float(), dim=-1) * adv.to(torch.float32)).sum(dim=-1)
    return (term * has_f).sum() / has_f.sum().clamp(min=1.0)


@dataclass
class Queued:
    step: int                      # buffer step
    env: int
    state: Any                     # ts.GameState clone at the decision
    options: np.ndarray            # legal flat actions
    taken: int                     # the action the rollout took
    sign: float                    # +1 when the US decides, -1 for the USSR


class ModeCounterfactual:
    """Queues sampled decisions during a rollout and prices their options at its end."""

    def __init__(self, subsample: int, playouts: int, auto_advance: bool, obs_features: int = 0,
                 chunk: int = 2048, max_plies: int = 2000, seed: int = 13579) -> None:
        if subsample < 1:
            raise ValueError("--mode-cf-subsample must be >= 1")
        if playouts < 1:
            raise ValueError("--mode-cf-playouts must be >= 1")
        self.subsample = int(subsample)
        self.playouts = int(playouts)
        self.auto_advance = bool(auto_advance)
        self.obs_features = int(obs_features)
        self.chunk = int(chunk)
        self.max_plies = int(max_plies)
        self.rng = np.random.default_rng(seed)
        self.queue: List[Queued] = []
        self.games_played = 0
        self.plies_played = 0

    def pick(self, rows: np.ndarray) -> np.ndarray:
        """Of the candidate rows, the ones sampled this step (each with probability 1/k)."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return rows
        return rows[self.rng.random(rows.size) < 1.0 / self.subsample]

    def enqueue(self, step: int, envs: np.ndarray, runner: Any, masks_np: np.ndarray,
                taken: np.ndarray, dp: np.ndarray) -> None:
        for e in np.asarray(envs, dtype=np.int64):
            e = int(e)
            opts = np.flatnonzero(masks_np[e])
            if opts.size < 2:
                continue
            self.queue.append(Queued(step=step, env=e, state=runner.get_state(e).clone(),
                                     options=opts, taken=int(taken[e]),
                                     sign=1.0 if int(dp[e]) == int(ts.Player.US) else -1.0))

    @torch.no_grad()
    def flush(self, net: Any, device: torch.device) -> List[Tuple[Queued, np.ndarray]]:
        """Play out every queued option; return (decision, Q per option, aligned with options)."""
        jobs: List[Tuple[int, int]] = []                   # (queue index, option index), x playouts
        for qi, q in enumerate(self.queue):
            for oi in range(q.options.size):
                jobs.extend([(qi, oi)] * self.playouts)
        values = np.zeros(len(jobs), dtype=np.float64)
        for lo in range(0, len(jobs), self.chunk):
            part = jobs[lo:lo + self.chunk]
            values[lo:lo + len(part)] = self._play(net, device, part)
        out: List[Tuple[Queued, np.ndarray]] = []
        k = 0
        for q in self.queue:
            n = q.options.size * self.playouts
            q_vals = values[k:k + n].reshape(q.options.size, self.playouts).mean(axis=1)
            out.append((q, q_vals))
            k += n
        self.queue = []
        return out

    def _play(self, net: Any, device: torch.device, part: Sequence[Tuple[int, int]]) -> np.ndarray:
        n = len(part)
        runner = ts.VectorizedBatchRunner(n, int(self.rng.integers(1, 1 << 30)))
        if self.obs_features:
            runner.set_obs_features([self.obs_features] * n, [self.obs_features] * n)
        sign = np.zeros(n)
        done = np.zeros(n, dtype=bool)
        value = np.zeros(n)
        for i, (qi, oi) in enumerate(part):
            q = self.queue[qi]
            st = q.state.clone()
            act = int(q.options[oi])
            if not ts.Engine.try_step_flat(st, act, self.auto_advance):
                raise RuntimeError(f"counterfactual option {act} was legal in the rollout's mask but the "
                                   f"engine refused it on the cloned state")
            runner.set_state(i, st)
            sign[i] = q.sign
        runner.refresh_all()
        term = np.asarray(runner.get_terminals(), dtype=bool)
        util = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
        value[term] = util[term] * sign[term]
        done |= term
        for i in np.flatnonzero(term):                      # over already: park on a fresh game
            runner.reset_game(int(i), int(self.rng.integers(1, 1 << 30)))
        plies = 0
        while not done.all():
            if plies >= self.max_plies:
                raise RuntimeError(f"{int((~done).sum())} counterfactual playouts still running after "
                                   f"{self.max_plies} plies")
            obs = torch.from_numpy(np.asarray(runner.get_observations())).to(device, torch.float32)
            masks_np = np.asarray(runner.get_action_masks())
            masks = torch.from_numpy(masks_np).to(device)
            logits = net(obs, masks)[0].float()
            acts = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1).cpu().numpy()
            # A finished game still sits in the runner, parked on a fresh one; step it with its first
            # legal action so the batch stays rectangular, and ignore it.
            if done.any():
                acts[done] = masks_np[done].argmax(axis=1)
            res = np.asarray(runner.step_flat_all(acts.tolist(), self.auto_advance))
            bad = int(((res == 0) & ~done).sum())
            if bad:
                raise RuntimeError(f"the engine refused a sampled playout action in {bad} games")
            plies += 1
            term = np.asarray(runner.get_terminals(), dtype=bool)
            util = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
            new = term & ~done
            value[new] = util[new] * sign[new]
            done |= new
            for i in np.flatnonzero(new):
                runner.reset_game(int(i), int(self.rng.integers(1, 1 << 30)))
        self.games_played += n
        self.plies_played += plies * n
        return value
