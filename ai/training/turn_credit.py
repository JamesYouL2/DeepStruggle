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

import io
import math
import multiprocessing as mp
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

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


#: A priced decision as it travels: observation (float16), legal mask, target (float16) over the flat
#: action space, evidence weight, gain over the policy's expectation, and its z.
Row = Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]


def make_row(c: Candidate, obs_features: int, beta: float, k_se: float, floor: float) -> "Row | None":
    v = c.values
    if v is None or v.shape[1] < 4:
        return None
    p = np.asarray(c.probs, dtype=np.float64)
    means = v.mean(axis=1)
    best = int(np.argmax(means))
    d_w = v[best] - p @ v                           # the best option over the policy's expectation, per world
    d = float(d_w.mean())
    se = float(d_w.std(ddof=1) / math.sqrt(len(d_w)))
    w = d * d / (d * d + k_se * se * se) if d > 0 else 0.0
    logit = np.log(np.maximum(p, floor)) + beta * means
    tgt = np.exp(logit - logit.max())
    tgt /= tgt.sum()
    obs = np.asarray(ts.extract_observation_features(c.state, decider(c.state), obs_features), dtype=np.float16)
    mask = np.asarray(ActionEncoder.get_legal_mask(c.state), dtype=bool)
    full = np.zeros(mask.shape[0], dtype=np.float16)
    full[c.options] = tgt
    return obs, mask, full, float(w), d, (d / se if se > 0 else 0.0)


def price_job(net: Any, device: torch.device, cands: List[Candidate], params: Dict[str, Any],
              seed: int) -> Tuple[List[Row], Dict[str, float]]:
    """Price the candidates (and up to `params['nested_budget']` nested decisions) into rows."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    nested: List[Nested] = []
    nested_p = float(params["nested_p"])
    price(net, device, cands, int(params["worlds"]), rng.getrandbits(30),
          nested=nested if nested_p > 0 else None, nested_p=nested_p)
    n_nested = min(len(nested), int(params["nested_budget"]))
    nc = [n.cand for n in rng.sample(nested, n_nested)] if n_nested > 0 else []
    if nc:
        price(net, device, nc, int(params["worlds"]), rng.getrandbits(30))
    rows = [make_row(c, int(params["obs_features"]), float(params["beta"]), float(params["k_se"]),
                     float(params["floor"])) for c in cands + nc]
    return [r for r in rows if r is not None], {
        "priced": float(len(cands)), "nested_found": float(len(nested)), "nested_priced": float(len(nc)),
        "seconds": time.perf_counter() - t0}


def _worker_main(jobs: Any, results: Any, model_bytes: bytes, device: str) -> None:
    """A pricing worker: its own copy of the network, fed (weights, decisions) and answering with rows
    one job at a time, in order."""
    torch.set_num_threads(2)
    dev = torch.device(device)
    net = torch.load(io.BytesIO(model_bytes), map_location=dev, weights_only=False).eval()
    while True:
        msg = jobs.get()
        if msg is None:
            return
        weights, items, params, seed = msg
        try:
            net.load_state_dict(torch.load(io.BytesIO(weights), map_location=dev, weights_only=True))
            cands = [Candidate(game=-1, choice=-1, state=ts.state_from_save_json(js), options=list(opts),
                               probs=list(pr), taken=int(tk)) for js, opts, pr, tk in items]
            results.put(("ok",) + price_job(net, dev, cands, params, seed))
        except Exception:                            # surfaced in the trainer, not swallowed here
            results.put(("error", traceback.format_exc(), {}))


class TurnCredit:
    """Queues sampled learner decisions during a rollout, prices them at its end, and trains on the
    improved targets."""

    def __init__(self, coef: float, budget: int = 64, worlds: int = 16, beta: float = 10.0,
                 k_se: float = 9.0, floor: float = 1e-3, nested_p: float = 0.0, nested_share: float = 0.33,
                 narrow_share: float = 0.5, cap: int = 12, buffer: int = 8192, steps: int = 4,
                 batch: int = 512, obs_features: int = 0, seed: int = 24680, workers: int = 0) -> None:
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
        #: With workers, pricing runs beside the next rollout: each flush sends this rollout's
        #: decisions and the current weights, then takes the previous flush's rows (one rollout late).
        self.workers = int(workers)
        self._procs: List[Any] = []
        self._jobs: List[Any] = []
        self._results: Optional[Any] = None
        self._outstanding = 0

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
        params = {"worlds": self.worlds, "nested_p": self.nested_p, "obs_features": self.obs_features,
                  "beta": self.beta, "k_se": self.k_se, "floor": self.floor,
                  "nested_budget": self.budget - len(cands)}
        self.stats["tc_narrow"] = float(k_narrow)
        if self.workers <= 0:
            rows, st = price_job(net, device, cands, params, self.rng.getrandbits(30))
            self._file(rows, st, device)
            self.stats["tc_seconds"] = time.perf_counter() - t0
            return
        self._start(net, device)
        buf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, buf)
        weights = buf.getvalue()
        parts: List[List[Tuple[str, List[int], List[float], int]]] = [[] for _ in range(self.workers)]
        for k, c in enumerate(cands):
            parts[k % self.workers].append((c.state.to_save_json(), c.options, c.probs, c.taken))
        p_w = dict(params, nested_budget=int(params["nested_budget"]) // self.workers)
        for w in range(self.workers):
            self._jobs[w].put((weights, parts[w], p_w, self.rng.getrandbits(30)))
        self._outstanding += 1
        t_wait = time.perf_counter()
        if self._outstanding > 1:                       # the previous flush's rows, one per worker
            assert self._results is not None
            rows_all: List[Row] = []
            st_all: Dict[str, float] = {}
            for _ in range(self.workers):
                kind, rows, st = self._get_result()
                if kind != "ok":
                    raise RuntimeError(f"turn-credit worker failed:\n{rows}")
                rows_all.extend(rows)
                for k, v in st.items():
                    st_all[k] = st_all.get(k, 0.0) + v
            self._outstanding -= 1
            st_all["seconds"] = st_all.get("seconds", 0.0) / self.workers
            self._file(rows_all, st_all, device)
        self.stats["tc_wait_seconds"] = time.perf_counter() - t_wait
        self.stats["tc_seconds"] = time.perf_counter() - t0

    def _start(self, net: Any, device: torch.device) -> None:
        if self._procs:
            return
        ctx = mp.get_context("spawn")
        buf = io.BytesIO()
        torch.save(net, buf)
        self._results = ctx.Queue()
        for _ in range(self.workers):
            q = ctx.Queue()
            p = ctx.Process(target=_worker_main, args=(q, self._results, buf.getvalue(), str(device)), daemon=True)
            p.start()
            self._jobs.append(q)
            self._procs.append(p)

    def _get_result(self) -> Tuple[str, Any, Dict[str, float]]:
        """The next worker answer; raises rather than waiting forever on a worker that died (an OOM
        in a worker's CUDA context kills it without a reply)."""
        import queue as _queue
        assert self._results is not None
        while True:
            try:
                return self._results.get(timeout=10.0)
            except _queue.Empty:
                dead = [p.pid for p in self._procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"turn-credit worker(s) {dead} died without answering")

    def close(self) -> None:
        for q in self._jobs:
            q.put(None)
        for p in self._procs:
            p.join(timeout=10)
        self._procs, self._jobs = [], []

    def _file(self, rows: List[Row], st: Dict[str, float], device: torch.device) -> None:
        for obs, mask, tgt, w, _d, _z in rows:
            self.ready.append((torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device),
                               torch.from_numpy(tgt).to(device), float(w)))
        if len(self.ready) > self.buffer:
            del self.ready[:len(self.ready) - self.buffer]
        self.stats.update({
            "tc_priced": st.get("priced", 0.0), "tc_nested_found": st.get("nested_found", 0.0),
            "tc_nested_priced": st.get("nested_priced", 0.0), "tc_pricing_seconds": st.get("seconds", 0.0),
            "tc_gain_mean": float(np.mean([r[4] for r in rows])) if rows else 0.0,
            "tc_clear_frac": float(np.mean([r[5] >= 3.0 for r in rows])) if rows else 0.0,
            "tc_weight_mean": float(np.mean([r[3] for r in rows])) if rows else 0.0,
            "tc_ready": float(len(self.ready))})

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
