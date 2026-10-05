"""Which search target carries information the network does not already have?

Search-driven training (C5, P15-X4b) trains the policy toward a target built by search at a sampled
fraction of decisions. The 64-simulation visit-count target was measured to be almost the prior
(top move the same 98.8% of the time, never different above a prior of 0.8), and a fine-tune on it
lost to its control (`research/log/E7_search_finetune.md`). This probe compares target forms
without training anything:

* for each position, every form's target distribution over the legal moves;
* where a form's top move differs from the prior's (a **departure**), the two moves are played out
  from paired copies -- same dice, same redeal of the cards the decider cannot see -- to the end of
  the game by the network on both sides, and the paired difference is the departure's
  **advantage** from the decider's side.

A useful target departs often and is right when it does: **gain per position** = departure rate x
mean advantage is the expected improvement of the decision if the policy adopted the target's top
move, which is what training toward it can at best buy.

Forms (honest search throughout -- one world sampled from the decider's side; the training
searcher's configuration otherwise):

* `prior` -- the network's policy, the reference every departure is measured from;
* `visits@N` -- root visit counts of PUCT search with N simulations (what P15-X4b trains on);
* `visits@64,pt1.5` / `visits@64,rpt2` -- the same with the priors tempered at every node / the root
  (`BatchedMCTSConfig.prior_temp` / `root_prior_temp`);
* `cq@64` -- Gumbel MuZero's improved policy computed from the same 64-simulation PUCT statistics:
  softmax(logits + sigma(completed Q)), so the target form alone changes, not the search;
  `cq@64,raw` -- the same without mctx's min-max rescaling of Q (which stretches any spread, however
  small, to [0, 1]): Q's own [-1, 1] range is mapped to [0, 1];
* `gchoice@N[,kK][,fpuF][,ptT]` -- the move a noise-free Gumbel root plays (ai/search/gumbel_root.py:
  the K most probable moves, sequential halving), as a one-hot target: the root's decision rather
  than its improved policy;
* `gumbel@N` -- a Gumbel root: k candidates by Gumbel-top-k over the logits, the N simulations
  split by sequential halving, each candidate's value from an ordinary batched search of the
  position it leads to, and the same improved-policy target. Unlike Gumbel MuZero, each halving
  phase searches a candidate's position afresh rather than growing one tree (the phases' visits and
  values are pooled), so the batched C++ search does the work unchanged.

Danihelka et al., "Policy improvement by planning with Gumbel", ICLR 2022; sigma and the completed
Q follow mctx (`qtransform_completed_by_mix_value`: c_visit 50, c_scale 0.1, values rescaled to
[0, 1] over the legal moves).

What it cannot say: the playouts value a move as THIS network continues from it, so a move whose
worth lies in a follow-up the network does not find reads as worse than it is; and the end of the
game is a noisy horizon, hence the pairing.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig, _BNode, decision_segment, settle
from ai.search.dmcts import determinize
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder

_UINT64 = 1 << 64
_US = int(ts.Player.US)
#: The training searcher samples 1 in 8 decisions (P15-X4b); positions are drawn the same way.
SAMPLE_PROB = 0.125
#: mctx's defaults for sigma.
C_VISIT, C_SCALE = 50.0, 0.1
DEFAULT_FORMS = ("visits@32", "visits@64", "visits@256", "visits@64,pt1.5", "visits@64,rpt2",
                 "cq@64", "cq@64,raw", "gumbel@32", "gumbel@64")


@dataclass
class Position:
    state: ts.GameState
    mover: int
    segment: str
    turn: int
    legal: List[int]
    prior: Dict[int, float]
    logits: Dict[int, float]
    value_mover: float                       # the network's value, from the mover's side
    targets: Dict[str, Dict[int, float]] = field(default_factory=dict)


# -- the network -------------------------------------------------------------------------------

def _forward(model, states: Sequence[ts.GameState], device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(logits, masks, value from the mover's side) for positions as their movers see them."""
    obs = np.stack([np.asarray(ts.extract_observation(s, acting_player(s)), dtype=np.float32)
                    for s in states])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
    with torch.no_grad():
        lg, v, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
    return lg.float().cpu().numpy(), masks, v.float().reshape(-1).cpu().numpy()


def collect_positions(model, n: int, seed: int, device="cpu", games: int = 64,
                      temperature: float = 1.0, max_steps: int = 3000) -> List[Position]:
    """`n` positions from the model's own play at the rollout temperature, each non-forced decision
    a candidate with probability SAMPLE_PROB, drawn from whole games (every phase represented)."""
    rng = np.random.default_rng(seed)
    pool: List[Position] = []
    base = seed
    # Whole games, every candidate kept, then n drawn uniformly: stopping as soon as n were found
    # would sample only the first decisions of the first games (all turn-1 headlines).
    while len(pool) < 2 * n:
        runner = ts.VectorizedBatchRunner(games, base)
        base += 1
        for _ in range(max_steps):
            terms = np.asarray(runner.get_terminals())
            if terms.all():
                break
            obs = torch.from_numpy(np.array(runner.get_observations(), copy=True)).to(device)
            masks = np.array(runner.get_action_masks(), copy=True)
            with torch.no_grad():
                lg, _v, _ = model(obs, torch.from_numpy(masks).bool().to(device))
            probs = torch.softmax(lg.float() / temperature, dim=-1).cpu().numpy().astype(np.float64)
            acts = []
            for i in range(games):
                legal = np.flatnonzero(masks[i])
                if terms[i] or len(legal) == 0:
                    acts.append(0)
                    continue
                if len(legal) > 1 and rng.random() < SAMPLE_PROB:
                    st = runner.get_state(i)
                    if not ts.Engine.is_terminal(st) and st.current_phase != ts.Phase.SETUP:
                        pool.append(_position(st))
                p = probs[i] * (masks[i] > 0)
                acts.append(int(rng.choice(len(p), p=p / p.sum())) if p.sum() > 0 else int(legal[0]))
            runner.step_flat_all(acts, True)
    out = [pool[int(i)] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    # The network's prior and value at each, in one batch.
    lg, masks, v = _forward(model, [p.state for p in out], device)
    for i, p in enumerate(out):
        legal = np.flatnonzero(masks[i])
        z = lg[i, legal] - lg[i, legal].max()
        pr = np.exp(z) / np.exp(z).sum()
        p.legal = [int(a) for a in legal]
        p.prior = {int(a): float(x) for a, x in zip(legal, pr)}
        p.logits = {int(a): float(x) for a, x in zip(legal, lg[i, legal])}
        p.value_mover = float(v[i])
    return out


def _position(st: ts.GameState) -> Position:
    return Position(state=st.clone(), mover=int(acting_player(st)), segment=decision_segment(st),
                    turn=int(st.turn), legal=[], prior={}, logits={}, value_mover=0.0)


# -- the improved policy ---------------------------------------------------------------------

def improved_policy(logits: Dict[int, float], prior: Dict[int, float], value_mover: float,
                    n: Dict[int, float], q_mover: Dict[int, float],
                    rescale: bool = True) -> Dict[int, float]:
    """softmax(logits + sigma(completed Q)), mctx's completed-by-mix-value form.

    Unvisited moves take the mixed value: the network's value blended with the prior-weighted mean
    of the visited moves' Q, weighted by the total visits. Completed Q is rescaled to [0, 1] over
    the legal moves before sigma, as mctx rescales."""
    acts = list(logits)
    total = sum(n.get(a, 0.0) for a in acts)
    visited = [a for a in acts if n.get(a, 0.0) > 0]
    if visited:
        pz = sum(prior[a] for a in visited)
        mean_q = sum(prior[a] * q_mover[a] for a in visited) / pz if pz > 0 else \
            sum(q_mover[a] for a in visited) / len(visited)
        v_mix = (value_mover + total * mean_q) / (1.0 + total)
    else:
        v_mix = value_mover
    cq = {a: (q_mover[a] if n.get(a, 0.0) > 0 else v_mix) for a in acts}
    lo, hi = min(cq.values()), max(cq.values())
    scale = (hi - lo) if hi - lo > 1e-8 else 1.0
    if not rescale:
        # Values are win-probability-like in [-1, 1]; map that range to [0, 1] instead of stretching
        # the observed spread, so a 0.001 difference stays a 0.001 difference.
        lo, scale = -1.0, 2.0
    max_n = max((n.get(a, 0.0) for a in acts), default=0.0)
    sig = {a: (C_VISIT + max_n) * C_SCALE * (cq[a] - lo) / scale for a in acts}
    z = {a: logits[a] + sig[a] for a in acts}
    m = max(z.values())
    e = {a: math.exp(z[a] - m) for a in acts}
    s = sum(e.values())
    return {a: e[a] / s for a in acts}


def _root_q(nd: _BNode, mover: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Visits and mean value (mover's side) per root action of a searched root."""
    n: Dict[int, float] = {}
    q: Dict[int, float] = {}
    for a, ni, wi in zip(nd.actions, nd.n, nd.w):
        n[int(a)] = float(ni)
        if ni > 0:
            v = wi / ni
            q[int(a)] = v if mover == _US else -v
    return n, q


# -- the forms ---------------------------------------------------------------------------------

class TargetBuilder:
    """Builds every requested form for a batch of positions."""

    def __init__(self, model, device="cpu", seed: int = 0, featurise_capacity: int = 4096,
                 gumbel_k: int = 8) -> None:
        self.model, self.device = model, device
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cap = featurise_capacity
        self.gumbel_k = gumbel_k
        self._subs: Dict[Tuple[int, bool], BatchedMCTS] = {}

    def _searcher(self, sims: int, determinize_: bool, prior_temp: float = 1.0,
                  root_prior_temp: float = 1.0) -> BatchedMCTS:
        cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                                advance_root=False, determinize=determinize_, node_filter="all",
                                subsample=1.0, seed=self.rng.getrandbits(31),
                                prior_temp=prior_temp, root_prior_temp=root_prior_temp)
        return BatchedMCTS(self.model, device=self.device, config=cfg, featurise_capacity=self.cap)

    def build(self, positions: Sequence[Position], forms: Sequence[str]) -> None:
        for form in forms:
            kind, _, rest = form.partition("@")
            if kind == "visits" or kind == "cq":
                sims_s, *opts = rest.split(",")
                pt, rpt = 1.0, 1.0
                rescale = True
                for o in opts:
                    if o == "raw":
                        rescale = False
                    elif o.startswith("rpt"):
                        rpt = float(o[3:])
                    elif o.startswith("pt"):
                        pt = float(o[2:])
                    else:
                        raise ValueError(f"unknown option {o!r} in {form!r}")
                roots = self._searcher(int(sims_s), True, pt, rpt)._search([p.state for p in positions])
                for p, r in zip(positions, roots):
                    p.targets[form] = self._from_root(p, r, kind, rescale)
            elif kind == "gumbel":
                self._gumbel(positions, int(rest), form)
            elif kind == "gchoice":
                self._gchoice(positions, rest, form)
            else:
                raise ValueError(f"unknown form {form!r}")

    def _from_root(self, p: Position, r: Optional[_BNode], kind: str,
                   rescale: bool = True) -> Dict[int, float]:
        if r is None or r.terminal or not r.actions:
            return dict(p.prior)
        n, q = _root_q(r, p.mover)
        n = {a: n.get(a, 0.0) for a in p.legal}       # visits on moves the real state forbids drop
        if kind == "visits":
            tot = sum(n.values())
            return {a: n[a] / tot for a in p.legal} if tot > 0 else dict(p.prior)
        return improved_policy(p.logits, p.prior, p.value_mover, n,
                               {a: v for a, v in q.items() if a in n}, rescale=rescale)

    def _gchoice(self, positions: Sequence[Position], rest: str, form: str) -> None:
        """The move a noise-free Gumbel root plays (ai/search/gumbel_root.py), as a one-hot target:
        `gchoice@N[,kK][,fpuF][,ptT]` -- N simulations, K candidates (default 4), FPU, prior
        temperature. Unlike `gumbel@N`, which trains toward the improved policy, this is the
        root's decision, the thing that measured strongest at play time."""
        sims_s, *opts = rest.split(",")
        k, fpu, pt = 4, 0.0, 1.0
        for o in opts:
            if o.startswith("fpu"):
                fpu = float(o[3:])
            elif o.startswith("pt"):
                pt = float(o[2:])
            elif o.startswith("k"):
                k = int(o[1:])
            else:
                raise ValueError(f"unknown option {o!r} in {form!r}")
        cfg = BatchedMCTSConfig(simulations=int(sims_s), temperature=0.0, auto_advance=True,
                                advance_root=False, determinize=True, node_filter="all",
                                subsample=1.0, seed=self.rng.getrandbits(31), prior_temp=pt,
                                fpu_reduction=fpu, gumbel_k=k, gumbel_scale=0.0)
        mcts = BatchedMCTS(self.model, device=self.device, config=cfg, featurise_capacity=self.cap)
        picks = mcts.best_actions([p.state for p in positions])
        for p, a in zip(positions, picks):
            a = int(a)
            p.targets[form] = {x: (1.0 if x == a else 0.0) for x in p.legal} if a in p.prior \
                else dict(p.prior)

    def _gumbel(self, positions: Sequence[Position], sims: int, form: str) -> None:
        """Gumbel root by sequential halving over the candidates' positions (module docstring)."""
        worlds: List[ts.GameState] = []
        cands: List[List[int]] = []
        g: List[Dict[int, float]] = []
        for p in positions:
            w = determinize(p.state.clone(), ts.Player(p.mover), self.rng)
            w.rng_state = self.rng.getrandbits(64) % _UINT64
            worlds.append(w)
            gum = {a: float(self.np_rng.gumbel()) for a in p.legal}
            g.append(gum)
            k = min(self.gumbel_k, len(p.legal))
            cands.append(sorted(p.legal, key=lambda a: -(gum[a] + p.logits[a]))[:k])
        n = [{a: 0.0 for a in p.legal} for p in positions]
        wsum = [{a: 0.0 for a in p.legal} for p in positions]
        phases = [max(1, math.ceil(math.log2(len(c)))) if len(c) > 1 else 1 for c in cands]
        alive = [list(c) for c in cands]
        for ph in range(max(phases)):
            jobs: List[Tuple[int, int, ts.GameState, int]] = []
            for i, p in enumerate(positions):
                if ph >= phases[i] or not alive[i]:
                    continue
                per = max(1, sims // (phases[i] * len(alive[i])))
                for a in alive[i]:
                    s = worlds[i].clone()
                    if not _legal(s, a):
                        continue
                    ts.Engine.step_flat(s, a)
                    settle(s, True)
                    jobs.append((i, a, s, per))
            for per in sorted({j[3] for j in jobs}):
                group = [j for j in jobs if j[3] == per]
                roots = self._searcher(per, False)._search([j[2] for j in group])
                for (i, a, _s, k), r in zip(group, roots):
                    if r is None:
                        continue
                    if r.terminal or not r.actions:
                        v_us = float(r.value_us)
                    else:
                        v_us = (float(r.value_us) + float(sum(r.w))) / (1.0 + float(sum(r.n)))
                    v = v_us if positions[i].mover == _US else -v_us
                    n[i][a] += k
                    wsum[i][a] += v * k
            for i, p in enumerate(positions):
                if ph >= phases[i] or len(alive[i]) < 2:
                    continue
                q = {a: wsum[i][a] / n[i][a] for a in p.legal if n[i][a] > 0}
                ip = improved_policy(p.logits, p.prior, p.value_mover, n[i], q)
                # The halving score g + logits + sigma(q): ranking by log(ip) + g is the same order.
                ranked = sorted(alive[i], key=lambda a: -(g[i][a] + math.log(max(ip[a], 1e-300))))
                alive[i] = ranked[:max(1, (len(ranked) + 1) // 2)]
        for i, p in enumerate(positions):
            q = {a: wsum[i][a] / n[i][a] for a in p.legal if n[i][a] > 0}
            p.targets[form] = improved_policy(p.logits, p.prior, p.value_mover, n[i], q)


def _legal(state: ts.GameState, action: int) -> bool:
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    return 0 <= action < len(mask) and bool(mask[action])


# -- paired playouts ---------------------------------------------------------------------------

def paired_advantage(model, items: Sequence[Tuple[Position, int, int]], pairs: int, seed: int,
                     device="cpu", batch: int = 2048, max_steps: int = 4000
                     ) -> List[Tuple[float, float]]:
    """For each (position, alternative, baseline): mean and standard error over `pairs` of the
    alternative's result minus the baseline's, from the mover's side, both played to the end of the
    game greedily by `model`. Pair j redeals the cards the mover cannot see and fixes the dice the
    same way in both branches."""
    starts: List[Tuple[int, int, ts.GameState]] = []    # (item, branch 0/1, state)
    for it, (p, alt, base) in enumerate(items):
        for j in range(pairs):
            r = random.Random((seed * 1_000_003 + it * 7919 + j) % (2 ** 61))
            world = determinize(p.state.clone(), ts.Player(p.mover), r)
            world.rng_state = r.getrandbits(64) % _UINT64
            for b, a in enumerate((alt, base)):
                s = world.clone()
                if not _legal(s, a):
                    raise RuntimeError(f"move {a} illegal after the redeal; legality should not "
                                       f"depend on the hidden cards here")
                ts.Engine.step_flat(s, a)
                starts.append((it, b, s))
    results = np.zeros(len(starts))
    for lo in range(0, len(starts), batch):
        chunk = starts[lo:lo + batch]
        runner = ts.VectorizedBatchRunner(len(chunk), seed)
        for i, (_it, _b, s) in enumerate(chunk):
            runner.set_state(i, s)
        runner.refresh_all()
        for _ in range(max_steps):
            terms = np.asarray(runner.get_terminals())
            if terms.all():
                break
            obs = torch.from_numpy(np.array(runner.get_observations(), copy=True)).to(device)
            masks = np.array(runner.get_action_masks(), copy=True)
            with torch.no_grad():
                lg, _v, _ = model(obs, torch.from_numpy(masks).bool().to(device))
            acts = lg.argmax(dim=1).cpu().numpy().tolist()
            runner.step_flat_all([0 if terms[i] else int(a) for i, a in enumerate(acts)], True)
        for i in range(len(chunk)):
            results[lo + i] = float(ts.Engine.get_terminal_utility(runner.get_state(i)))
    out: List[Tuple[float, float]] = []
    per_item = 2 * pairs
    for it, (p, _alt, _base) in enumerate(items):
        r = results[it * per_item:(it + 1) * per_item].reshape(pairs, 2)
        sign = 1.0 if p.mover == _US else -1.0
        d = sign * (r[:, 0] - r[:, 1]) / 2.0          # utility is +-1; a win-loss swing is 1.0
        out.append((float(d.mean()), float(d.std(ddof=1) / math.sqrt(pairs)) if pairs > 1 else 0.0))
    return out


# -- rows and the report -----------------------------------------------------------------------

def rows(positions: Sequence[Position], forms: Sequence[str], verdicts: Dict[Tuple[int, int], Tuple[float, float]]
         ) -> List[Dict[str, Any]]:
    """One JSON row per position: prior top move, each form's top move, its TV distance from the
    prior and, where it departs, the paired advantage."""
    out: List[Dict[str, Any]] = []
    for idx, p in enumerate(positions):
        top0 = max(p.prior, key=lambda a: p.prior[a])
        row: Dict[str, Any] = {"segment": p.segment, "turn": p.turn, "mover": p.mover,
                               "prior_top": top0, "prior_top_p": p.prior[top0],
                               "n_legal": len(p.legal), "forms": {}}
        for f in forms:
            t = p.targets[f]
            top = max(t, key=lambda a: (t[a], p.prior[a]))
            tv = 0.5 * sum(abs(t.get(a, 0.0) - p.prior[a]) for a in p.legal)
            ent = {"top": top, "tv": tv}
            if top != top0:
                adv, se = verdicts[(idx, top)]
                ent.update(adv=adv, se=se)
            row["forms"][f] = ent
        out.append(row)
    return out


def report(all_rows: Sequence[Dict[str, Any]], forms: Sequence[str], sims_of: Dict[str, int]) -> str:
    n = len(all_rows)
    lines = [f"# Search target forms: departures from the prior, checked by paired playouts",
             "", f"{n} positions. A departure is a form whose top move differs from the prior's; its "
             "advantage is the paired playout difference (alternative minus the prior's move, from "
             "the mover's side, +1 = always turns a loss into a win). Gain per position = departure "
             "rate x mean advantage.", "",
             "| form | sims | departs | mean advantage on departures | confirmed (> 2 SE) | refuted (< -2 SE) | **gain per position** | mean TV from prior |",
             "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in forms:
        dep = [r["forms"][f] for r in all_rows if "adv" in r["forms"][f]]
        tv = np.mean([r["forms"][f]["tv"] for r in all_rows]) if all_rows else 0.0
        if dep:
            adv = np.array([d["adv"] for d in dep])
            se = math.sqrt(sum(d["se"] ** 2 for d in dep)) / len(dep)
            conf = sum(d["adv"] > 2 * d["se"] for d in dep)
            ref = sum(d["adv"] < -2 * d["se"] for d in dep)
            gain = adv.sum() / n
            lines.append(f"| {f} | {sims_of.get(f, 0)} | {len(dep)} ({100 * len(dep) / n:.1f}%) | "
                         f"{adv.mean():+.3f} ± {se:.3f} | {conf} | {ref} | **{1000 * gain:+.2f}‰** | {tv:.3f} |")
        else:
            lines.append(f"| {f} | {sims_of.get(f, 0)} | 0 | — | 0 | 0 | 0 | {tv:.3f} |")
    lines += ["", "## By segment: departures (gain per position, ‰)", "",
              "| segment | positions | " + " | ".join(forms) + " |",
              "|:---|---:|" + "---:|" * len(forms)]
    segs = sorted({r["segment"] for r in all_rows})
    for s in segs:
        rs = [r for r in all_rows if r["segment"] == s]
        cells = []
        for f in forms:
            dep = [r["forms"][f]["adv"] for r in rs if "adv" in r["forms"][f]]
            cells.append(f"{len(dep)} ({1000 * sum(dep) / len(rs):+.1f})")
        lines.append(f"| {s} | {len(rs)} | " + " | ".join(cells) + " |")
    lines += ["", "## By the prior's confidence: departure rate", "",
              "| prior top p | positions | " + " | ".join(forms) + " |",
              "|:---|---:|" + "---:|" * len(forms)]
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        rs = [r for r in all_rows if lo <= r["prior_top_p"] < hi]
        if not rs:
            continue
        cells = [f"{100 * sum('adv' in r['forms'][f] for r in rs) / len(rs):.1f}%" for f in forms]
        lines.append(f"| {lo:.2f}–{min(hi, 1.0):.2f} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def sims_of(forms: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for f in forms:
        _k, _, rest = f.partition("@")
        out[f] = int(rest.split(",")[0])
    return out


def dump(rows_: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows_:
            fh.write(json.dumps(r) + "\n")
