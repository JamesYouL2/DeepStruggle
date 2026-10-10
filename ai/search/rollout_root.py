"""A rollout root: the network's top moves, each played out in many sampled worlds, then a cautious choice.

The Ataraxos recipe for Stratego (Sokota et al., 2025) at test time: sample possible worlds, run
depth-limited rollouts by the network from each candidate move, average, and move the policy only as
far as the evidence carries it. Here, at one decision:

1. **Candidates.** The network's `k` most probable legal moves (the search bank found the better
   move inside the top 4 for ~92% of the network's errors).
2. **Worlds.** `worlds` independent draws of the mover's unseen cards and of the dice
   (`determinize` plus a fresh random stream). Every candidate is played in every world, so two
   candidates' values in one world share the cards and the dice: differences are paired.
3. **Rollouts.** After the candidate, the network plays both sides greedily through `horizon`
   action-round boundaries (0: none -- the critic reads the position the move leads to), then the
   critic reads the leaf from the mover's side; a finished game counts its result. A rollout move
   that ends the game against its own player on the spot is replaced by that player's next-best
   move, so a candidate is not credited with an opponent's blind suicide.
4. **Choice** (`rule`): `argmax` -- the best mean value; `z<x>` -- the best mean value only if its
   paired lead over the network's own move is at least x standard errors, else the network's move;
   `kl<t>` -- argmax of log pi(a) + Q(a) / t over the candidates, one mirror-descent step from the
   network's policy (Q the mover's value in [-1, 1]).

The defaults are the measured player, `rollout:<ckpt>:4:16:4:z2`
(research/log/E7_search_depth_and_value.md section 4). The budget is network rows: k x worlds x
(1 + rollout steps) plus the root. Honest: candidates come from the real position, values only from
sampled worlds. A world where a candidate is illegal (The Cambridge Five can change the legal set)
is dropped for every candidate of that position.

The playouts are P32 B4''s `ai.search.turn_pricing.price` -- every candidate of every world on one
`VectorizedBatchRunner`, the network read in its own observation feature set, the critic read from
the mover's side at the stop -- with its `horizon`, `shared_dice` and `guard` switches; the root's
own forward is the only per-position network call outside it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.paired_playouts import paired_diff
from ai.search.pimcts import acting_player
from ai.search.turn_pricing import Candidate, price
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features

#: A rollout that has not met its horizon in this many plies is an error, not a leaf: reading an
#: unfinished rollout where it stands would value it silently at a depth nobody asked for.
_MAX_ROLLOUT_STEPS = 400
#: Rows per network call at the root.
_ROWS = 4096


@dataclass
class RolloutConfig:
    k: int = 4
    worlds: int = 16
    horizon: int = 4
    rule: str = "z2"
    seed: int = 0


def parse_rule(rule: str) -> Tuple[str, float]:
    if rule == "argmax":
        return "argmax", 0.0
    if rule.startswith("z"):
        return "z", float(rule[1:] or 2.0)
    if rule.startswith("kl"):
        return "kl", float(rule[2:] or 0.05)
    raise ValueError(f"unknown rule {rule!r}: argmax, z<x> or kl<t>")


class RolloutRoot:
    def __init__(self, model: Any, config: RolloutConfig, device: Any = "cpu") -> None:
        self.model = model
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.rule, self.param = parse_rule(config.rule)
        self.features = model_obs_features(model)
        self._rng = random.Random(config.seed)
        #: Per position of the last `choose`: the candidates, each one's mean value and paired lead
        #: over the network's move with its standard error, and the worlds used.
        self.last_stats: List[Dict[str, Any]] = []
        #: Network rows of the last `choose`: the root plus every playout step and leaf.
        self.rows = 0

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _forward(self, states: Sequence[ts.GameState]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Logits, masks and each position's value for its own acting player."""
        lgs, vs, ms = [], [], []
        for lo in range(0, len(states), _ROWS):
            chunk = states[lo:lo + _ROWS]
            obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), self.features),
                                       dtype=np.float32) for s in chunk])
            masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in chunk])
            with torch.no_grad():
                lg, v, _ = self.model(torch.from_numpy(obs).to(self.device),
                                      torch.from_numpy(masks).to(self.device))
            lgs.append(lg.float().cpu().numpy())
            vs.append(v.float().reshape(-1).cpu().numpy())
            ms.append(masks)
            self.rows += len(chunk)
        return np.concatenate(lgs), np.concatenate(ms), np.concatenate(vs)

    def choose(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        self.rows = 0
        lg, masks, _ = self._forward(states)
        cands: List[List[int]] = []
        for i in range(len(states)):
            z = np.where(masks[i].astype(bool), lg[i], -np.inf)
            cands.append([int(a) for a in np.argsort(-z)[:self.cfg.k] if np.isfinite(z[a])])
        priced: Dict[int, Candidate] = {}
        for i, st in enumerate(states):
            if len(cands[i]) >= 2 and not ts.Engine.is_terminal(st):
                priced[i] = Candidate(game=i, choice=0, state=st, options=list(cands[i]), probs=[],
                                      taken=cands[i][0])
        if priced:
            self.rows += price(self.model, self.device, list(priced.values()), self.cfg.worlds,
                               self._rng.getrandbits(30), max_plies=_MAX_ROLLOUT_STEPS,
                               horizon=self.cfg.horizon, shared_dice=True, guard=True)
        picks: List[int] = []
        self.last_stats = []
        for i in range(len(states)):
            raw = cands[i][0] if cands[i] else int(np.argmax(np.where(masks[i].astype(bool), lg[i], -np.inf)))
            c = priced.get(i)
            values = c.values if c is not None else None
            if values is None or values.shape[1] == 0:
                picks.append(raw)
                self.last_stats.append({"candidates": cands[i], "worlds": 0})
                continue
            q = {a: float(values[j].mean()) for j, a in enumerate(cands[i])}
            lead: Dict[int, Tuple[float, float]] = {}
            for j, a in enumerate(cands[i]):
                m, se = paired_diff(values[j], values[0])
                lead[a] = (m, 0.0 if math.isnan(se) else se)
            if self.rule == "argmax":
                pick = max(cands[i], key=lambda a: q[a])
            elif self.rule == "z":
                best = max(cands[i], key=lambda a: q[a])
                m, se = lead[best]
                pick = best if best != raw and m > 0 and (se == 0.0 or m / se >= self.param) else raw
            else:
                z = lg[i] - np.max(lg[i][masks[i].astype(bool)])
                logp = {a: float(z[a] - math.log(np.exp(z[masks[i].astype(bool)]).sum())) for a in cands[i]}
                pick = max(cands[i], key=lambda a: logp[a] + q[a] / self.param)
            picks.append(int(pick))
            self.last_stats.append({"candidates": cands[i], "worlds": int(values.shape[1]), "q": q,
                                    "lead": {a: list(v) for a, v in lead.items()}})
        return picks


class RolloutRootAgent:
    """A tournament entrant (`select_actions_batch`, the state-based batch hook)."""

    def __init__(self, model: Any, name: str, config: RolloutConfig, device: Any = "cpu") -> None:
        self.model = model
        self.name = name
        self.root = RolloutRoot(model, config, device=device)

    def reseed(self, seed: int) -> None:
        self.root.reseed(seed)

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        picks = self.root.choose(states)
        for i, (st, a) in enumerate(zip(states, picks)):
            if not np.asarray(ActionEncoder.get_legal_mask(st))[a]:
                raise RuntimeError(f"rollout root returned an illegal action {a}")
        return picks

    def select_action(self, state: ts.GameState, *_: Any, **__: Any) -> int:
        return self.select_actions_batch([state])[0]


def rollout_spec_config(spec: str) -> Tuple[str, RolloutConfig, str]:
    """rollout:<checkpoint>[:k[:worlds[:horizon[:rule]]]] -> (checkpoint, config, label); an
    omitted field is the measured player's (4, 16, 4, z2)."""
    parts = spec.split(":")
    if parts[0].lower() != "rollout" or len(parts) < 2:
        raise ValueError(f"not a rollout spec: {spec!r}")

    def field(i: int) -> Optional[str]:
        return parts[i] if len(parts) > i and parts[i] else None

    d = RolloutConfig()
    cfg = RolloutConfig(k=int(field(2) or d.k), worlds=int(field(3) or d.worlds),
                        horizon=int(field(4) or d.horizon), rule=field(5) or d.rule)
    parse_rule(cfg.rule)
    return parts[1], cfg, f"rollout-k{cfg.k}-w{cfg.worlds}-h{cfg.horizon}-{cfg.rule}"
