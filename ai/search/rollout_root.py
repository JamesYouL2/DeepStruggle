"""A rollout root: the network's top moves, each played out in many sampled worlds, then a cautious choice.

The Ataraxos recipe for Stratego (Sokota et al., 2025) at test time: sample possible worlds, run
depth-limited rollouts by the network from each candidate move, average, and move the policy only as
far as the evidence carries it. Here, at one decision:

1. **Candidates.** The network's `k` most probable legal moves (the search bank found the better
   move inside the top 4 for ~92% of the network's errors).
2. **Worlds.** `worlds` independent draws of the mover's unseen cards and of the dice. Every
   candidate is played in every world, so two candidates' values in one world share the cards and
   the dice: differences are paired.
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

Everything but the network calls is C++ shared with the browser (bindings/rollout_batch.hpp):
`ts.rollout_candidates` picks step 1 from the root's logits, `ts.RolloutBatch` is steps 2 and 3,
and `ts.rollout_decide` is step 4 with the statistics it reports. This file is the loop that feeds
the network's logits and values in -- the same loop web/ui/src/analysis/rollout.ts runs with
onnxruntime -- so the two searchers cannot drift apart in anything but the network's arithmetic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.gumbel_root import sigma_completed
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features

#: A playout that has not met its horizon in this many plies is an error, not a leaf: reading an
#: unfinished rollout where it stands would value it silently at a depth nobody asked for.
_MAX_ROLLOUT_STEPS = 400
#: Rows per network call.
_ROWS = 4096


@dataclass
class RolloutConfig:
    k: int = 4
    worlds: int = 16
    horizon: int = 4
    rule: str = "z2"
    seed: int = 0


def parse_rule(rule: str) -> Tuple[str, float]:
    """(kind, parameter) of a rule -- argmax, z<x> or kl<t>; ValueError otherwise. The grammar is
    the engine's (`ts.rollout_parse_rule`), so a spec means the same thing here and in the page."""
    return ts.rollout_parse_rule(rule)


class RolloutRoot:
    def __init__(self, model: Any, config: RolloutConfig, device: Any = "cpu", leaf_model: Any = None) -> None:
        """`leaf_model` (P33a): another network's critic reads the leaves, while `model` still chooses
        the candidates and plays the playouts -- the two channels a network reaches the search by,
        separated. It must read the same observation (the same feature blocks)."""
        self.model = model
        self.leaf_model = leaf_model
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.rule, self.param = parse_rule(config.rule)
        self.features = model_obs_features(model)
        if leaf_model is not None and model_obs_features(leaf_model) != self.features:
            raise ValueError(f"the leaf network reads observation features {model_obs_features(leaf_model)!r}, "
                             f"the root's network {self.features!r}: their leaves would not be the same rows")
        self._rng = random.Random(config.seed)
        #: Per position of the last `choose`: the candidates, each one's mean value and paired lead
        #: over the network's move with its standard error, and the worlds used.
        self.last_stats: List[Dict[str, Any]] = []
        #: Network rows of the last `choose`: the root plus every playout ply and leaf.
        self.rows = 0

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _net(self, obs: np.ndarray, masks: np.ndarray, model: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """Logits and values for observation rows already built, in chunks of _ROWS (`model`, else the
        root's own network)."""
        net = self.model if model is None else model
        lgs, vs = [], []
        for lo in range(0, obs.shape[0], _ROWS):
            with torch.no_grad():
                lg, v, _ = net(torch.from_numpy(obs[lo:lo + _ROWS]).to(self.device),
                               torch.from_numpy(masks[lo:lo + _ROWS]).to(self.device))
            lgs.append(lg.float().cpu().numpy())
            vs.append(v.float().reshape(-1).cpu().numpy())
        self.rows += int(obs.shape[0])
        if not lgs:
            return np.zeros((0, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.concatenate(lgs), np.concatenate(vs)

    def _forward(self, states: Sequence[ts.GameState]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Logits, masks and each position's value for its own acting player."""
        obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), self.features),
                                   dtype=np.float32) for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        lg, v = self._net(obs, masks)
        return lg, masks, v

    def _playouts(self, batch: Any) -> None:
        """Drive the batch to its leaves and read them with the critic."""
        for _ in range(_MAX_ROLLOUT_STEPS):
            if batch.all_done():
                break
            obs, masks = batch.fill_live()
            lg, _v = self._net(np.ascontiguousarray(obs), np.ascontiguousarray(masks))
            batch.step(np.ascontiguousarray(lg, dtype=np.float32))
        else:
            raise RuntimeError(f"{batch.live()} playouts still running after {_MAX_ROLLOUT_STEPS} plies")
        if batch.refused():
            raise RuntimeError(f"the engine refused a playout action in {batch.refused()} rows")
        obs, masks = batch.fill_leaves()
        if obs.shape[0]:
            _lg, v = self._net(np.ascontiguousarray(obs), np.ascontiguousarray(masks), self.leaf_model)
            batch.set_leaf_values(np.ascontiguousarray(v, dtype=np.float32))

    def choose(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        self.rows = 0
        lg, masks, _ = self._forward(states)
        lg = np.ascontiguousarray(lg, dtype=np.float32)
        cands: List[List[int]] = [ts.rollout_candidates(lg[i], masks[i], self.cfg.k) for i in range(len(states))]
        batch = ts.RolloutBatch(horizon=self.cfg.horizon, shared_dice=True, guard=True,
                                features=self.features, seed=self._rng.getrandbits(64))
        position_of: Dict[int, int] = {}
        for i, st in enumerate(states):
            if len(cands[i]) >= 2 and not ts.Engine.is_terminal(st):
                position_of[i] = batch.positions()
                batch.add(st, cands[i], self.cfg.worlds)
        if batch.rows():
            self._playouts(batch)
        picks: List[int] = []
        self.last_stats = []
        for i in range(len(states)):
            raw = cands[i][0] if cands[i] else int(np.argmax(np.where(masks[i].astype(bool), lg[i], -np.inf)))
            p = position_of.get(i)
            if p is None:
                picks.append(raw)
                self.last_stats.append({"candidates": cands[i], "worlds": 0})
                continue
            d = ts.rollout_decide(batch, p, cands[i], lg[i], masks[i], self.cfg.rule)
            if d["worlds"] == 0:
                picks.append(raw)
                self.last_stats.append({"candidates": cands[i], "worlds": 0})
                continue
            picks.append(int(d["pick"]))
            self.last_stats.append({
                "candidates": cands[i], "worlds": int(d["worlds"]),
                "q": {a: float(q) for a, q in zip(cands[i], d["q"])},
                "lead": {a: [float(m), float(se)] for a, m, se in zip(cands[i], d["lead"], d["lead_se"])},
            })
        return picks


class RolloutTeacher:
    """Training targets from a rollout root: R32's online improved-policy targets (P32 T1) with the
    rollout root in place of the Gumbel root. At a searched decision the target over every legal move
    is softmax(logit + sigma(completed Q)) -- the formula R32's Gumbel targets used
    (`ai.search.gumbel_root.sigma_completed`) -- with Q each candidate's mean rollout value, n its
    worlds, and the unsearched moves at the mixed value. The same interface as `BatchedMCTS`
    (`should_search`, `improved_policies`), so the trainer's search-CE path takes either."""

    def __init__(self, model: Any, config: RolloutConfig, device: Any = "cpu", subsample: float = 1.0,
                 node_filter: str = "all", seed: int = 86420) -> None:
        if node_filter not in ("all", "card_playmode"):
            raise ValueError(f"unknown node_filter {node_filter!r}")
        self.root = RolloutRoot(model, config, device=device)
        self.subsample = float(subsample)
        self.node_filter = node_filter
        self._rng = random.Random(seed)
        self.model = model
        self.root.reseed(seed + 1)

    def should_search(self, state: ts.GameState) -> bool:
        if self.node_filter == "card_playmode" and int(state.ctx().decision_type) not in (
                int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)):
            return False
        return self.subsample >= 1.0 or self._rng.random() < self.subsample

    def improved_policies(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        states = list(states)
        self.root.choose(states)                  # fills last_stats: candidates, worlds, Q
        lg, masks, v = self.root._forward(states)
        out: List[Tuple[List[int], np.ndarray]] = []
        for i, st in enumerate(self.root.last_stats):
            acts = [int(a) for a in np.flatnonzero(masks[i])]
            if not acts:
                out.append(([], np.zeros(0)))
                continue
            logits = {a: float(lg[i][a]) for a in acts}
            if st["worlds"] > 0:
                q = {int(a): float(x) for a, x in st["q"].items()}
                n = {a: float(st["worlds"]) for a in q}
                sig = sigma_completed(logits, float(v[i]), n, q)
            else:                                   # nothing searched: the teacher's own policy
                sig = {a: 0.0 for a in acts}
            z = np.array([logits[a] + sig[a] for a in acts], dtype=np.float64)
            z = np.exp(z - z.max())
            out.append((acts, z / z.sum()))
        return out


class RolloutRootAgent:
    """A tournament entrant (`select_actions_batch`, the state-based batch hook)."""

    def __init__(self, model: Any, name: str, config: RolloutConfig, device: Any = "cpu", leaf_model: Any = None) -> None:
        self.model = model
        self.name = name
        self.root = RolloutRoot(model, config, device=device, leaf_model=leaf_model)

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


class TeacherPolicyTargets:
    """The control for the rollout teacher: at a searched decision the target is the frozen teacher's
    own policy over the legal moves, with no search -- R39's mechanics with the rollout root taken out.
    The same interface as `RolloutTeacher` (`should_search`, `improved_policies`)."""

    def __init__(self, model: Any, device: Any = "cpu", subsample: float = 1.0, node_filter: str = "all",
                 seed: int = 97531) -> None:
        if node_filter not in ("all", "card_playmode"):
            raise ValueError(f"unknown node_filter {node_filter!r}")
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.features = model_obs_features(model)
        self.subsample = float(subsample)
        self.node_filter = node_filter
        self._rng = random.Random(seed)

    def should_search(self, state: ts.GameState) -> bool:
        if self.node_filter == "card_playmode" and int(state.ctx().decision_type) not in (
                int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE)):
            return False
        return self.subsample >= 1.0 or self._rng.random() < self.subsample

    def improved_policies(self, states: Sequence[ts.GameState]) -> List[Tuple[List[int], np.ndarray]]:
        states = list(states)
        if not states:
            return []
        obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), self.features),
                                   dtype=np.float32) for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        with torch.no_grad():
            lg, _v, _ = self.model(torch.from_numpy(obs).to(self.device), torch.from_numpy(masks).to(self.device))
        lg_np = lg.float().cpu().numpy()
        out: List[Tuple[List[int], np.ndarray]] = []
        for i in range(len(states)):
            acts = [int(a) for a in np.flatnonzero(masks[i])]
            if not acts:
                out.append(([], np.zeros(0)))
                continue
            z = np.array([lg_np[i][a] for a in acts], dtype=np.float64)
            z = np.exp(z - z.max())
            out.append((acts, z / z.sum()))
        return out


class SparseRolloutAgent:
    """A rollout root applied at a random `frac` of decisions, the network's greedy move at the rest -- to
    measure whether its corrections help one at a time (the network following up) or only in sequence (the
    search following up). `sparse:<frac>:rollout:...` in tournament specs."""

    def __init__(self, inner: RolloutRootAgent, frac: float, seed: int = 24601) -> None:
        if not 0.0 < frac <= 1.0:
            raise ValueError("sparse: the searched fraction must be in (0, 1]")
        self.inner = inner
        self.frac = float(frac)
        self.name = f"{inner.name}@{frac:g}"
        self._rng = random.Random(seed)
        self.searched = 0
        self.decisions = 0

    def reseed(self, seed: int) -> None:
        self.inner.reseed(seed)
        self._rng = random.Random(seed + 1)

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        lg, masks, _ = self.inner.root._forward(states)
        picks = [int(np.argmax(np.where(masks[i].astype(bool), lg[i], -np.inf))) for i in range(len(states))]
        chosen = [i for i in range(len(states)) if self._rng.random() < self.frac]
        if chosen:
            for i, a in zip(chosen, self.inner.select_actions_batch([states[i] for i in chosen])):
                picks[i] = a
        self.decisions += len(states)
        self.searched += len(chosen)
        return picks

    def select_action(self, state: ts.GameState, *_: Any, **__: Any) -> int:
        return self.select_actions_batch([state])[0]


def rollout_spec_leaf(spec: str) -> Optional[str]:
    """The leaf checkpoint of rollout:<checkpoint>:k:worlds:horizon:rule:<leaf checkpoint> (P33a), or
    None: the root's own network reads its leaves."""
    parts = spec.split(":")
    return parts[6] if len(parts) > 6 and parts[6] else None


def rollout_spec_config(spec: str) -> Tuple[str, RolloutConfig, str]:
    """rollout:<checkpoint>[:k[:worlds[:horizon[:rule[:leaf checkpoint]]]]] -> (checkpoint, config,
    label); an omitted field is the measured player's (4, 16, 4, z2), and the leaf is
    `rollout_spec_leaf`'s."""
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
