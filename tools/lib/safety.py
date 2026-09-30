"""A safety layer over any agent: take a certain win, refuse a certain loss.

The goal probes (research/log/E4_goal_probes.md) found the E4 networks taking a forced win only
59-70% of the time and losing to their own DEFCON choice in 1-8% of games. Both are decided by
`ai.eval.safety.classify_legal_actions` -- the same classifier those probes score against, and the
one `HeuristicV2Agent` wraps HeuristicBot in. `SafetyAgent` puts it around any agent:

* if some legal move wins with certainty, it plays one -- the base's own choice when that is one
  of them, else the winning move the base rates highest;
* if the base's move loses with certainty and some move does not, it plays the base's
  highest-rated move among those that do not.

Otherwise the base's move stands. Only certain outcomes count ("win"/"loss"); a line the opponent
would have to choose ("risky") is left to the base.

Hidden information: the classifier follows forced continuations, which can draw cards, so it runs
on a determinized copy (the opponent's unseen cards and the deck reshuffled, as Doctrine and the
searcher do) and reads nothing the player could not see.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch

import ts_engine as ts
from ai.eval.safety import classify_legal_actions
from bindings.action_encoder import ActionEncoder
from bot.doctrine.policy import determinize


def _policy_probs(agent: object, state: ts.GameState, player: ts.Player,
                  mask: np.ndarray) -> Optional[np.ndarray]:
    """The base agent's move probabilities here, when it is a network that can say."""
    from tools.lib.player_agent import EnsembleAgent, NeuralAgent, OnnxAgent

    obs = np.asarray(ts.extract_observation(state, player), dtype=np.float32)[None, :]
    m = mask.astype(np.uint8)[None, :]
    if isinstance(agent, OnnxAgent):
        z = agent.logits(obs, m)[0]
    elif isinstance(agent, EnsembleAgent):
        p = np.zeros(mask.shape, dtype=np.float64)
        for member in agent.members:
            x = member.logits(obs, m)[0]
            e = np.exp(x - x.max())
            p += e / e.sum()
        z = np.log(np.maximum(p, 1e-300))
    elif isinstance(agent, NeuralAgent):
        with torch.no_grad():
            logits, _, _ = agent.model(torch.from_numpy(obs).to(agent.device),
                                       torch.from_numpy(m).to(agent.device))
        z = logits[0].detach().cpu().numpy().astype(np.float64)
    else:
        return None
    z = np.where(mask > 0, z, -np.inf)
    z = z - z[mask > 0].max()
    p = np.exp(z)
    return p / p.sum()


class SafetyAgent:
    """`base`, with certain wins taken and certain losses refused (module docstring)."""

    def __init__(self, base: object, seed: int = 0):
        self.base = base
        self.name = f"safe({getattr(base, 'name', 'agent')})"
        self.merged_influence = bool(getattr(base, "merged_influence", False))
        self.rng = random.Random(seed)
        #: How often each override fired.
        self.stats: Dict[str, int] = {"decisions": 0, "took_win": 0, "refused_loss": 0}

    def reseed(self, seed: int) -> None:
        self.rng = random.Random(seed)
        reseed = getattr(self.base, "reseed", None)
        if callable(reseed):
            reseed(seed + 1)

    def select_action(self, state: ts.GameState, player: ts.Player,
                      temperature: float = 0.1) -> int:
        choice = int(getattr(self.base, "select_action")(state, player, temperature=temperature))
        self.stats["decisions"] += 1
        mask = np.asarray(ActionEncoder.get_legal_mask(state, self.merged_influence))
        if int(mask.sum()) < 2:
            return choice
        kinds = classify_legal_actions(determinize(state, player, self.rng), player)
        wins = [a for a, k in kinds.items() if k == "win" and mask[a]]
        if wins:
            if choice in wins:
                return choice
            self.stats["took_win"] += 1
            return self._best(state, player, mask, wins)
        if kinds.get(choice) == "loss":
            alive = [a for a, k in kinds.items() if k != "loss" and mask[a]]
            if alive:
                self.stats["refused_loss"] += 1
                return self._best(state, player, mask, alive)
        return choice

    def _best(self, state: ts.GameState, player: ts.Player, mask: np.ndarray,
              among: List[int]) -> int:
        p = _policy_probs(self.base, state, player, mask)
        if p is None:
            return among[0]
        return max(among, key=lambda a: p[a])
