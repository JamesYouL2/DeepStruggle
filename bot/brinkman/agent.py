"""Brinkman as a tournament agent (`PlayerAgent`) and as a match-loop bot (`BaseBot`)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import ts_engine as ts
from bot.base_bot import BaseBot
from bot.brinkman.evaluator import Weights
from bot.brinkman.policy import ROLL_DIE_INDEX, BrinkmanPolicy


class BrinkmanAgent:
    """The `tools.lib.player_agent.PlayerAgent` protocol: a flat action for the player to move.

    Never reads what its player could not see: the search runs on a determinized copy of the
    state (`policy.determinize`)."""

    def __init__(self, name: str = "Brinkman", seed: int = 0, weights: Optional[Weights] = None):
        self.name = name
        self.policy = BrinkmanPolicy(weights=weights, seed=seed)

    def select_action(self, state: ts.GameState, player: ts.Player, temperature: float = 0.1) -> int:
        ctx = state.ctx()
        if ctx.decision_type == ts.DecisionType.ROLL_DIE and ctx.decision_player == ts.Player.NONE:
            return ROLL_DIE_INDEX
        return int(self.policy.choose(state))


class BrinkmanBot(BaseBot):
    """The same player for `tools/play_match.py`, which routes engine-state bots through
    `select_from_state` (see `wants_game_state`)."""

    wants_game_state = True

    def __init__(self, role: str, name: Optional[str] = None, seed: int = 0):
        super().__init__(role, name or "Brinkman")
        self.agent = BrinkmanAgent(name=self.name, seed=seed)

    def select_flat_action(self, state: ts.GameState, player: ts.Player) -> int:
        return self.agent.select_action(state, player)

    def select_from_state(self, state: ts.GameState) -> Dict[str, Any]:
        player = state.ctx().decision_player
        if player == ts.Player.NONE:
            player = state.phasing_player
        flat = self.select_flat_action(state, player)
        if not np.asarray(ts.Engine.get_flat_action_mask(state))[flat]:
            raise RuntimeError(f"Brinkman chose flat action {flat}, illegal at decision_type="
                               f"{int(state.ctx().decision_type)}")
        ma = ts.decode_flat_action(state, flat)
        return {"decision_type": int(ma.decision_type), "primary_id": int(ma.primary_id),
                "secondary_id": int(ma.secondary_id), "flags": int(ma.flags)}

    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Brinkman searches the engine GameState; use select_flat_action.")

    def reset(self) -> None:
        pass
