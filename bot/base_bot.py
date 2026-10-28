"""Abstract BaseBot Interface for Twilight Struggle AI Agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import sys

try:
    import ts_engine as ts
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _build = os.path.join(_root, "build")
    if os.path.exists(_build) and _build not in sys.path:
        sys.path.insert(0, _build)
    import ts_engine as ts


#: The decision types the engine hands a player: SELECT_CARD, SELECT_PLAY_MODE, SELECT_OP_MODE,
#: POINT_NODE and CHOOSE_BRANCH. CHOOSE_TIMING_BRANCH was retired by P17 (the resolution node
#: carries the timing) and a ROLL_DIE belongs to nobody, so neither ever reaches a bot.
PLAYER_DECISION_TYPES = frozenset({1, 2, 4, 5, 6})

#: The action that closes a node: the engine guarantees it is legal whenever nothing else is (its
#: own anti-deadlock rule), whatever `allow_early_stop` says.
CONFIRM_DONE_FLAG = 0x80


class BaseBot(ABC):
    """Abstract base class for all Twilight Struggle bot clients and agents."""

    def __init__(self, role: str, name: Optional[str] = None):
        self.role: str = role.upper()  # "US" or "USSR"
        self.name: str = name or self.__class__.__name__

    @abstractmethod
    def select_action(self, state: Dict[str, Any], legal_actions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Selects an action dictionary according to the WebSocket / REST protocol.

        Args:
            state: Serialized GameStateDict representation of the game.
            legal_actions: Dict containing decision_type, valid_ids, allow_early_stop.

        Returns:
            Dict representing micro-action or None if no valid action is possible.
        """
        ...

    def require_player_decision(self, d_type: int) -> None:
        """Raise on a decision type no player is ever asked. A bot that answered one anyway --
        with a random or first id -- would hide an engine or protocol defect behind a move."""
        if int(d_type) not in PLAYER_DECISION_TYPES:
            raise ValueError(f"{self.name}: decision_type {d_type} is not a player decision "
                             f"(expected one of {sorted(PLAYER_DECISION_TYPES)})")

    @staticmethod
    def confirm_done(d_type: int) -> Dict[str, Any]:
        """The CONFIRM_DONE action for a node with no other legal action."""
        return {"decision_type": d_type, "primary_id": 0, "secondary_id": 0, "flags": CONFIRM_DONE_FLAG}

    def select_flat_action(self, state: ts.GameState, player: ts.Player) -> int:
        """Selects a 212-dim flat action index directly against the C++ engine.

        Subclasses may override this for vectorized or high-speed flat inference.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement select_flat_action directly.")

    def reset(self) -> None:
        """Resets any internal bot state (memory, search tree, rollout cache)."""
        pass
