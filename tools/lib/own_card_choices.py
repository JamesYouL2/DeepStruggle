"""How a card's owner uses it: every card choice while the card is in its hand, and how each holding ended.

Shared by `tools/scripts/checkpoint_report.py` (the OPEC, Alliance for Progress and KAL-007
sections) and `tools/scripts/card_event_trajectory.py` (one card across a run's checkpoints), which
imported it from the report script before it lived here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

import ts_engine as ts

from bindings.action_encoder import ActionEncoder

#: The play-mode actions: the event first, and the space race.
EVENT: int = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.EVENT)
SPACE: int = ActionEncoder.PLAY_MODE_OFFSET + int(ts.Resolution.SPACE)

_RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rules", "cards.json")


def _card_ids() -> Dict[str, int]:
    with open(_RULES, encoding="utf-8") as f:
        return {str(c["name"]): int(c["id"]) for c in json.load(f)}


#: Card name (as in rules/cards.json) -> card id.
CARD_IDS: Dict[str, int] = _card_ids()


class OwnCardChoices:
    """Every card choice of `owner` (headline or action round, not inside an event) while `card`
    is in its hand: whether the event could trigger, `feature(st)` (a number that conditions the
    event's worth), and whether the card was played and how -- "headline", or in a round "event",
    "Ops" or "space race". `holding` numbers the spells of the card in the owner's hand."""

    def __init__(self, card: int, owner: Any, feature: Callable[[ts.GameState], int]) -> None:
        self.card, self.owner, self.feature = card, owner, feature
        self.choices: List[Dict[str, Any]] = []
        self.pending: Dict[int, Dict[str, Any]] = {}
        self.held: Dict[int, bool] = {}
        self.spell: Dict[int, int] = {}

    def see(self, g: int, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        sel = self.pending.pop(g, None)
        if sel is not None:
            if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and int(ctx.pending_op_card) == self.card:
                sel["played"] = "event" if a == EVENT else "space race" if a == SPACE else "Ops"
            else:
                sel["played"] = "not a play"
        holding = ts.in_hand_of(st.get_card_location(self.card), self.owner)
        if holding and not self.held.get(g, False):
            self.spell[g] = self.spell.get(g, 0) + 1
        self.held[g] = holding
        if (not holding or int(ctx.resolving_card) != 0 or ctx.decision_player != self.owner
                or ctx.decision_type != ts.DecisionType.SELECT_CARD
                or st.current_phase not in (ts.Phase.HEADLINE, ts.Phase.ACTION_ROUND)):
            return
        headline = st.current_phase == ts.Phase.HEADLINE
        chosen = a < ActionEncoder.PLAY_MODE_OFFSET and int(ts.decode_flat_action(st, a).primary_id) == self.card
        rec: Dict[str, Any] = {"game": g, "holding": self.spell[g], "turn": int(st.turn),
                               "phase": "headline" if headline else "ar",
                               "legal": bool(ts.CardHandlers.can_trigger_event(st, self.card, self.owner)),
                               "feature": self.feature(st), "played": "headline" if chosen and headline else None}
        self.choices.append(rec)
        if chosen and not headline:
            self.pending[g] = rec

    def done(self, g: int) -> None:
        for d in (self.pending, self.held, self.spell):
            d.pop(g, None)


def holding_outcomes(choices: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Per holding (game, spell): its choices where the event could trigger, the best `feature`
    among them, and how the holding ended -- the card's play, if it was played at one of them."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for c in choices:
        h = out.setdefault((c["game"], c["holding"]), {"max": -1, "played": None, "at": None, "legal": False})
        if c["legal"]:
            h["legal"] = True
            h["max"] = max(h["max"], int(c["feature"]))
        if c["played"] in ("headline", "event", "Ops", "space race"):
            h["played"], h["at"] = c["played"], int(c["feature"])
    return out


def pct(k: int, n: int) -> str:
    """`k` of `n` as a percentage with one decimal, or a dash when there is nothing to divide."""
    return "—" if n == 0 else f"{100 * k / n:.1f}%"
