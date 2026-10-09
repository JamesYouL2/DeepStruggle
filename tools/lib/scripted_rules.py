"""Scripted rules: moves a script forces on top of a policy, one implementation for every caller.

The tournament plays them over an agent (`script:<name>:<spec>`, tools/lib/batch_tournament.py) and
the trainer forces them as environment in seeded games (`--seed-scenarios`,
ai/training/show_and_decide.py). Each rule is `(state, legal mask) -> flat action or None`: the
move to force at this decision, or None to leave it to whoever is playing.

Engine and numpy only, so the trainer can import it without the tournament's dependencies.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy.typing as npt

import ts_engine as ts

_CMC, _LONE_GUNMAN, _ORTEGA, _CHE = 40, 62, 91, 107
_CUBA, _COSTA_RICA = 71, 68
_INFLUENCE_SLOT = 112     # OPS_INFLUENCE: on an opponent's card, Ops first, then its event
_NODE = 116               # POINT_NODE slots: 116 + country id


def _che_targets() -> List[int]:
    """Countries Che can coup: non-battlegrounds in Central America, South America and Africa,
    least stable first."""
    regions = (ts.Region.CENTRAL_AMERICA, ts.Region.SOUTH_AMERICA, ts.Region.AFRICA)
    infos = [ts.MapData.get_country_info(i) for i in range(84)]
    return [int(c["id"]) for c in sorted(infos, key=lambda c: (c["stability"], c["id"]))
            if c["region"] in regions and not c["battleground"]]


_CHE_TARGETS: List[int] = []


def _script_cmc_combo(st: "ts.GameState", mask: npt.NDArray[Any]) -> Optional[int]:
    """The US's Cuban Missile Crisis combo (owner, 2026-10-09). Holding the crisis and Lone
    Gunman, Che or Ortega, with no USSR Influence in Cuba: headline the crisis, then play the
    other card. Ortega and Che are played for Influence (Ops first, then the event) placed where
    the event's coup can reach it -- Ortega into Cuba or Costa Rica, Che into a non-battleground
    in the Americas or Africa -- so the USSR has something to coup. Anything else is the agent's."""
    ctx = st.ctx()
    if ctx.decision_player != ts.Player.US:
        return None
    dt = ctx.decision_type
    us = ts.Player.US
    held = lambda c: ts.in_hand_of(st.get_card_location(c), us)
    under = st.has_flag(ts.EffectBits.CMC_ACTIVE_US)
    if dt == ts.DecisionType.SELECT_CARD:
        if st.current_phase == ts.Phase.HEADLINE:
            if mask[_CMC - 1] and st.get_country(_CUBA).ussr_influence == 0 \
                    and any(held(c) for c in (_LONE_GUNMAN, _CHE, _ORTEGA)):
                return _CMC - 1
        elif under:
            for c in (_LONE_GUNMAN, _ORTEGA, _CHE):
                if mask[c - 1]:
                    return c - 1
        return None
    card = int(ctx.pending_op_card)
    if not under or card not in (_ORTEGA, _CHE):
        return None
    if dt == ts.DecisionType.SELECT_PLAY_MODE and mask[_INFLUENCE_SLOT]:
        return _INFLUENCE_SLOT
    if dt == ts.DecisionType.POINT_NODE and ctx.op_mode == ts.OpMode.INFLUENCE:
        if card == _ORTEGA:
            targets = [_CUBA, _COSTA_RICA]
        else:
            if not _CHE_TARGETS:
                _CHE_TARGETS.extend(_che_targets())
            targets = _CHE_TARGETS
        for t in targets:
            if mask[_NODE + t]:
                return _NODE + t
    return None


#: Scripted rules a `script:<name>:` agent plays on top of its own policy, by name. Each returns
#: the move to force, or None to leave the decision to the agent.
SCRIPTS: Dict[str, Callable[["ts.GameState", npt.NDArray[Any]], Optional[int]]] = {
    "cmc-combo": _script_cmc_combo,
}


