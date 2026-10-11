"""What the position-bank tools share: a position's seed, a move's name, the engine's exact
verdict on a move, and the searched continuation a `--continue-with SIMS:K` flag asks for.

Used by `tools/scripts/bank_playouts.py` (the disagreement bank's playouts and the blunder
census's validation), `blunder_census.py`, `blunder_bank.py` and `disagreement_bank.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import require_e4_view, rules_json
from tools.lib.player_agent import load_agent, resolve_device

CARD_NAMES: Dict[int, str] = {int(c["id"]): str(c["name"]) for c in rules_json("cards.json")}


def position_seed(rid: str, seed: int) -> int:
    """A position's seed: its row id (16 hex digits) mixed with the run's seed, kept below 2**63."""
    return (int(rid, 16) ^ (seed * 0x9E3779B97F4A7C15)) & 0x7FFF_FFFF_FFFF_FFFF


def move_name(st: ts.GameState, a: int) -> str:
    """A move as a reader says it: the card, or EVENT / SPACE / OPS (influence, coup, realign)."""
    off = int(ActionEncoder.PLAY_MODE_OFFSET)
    if a < off:
        cid = int(ts.decode_flat_action(st, a).primary_id)
        return CARD_NAMES.get(cid, f"card {cid}")
    mode = a - off
    return {0: "event", 1: "space", 2: "Ops: influence", 3: "Ops: coup", 4: "Ops: realign"}.get(
        mode, ActionEncoder.get_action_name(st, a))


def exact_mistake(greedy: int, labels: Dict[int, str]) -> Optional[str]:
    """The engine's own verdict on the move `greedy` from `classify_legal_actions`' labels, not an
    estimate: "exact-missed-win" where a legal move wins on the spot and greedy's does not,
    "exact-forced-loss" where greedy ends the game against the mover while some legal move does
    not, None otherwise."""
    if any(l == "win" for l in labels.values()) and labels.get(greedy) != "win":
        return "exact-missed-win"
    if labels.get(greedy) == "loss" and any(l != "loss" for l in labels.values()):
        return "exact-forced-loss"
    return None


def searched_continuation(model: str, sims_k: str) -> Tuple[Any, str]:
    """`--continue-with SIMS:K`: a Gumbel root of the model playing both sides of the playouts,
    and the spec as a row records it. The caller reseeds the agent per position, so one
    position's play does not depend on which positions preceded it."""
    sims, k = (int(x) for x in sims_k.split(":"))
    spec = f"gumbel:{model}:{sims}:{k}"
    agent = load_agent(spec, device=resolve_device("cuda"))
    require_e4_view(spec, agent)
    return agent, f"gumbel:{sims}:{k}"
