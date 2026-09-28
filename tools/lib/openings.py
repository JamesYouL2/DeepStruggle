"""Named opening setups that can be forced on an agent in place of its own placements.

A trained policy fights where it was placed and rarely opens a new front, so its own setup
decides much of what the rest of the game can look like. Forcing a known opening is how that gets
separated: same agent, same engine, different starting board.

One definition, used by both the probe that measures the effect (`ai/eval/forced_setup.py`) and
the replay generators that produce games to watch (`tools/lib/self_play.py`,
`tools/play_match.py`). If these drifted, a replay labelled "the human opening" would not be the
opening the numbers were measured on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts
from bindings.action_encoder import ActionEncoder

#: Flat action index of country `cid` at a POINT_NODE (`bindings/action_encoder.py`).
NODE_OFFSET = ActionEncoder.NODE_OFFSET

#: 6 USSR placements, then 7 US in Western Europe, then 2 US bonus.
SETUP_DECISIONS = 15

WEST_GERMANY, ITALY, EAST_GERMANY, POLAND, YUGOSLAVIA, IRAN = 7, 10, 14, 15, 18, 25
CANADA, FRANCE, HUNGARY, SOUTH_KOREA = 0, 8, 17, 44

#: The standard human opening, as (country, points) in placement order. The USSR overcontrols
#: Poland and East Germany by a point each; the US takes West Germany and Italy with a buffer and
#: spends its two bonus points on Iran.
USSR_OPENING: Sequence[Tuple[int, int]] = ((EAST_GERMANY, 1), (POLAND, 4), (YUGOSLAVIA, 1))
US_OPENING: Sequence[Tuple[int, int]] = ((WEST_GERMANY, 4), (ITALY, 3), (IRAN, 2))


def expand(opening: Sequence[Tuple[int, int]]) -> List[int]:
    """(country, points) pairs to one country id per influence point, in placement order."""
    return [cid for cid, n in opening for _ in range(n)]


#: A symmetric USSR opening that commits nothing to East Germany, so the two US setups below face
#: the same board and differ only in whether the US holds West Germany.
USSR_POLAND_HUNGARY: Sequence[Tuple[int, int]] = ((POLAND, 3), (HUNGARY, 3))
#: The setup M2d s3 settled on from 185M: West Germany left empty, Canada taken instead.
US_NO_WEST_GERMANY: Sequence[Tuple[int, int]] = (
    (CANADA, 2), (ITALY, 2), (FRANCE, 3), (IRAN, 1), (SOUTH_KOREA, 1))

OPENINGS: Dict[str, Dict[str, List[int]]] = {
    "human": {"US": expand(US_OPENING), "USSR": expand(USSR_OPENING)},
    "ph_west_germany": {"US": expand(US_OPENING), "USSR": expand(USSR_POLAND_HUNGARY)},
    "ph_no_west_germany": {"US": expand(US_NO_WEST_GERMANY), "USSR": expand(USSR_POLAND_HUNGARY)},
}


def scripted_setup_index(state: ts.GameState, side: str, opening: str,
                         cursor: Dict[str, int]) -> Optional[int]:
    """The flat action index of `side`'s next scripted placement, or None if there is none.

    None means "let the agent decide": either setup is over, or this side's script is spent.
    An *illegal* scripted placement raises instead, because falling through to the agent would
    produce a partly-forced opening -- which is neither opening, and would be reported as one.

    `cursor` is per-side and mutated in place; pass `{"US": 0, "USSR": 0}` at the start of a game.
    """
    if state.current_phase != ts.Phase.SETUP:
        return None
    script = OPENINGS[opening][side]
    k = cursor[side]
    if k >= len(script):
        return None
    idx = NODE_OFFSET + script[k]
    mask = np.asarray(ts.ActionMask.generate_flat_mask(state))
    if not mask[idx]:
        raise RuntimeError(
            f"{side} placement {k} of the '{opening}' opening (country {script[k]}, flat action "
            f"{idx}) is not legal at this setup decision -- refusing to fall back to the agent.")
    cursor[side] = k + 1
    return idx


def acting_side(state: ts.GameState) -> str:
    """Which side is making this decision.

    The decision player, never `phasing_player`: the USSR is the phasing player for the whole of
    setup, so keying off it sends every US placement down the USSR script.
    """
    return "US" if state.ctx().decision_player == ts.Player.US else "USSR"


def play_scripted_setup(state: ts.GameState, opening: str) -> ts.GameState:
    """A copy of `state`, a fresh game in SETUP, with the whole setup played by `opening`'s script
    for both sides. Raises if the script does not cover the setup or a placement is illegal, rather
    than returning a partly scripted opening."""
    from bindings.settle import SettleMode, settle

    st = state.clone()
    cursor = {"US": 0, "USSR": 0}
    for _ in range(4 * SETUP_DECISIONS):
        if st.current_phase != ts.Phase.SETUP:
            return st
        if st.ctx().decision_player == ts.Player.NONE:
            settle(st, SettleMode.CHANCE)
            continue
        idx = scripted_setup_index(st, acting_side(st), opening, cursor)
        if idx is None:
            raise RuntimeError(f"the '{opening}' opening ran out before setup ended ({cursor})")
        ts.Engine.step_flat(st, idx, False)
        settle(st, SettleMode.CHANCE)
    raise RuntimeError("setup did not end")


class ScriptedSetupOverride:
    """Replaces an agent's setup placements with a named opening inside a batched game loop.

    One instance per batch of games. Each game keeps a cursor per side, so a side's k-th setup
    placement is the script's k-th point whichever agent is on the other side, and a scripted side
    can face a side that sets up for itself. Setup rows are recognised from the observation's phase
    slot (exactly 0 in SETUP), the test training uses."""

    def __init__(self, num_games: int) -> None:
        self.cursor = {"US": np.zeros(num_games, dtype=np.int32),
                       "USSR": np.zeros(num_games, dtype=np.int32)}

    def apply(self, actions: np.ndarray, obs: np.ndarray, masks: np.ndarray,
              d_players: np.ndarray, rows: np.ndarray, opening: str) -> None:
        """Overwrite `actions[rows]` in place wherever the row is a setup placement."""
        from ai.training.rollout_buffer import setup_phase_slot

        rows = np.asarray(rows)
        if len(rows) == 0:
            return
        setup_rows = rows[np.asarray(obs)[rows, setup_phase_slot()] < (0.5 / 6.0)]
        for r in setup_rows:
            side = "US" if int(d_players[r]) == 1 else "USSR"
            k = int(self.cursor[side][r])
            script = OPENINGS[opening][side]
            if k >= len(script):
                raise RuntimeError(f"game {r}: {side} has no scripted placement {k} in '{opening}'")
            idx = NODE_OFFSET + script[k]
            if not masks[r][idx]:
                raise RuntimeError(f"game {r}: {side} placement {k} of '{opening}' (country "
                                   f"{script[k]}) is not legal here")
            actions[r] = idx
            self.cursor[side][r] = k + 1
