"""P27: an influence ops play, solved exactly, and the policy's choice measured against it.

An ops play spent on influence is a block of POINT_NODE decisions by one player with no chance
node between them: no dice, no card draw, no opponent move. So every legal allocation can be
enumerated from the position where the play starts, and each one scored. That turns short-horizon
"no plan" failures that are easy to eyeball into rates:

* **missed contested battleground:** some allocation takes control of a battleground the opponent
  has influence in (or controls), and the policy's allocation takes none;
* **uncontested instead:** in those same positions, the policy gains control of a country the
  opponent has no influence in;
* **reinforcing:** points the policy puts into countries it already controlled when the play
  started;
* and, against the network's own critic: the value it assigns to its own allocation against the
  best allocation by the same critic (`critic_regret`), and whether that critic-best allocation
  takes the contested battleground.

The comparison is the diagnosis. If the critic prefers the allocation that takes the battleground
and the policy does not play it, the network "knows" but cannot execute a multi-point plan one
point at a time: a learning problem in the policy. If the critic prefers the bad allocation too,
it is a value problem.

A block is identified from the decision context (engine/src/state_machine.cpp): POINT_NODE,
op_mode INFLUENCE, a pending ops card, no event resolving, and remaining_steps equal to the play's
pending ops value at its first point. It ends at the first decision that is not another point of
the same play. Stability and battleground flags come from `rules/map.json`, the spec the engine
implements; its country order is checked against the engine's.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

NODE_OFFSET = 116
N_COUNTRIES = 84
CONFIRM_DONE = 208
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@lru_cache(maxsize=1)
def country_table() -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """(stability[84], battleground[84], names) from rules/map.json, checked against the engine."""
    with open(os.path.join(_ROOT, "rules", "map.json"), encoding="utf-8") as f:
        countries = sorted(json.load(f)["countries"], key=lambda c: int(c["id"]))
    if len(countries) != N_COUNTRIES:
        raise ValueError(f"map.json has {len(countries)} countries, expected {N_COUNTRIES}")
    for c in countries:
        if ts.MapData.get_country_name(int(c["id"])) != c["name"]:
            raise ValueError(f"map.json country {c['id']} is {c['name']!r}, the engine calls it "
                             f"{ts.MapData.get_country_name(int(c['id']))!r}")
    stab = np.array([int(c["stability"]) for c in countries], dtype=np.int32)
    bg = np.array([bool(c["battleground"]) for c in countries], dtype=bool)
    return stab, bg, tuple(c["name"] for c in countries)


def influence(state: "ts.GameState") -> np.ndarray:
    """[2, 84] influence: row 0 US, row 1 USSR."""
    out = np.zeros((2, N_COUNTRIES), dtype=np.int32)
    for i in range(N_COUNTRIES):
        c = state.get_country(i)
        out[0, i] = int(c.us_influence)
        out[1, i] = int(c.ussr_influence)
    return out


def controlled(inf: np.ndarray, side: int) -> np.ndarray:
    """Boolean [84]: countries `side` (0 US, 1 USSR) controls."""
    stab, _, _ = country_table()
    return (inf[side] - inf[1 - side]) >= stab


def _ctx(state: "ts.GameState") -> "ts.DecisionContext":
    return state.ctx()


def is_block_start(state: "ts.GameState") -> bool:
    """The first point of an influence ops play."""
    if ts.Engine.is_terminal(state):
        return False
    c = _ctx(state)
    return (int(c.decision_type) == int(ts.DecisionType.POINT_NODE)
            and int(c.op_mode) == int(ts.OpMode.INFLUENCE)
            and int(c.pending_op_card) != 0 and int(c.resolving_card) == 0
            and int(c.remaining_steps) > 0
            and int(c.remaining_steps) == int(c.pending_ops_value)
            and int(state.current_phase) == int(ts.Phase.ACTION_ROUND))


def _in_block(state: "ts.GameState", card: int, player: int) -> bool:
    if ts.Engine.is_terminal(state):
        return False
    c = _ctx(state)
    return (int(c.decision_type) == int(ts.DecisionType.POINT_NODE)
            and int(c.op_mode) == int(ts.OpMode.INFLUENCE)
            and int(c.pending_op_card) == card and int(c.resolving_card) == 0
            and int(c.decision_player) == player)


def _legal(state: "ts.GameState") -> np.ndarray:
    return np.flatnonzero(np.asarray(ts.Engine.get_flat_action_mask(state, False)))


@dataclass
class Allocation:
    end: "ts.GameState"          # the first decision after the block
    actions: Tuple[int, ...]
    inf_end: np.ndarray


def enumerate_block(start: "ts.GameState", max_allocations: int = 50_000) -> List[Allocation]:
    """Every distinct end board of the influence play starting at `start`.

    Points are tried in non-decreasing country order, so each multiset is reached once; a
    transposition set on (influence, remaining points) drops duplicates the engine's cost rule
    still produces. The engine judges legality at every step. `max_allocations` bounds the
    search; exceeding it raises rather than returning a partial set.
    """
    if not is_block_start(start):
        raise ValueError("not the first point of an influence ops play")
    c0 = _ctx(start)
    card, player = int(c0.pending_op_card), int(c0.decision_player)
    out: List[Allocation] = []
    seen: set = set()

    def rec(state: "ts.GameState", acts: Tuple[int, ...], last_node: int) -> None:
        for a in _legal(state):
            a = int(a)
            if NODE_OFFSET <= a < NODE_OFFSET + N_COUNTRIES:
                if a - NODE_OFFSET < last_node:
                    continue
            elif a != CONFIRM_DONE:
                continue
            nxt = state.clone()
            ts.Engine.step_flat(nxt, a, True, False)
            inf = influence(nxt)
            if _in_block(nxt, card, player):
                key = (inf.tobytes(), int(_ctx(nxt).remaining_steps), False)
                if key in seen:
                    continue
                seen.add(key)
                rec(nxt, acts + (a,), a - NODE_OFFSET if a != CONFIRM_DONE else last_node)
            else:
                key = (inf.tobytes(), -1, True)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Allocation(nxt, acts + (a,), inf))
                if len(out) > max_allocations:
                    raise RuntimeError(f"more than {max_allocations} allocations")

    rec(start.clone(), (), 0)
    return out


def play_block(start: "ts.GameState",
               choose: Callable[["ts.GameState"], int]) -> Allocation:
    """The allocation `choose` (state -> flat action) plays from `start`, on a copy."""
    c0 = _ctx(start)
    card, player = int(c0.pending_op_card), int(c0.decision_player)
    st = start.clone()
    acts: List[int] = []
    while True:
        a = int(choose(st))
        acts.append(a)
        ts.Engine.step_flat(st, a, True, False)
        if not _in_block(st, card, player):
            return Allocation(st, tuple(acts), influence(st))


@dataclass
class BlockReport:
    ops: int
    side: int                                   # 0 US, 1 USSR
    n_allocations: int
    takeable_contested: bool
    policy_takes_contested: bool
    policy_gains_uncontested: bool
    policy_points: int
    policy_reinforce_points: int
    policy_bg_gain: int
    max_bg_gain: int
    critic_policy: float = float("nan")
    critic_best: float = float("nan")
    critic_best_takes_contested: bool = False
    critic_rank_of_policy: int = -1             # 0 = the critic's own favourite
    extra: Dict[str, float] = field(default_factory=dict)


def analyse(start: "ts.GameState", allocations: Sequence[Allocation], policy: Allocation,
            values: Optional[np.ndarray] = None, policy_value: Optional[float] = None) -> BlockReport:
    """The P27 measures for one play. `values[i]` is the critic's value of allocation i's end
    state from the mover's side, and `policy_value` the same for the policy's allocation."""
    _, bg, _ = country_table()
    c0 = _ctx(start)
    side = 0 if int(c0.decision_player) == int(ts.Player.US) else 1
    inf0 = influence(start)
    ctl0 = controlled(inf0, side)
    contested_bg = bg & (~ctl0) & (inf0[1 - side] > 0)
    uncontested = (~ctl0) & (inf0[1 - side] == 0)

    def gains(inf: np.ndarray) -> np.ndarray:
        return controlled(inf, side) & (~ctl0)

    takes = [bool((gains(a.inf_end) & contested_bg).any()) for a in allocations]
    bg_gain = [int((gains(a.inf_end) & bg).sum()) for a in allocations]
    pg = gains(policy.inf_end)
    placed = np.clip(policy.inf_end[side] - inf0[side], 0, None)
    rep = BlockReport(
        ops=int(c0.pending_ops_value), side=side, n_allocations=len(allocations),
        takeable_contested=any(takes),
        policy_takes_contested=bool((pg & contested_bg).any()),
        policy_gains_uncontested=bool((pg & uncontested).any()),
        policy_points=int(placed.sum()),
        policy_reinforce_points=int(placed[ctl0].sum()),
        policy_bg_gain=int((pg & bg).sum()),
        max_bg_gain=max(bg_gain) if bg_gain else 0,
    )
    if values is not None and len(values) == len(allocations) and policy_value is not None:
        best = int(np.argmax(values))
        rep.critic_best = float(values[best])
        rep.critic_policy = float(policy_value)
        rep.critic_best_takes_contested = takes[best]
        rep.critic_rank_of_policy = int((values > policy_value + 1e-6).sum())
    return rep
