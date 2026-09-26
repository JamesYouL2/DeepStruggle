"""P27: an influence ops play, solved exactly, and the policy's choice measured against it.

An ops play spent on influence is a block of POINT_NODE decisions by one player with no chance
node between them: no dice, no card draw, no opponent move. So every legal allocation can be
enumerated from the position where the play starts, and each one scored. That turns short-horizon
"no plan" failures that are easy to eyeball into rates:

* **missed contested battleground:** some allocation takes control of a *contested* battleground
  -- one neither side controls and both sides can reach (own influence there or in a neighbour, or
  next to the own superpower) -- and the policy's allocation takes none. Taking *any* contested
  battleground counts: choosing one over another is not judged here, because which one is right
  is a longer-horizon question the rules cannot settle;
* **what it does instead:** in those positions, where the policy's points went (the contested
  battleground without reaching control, another battleground, its own countries, uncontrolled or
  opponent-held non-battlegrounds) and what they achieved (control of something else, an opponent's
  control broken, points left unspent);
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
def adjacency() -> Tuple[np.ndarray, np.ndarray]:
    """(adj[84, 84], superpower_adjacent[2, 84]) from rules/map.json; row 0 of the second is the
    USA, row 1 the USSR."""
    with open(os.path.join(_ROOT, "rules", "map.json"), encoding="utf-8") as f:
        countries = sorted(json.load(f)["countries"], key=lambda c: int(c["id"]))
    idx = {c["name"]: int(c["id"]) for c in countries}
    adj = np.zeros((N_COUNTRIES, N_COUNTRIES), dtype=bool)
    sp = np.zeros((2, N_COUNTRIES), dtype=bool)
    for c in countries:
        for nb in c.get("neighbours", []):
            adj[int(c["id"]), idx[nb]] = adj[idx[nb], int(c["id"])] = True
        side = {"USA": 0, "USSR": 1}.get(str(c.get("superpower_adjacent")))
        if side is not None:
            sp[side, int(c["id"])] = True
    return adj, sp


def access(inf: np.ndarray, side: int) -> np.ndarray:
    """Boolean [84]: countries `side` can place influence in -- own influence there or in a
    neighbour, or adjacent to its own superpower."""
    adj, sp = adjacency()
    own = inf[side] > 0
    return own | (adj[:, own].any(axis=1)) | sp[side]


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
    policy_gains_uncontested: bool             # gains control of a non-contested country
    policy_points: int
    policy_reinforce_points: int
    policy_bg_gain: int
    max_bg_gain: int
    #: where the policy's points went, by the country's state when the play started
    points_by_class: Dict[str, int] = field(default_factory=dict)
    #: what the policy's allocation achieved
    outcomes: Dict[str, bool] = field(default_factory=dict)
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
    opp_ctl0 = controlled(inf0, 1 - side)
    contested_bg = bg & (~ctl0) & (~opp_ctl0) & access(inf0, side) & access(inf0, 1 - side)
    uncontested = (~ctl0) & (~contested_bg)

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
    classes = {
        "contested battleground": contested_bg,
        "other uncontrolled battleground": bg & (~ctl0) & (~opp_ctl0) & (~contested_bg),
        "opponent-controlled battleground": bg & opp_ctl0,
        "own-controlled country": ctl0,
        "uncontrolled non-battleground": (~bg) & (~ctl0) & (~opp_ctl0),
        "opponent-controlled non-battleground": (~bg) & opp_ctl0,
    }
    rep.points_by_class = {k: int(placed[m].sum()) for k, m in classes.items()}
    opp_end = controlled(policy.inf_end, 1 - side)
    left = 0
    if not ts.Engine.is_terminal(policy.end) and policy.actions and policy.actions[-1] == CONFIRM_DONE:
        left = 1
    rep.outcomes = {
        "takes a contested battleground": rep.policy_takes_contested,
        "partial on a contested battleground": bool((placed[contested_bg] > 0).any()
                                                    and not rep.policy_takes_contested),
        "takes another battleground": bool((pg & bg & ~contested_bg).any()),
        "takes a non-battleground": bool((pg & ~bg).any()),
        "breaks an opponent's control": bool((opp_ctl0 & ~opp_end).any()),
        "stops with points unspent": bool(left),
        "no control change at all": bool(not pg.any() and not (opp_ctl0 & ~opp_end).any()),
    }
    if values is not None and len(values) == len(allocations) and policy_value is not None:
        best = int(np.argmax(values))
        rep.critic_best = float(values[best])
        rep.critic_policy = float(policy_value)
        rep.critic_best_takes_contested = takes[best]
        rep.critic_rank_of_policy = int((values > policy_value + 1e-6).sum())
    return rep
