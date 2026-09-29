"""Brinkman's board value: struggler's evaluator terms on ts_engine's map.

A port of `struggler/bots/strategic/evaluator.py`, which was written as pure functions over an
indexed snapshot precisely so that it could be moved. The terms are unchanged -- country value
(control, progress toward it, a reserve past it, and access to battlegrounds), region scoring
now, and the fitted per-country weights -- and so are the shipped default weights. What changed
is only the index space: countries are ts_engine's 0..83, read from `ts.MapData`, so a board is
read straight off a `GameState` with no name translation.

Dropped in the port: the Zobrist digest and the incremental `Position.place` bookkeeping, which
existed to make struggler's Python sandbox affordable. Here a position is rebuilt from the engine
state it describes, which cannot go stale.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import ts_engine as ts

US, USSR = 0, 1
NOBODY = -1
N_COUNTRIES = 84

# Region indices follow ts.Region: Europe, Asia, Middle East, Africa, Central/South America.
EUROPE, ASIA, MIDDLE_EAST, AFRICA, CENTRAL_AMERICA, SOUTH_AMERICA = range(6)
REGIONS = (EUROPE, ASIA, MIDDLE_EAST, AFRICA, CENTRAL_AMERICA, SOUTH_AMERICA)

# (presence, domination, control) VP; Europe's control is an automatic win, priced below.
SCORING_VP: Dict[int, Tuple[int, int, Optional[int]]] = {
    EUROPE: (3, 7, None),
    ASIA: (3, 7, 9),
    MIDDLE_EAST: (3, 5, 7),
    AFRICA: (1, 4, 6),
    CENTRAL_AMERICA: (1, 3, 5),
    SOUTH_AMERICA: (2, 5, 6),
}

#: What turning the position into a certain win is worth, in VP (struggler `stakes.py`).
AUTO_VICTORY_VP = 20.0
GAME_SWING_VP = 2 * AUTO_VICTORY_VP
EUROPE_CONTROL_VP = GAME_SWING_VP

#: P(reach becomes control by the next scoring), per stability 1..4 (struggler, measured).
CONVERSION_P = (0.406, 0.303, 0.304, 0.154)
CONVERSION_P_POOLED = 0.287

_DATA = Path(__file__).resolve().parent / "data" / "fitted_country_weights.json"


@dataclass(frozen=True)
class Weights:
    """struggler's `StrategicWeights` defaults, as shipped (policy.py, 2026-09-27)."""

    country_vp_scale: float = 2.795
    progress: float = 2.8
    reserve: float = 0.35
    access: float = 1.5
    access_decay: float = 1.445
    region: float = 2.6
    vp_base: float = 0.5
    vp_swing: float = 3.0
    military: float = 2.0
    scoring_final: float = 1.0
    coup_discount: float = 0.9


@dataclass(frozen=True)
class Terrain:
    """The static map, indexed by ts_engine country id."""

    names: Tuple[str, ...]
    neighbors: Tuple[Tuple[int, ...], ...]
    stability: Tuple[int, ...]
    battleground: Tuple[bool, ...]
    region_of: Tuple[int, ...]
    southeast_asia: FrozenSet[int]
    #: Countries adjacent to each superpower, [US, USSR].
    home: Tuple[FrozenSet[int], FrozenSet[int]]
    members: Dict[int, Tuple[int, ...]]
    #: A member of each region outside Southeast Asia, whose urgency is the region's own.
    region_anchor: Dict[int, int]
    #: Fitted per-side control weights, [US, USSR][country].
    fitted: Tuple[Tuple[float, ...], Tuple[float, ...]]
    thailand: int
    taiwan: int


@functools.lru_cache(maxsize=1)
def terrain() -> Terrain:
    names: List[str] = []
    neighbors: List[Tuple[int, ...]] = []
    stability: List[int] = []
    battleground: List[bool] = []
    region_of: List[int] = []
    sea: List[int] = []
    home: Tuple[List[int], List[int]] = ([], [])
    for i in range(N_COUNTRIES):
        info = ts.MapData.get_country_info(i)
        names.append(str(info["name"]))
        # Sorted by name, as struggler sorts: a float sum must not depend on neighbour order.
        nb = [int(n) for n in info["neighbors"] if int(n) < N_COUNTRIES]
        neighbors.append(tuple(sorted(nb, key=lambda k: ts.MapData.get_country_name(k))))
        stability.append(int(info["stability"]))
        battleground.append(bool(info["battleground"]))
        region_of.append(int(info["region"]))
        if info["in_southeast_asia"]:
            sea.append(i)
        adj = str(info["superpower_adjacent"])
        if adj == "US":
            home[US].append(i)
        elif adj == "USSR":
            home[USSR].append(i)
    members = {r: tuple(i for i in range(N_COUNTRIES) if region_of[i] == r) for r in REGIONS}
    anchor = {r: next(i for i in members[r] if i not in sea) for r in REGIONS}
    data = json.loads(_DATA.read_text())["weights"]
    fitted = (tuple(float(data[n]["US"]) for n in names), tuple(float(data[n]["USSR"]) for n in names))
    return Terrain(
        names=tuple(names), neighbors=tuple(neighbors), stability=tuple(stability),
        battleground=tuple(battleground), region_of=tuple(region_of), southeast_asia=frozenset(sea),
        home=(frozenset(home[US]), frozenset(home[USSR])), members=members, region_anchor=anchor,
        fitted=fitted, thailand=names.index("Thailand"), taiwan=names.index("Taiwan"),
    )


@dataclass
class Position:
    """One board: influence per side, and the control and reach it implies."""

    inf: Tuple[List[int], List[int]]
    control: List[int] = field(default_factory=list)
    reach: Tuple[List[bool], List[bool]] = field(default_factory=lambda: ([], []))

    @classmethod
    def of(cls, state: ts.GameState, t: Optional[Terrain] = None) -> "Position":
        t = t or terrain()
        us = [0] * N_COUNTRIES
        ussr = [0] * N_COUNTRIES
        for i in range(N_COUNTRIES):
            c = state.get_country(i)
            us[i] = int(c.us_influence)
            ussr[i] = int(c.ussr_influence)
        pos = cls(inf=(us, ussr))
        pos.refresh(t)
        return pos

    def refresh(self, t: Terrain) -> None:
        us, ussr = self.inf
        self.control = [
            US if us[i] - ussr[i] >= t.stability[i] else USSR if ussr[i] - us[i] >= t.stability[i] else NOBODY
            for i in range(N_COUNTRIES)
        ]
        reach: Tuple[List[bool], List[bool]] = ([False] * N_COUNTRIES, [False] * N_COUNTRIES)
        for s in (US, USSR):
            inf_s, r = self.inf[s], reach[s]
            for i in range(N_COUNTRIES):
                r[i] = i in t.home[s] or inf_s[i] > 0 or any(inf_s[k] > 0 for k in t.neighbors[i])
        self.reach = reach


def region_urgency(t: Terrain, region: int, urgency: Sequence[float]) -> float:
    return urgency[t.region_anchor[region]]


def fitted_importance(t: Terrain, urgency: Sequence[float], i: int, s: int) -> float:
    regional = region_urgency(t, t.region_of[i], urgency)
    value = t.fitted[s][i] * regional
    if i in t.southeast_asia:
        value += (2.0 if i == t.thailand else 1.0) * (urgency[i] - regional)
    return value


def importance(t: Terrain, w: Weights, urgency: Sequence[float], i: int, s: int) -> float:
    return w.country_vp_scale * fitted_importance(t, urgency, i, s)


def conversion_p(stability: int) -> float:
    return CONVERSION_P[min(max(stability, 1), len(CONVERSION_P)) - 1]


@functools.lru_cache(maxsize=None)
def route_weight(stability: int, base: float, routes: int) -> float:
    """The symmetric share of the capped geometric value of `routes` routes (struggler)."""
    decay = base * (1.0 - CONVERSION_P_POOLED) / (1.0 - conversion_p(stability))
    return sum(decay ** -step for step in range(routes)) / routes


def access(t: Terrain, pos: Position, i: int, s: int, w: Weights, urgency: Sequence[float]) -> float:
    """Reach a holding in `i` gives side `s` into uncontrolled battlegrounds the opponent cannot
    already reach, shared among the routes into each."""
    inf_s = pos.inf[s]
    reach_them = pos.reach[1 - s]
    total = 0.0
    for n in t.neighbors[i]:
        if t.battleground[n] and pos.control[n] != s and not reach_them[n]:
            routes = 1
            for m in t.neighbors[n]:
                if m != i and inf_s[m] > 0:
                    routes += 1
            if n in t.home[s]:
                routes += 1
            if inf_s[n] > 0:
                routes += 1
            weight = route_weight(t.stability[n], w.access_decay, routes)
            total += weight * importance(t, w, urgency, n, s) / t.stability[n]
    return total


def country_value(t: Terrain, pos: Position, i: int, s: int, w: Weights, urgency: Sequence[float]) -> float:
    us, ussr = pos.inf[US][i], pos.inf[USSR][i]
    own, opp = (us, ussr) if s == US else (ussr, us)
    margin = own - opp
    stability = t.stability[i]
    mine = importance(t, w, urgency, i, s)
    theirs = importance(t, w, urgency, i, 1 - s)
    value = mine if margin >= stability else -theirs if margin <= -stability else 0.0
    fraction = max(-1.0, min(1.0, margin / stability))
    value += w.progress * (mine * fraction if fraction > 0 else theirs * fraction)
    value += w.reserve * (mine * min(2, max(0, margin - stability))
                          - theirs * min(2, max(0, -margin - stability)))
    access_own = access(t, pos, i, s, w, urgency) if own > 0 else 0.0
    access_opp = access(t, pos, i, 1 - s, w, urgency) if opp > 0 else 0.0
    return value + w.access * (access_own - access_opp)


Overrides = Tuple[FrozenSet[int], FrozenSet[int]]
NO_OVERRIDES: Overrides = (frozenset(), frozenset())


def scoring_overrides(t: Terrain, pos: Position, region: int, *, formosan: bool, shuttle: bool) -> Overrides:
    extra: FrozenSet[int] = frozenset()
    ignored: FrozenSet[int] = frozenset()
    if formosan and region == ASIA and pos.control[t.taiwan] == US:
        extra = frozenset((t.taiwan,))
    if shuttle and region in (MIDDLE_EAST, ASIA):
        for i in t.members[region]:
            if t.battleground[i] and pos.control[i] == USSR:
                ignored = frozenset((i,))
                break
    return extra, ignored


def region_vp(t: Terrain, pos: Position, region: int, overrides: Overrides = NO_OVERRIDES) -> float:
    """Net VP for the US from scoring `region` now. Europe Control stands in as the whole game."""
    presence_vp, domination_vp, control_vp = SCORING_VP[region]
    extra, ignored = overrides
    counts = ([0, 0, 0], [0, 0, 0])  # controlled, battlegrounds, bonus
    total_bg = 0
    for i in t.members[region]:
        is_bg = t.battleground[i] or i in extra
        total_bg += is_bg
        holder = NOBODY if i in ignored else pos.control[i]
        if holder == NOBODY:
            continue
        tally = counts[holder]
        tally[0] += 1
        tally[1] += is_bg
        tally[2] += is_bg + (i in t.home[1 - holder])

    def value_for(s: int) -> Optional[float]:
        side_count, side_bg, bonus = counts[s]
        opp_count, opp_bg, _ = counts[1 - s]
        if total_bg > 0 and side_bg == total_bg and side_count > opp_count:
            return None if control_vp is None else float(control_vp + bonus)
        if side_count > opp_count and side_bg > opp_bg and side_count > side_bg:
            return float(domination_vp + bonus)
        if side_count > 0:
            return float(presence_vp + bonus)
        return float(bonus)

    us_value = value_for(US)
    if us_value is None:
        return EUROPE_CONTROL_VP
    ussr_value = value_for(USSR)
    if ussr_value is None:
        return -EUROPE_CONTROL_VP
    return us_value - ussr_value


def board_value(t: Terrain, pos: Position, s: int, w: Weights, urgency: Sequence[float],
                overrides: Optional[Dict[int, Overrides]] = None) -> float:
    """Every country's value and every region's weighted scoring now, for side `s`."""
    sign = 1 if s == US else -1
    ov = overrides or {}
    countries = sum(country_value(t, pos, i, s, w, urgency) for i in range(N_COUNTRIES))
    regions = w.region * sum(region_urgency(t, r, urgency) * sign * region_vp(t, pos, r, ov.get(r, NO_OVERRIDES))
                             for r in REGIONS)
    return countries + regions
