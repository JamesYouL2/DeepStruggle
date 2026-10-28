"""A readable breakdown of what a scoring card would score in a position.

The arithmetic is the engine's, never re-derived here: the region scorings come from
`ts.Scoring.evaluate_region` (statuses, battlegrounds, the +1 per country adjacent to the
enemy superpower, Shuttle Diplomacy and Formosan Resolution), and Southeast Asia from running
`ts.Scoring.score_southeast_asia` on a clone. This module only lists the countries behind the
numbers. An earlier copy re-implemented the scoring rules, missed the superpower-adjacency
bonus and mapped card 88 (Marine Barracks Bombing) as Central America Scoring (card 37).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import ts_engine as ts

NUM_COUNTRIES: int = 84

#: Each regional scoring card and the region it scores.
REGION_SCORING_CARDS: Dict[int, Tuple[ts.Region, str]] = {
    1: (ts.Region.ASIA, "ASIA"),
    2: (ts.Region.EUROPE, "EUROPE"),
    3: (ts.Region.MIDDLE_EAST, "MIDDLE EAST"),
    37: (ts.Region.CENTRAL_AMERICA, "CENTRAL AMERICA"),
    79: (ts.Region.AFRICA, "AFRICA"),
    81: (ts.Region.SOUTH_AMERICA, "SOUTH AMERICA"),
}

SOUTHEAST_ASIA_SCORING: int = 38


def _side_name(player: ts.Player) -> str:
    if player == ts.Player.US:
        return "US"
    if player == ts.Player.USSR:
        return "USSR"
    return "NONE"


def _net_line(net: int) -> str:
    if net > 0:
        swing = f"US gains {net} VP"
    elif net < 0:
        swing = f"USSR gains {-net} VP"
    else:
        swing = "No VP swing"
    return f"│ • Net VP Outcome: {net:+d} VP ({swing})"


def _country_entry(state: ts.GameState, cid: int) -> Tuple[str, ts.Player]:
    info = ts.MapData.get_country_info(cid)
    country = state.get_country(cid)
    control = ts.Scoring.get_country_control(state, cid)
    entry = (f"{info['name']} (US:{country.us_influence}/USSR:{country.ussr_influence}, "
             f"Stab:{info['stability']})")
    if info["battleground"]:
        entry = f"★ [BG] {entry}"
    return entry, control


def _southeast_asia(state: ts.GameState) -> str:
    probe = state.clone()
    probe.victory_points = 0
    ts.Scoring.score_southeast_asia(probe)
    net = probe.victory_points

    lines: List[str] = ["┌─── 📊 REGIONAL SCORING AUDIT: SOUTHEAST ASIA (Card #38) ───",
                        "│ • Southeast Asian Countries (1 VP each, Thailand 2 VP):"]
    for cid in range(NUM_COUNTRIES):
        if not ts.MapData.get_country_info(cid)["in_southeast_asia"]:
            continue
        entry, control = _country_entry(state, cid)
        lines.append(f"│     - {entry} -> Control: {_side_name(control)}")
    lines.append(_net_line(net))
    lines.append("└─────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def format_regional_scoring_breakdown(state: ts.GameState, card_id: int) -> str:
    """The countries, statuses and VP a scoring card would score in `state`, as text.

    Raises ValueError for a card that is not a scoring card.
    """
    if card_id == SOUTHEAST_ASIA_SCORING:
        return _southeast_asia(state)
    if card_id not in REGION_SCORING_CARDS:
        raise ValueError(f"card {card_id} is not a scoring card")
    region, region_name = REGION_SCORING_CARDS[card_id]
    summary = ts.Scoring.evaluate_region(state, region)

    us_controlled: List[str] = []
    ussr_controlled: List[str] = []
    for cid in range(NUM_COUNTRIES):
        if ts.MapData.get_country_info(cid)["region"] != region:
            continue
        entry, control = _country_entry(state, cid)
        if control == ts.Player.US:
            us_controlled.append(entry)
        elif control == ts.Player.USSR:
            ussr_controlled.append(entry)

    lines: List[str] = [f"┌─── 📊 REGIONAL SCORING AUDIT: {region_name} (Card #{card_id}) ───"]
    for side, countries, total, bgs in (
            ("US", us_controlled, summary.us_countries, summary.us_battlegrounds),
            ("USSR", ussr_controlled, summary.ussr_countries, summary.ussr_battlegrounds)):
        lines.append(f"│ • {side} Controlled ({total} total, {bgs} BG):")
        lines.extend(f"│     - {c}" for c in countries or ["(None)"])
    for side, status, bgs, adjacent, score in (
            ("US", summary.us_status, summary.us_battlegrounds, summary.us_superpower_adjacent,
             summary.us_score),
            ("USSR", summary.ussr_status, summary.ussr_battlegrounds,
             summary.ussr_superpower_adjacent, summary.ussr_score)):
        lines.append(f"│ • {side + ' Standing:':<15} {status.name:<10} -> {score} VP "
                     f"({bgs} BG, {adjacent} adjacent to the enemy superpower)")
    if region == ts.Region.EUROPE:
        lines.append("│ • (Control of Europe wins the game outright; it scores no VP)")
    lines.append(_net_line(summary.net_delta))
    lines.append("└──────────────────────────────────────────────")
    return "\n".join(lines)
