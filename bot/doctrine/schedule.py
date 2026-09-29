"""How much each region will still score: the per-country `urgency` vector the evaluator reads.

A compact version of struggler's `schedule.py` + `public_cards.py`. A scoring card's future
occurrence mass is summed over the same buckets:

  1. this turn    -- 1.0 if we hold it; P(the opponent holds it) if it is unseen,
  2. this cycle   -- P(it is still in the pile) x P(the pile deals it before the game ends),
  3. next cycle   -- a discarded (or about-to-be-played) card coming back after the reshuffle,
                     or a card whose era has not entered the deck yet,
  5. final score  -- the measured odds the game reaches final scoring (not Southeast Asia).

The deck walk is simpler than struggler's -- a fixed deal size per era rather than its measured
per-turn one -- which is the fidelity this port gives up first. Everything reads public
information only: our hand, the opponent's cards we have seen and the *count* of the rest, the
pile size, and the discard pile.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import ts_engine as ts

from bot.doctrine.evaluator import (AFRICA, ASIA, CENTRAL_AMERICA, EUROPE, MIDDLE_EAST, N_COUNTRIES,
                                    SOUTH_AMERICA, Terrain, Weights)

LAST_TURN = 10
#: P(the game reaches final scoring | it reached this turn), turns 1..10 (struggler: BPA 2026
#: round 4, six of 27 games reached final scoring).
FINAL_SCORING_ODDS = (6 / 27, 6 / 26, 6 / 26, 6 / 25, 6 / 22, 6 / 16, 6 / 15, 6 / 13, 6 / 8, 6 / 6)

SEA_SCORING = 38
#: Scoring card id -> the region it scores (Southeast Asia's is handled separately).
SCORING_REGION: Dict[int, int] = {1: ASIA, 2: EUROPE, 3: MIDDLE_EAST, 37: CENTRAL_AMERICA,
                                  79: AFRICA, 81: SOUTH_AMERICA}
#: Cards that enter the draw deck at the start of turn 4 and turn 8 (the China Card excluded).
_ERA_ENTRY = {1: 4, 2: 8}


def _entering(turn: int) -> int:
    era = {4: 1, 8: 2}.get(turn)
    if era is None:
        return 0
    return sum(1 for c in range(1, 111)
               if c != 6 and int(ts.CardData.get_card_info(c)["era"]) == era)


_ENTERING = {t: _entering(t) for t in range(1, LAST_TURN + 1)}


def _deal_size(turn: int) -> int:
    """Cards drawn at the start of `turn`: both hands refilled after a turn of play."""
    if turn == 4:
        return 2 * 7 + 2          # seven rounds played, and the hand limit rises from 8 to 9
    return 2 * (6 if turn <= 3 else 7)


def _hand(state: ts.GameState, p: ts.Player) -> List[int]:
    return [c for c in range(1, 111) if ts.in_hand_of(state.get_card_location(c), p)]


def urgency(state: ts.GameState, me: ts.Player, t: Terrain, w: Weights) -> Tuple[float, ...]:
    """Per country, the summed occurrence mass of every scoring card that counts it."""
    opp = ts.Player.USSR if me == ts.Player.US else ts.Player.US
    turn = int(state.turn)
    horizon = LAST_TURN - turn
    # Only the opponent's cards we cannot see compete with the pile for an unseen card. A card of
    # theirs we know is accounted for by name in `mass`, so counting it here would make every
    # unseen scoring card likelier to be in their hand the more of their hand we know.
    theirs = sum(1 for c in _hand(state, opp) if not ts.known_to_opponent(state.get_card_location(c)))
    pile = sum(1 for c in range(1, 111) if state.get_card_location(c) == ts.CardLocation.DRAW_DECK)

    # Walk the pile forward: when does it run out, and what share of it is dealt before then?
    reshuffle_in = horizon + 1
    remaining = pile
    shortfall = 0      # cards the exhausting deal takes from the recycled pile
    for k in range(1, horizon + 1):
        remaining += _ENTERING[turn + k]
        deal = _deal_size(turn + k)
        if deal >= remaining:
            reshuffle_in = k
            shortfall = deal - remaining
            break
        remaining -= deal
    pile_dealt = 1.0 if reshuffle_in <= horizon else (
        0.0 if pile == 0 else min(1.0, (pile - remaining) / max(1, pile)))
    # After a reshuffle: the recycled pile is roughly every entered, unremoved card outside the
    # two hands, and it deals until the game ends.
    after = 0.0
    if reshuffle_in <= horizon:
        entered = sum(1 for c in range(1, 111)
                      if c != 6 and state.get_card_location(c) not in
                      (ts.CardLocation.UNAVAILABLE, ts.CardLocation.REMOVED_FROM_GAME))
        entered += sum(_ENTERING[turn + k] for k in range(1, reshuffle_in + 1))
        recycled = max(1, entered - 2 * 9)
        # The deal that empties the old pile finishes from the reshuffled one, so its unfilled
        # part is drawn from the recycled cards too, before any later deal.
        deals = shortfall + sum(_deal_size(turn + k) for k in range(reshuffle_in + 1, horizon + 1))
        after = min(1.0, deals / recycled)

    final = FINAL_SCORING_ODDS[min(LAST_TURN, max(1, turn)) - 1] * w.scoring_final

    def mass(card: int) -> float:
        loc = state.get_card_location(card)
        once = card == SEA_SCORING
        total = 0.0
        if loc == ts.CardLocation.REMOVED_FROM_GAME or (once and loc == ts.CardLocation.DISCARD_PILE):
            return 0.0
        if ts.in_hand_of(loc, me):
            total += 1.0
            if not once:
                total += after
        elif ts.in_hand_of(loc, opp) and ts.known_to_opponent(loc):
            total += 1.0
            if not once:
                total += after
        elif loc == ts.CardLocation.DISCARD_PILE:
            total += after
        elif loc == ts.CardLocation.UNAVAILABLE:
            era = int(ts.CardData.get_card_info(card)["era"])
            if _ERA_ENTRY.get(era, 99) <= LAST_TURN:
                total += 1.0
        else:  # unseen: the draw pile or the opponent's hidden hand
            pool = theirs + pile
            p_opp = theirs / pool if pool else 0.0
            total += p_opp + (1.0 - p_opp) * pile_dealt
            if not once:
                total += after
        if not once:
            total += final
        return total

    region_mass = {region: mass(card) for card, region in SCORING_REGION.items()}
    sea_mass = mass(SEA_SCORING)
    out: List[float] = []
    for i in range(N_COUNTRIES):
        u = region_mass[t.region_of[i]]
        if i in t.southeast_asia:
            u += sea_mass
        out.append(u)
    return tuple(out)


def rounds_left(state: ts.GameState) -> int:
    total = 6 if int(state.turn) <= 3 else 7
    if state.current_phase == ts.Phase.HEADLINE:
        return total
    return max(1, total - int(state.action_round) + 1)

