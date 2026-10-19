"""tools/scripts/destal_values.py: the packages it prices, the worlds it deals, the deck it repairs
and the model it fits."""
from typing import Dict

import numpy as np
import ts_engine as ts

from bindings.settle import SettleMode, settle
from tools.lib.openings import play_scripted_setup
from tools.scripts.destal_values import (COUNTRIES, country_ids, design, fit, packages,
                                         repair_turn1_deck, world_start)


def _turn1() -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, 7)
    st = play_scripted_setup(st, "human")
    settle(st, SettleMode.CHANCE)
    assert int(st.turn) == 1
    return st


def _where(st: ts.GameState) -> Dict[int, int]:
    return {c: int(st.get_card_location(c)) for c in range(1, 111)}


def test_every_package_of_four_points_at_most_two_a_country() -> None:
    ids = country_ids()
    pk = packages([ids[n] for n in COUNTRIES], 4)
    assert len(pk) == 615  # the t^4 coefficient of (1 + t + t^2)^10
    assert len(set(pk)) == len(pk)
    assert all(len(p) == 4 and max(p.count(c) for c in p) <= 2 for p in pk)


def test_a_world_redeals_both_hands_and_nothing_else() -> None:
    st = _turn1()
    before = _where(st)
    a, b = world_start(st, 11, "both"), world_start(st, 11, "both")
    assert _where(a) == _where(b) and a.rng_state == b.rng_state  # a world is its seed
    worlds = [world_start(st, s, "both") for s in range(8)]
    for w in worlds:
        after = _where(w)
        for loc in set(before.values()):
            assert sum(v == loc for v in after.values()) == sum(v == loc for v in before.values())
        seen = (ts.CardLocation.DISCARD_PILE, ts.CardLocation.REMOVED_FROM_GAME,
                ts.CardLocation.ONGOING_EVENT, ts.CardLocation.UNAVAILABLE)
        assert all(after[c] == v for c, v in before.items() if v in [int(x) for x in seen])
    ussr = int(ts.CardLocation.HAND_USSR_UNKNOWN)
    assert len({tuple(c for c, v in _where(w).items() if v == ussr) for w in worlds}) > 1
    kept = [world_start(st, s, "mover") for s in range(4)]
    assert all({c for c, v in _where(w).items() if v == ussr} ==
               {c for c, v in before.items() if v == ussr} for w in kept)


def test_the_repair_returns_only_cards_nobody_can_have_discarded() -> None:
    st = _turn1()
    deck = [c for c in range(1, 111) if st.get_card_location(c) == ts.CardLocation.DRAW_DECK]
    held = [c for c in range(1, 111) if ts.hand_holder(st.get_card_location(c)) != ts.Player.NONE]
    spurious, played, event_discard = deck[0], deck[1], held[0]
    for c in (spurious, played, event_discard):
        st.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    name = ts.CardData.get_card_name
    log = {"hands": {"1": {"us": [name(c) for c in held if c != event_discard] + [name(event_discard)],
                           "ussr": []}},
           "all_turns": [{"num": "1", "card": name(played)}]}
    assert repair_turn1_deck(st, log) == [spurious]
    assert st.get_card_location(spurious) == ts.CardLocation.DRAW_DECK
    assert st.get_card_location(played) == ts.CardLocation.DISCARD_PILE
    assert st.get_card_location(event_discard) == ts.CardLocation.DISCARD_PILE


def test_the_fit_recovers_point_values_up_to_their_common_constant() -> None:
    countries = list(range(5))
    pk = packages(countries, 4)
    x = design(pk, countries)
    true = np.array([0.03, 0.01, 0.0, -0.01, -0.03, 0.02, -0.02, 0.0, 0.01, -0.01])
    rng = np.random.default_rng(0)
    y = 0.5 + rng.normal(0, 0.1, size=(4000, 1)) + x @ true + rng.normal(0, 0.05, size=(4000, len(pk)))
    beta, se = fit(y, x)
    want = true - true[:5].mean()
    assert np.all(np.abs(beta - want) < 4 * se + 1e-9)
    assert abs(beta[:5].mean()) < 1e-12
