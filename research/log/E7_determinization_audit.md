# Determinization audit (2026-10-13): no leak; a few deductions left unused

**Conclusion.** The determinization (C++ rollout root and Python D-MCTS) deals only the opponent's
unknown cards and the deck and never leaks a hidden card; the engine's `_KNOWN` marks explain every
revealed card. A few deductions (The Cambridge Five, Missile Envy, Our Man in Tehran, held cards)
are left unused.

**Question** (owner): does the rollout's determinization respect discards and, above all, the opponent's known cards?

**What determinizes.** `ts.determinize` (C++, `bindings/rollout_batch.cpp`: the rollout root, both builds) and
`ai.search.dmcts.determinize` (Python: turn pricing, D-MCTS) do the same thing: the pool is the opponent's
`HAND_*_UNKNOWN` cards plus the `DRAW_DECK`, shuffled and dealt back with both counts kept. Untouched: the discard
pile, removed cards, ongoing events, the China card (fixed at `ONGOING_EVENT`, its holder in `china_card_holder`),
`PEEKED_TEMP`, the decider's own hand, and every `HAND_*_KNOWN` card. An opposing headline chosen but not revealed is
forgotten and re-asked from the world's hand. Each world gets fresh dice (`rng_state` reseeded per world, per option
in the pricing), so rollouts never replay the real game's dice.

**Whether the engine marks what is known.** That is only as good as the `_KNOWN` marks, so every card-location
transition was logged over 800 random games (two seeds; scripts `detz/transitions.py`, `detz/known_steps.py` in the
job's scratch directory):

* Nothing enters an `_UNKNOWN` hand from a public place except through the deck: the discard → hand and
  unavailable → hand moves are the deal's reshuffle and the era transitions, inside one step.
* Nothing goes `_KNOWN` → `_UNKNOWN`.
* Every card that becomes known is explained: the deck running out during a deal (2,164 -- both hands are then the
  complement of what is public, and the engine reveals both), CIA Created (840), Lone Gunman (123), The Cambridge Five
  (99, the US's scoring cards), Grain Sales (18), SALT Negotiations (15, a card taken from the discard), Missile Envy
  (10, both halves of the swap, as a headline too), Aldrich Ames (7).

**The network's observation** gives an opponent's card its own slot only when it is known; an unknown one reads as
"deck or hidden" (`engine/src/observation.cpp`). So a rollout's opponent policy cannot see the decider's hidden cards
either.

**Deductions the worlds do not use** (worlds a player could rule out -- not leaks):

* The Cambridge Five: the USSR learns the US holds no *other* scoring card; a world may deal it one.
* Missile Envy: the giver's remaining cards are worth no more Ops than the card given.
* Our Man in Tehran: the US knows which cards went back into the deck, so none of them is in the USSR hand.
* Cards held over a turn end cannot be scoring cards.
