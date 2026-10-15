# Missile Envy removed a starred card whose Event never fired (fixed, E4)

**Conclusion.** When Missile Envy hands the player the opponent's own starred card for Operations,
its Event does not occur, so the card must be **discarded**; the engine **removed it from the game**
(constructed case: NATO taken by Missile Envy ended `REMOVED_FROM_GAME`). Fixed (`0aa3dc0`, part of
the E4 boundary): both routes into the use-for-Ops branch (`mid_war.cpp` untied, `card_dispatcher.cpp`
tied) now set `suppress_op_card_event = 1`, as UN Intervention already did. Pinned by
`MidCardsTest.Card49_MissileEnvy_StarredOpponentCardIsDiscardedNotRemoved`.

**Cause, and the hazard that remains.** The branch set `timing_branch = 255`, which stops the Event
*firing*, but relocation asks `event_occurred_on_ops_play`, which reads only
`suppress_op_card_event`. Any future path that blocks an Event by timing rather than by that flag
inherits the bug.

**Reachability.** Common in principle (NATO, Marshall Plan, De-Stalinization, Socialist Governments
are starred 3–4 Ops cards); a sweep of the human corpus for an affected game was never done. Context:
[`../../archive/E3_ladder/plans/P17_stack_layout.md`](../../archive/E3_ladder/plans/P17_stack_layout.md).
