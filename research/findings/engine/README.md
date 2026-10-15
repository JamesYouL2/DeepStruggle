# findings/engine — what the simulator does

Pinned to a commit. A finding here can invalidate **absolute** numbers wholesale: when a rules defect
or a lying instrument is found, the ratings measured through it are gone, not re-sized.

| file | what it settles |
|:---|:---|
| [`engine_revisions.md`](engine_revisions.md) | the E1 → E7 letters, what each boundary changed, and which comparisons each breaks; E7 = E6 in strength |
| [`engine_change_decision_stream.md`](engine_change_decision_stream.md) | P14 moved **no** decision (1,068 games, 385,812 steps, masks included); how to measure the next change |
| [`hidden_information_legality.md`](hidden_information_legality.md) | the legal action **set** can depend on the opponent's hidden hand (The Cambridge Five), which breaks determinized search |
| [`event_completion_categories.md`](event_completion_categories.md) | PARTIAL_OK / BEST_EFFORT / MUST_COMPLETE: early stop and fizzling are two facts, and the fixes are coupled (P17 §6a) |
| [`flattening_card_play.md`](flattening_card_play.md) | P17's merged card-resolution node (−17.5% decisions) and the end of `INFLUENCE` meaning "decline" |
| [`missile_envy_starred_removal.md`](missile_envy_starred_removal.md) | Missile Envy removed a starred card whose Event never fired (fixed at E4) |
| [`ply_headline_order.md`](ply_headline_order.md) | `ply` now orders headlines by resolution; the correction is 0.00026 SD, so earlier numbers stand |

Open defects live in [`../../../BUGS.md`](../../../BUGS.md); a fixed bug's *consequence* is written
up here. Instruments that reported confident nonsense: [`../../log/measurement_bugs.md`](../../log/measurement_bugs.md),
checklist in [`../../method/measurement_pitfalls.md`](../../method/measurement_pitfalls.md).
