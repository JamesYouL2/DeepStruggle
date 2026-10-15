# An engine change that moves no decision

**Conclusion.** An engine change invalidates comparisons that straddle it only if it moves the
decision stream, and that can be measured cheaply instead of assumed. P14 — `StateMachine::step`
validating against `generate_flat_mask_212` instead of re-deriving each rule, the mask no longer
emitting unplayable actions, and the Missile Envy forced-play rule fixed (`0d2f499`;
[`../../archive/E3_ladder/plans/P14_one_definition_of_legality.md`](../../archive/E3_ladder/plans/P14_one_definition_of_legality.md))
— moved **nothing**: the same games through `b5f2c06` and the fixed build, comparing outcome, chosen
action **and the legal mask's popcount and hash** at every step:

| policy | games | steps | divergences |
|---|---:|---:|---:|
| policy-free deterministic walk | 300 | 55,786 | 0 |
| E3-20-28 @160M / @320M, argmax (pooled) | 512 | 234,473 | 0 |
| E3-17-25 @160M, argmax (unpooled) | 256 | 95,553 | 0 |
| **total** | **1,068** | **385,812** | **0** |

So the letter stayed E3 and `E3-20-28` remained the matched baseline for `E3-21-28`. The licence
comes from the measurement, not from the change looking small.

**The qualification.** The sweep covers fresh self-play. One archived replay
(`h2_480M_selfplay_20260502.tslog.json`, step 631) records a pass at a `SELECT_CARD` node where
passing was never legal and the pre-P14 `step` accepted it: **`step` used to be the more permissive
side**, which a sweep asking "does the mask offer something step refuses?" cannot see. Ten of eleven
archived replays reproduce exactly; the eleventh is kept in `data/replays/archive_pre_p14/`.

**Why nothing moved.** A 1.54M-position sweep of sampled self-play found zero natural mask/step
disagreements; the three C++ tests P14 broke were hand-built states, two of them asserting a coup
the rules forbid.

**How to repeat it.** `git worktree add --detach <tmp> <old-commit>`, build each engine into its own
directory, run the same dump script under each build in separate processes (two `ts_engine` modules
cannot share an interpreter), diff the masks.
