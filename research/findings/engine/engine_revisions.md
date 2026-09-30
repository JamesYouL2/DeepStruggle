# Engine revisions — which game each arm was actually trained on

What the simulator did at each point in the project's history, what each change to it fixed, and
what that change invalidated. **Update discipline: rewritten in place** — when a revision boundary
is re-dated or a new one opens, this file changes and the experiments stay in `../../log/`.

The three-letter scheme itself (when the letter bumps, what a name means) is
[`../../method/run_nomenclature.md`](../../method/run_nomenclature.md); the arms sitting on each
revision are [`../../runs.md`](../../runs.md). This file is the other half: not *what a name
means* but *what the engine did*, because an Elo number is a statement about a game and three
different games have been played here.

**The letter is coarse.** It has six values and the project has had several dozen `fix(engine)`
commits. A letter bumps when someone judged that the decision stream moved; the commits inside a
letter are the ones judged not to move it, and in two cases that judgement was measured rather
than assumed ([below](#changes-held-inside-e3-by-measurement)).

## The ladder

| letter | boundary commit(s) | date | what changed in the game |
|:---|:---|:---|:---|
| **E1** | everything before `9fa5b05` | to 2026-09-09 | a starred card spent for Operations was **deleted from the game**. Arms A–G and everything older. Not runnable today: their observation layouts are gone and `check_checkpoint_layout` refuses their checkpoints by width |
| **E2** | `9fa5b05`, `341088e` (+ the 09-10 batch below) | 2026-09-10 | the starred-card fix, plus observation v2.3. Arms H, H2, I |
| **E3** | `76e7385`, `6381cf4` | 2026-09-11 | the Aldrich Ames discard and the Star Wars pick made mandatory. Everything from P1 onward |
| **E4** | `7057251`, `0aa3dc0` (P17) | 2026-09-19 | Grain Sales flattened to one decision, Missile Envy's starred-card removal fixed, action space repacked 212 → 220. The registry restarted here ([`../../runs.md`](../../runs.md)) |
| **E5** | `b19b958`, `8d05d94` | 2026-09-26 | influence Ops spent in full; each Ops modifier carries its own limit; an owed Event no longer inherits the Ops play's stop ([below](#the-e4--e5-boundary)) |
| **E6** | `c1df6f5` + `c248bfb` | 2026-09-29 | two rules fixes: the era transitions no longer shuffle the discard back into the deck (rule 4.4), and Kitchen Debates spent for Operations is discarded instead of staying in the US hand. `c1df6f5` alone was briefly called E6 and is abandoned ([below](#the-e5--e6-boundary)) |

## E1 was not one engine either

Two waves of rules fixes landed inside E1 and neither bumped a letter, because the scheme did not
exist yet.

* **2026-09-05** — per-card headline decision frames (`df480ff`), Defectors cancelling the USSR
  headline however it reaches the table (`eee34c2`), Shuttle Diplomacy (`a67d62a`), the
  scoring-card trap rule and the last turn's hand fill (`1073ec8`), among others.
  [`../../log/engine_reanchor_and_human_control.md`](../../log/engine_reanchor_and_human_control.md)
  §7 re-anchored the ladder after them and is the project's first evidence on what an engine
  change costs: **the ordering survived** (K=40 > control > HeuristicBot > Random, and K=40's rate
  against the anchor read 90.2% against 88.9% before the change) while **everything measured
  through the self-play distribution did not** — the §4 deficiency tables, ending mixes and
  battleground counts had to be re-measured.
* **2026-09-06** — a free coup from Che or Ortega is still a coup (`567179b`). Before it, those
  two events ran their own target lists without consulting `Operations::can_coup` and offered
  coups against countries the opponent had no influence in; couping a battleground took DEFCON
  2 → 1 and ended the game. **29 of the 64 "forced wins" the human-corpus metric counted were
  these illegal coups**, which is most of why that result was withdrawn
  ([`../../log/engine_reanchor_and_human_control.md`](../../log/engine_reanchor_and_human_control.md)
  §8.4, and `engine/AGENTS.md` §8). A policy trained against it learned that an opponent's Ortega
  is a free win whenever a battleground sits next to Nicaragua.

## The E1 → E2 boundary is a batch, not a commit

"The starred events change" names two commits on 2026-09-10, and eight more landed the same day.
Anything that dates the assumption in
[`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md)
is dating this whole batch.

| commit | what it changed |
|:---|:---|
| `9fa5b05` | a starred card **spent for Operations** is discarded, not removed. Measured over 600 games: 19 of 19 starred cards played for Ops by their own side had been permanently removed; after, 12 of 12 discarded and 0 removed |
| `341088e` | the other half — a starred card whose event **fires and cannot occur** is discarded. Seven cards can reach that state; six of them were being removed |
| `0ad8598` | the event-occurred rule reaches the five sites outside the state machine that had drifted from it |
| `8990ba4` | the pending die roll gets named fields, and forced dice leave the state |
| `fc391f7`, `d72b21c` | `temp_cards` removed; the cards a decision is about are read from where the cards are |
| `d6f7ae0` | `node_counts` packed; event chains nest six deep instead of three |
| `e758b6d` | Socialist Governments enforces its own per-country cap |
| `9f55470`, `9591470` | Europe Control becomes its own ending rather than an ordinary 20 VP win |
| `dc08a39` | observation **v2.3** — a separate axis; see below |

The starred bug's consequence was structural rather than local: every starred own-card Ops play
deleted a card from the game, so the deck shrank, reshuffles came early, and those events could
never appear again for either side. It stood for 380 of the repository's 389 commits, and the
observation's `removed` and `not_in_game` slots carried wrong values throughout. The human corpus
could not have caught it — the converter checks per-country influence and never looks at card
locations (`9fa5b05`'s own commit message).

**What this boundary invalidated, as the record states it.**
[`../../log/corrected_engine_arms_H_I.md`](../../log/corrected_engine_arms_H_I.md): *"Nothing before
this is comparable to it. The fix changes the decision stream, so every Elo in §1–§24 is on a
different ladder."* Note that the same section then rates pre-fix arms D and E **in the
corrected-engine field**, at 1739.7 and 1763.2, below the post-fix arms — so the record asserts
incomparability and performs the comparison on the same page. Both readings are defensible (an old
checkpoint is a valid *opponent* even when its rating is not on the same ladder) but the tension is
real and is one of the reasons the assumption document exists.

## The E2 → E3 boundary

`76e7385` and `6381cf4`, 2026-09-11: the Aldrich Ames discard and the Star Wars pick stop being
optional. Two cards, both rare.

The standing handicap statement is
[`../../method/run_nomenclature.md`](../../method/run_nomenclature.md) rule 2 — an E2 checkpoint
evaluated on an E3 engine *"is playing a game it never trained on; the handicap is small for the
two mandatory-choice cards but it is not zero, and it biases in favour of the E3 arm"* — repeated
in [`../../log/P9_architecture.md`](../../log/P9_architecture.md) as *"the handicap is small — two
rare cards — but it runs against the reference, so it flatters the E3 arms slightly."*
**Neither statement has a measurement behind it.** Nothing in the record puts a number on the two
cards' frequency or on what the handicap is worth in Elo, and `E2-02-21-480M` is nonetheless the
project's standing rating anchor and the reference every P1 arm was rated against.

## The E4 → E5 boundary

Three rules fixes, 2026-09-26, all in [`../../log/P27_ops_block_stage1.md`](../../log/P27_ops_block_stage1.md):

* **Influence Ops are spent in full** (`b19b958`). An influence play offered CONFIRM_DONE at every
  point; it now does only when no country can take the next point. Realignment keeps its stop.
* **Each Ops modifier carries its own limit** (`b19b958`). Containment/Brezhnev to at most 4 (5
  for the China Card in Asia), Red Scare/Purge to no less than 1 on the final value, Vietnam Revolts
  beyond both. Only a 1-Op card under both Purge and Vietnam Revolts changes value (2 → 1).
* **An owed Event does not inherit the Ops play's stop** (`8d05d94`). After a realignment, Warsaw
  Pact Formed's mandatory choice could be declined.

The letter bumps because the legal mask changes: in random play the stop disappears from about a
sixth of decisions. What an E4 checkpoint *does* barely changes. E4-61-44@720M stopped an influence
play early 0.06 times per game at temperature 0.1, and the observation keeps the value E4
checkpoints were trained on (`ALLOW_EARLY_STOP` reads 1 throughout an influence play). An E4 checkpoint
evaluated on E5 is therefore a small, measured handicap rather than an unknown one. **E5-01**
retrained E4-61's configuration on E5 to test whether training notices at all, and it does not.
Against E4-61 the pair's late mean is +2 Elo, and the head to head is 51.0% ± 0.6 over 7,200 games
([`../../log/E5_01_engine_bump.md`](../../log/E5_01_engine_bump.md)). This is the first arm in the
repository re-run across a letter boundary.

## The E5 → E6 boundary

**One rules fix, 2026-09-28: the era transitions leave the discard pile alone.** At the start of
turn 4 the engine added the Mid War cards *and* shuffled the whole discard pile back into the
deck, and did the same with the Late War cards at turn 8. Rule 4.4 says the discards "remain in the
discard pile for now, but will be reshuffled into the deck in the next reshuffle", which is when
the deck runs out. `rules/rules.md` (the project's own spec) always said the same; the engine never
implemented it.

* **Since when.** `4f103fc` (2026-08-21), the second day of the engine, which made the era
  mechanism work at all (before it, every card sat in the deck from turn 1) and added the two
  reshuffles in the same change. **Every letter from E1 to E5 has it.**
* **Found by** the owner in a workbench game: Red Scare/Purge, spent at turn 3 AR2, was dealt to
  the US at turn 4 along with the other ten discards (`data/ts_red_scare_bug.json`).
* **The human corpus confirms the rule.** Of the 300 games, `*RESHUFFLE*` appears at turn 3 in all
  283 that reach it and at turn 7 in 199 of 210, and never at turn 4 (274 games) or turn 8 (192).
* **Why nothing caught it.** The converter's hand solver has the rule right
  (`tools/lib/ts_replayer_hands.py`, "the era brings in is separate and does not disturb the
  pile"), but the converter then sets each turn's hands from the log (`_set_hand`), so the engine's
  own deck was never checked against a real game.
* **What changes in play.** Cards spent in turn 3 after its reshuffle -- the non-starred scoring
  cards among them -- wait until the deck runs out (about turn 7) instead of returning at turn 4;
  the Mid War discards likewise stay out at turn 8. The decks of turns 4–7 and 8–10 are different
  games, and so is when a scoring card comes round.

The legal mask and the observation layout are unchanged; the decision stream changes from the
turn-4 deal on. E5 checkpoints load and play on E6, but learned the old deck.

**A second fix joins E6 (2026-09-29): Kitchen Debates spent for Operations.** The owner played
Kitchen Debates for Operations as the US at turn 6 AR3 and again at turn 7 AR6 of one workbench
game (`data/ts_workbench_2026-09-28T23-54-31-730Z.tslog.json`); it never left the US hand.
`relocate_played_card` skipped Kitchen Debates unconditionally, because its handler placed the
card itself -- a leftover from before `event_has_effect` existed (`0ad8598`, 2026-09-10). A US
play for Operations runs no Event, so no handler, and the card stayed, playable every action
round. The fix removes the special case: the handler scores and every caller relocates the card
on its own `event_has_effect` answer, as for NATO, Solidarity and Our Man in Tehran.

* **Since when:** `0ad8598`, so E3, E4, E5 and the abandoned intermediate E6 all have it.
* **The models exploit it:** in 4 of 12 E5 self-plays (E5-11-43 and E5-21-43 at 560M) the US
  replays Kitchen Debates within a turn, once in five straight action rounds (turn 9 of
  `e5_21_43_560M_selfplay_seed6`) -- a permanent spare card for the US.
* **The intermediate engine.** `c1df6f5` (the era fix alone) was called E6 for a day. E6-01-43
  and the first E6-02-44 ran on it; the owner abandoned it and E6 is the two fixes together. The
  E6-01 finding (40–120M on the corrected deck buys nothing measurable) is a statement about the
  deck change only, and both of its arms carried the Kitchen Debates bug.

## Changes held inside E3 by measurement

Two `engine/` changes were deliberately *not* given a new letter, and in both cases the record says
what was measured rather than what was assumed. That is the practice this project has converged on.

| change | what was measured | where |
|:---|:---|:---|
| `d124cff`, `09c2498` — reject a mismatched `decision_type`; a chance node's only legal action rolled 255 instead of a die | no `ROLL_DIE` node is ever handed to an agent (0 in 879 single-env and 12,800 vectorized env-steps) and the runner always forces die 0 | [`../../runs.md`](../../runs.md), *E3-21* |
| **P14** — `StateMachine::step` validates against the mask; the Missile Envy forced-play rule fixed (`0d2f499`) | 1,068 games / 385,812 steps under four policies, comparing outcome, length, chosen action **and the legal mask itself** at every step: 0 divergences | [`engine_change_decision_stream.md`](engine_change_decision_stream.md) |

Both kept `E3-20-28` usable as a matched baseline for `E3-21-28`, which is the difference between
running one arm and two.

## The observation is a separate axis, and a harder one

The letter tracks the *game*. The observation tracks what the network is told about it, and it has
its own history — legacy 4,293 floats, v2.1, v2.2, then **v2.3 at 3,824 floats, the only one that
still exists**. The two axes moved together at the E1 → E2 boundary (`dc08a39` landed with the
starred fix) which is why arms D through G cannot be re-rated even in principle: v2.2 is retired
with its `temp_card_count` slot.

An observation change is stricter than an engine change in one specific way. An engine change
alters what the same checkpoint *plays*; an observation change makes the checkpoint unloadable, or
worse, loadable and silently misreading — the failure mode that produced four separate measurement
bugs ([`../../method/measurement_pitfalls.md`](../../method/measurement_pitfalls.md), *the input the
model was given*). `check_obs_width` and `check_checkpoint_layout` now refuse by width, which
catches a changed *width* and cannot catch changed *content* at the same width. What each layout
was worth is [`../../log/observation_layout.md`](../../log/observation_layout.md).

## The 2026-09-28 rules audit moves the decision stream

Seven rules fixes (`BUGS.md` ENG-2, `engine/AGENTS.md` §12). They change what is legal and what it
is worth -- Brush War odds, when UN Intervention is an Event, when We Will Bury You pays --
so by [`../../method/run_nomenclature.md`](../../method/run_nomenclature.md)
this is a letter boundary, and every E6 checkpoint now plays a game it was not trained on. The
deterministic walk in `tests/engine_logic/test_observation_golden.py` diverged at decision 311 of
480; every observation before that point was bit-identical, so the extractor is unchanged and only
the game moved. The golden
was regenerated for that reason. **The letter itself has not been bumped** -- naming the next
engine is the owner's call.

## Open, and worth knowing

* **One arm has been re-run across a letter boundary: E5-01**, E4-61's configuration on E5, with
  no effect (+2 Elo). No configuration was trained on E1 and again on E2, or on E2 and again on E3.
  The E4 → E5 change is small, so this says nothing about the larger earlier boundaries. See
  [`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md).
* **`BUGS.md` ENG-1** (UN Intervention offers companion cards the rules forbid) was open through
  E3, which means that engine was known to be playing a slightly wrong game. It is fixed.
* **ENG-3's own text predicted the opposite of what P14 measured.** It said of the forced-play fix:
  *"they change what is legal, so they change the decision stream and invalidate comparisons across
  the change."* The change landed and moved no decision in 385,812 steps. The prediction was a
  reasonable default; the measurement is what settled it.
