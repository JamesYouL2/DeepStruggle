# Known bugs

Defects that are known, reproducible, and not yet fixed. A bug leaves this file when it is fixed
and has a regression test, not when it is understood.

Anything here is a thing the code gets *wrong*, not a thing it does not do yet. Missing features
and unfinished work belong in the documentation for the area they concern.

---

## ENG-1 — UN Intervention offers companion cards the rules forbid

**Area:** engine · **Severity:** medium · **Status:** FIXED (P17 6a)

The rule now lives on the reachable path -- a `UN_INTERVENTION` case in
`CardHandlers::get_event_action_mask`'s `SELECT_CARD` switch, requiring an
opponent-associated, non-scoring card in hand. The copy in `action_mask.cpp` was left in
place but made identical, including the non-scoring half it had been missing, so the two
spellings cannot disagree again. Two spellings of one rule is how this survived: the
unreachable copy was right, so nobody reading it saw a bug.

Regression test drives `generate_flat_mask_212` rather than the handler directly -- a test
against the wrong function would have passed throughout the bug's life.

UN Intervention (#32) reads "play this card simultaneously with a card containing your opponent's
associated Event". The companion must therefore be an opponent-associated, non-scoring card. The
action mask offers **every card in the player's hand**, including scoring cards and the player's
own cards.

**Where.** Three pieces, and the middle one is the defect:

* `engine/src/events/early_war.cpp` — `trigger_un_intervention` gets the *gate* right: it only
  raises the decision when the player actually holds an opponent, non-scoring card. It then sets
  `ctx().resolving_card`.
* `engine/src/action_mask.cpp` — the `resolving_card != 0` branch runs first and delegates to
  `CardHandlers::get_event_action_mask`, returning before the dedicated UN Intervention branch
  below it. That branch has the rule right and is unreachable.
* `engine/src/card_dispatcher.cpp` — `get_event_action_mask`'s `SELECT_CARD` switch has no case
  for UN Intervention, so it falls to `default:`, which offers the whole hand.

So the rule exists twice and the reachable copy is the wrong one.

**Effect.** Nothing is corrupted. The resolution handler re-validates the choice and, on an
illegal one, falls through without discarding the named card or granting its Operations — so a
scoring card named this way stays in hand. But UN Intervention is spent, no event fires, no
Operations are granted, and the action round ends. **The player silently loses a whole action
round.**

That silent absorption is the second half of the bug, and the reason it went unnoticed: an illegal
action is swallowed rather than rejected.

**Scale.** About half of every offered companion list is illegal by rule. Trained policies pick an
illegal one in the low single digits of percent of the times they play the card.

**Fix.** Add a `case card_ids::UN_INTERVENTION:` to the `SELECT_CARD` switch in
`get_event_action_mask`, mirroring the test the resolution handler already applies
(`in_hand_of && side == opponent && !is_scoring_card`), and **delete** the now-provably-dead branch
in `action_mask.cpp` rather than leaving a second copy of a rule nothing reaches. A regression test
must pin the companion mask to opponent non-scoring cards only — the bug is precisely that a
correct filter existed and was never consulted.

**Why it is still open.** The fix changes the decision stream, so every trained checkpoint is then
playing a game it was not trained on, and results measured before and after are not comparable.
It is deferred to a point where that cost is acceptable, not because the fix is unclear.

---

## ENG-2 — rules audit, 2026-09-28: seven rules the engine got wrong

**Area:** engine · **Severity:** medium · **Status:** FIXED (2026-09-28)

War rolls ignored the defender's superpower; the Vietnam Revolts ladder fell back to a card's
printed Ops rather than the event's grant; UN Intervention could be played as an Event with no companion; We Will Bury You
skipped a trapped US round; NORAD armed on DEFCON being set to 2 at 2; the Grain Sales route to UN
Intervention paid no U-2 Incident; and Summit counted
the scoring-only Shuttle Diplomacy and Formosan Resolution.

Each is listed with its source in `engine/AGENTS.md` §12 and pinned by
`engine/tests/test_rules_audit.cpp`. All change the decision stream: the engine with them is
**E7**, expected not to differ measurably from E6 because every fix is on a rare situation
(`research/findings/engine/engine_revisions.md`).

The audit also found the era decks reshuffling the discard pile in at turns 4 and 8; `main` had
already fixed that independently (`c1df6f5`, engine letter E6), so it is not part of ENG-2.
`tests/replayer` passes against the human corpus with these fixes in place.

---

## TEST-1 — the differential fuzzing suite does not run

**Area:** tests · **Severity:** low · **Status:** open

`tests/differential/` cross-checks this engine against an independent implementation. Its modules
fail at *import*, which would abort a whole pytest run, so `tests/conftest.py` skips them at
collection and they sit behind the `differential_fuzz` marker and a `--run-fuzz` flag.

The consequence is that the repository's cross-engine check is not currently a check. Nothing else
depends on it, and no other suite is affected.

**Fix.** Repair the imports and re-enable collection, or remove the suite. Leaving it collectable
but broken is the one option to avoid, since a suite that aborts the run is worse than one that is
honestly gated.

---

## CONV-1 — the human-log converter holds cards in a hand that the player did not have

**Area:** human replay conversion (`tools/lib/ts_replayer_*`) · **Severity:** high · **Status:** open

A converted human position can show the deciding player holding cards they never had, and lacking
cards they did. The board and the VP are right -- the converter checks those against the log at
every entry -- but nothing checks a hand against the rules or against the log's hand list, so
the error is silent.

**What it looks like.** The converter starts a turn with filler cards in a hand and adds the
logged cards as the turn reaches them, without taking the fillers back out:

* replay 14, turn 6, US: at the headline the hand holds Blockade and Allende, which appear nowhere
  in the log's US hand for the turn, and lacks Junta, which the US plays at AR2. Junta is added at
  AR1 and neither filler leaves, so the US holds 9 cards (the China Card aside) after its headline,
  one more than the rules allow;
* replay 30, turn 4, US: Africa Scoring, Portuguese Empire Crumbles and Flower Power -- drawn at AR7
  by Our Man in Tehran and discarded on the spot -- are in the US hand from AR1, three over;
* replay 147, turn 9, USSR: 14 cards at AR1. That turn is also where the record stops, which the
  converter does flag (`truncated_at`), but it still hands the turn's decisions to `on_decision`.

**How much.** Over the 274 distinct corpus games, 178 have a hand over the legal limit in some
action round of a trusted turn: 291 game-turn-sides, 2,590 of 111,194 human decisions, usually 1-3
cards over. That counts only *over the limit*; a filler that stands in for a missing card at the
right count is not seen by it. In the disagreement bank, 1,143 of 28,732 positions (4%) are over
the limit or have a logged card still in the draw deck (`tools/scripts/bank_hand_check.py`).
Turns with a drawing event (Our Man in Tehran, Ask Not, Grain Sales, Missile Envy, Star Wars) are
about two thirds of the affected turns; the rest have none.

**What it touches.** Every observation built from a converted position carries the player's own
hand: the human BC dataset (`tools/build_human_dataset.py`), human-agreement probes, the census and
disagreement-bank tools. A "network" or "search" move at an affected position can be a card the
human never held.

**Mitigation in place (not a fix).** `tools/scripts/event_play_census.feed_corpus_game` no longer
passes on decisions from a game's untrusted turn (`untrusted_turn`: the turn of `truncated_at` or
`failure`), with a regression test on replay 147; `bank_hand_check.py` excludes the flagged bank
positions. A fix belongs in the converter's hand model -- the cards in each hand at each entry
should be exactly the solved hand, with cards drawn mid-turn entering at the event that draws them
-- and the regression test should check every converted position's hands against the rules'
limit and the log's hand list.

---

## TODO — no forced-deal affordance, so replays cannot survive a shuffle change

**Status:** DONE (P17). `GameState.set_forced_deal(player, cards)` names the cards the next deal
gives a player; `get_forced_deal_remaining` reports what is left of it. Per player, not one queue
in deal order, so a recording survives a change to the deal algorithm and not merely to the
shuffle. Consumed by one deal and then cleared, so a leftover cannot be applied to a deal it was
never recorded for. Inert in normal play. A named card that is not in the draw deck when the deal
reaches it is reported as an anomaly and that draw falls back to the RNG, rather than being
silently mis-dealt.

The premise for deferring it expired: the engine changed under P17 and the pre-merge replays had
to be archived because they could no longer be driven. Adding it now is what stops the next
breaking change from costing the same thing again.

Original report follows.

**Status (original):** deferred by the owner. Not urgent while the engine is not changing between
a replay's generation and its playback.

Die rolls have a designed override: a `ROLL_DIE` action carries the acting player's value in
`primary_id` and the opponent's in `secondary_id`, and `state_machine.cpp` documents it as "the
only source of a forced die". Card deals have no equivalent -- `StateMachine::deal_cards_to_hands`
draws from the deck through `state.rng_state` with no way for a caller to supply the result.

**Consequence.** A replay cannot be made fully self-contained. Even with the starting position
saved (`to_save_dict`/`state_from_save_dict`, added) and every die recorded, re-driving a recorded
game on an engine whose shuffle or draw order has changed diverges at the first deal -- silently,
producing a different game rather than an error.

**Fix when it matters.** A forced-deal affordance mirroring the forced die: the replay source
supplies the cards a deal produces, inert in normal play. Record *what was dealt* rather than deck
order, since that also survives a change to the deal algorithm itself and not merely to the
shuffle. See `research/plans/P13_one_game_driver.md` § "a replay must carry its own randomness".
