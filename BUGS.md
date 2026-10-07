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

## CONV-1 — the human-log converter held cards in a hand that the player did not have

**Area:** human replay conversion (`tools/lib/ts_replayer_*`) · **Severity:** high · **Status:** FIXED

A converted position could show the deciding player holding cards they never had. The board and
the VP were right -- the converter checks those against the log at every entry -- but nothing
checked a hand, so the error was silent. Before the fix, 178 of the corpus's 274 games held a hand
over the rules' limit in some action round of a trusted turn (2,590 of 111,194 decisions).

**Causes, and what changed.**

* *Our Man in Tehran.* The cards it shows the US are named in the turn's hand list, because the US
  saw them, but they come off the draw pile and are never held. The solver rightly left them out
  of the deal; the converter then seated every listed card the deal lacked as a late arrival, from
  the turn's first entry -- so they were in the US hand from AR1 (replay 30, turn 4: three cards).
  `_peeked_this_turn` now keeps them out of hands.
* *Reveals were not evidence.* A card revealed out of a hand (CIA Created, "Lone Gunman", Aldrich
  Ames) was in it, but the solver never read reveals, so it dealt other cards and the revealed
  ones came back as late arrivals: both at once (replay 147, turn 9: CIA Created reveals eight
  USSR cards; 14 held at AR1). `GameFacts.revealed` now requires them in the deal unless they
  arrived mid-turn. Where a turn's reveals make the model unsatisfiable, the converter finds the
  fewest such turns, solves without their reveals, and records them in
  `Conversion.reveal_conflict_turns`: replay 212 turn 6 ("Lone Gunman" reveals six US cards and
  the US then plays a seventh) and replay 321 turn 10 (Aldrich Ames reveals a card the model has
  in the discard pile). Those hands are not known to be right, so the turns' 65 decisions are
  driven but not emitted, to the samples or to `on_decision`. `solve_hands` itself never drops a
  reveal unless its caller asks to be told which: called plainly, it returns None. 256 of 274
  games are solved, as before.
* *Every late arrival came at the turn's start.* Only Ask Not and SALT had an arrival entry;
  anything else defaulted to the turn's first entry, i.e. AR1. `_arrival_entry` now places a card
  received from the opponent (Missile Envy, Grain Sales) at the entry that hands it over, an Ask
  Not draw at the Ask Not entry, and any other at the first entry that shows the side with it.
  Ask Not's draws need the second rule as well as `_mid_turn_acquisitions`: that function picks
  the draws by a preference among the turn's cards, and where the solved deal leaves out a
  different card, that card was drawn all the same (replay 35, turn 7: Brush War, drawn at the
  headline and played at AR6, was out of the US hand through AR1-AR5). A card the turn's list
  names that nothing shows the side holding is not seated at all, and is counted in
  `Conversion.listed_never_held` (replay 285, turn 7: both lists name ABM Treaty, which the USSR
  headlines, reclaims with SALT and plays; it sat in the US hand too).

**After.** 39 games / 186 decisions are over the limit, by one card in 180 of them, and every such
turn but two has a mid-turn acquisition (SALT, Ask Not, Missile Envy, Grain Sales) that makes the
extra card legal; the two are in replays 212 and 321, whose turns are now left out. The Ask Not
and listed-card rules also shrink the hands that came out *short*: below each round's limit by two
or more, 209 decisions in 64 games before them and 156 in 42 after -- what remains includes the
events that discard from a hand. The converted board and VP are unchanged (`tests/replayer`, and
`-m corpus_full`), and so are the decisions but for the two left-out turns. Regression tests:
`tests/replayer/test_replay_hand_contents.py`.

**Rebuild** everything made from converted human positions -- the human BC dataset
(`tools/build_human_dataset.py`), human-agreement probes, the card census and the disagreement
bank -- since the hands, and so the observations, changed.

`tools/lib/corpus_driver.feed_corpus_game` also stops passing on decisions from a game's untrusted
turn (where the record stops or the conversion fails), which the converter already kept out of its
own training data.

**Open: the committed verdict bank predates the final converter.** Of the 100 positions in
`ai/eval/banks/disagreement_verdicts.jsonl`, 33 are no longer produced by the converter they were
committed with (34 after the Ask Not and listed-card rules): 32 differ in what the network
observes -- in several the mover's hand holds other cards, or fewer -- and one (replay 147, turn
9) is in a turn now left out as untrusted. 28 of them carry marks. The reviewer judged those exact
positions, so the verdicts are not moved here; carrying them to the current positions, or
re-reviewing, is the owner's decision. `tests/training/test_disagreement_bank.py` checks that each
position loads and each mark is legal, not that the converter still produces the position.

---

## BIND-1 — a long disagreement-bank run can stop on a nanobind instance collision

**Area:** bindings · **Severity:** low · **Status:** open, not reproduced

Reported with the disagreement bank (`tools/README.md` §7b): a long `disagreement_bank.py` scan
occasionally stops on a nanobind instance collision, and the scan was made resumable
(`--resume`) to work around it rather than fixed. No traceback or reproducer was recorded. A
collision means two Python objects claimed one C++ address, which points at an object's lifetime
-- a `GameState` or a runner slot freed while a Python reference still names it -- and the same
fault could corrupt a position silently rather than stop the run.

**Next.** Reproduce under the sanitizer build (`tools/scripts/run_asan.sh`) with a long scan, and
record the traceback here.

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
