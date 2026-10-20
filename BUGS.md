# Known bugs

Defects that are known, reproducible, and not yet fixed. A bug leaves this file when it is fixed
and has a regression test, not when it is understood. Missing features and unfinished work belong
in the documentation for the area they concern.

Fixed bugs keep a one-line entry at the end, because code and research documents cite their IDs;
the full write-up is in git history.

---

## TEST-1 — the differential fuzzing suite does not run

**Area:** tests · **Severity:** low · **Status:** open

`tests/differential/` cross-checks this engine against an independent implementation
(`external/struggler`). Its modules fail at *import*, so `tests/conftest.py` keeps them out of
collection and they sit behind the `differential_fuzz` marker and a `--run-fuzz` flag. The
repository's cross-engine check is therefore not currently a check; nothing else depends on it.

**Fix.** Repair the imports and re-enable collection, or remove the suite. Do not leave it
collectable but broken: a suite that aborts the run is worse than one that is honestly gated.

---

## CONV-1 (remainder) — the committed verdict bank predates the final converter

**Area:** human replay conversion / eval · **Severity:** low · **Status:** open, owner's decision

The converter fix itself is done (below). But of the 100 positions in
`ai/eval/banks/disagreement_verdicts.jsonl`, 34 are no longer produced by the current converter:
nearly all differ in what the network observes (the mover's hand holds other cards, or fewer),
and one (replay 147, turn 9) is in a turn now left out as untrusted. 28 of the first 33 carry
marks. The reviewer
judged those exact positions, so the verdicts were not moved; carrying them to the current
positions, or re-reviewing, is the owner's call. `tests/training/test_disagreement_bank.py`
checks that each position loads and each mark is legal, not that the converter still produces it.

They also predate CONV-2: 94 of the 100 positions hold in the discard pile a card the log had not
spent by that turn, 7.1 on average, and the network observes the pile.

---

## BIND-1 — a long disagreement-bank run can stop on a nanobind instance collision

**Area:** bindings · **Severity:** low · **Status:** open, not reproduced

A long `tools/scripts/disagreement_bank.py` scan occasionally stops on a nanobind instance
collision (`tools/README.md` §7b); the scan was made resumable (`--resume`) rather than fixed. No
traceback or reproducer was recorded. A collision means two Python objects claimed one C++
address -- an object's lifetime, e.g. a `GameState` or runner slot freed while Python still names
it -- and the same fault could corrupt a position silently instead of stopping the run.

**Next.** Reproduce under the sanitizer build (`tools/scripts/run_asan.sh`) with a long scan, and
record the traceback here.

---

## CONV-3 — from turn 4 the converted piles still drift from the log's

**Area:** human replay conversion · **Severity:** low · **Status:** open

Since CONV-2 the discard pile at a converted decision is the log's through turn 3
(`tests/replayer/test_replay_deck_and_discards.py`), and 461 of the corpus's 469 `*RESHUFFLE*`
markers fall on the entry where the engine reshuffles. From turn 4 a decision's pile still holds
0.34 cards the log has not spent and lacks 0.06 it has, on average. Part of the 0.34 is the
measurement's own: an ongoing card such as Shuttle Diplomacy, played before a reshuffle and
discarded after it, is counted as never spent. What is known of the rest:

* **Draws the humans' game reshuffled in the middle of.** The engine's Our Man in Tehran draws at
  most what the deck holds and reshuffles only an empty deck; at turn 7 AR2 of replay 207 the log
  reshuffles inside the draw with two cards left. Ask Not's replacement draws do the same at turn
  7 AR7 of replay 272. The engine reshuffles a turn or two later instead, and every pile in
  between is wrong. Engine behaviour, so the owner's call (invariant 11).
* **A card received through Missile Envy is missing from the receiver's solved hand** until the
  entry that shows the receiver with it. In replay 109 the USSR headlines Missile Envy at turn 5;
  the engine holds it as `HAND_US_KNOWN`, the solved US hand for the USSR's AR1 leaves it out, and
  at those decisions it sits in the deck. Missile Envy is the card most often missing (339
  decisions in 74 games); only replay 109 has been traced.
* Not traced: four turn-7 deals where the log's deck runs out and the engine's does not (replays
  59, 119, 228, 302), and two mid-round markers (156 turn 6 AR4, 314 turn 7 AR6).

---

## Fixed (kept for the IDs cited elsewhere)

| ID | what it was | fixed in |
|:---|:---|:---|
| ENG-1 | UN Intervention's companion mask offered every card in hand (the correct filter in `action_mask.cpp` was unreachable); an illegal pick silently cost the action round | `3ef2d92` (P17 6a); `engine/AGENTS.md` §9, test in `engine/tests/test_cards_early.cpp` |
| ENG-2 | 2026-09-28 rules audit: seven rare-situation rules wrong (war-roll defender, Vietnam Revolts ladder, UN Intervention as an event, We Will Bury You on a trapped round, NORAD at 2, Grain Sales→UN with U-2, Summit counting scoring-only cards). Defines engine **E7** | `acc68d6`; `engine/AGENTS.md` §12, `engine/tests/test_rules_audit.cpp`, `research/findings/engine/engine_revisions.md` |
| CONV-1 | the human-log converter seated cards in hands the player never held (Our Man in Tehran peeks, reveals not used as evidence, late arrivals placed at AR1): over-limit hands in trusted turns 178 games / 2,590 decisions → 39 / 186. Rebuild anything made from converted human positions after it | `8aec4db`; `tests/replayer/test_replay_hand_contents.py` |
| CONV-2 | the human-log converter put every card the solved hands leave out in the discard pile, not the deck: at 99.7% of converted decisions the pile held cards nobody had played (14.2 on average) and the engine reshuffled where the humans had not, 24 games losing a SALT pick to it. Terrorism's random discard was never steered, and Five Year Plan's took the wrong card in 5 of 237 entries. Now 0.34 per decision from turn 4 (CONV-3), none before. Rebuild anything made from converted human positions after it | `0f112d7`; `tests/replayer/test_replay_deck_and_discards.py`, `tests/replayer/test_replay_five_year_plan_discard.py` |
| TODO (forced deal) | no way to drive a deal from a recording, so replays could not survive a shuffle change | `dfadce5`: `GameState.set_forced_deal` / `get_forced_deal_remaining`; `tests/bindings/test_forced_deal.py` |
