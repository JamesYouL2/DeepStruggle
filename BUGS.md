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

## Fixed (kept for the IDs cited elsewhere)

| ID | what it was | fixed in |
|:---|:---|:---|
| ENG-1 | UN Intervention's companion mask offered every card in hand (the correct filter in `action_mask.cpp` was unreachable); an illegal pick silently cost the action round | `3ef2d92` (P17 6a); `engine/AGENTS.md` §9, test in `engine/tests/test_cards_early.cpp` |
| ENG-2 | 2026-09-28 rules audit: seven rare-situation rules wrong (war-roll defender, Vietnam Revolts ladder, UN Intervention as an event, We Will Bury You on a trapped round, NORAD at 2, Grain Sales→UN with U-2, Summit counting scoring-only cards). Defines engine **E7** | `acc68d6`; `engine/AGENTS.md` §12, `engine/tests/test_rules_audit.cpp`, `research/findings/engine/engine_revisions.md` |
| CONV-1 | the human-log converter seated cards in hands the player never held (Our Man in Tehran peeks, reveals not used as evidence, late arrivals placed at AR1): over-limit hands in trusted turns 178 games / 2,590 decisions → 39 / 186. Rebuild anything made from converted human positions after it | `8aec4db`; `tests/replayer/test_replay_hand_contents.py` |
| TODO (forced deal) | no way to drive a deal from a recording, so replays could not survive a shuffle change | `dfadce5`: `GameState.set_forced_deal` / `get_forced_deal_remaining`; `tests/bindings/test_forced_deal.py` |
