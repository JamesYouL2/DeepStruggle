# Engine revisions — which game each arm was actually trained on

What the simulator did at each point, what each change fixed, and what it invalidated. **Rewritten
in place**; the experiments are in `../../log/`. When the letter bumps is
[`../../method/run_nomenclature.md`](../../method/run_nomenclature.md); the arms on each revision are
[`../../runs.md`](../../runs.md).

**The letter is coarse.** A letter bumps when the decision stream was judged to move; the many
`fix(engine)` commits inside a letter were judged not to, and twice that was measured
([below](#changes-held-inside-a-letter-by-measurement)).

## The ladder

| letter | boundary commit(s) | date | what changed in the game |
|:---|:---|:---|:---|
| **E1** | before `9fa5b05` | to 2026-09-09 | a starred card spent for Operations was **deleted from the game**. Arms A–G. Not runnable: their observation layouts are gone and `check_checkpoint_layout` refuses them by width |
| **E2** | `9fa5b05`, `341088e` (+ the 09-10 batch) | 2026-09-10 | the starred-card fix, plus observation v2.3. Arms H, H2, I |
| **E3** | `76e7385`, `6381cf4` | 2026-09-11 | the Aldrich Ames discard and the Star Wars pick made mandatory |
| **E4** | `7057251`, `0aa3dc0` (P17) | 2026-09-19 | Grain Sales flattened to one decision, Missile Envy's starred-card removal fixed, action space 212 → 220. The registry restarted here |
| **E5** | `b19b958`, `8d05d94` | 2026-09-26 | influence Ops spent in full; each Ops modifier carries its own limit; an owed Event no longer inherits the Ops play's stop |
| **E6** | `c1df6f5` + `c248bfb` | 2026-09-29 | the era transitions no longer shuffle the discard back (rule 4.4); Kitchen Debates spent for Operations is discarded |
| **E7** | `acc68d6`–`9cd3ed9`, merged as `dec2768` | 2026-09-30 | the 2026-09-28 rules audit: seven fixes, each on a rare situation (`BUGS.md` ENG-2, `engine/AGENTS.md` §12) |

## What each boundary showed

* **Inside E1**, two waves of rules fixes (2026-09-05, 09-06) bumped nothing because the scheme did
  not exist yet. The re-anchor after them is the first evidence on what an engine change costs:
  **rankings survived** (K=40's rate against the anchor 90.2% against 88.9%) while everything read
  off the self-play distribution had to be re-measured. The 09-06 free-coup fix (`567179b`, Che /
  Ortega couping where the opponent had no influence) accounted for **29 of the 64 "forced wins"**
  the human-corpus metric had counted
  ([`../../log/engine_reanchor_and_human_control.md`](../../log/engine_reanchor_and_human_control.md) §7, §8.4).
* **E1 → E2 is a batch**, not one commit: `9fa5b05` and `341088e` (a starred card spent for Ops or
  whose event cannot occur is discarded, not removed — 19 of 19 had been removed over 600 games),
  plus eight same-day fixes (event-occurred rule at five sites, named die-roll fields, `temp_cards`
  removed, six-deep event chains, Socialist Governments' cap, Europe Control as its own ending) and
  observation v2.3 (`dc08a39`). The starred bug stood for 380 of the first 389 commits: the deck
  shrank and those events never recurred. The record declares pre-fix Elo incomparable and then
  rates arms D and E in the corrected field ([`../../log/corrected_engine_arms_H_I.md`](../../log/corrected_engine_arms_H_I.md));
  an old checkpoint is a valid opponent even when its rating is not on the ladder.
* **E2 → E3** (two rare mandatory choices): the handicap to an E2 checkpoint on E3 is described as
  "small but not zero" and was **never measured**, although `E2-02-21-480M` was the P1 anchor.
* **E4 → E5** ([`../../log/P27_ops_block_stage1.md`](../../log/P27_ops_block_stage1.md)): the mask
  changes (the stop disappears from ~1/6 of random-play decisions) but an E4 checkpoint barely
  notices (E4-61-44@720M stopped early 0.06 times a game). **E5-01** retrained E4-61 on E5: +2 Elo,
  51.0% ± 0.6 over 7,200 games — the first re-run across a letter
  ([`../../log/E5_01_engine_bump.md`](../../log/E5_01_engine_bump.md)).
* **E5 → E6**: the era reshuffle had been wrong since `4f103fc` (day two of the engine); the human
  corpus confirms the rule (`*RESHUFFLE*` at turns 3 and 7, never 4 or 8) but the converter sets
  hands from the log, so the engine's deck was never checked. Kitchen Debates stayed in the US hand
  after an Ops play since `0ad8598`, and E5 models exploited it (4 of 12 self-plays). `c1df6f5`
  alone was briefly "E6"; E6-01 and the first E6-02 ran on it and are void.
* **E6 → E7**: the seven audit fixes are all rare (Brush War odds next to a superpower, UN
  Intervention without a companion, We Will Bury You on a Quagmire round, NORAD, U-2 through Grain
  Sales, Summit with Shuttle Diplomacy, Vietnam Revolts budget). The observation golden diverged at
  decision 311 of 480 and was regenerated; mask and layout unchanged, E6 checkpoints play on E7.
  **The re-run, E7-01-44 (E6-12-44's recipe on E7), is level**: 500–560M snapshots US −1.6 ± 0.6,
  USSR +0.5 ± 0.5, 47.1% head to head; SWAs 49.8% ± 1.1 — inside the pre-registered ±3 bounds, so
  **E6 measurements stay comparable to E7 ones**. E7's slower training (88k against 96k steps/s)
  was the host: six void smokes put current code on E7 at 93k, on E6 at 90k, old code on E6 at 92k;
  host throughput varies ~±7% over a day, so compare steps/s only between runs made together.

## Changes held inside a letter by measurement

| change | what was measured | where |
|:---|:---|:---|
| `d124cff`, `09c2498` — reject a mismatched `decision_type`; a chance node rolled 255 | no `ROLL_DIE` node ever reaches an agent (0 in 879 + 12,800 env-steps) | E3 registry ([`../../archive/E3_ladder/runs.md`](../../archive/E3_ladder/runs.md), *E3-21*) |
| **P14** — `StateMachine::step` validates against the mask; Missile Envy forced play (`0d2f499`) | 1,068 games / 385,812 steps, four policies, outcome, action **and mask** compared: 0 divergences | [`engine_change_decision_stream.md`](engine_change_decision_stream.md) |

`BUGS.md` ENG-3 had predicted the P14 fix would move the decision stream; it moved nothing.

## The observation is a separate axis

Layouts: legacy 4,293 floats, v2.1, v2.2, then **v2.3 at 3,824, the only one left**. An observation
change makes a checkpoint unloadable or, at equal width, silently misreading; `check_obs_width` and
`check_checkpoint_layout` catch width only. **Since 2026-10-01 additions are appended blocks**
(`obs_features`), so the base stays bit-identical: the first, `OPS_BUDGET` (3 floats), left 48,000 of
48,000 positions identical and changed only the fingerprint (5419a582… → 1d11c2f2…). What each layout
was worth: [`../../log/observation_layout.md`](../../log/observation_layout.md).

## Open

Only E5-01 (a small boundary) and E7-01-44 have been re-run across a letter; nothing was trained on
both sides of E1/E2 or E2/E3. See
[`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md).
