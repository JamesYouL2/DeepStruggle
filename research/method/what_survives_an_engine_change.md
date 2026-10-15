# What survives an engine change — the assumption this record runs on

> While engine changes can make old arms incomparable with new, they (after the starred events
> change) don't affect the relative effect of training approaches. — the owner

A **methodological assumption**, not a result (hence `method/`): the *sign and rough size* of a
difference between two training configurations, each measured against its own matched control
inside one engine, transfers across a revision; the *absolute* rating of either does not. Its scope
starts at the starred-events batch of **2026-09-10** (E1 → E2: `9fa5b05`, `341088e` and eight
same-day commits including observation v2.3;
[`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md)), because before it
the deck itself shrank — a different game, not a noisier one. **Living reference.**

## What it licenses, and what it does not

**Licenses:** carrying a matched comparison (or a rejection — replacing the dense logit by per-entity
heads cost 219 Elo) across a later fix; not re-running a control when a fix leaves the decision
stream alone.

**Does not license:**

* comparing Elo, win rates or any absolute rating across revisions;
* comparing anything read through the self-play distribution — ending mixes, ply, battleground
  counts, blunder rates, and training dynamics (`entropy`, `clip_frac`, `adv_std_raw`, `kl_div`,
  critic metrics: fitted on E4 and checked on E3 they shrank to noise or reversed sign;
  [`detecting_collapse.md`](detecting_collapse.md));
* skipping the measurement because a change "looks small";
* anything about an observation change — a third axis that makes a checkpoint misread, silently at
  equal width;
* ignoring seeds and stopping point, the two confounds that actually invalidated results here
  ([`../archive/E3_ladder/findings/seed_variance.md`](../archive/E3_ladder/findings/seed_variance.md)).

## Evidence

**For:**

* **Re-runs across a letter, both level:** E5-01 (E4-61's recipe retrained on E5: +2 Elo, 51.0% ±
  0.6 over 7,200 games; [`../log/E5_01_engine_bump.md`](../log/E5_01_engine_bump.md)) and E7-01-44
  (E6-12-44's recipe on E7: 49.8% ± 1.1 on SWAs). Both boundaries were small.
* **The E1 re-anchor:** four players' ordering survived the 2026-09-05 fixes while every
  distributional table had to be re-measured
  ([`../log/engine_reanchor_and_human_control.md`](../log/engine_reanchor_and_human_control.md) §7)
  — rankings, not effect sizes, and before the scope begins.
* **P14 moved no decision** (1,068 games, 385,812 steps, masks included;
  [`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md)),
  and `BUGS.md` ENG-3 had predicted it would. The `ROLL_DIE` pair was likewise held at E3 by
  measurement. Both are changes that touch nothing self-play reaches.

**Against, or thin:**

* **No boundary that genuinely changes play has been crossed by a re-run** — not the starred fix,
  not E2 → E3, not E5 → E6's deck change (E6-01 retrained only 40–120M on it).
* **The cross-boundary anchor was never sized:** every P1 arm was rated against `E2-02-21-480M` on
  E3, a handicap called "small but not zero" with no number behind it.
* **A fix batch reversed two relative search conclusions** at `611d83e` (privileged vs honest search
  100% vs 58.3% → +2.5 pp ± 5.5; 384 vs 96 sims → −1.7 pp ± 5.6), though at 3–10× the sample size
  and mostly through harness fixes ([`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md) §9).
* **Game shape does not replicate even inside one engine** (arm H vs its replicate H2;
  [`../log/corrected_engine_arms_H_I.md`](../log/corrected_engine_arms_H_I.md)).

## Standing verdict

**Plausible, supported on small boundaries, untested on one that moves play.** Use it as a working
assumption with a cheap escape hatch: measure the decision stream (both builds, fixed seeds, diff
action and mask hash), record the verdict in the arm's row, and if the stream moves, re-run one arm
across the boundary — that is the experiment that would settle it.
