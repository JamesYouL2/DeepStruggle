# E6-01 — does a model that knows the corrected deck beat one that does not?

**Question (owner, 2026-09-28).** E6 fixes rule 4.4: the era transitions at turns 4 and 8 no
longer shuffle the discard back into the deck ([`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md),
*The E5 → E6 boundary*). Every model so far learned the old deck. Does 40M of fine-tuning on E6
give a model that beats the untuned one?

**Arm.** E6-01-43: E5-21-43@560M (the best model) resumed from its 560M state on E6 for 40M,
with exactly the flags of its own E5 continuation's first 40M. `launch_flags --diff` against that
continuation shows only `--train-steps`. Both ran without exploiter training, with the three
published E5-22 exploiters as a static league pool. The continuation's 600M snapshot is
therefore the same 40M on the old deck.

All games were played on E6, greedy, `tools/tournament.py`:

* **A** = E6-01-43@600M, the fine-tuned model;
* **B** = E5-21-43@560M, the untuned model;
* **C** = E5-21-43@600M, from the E5 continuation.

Reports: `data/reports/e6_01_{bars_e5_21_560_twin,rr,trend_570,trend_580,trend_590}.md`.

## Result

**Bars:** B's twin self-play on E6 gives US 48.9% and USSR 50.2%, 2,000 games per seat.

| pairing, 2,000 per seat | A as US | A as USSR | overall |
|:---|---:|---:|---:|
| **A vs B** (bars 48.9 / 50.2) | 47.4 → **−1.5 ± 1.1** | 53.0 → **+2.8 ± 1.1** | **50.2 ± 1.5** |
| A vs C | 50.0 | 57.8 | 53.9 ± 1.5 |
| B vs C | 51.6 | 52.5 | 52.1 ± 1.5 |

Elo, same field: A 1512, B 1504, C 1485.

Trend of A's earlier snapshots against B, 500 per seat: 46.5% (570M), 47.5% (580M), 47.6%
(590M), then 50.2% at 600M.

## Reading

* **Not accepted by the pre-registered rule.** Against the untuned model the fine-tuned one is
  level overall (50.2 ± 1.5). By seat it is +2.8 as USSR, but −1.5 as US, below its bar.
* **The win over C is not evidence of an E6 effect.** A beats C (53.9%), with the interval
  clear of 50%. But the untuned B also beats C, 52.1%. C, the same 40M on the old deck, is the
  weaker of the three: the E5 continuation drifted down after 560M, as the E5-21 log already
  recorded. Measured against C, A leads B by about 1.8 points, which is within noise.
* **40M of E6 does not buy measurable strength.** Either an E5-trained policy barely suffers
  from the corrected deck, or 40M is too short for it to learn what to do differently. What the
  deck changes (which cards and scoring cards come round in turns 4–7 and 8–10) is information
  the policy only reads through the observation's card-location features, so the adjustment may
  be slow.
* **E5 checkpoints stay usable on E6** as opponents and starting points, with no measurable
  handicap at this resolution.
* One seed; each seat's rate has a standard error of about 1.1 points at 2,000 games.
