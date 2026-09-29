# E6-01 — does a model that knows the corrected deck beat one that does not?

> **Ran on an abandoned intermediate engine** (`c1df6f5`, the era fix alone). On 2026-09-29 E6 was redefined as that fix plus the Kitchen Debates fix ([`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md)). Every arm here, E6-01 and its E5 controls alike, carried the Kitchen Debates bug, so the comparison still isolates the deck change; it says nothing about Kitchen Debates.

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

## Continued 600 → 680M (2026-09-28)

Owner: "tune it for 80M more steps to just check what's happening". Same flags, resumed from the
600M end state. The control is E5-21-43's E5 continuation at the same steps, with the same flags
on the old deck; its E5-23 exploiters published nothing, so its pool did not change either.

Round robin on E6, greedy, 2,000 games per seat per pairing (`data/reports/e6_01b_rr.{md,json}`).
The bars are B's twin self-play on E6: US 48.9, USSR 50.2.

**Against B = E5-21-43@560M:**

| E6-01-43 at | overall | US vs bar | USSR vs bar |
|:---|---:|---:|---:|
| 620M | 48.9 ± 1.5 | −2.0 | +0.6 |
| 640M | 46.3 | −2.0 | −4.6 |
| 660M | 46.8 | −2.0 | −3.7 |
| 680M | **54.1** | **+1.7** | **+7.4** |
| *E5 control, 640M / 680M* | *52.5 / 52.4* | *+0.7 / +3.8* | *+4.4 / +2.0* |

Each seat's error is ± 1.1.

**Against the E5 control at matched and neighbouring steps:**

| E6-01-43 | vs E5@640M | vs E5@680M |
|:---|---:|---:|
| 620M | 46.3 | 46.6 |
| 640M | 42.4 | 43.6 |
| 660M | 46.4 | 45.6 |
| 680M | 51.7 | **50.3 ± 1.5** |

Elo, same field: E6 at 620–680M is 1488 / 1474 / 1486 / 1527; B is 1498; the E5 control is
1514 / 1514.

**Reading.**

* **Snapshot-to-snapshot variation dominates.** Against the same opponent, E6-01-43 swings from
  46.3% (640M) to 54.1% (680M), about 8 points across 40M. No single snapshot can carry the
  question.
* **The 680M snapshot passes the per-seat rule against B** (US +1.7, USSR +7.4). But so does the
  E5 control at 680M (US +3.8, USSR +2.0). Both are the same further training from 560M, and at
  matched steps they are level (50.3 ± 1.5).
* **Over the block, the E6-trained snapshots are not better than the E5-trained ones.**
  * The eight E6-vs-E5 pairings average **46.6%**, about 3 points in the E5 model's favour, on
    the E6 engine.
  * Against B, the E6 block averages 49.0% and the E5 block 52.5%.
* **Conclusion: no evidence that learning the corrected deck helps, over 120M of fine-tuning.**
  * The E5-trained policy plays E6 at least as well as a policy tuned on it.
  * The deck change is real, but it does not move what these policies do by enough to show at
    this resolution.
  * The deficit of the E6 block is more likely the fine-tune's own trajectory noise than a
    systematic cost. A second fine-tune from the same state, with another seed, would separate
    the two.
