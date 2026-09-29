# P29 bet 3 — the merged-influence view trains cleanly on E6 but plays ~235 Elo weaker

Plan: [`../plans/P29_big_bets_from_scratch.md`](../plans/P29_big_bets_from_scratch.md), bet 3.

**Arm.** E6-09-44: E6-03-44's recipe plus `--merged-influence`, from scratch to 560M, seed 44, solo.
It ran without `--ladder-head-center`, which the trainer refuses in the merged view.
`launch_flags --diff` against E6-03-44 shows exactly those two differences.

**Pre-registered.**
* **Strength:** against E6-03-44 at the late block (500/520/540/560M), on snapshots and on the
  480–560M soup, per seat against the panel. The owner's rule applies: better in one seat and not
  worse in the other.
* **Collapse:** a US seat pinned at ≤ 10% that has not recovered by 240M closes the bet.

## Training: no collapse

This is the first merged-view run with no collapse; P23's six E4 runs all collapsed.

| step | 20M | 40M | 100M | 160M | 200M | 240M | 280M | 400M | 560M |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| US self-play win rate | 42% | 49% | 47% | 49% | 38% | 40% | 46% | 43% | ~43% |

* Entropy fell together in both seats: 2.9 at the start, 1.6 at 20M, 1.07 at 100M, 0.67 at 560M.
* P23's collapses left the US seat stuck at 2.5–2.7. The 200–260M dip to 38–40% matches one the
  seed-43 control went through at the same steps.
* Against HeuristicBot it reached 100% by the fourth evaluation and stayed there.

So P29's diagnosis stands: the old sharpening rollout bands, not the view, caused P23's collapses.

## Strength: much weaker than its control, in both seats

Round robin, greedy, 1,000 games per seat (`data/reports/p29_bet3_rr.{md,json}`). ± is one
standard error.

**Panel rule, merged minus control:**

| | US | USSR |
|:---|---:|---:|
| snapshots (500–560M) | **−35.3 ± 0.6** | **−25.0 ± 0.6** |
| soups (480–560M) | **−30.1 ± 1.3** | **−22.6 ± 1.3** |

**Head to head:**

| | overall | US | USSR |
|:---|---:|---:|---:|
| snapshots | 21.1% | 16.4 | 25.9 |
| soups | 19.4% | 16.6 | 22.1 |

**Elo:** snapshots 1377 against 1612, soups 1465 against 1694, about −235 on either reading.

## A sanity check on the evaluation

* Packed tournaments give each agent its own view (`batch_tournament.py:181–188`).
* The loader builds E6-09-44's snapshots and soup with `merged_influence=True` and
  `head_center=False`, as trained. E6-03-44's are built with the E4 view and centred heads.
* The gap is therefore not an evaluation artefact.

## Goal probes at 560M

These are self-play in each model's own view (`data/reports/p29_bet3_goal_probes.{md,json}`):

| measurement | E6-09-44 (merged) | E6-03-44 |
|:---|---:|---:|
| US opening West Germany ≥ 4 | 0% | 100% |
| mean final turn | 7.35 | 8.20 |
| empty battlegrounds, turn 8 | 0.18 | 1.66 |
| uncontrolled battlegrounds, turn 8 | 1.88 | 5.01 |
| **avoidable losses avoided** | **83.5%** | **94.1%** |
| forced wins taken | 58.7% | 61.7% |
| own-DEFCON loss with an alternative | 0.7% | 1.3% |
| space exit spent on own/neutral card | 0.0% | 3.2% |

Its own games are more battleground-heavy and end earlier, but it fails to avoid an avoidable
loss about three times as often. Against outside opponents the style loses badly.

**Speed and decisions:**
* Late-run median 87.6k steps/s against 81.6k (+7%).
* Mean ply 116.5 against 121.5 (−4%), not the −11% the census predicted.
* The 3.3 h wall time for 560M is not the view's cost; at 87.6k steps/s the run needs 1.8 h, so
  the rest is host suspension.

## Verdict: bet 3 not adopted

* **Both seats are far worse than the control, on snapshots and on the soup.** At −235 Elo the
  gap is more than twice the between-seed variance (~100), so a seed-43 replicate would not change
  the decision.
* **What it settles.**
  * The merged view no longer collapses under flat 1.0, so P23's collapse question is answered.
  * But at this recipe and budget the view learns a worse policy, not a better one.
* **Not diagnosed.** The drop in avoidable losses avoided (94 → 84%) and the abandoned West
  Germany-4 opening are the leads.
  * The first plausible cause is P29 bet 4's: at the merged op-choice node the network chooses the
    first placement without seeing the Ops it is spending.
  * Testing that needs the owner's observation approval. It is not re-run here.

## Saturated, or still training? (owner's question, 2026-09-29)

**It saturated.** Both runs' 40M snapshots, 40–560M, were rated in one field with E6-03-44@550M:
29 models, greedy, 400 games per seat (`data/reports/p29_bet3_trace.{md,json}`). Each Elo value
carries roughly ±10–15 of noise.

| step | 40M | 80M | 160M | 240M | 320M | 400M | 480M | 520M | 560M |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E6-09-44 (merged) | 1113 | 1242 | 1379 | 1395 | 1441 | 1482 | 1452 | 1486 | 1478 |
| E6-03-44 (control) | 1266 | 1390 | 1490 | 1576 | 1664 | 1684 | 1717 | 1701 | 1734 |
| gap | −152 | −148 | −112 | −181 | −223 | −202 | −265 | −215 | −255 |

Elo gained per stretch:

| stretch | merged | control |
|:---|---:|---:|
| 80→240M | +153 | +186 |
| 240→400M | +87 | +107 |
| 400→560M | **−4** | **+50** |

* **The merged view levels off around 400M.** The control is still climbing through 560M.
* **It is behind from the first snapshot and learns more slowly at every stage**, so the gap widens
  from about −150 to about −250.
* **A longer run would not close it.** The view is not a slower learner that catches up; at this
  recipe it learns more slowly and levels off lower.
