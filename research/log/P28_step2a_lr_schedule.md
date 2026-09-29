# P28 step 2a — a stepped learning rate does not cut the spread; averaging at a constant rate wins

Plan: [`../plans/P28_strength_on_E6.md`](../plans/P28_strength_on_E6.md), step 2a.

**Arms.** Both continue E6-04-44 from its 560M end state to 760M, with its flags and the published
E6-05 exploiters as a static league pool:

* **E6-06-44:** constant rate 3e-4.
* **E6-07-44:** `--lr-schedule step`, 3e-4 until 620M, then 1e-4 until 680M, then 3e-5.

`launch_flags --diff` shows only the schedule between them.

**Pre-registered.** First the late spread: the standard deviation across the nine 680–760M
snapshots of each one's result against the panel peak, E6-03-44@550M. Then the panel rule at
700/720/740/760M. Adopt the schedule if the spread halves and the panel result is not worse.

## Results

The round robin was greedy, 1,000 games per seat, over both arms' 680–760M snapshots and the three
neural panel members (`data/reports/p28_2a_rr.{md,json}`).

**Spread against the panel peak**, per snapshot, overall. The binomial noise of one cell alone is
1.12 points.

| arm | 680 | 690 | 700 | 710 | 720 | 730 | 740 | 750 | 760 | SD | mean |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E6-06-44 (constant) | 59.7 | 57.2 | 60.2 | 58.1 | 55.6 | 55.9 | 57.8 | 59.8 | 54.4 | **2.03** | 57.6 |
| E6-07-44 (stepped) | 58.0 | 61.0 | 61.5 | 60.5 | 59.9 | 57.0 | 57.9 | 55.7 | 60.5 | **1.99** | 59.1 |

**Panel rule, late block.** E6-07 − E6-06: US +0.2 ± 0.6, USSR +3.4 ± 0.6. Head to head over the late
block, E6-07 scores 51.0%. Elo is within about 5 points.

**Averaging each arm's 680–760M snapshots** (`weight_soup.py`, `data/reports/p28_2a_soups.{md,json}`):

| soup | Elo | vs panel, US | vs panel, USSR |
|:---|---:|---:|---:|
| **E6-06-44 (constant)** | **1662** | **81.8** | **81.7** |
| E6-07-44 (stepped) | 1603 | 75.3 | 78.7 |
| E6-04-44, 480–560M (step 1's best) | 1641 | 79.5 | 81.9 |

Head to head between soups:

| pairing | overall | US | USSR |
|:---|---:|---:|---:|
| constant soup vs stepped soup | **57.8%** | 56.5 | 59.1 |
| constant soup vs the 480–560M soup | 53.7% | 51.0 | 56.4 |

## Reading

* **The schedule is not adopted.** Its pre-registered mechanism failed: the spread did not shrink at
  all (1.99 against 2.03). Every E6-07 snapshot from 690M on was trained at 3e-5, and they still
  scatter about 1.7 points beyond binomial noise.
  * That spread is therefore not step-size noise in the weights. Tiny weight changes flip enough
    greedy decisions to move a snapshot's result.
  * Averaging smooths exactly that, which would explain why soups gain so much.
* **Raw snapshots at the lower rate are a little better as USSR** (+3.4 against the panel).
  * The soups reverse it. A soup of the constant-rate run beats a soup of the scheduled run 57.8%.
  * At a low rate the snapshots stay close together, so there is little to average. At a constant
    rate they spread around a better centre, and the average finds it.
* **Training at a constant rate keeps paying once averaged.** The 680–760M soup beats the 480–560M
  soup 53.7%, about +20 Elo for 200M more steps. The raw snapshots of the same run hide this
  entirely: their Elo is flat at 1506–1538.
  * The ~1M-game "plateau" is therefore largely an artefact of rating raw snapshots.
* **The recipe to carry forward is a constant rate plus weight averaging.** This is step 2b (E6-08-44,
  EMA τ 10M, running). The soups that worked averaged over 80M. If τ 10M is too narrow to match them,
  the next cell is a wider τ (40M) or a periodic soup of the last 80M as the rated model.
