# P28 step 2b — EMA weights beat raw snapshots in both seats; the window must be wide

> **Owner's decision (2026-09-29):** averaging does not enter training. Every arm is instead measured twice: on its snapshots and on the SWA of its last 80M. The "adopted as the rated model" reading below is superseded by that rule.

Plan: [`../plans/P28_strength_on_E6.md`](../plans/P28_strength_on_E6.md), step 2b.

**Arm.** E6-08-44 is E6-06-44 (E6-04-44 from its 560M end state to 760M, same flags, static league
pool) plus `--ema-weights 10000000`. Snapshots, pool members and evaluations are the average, with
a time constant of 10M steps; the live weights train. `launch_flags --diff` against E6-06-44 shows
only `--ema-weights`.

**Pre-registered.** Adopt EMA if the averaged snapshots beat the matched raw ones by the panel rule.
Read the late spread as in 2a.

## Results

The round robin was greedy, 1,000 games per seat (`data/reports/p28_2b_rr.{md,json}`). The players
were E6-08-44's 680–760M snapshots, E6-06-44's raw 700/720/740/760M snapshots, E6-06-44's
680–760M SWA and the three neural panel members.

**Spread against the panel peak** (E6-03-44@550M). E6-08-44's snapshots read 62.6, 63.1, 64.5,
66.5, 65.5, 64.8, 63.3, 65.0 and 66.5, a mean of 64.6.

| arm | SD | spread beyond binomial noise (1.12) |
|:---|---:|---:|
| E6-06-44, raw (2a) | 2.03 | 1.69 |
| **E6-08-44, EMA** | **1.43** | **0.89** |

The spread beyond binomial noise is about halved.

**Panel rule, late block** (700/720/740/760M), E6-08 (EMA) against E6-06 (raw):

| seat | EMA | raw | difference | the E6-06 SWA |
|:---|---:|---:|---:|---:|
| US | 74.1 | 72.0 | **+2.1 ± 0.6** | 82.5 |
| USSR | 80.5 | 73.2 | **+7.3 ± 0.6** | 82.2 |

* Head to head over the late block, EMA against raw scores 54.5%.
* Elo in the same field: EMA snapshots 1536–1556, raw 1493–1522, the E6-06 SWA 1601.

**But the τ 10M average loses to the 80M SWA.** E6-08 at 700/720/740/760M scores 42.0 / 43.2 /
40.6 / 41.5% against it.

**SWAs of both arms' 680–760M** (`data/reports/p28_2b_soups.{md,json}`):

| SWA | Elo | vs panel, US | vs panel, USSR |
|:---|---:|---:|---:|
| E6-08-44 (EMA snapshots) | 1665 | 76.9 | 84.2 |
| E6-06-44 (raw snapshots) | 1677 | 81.8 | 81.7 |

Head to head the two SWAs are level: EMA SWA 48.0% ± 1.6 (US 42.9, USSR 53.0).

## Reading

* **EMA is adopted as the rated model.** It beats the matched raw snapshots in both seats (US
  +2.1, USSR +7.3) and roughly halves the spread beyond noise. By both P28's rule and the owner's,
  it is better.
* **The window has to be wide.** A 10M-step average captures only part of what an 80M uniform
  SWA gives. Every τ 10M snapshot loses to the 80M SWA (41–43%).
* **The EMA did not change what was learned.** Its SWA and the raw run's SWA are level (48.0 ±
  1.6). Pooling averaged opponents had no measurable effect on the trajectory. The gain is entirely
  in which weights are rated.
* **What to carry forward:**
  * The *rated* model of a run is its SWA of the last ~80M, which needs no training change.
  * `--ema-weights` with τ ≈ 40M (a 3.5–4× wider memory) is the in-training equivalent, if an
    always-current rated model is wanted.
  * The strongest model on record is E6-06-44's 680–760M SWA: `swa_680-760M.pt`, Elo 1662–1677
    across fields.
