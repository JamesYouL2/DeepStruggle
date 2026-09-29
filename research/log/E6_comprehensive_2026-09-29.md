# The comprehensive E6 tournament (2026-09-29): every arm on its snapshots and on its 80M soup

The owner's measurement rule: every arm is measured twice, on its snapshots and on the uniform soup of
its last 80M ([`../plans/P28_strength_on_E6.md`](../plans/P28_strength_on_E6.md), *Measurement rule*).
This tournament puts every E6 arm into one field that way, after E6-03-43 (the control recipe on
seed 43) finished.

**The field** (`data/reports/e6_comprehensive.{md,json}`), 34 models, greedy, 500 games per seat per
pairing, anchored at HeuristicBot = 1500:

* each arm's four late snapshots: 500/520/540/560M for the from-scratch arms, 700/720/740/760M for
  the 560→760M continuations;
* each arm's soup: 480–560M or 680–760M;
* the three neural panel members (E6-03-44 at 80/240/550M);
* E5-21-43@560M, the E5 best model, played on E6;
* HeuristicBot.

"vs panel" is the mean over the three panel members, per seat.

| arm | what it is | snapshots: mean Elo | snapshots vs panel, US / USSR | **soup Elo** | soup vs panel, US / USSR |
|:---|:---|---:|:---|---:|:---|
| **E6-06-44** | E6-04-44 continued to 760M, constant rate | 2467 | 72.3 / 73.0 | **2560** | 82.4 / 81.0 |
| **E6-08-44** | the same with EMA (τ 10M) snapshots | **2508** | 74.6 / 80.7 | **2555** | 79.4 / 83.1 |
| E6-04-44 | league + setup credit from 310M (adopted) | 2434 | 68.3 / 73.0 | 2541 | 80.7 / 81.4 |
| E6-07-44 | E6-06-44 with the rate stepped to 3e-5 | 2481 | 72.8 / 76.2 | 2504 | 76.7 / 78.7 |
| E6-03-43 | control recipe, seed 43 | 2393 | 69.1 / 62.9 | 2482 | 79.6 / 72.9 |
| E6-03-44 | control recipe, seed 44 | 2397 | 65.4 / 72.0 | 2476 | 74.8 / 79.7 |
| *E5-21-43@560M* | *the E5 best model, on E6* | *2441* | *66.9 / 72.5* | — | — |

Panel members: 2125 / 2278 / 2406 Elo.

**Soup against soup**, overall %, row against column:

| | E6-06 | E6-08 | E6-04 | E6-03-43 | E6-03-44 | E6-07 |
|:---|---:|---:|---:|---:|---:|---:|
| E6-06-44 | — | 49.4 | 53.1 | 63.7 | 61.2 | 57.6 |
| E6-08-44 | 50.6 | — | 51.9 | 60.4 | 61.3 | 56.9 |
| E6-04-44 | 46.9 | 48.1 | — | 58.7 | 58.0 | 58.1 |
| E6-03-43 | 36.3 | 39.6 | 41.3 | — | 54.2 | 45.3 |
| E6-03-44 | 38.8 | 38.7 | 42.0 | 45.8 | — | 47.6 |
| E6-07-44 | 42.4 | 43.1 | 41.9 | 54.7 | 52.4 | — |

**Against E5-21-43@560M:**

| soup | overall |
|:---|---:|
| E6-06-44 | 69.4% |
| E6-08-44 | 69.1% |
| E6-04-44 | 64.8% |
| E6-07-44 | 58.6% |
| E6-03-44 | 57.7% |
| E6-03-43 | 52.8% |

## Reading

* **The strongest models are the soups of the 560→760M continuations of E6-04-44.**
  * E6-06-44's soup (2560) and E6-08-44's (2555) are level head to head: 49.4 / 50.6.
  * Each beats the E5 best model 69%, and the adopted E6-04-44 soup 52–53%.
* **Soups beat snapshots on every arm,** by 23–110 Elo. The soup gap is largest where snapshots are
  noisiest (E6-04-44: +107; E6-03-43: +89), and smallest where they are already smooth (E6-07-44,
  trained at 3e-5 from 680M: +23; E6-08-44, EMA snapshots: +47).
* **The league + credit recipe (E6-04) beats the plain recipe** on snapshots (+37 Elo over
  E6-03-44), on soups (58.0% soup against soup), and against the seed-43 plain run too (58.7%).
* **Seed 43's plain run is level with seed 44's**: snapshots 2393 against 2397, soups 2482 against
  2476, head to head 54.2%. Its seat split against the panel is different, US 69.1 / USSR 62.9
  against 65.4 / 72.0. The panel is seed 44's own lineage, so part of that split may be a
  lineage-style effect rather than a seed-43 weakness. E6-03-43's own panel members should join
  the panel.
* **Training past 560M still pays when measured on soups.**
  * E6-06-44's 680–760M soup beats E6-04-44's 480–560M soup 53.1%.
  * The raw snapshots would not show it: E6-06's snapshots are within about 30 Elo of E6-04's.
