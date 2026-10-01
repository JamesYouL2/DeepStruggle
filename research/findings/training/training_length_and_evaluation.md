# Training length, and how a recipe is measured

Measured 2026-09-30 – 10-01 on E7 (training findings).

## Recipes are judged at 560M before they saturate

The shallow recipe (E7-01-44 continued as E7-02-44), traced in one field every 80M:

| steps | 400M | 560M | 720M | 880M | 1,040M | 1,200M |
|:---|---:|---:|---:|---:|---:|---:|
| Elo | 1451 | 1505 | 1565 | 1603 | 1616 | 1630 |

* About +100 Elo from 560M to 880M, then +27 to 1,200M. **It saturates at about 900M–1B steps,**
  roughly 3 h solo.
* The 1,120–1,200M SWA beats the 480–560M SWA 64.4% (+110 Elo), in both seats.
* **The deep recipe saturates earlier, at about 560M** (E7-06-44, plain M2d to 1,200M: its snapshots
  stay at about 1,500–1,550 from 560M on). The league line gained only about +20 Elo per 200M
  from 560M to 760M.
* **Consequence.** Every arm since P28 was compared at 560M, before saturation. An ordering at 560M
  is a statement about the first 60% of the curve, not about the base model a recipe reaches.
  Logs: [`../../log/E7_02_44_saturation.md`](../../log/E7_02_44_saturation.md).

## SWA and the model soup are measuring devices, not the product

The owner's framing (2026-10-01): **base-model quality is the goal.** SWA and model soups are
ways to read a recipe with less noise; they do not enter training.

* **SWA** (uniform average of one run's last 80M): +50 to +110 Elo over the run's snapshots, more
  on the deep trunk.
* **Model soup** (average of branches of one trained state; never across seeds, which breaks the
  net): +42 Elo over the best SWA on the deep line, +50 on the shallow line.
* The shallow soup (E7-02/03/04/05-44 at 1,200M, 2613) is the strongest model measured
  ([`../../log/E7_shallow_soup.md`](../../log/E7_shallow_soup.md)). Its best single ingredient was
  the learning-rate step-down branch (E7-04-44), which is a training-recipe signal.

## Other measured facts

* **E7 does not change strength.** E7-01-44 (E6-12-44's recipe on E7) is level with E6-12-44 on
  E7: 47.1% head to head on snapshots, 49.8% on SWAs
  ([`../engine/engine_revisions.md`](../engine/engine_revisions.md)).
* **The league adds little on the shallow trunk.** About +10 Elo on snapshots (USSR +2.3, US +1.3)
  and nothing on the SWA (E6-13-44 against E6-12-44, one seed).
* **Throughput.**
  * Two shallow arms at once give +20% total steps/s (110k against 91.7k alone); three add
    nothing.
  * The host drifts by about ±7% over a day, so compare speeds only between runs made together.
