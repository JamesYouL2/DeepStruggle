# Training length, and how a recipe is measured

E7, 2026-09-30 – 10-14. The owner's framing: **base-model quality is the goal**; a recipe is judged
at its plateau, and SWA and soups are ways to read it with less noise, not part of training.

## Runs keep climbing long after they look saturated

* **The first "saturation" was premature.** E7-02-44's trace rose ~+100 Elo from 560M to 880M and
  +27 to 1,200M ([`../../log/E7_02_44_saturation.md`](../../log/E7_02_44_saturation.md)), but the
  same seed-44 line then gained +54 to 2,000M, met the pre-registered plateau rule at 2,800M, and
  rose again from ~2,960M (the league's control leg, bit-identical to the league until its first
  publish) to a real plateau at **~3,700M**
  ([`../../log/E7_line_to_2000M.md`](../../log/E7_line_to_2000M.md),
  [`../../log/E7_league_from_2800M.md`](../../log/E7_league_from_2800M.md)).
* **Later lines plateau later still** (rule: the mean of the last two 400M windows against fixed
  references within +1.5 points of the two before): the A8 heads run at **6,800M**
  ([`../../log/P31_branch_arms_B1_B2.md`](../../log/P31_branch_arms_B1_B2.md)), R48 at **9,200M**,
  A13+R48 at **11,200M** ([`../../runs.md`](../../runs.md)).
* **The deep (M2d) trunk plateaus at ~560M** and ends ~120 Elo under the shallow one
  ([`../../log/E7_06_44_deep_to_saturation.md`](../../log/E7_06_44_deep_to_saturation.md)).
* **Consequence.** Every arm compared at 560M (P28 onward) was compared on the first part of the
  curve. A late branch from a plateau (400M from 6,400M) is the cheap way to compare recipes now.

## SWA and soups

* **SWA** (uniform average of a run's last 80M): +50 to +110 Elo over snapshots on the early lines;
  at the A8 plateau the SWA beats its raw 6,800M snapshot 53.8% (US +8.2, USSR +0.6). Part of that
  gain is optimiser noise: a learning-rate step-down (R46) or a 4× minibatch (R47) puts it into the
  raw weights (raw against raw 56.9% / 56.0%), a one-time +15–25 Elo, not a higher ceiling.
* **Model soup** (average of SWAs of branches from one trained state, never across independent
  seeds): +42 over the best SWA on the deep E6 line; on the leaderboard the A8 heads soup is +45 over
  its SWA (1625 against 1580) and the R34 soup +21 over R34's SWA.

## Other measured facts

* **E7 does not change strength** — E7-01-44 is level with E6-12-44 on E7
  ([`../engine/engine_revisions.md`](../engine/engine_revisions.md)).
* **The league adds nothing measurable** on the shallow trunk: +10 Elo on snapshots, 0 on SWAs
  (E6-13-44), and level with its control from 2,800M (E7-21-44); exploiters rarely reach the publish
  bar from a plateau main.
* **Throughput:** two shallow arms at once give +20% total steps/s (110k against 91.7k), three add
  nothing; the host drifts ~±7% over a day, so compare speeds only between runs made together.
