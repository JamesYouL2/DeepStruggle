# Entropy inflation — a second failure mode, invisible to the collapse detector

**Conclusion (E4, one arm).** `E4-03-01@160M`, the P21 anchor continued 80 → 160M, lost **~330 Elo**
(2093 at 80M; −296 in the single 10M interval to 105M, trough −551 at 125M, −331 at 160M) while every
side-collapse indicator stayed healthy: `adv_std_raw` 0.217–0.262 (collapsed arms 0.004–0.031),
`us_episode_frac` 0.30, `collapse_check.py` correctly **CLEAN**. The policy did not turn one-sided; it
turned **indecisive** — Elo mirrors policy entropy inversely across the leg (1.27 → 1.96 → 1.72).
Companion to [`side_collapse.md`](side_collapse.md).

* **Not a configuration error:** `launch_flags --diff` shows only `--train-steps`, the pool restored
  (11 of 12), and no schedule keys off `train_steps`.
* **Signature across four arms:** sustained entropy inflation marks both failures (this anchor, and
  the side-collapsed `E4-08-14@160M`, 1.35 → 1.87), while healthy arms rise transiently and return —
  so entropy may be the more general indicator.
* **Mechanism, unverified:** `kl_div` falls (0.043 → 0.027) while entropy rises — the policy tracks a
  reference that drifts flatter, a ratchet through `π_ref` refreshes.
* **Not established:** n = 1, no replicate (`seed = None`).

**Consequence for P21:** M2d at 160M rates +110 over the anchor's *best* state (2203 against 2093 at
80M), not the +466 a 160M-vs-160M comparison shows
([`../../log/P21_M2d_160M_slope.md`](../../log/P21_M2d_160M_slope.md)). A reference that loses 330 Elo
when trained further is not fixed; rungs compared at 80M are unaffected. Data:
`data/reports/P21_anchor_decline_trajectory.json`.
