# Detecting a collapse from internal metrics

Win rate against a frozen opponent is ground truth but arrives only per snapshot. What do the
training loop's own metrics say in between? **Living reference.**

## Scope rule

A **lineage** is an engine revision + action space + architecture + observation. What one lineage
established is **which instrument sees which failure mode — that transfers. No number does.**
Fitted on E4's collapse and checked on E3's: the entropy gap (standardised difference 5.6 on E4)
was noise on E3, and `clip_frac` and `adv_std_raw` reversed direction. So every trigger below is
relative to the run's own baseline.

## What transfers

1. **There is no single detector, because there is no single collapse.** Each mode is invisible to
   the other's instrument.
2. **Pool starvation is structural:** is `opp_pool_size` logged at all, and does it grow? A pool
   that never passes one opponent *is* the failure. **Being beaten is not starvation**: E3-37-31, a
   healthy arm, beat its pool > 90% for 947 of 1,229 iterations while the pool grew to 8.
3. **KL domination** shows as `kl_div` orders of magnitude above the run's own median; pool
   instruments are blind to it. Both arms that showed it used search CE (E3); untested since.
4. **The two are mutually blind** (`E3-19-23` one-sided with normal KL; `E3-31-28` KL blown up,
   balance never flagged).
5. **Critic quality never separates collapsed from healthy** — a degenerate policy is an easy
   prediction problem (on E4 `explained_variance` → 1.0 marks *entry* into a side collapse,
   [`../findings/training/side_collapse.md`](../findings/training/side_collapse.md)).
6. **`critic_base_rate` (`max(p, 1 − p)`) carries no direction** and fired wrongly three times on
   E4-02-01.
7. **Side balance is description, not an alarm:** over trailing windows of 100–400 iterations the
   pinned fraction overlaps between the best run in the record (0.50–0.72) and a heavily one-sided
   one (0.54–0.87).
8. **Dynamics metrics** (`entropy`, `clip_frac`, `adv_std_raw`, `logratio_max`) say what kind of
   wrong once something has fired; no threshold on them survived a second dataset. Sustained
   entropy inflation marked both E4 failures ([`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md))
   but is four arms, not a detector.
9. **The arbiter is per-seat win rate against a frozen reference.**
10. **Validate a candidate signature out of sample, in the same lineage**, before acting on it.

## The protocol (`tools/scripts/watch_run.py`, `health_alarms`)

| | condition |
|:---|:---|
| **alarm: no pool** | `opp_pool_size` absent from the metrics |
| **alarm: pool stuck** | `opp_pool_size` never above 1 over ≥ 100 iterations |
| **alarm: KL spike** | `kl_div` ≥ 10× the run's own median (with an absolute floor) |
| description only | `us_episode_frac` (rows with `episodes_completed > 0` only), `entropy`, `clip_frac`, `adv_std_raw`, `opp_win_rate_mean`, `logratio_max` |
| never | critic quality, `critic_base_rate`, side balance as a trigger, `steps_per_sec`, win rate against HeuristicBot |
| confirm before acting | per-seat win rate against a frozen reference |

Replayed over the 54 runs then on record: 23 fired, every alarm on a run already known to be bad
(20 unpooled arms, E3-24-28's stuck pool, both KL-domination collapses at 56× and 1070×), no false
positives. Two triggers were dropped for firing on healthy arms: `opp_win_rate_mean > 0.9` and any
one-sidedness threshold.

**Measurement trap:** E3-24-28 logs `episodes_completed = 0.0` on every row, and a sweep computing
`won_us / max(1, episodes_completed)` turned missing data into "100% one-sided". Require
`episodes_completed > 0` and report excluded rows.

## Provenance

Pool starvation: [`../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`](../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md)
and its recurrence on E4 ([`../log/E4_pool_starvation_recurrence.md`](../log/E4_pool_starvation_recurrence.md)).
KL domination: [`../archive/E3_ladder/log/P15_kl_domination.md`](../archive/E3_ladder/log/P15_kl_domination.md).
Why `critic_base_rate` was dropped: [`../archive/E3_ladder/log/P15_control_per_seat.md`](../archive/E3_ladder/log/P15_control_per_seat.md).
E4 reference: the cold-start pooled `E4-04-01` (80M) had entropy 0.76–1.92, max `kl_div` 0.095, its
pool at capacity 12, `us_episode_frac` mean 0.443 with 0% pinned; `kl_div` stayed ≤ 0.12 on every E4
leg, so E4 never showed KL domination.
