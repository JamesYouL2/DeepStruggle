# Side collapse: a reproducible, escapable learning-arrest

**Conclusion (E4, P21 rung M2d: grouped input, country head only, pooled 0.3/12, cold start).** One
seat — always the **US** in every episode logged — stops winning: `us_episode_frac` falls to ~0 and
stays there. It is an **arrest, not a degradation** (the arm stops improving; before onset it is
level with clean seeds). About **17% of seeds enter** it by 80M (5 of 30), and entry is **sticky but
not absorbing**: of seven arms scored COLLAPSED and continued 80M further, **five recovered**. A
recovered collapse costs **compute, not strength** (−1.4 Elo, 0.05 pooled SD, against never-collapsed
seeds at 160M); the one that never recovered (seed 14) cost **~−394** at 240M
([`../../log/E4_collapse_is_recoverable.md`](../../log/E4_collapse_is_recoverable.md)). Recurrent
collapses in the late-dynamics legs were largely a drained opponent pool: on a fixed pool the seed-5
continuation did not collapse (3 of 3; [`../../log/P25_pool_resume_bug.md`](../../log/P25_pool_resume_bug.md)).

## What is established

* **Bitwise reproducible.** Three 80M runs of seed 1, hours apart, gave byte-identical checkpoints
  and metrics with every cuDNN determinism flag off: this model (dense `Linear`, GELU, LayerNorm,
  `allow_tf32=False`) has no non-deterministic kernel. A property of the configuration, to be
  re-verified for anything with attention.
* **Detection of entry, not outcome.** Minimum `adv_std_raw` 0.0036–0.0313 in arms that entered
  against 0.087–0.198 clean, and maximum `explained_variance` 0.999–1.000 against 0.974–0.991 —
  no overlap over 30 seeds. The critic looks *excellent* exactly when the run is failing
  ([`../../method/detecting_collapse.md`](../../method/detecting_collapse.md)). Neither predicts
  escape (seed 21 sat at 0.0226 and escaped). The mechanism transfers between lineages; the
  threshold does not (`adv_std_raw < 0.01` had 0 hits in 47 E3 runs).
* **Onset does not predict persistence** (35–72M; the earliest one escaped), and five candidate
  predictors — depth, duration, exact-zero run length and count, onset timing — were falsified.
* **Per seat** ([`../../log/E4_collapse_census_per_seat.md`](../../log/E4_collapse_census_per_seat.md)):
  both seats show equal advantage spreads at the first logged collapse; only the losing seat's
  entropy rises. Loosening the update late brings one on; a slow π_ref delays it.
* **Cause: no single seed stream is sufficient** — eight arms moving initialisation, sampling,
  deals or the opponent draw one at a time were all clean
  ([`../../log/E4_collapse_attribution.md`](../../log/E4_collapse_attribution.md)); necessity is
  untested (an all-clean result has p 0.34 at the base rate).
* **Not the only failure mode:** [`entropy_inflation.md`](entropy_inflation.md) lost ~330 Elo with
  every indicator here healthy.

## Withdrawn claims

| claim | why it was wrong |
|:---|:---|
| 4 of 30 terminal collapses (13.3%), CI [5.3, 29.7]% | censored: all four recover past 80M; it counted arms *inside an episode* at the cutoff |
| collapse costs ~300 Elo (≈1697 against 1993 clean) | ratings of mid-episode snapshots; at 160M the same arms are level |
| an absorbing state, a clean bifurcation | seed 21 was pinned 45 rows and climbed out to 0.343 |
| "the country head alone buys nothing", "the heads are superadditive" | read off a collapsed arm's Elo |
| "M2d reaches the anchor" | a four-seed mean at the luckiest end of all 495 draws |
| a partial per-entity correction distorts the card-vs-country trade-off | the mask never offers both blocks (0.00% of 25,600 decisions) |

The detector itself was wrong twice (one-sided; pinned fraction instead of terminal state).
