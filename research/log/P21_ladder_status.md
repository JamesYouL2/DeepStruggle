# P21 architecture ladder — status at 2026-09-21

**Conclusion.** On E4 the late-E3 bundle's +438 Elo over a flat MLP is mostly one mechanism: the
per-entity **country** head (+282 on top of grouped projections, +106); the card head is worth
nothing. M2d (country head only) reaches 99.5% of full M2 at 1.4x its throughput and was still
climbing steeply at 80M (+160 to 160M). The M2 family closed with nothing beating M2d; M3–M5 were
never run (blocked on P22, then superseded by P29/P30).

Everything below comes from **one tournament**,
`/workspace/data/reports/P21_M2d_160M_slope.json` (22 entrants, 100 games per side,
`temperature 0.0`). Bradley-Terry ratings are field-relative, so these numbers may be compared
with each other and with nothing else.

Sources: [`P21_M0_flat_mlp.md`](P21_M0_flat_mlp.md), [`P21_M1_grouped.md`](P21_M1_grouped.md),
[`P21_M2_lookup.md`](P21_M2_lookup.md),
[`P21_M2d_country_head_collapse.md`](P21_M2d_country_head_collapse.md),
[`P21_M2d_160M_slope.md`](P21_M2d_160M_slope.md),
[`E4_collapse_attribution.md`](E4_collapse_attribution.md),
[`../findings/training/side_collapse.md`](../findings/training/side_collapse.md).

## Where the ladder stands

| rung | what it adds | Elo @80M | steps/s | Elo/GPU-h |
|:---|:---|---:|---:|---:|
| **M0** flat MLP | nothing — the floor | 1651.7 | 63,187 | 431 |
| E4-04-01 | the *default* architecture | 1689.7 | 14,057 | 120 |
| **M2e** card head only | `pe_card`, no `pe_country` | 1746.3 | 46,313 | 513 |
| **M1** grouped projections | board/card/global each get their own dense projection | 1758.1 | 59,561 | 692 |
| **M2d** country head only | `pe_country`, no `pe_card` | **2040.3** (6 seeds) | ~50,800 | **1266** |
| **M2** both heads | `pe_country` + `pe_card` | 2050.2 | 35,744 | 885 |
| **anchor** E4-03-01@80M | the late-E3 bundle | 2089.7 | 11,732 | 311 |
| — HeuristicBot | | 1500.0 | — | — |

**The decomposition is now clean.** Against the 438 Elo from M0 to the anchor:

* grouped projections (M1) are worth **+106**
* the **country** per-entity head is worth **+282** on top of M1 — the single largest mechanism
* the **card** per-entity head alone is worth **−12**, i.e. nothing

M2d reaches **99.5% of full M2** at **1.4× its throughput**, and closes **89%** of the
MLP→anchor gap with no attention, no identity vectors, no pooling and no graph convolution.

## Settled

**1. Structure is worth far less than the late-E3 bundle implied.** The E3-era claim was ~110 Elo
for "architecture vs MLP". On this engine the *default* architecture is worth **+38** over a plain
MLP, while the late-E3 bundle is worth **+438** — and most of that +438 is one mechanism, the
country lookup.

**2. The card head buys nothing, and why is NOT established.** `forward_pass_trace.md` predicted
`pe_card` would be near-inert and that `pe_country` would carry the result. The prediction held —
but the reason it gave (that `pe_card` cannot distinguish 95 of 110 cards without an identity
vector) does not survive inspection, because only **4 of a card's 14 slots are static**. Ops, era,
one-time and is-scoring are constant; the other ten — location, playability, and the side slot
that flips with perspective — move constantly. A card is not a static object in this observation.

An earlier version of this section explained the −12 by saying `pe_card` addresses card *types*
and that type-level information is most of what card timing needs. That is an argument for why the
head *would* work, carried over from when seed 1's collapse made `pe_card` look load-bearing, and
it is incoherent as an explanation for a null result. It is withdrawn. The measurement stands; the
mechanism is open.

**3. The rung is still improving steeply at 80M: +159.8 Elo from 80M to 160M**, within-seed over
six seeds, sd 33.7. An 80M measurement understates M2d.

**4. Per GPU-hour, M2d at 160M dominates the anchor at 80M** — +56 to +138 Elo for 46–52% of the
wallclock, because the architecture is 4.3× cheaper per step. At matched *steps and 80M* the
anchor is still 54 Elo ahead. At matched steps and 160M the anchor is far behind, but only
because it fell (see Open, first item) — that margin is not M2d's to claim.

**5. The side collapse is real and not attributable to any single seed stream — but it is
recoverable, and mostly free.** Entry reaches ~12.5% of seeds by 80M. It is **not** an outcome:
5 of 7 arms scored COLLAPSED recover when continued 80M further, and a collapse that recovers
costs **−1.4 Elo** against arms that never collapsed. One that does not recover costs **−394**.
The "costs ~250 Elo and lands below M1" reading is withdrawn — it was measured on checkpoints
taken mid-episode
([`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md)). Splitting `--seed` into initialisation / sampling / deals / opponent-draw
and moving one at a time, in both directions, left all eight arms clean: **no single stream is
sufficient**, initialisation refuted in both directions. Entering the pinned state and escaping
costs nothing detectable.

**6. There is a second failure mode, and nothing here detects it.** The anchor's 330 Elo fall came
with `adv_std_raw` between 0.217 and 0.262 and `us_episode_frac` at 0.301 — every collapse
indicator healthy and correctly so. The policy became *indecisive*, not one-sided, and Elo tracked
policy entropy inversely across the whole leg. The side-collapsed M2d arm shows the same sustained
entropy inflation, so entropy may be the more general indicator.
[`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md).

## Open at the time, and what became of it

* **The anchor at 160M.** Measured: `E4-03-01@160M` lost ~330 Elo over its second 80M
  ([`../findings/training/entropy_inflation.md`](../findings/training/entropy_inflation.md)), so
  its +465.8 matched-steps gap to M2d is the anchor falling, not M2d pulling ahead. The fair
  statement is M2d@160M (2203.1) against the anchor's best (2093.3 @80M): **+110 Elo at twice the
  steps and 46% of the wallclock.** That leg was later found to be a resume on the drained pool
  ([`E4_long_runs.md`](E4_long_runs.md), last section), so its cause is confounded.
* **Whether collapse is M2d-specific.** No: M2.5b enters on 3 of 3 seeds
  ([`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md)).
* **Why escape happens.** Branching one collapse under new seeds: entry is not fixed by the 5M
  state, and a new seed does not stop escape ([`E4_collapse_is_recoverable.md`](E4_collapse_is_recoverable.md)).
* **Whether `ref_update_freq` drives entropy inflation.** A slower π_ref *raised* entropy from
  scratch, which is evidence against the ratchet ([`E4_dynamics_ref_lambda.md`](E4_dynamics_ref_lambda.md)).
* **Side balance.** Every ladder arm leaned USSR (+2.5 to +16.9 pp) except the anchor (−2.7); no
  ladder rung targeted it.

## What the ladder did next

The remaining M2 rungs (M2a/b/c, M2.5, M2.5b, M2.5c) all lost to M2d on both seeds at 80M
([`P21_M2abc_head_inputs.md`](P21_M2abc_head_inputs.md), [`P21_identity_rungs.md`](P21_identity_rungs.md)).
M3 (card↔card attention), M4 (card→country cross-attention) and M5 (flatten → pooling) were blocked
on P22's `board_mode`/`card_mode` split and never run; the plan is
[`../plans/P21_architecture_ladder.md`](../plans/P21_architecture_ladder.md).

Two protocol lessons from the M2d sweep carried forward: run the collapse detector on every arm and
never read a collapsed arm as a strength measurement (seed 1's collapse produced the wrong
conclusion "the country head buys nothing" by ~300 Elo), and never compare Elo across tournaments.
Seed 13 is M2d's representative checkpoint (rank 3 of 6 at both 80M and 160M); the rest of the
sweep moved to `data/checkpoints/archive/P21_M2d/`.
