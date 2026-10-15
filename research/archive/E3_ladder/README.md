# The E3 ladder — archived 2026-09-19

Everything here was measured on the **pre-P17 engine**. P17 repacked the action space 212 → 220,
flattened Grain Sales to one decision and fixed Missile Envy's starred-card removal, so **no Elo
number, win rate or blunder rate below is comparable to anything measured after it**. The ladder is
reset; E4 re-measures from scratch.

Kept because the *lessons* outlive the numbers. Read this page; open the originals only to check a
specific claim.

| | |
|:---|:---|
| checkpoints, replays, datasets, reports | `/workspace/data/archive/E3_ladder/` (45G + 2.3G + 137M) |
| research logs | `log/` — 20 documents |
| plans | `plans/` — P1, P9, P10, P14, P15, P17 (×3); the rest deleted, in git history |
| training findings | `findings/` — 7 documents |

## What we learned that still holds

These are about method and configuration, not about any particular policy, so they survive the
engine change.

**An RL run needs an opponent pool.** Without one the policy trains against its own current self,
one seat runs away, the games shorten and the critic loses all skill. Seen twice from two unrelated
causes — a `dirname` path bug that shrank a pool to one (`log/P15_X4b_collapse_is_pool_starvation.md`),
and a launch that never enabled a pool at all (`../../log/E4_pool_starvation_recurrence.md`). Shared
fingerprint: `mean_turn` ≈ 4, ~60% of wins ending in DEFCON 1, the policy beating its available
opponent ~99%. Now `CLAUDE.md` invariant 14.

**Per-seat rating against a frozen anchor is the only measure that separates "one side collapsed"
from "both sides improved unevenly."** Self-play side split and `critic_base_rate` cannot, and
`P15_control_per_seat.md` records the answer they gave coming out **backwards**.

**No live training metric rates an arm after ~120M** (`log/P15_X0_frozen_anchors.md`). Frozen
anchors and a round robin are needed; the in-run instruments saturate.

**Throughput figures are close to meaningless without the GPU and the concurrency**
(`findings/throughput.md`). Two runs sharing a card depress each other — an E3-vs-E4 comparison had
to discard `E3-30-28` for exactly this, because it overlapped `E3-31-28`.

**One network can hold both seats** (`log/P15_two_seat_capacity.md`) — so a seat collapse is a
training pathology, not a capacity limit.

**Sample at the policy's own temperature, not sharper** (`log/P15_rollout_temperature.md`).

**The perspective negation in GAE is load-bearing** (`findings/value_bootstrap_perspective.md`):
`compute_gae` bootstraps across a change of mover by negating the next value.

**Search transfers, then RL erodes it.** One offline round of expert iteration bought +47.7 Elo,
and it did not survive RL (`log/P15_X4a_distillation.md`, `log/P15_X4b_search_during_rl.md`). Most
of the signal was *not* at card/play-mode nodes, which averaged 4.1 legal actions with the policy
already agreeing — scoping search to them was wrong
(`findings/which_decisions_to_search.md`).

**The KL term can dominate the policy gradient** — 300x on half of `E3-31-28`'s iterations
(`log/P15_kl_domination.md`).

**Opening influence placement froze for 155M steps** and was only unfrozen by a resume
(`log/P15_setup_placement.md`) — worth re-checking early in any new ladder.

## Open questions carried into E4, and where they went

1. **Does the oscillate-then-stall cycle survive a healthy pool?** Replaced rather than answered:
   with a pool, runs climb to a plateau, and side collapses are recoverable delays
   ([`../../log/E4_collapse_is_recoverable.md`](../../log/E4_collapse_is_recoverable.md)).
2. **Does a slower π_ref refresh help?** From scratch it loses; switched on late it helped on
   drained-pool legs only ([`../../findings/training/ref_update_freq_open_ablation.md`](../../findings/training/ref_update_freq_open_ablation.md)).
3. **Where is the search signal?** Spread over ~7% of decisions rather than a tail
   ([`../../log/E7_blunder_census.md`](../../log/E7_blunder_census.md)); search as a teacher is
   P32/P33 ([`../../plans/README.md`](../../plans/README.md)).
4. **How many seeds does an arm need?** [`../../log/variance_and_noise.md`](../../log/variance_and_noise.md).
5. **Does the DEFCON-1 blunder persist?** Engine-proven blunders are rare on the E7 heads soup
   ([`../../log/E7_blunder_census.md`](../../log/E7_blunder_census.md)).
6. **Is pooled better than non-pooled?** A pool is required: `CLAUDE.md` invariant 14.

## One measurement caveat that outlived its document

`ai/eval/blunders.py` carried a pre-P17 play-mode map until 2026-09-19, so **every blunder rate
logged by an E4 run before that fix is wrong** — Influence plays were scored as Space Race plays.
E3's own numbers are fine; that map was correct for the engine E3 ran on. Re-measure from snapshots
rather than trusting a training log.
