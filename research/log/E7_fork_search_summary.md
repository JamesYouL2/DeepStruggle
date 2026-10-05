# Search and card-play experiments on the fork (2026-09-29 to 2026-10-04): summary

Everything the fork ([JamesYouL2/DeepStruggle](https://github.com/JamesYouL2/DeepStruggle)) ran on
search, and on teaching the network card play, with what did and did not work, and three
recommendations at the end (the first since withdrawn, see below). The code behind the E7 runs below is on main with these notes, every
part off by default: the search node filters, `--playout-*`, the `CARD_EFFECTS` block and head,
and whole-placement search. The older play-time search variants (Gumbel root, truncation, round
search) and the evaluation tools (doctrine census, CI tournament workflow) stay on the fork's
branches, cited by branch and commit.

## Runs

Every E7 fine-tune here warm-starts from **E7-04-44@1200M** (`snapshot_1200029696steps.pt`) and
trains 80M more steps. **E7-75-45 is the control**: that warm start with nothing else changed,
seed 45, constant lr 3e-5, opponent pool 0.3. Its flags, printed by `launch_flags.py` from its
`metadata.json`:

```
tools/train.py --arch ladder --block-lambda off --drop-static --ladder-aggregation flatten \
  --no-ladder-card-lookup --ladder-card-lookup-dim 0 --ladder-card-lookup-heads 0 \
  --ladder-card-lookup-identity-dim 0 --ladder-entity-dim 16 --ladder-entity-proj-dim 256 \
  --ladder-head-center --ladder-head-context --ladder-head-entities country --ladder-head-static \
  --ladder-hidden-dim 480 --ladder-input-mode grouped --ladder-res-blocks 0 --per-entity-heads 64 \
  --lr 3e-05 --opponent-frac 0.3 --opponent-self-pool --resume-every-steps 10000000 \
  --seed 45 --seed-env 45 --seed-init 45 --seed-pool 45 --seed-sampling 45 \
  --eval-opponents heuristic random --train-steps 80000000 \
  --warmup-checkpoint <E7-04-44>/snapshot_1200029696steps.pt --run-name E7-75-45
```

Training is deterministic: E7-92-45, the same command on a later commit, reproduced E7-75-45's
weights bit for bit. Each arm adds only its own flags:

| run | added flags | note |
|:---|:---|:---|
| E7-76-45 | `--play-mode-temp 2.0` | [`E7_playout_pg_finetune.md`](E7_playout_pg_finetune.md) |
| E7-85-45 | `--playout-adv 3.0 --playout-mode joint --playout-hidden redeal --playout-pairs 16 --playout-batch 64 --playout-sample-frac 0.008` | same |
| E7-86-45 | E7-85-45's plus `--play-mode-temp 2.0` | same |
| E7-90-45 | `--obs-features card_effects --ladder-card-effects-head 64` | [`E7_card_effects_block.md`](E7_card_effects_block.md) |
| E7-91-45 | `--obs-features card_effects` | same |
| E7-93-45 | `--search-ce-coef 0.5 --search-sims 64 --search-node-filter all --search-subsample 0.125` | [`E7_search_finetune.md`](E7_search_finetune.md) |
| E7-94-45 | `--search-ce-coef 0.5 --search-sims 64 --search-node-filter late --search-subsample 0.5` | same |

Each run's `metadata.json` is in its release; `launch_flags.py <E7-75-45> --diff <arm>` checks an
arm against the control (`--search-subsample 0.125` is the default, so it does not print).

## Measured

| experiment | result | verdict | where |
|:---|:---|:---|:---|
| **Search budget sweep**, E5-11-43@560M, honest search 2-128 sims | nothing below ~16 sims; +28 Elo at 32, +52 to +54 at 64 and 128 | 64 is the knee | fork `workshop`, `research/log/E5_search_budget_sweep.md` |
| ↳ tie-break bug found by the sweep | tied root visits took the first listed move, so a 2-sim search scored below its own prior | fixed, on main (`pr/search-tiebreak`) | — |
| **Gumbel AlphaZero root** (`gumbel_k`, sequential halving), E6-06-44 soup, 64 sims, 512 games | honest Gumbel k=2 vs honest PUCT **+6 ± 15 Elo**; both only +12 to +20 over the raw net | **level with PUCT** at play time | fork `feat/mcts-gumbel` `b249e6c` |
| ↳ same, privileged (true hidden cards) | k=2 +158, k=4 +103, k=8 +95 Elo vs raw net | not comparable: privileged, and no untruncated privileged-PUCT control | runs `36733976167`, `36733990044`, `36734002465` |
| **Truncated search** (cut after N half rounds), privileged, 64 sims | cut 1 / 2 / 4: +17 / +55 / +88 Elo vs raw net | **a shallower cut is worse**; never run honest | fork `feat/mcts-truncation` `9a7c300` |
| **Dense action-round search** (`roundsearch:`) | 12-game smoke only; without a reply stage it ends a turn holding a scoring card | **horizon bug, never measured for strength** | fork `feat/round-search-ci` |
| **What the training searcher's target is**, E6-06-44 soup, 4,000 positions, 64 sims | the target's argmax equals the prior's 98.8% of the time, and never differs once the prior's top move is above 0.8; one world's target is within TV 0.034 of eight worlds' | **the 64-sim visit target is the prior** | fork `exp/hungary-openings`, `research/fork/determinization_targets.md` |
| ↳ determinized worlds knew the opponent's unrevealed headline | every determinized search at a USSR headline resolved the true US headline | fixed, on main (`pr/determinize-headline`) | — |
| **Where search gains on the soup**, 64 sims honest, 8,000 games per row | everything +4.0; Ops influence +1.9; turns 8-10 +2.7; card + play mode +1.4 | influence and the late game | [`E7_search_segments.md`](E7_search_segments.md) |
| **Offline distillation** of bank-confirmed card spots (paired playouts, denoised) | E7-04-44@1200M+denoised-e20: 48.8% vs parent; Star Wars 48%, OPEC 64%, Alliance for Progress 58% | **the only method that moved the spots**, at a small strength cost | fork `exp/hungary-openings` |
| **Paired-playout policy gradient inside RL**, 80M fine-tunes | 47.5% / 46.0% vs control; spots unmoved; entropy −20% | **fails** | [`E7_playout_pg_finetune.md`](E7_playout_pg_finetune.md) |
| **Card-effects observation block** (+ head), 80M fine-tunes | 50.6% / 49.2% vs control; spots unmoved; head used but uncorrelated with event value | **no effect** | [`E7_card_effects_block.md`](E7_card_effects_block.md) |
| **Whole-placement search**, soup, Ops influence only, 64 sims per point, 8,000 games per pairing | vs soup **+2.6** (point-by-point +1.9); vs point-by-point 50.65% (+0.65 ± 0.56) | **better, small**; head to head 1.2 SE | `ai/search/placement_search.py`, CI `37227904004` |
| **Search target forms vs the prior**, 4,000 positions, paired playouts of each departure | visits@64 departures +0.001 ± 0.002 (1,024 pairs); Gumbel's improved policy departs on 15-48% with no gain | **no form carries a measurable gain; Gumbel as published adds noise** | [`E7_search_target_forms.md`](E7_search_target_forms.md) |
| **Search-driven fine-tune**, visit-count CE at 64 sims, 80M | 43.3% (all decisions) / 45.3% (turns 8-10) vs control; spots unmoved | **fails** | [`E7_search_finetune.md`](E7_search_finetune.md) |

## What did not work, and the common thread

* **Every in-RL target trained towards a judge built from the network itself lost or was level**
  at 80M: playouts (−2.5 to −4), search visit counts (−5 to −7), a card-effects input (level). The
  judge is the network's own continuation or its own prior plus 64 simulations, and pulling the
  policy towards it sharpens the policy without improving it.
* **64-simulation PUCT cannot leave a confident prior.** Its target matches the prior's argmax
  98.8% of the time, so visit-count CE mostly teaches the prior back; where the prior is unsure the
  target is noisy. That is the likeliest reason the search fine-tune lost.
* **At play time, Gumbel's root and truncation add nothing over plain honest PUCT** at 64
  simulations; round search was never brought to a measurement.
* **The card leaks are about value, not information.** Handing the network each card's exact
  current effect changed card play but not in the direction of the event's value.

## Recommendations

**1. Gumbel's improved-policy target for C5, instead of visit counts.**

> **Withdrawn (2026-10-05).** The order below was followed -- its first step, the probe, was run
> before any training -- and the target failed it: at 32-64 simulations the completed-Q target
> departs from the prior on 15-48% of positions with no gain (confirmed = refuted), because mctx's
> min-max rescale stretches tiny Q gaps into large logit shifts; and the visit target's departures
> are no better than the prior either. See [`E7_search_target_forms.md`](E7_search_target_forms.md).
> The original recommendation is kept as written.

Gumbel MuZero (Danihelka et
al., ICLR 2022) trains the policy towards softmax(logits + σ(completed Q)): the prior moved by the
searched value of each action, with unvisited actions filled in from the value estimate. It is built
for 16-64 simulations, comes with a policy-improvement property, and moves the target
toward a better action even when that action's prior is low -- exactly where 64-sim PUCT's visit
target cannot go (above). It changes how the target is computed, not the search, and the root
(Gumbel-top-k plus sequential halving) already exists on the fork (`feat/mcts-gumbel`, `b249e6c`,
with tests). Order:

* first, re-run the determinization-targets probe with the completed-Q target: does it depart from
  the prior on positions where the PUCT target does not, and do those departures check out on
  paired playouts?
* only then, a fine-tune arm exactly like E7-93-45 with the new target, against E7-75-45 (the
  control is deterministic, so the comparison is exact), tracking agreement on fresh positions and
  entropy, which is where the playout arms showed their failure first.

**2. C2 with the C++ labeller, labelled densely.** C2 (`--aux-card-events`) is the one P30 arm with
a positive signal (US +1.5 on snapshots and +3.2 on the SWA, USSR level,
[`P30_branch_arms_c2_play_mode_temp.md`](P30_branch_arms_c2_play_mode_temp.md)), and its head is
shown to reach the trunk ([`P30_c2_card_probe_and_branch_soups.md`](P30_c2_card_probe_and_branch_soups.md)).
It labels only 0.05% of decisions because the Python labeller costs ~2-6 ms a position. The fork
has an exact C++ port (`engine/src/card_effects.cpp`, in this change as `card_effects_label`; equal to
`card_event_targets.label` on the same states and dice, `tests/bindings/test_card_effects.py`) at
0.27 ms, about 20× faster. Used **only as C2's label source** -- no observation block, no change to
the game -- it lets C2 label 10-20× more decisions at the same cost. Run as C2's seed-43 replicate
with the denser labels; the remaining work is pointing `--aux-card-events`' labelling at it.

**3. Search an influence placement as one decision: all its points at once.** Ops influence
placement is the segment where search gains most (+1.9 of +4.0,
[`E7_search_segments.md`](E7_search_segments.md)), and the search spends its budget there worst.
Each point is a separate tree level, so a 4-Ops placement is 4 levels deep before the opponent
moves, and `BatchedMCTS` has no transposition table: placing Poland then Hungary and Hungary then
Poland are different paths to the same board, each searched separately. At 64 simulations most of
the budget goes on the order of the points rather than on which placement to make.

Instead, at the first point of a placement: sample k complete placements from the network's own
point-by-point policy (keeping its argmax placement, deduplicated as multisets of countries), apply
each whole, and split the simulations among them by sequential halving, as the Gumbel root does
for single moves. The chosen placement is then played point by point, so the network, its action
space and the training data are unchanged. This is not the merged-influence view (P29 bet 3,
[`P29_bet3_merged_view.md`](P29_bet3_merged_view.md)), which changed what the network is trained
on; here only the search is changed.

**Measured** (`search:<soup>:64:determinize:ops_influence:placement=8`, CI `37227904004`, 4,000
games per side per pairing): against the soup **52.6%** where point-by-point search scores 51.9%
(+2.6 against +1.9; the point-by-point pairing reproduces the segment tournament's +1.86), and
50.65% ± 0.56 head to head against point-by-point -- better in both seats against the soup, 1.2
standard errors head to head. A small, plausible gain at equal budget; with 8 candidates sharing
64 simulations per point, each gets few, so the budget is the next thing to vary.
