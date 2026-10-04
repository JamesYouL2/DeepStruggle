# Search and card-play experiments on the fork (2026-09-29 to 2026-10-04): summary

Everything the fork ([JamesYouL2/DeepStruggle](https://github.com/JamesYouL2/DeepStruggle)) ran on
search, and on teaching the network card play, with what did and did not work. The code stays on
the fork's branches (cited by branch and commit); what is worth having on main is the two
recommendations at the end.

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

**1. Gumbel's improved-policy target for C5, instead of visit counts.** Gumbel MuZero (Danihelka et
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
has an exact C++ port (`engine/src/card_effects.cpp`, `feat/card-effects-obs` `8f1e8c5`; equal to
`card_event_targets.label` on the same states and dice, `tests/bindings/test_card_effects.py`) at
0.27 ms, about 20× faster. Used **only as C2's label source** -- no observation block, no change to
the game -- it lets C2 label 10-20× more decisions at the same cost. Run as C2's seed-43 replicate
with the denser labels. It is an engine addition, so it needs the owner's approval.
