# Paired-playout policy gradient inside RL, as an E7-04-44 fine-tune (2026-10-03): it fails

**Question.** The expert review ([`expert_review_E7.md`](expert_review_E7.md)) found card-play leaks
worth 2-15 points at specific spots, and argued that RL cannot see them: one game's ±1 is too noisy
to credit a single card decision. Its candidate 4 was to give RL a clean per-position signal: play
each candidate move out from paired copies of the position, and use the paired differences as the
policy gradient. This note tests that candidate, and play-mode exploration (candidate 1) beside it,
as fine-tunes.

**Verdict.** Both playout arms are weaker than their control (47.5% and 46.0% head to head). The
review's spot rules did not move: Star Wars ahead in space 0%, OPEC at 5+ VP 6-10%. Exploration
alone is level with the control and also leaves the spots where they were, which agrees with
E7-14-44 ([`P30_branch_arms_c2_play_mode_temp.md`](P30_branch_arms_c2_play_mode_temp.md)). Not
worth pursuing in this form.

Code on the fork, branch
[`feat/playout-advantage`](https://github.com/JamesYouL2/DeepStruggle/tree/feat/playout-advantage):
`ai/training/playout_advantage.py`, `--playout-*` in `tools/train.py`, off by default. Not proposed
for main.

## The method

* **Sampling.** A sampled fraction of the learner's decisions is recorded with the true GameState.
  The decision kinds are choosing a card, choosing how to play it, and choices inside events.
  The last is `CHOOSE_BRANCH`, which holds Wargames' "end the game".
* **Candidates.** The policy's 4 likeliest legal moves.
* **Paired playouts.** Each candidate is played from 16 copies of the position. Copy j of every
  candidate gets the same dice and the same redeal of the cards the decider cannot see. The current
  network plays both sides greedily to the end of the turn, and the critic values the result.
* **The loss.** The all-actions policy gradient over the candidates: the value of each candidate
  minus the policy-weighted mean, divided by the batch's standard deviation.

**Redeal or the true hand?** Measured on CI (`playout_hidden_variance.py`, 384 decisions of
E7-04-44@1200M, 8 hands × 8 dice). The hidden hand is ~9% of a pair's noise and the dice the rest.
Keeping the true hand shares that 9% across every pair, so reliability stops at ~0.75. A redeal
averages it out:

| pairs | reliability, true hand | reliability, redeal |
|---:|---:|---:|
| 8 | 0.61 | 0.71 |
| 16 | 0.68 | 0.83 |
| 32 | 0.73 | 0.91 |

The redeal shows no bias against the true hand (favourite's label −0.0016, z −0.5). Card choice is
the noisiest kind: 0.41 at 8 pairs even with a redeal.

## Arms

All arms warm-start from E7-04-44@1200M, with E7-71-45's flags otherwise: lr 3e-5 constant,
seed 45, opponent pool 0.3. They ran on one RunPod RTX 4090, torch 2.13.0+cu129, engine
`1d11c2f2`. Release [`e7-75-87-45`](https://github.com/JamesYouL2/DeepStruggle/releases/tag/e7-75-87-45)
holds the four 80M finals.

| arm | change | steps | result |
|:---|:---|---:|:---|
| E7-75-45 | control | 80M | — |
| E7-74-45 | playout, 2 optimiser steps of its own per iteration, coef 1 | 80M | **no effect:** `playout_p_best` 0.49 → 0.47 over the run |
| E7-76-45 | `--play-mode-temp 2.0` | 80M | level (49.9% / 48.9%) |
| E7-77-45 | E7-74-45 + temperature 2.0 | 80M | the playout half had no effect, as in E7-74-45 |
| E7-78/79-45 | playout added to every PPO minibatch ("joint"), coef 1 / 3 | 20M | moves the policy on buffered positions |
| E7-80..84-45 | dose check 2: separate baseline; joint at coef 1/3, batch 256/64, sample 0.002/0.008 | 20M | the buffer gain is mostly memorisation |
| E7-85-45 | joint, coef 3, batch 64, sample 0.008 | 80M | **47.5%** against the control |
| E7-86-45 | E7-85-45 + temperature 2.0 | 80M | **46.0%** against the control |
| E7-87-45 | separate, the same sampling (fresh-metric baseline) | 80M | — |

**Why E7-74-45 did nothing.** An iteration takes 64 PPO optimiser steps (65,536 / 4,096 × 4
epochs). Two playout steps that share Adam's state with them move the policy by nothing measurable,
although the labels were reliable (split-half 0.82). Adding the term to every PPO minibatch
(`--playout-mode joint`) fixed that.

**Buffer agreement or generalisation?** `playout_p_best` (the policy's mass on the playout-best
candidate in the update batch) rose to 0.57 at batch 256, but those positions are reused hundreds of
times. `playout_fresh_p_best` reads the policy at each newly labelled decision, before any update
has seen it:

| arm (80M) | fresh p on the best, last quarter | fresh value given up, last quarter | entropy at end |
|:---|---:|---:|---:|
| E7-85-45 playout | 0.478 ± 0.002 | 0.0297 | 0.246 |
| E7-86-45 playout + temperature | 0.471 ± 0.002 | 0.0298 | 0.240 |
| E7-87-45 baseline | 0.466 ± 0.002 | 0.0304 | 0.300 |

Generalisation is real but small: about +1 point on the playout-best move. The sharpening is large,
with entropy 0.30 → 0.24. KL to π_ref stayed at 0.002 in every arm.

## Strength and card play

Round robin on CI (`37157119381`), 2,000 games per side per pairing, temperature 0.1:

| model | Elo | against E7-75-45 |
|:---|---:|---:|
| E7-75-45 control | 1511 | — |
| E7-76-45 temperature | 1506 | 49.9% |
| E7-85-45 playout | 1495 | **47.5%** (about −3 SE) |
| E7-86-45 both | 1488 | **46.0%** (about −5 SE) |

Both playout arms are below the control in both seats (45.5-48.6%).

Doctrine census, 4,000 greedy games each (CI `37151939991`, `37151938283`, `37157120790`,
`37157122263`):

| rule / card | control | temperature | playout | both |
|:---|---:|---:|---:|---:|
| US events Star Wars when ahead in space | 0% | 0% | 0% | 1% |
| USSR events OPEC at 5+ VP | 21% | 7% | 6% | 10% |
| US events Alliance for Progress at 5+ VP | 15% | 17% | 22% | 19% |
| Wargames evented, US / USSR | 0.0 / 0.5% | 0.0 / 0.0% | 0.0 / 0.2% | 0.3 / 0.9% |
| Star Wars evented, all plays | 30% | 17% | 42% | 28% |
| Special Relationship evented, all plays | 2.7% | 1.9% | 4.1% | 10.3% |
| event share, all plays | 41.9% | 41.6% | 42.0% | 42.3% |

Some cards are evented more overall. The spot rules, where the review measured the event to be
worth points, did not move. Exploration reshuffled the event mix: Suez Crisis +18 points, OPEC −10.

For comparison, the offline distillation of bank-confirmed spots (E7-04-44@1200M+denoised-e20) moved
the same rules to 48% (Star Wars) / 64% (OPEC) / 58% (Alliance for Progress), at 48.8% against its
parent.

## Reading

* **The judge is the problem, not the dose.** The labels say what this model's greedy continuation
  thinks of a move after one turn. Training the policy towards that judge sharpens it (entropy −20%),
  costs 2.5-4 points, and buys about 1 point of agreement with the judge on new positions. This is
  the review's standing caveat at scale: a playout values a move given how *this* model plays
  afterwards.
* **Exploration does not find the spots.** Tried events are still credited by one game's result.
  The event mix drifts both ways and strength stays level. This agrees with E7-14-44, which branched
  E7-02-44 for 380M steps.
* **What could still work** is a judge better than the policy rather than an echo of it: deeper
  playouts, a search continuation (C5 measured honest search at +8-10), or the bank's own labels as in
  the distillation. That is a different experiment.

## Reproduction

* Code `feat/playout-advantage` at `23f746b` (the 80M arms) and `4da297d` (E7-74..77-45). Census rule
  change (Five Year Plan / Aldrich Ames counted with one or zero other cards in hand) on
  `exp/hungary-openings` at `c6310e0`.
* Launch: E7-71-45's flags plus the arm's, recorded in each run's `metadata.json`. The run
  directories are on the RunPod network volume and the finals in the release above.
