# Search-driven training as an E7-04-44 fine-tune (2026-10-04): both arms lose to the control

**Question.** C5 ([`../plans/P30_base_model_quality.md`](../plans/P30_base_model_quality.md)) is
gated on search headroom, and the gate passes ([`P30_search_headroom_e7.md`](P30_search_headroom_e7.md)).
Before a from-scratch C5 run, does the existing search-target path (P15-X4b: visit-count
cross-entropy from a searched fraction of decisions,
[`../archive/E3_ladder/log/P15_X4b_search_during_rl.md`](../archive/E3_ladder/log/P15_X4b_search_during_rl.md))
improve a saturated model when added as a fine-tune?

**Verdict.** No. Both arms are clearly weaker than their control after 80M steps: **43.3%** and
**45.3%** head to head (about −44 and −35 Elo, 6-8 standard errors), below it in both seats. The
census spots did not move. Visit-count targets at 64 simulations are not a usable training signal
in this form.

Code: the trainer's existing P15-X4b path (`--search-ce-coef`, `--search-sims`,
`--search-subsample`); the only addition is that `--search-node-filter` accepts the segment and
turn-range filters of [`E7_search_segments.md`](E7_search_segments.md) (`late` here). Launch flags
in [`E7_fork_search_summary.md`](E7_fork_search_summary.md#runs).

## Arms

Both warm-start from E7-04-44@1200M with E7-75-45's flags (lr 3e-5 constant, seed 45, opponent pool
0.3; [`E7_playout_pg_finetune.md`](E7_playout_pg_finetune.md)), plus `--search-ce-coef 0.5
--search-sims 64` (honest search, C++ tree, targets deferred to the end of the rollout). Each
searches about 12% of the learner's decisions, so the two cost the same.

| arm | searched decisions | why |
|:---|:---|:---|
| E7-93-45 | all, 1 in 8 (`all`, subsample 0.125) | the P15-X4b recipe |
| E7-94-45 | turns 8-10 only, 1 in 2 (`late`, subsample 0.5) | where search gains most ([`E7_search_segments.md`](E7_search_segments.md)) |

The control is **E7-75-45**: training is deterministic, and a control re-run on a later commit
(E7-92-45) reproduced E7-75-45's weights bit for bit, so the control is exactly these arms without
the search term. One RunPod RTX 4090, both arms at once, 12-14k steps/s each; engine `26686521`,
torch 2.13.0+cu129. Finals in release
[`e7-93-94-45`](https://github.com/JamesYouL2/DeepStruggle/releases/tag/e7-93-94-45).

Early in E7-93-45 (2.6M steps) the targets were clean: no illegal target mass, 0.01% of visits
dropped, target entropy 0.41 against the policy's 0.38, the search term 35% of the policy gradient,
KL to π_ref 0.003 per iteration.

## Strength

Round robin on CI (`37224143219`), 2,000 games per side per pairing, temperature 0.1:

| model | Elo | score against E7-75-45 | as USSR / as US |
|:---|---:|---:|---:|
| E7-75-45 control | 1527 | — | — |
| E7-94-45 late, 1 in 2 | 1491 | **45.3%** | 45.2 / 44.5 (win %) |
| E7-93-45 all, 1 in 8 | 1482 | **43.3%** | 42.2 / 43.6 (win %) |

E7-94-45 against E7-93-45: 50.5%.

## Card play

Doctrine census, greedy self-play (CI `37224149541`, `37224155356`; control `37178033890`):

| rule | control | E7-93-45 | E7-94-45 |
|:---|---:|---:|---:|
| US events Star Wars when ahead in space | 0% | 0% | 1% |
| USSR events OPEC when it scores 5+ VP | 21% | 22% | 22% |
| US events Alliance for Progress when it scores 5+ VP | 15% | 12% | 13% |
| Captured Nazi Scientist evented (either side) | 83% | 89% | 90% |
| USSR plays Five Year Plan with ≤1 other card in hand | 58% | 43% | 57% |
| US plays Aldrich Ames Remix with ≤1 other card in hand | 88% | 79% | 44% |

The spots the expert review priced did not move; two hand-management rules got worse. The census counts action-round play-mode decisions only (the card already chosen), so a headline is never one of its plays. The Star Wars rule's 0% is therefore "never evented in an action round while ahead": the control evented Star Wars in 143 of its 479 US plays with the event legal (30%), every one of them a headline, consistent with the 24% of holdings in [`event_census_soup.md`](event_census_soup.md), which counts headlines.

## Reading

* **This is the third in-RL target to lose to its control at 80M.** Paired-playout policy gradient
  lost 2.5-4 points ([`E7_playout_pg_finetune.md`](E7_playout_pg_finetune.md)); the card-effects
  block was level ([`E7_card_effects_block.md`](E7_card_effects_block.md)); search targets lose 5-7.
  Search wins at play time (+4 against the soup at 64 simulations), but training the policy
  towards its 64-simulation visit counts does not transfer that gain.
* **P15-X4b's +165.6 Elo does not carry over.** That was one seed at 20M on E3, against a control
  that was itself declining. Here the control is healthy and exactly reproducible.
* **Not tested:** a cold start with search from step 0 (C5 as planned), other coefficients or
  simulation counts, and target forms other than visit counts. The training metrics and logs are on
  the RunPod network volume, not downloaded; the drift of the search term's gradient share and of
  entropy over the run is therefore not reported.

## Reproduction

Launch: E7-75-45's flags plus the arm's, recorded in each run's `metadata.json` (in the release).
Census on the same branch, `doctrine_census.yml`, 500 games × 8 runners.
