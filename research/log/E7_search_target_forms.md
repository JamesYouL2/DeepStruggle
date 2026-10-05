# Search target forms against the prior (2026-10-05): no form at 32-64 simulations carries a measurable gain, and Gumbel's improved policy adds noise

**Question.** C5 trains the policy toward a search target at a sampled fraction of decisions. The
visit-count target lost as a fine-tune ([`E7_search_finetune.md`](E7_search_finetune.md)), and
[`E7_fork_search_summary.md`](E7_fork_search_summary.md) recommended Gumbel MuZero's improved
policy instead. Rather than train on each candidate, this measures them: where a target's top move
differs from the network's prior, is the target's move actually better?

**Answer.** No target form tested is measurably better than the prior at C5's budget. The
visit-count target rarely departs from the prior, and when it does the move is no better (64
simulations: +0.001 ± 0.002 win probability per departure; 256: −0.004 ± 0.001). Gumbel's improved
policy as published departs on 15-48% of positions, including a third of those where the prior is
above 0.95, and those departures are neutral to slightly harmful. **Recommendation 1 of the summary
is withdrawn.**

## Method

Probe `ai/eval/target_forms.py` + `tools/target_forms_probe.py` (in this change; the runs used
fork branch `exp/target-forms` at `1bcc6a4` / `d775a18`, whose CI workflow sharded it over 16
runners and stays on the fork). One part locally:
`PYTHONPATH=.:build/release python tools/target_forms_probe.py --model <ckpt.pt> --positions 250
--seed 1 --pairs 32 --dump part-1.jsonl`, then `--merge part-*.jsonl --output-md report.md`.

* **Model.** `soup5@1200M+2800M.pt` (sha256 `0b62748fa869`): the shallow soup's four ingredients
  and E7-20-44@2,800M averaged uniformly; fork release
  [`e7-20-44-soups`](https://github.com/JamesYouL2/DeepStruggle/releases/tag/e7-20-44-soups).
* **Positions.** 4,000 (16 runners x 250) from the model's own self-play at the rollout temperature,
  every non-forced decision a candidate with probability 1/8 as the training searcher samples them,
  drawn uniformly from whole games. Setup excluded.
* **Targets.** Honest search throughout: one world sampled from the mover's side, the training
  searcher's configuration otherwise (temperature 0, auto-advance, no root noise).
  * `visits@N`: PUCT root visit counts, the P15-X4b target. `,pt1.5` / `,rpt2`: priors tempered at
    every node / at the root (`BatchedMCTSConfig.prior_temp` / `root_prior_temp`).
  * `cq@N`: Gumbel MuZero's improved policy softmax(logits + sigma(completed Q)) computed from the
    same PUCT statistics, so only the target formula changes. sigma and the completed Q follow mctx
    (`qtransform_completed_by_mix_value`: c_visit 50, c_scale 0.1, Q rescaled min-max over the legal
    moves). `,raw`: Q's own [-1, 1] mapped to [0, 1] instead of the min-max rescale.
  * `gumbel@N`: a Gumbel root -- 8 candidates by Gumbel-top-k over the logits, N simulations split by
    sequential halving -- and the same improved-policy target. Each halving phase searches a
    candidate's position afresh with the batched search (visits and values pooled across phases)
    rather than growing one tree.
* **Verdict.** Where a form's top move differs from the prior's (a *departure*), both moves are
  played to the end of the game greedily by the model, from paired copies: pair j redeals the cards
  the mover cannot see and fixes the dice identically in both branches. **Advantage** = mean paired
  difference in result from the mover's side, in win probability (+0.01 = one point). **Gain per
  position** = departure rate x mean advantage. A move paired with itself scores exactly 0 (tested).

## Result

At 32 simulations (`--search-sims`'s default), 32 pairs per departure, CI `37252517871`:

| form | departs | mean advantage on departures | confirmed / refuted (2 SE) | gain per position | mean TV from prior |
|:---|---:|---:|---:|---:|---:|
| visits@32 | 51 (1.3%) | +0.002 ± 0.013 | 1 / 2 | +0.02‰ | 0.033 |
| visits@32, pt1.5 | 69 (1.7%) | −0.005 ± 0.011 | 1 / 2 | −0.09‰ | 0.060 |
| visits@32, rpt2 | 102 (2.5%) | −0.004 ± 0.009 | 3 / 4 | −0.09‰ | 0.110 |
| **cq@32** | **1,731 (43.3%)** | −0.003 ± 0.002 | 31 / 34 | **−1.19‰** | 0.418 |
| cq@32, raw | 52 (1.3%) | +0.001 ± 0.011 | 2 / 3 | +0.02‰ | 0.014 |
| **gumbel@32** | **610 (15.2%)** | −0.004 ± 0.003 | 13 / 13 | **−0.61‰** | 0.143 |
| visits@64 | 98 (2.5%) | +0.004 ± 0.009 | 5 / 3 | +0.11‰ | 0.043 |
| cq@64 | 1,913 (47.8%) | −0.004 ± 0.002 | 41 / 46 | −2.06‰ | 0.468 |
| gumbel@64 | 693 (17.3%) | −0.002 ± 0.003 | 13 / 13 | −0.38‰ | 0.163 |
| visits@256 | 227 (5.7%) | −0.001 ± 0.006 | 5 / 7 | −0.07‰ | 0.077 |

Departure rate by the prior's confidence (top-move probability):

| prior top p | positions | visits@32 | cq@32 | gumbel@32 | visits@256 |
|:---|---:|---:|---:|---:|---:|
| < 0.5 | 283 | 11.3% | 71.0% | 64.0% | 22.3% |
| 0.5–0.8 | 810 | 2.2% | 56.5% | 34.1% | 13.5% |
| 0.8–0.95 | 673 | 0.1% | 54.5% | 14.6% | 3.1% |
| ≥ 0.95 | 2,234 | 0.0% | 31.6% | 2.5% | 1.5% |

**Follow-up at higher precision**, the two visit forms only, the same 16 x 250 positions, 1,024 pairs
per departure (CI `37255876807`):

| form | departs | mean advantage on departures | confirmed / refuted |
|:---|---:|---:|---:|
| visits@64 | 95 (2.4%) | **+0.001 ± 0.002** | 9 / 9 |
| visits@256 | 220 (5.5%) | **−0.004 ± 0.001** | 15 / 23 |

No segment (card choice, play mode, influence, coups, events, headlines) shows a consistent gain
for any form.

## Reading

* **Gumbel's improved policy as published is the wrong target here.** It moves the top move on
  15-48% of positions, a third of them where the prior is above 0.95, and the moves it picks are no
  better: confirmed and refuted counts are equal, as noise would give. The min-max rescale of Q is
  the cause -- at 32-64 simulations the searched values differ by hundredths, and the rescale
  stretches any spread to [0, 1], several logits through sigma. Without it (`raw`) the target
  collapses back onto the visit target.
* **The visit target departs rarely and gains nothing measurable.** At 1,024 pairs the 64-simulation
  departures are +0.001 ± 0.002; the 256-simulation ones are slightly worse. This fits E7-93/94-45:
  training toward the target sharpened the policy and lost strength.
* **Search still wins at play time** (+4.9 for 64-simulation honest search on E7-20-44@2,800M, all
  decisions). The two are compatible: here a search move is continued by the plain network, while
  at play time search also chooses the follow-up. A target taken one decision at a time is what
  distillation asks the policy to learn, and on this judge it carries no information beyond the
  prior.

## What this does not say

* **The judge is the network's own continuation.** A move whose worth lies in a follow-up the
  network does not find reads as worse than it is; a search-continued judge was not run.
* One model, one budget range (32-256); prior temperatures only at 32.
* The Gumbel root re-searches each candidate per phase instead of growing one tree; the cq forms
  use PUCT statistics, so the formula's failure does not depend on that.
