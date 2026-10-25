# P38 — scaling the network reliably: the stability ladder, rung by rung (2026-10-24)

**Goal (owner).** Train bigger networks reliably, so that a capacity result can be read at saturation
instead of lost to a collapse, and so that a design can be trusted to scale with compute before
compute is bought. The instrument is [`../method/network_stability.md`](../method/network_stability.md);
this plan is the sequence of rungs and what each tries.

**What the record fixes.** Two measured failures, both in the trunk-and-heads geometry rather than in
strength: the residual stream grows without bound in a few dimensions (A22: RMS 1.3 → 37 by 2,400M,
the largest dimension 46× the median) and the rank of what the heads read collapses (A25: 64 → 22 of
768 while A8's rises 51 → 94 of 480), even with a LayerNorm after the stack; and the policy heads'
weights grow to 20–50× a fresh head's and lose plasticity (P37a). Width alone without blocks (A26)
passes the checks, and at a learning rate scaled down for the width (2e-4) its per-update step matches
the control's -- the µP observation found by hand -- but a smaller rate did not by itself make it better
than the control. **Decreasing the learning rate is not the fix; bounding the norms is the candidate.**

**The recipes, read in detail.**

| | ours (A22 / A25) | SimBa (Lee et al., ICLR 2025) | SimbaV2 (Lee et al., ICML 2025) |
|:---|:---|:---|:---|
| input | fusion Linear + LayerNorm + GELU | RSNorm running statistics, linear stem | RSNorm, a shift constant, ℓ2 to the unit sphere |
| block | pre-LN, Linear d→d, LN, Linear, dropout 0.05, add | pre-LN, Linear d→4d, ReLU, Linear, add | stream at unit ℓ2 norm; h̃ = ℓ2(W₂ ReLU((W₁h) ⊙ s)); h ← ℓ2((1−α)h + α·h̃), α learnable per dimension, init 1/(L+1) |
| after the stack | none (A22) / LayerNorm (A25) | LayerNorm | nothing needed |
| weights | AdamW, decay 1e-4 | AdamW, decay 1e-2 | projected to unit norm after every step, no decay; learnable scalers (init √(2/d)) restore scale |
| heads | shared trunk, policy and value | critic scaled, actor kept small ("scaling the actor tends to hurt") | linear with a scaler |
| tested on-policy | -- | PPO on Craftax, 1B steps | no (SAC only) |

SimBa v1's answer to the growing stream is the final LayerNorm, which A25 had and which did not stop the
rank collapse. SimbaV2 removes both measured failures by construction: the stream cannot grow because it
is renormalised at every block, the weights cannot grow because they are projected back after every
step; it scaled in depth to 8 blocks where SimBa plateaued at ~2. It reports norms and effective learning
rate, not rank, so whether the sphere preserves rank is what our 60M check measures first. **Not
transferred:** the categorical critic and reward scaling (continuous-control return scales; our reward is
±1 and the categorical head settled −28 on E3), RSNorm (our observation is 0/1 and fractions by design;
the fusion LayerNorm is the stem). **Carried regardless:** capacity belongs in the trunk and the value
path; the policy heads stay small and are reset (P37's first branch).

## The rungs

Each rung is **one change** from a rung that passes every check, on the control's recipe (R1, 4,096
minibatch) unless the change is the recipe; `--ladder-res-dropout 0` on every block rung. Codes at
launch. The control is `E8-A8-R1-S44-2` (A8 on R1, E8 with ENG-5, running long).

| rung | change | from | what it tests |
|:---|:---|:---|:---|
| **0** | A26 at lr 2e-4 (R84), running | A26 | the width reference with the step size matched |
| **1** | **weight decay 1e-2** (SimBa's; ours is 1e-4), norms and biases excluded | rung 0 | the cheapest knob against the measured head-norm growth |
| **2** | **the SimbaV2 trunk, 2 blocks**: ℓ2-normalised stream, lerp residual with α init 1/(L+1), Linear d→4d + ReLU, no dropout, weights projected to unit norm after each optimiser step with learnable scalers; the fusion LayerNorm kept as the stem | the better of 0 / 1 | depth with the stream and the weights bounded by construction (replaces A25's block, not a patch to it) |
| **3** | the same trunk, **4 blocks** | rung 2 | depth scaling; SimBa v1 plateaued at ~2 |
| **4** | **the token encoder** (A33's two attention layers) on the rung-3 trunk | rung 3 | attention on a trunk that holds rank |
| **5** | width and depth together (fusion 1024 / 8 blocks) | rung 3 or 4 | the first network that could use a cloud budget |

Rung 2's small α at initialisation makes a fresh deep network behave almost like the shallow A8 at step
0, which is also what depth-µP's 1/√L residual scaling buys; the rung covers both ideas. The prepared
`--ladder-res-post-norm` (a LayerNorm after every residual add) stays available as the fallback if rung
2's build is delayed: it bounds the stream but not the weights.

## Checks and reads, per rung

From [`../method/network_stability.md`](../method/network_stability.md), stopped at the check, not the read:

1. **The training log from 30M**: `logit_lse_absmax` within ~2× the control's; per-update approximate
   KL at the control's level (the step size); entropy not falling faster; explained variance within 0.03.
2. **Geometry at 60M**: the head input's effective rank ≥ 2× its fresh value (the networks that train
   well: 2.3–3.2×; every collapsing trunk 0.4–1.3×); the largest dimension's RMS over the median; the
   residual stream's RMS where there is one; **head weight norm over a fresh head's**, logged every 50M
   from now on as the plasticity trajectory.
3. **Plasticity at 400M** (P37a's probe): a fresh head on the frozen trunk fits random labels at least
   as well as a fresh trunk of the same shape.
4. **Strength** at 400 / 800 / 1,600M: 80M-window SWAs against A13+R48's SWA (12,730–12,800M) beside
   the control's at matched steps; head to head with the control once both pass 1,600M; a rung that is
   level at 1,600M and holds rank continues to its own plateau by the plain line's rule.

## Decision rules

* A rung **passes** when checks 1–3 hold and its 1,600M read is level with or above the control's. A
  pass is the base of the next rung.
* A rung that fails check 1 or 2 is stopped at 60–100M; the change is recorded as the cause and the
  next rung is built from the last pass.
* Rung 1 is adopted into the recipe (an R code carried by every later rung) if the head norm's trajectory
  bends below the control's without a strength cost.
* **Capacity pays** is declared only by a rung's plateau above A13+R48's by the per-seat rule raw and
  under the corrected keyed root -- the P35 verdict, now read on a trunk that holds rank.
* The cloud question (a few thousand dollars) is asked only after rung 5 passes and a two-point slope
  exists (rung 3 against rung 5 at matched steps).

## Sequencing

One GPU slot for the ladder beside the long control; the encoder flagship (A33, paused at 180M) is
superseded by rung 4 and is not resumed on its old trunk. Rung 1 launches now beside rung 0 if memory
allows (both are A26-sized), otherwise after rung 0's 1,600M read. Rung 2's build (the hyperspherical
block, the weight projection step, the scaler parameters, `launch_flags`/`run_codes` coverage, a
parity test that a fresh network's forward is unchanged in eval) proceeds meanwhile. P37's head-reset
branch runs in the same slot when a rung is between reads.

## Not in this plan

Learning-rate sweeps on their own (rung 0 is the one that stands); the categorical critic; RSNorm;
any target or search change (P34 / P36, closed).

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

| rung | run | result |
|:---|:---|:---|
| 0 | `E8-A26-R84-S44_20261024_101857` (A26, lr 2e-4) | **Does not pass**: checks hold (step size at the control's, effective rank 3.0–4.2x fresh, outlier dimension near A8's), strength does not -- against A13+R48's SWA 8.0 / 13.8 / 16.0 / 19.5% at 400 / 800 / 1,200 / 1,600M, the control 9.8 / 16.4 / 18.4 / 24.5%; head dormant 2.1% at 1,600M (control 0.0%). Stopped at 1,926M for rung 1 |
| 1 | `E8-A26-R85-S44_20261024_204210` (rung 0 + weight decay 1e-2, norms and biases undecayed; R85) | Running (launched 2026-10-24 20:42 UTC). **60M checks pass**: effective rank 125 of 768, 3.1x fresh (control 97 of 480); per-update KL 0.0165 at 50–60M (control 0.0158); head weights over a fresh head's below the control's (policy head 1.8x vs 2.1x, play-mode head 3.7x vs 5.4x) -- the decay bends the trajectory; logits level (\|lse\| max 17.5 vs 20.6). **400M: 7.9%** (US 7.2 / USSR 8.5; rung 0 8.0%, control 9.8%); head weights well below the control's (policy head 2.8x vs 4.3x a fresh head's, play-mode head 6.7x vs 12.0x), logit spread narrower (worst 44 vs 75), effective rank 169 of 768, 3.8x fresh. **800M: 11.6%** (US 9.4 / USSR 13.1; rung 0 13.8%, control 16.4%) -- the decay costs strength (−2.2 against rung 0, the gap to the control −4.8) while keeping the heads small (policy head 3.1x a fresh head's, control 6.0x; head dormant 0.0%) and the rank highest yet (190 of 768). **Stopped at 810M (2026-10-25 01:05 UTC)**: it cannot plausibly reach the control by 1,600M and rung 0 is the better of rungs 0 / 1, so rung 2 is launched on rung 0's recipe now instead of five hours later (owner away; P38's 'better of 0 / 1'). `--weight-decay` / `--no-decay-norms-biases` and the per-iteration `head_norm_over_fresh_*` added for it (5be8ee5) |
| 2 (build) | -- | **Built, not launched**: `--ladder-sphere-blocks N` / `--ladder-sphere-expansion` (`ai/models/hyperspherical.py`): stream on the unit sphere, h ← ℓ2(h + α ⊙ (ℓ2(W₂ ReLU((W₁h) ⊙ s)) − h)), α per dimension init 1/(L+1), W₁ d→4d and W₂ without bias, rows re-projected to unit norm by a post-step hook on the optimiser (every step site), heads read √d · h; replaces the residual blocks and the trunk norm. Tests: the stream stays on the sphere under arbitrary weights, rows are unit after every step, config and weight detection round-trip; an end-to-end CPU smoke run through `tools/train.py` trains, logs, snapshots and reloads it. Launches in the ladder slot from the better of rungs 0 / 1 |
| 2 | `E8-A27-R84-S44_20261025_010608` (A27: A26 + 2 sphere blocks, on rung 0's recipe R84) | **Fails the step-size check, stopped at 70M (2026-10-25 01:38 UTC)**: per-update approximate KL 0.049–0.062 through 60M (control 0.016–0.019; clip fraction 0.34 vs 0.23), KL to pi_ref max 0.07–0.08 (control 0.04); geometry passed -- effective rank 116 of 768, 2.7x fresh, logits small (worst spread 33 vs 47), head weights below the control's, entropy high (1.09 vs 0.72 at 60–80M), explained variance 0.905 (0.885). Adam's step is large against unit-norm rows' entries (~0.02–0.04); relaunched at lr 1e-4 (E8-A27-R82-S44) |
| 2b | `E8-A27-R82-S44_20261025_013823` (the same at lr 1e-4, R82) | Running (launched 2026-10-25 01:38 UTC). **60M: the step size does not follow the learning rate** -- per-update approximate KL 0.050 at 50–60M (at 2e-4 0.057, control 0.016), clip fraction 0.32 (0.34; 0.23): halving the rate cut it 13%, not the ~4x that rung 0's story predicts, so on this trunk the step is set by something structural (PPO's clip range reached on every update is the candidate), not by the rate. Geometry clean: effective rank 119 of 768, 2.9x fresh, the smallest heads yet (policy head 1.5x a fresh head's, control 2.1x), no dormant units, logits small; KL to pi_ref max 0.062–0.066 (control 0.04), entropy 1.08 (0.72), explained variance 0.914 (0.885). **Kept past the failed check 1 by exception** (owner away): with no learning-rate fix to try, only the 400M read can say whether the large updates cost strength. **400M: 2.3%** (US 2.7 / USSR 1.9; control 9.8%) and the effective rank falling, 119 → 84 of 768 (2.0x fresh; control 103 of 480), largest dimension 9.1x the median: the step-size check was a real warning and keeping it past the check was wrong. **Stopped at 409M (2026-10-25 04:46 UTC)**; rung 2c scales the trunk's output to RMS 0.5 |
| 2c | `E8-A28-R84-S44_20261025_044640` (A28: the sphere trunk with output RMS 0.5, on R84) | **Fails the step-size check, stopped at 69M (2026-10-25 05:17 UTC)**: per-update approximate KL 0.056 at 50–60M, identical window by window to rung 2's (0.057) -- the output scale changes nothing, as halving the learning rate changed little (−13%); the sphere trunk's large per-iteration movement is inherent to it under our PPO update, cause not found. Geometry clean (effective rank 97 of 768, 2.6x fresh, no dormant units). The slot goes to P38's named fallback, post-LN residual blocks (E8-A29-R84-S44) |
| fallback | `E8-A29-R84-S44_20261025_051812` (A29: A26 + 2 post-LN residual blocks, dropout 0, on R84) | **Fails the step-size check, stopped at 70M (2026-10-25 05:43 UTC)**: per-update KL 0.043 at 50–60M (control 0.016), geometry clean (effective rank 112 of 768, 2.9x fresh). **Every trunk with blocks after the fusion takes 2.6–3.6x the control's step per update on the 4,096 minibatch** (per-update approximate KL at 50–60M: A8 control 0.016; A26, no blocks, 0.017 at 2e-4; A34 / A25 / A29 0.041–0.043; the sphere trunk 0.050–0.057 at lr 2e-4, 1e-4 and output RMS 0.5 alike; A22 with dropout 0.05 0.022), and the 16k minibatch brings it back (A25 on R47: 0.019, the one block network level with the control to 1,600M). The lever is the number of gradient steps per iteration, not the learning rate; relaunched at 16k (E8-A29-R86-S44) |
| fallback, 16k | `E8-A29-R86-S44_20261025_054307` (A29 on R86: R84 at the 16,384 minibatch) | Running (launched 2026-10-25 05:43 UTC). **60M: every check passes -- the first trunk with blocks to do so**: per-update approximate KL 0.017 at 50–60M (control 0.016; the same trunk on 4,096 0.043), clip fraction 0.213 (0.232); effective rank 102 of 768, 2.6x fresh, largest dimension 2.2x the median, the smallest heads yet (policy head 1.5x a fresh head's, control 2.1x), no dormant units, logits small (worst spread 32 vs 44). **400M: strength level with the control, rank collapsing**: 9.5% against A13+R48's SWA (US 9.0 / USSR 9.5; control 9.8%) -- the best of the block networks at matched steps -- but the head input's effective rank fell 102 → 37 of 768, below its fresh value (42; control 120 of 480), top-10 components 59% of the variance: post-LN bounds the stream's size, not its rank. Heads still small (policy head 2.4x a fresh head's, control 4.3x). **Stopped at 776M (owner, 2026-10-25)** for the sphere trunk on 16k (E8-A27-R86-S44), which held rank where post-LN did not |
| 2, 16k | `E8-A27-R86-S44_20261025_102344` (A27 on R86: the sphere trunk at the 16,384 minibatch) | Running (launched 2026-10-25 10:23 UTC). **60M: passes, narrowly**: per-update approximate KL 0.024 at 50–60M (control 0.016; the same trunk on 4,096 0.057) with the clip fraction at the control's (0.230 vs 0.232) -- the 16k minibatch takes the step from 3.6x the control's to 1.5x; effective rank 91 of 768, exactly 2.0x fresh (control 105 of 480, 2.2x); the smallest heads yet (policy head 1.5x a fresh head's), logits small, dormant units under 1%. The 400M rank check is the test the post-LN trunk failed. **400M: holds rank, learns slower**: effective rank 91 of 768, unchanged from 60M, 2.0x fresh -- where the post-LN trunk on 16k fell 102 → 37 -- so the first trunk with blocks to keep both the step and the rank; but 7.3% against A13+R48's SWA (US 4.9 / USSR 9.3; control 9.8%, post-LN 9.5%, rung 0 8.0%); head dormant 4.3% (control 0.7%), largest dimension 7.9x the median. Continues to the 1,600M pass read |
