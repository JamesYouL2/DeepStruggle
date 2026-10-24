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
| 1 | `E8-A26-R85-S44_20261024_204210` (rung 0 + weight decay 1e-2, norms and biases undecayed; R85) | Running (launched 2026-10-24 20:42 UTC); `--weight-decay` / `--no-decay-norms-biases` and the per-iteration `head_norm_over_fresh_*` added for it (5be8ee5) |
