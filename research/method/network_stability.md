# Early stability checks for a network trained from scratch

Owner, 2026-10-23: find a way to train bigger networks reliably, and get there gradually. The scaled flagship
(A21/A22) spent 4,645M steps before it was stopped, yet its failure was visible in its training log at 30M; the
next attempt (A25) fixed that symptom and failed in a way visible at 400M. These are the checks that would have
caught both, cheapest first, with the numbers measured on the networks that worked and the ones that did not. The
network they all start from is A8 ([`a8_architecture.svg`](a8_architecture.svg)).

`ai/eval/network_health.py` runs checks 1 and 2:

```bash
# 1. the training log, against the control's at matched steps (free; from 30M)
PYTHONPATH=.:build/release .venv/bin/python -m ai.eval.network_health --runs <new run dir> <control run dir>
# 2. checkpoint geometry on shared self-play positions (CPU, a few minutes; at 60-100M and at 400M)
PYTHONPATH=.:build/release .venv/bin/python -m ai.eval.network_health \
  --checkpoints <new>/snapshot_<N>steps.pt <control>/snapshot_<N>steps.pt --out <report.json>
```

The control is `E8-A8-R1-S44-2` (A8 on R1, E8 with ENG-5), training long for exactly this purpose.

## 1. The training log (from 30M)

| series (per 50M window) | healthy | failing |
|:---|:---|:---|
| `logit_lse_absmax`, max | within ~2x the control's: A8 15–30 through 60M, A25 (trunk norm) 31–40 | A22 239–474 at 30–70M, A34 160–200; flag > 3x the control |
| `kl_div` (to π_ref), max | ≤ 0.07 (A25 on 16k), ≤ 0.12 early on 4,096 | A22 on R70 0.3–3.3 per 100M window |
| `entropy`, median | not falling faster than the control's | A22 0.36 against A8's 0.50 at 350–400M |
| `explained_variance`, median | within ~0.03 of the control's | A22 0.80 against 0.88 |

The first row is the one that matters most: A22's failure was in it from 20M.

## 2. Checkpoint geometry (at 60–100M, and at 400M at the latest)

Measured on 4,096 positions of 16 self-play games of the first checkpoint given (every decision type).

| measure | A8 control | failing |
|:---|:---|:---|
| logits, worst decision type's spread (max − min legal logit), max | ~100 (2,400M) | A22 at 400M: 13,926 (play mode, Ops mode, branch) |
| head input (what the heads read): effective rank | **grows** with training: 51 fresh → 94 on card decisions, 130 on all decisions at 2,400M | **falls**: A25 64 fresh → 24 (400M) → 22 (2,400M) of 768; A22 14 at 400M |
| head input: largest dimension's RMS over the median one | ~8 | A25 21 |
| residual stream before the trunk norm (networks that have one): RMS | — | A25 1.3 → 8 → 25 → 37 at 0 / 400 / 1,600 / 2,400M, largest dimension 49x the median |
| policy head weight norm / a fresh head's | a trajectory: 9–10x at 2,000–2,400M, 15x at 6,400M, 20x at the A13+R48 plateau | — (plasticity, not a stop signal; see P37) |
| policy head dormant units, τ 0.1 | 0.3–5% at 2,400M; 13.5% at the plateau | A25 8.5% at 2,400M |

The effective rank is the check A25 needed: its logits were bounded, its strength was level with the control's to
1,600M, and its features had already collapsed to a quarter of A8's.

**At 60M already** (`data/reports/network_health_bigger_60M.json`; 4,096 positions of A26's self-play, every
network against its own shape freshly initialised on the same positions), the rank's growth separates every network
that scaled from every one that did not, before the logits or the strength do:

| network | head input effective rank, fresh → 60M | growth | logits, \|lse\| max / worst spread | outcome |
|:---|:---|---:|:---|:---|
| A8 control (no residual blocks) | 42.5 → 96.5 of 480 | 2.3x | 21 / 44 | healthy |
| A26 (width only, no residual blocks) | 38.4 → 124.3 of 768 | 3.2x | 24 / 52 | passes so far |
| A25 on R1 (4 blocks, trunk norm) | 49.7 → 66.6 | 1.3x | 19 / 40 | failed (3.1% at 400M) |
| A25 on R47 (4 blocks, trunk norm) | 49.7 → 61.9 | 1.2x | 20 / 35 | rank 22 by 2,400M |
| A34 (4 blocks, no dropout, no norm) | 49.6 → 61.4 | 1.2x | 167 / 334 | stopped |
| A22 on R1 (4 blocks) | 49.6 → 25.7 | 0.5x | 123 / 245 | failed |
| A22 on R70, the flagship | 49.6 → 18.7 | 0.4x | 83 / 131 | failed |

**Rule: at 60M the head input's effective rank should be at least ~2x its fresh value** (the networks that train well:
2.3–3.2x; every residual-block trunk: 0.4–1.3x).

## 3. Plasticity probe (400M and later; CPU, ~10 minutes)

P37a's fits on 20,000 of P36c's positions ([`../plans/P37_known_playbook_remainder.md`](../plans/P37_known_playbook_remainder.md)):
a **fresh head on the frozen trained trunk** should fit random labels at least as well as a fresh trunk of the same
shape -- A8's trained trunk 98.7% against a fresh one's 93.5%; A25's 71.7% against 88.1% (80 epochs). The network's
**own head** on the same task is the plasticity trajectory (A8: 47% at 2,000M, 35% at 6,400M, 33% at the plateau).

## 4. Strength

Reads against A13+R48's SWA 12,730–12,800M (80M-window SWAs, raw, 1,000 deals a side) at 400M and every 400–800M,
beside the control's at matched steps, and one head-to-head with the control once both pass 1,600M.

## Rules for getting to bigger networks

* **One change at a time, from a network that passes every check.** The ladder so far: A8 → A26 (width: A13's
  512 projections, fusion 768, no residual blocks) → depth with a bounded stream → attention.
* **The control's recipe** (R1, 4,096 minibatch) unless the change is the recipe; a size change on another
  minibatch is two changes.
* **Residual blocks need `--ladder-res-dropout 0`**: the rollouts act in eval mode and the update in train mode, and a
  checkpoint's weights do not record its dropout, so a reloaded network has the historical 0.05 whatever it trained with.
* **A residual stream must be bounded, not only its output.** `--ladder-trunk-norm` stopped A22's logit blow-up and
  left the stream growing in a few dimensions, which the norm then turned into a rank collapse (A25).
* **Stop at the check, not at the read.** A network that fails check 1 or 2 is stopped there; strength reads only
  confirm.
