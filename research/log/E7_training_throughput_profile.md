# E7 training throughput: where the time goes, and what stands between it and 2× (2026-10-05)

The owner asked, after the network shrank to E7's shallow recipe (1.33M parameters), what larger
batches would buy and what stops training from being twice as fast. Nothing here changed training
code; the scripts are in `data/logs/perf/` (`e7_bench.py`, `e7_pe_head.py`, `e7_pe_equiv.py`,
`e7_sgemm.py`) and the job's `bench_batch.sh` / `phase_prof.py`.

## Larger batches: the env count matters, the minibatch does not

`tools/train.py` with E7-20-44's flags, 5 minutes per config, steady steps/s from the training
clock (noise about ±5%):

| envs \ minibatch | 4,096 | 8,192 | 16,384 | 32,768 |
|---:|---:|---:|---:|---:|
| 512 | **96.3k** (recipe) | 98.4k | 92.4k | 98.4k |
| 1,024 | 104.7k | 100.6k | 99.2k | -- |
| 2,048 | **114.0k** | -- | 113.6k | 112.2k |

* **The minibatch does nothing** for speed. The update's cost is its total arithmetic (4 epochs over
  the rollout), which a minibatch size only reslices.
* **2,048 envs is ~+18%**, by amortising each rollout step's fixed costs over 4× the rows. It is
  not free for learning: 262,144 steps per iteration, so 256 gradient steps between rollouts and a
  reference refresh every iteration. Untested for strength.

## Where an iteration goes

py-spy, non-blocking, main thread only (`--idle`, 100 Hz), 5 minutes of the E7 recipe with the
opponent pool; the profiled run trained at 96.4k steps/s, the same as unprofiled, so the shares
are not distorted:

| phase | share of wall |
|:---|---:|
| update: backward | 29.3% |
| update: forward + loss | 17.5% |
| update: optimizer step + grad clip | 4.4% |
| rollout: C++ step + observation (`step_flat_all`) | 10.4% |
| rollout: rest of `env.step` (Python) | 5.8% |
| rollout: forward (CUDA-graph replays, learner + pool opponent) | 8.5% |
| rollout: observation copy to the GPU (7.8 MB per step, pageable) | 7.5% |
| rollout: other Python (sampling, per-seat stats, buffer, critic tracker, GAE) | ~16% |

Timed directly on the trainer (`e7_bench.py`, 512 envs, no pool, TF32): **rollout 0.266 s, update
0.361 s** per 65,536-step iteration. torch.profiler inside each phase:

* **The update is GPU-bound:** 0.357 s of kernel time in 0.364 s of wall, 25,443 kernels.
* **The rollout is CPU- and sync-bound:** the GPU is busy 0.103 s of 0.33 s. Every step is serial
  -- build observations (C++), copy them, forward, sync the actions back, Python bookkeeping, step
  -- and nothing overlaps.

## The update's largest single cost is the per-country head, and most of it is redundant

`pe_country` is a 90 → 64 → 1 MLP run on all 84 countries of every sample: 344,064 rows per
minibatch of 4,096. Its input is `cat[raw country slots (26), ctx (64)]`, where `ctx =
pe_trunk(h)` is **the same vector for all 84 countries**. Two consequences:

* **No tensor cores.** K = 90 is not 8-aligned, so cuBLAS runs it as plain fp32
  (`ampere_sgemm_64x64_tn`, 15% of a forward+backward by itself) despite TF32.
* **The [4096, 84, 90] concatenation and the 64 ctx columns are recomputed 84 times.**

`Linear(cat[x, ctx]) = x·Wxᵀ + (ctx·Wcᵀ + b)` computes the same function with the shared term once
per sample and broadcast (the same parameters, so checkpoints are unaffected). Measured:

| | fwd+bwd, minibatch 4,096 | rollout fwd, 512 |
|:---|---:|---:|
| as built | 4.74 ms | 0.34 ms |
| ctx term split out | **3.59 ms** | 0.32 ms |

Equivalence: on real mid-game positions the two agree to 1.0–1.4e-4 in logits under both TF32 and
exact fp32 (rounding order only). Uniform-random inputs give logits up to ~87, where TF32's 1e-3
relative rounding is ~0.08; that is TF32, not the rewrite.

## The levers, measured on the trainer (512 envs, minibatch 4,096, no pool)

| variant | rollout s | update s | steps/s |
|:---|---:|---:|---:|
| as trained (TF32) | 0.266 | 0.361 | 104k |
| **ctx split** (same function) | 0.267 | **0.289** | **118k (+13%)** |
| bf16 autocast in the update only | 0.265 | 0.263 | 124k (+19%) |
| ctx split + bf16 update | 0.265 | 0.239 | 130k (+25%) |

(Production with the opponent pool runs ~8% below this harness: 96k against 104k.)

## What stops 2×

After the update levers, **the rollout is the floor**: 0.265 s per iteration caps training at
~250k steps/s even with a free update, and at 2.07 ms per step it is already half the iteration.
2× (~0.31 s per iteration) needs both halves at ~0.16 s:

* **Update, 0.36 → ~0.16 s:** ctx split (exact function), then bf16 in the update (P26 measured
  it, never adopted: it changes numerics and needs P26's gates), then `torch.compile` to fuse the
  elementwise kernels that are the other ~40% of the update (opt-in since P26; its A/B was "not
  shown harmless").
* **Rollout, 0.27 → ~0.16 s**, which no numerics option moves:
  * more envs (2,048: +18%) amortise the per-step fixed costs;
  * two per-step host syncs (`actions_t.cpu()` and the critic tracker's `v_win_t.cpu()`) and the per-step
    statistics kernels (218 kernels per step in all, forward included) could be cut to one sync;
  * pipelining two halves of the envs, so the C++ step of one half overlaps the copy and forward
    of the other, hides the GPU part (~0.8 ms of 2.07 ms per step).
* **The structural option:** collect iteration k+1's rollout while iteration k's update runs. The
  GPU is idle ~70% of the rollout and the CPU idle during the update, so the iteration would cost
  max(rollout, update) instead of their sum -- ~250k steps/s with the update levers, ~2.4×. It
  makes the data one update stale, which is a recipe change and needs an A/B.

## Found on the way

`tools/scripts/launch_flags.py` printed `--no-ladder-card-self-attention` and
`--no-ladder-cross-attention` for E7-20-44, which `train.py` rejects, and omitted
`--ladder-head-context` / `--ladder-head-static`, which `--arch ladder` requires. Its output for
an E7 run does not launch as printed.

## The shared-context head under `torch.compile` (2026-10-05)

**The same function.** Against the head as built, E7-20-44@3,600M on 256 real mid-game positions:
log-probs over legal actions and every parameter's gradient differ by 1.5e-4 and 1.7e-5 (relative)
in fp32, and by **3e-13 and 4e-14 in float64** (`data/logs/perf/e7_pe_grad_equiv{,64}.py`). The
fp32 difference is summation order only.

**Timed on the trainer** (`e7_bench.py --compile`, the trainer's own `--compile-update`, which
compiles the update's forwards; the rollout keeps its CUDA graphs). 512 envs, minibatch 4,096, no
pool, TF32; two passes, alternating:

| `--compile-update` | head | update s | steps/s |
|:---|:---|---:|---:|
| off | as built | 0.364 / 0.364 | 102.3k / 102.6k |
| off | shared context | 0.290 / 0.290 | 118.4k / 117.7k |
| default | as built | 0.282 / 0.281 | 118.7k / 117.5k |
| default | shared context | **0.218 / 0.213** | **134.6k / 136.1k** |
| max-autotune | as built | 0.276 / 0.276 | 119.7k / 119.3k |
| max-autotune | shared context | 0.214 / 0.209 | 132.1k / 137.1k |

* **The two gains compose.** Under compile the shared-context head still takes ~0.065 s off the
  update, as it does eagerly (0.074 s): compile fuses the elementwise work but does not remove the
  84-fold duplicated matmul columns.
* **Together: update 0.364 → 0.213 s (−41%), +32% end to end.** Compile alone and the head alone
  are each worth ~+15%.
* **max-autotune buys nothing over default** here (within noise), at a much longer first compile.
* The rollout (0.27 s) is untouched by either and is now ~56% of the iteration.
