# Search performance profile (2026-10-03): what takes the time in search evals and search training

The owner asked for this before optimising search ("benchmark what takes time in running evals and
training with search"), after search was found to add about +10 points on the best model
([`P30_search_headroom_e7.md`](P30_search_headroom_e7.md)).

**Method.** Sampling profiles with py-spy (`--native`, so time in the C++ engine and in torch is
attributed to the Python line that called it), on the real workloads. Engine 1d11c2f2. Profiles and
logs are in the job's scratch directory and are not kept.

* **Eval:** the soup against `search:<soup>:256:determinize`, 200 games per side, `--pack-pairs 1`
  (the configuration of the search-headroom measurement): about 3 minutes, 21,478 samples on the
  decision path.
* **Training:** `tools/train.py` on the shallow recipe, cold start, with
  `--search-ce-coef 0.1` (defaults: 32 simulations, `card_playmode`, 1 in 8) against the same command
  without search. About 5 minutes each, written to scratch under throwaway names
  (E7-98-44, E7-99-44) and discarded.

## Eval: CPU-bound Python tree work

GPU utilisation averaged **6%**. One search of 200 positions at 256 simulations takes 6.0 s
(30 ms per position, about 23 ms per simulation round of 200 leaves).

| where | share of the decision path |
|:---|---:|
| `_select` (PUCT over the children, in Python) | **37.0%** |
| `_featurise` -- of which `refresh_all` 20.4% | **20.8%** |
| `_evaluate_batch` outside the forward (per-node priors, transfers) | 11.3% |
| `_descend` (clone, step, settle a new child) | 8.6% |
| `_backup` | 8.0% |
| network forward (`_encode`, heads) | about 7% |
| everything else | about 7% |

* **Selection is a Python loop over about 19 children** (visit-weighted branching 18.6; leaves at
  mean depth 6.3, max 45). It re-sums every child's visit count on every visit (lines 226–227, 7.6%
  on their own).
* **`refresh_all` rebuilds all 4,096 slots of the featuriser for a batch of about 200.**
  `BatchedMCTSAgent` sizes the featuriser at 4,096. Measured with 200 states set: 8.25 ms per call at
  capacity 4,096, 6.10 ms at 1,024, and 2.09 ms at 256. That is about 6 ms of a 23 ms round spent on
  empty slots.

## Training: about 8× slower, small batches and per-leaf Python

Throughput, same harness, early in a cold run: **57–62k steps/s without search, 7.1–8.1k with
it**. The searcher answers about 25 positions per environment step (512 environments × ~42% card or
play-mode decisions × 1/8), at 32 simulations, so every environment step makes 32 network calls of
about 25 leaves each.

| where | share of the training loop |
|:---|---:|
| `_featurise` on the **per-node path** (`extract_observation` + `get_legal_mask` per leaf) | **19.7%** |
| network forward at tiny batches (`_encode`, `_policy_logits`, value, entity heads) | about 32% |
| `_evaluate_batch` outside the forward | 16.0% |
| `_select` + `_descend` + `_make_node` + `_backup` | 11.2% |
| the PPO update and rollout proper (`train_step`, `collect_rollouts`, env step) | about 10% |
| everything else | about 11% |

* **The training searcher has no C++ featuriser.** `NashPGTrainer` builds `BatchedMCTS` with the
  default `featurise_capacity=0`, so every leaf is featurised one at a time from Python, which the
  searcher's own docstring prices at about 10× the batched path.
* **The forward runs at batch ~25**, where it is launch latency, not compute (the searcher's
  docstring: 0.825 ms at batch 1 against 2.19 ms at batch 512).
* **The search does not need to happen at the step.** It only produces targets; the actions were
  already sampled from the raw policy and the weights do not change during a rollout. So the
  positions could be cloned during the rollout and searched once at its end: about 3,200 positions
  per rollout (25 × 128 steps) in one batched search instead of 128 searches of 25.

## Reading: what an optimisation can buy

* **Python-only fixes (no engine or bindings change):**
  * size the featuriser to the batch (eval), and give the training searcher one;
  * keep each node's visit total incrementally;
  * extract priors for the whole batch at once;
  * in training, defer the search to the end of the rollout.

  Expected about 1.5–2× on the eval search, and about 2× on the search part of training (the
  per-leaf Python floor of ~30–40 µs remains).
* **Moving the tree into C++** (selection, expansion, clone/step/settle and featurisation into one
  buffer; Python only runs the network on each leaf batch) removes the dominant cost in both
  profiles. Rough expectation: 5–10× on the eval search, and training with search at about 5× its
  current throughput (from ~7.5k towards ~35–40k steps/s at 32 simulations, 1 in 8). It is a new C++
  component next to the engine, so it needs the owner's approval.

## After the optimisation (2026-10-03, same day)

Three changes, each committed separately on `hand-knowledge-tracking`:

1. **Python fixes** -- featurisers sized to the batch in power-of-two buckets, incremental visit
   totals, priors extracted for the whole batch. Identical visit counts; 200 positions × 256
   simulations 6.5 s → 5.5 s.
2. **Training: search deferred to the end of the rollout.** Positions are cloned at their step and
   searched in one batch of ~3,200 (chunks of 1,024) with the batched featuriser. The weights do
   not change during a rollout and search never acts, so a target is what it would have been at
   its step; `test_deferred_targets_land_on_the_rows_they_describe` checks every target against its
   own row's legal mask. Training with search: 7.5k → 16k steps/s.
3. **The tree in C++** (`ts_engine.BatchedSearch`, `bindings/batched_search.hpp`, the default
   backend). PUCT, expansion, settling and featurisation in C++ across the OpenMP pool; the
   network stays in Python, with two binding crossings per simulation round. It draws chance seeds
   from the caller's `random.Random` (a C++ copy of CPython's MT19937, state handed in and back) in
   the Python tree's order, so it is bit-identical to the Python tree: tests require identical visit
   counts and generator states, and a 400-game search tournament replays the same games.

| measurement | before | after |
|:---|---:|---:|
| 200 positions × 256 simulations (soup, honest) | 6.5 s | **0.27 s** (24×) |
| 3,200 positions × 32 simulations (one training rollout's targets) | -- | 0.61 s |
| eval: soup vs honest search, 400 games | 1,848 s | **189 s** (9.8×) |
| training with search (32 sims, 1 in 8 card decisions) | 7.1–8.1k steps/s | **~36k steps/s** (4.8×; without search 57–62k) |

What remains in the C++ path is the network forward and its transfers (about a third) and the
per-leaf observation extraction, now spread across cores.

### Correction: training throughput without the profiler

The training figures above were taken under py-spy (`--native`), which slowed the unsearched run by
about a third and the searched runs much less, so they understate the baseline. Re-timed without
the profiler, same flags, the same early stretch of a cold run (0.6–2.6M steps); the "before" is
commit 6da56a5 built in a scratch worktree:

| training | steps/s |
|:---|---:|
| no search | **81–90k** (E7-01-44 logged 75–87k over 5–20M) |
| with search, before (Python tree, searched at every step) | **~9.2k** |
| with search, after (deferred to rollout end, C++ tree) | **37–41k** |

So the speed-up is **4.3×**, and search now costs about 2.2× the unsearched throughput, not the
"8×" (before) or "1.6×" (after) a profiled baseline suggested.

## Pinned buffers and pipelined halves (2026-10-04)

Two more changes, both bit-identical between the C++ and Python trees:

* **Page-locked leaf buffers.** C++ writes leaves straight into pinned torch tensors owned by the
  Python driver, so the copies to and from the GPU are asynchronous.
* **Evaluation groups.** From 1,024 trees on, the trees are split into two halves; each half's
  leaves go to the network on their own, and the C++ tree selects one half while the GPU evaluates
  the other. The network's output for a row depends on the batch it is evaluated in (measured on
  the GPU: no row of a split batch matched the whole batch bit for bit), so the grouping is part of
  the search's semantics and the Python tree evaluates the same halves (`search_groups`). Tests on
  CPU and CUDA hold the two trees to identical visit counts and generator states with two groups,
  and a 400-game match gave identical games from both (182-216-2, at an earlier threshold of 128).

The threshold was first 128 trees; halves of ~100 only doubled the network calls (200 × 256: 0.79 s
against 0.27 s as one batch), so it is 1,024. Below it the search is exactly what it was before
the split: the 400-game honest-soup match replays to the original 157-237-6.

All on an idle machine (E7-17-44 finished), unprofiled:

| measurement | single batch, unpinned | **now** |
|:---|---:|---:|
| 200 positions × 256 simulations | 0.27 s | **0.20 s** |
| 3,200 positions × 32 simulations | 0.61 s | **0.26 s** |
| eval: soup vs honest search, 400 games | 189 s | **133 s** (14× the original Python search's 1,848 s) |
| training, no search | 81–90k steps/s (loaded) | **95–97k** |
| training with search (32 sims, 1 in 8 card decisions) | 37–41k | **45–47k** |

Search's own share of a training rollout fell from about 0.92 s to about 0.73 s; the isolated
search of one rollout's positions is now 0.26 s, so most of what remains is outside the C++ tree --
the Python work per queued position (cloning, determinizing each root, the real-mask check when the
targets are written).
