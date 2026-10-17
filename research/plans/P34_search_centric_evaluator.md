# P34 — the search-centric evaluator: a leaf critic trained on playout-labelled positions (2026-10-17)

**Goal.** A player much stronger than today's (the owner's figure: ~2500 on the E7 scale, against
1852 for the best searched player and 1682 raw), by changing what the evaluator is trained on. The
deployed player is the network under the rollout root; the root keeps the critic and re-decides the
policy ([P33](P33_search_preserved_gains.md)), and every critic measured is at chance exactly where
the searched player still loses ([`../log/P33_diagnostics.md`](../log/P33_diagnostics.md), P33c).
This plan trains a **second network**, read only at the root's leaves, on positions **labelled by
playouts** rather than by one game outcome each -- the way chess evaluators were trained on search
scores -- and iterates it by generation.

**Why this and not more of the same.** Every single big lever has been pulled once on the RL loop
without moving its plateau (capacity, tokens, human data, the league, search as a policy teacher,
turn credit; P33's opening section). The leaks the searched player keeps are things no evaluator sees
within four action rounds and some only at a later scoring; the search's own gain keeps rising with
worlds (+90 / +177 / +214 at 4 / 16 / 64) with no ceiling in sight. So the ceiling is the evaluator's
horizon of understanding, and the critic's training signal -- a λ-return of sampled self-play, one
outcome per position with a standard deviation near 0.5 against candidate differences of 0.04 -- is
what holds it there. A 32-world playout label has a standard error near 0.09 and is unbiased for the
greedy value: per label about 30× the information of a game outcome, so a million labels is roughly
twice the value information of the whole RL run, concentrated on decision states.

**What this plan is not.** Not policy distillation of search (seven forms null), not search tuning,
not a representation change: the owner holds history out of scope for now -- the network misplays on
current-state information alone, so the evaluator reads the same 3,824-float observation.

## Nomenclature (owner, 2026-10-17)

Four new kinds of thing appear here; one new letter.

* **`D<n>`, a labelling recipe** -- the one new code: the source policy's temperature, decisions kept
  per game, worlds, the stop (game end, turn end), the label columns and the device. A flag set, like
  an R code, registered with the other codes once `tools/scripts/run_codes.py` knows the labelling
  tool; until then defined in [`../datasets.md`](../datasets.md).
* **A dataset instance** is a model under a labelling recipe, spelled like a player under a search:
  `<generator model>~D1(seed=44,games=20000)`. The generator played the positions and the playouts, so
  the generation is in the name. Instances are ledgered in [`../datasets.md`](../datasets.md): size,
  rows, hashes, where the files live.
* **An evaluator** is a trained network: an **A code** for its shape (value-only, so new codes) and an
  **R code** for the supervised recipe, which pins its dataset the way R32 pins a teacher checkpoint.
  From scratch `E7-A<n>-R<n>-S44`; fine-tuned from the RL network's weights
  `<RL model>+A<n>-R<n>`, the branch grammar A13 used. Snapshots, SWAs and soups keep the model grammar.
* **A two-network player**: the leaf network is a parameter of the search spec,
  `<policy model>~rollout(k=4,worlds=16,horizon=4,rule=z2,stop=earlier,leaf=own,leaf_net=<evaluator>)`
  (`leaf=` is already the leaf's *side*; `leaf_net=` is the network that reads it). The leaderboard
  shows the full id; a display alias is a page matter.
* **Feeding the RL side** is a recipe code on the RL run, pinning the evaluator it reads.
* **Plan steps** that train nothing are `P34a`, `P34b`, …; every fit is an A+R pair.

A generation then reads as one chain: the RL model names the dataset, the dataset names the evaluator
through its R code, the evaluator names the player through its search spec, and the next RL recipe
names the evaluator. No generation counter is needed.

## The dataset (generation 1) -- `P34a`

* **Positions.** Greedy self-play of the best raw network (A13+R48's SWA, `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M`)
  at temperature 0.25 for diversity, 20,000 games, one decision in eight kept: ~1.1M decision
  positions. Held out by game, 5%. With `leaf=own` a leaf *is* the mover's own next decision, so
  decision states are the leaf distribution, read in the decider's perspective the critic is trained in.
* **Main labels (D1).** From each position, 32 worlds dealt from the mover's view
  (`ai.search.dmcts.determinize` / `ts.determinize`), fresh dice per world, both sides played greedily
  by the generator to the game's end through the safe-playout rule (`ai/eval/paired_playouts.play_safe`:
  no on-the-spot suicide while another option exists in the mover's round). Per world, averaged: the
  mover's score (1 / ½ / 0), the VP margin at the end (/20), VP at the current turn's end, DEFCON at the
  turn's end, each region's VP swing at its next scoring (masked where it is never scored). ~8,000
  network rows a label; ~9 × 10⁹ rows in all.
* **Sibling slice.** For 50,000 of the decisions, the policy's top four candidates each played in the
  same 16 worlds; the leaf after each candidate (the mover's own next decision) captured as a position
  of its own with that world's outcome. Gives the fit a paired loss on sibling leaves of one world,
  gives evaluation a paired accuracy, and covers leaves after moves the policy would not play -- which
  the search visits and self-play does not.
* **Exact slice.** 2,000 turn-10 last-round positions solved exactly (`tools/scripts/blunder_bank.py
  prove`, dice averaged, the opponent's best reply). Not a training set: the playout label against the
  exact value on the same positions is the **bias of greedy follow-up play**, the number that says how
  much the labels inherit from the generator's own mistakes.
* **Where it runs.** On the GPU, two ways. *Now:* the Python playout loop with a CUDA policy, ~200–300k
  rows/s a process (the loop's ~9 µs a row on one core is the limit; measured 2026-10-17: 15k rows/s a
  CPU core, 34k on 4 threads, flat beyond), several processes sharing the GPU, at a proportional cost
  to a training run sharing it. *To build:* a game-end stop with outcome capture in the C++ rollout
  batch (`bindings/rollout_batch.hpp`; a bindings change, **the owner's approval**) -- inference-bound
  at batches ≥ 8k, several million rows/s, the engine side ~2M rows/s on this machine's cores:
  **about an hour for the generation-1 set.** The same path is what the reverse curriculum needs
  (a stop at the turn's end, read by the evaluator). 20 CI runners (~600k rows/s, ~4 h) remain the
  fallback and the home of the exact slice. GPU labels are not bit-identical to CPU ones (near-tie
  argmaxes); the dataset records device and seed.
* **Reads before any fit.** The exact-slice bias (mean and by DEFCON / side); label SE against the
  32-world budget; the held-out split's independence (no position token in both halves).

## The evaluator -- fits as A+R pairs

* **Architectures, all on the same dataset** (each an A code, value-only: no policy heads):
  the current ladder trunk fine-tuned from the RL weights (`<RL model>+A<n>-R<n>`); the same shape from
  scratch; the deep variant (8 residual blocks at width 768, ~11M parameters, A3's shape); the token
  transformer (A6's shape). Supervised training of even the largest is under an hour of GPU, so this
  is the one place architecture is compared cheaply -- **and the first clean capacity test the project
  can run**: if the deep variant fits what the shallow one cannot, capacity pays on a rich signal.
* **Recipe (R).** Cross-entropy on the mover's win probability, squared error on the VP targets, the
  auxiliaries at a small weight, a paired-difference loss on sibling leaves of one world at 10× weight
  (the value probe's finding: the paired term is what lifted sibling correlation, 0.25 → 0.59), early
  stopping on held-out games. A supervised value trainer over the dataset (`tools/train.py --mode
  value-fit`, or `tools/value_probe.py` extended from head-only to whole-net), with the recipe's flags
  as its code.
* **Reads, in order.** (1) Held-out label fit against the RL critic's fit on the same positions.
  (2) Sibling paired accuracy on the held-out slice. (3) P33c's leak-pair ranking on the confirmed
  pairs (`ai/eval/leak_critic.py`), which the training set must not contain. (4) **The only number
  that counts** (`P34b`): the rollout root with the evaluator at the leaf, `stop=earlier, leaf=own`,
  16 worlds, 1,000 deals a side against the best searched player (the R34 soup under the same root,
  1800 at 16 worlds), per seat by the rule; then on the leaderboard with `--strong`. ~25 min a read.
* **Decision rule.** The evaluator becomes the product leaf when its searched player beats the current
  best searched player by the per-seat rule. If it fits the labels but the games do not move → the
  labels are biased by the follow-up (the exact slice says how much) and the fix is on the label side.
  If it does not fit the labels → capacity: the next architecture up. If the deep variant fits and the
  shallow does not → the capacity result, recorded as such.

## Generations

* **Generation k.** Positions and playouts from the RL network of generation k; the evaluator trained;
  the searched player measured. The playout policy is always the small network -- the evaluator never
  plays -- so labelling cost does not grow with the evaluator.
* **Feeding the RL side** (recipe codes on the RL run; measured separately, each against the unchanged
  control, 400M from the current plateau). (a) A squared-error term toward the evaluator's reading on a
  subsample of rollout positions (the evaluator forward on, say, 1 in 16 rows): sharper GAE
  advantages, the route by which the *policy* can move past its plateau. (b) The RL critic's head
  initialised from the dataset, then resumed. **Risk:** the shared trunk -- a value-target shift
  perturbs the policy, and R44 showed how fast a resumed network snaps back -- so the first cell runs
  at a low weight under the usual stability panel (entropy, KL to π_ref, pool win rate), read raw and
  searched.
* **Reverse curriculum (from generation 2).** Once the evaluator is trusted from turn t on, positions
  in turn t − 1 are labelled by one-turn playouts read by the evaluator at the turn's end instead of
  game-end playouts: ~3,000 rows a label against ~16,000 at 64 worlds. The exact slice anchors turn
  10. Game-end labels on a 100k sample stay as the check that the curriculum loses nothing. This is
  where the placement leaks, which resolve only at a later scoring, should start to move.
* **Stopping.** The programme continues while each generation's searched player beats the previous
  one. A level generation whose evaluator still fits its labels means the labels have hit the
  follow-up's ceiling; the next lever is then a stronger playout policy on a smaller set (the sparse
  searched player at 10% of moves, ~90× a label's cost), not a bigger evaluator.

## What has to be built

1. **The labelling tool** (`tools/value_labels.py`): stages `positions`, `label`, `merge`, shardable
   over CI like `tools/search_bank.py`, reusing `paired_playouts` with outcome capture; the dataset
   written with its D code, generator hash, device and seed.
2. **The C++ game-end stop with outcome capture** in the rollout batch -- the bindings change waiting
   on the owner; the Python path is the interim.
3. **The supervised value trainer** with the paired loss and the architecture choices, writing a
   checkpoint the rollout root can load as a leaf network.
4. **`leaf_net=` in the player grammar** (`tools/lib/player_spec.py`; P33a already runs the
   two-network root from a spec string) so the leaderboard can rate a two-network player.
5. **A value-distillation term** in the RL trainer, for generation 2.
6. **`D` codes in `run_codes.py`** and the [`../datasets.md`](../datasets.md) ledger.

## Costs (generation 1)

| step | cost |
|:---|---:|
| positions, 20,000 greedy games | minutes (GPU) |
| main labels, 1.1M × 32 worlds | ~1 h GPU on the C++ path; ~4 h on 20 CI runners; ~1 day on this CPU |
| sibling slice, 50k × 4 × 16 | ~0.3 of the main set |
| exact slice, 2,000 positions | CI, the prover's budget |
| each evaluator fit | < 1 h GPU |
| each games read | ~25 min GPU |

## Decision rules, collected

* `P34a`: the exact-slice bias is reported before any fit; a bias above ~5 points of win probability
  on average puts the label-side fix (a stronger playout policy on a subset) ahead of the fits.
* Fits: ranked by held-out fit, sibling accuracy and the leak-pair ranking; only the best two go to
  games.
* `P34b`: adopt the evaluator as the product leaf by the per-seat rule against the best searched
  player; the RL-side term is adopted by the same rule read raw and searched, with the stability panel.
* Generations continue while the searched player improves; the curriculum is validated against
  game-end labels on a 100k sample each generation.

## Runs

(evaluators mirror [`../runs.md`](../runs.md), datasets [`../datasets.md`](../datasets.md))

**What was built (2026-10-17).** The C++ labeller (`bindings/playout_labels.*`; every world equals a
step-by-step Python `play_safe` playout of it) with region swings from the engine's `ScoringTap`;
`tools/value_labels.py` (`positions` / `label` / `exact` / `merge`, D codes derived by
`run_codes.py`); `tools/train.py --mode value-fit`; `leaf_net=` in the player grammar. The labeller
runs at ~1.1M network rows/s on this machine (two labellers alternating, a CUDA graph, exact
one-byte observation codes); the engine's observation extraction is the limit.

**`P34a`, generation 1.** `…+A13@11920..12000M~D1(seed=44,games=20000)`: 1,115,626 positions from
20,000 games (31 s), labelled in 2 h 22 min (9.4 × 10⁹ network rows); label standard error 0.064 at 32
worlds; 2,794,624 sibling rows.

* **Exact slice:** 925 of 2,000 turn-10 last-round positions solved in all 4 worlds; label − exact
  **−1.3 ± 0.4 points** of win probability (US mover −2.1, USSR +1.5), mean |gap| 2.8, r = 0.96 --
  under the 5-point rule, so the labels are used as they are. The solvable positions are mostly
  decided (888 of 925); on the 37 contested ones greedy follow-up loses 16 points to exact play,
  which is where the labels inherit the generator.

**The fits** (held out by game: 56,193 positions; label noise, mean SE², 0.0050 of the 0.0925 label
variance):

| evaluator | held-out score MSE (excess over noise) | cross-entropy | sibling pair accuracy |
|:---|---:|---:|---:|
| the RL critic as it is (step 0) | 0.0111 (0.0061) | 0.5006 | 0.519 |
| **`…+A13@12000M+A15-R52`**, fine-tuned | **0.0093 (0.0043)** | **0.4957** | 0.533 |
| `E7-A15-R53-S44`, same shape from scratch | 0.0128 (0.0078) | 0.5059 | 0.528 |
| `E7-A16-R53-S44`, deep (12.4M parameters) | 0.0127 (0.0078) | 0.5060 | 0.525 |
| `E7-A17-R53-S44`, token layers | 0.0127 (0.0077) | 0.5054 | 0.515 |

The RL critic already explains ~93% of the label variance net of noise; fine-tuning removes 30% of
what is left. A million labels do not train an evaluator from scratch to the RL critic's level, and
neither added capacity nor tokens helps there: **no capacity signal at this data size**. The
sibling pair accuracy is near chance for all -- the siblings' single playouts share the deal but
not their dice once the moves differ (correlation 0.22), so the paired target is mostly noise.

**`P34b`.** A13+R48's root (16 worlds, `stop=earlier`, `leaf=own`) with its leaves read by the
fine-tuned evaluator, strong tier: **1804 ± 4**, against **1794 ± 3** for the same root on its own
critic -- +10, about two standard errors. Real and small, in line with the label fit.

**Keyed chance** (`bindings/keyed_chance.hpp`, from this result): each world's dice and deals tied to
the situation they are drawn in, so siblings stay paired longer. Within-world correlation of two
candidates' results on 3,000 slice positions: 0.22 shared stream, 0.33 keyed dice, **0.39** keyed dice
plus one deck order per world; the mean score unchanged. The keyed root's strong-tier read is the
next result.
