# Racing the rollout root's worlds (2026-10-26): 64 worlds' strength for 1.5x the cost of 32

**Question.** The deployed rollout root (`4:32:4:z2`, `ai/search/rollout_root.py`) plays every one of its
four candidates in every one of its 32 worlds. Most challengers are clearly behind the network's own
move long before 32 worlds, and z2 almost never picks a challenger that trails it halfway. Can the root
stop playing those challengers early without changing what it picks?

**Answer.** Yes. Two stages gain as much over the network's move as the deployed root, to within
−0.002 ± 0.002 points ×100 a decision on held-out worlds and departing from it as often (4.8%), for 106
playouts a decision instead of 128 (−17%). That is the gain, not a count of identical picks, which the
reliability rows were not scored for:

1. Every candidate plays the first 16 worlds.
2. A challenger whose paired lead over candidate 0 is below 0 leaves the race.
3. The other 16 worlds are played only by candidate 0 and the challengers left.
4. z2 reads every world.

This is now `RolloutBatch.race` (`bindings/rollout_batch.hpp`), shared by Python and the web page, and
is spelled `race=16` in a player id and `:race16` in a spec. It is off by default, so every measured
player is unchanged.

The same rule at 64 worlds (16 → 64) costs 191 playouts, against 256 for 64 worlds uniform, and gains
the same as 64 worlds uniform. In games on A13+R48 (2,000 games a side each, `earlier:own`), 64 worlds
beats 32 by ~12 Elo, and **race 16 → 64 keeps all of it**: 51.6% against 4:32, where 4:64 scores 51.8%.
It spends ~1.5× the time of 4:32, against ~2× for 4:64. Race 16 → 32 plays even with 4:32 (49.0 ± 0.8%,
1.2 SE) for ~83% of its time.

## Method

The data are the sibling rows of the P34 reliability run: CI `40373298241` on the fork, branch
`exp/p34-reliability` (see `research/log/E7_p34_d3_worlds.md` on branch `pr/p34-d3-closed`).

* **Decisions.** 3,207 decisions of the A13 network: 2,321 entry decisions and 886 uniform ones. Each
  had four candidates and at least 120 of its 128 keyed worlds valued, at horizon 4 with greedy
  follow-up, the root's own leaf.
* **Scoring.** A policy selects on one parity of worlds (at most 64) and is scored on the other 64:
  the held-out mean of its pick minus candidate 0's, ×100 a decision. Both parities are averaged.
* **Uncertainty.** "vs deployed" is the paired difference against `4 × 32, z2`, with a bootstrap SE
  over decisions (400 draws).
* **Cost.** A playout is one candidate in one world.

| policy | playouts | departs | gain over the network | vs deployed |
|:---|---:|---:|---:|---:|
| 16 worlds, z2 | 64 | 4.0% | +0.040 | −0.077 ± 0.024 |
| **32 worlds, z2 (deployed)** | 128 | 4.8% | +0.118 | 0 |
| 48 worlds, z2 | 192 | 5.0% | +0.150 | +0.033 ± 0.020 |
| 64 worlds, z2 | 256 | 5.3% | +0.169 | +0.051 ± 0.024 |
| halving 16 + 32 (best challenger only) | 128 | 4% | +0.127 | +0.009 ± 0.021 |
| **race 16 → 32** | **106** | 4.8% | +0.116 | **−0.002 ± 0.002** |
| race 8 → 32 | 91 | 4.6% | +0.114 | −0.004 ± 0.006 |
| race 16 → 48 | 148 | 5.5% | +0.152 | +0.034 ± 0.020 |
| race 16 → 64 | 191 | 5.2% | +0.165 | +0.047 ± 0.024 |

Other variants did no better than the two-stage race:

* checking the leads every 8 worlds (103 playouts at cap 32, the same picks);
* a stricter drop (a challenger leaves only when lead + 1 SE < 0: 118 playouts, no better);
* accepting a challenger early at z 2.5 or 3 (the picks only got worse).

So the race is one check, at `race` worlds.

## In games

The network is A13+R48's SWA (`E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13_20261014_105711/swa_11920-12000M.pt`).
The fork's `searcher_tournament.yml` ran three matches at 2,000 games a side, temperature 0, on 40 runners.
Each match is against `4:32:4:z2:earlier:own`. A draw counts as half a win.

| entrant | CI run | score vs 4:32 | Elo | match wall time |
|:---|:---|---:|---:|---:|
| `4:64:4:z2:earlier:own` | `40409329043` | 51.8 ± 0.8% | +12 | 5,734 s |
| `4:64:4:z2:earlier:own:race16` | `40432479373` | 51.6 ± 0.8% | +11 | 4,826 s |
| `4:32:4:z2:earlier:own:race16` | `40432490347` | 49.0 ± 0.8% | −7 | 3,491 s |

The report gives no per-player timing. Assume the 4:32 player's share of a match is the same every
time, and that a player's time scales with its worlds, so 4:64 takes twice 4:32's share. The first
match then puts 4:32 at ~1,911 s. The racers come out at about:

* race 16 → 64: ~2,915 s, which is 1.53× 4:32 and 0.76× 4:64;
* race 16 → 32: ~1,580 s, which is 0.83× 4:32.

Those are the playout ratios measured offline. The estimate is rough, because CI runners differ.

The race 16 → 32 score of 49.0% is 1.2 SE below even. Offline, its picks match 4:32's to
−0.002 ± 0.002, so this is most likely noise. A cheaper 4:32 still costs nothing to keep.

## What was built

* **`RolloutBatch::race(position, worlds[, hands])`.** Call it once the position's rows are valued.
  * It drops the challengers whose lead is below 0. A lead of exactly 0 stays.
  * It adds `worlds` more worlds for the candidates left. Their world indices continue, and the
    stream continues the batch's own (or uses `hands`, for a belief).
  * If no challenger is left, it adds nothing.
* **`decide`.** It picks among the in-race candidates, over every world they were all valued in.
  * A dropped candidate keeps its statistics on its own worlds and is never picked.
  * If no candidate is dropped, a raced batch of one position is the unraced batch at the full world
    count, row for row and decision for decision (`tests/bindings/test_rollout_batch.py`). In a batch of
    several positions the race's worlds are drawn after every position's first ones, so they are other
    worlds, and the two batches agree only in distribution.
* **Python.** `RolloutConfig.race`, the spec token `:race<n>`, the player id `race=<n>`
  (`tools/lib/player_spec.py`), and `last_stats["dropped"]`.
* **Web page.**
  * `ts_rollout_race` / `ts_rollout_in_race` in the WebAssembly build.
  * The two-stage loop in `web/ui/src/analysis/rollout.ts`, read from the `search=` value's `:race<n>`.
  * With a belief, all the worlds' hands are dealt in one draw, as Python's root deals them.
  * `tests/web/test_wasm_engine.py` holds a raced batch bit-identical across the two builds.

## Caveats

* **A survivor's statistics are read on the worlds that selected it.** A challenger stays in the race
  because its lead on the first worlds was not below 0, and `decide` then reads its lead over every
  world, those first ones included, so a survivor's lead and z read a little high. The offline scoring
  selects on one parity of worlds and scores on the other, so the gains above already pay for it; but
  `last_stats["lead"]` of a raced root is not an unbiased estimate, and a dropped candidate's statistics
  cover only the first worlds. `RolloutTeacher`, whose targets read every candidate's Q over every
  world, refuses a raced config (as `distill_targets.py`'s `root:` teachers do).

* **The saving was planned in candidate-worlds.** A playout's network rows depend on how long it runs,
  and the root forward and the leaves are unchanged. The tournaments' wall times track the playout
  ratios anyway (*In games*), but only as a rough estimate.
* **The second stage is a second round of playouts** after the first is valued. In a batch over many
  positions (the tournament's), both stages are batched across the positions. In the page, one
  position is searched at a time, so the second stage is a second, smaller loop.
* **One network, one opponent.** The game results are A13+R48 against 4:32 only. 64 worlds did not
  separate from 32 on R39 (55.3% vs 55.9% against 4:16), nor clearly on the E7 leaderboard (1839 vs
  1832, ±4).
