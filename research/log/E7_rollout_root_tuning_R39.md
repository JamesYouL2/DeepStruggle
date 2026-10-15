# Which searcher recovers R39's placement leak, and tuning the rollout root on R39 (2026-10-13)

**Question (owner).** [The R39 leak](E7_leak_by_kind_R39.md) still puts influence placement first
(+11.2 ± 3.3 points a game, 61%). Experiment 1: does any searcher recover it? Then use the same
measurements to tune the rollout root.

**Answer.**

* **The deployed rollout root already recovers the whole placement leak.**
  * Rollout root k = 4, 16 worlds, horizon 4, rule z2 (`4:16:4:z2`): +11.8 ± 3.3 a game on the
    large-regret placement rows.
  * The reference search's best move scores +11.2 ± 3.3 on the same rows. Gumbel@256 recovers +9.4.
  * So placement is a leak of R39 raw, not of the searched player: no searcher is missing here.
* **More candidates do not help, as the recall check predicted.**
  * On R34, the reference's best placement point was inside the network's top 4 for ~75% of
    large-regret departures (by weight), and nearly all the gain was there.
  * Here k = 8 is −0.3 ± 1.5 and k = 16 +0.2 ± 1.4 against k = 4. More worlds add +1.5 to +1.7.
* **Offline, across every decision kind, more worlds and a looser choice rule help; depth and width
  do not** (large-regret rows, against `4:16:4:z2`):
  * 64 worlds +11.5 ± 2.0, at 4× the network rows;
  * 32 worlds with rule z1 +10.2 ± 2.2, at 2×;
  * 32 worlds, argmax +9.1 ± 2.4;
  * 32 worlds, z2 +6.4 ± 1.5;
  * k = 6 with 32 worlds +6.5 ± 2.1;
  * k = 8 or 16: about 0;
  * horizon 8: −5.0 ± 2.1;
  * Gumbel@256: −19.2 ± 5.3.
* **In games, 32 worlds at z2 is the tuned rollout root: +41 ± 8 Elo over the deployed `4:16:4:z2`**
  on R39, at twice the network rows ([below](#tournaments)).
  * 64 worlds adds nothing more (+37 ± 8).
  * The looser z1 rule loses the gain (−3 ± 8 at 32 worlds, +17 at 64), though it looked best
    offline. It departs from the network at 30% of positions, against 18% for z2, and the extra
    departures fall where the playout judge is only noise.
  * The offline judge ranked the worlds axis rightly and the rule axis wrongly. It has misranked
    searchers before (Gumbel@1024 above @256, `E7_search_depth_and_value.md`).

## Experiment 1: placement

On the 106 large-regret placement rows (population-weighted, 512 paired continuations a move, R39
greedy both sides), gain over R39 raw a game:

| searcher | gain a game | against `4:16:4:z2` | departs from raw |
|:---|---:|---:|---:|
| rollout `4:16:4:z2` (deployed) | **+11.8 ± 3.3** | — | 41% |
| rollout `4:32:4:z2` | +13.3 ± 3.2 | +1.5 ± 0.7 | 49% |
| rollout `4:64:4:z2` | +13.6 ± 3.2 | +1.7 ± 0.7 | 52% |
| rollout `6:32:4:z2` | +13.4 ± 3.3 | +1.5 ± 0.7 | 48% |
| rollout `8:16:4:z2` | +11.5 ± 2.8 | −0.3 ± 1.5 | 45% |
| rollout `16:16:4:z2` | +12.1 ± 3.8 | +0.2 ± 1.4 | 57% |
| rollout `4:16:8:z2` | +11.9 ± 3.1 | +0.1 ± 0.8 | 31% |
| rollout `4:32:4:z1` | +13.0 ± 3.3 | +1.1 ± 0.6 | 59% |
| rollout `4:32:4:argmax` | +13.5 ± 3.3 | +1.7 ± 0.8 | 67% |
| Gumbel k = 8 @256 | +9.4 ± 2.6 | −2.4 ± 1.6 | 74% |
| the R34 soup + `4:16:4:z2` (R39's teacher) | +6.3 ± 3.4 | −5.6 ± 2.3 | 60% |
| the R34 soup raw | +1.1 ± 1.0 | −10.7 ± 3.3 | 30% |

The reference best move, from the leak's own playouts (1,024 pairs, another seed): +11.2 ± 3.3.

* **What this means for training.** The placement leak is what R39 raw has not absorbed from a
  searcher that already fixes it. R39 trained against the soup's rollout root, and that teacher
  recovers only about half of it on R39's positions (+6.3). R39's own rollout root recovers all of it.
* **The whole-placement search on `exp/placement-search`** (one decision for a complete allocation)
  is not needed for this leak, so it was not ported.

## Tuning: every decision kind

The same run, all 1,352 large-regret rows, gain a game against `4:16:4:z2`:

| searcher | network rows a search | against `4:16:4:z2` | departs | reproduced by another seed |
|:---|---:|---:|---:|---:|
| `4:16:4:z2` (deployed) | 885 | — | 29% | 65% |
| `4:64:4:z2` | 3,539 | **+11.5 ± 2.0** | 41% | 74% |
| `4:32:4:z1` | 1,770 | **+10.2 ± 2.2** | 49% | 63% |
| `4:32:4:argmax` | 1,770 | +9.1 ± 2.4 | 64% | 70% |
| `6:32:4:z2` | 2,287 | +6.5 ± 2.1 | 35% | 62% |
| `4:32:4:z2` | 1,770 | +6.4 ± 1.5 | 34% | 69% |
| `8:16:4:z2` | 1,324 | −0.8 ± 2.5 | 31% | 59% |
| `16:16:4:z2` | 1,591 | −0.8 ± 2.2 | 33% | 52% |
| `4:16:8:z2` | 1,738 | −5.0 ± 2.1 | 20% | 37% |
| Gumbel k = 8 @256 | 252 | −19.2 ± 5.3 | 61% | 48% |
| the soup + `4:16:4:z2` | 892 | −10.9 ± 4.8 | 44% | 75% |

"Departs" is on the large-regret rows; "reproduced by another seed" is over all 2,423 positions.
Network rows are per searched position, averaged.

* **Worlds are the lever.** More worlds raise both the gain and the reproducibility (65% → 74% at
  64). This matches R32's Experiment 2, where 32 worlds was +22 Elo over 16 in games and still rising
  (`research/log/R32_rollout_cpu_experiments.md` on `pr/r32-search`).
* **A looser rule gains at large-regret positions.** z1 at 32 worlds matches 64 worlds at z2 for
  half the rows. But it departs from the network at 30% of all positions against 18%, and outside
  the large-regret rows its gain cannot be read (+3.9 ± 24.0 against `4:16:4:z2`). The games
  ([below](#tournaments)) say those extra departures lose: z1 gives back the whole gain of the
  extra worlds.
* **Width does not help and depth hurts.** k = 8 and k = 16 add nothing, as on placement. Horizon 8
  departs half as often and is −5.0: longer greedy rollouts blur the difference between candidates.
* **Gumbel@256 now trails the rollout root by 19 points a game** on the rows where it should be
  strongest.

## Tournaments

R39 with each candidate against R39 with `4:16:4:z2`: 2,000 games each (1,000 a side), temperature
0, draws as half. Seat columns are the candidate's win rate in that seat.

| candidate | CI run | score against `4:16:4:z2` | Elo | as US | as USSR | network rows |
|:---|:---|---:|---:|---:|---:|---:|
| **`4:32:4:z2`** | `38393748928` | **55.9% ± 1.1** | **+41 ± 8** | 63.2% | 47.7% | 2× |
| `4:64:4:z2` | `38393765155` | 55.3% ± 1.1 | +37 ± 8 | 62.2% | 47.6% | 4× |
| `4:64:4:z1` | `38393795017` | 52.5% ± 1.1 | +17 ± 8 | 58.9% | 45.2% | 4× |
| `4:32:4:z1` | `38393779853` | 49.5% ± 1.1 | −3 ± 8 | 56.0% | 42.4% | 2× |

* **Worlds: 16 → 32 is +41 Elo; 32 → 64 is nothing** (−4 ± 11, the two read against the same
  reference). R32's Experiment 2 found +22 for 32 worlds on R32.
* **Rule: z2 beats z1 by ~40 Elo at either width.** The offline grid had z1 ahead on large-regret
  rows (+10.2 against +6.4). In games, the departures z1 adds where the evidence is thin cost more
  than its extra corrections gain.
* **Follow-up:** at 32 worlds, the stricter z3 and z2.5 against `4:32:4:z2`, 1,000 a side, were
  launched (CI `38421746865`, `38421768420`); their result is not recorded here, and `4:32:4:z2`
  stayed the deployed root ([`E7_searched_leak_R39.md`](E7_searched_leak_R39.md)).

Tables: [`4:32:4:z2`](E7_rollout_root_tuning_R39/tournament_w32z2.md),
[`4:64:4:z2`](E7_rollout_root_tuning_R39/tournament_w64z2.md),
[`4:32:4:z1`](E7_rollout_root_tuning_R39/tournament_w32z1.md),
[`4:64:4:z1`](E7_rollout_root_tuning_R39/tournament_w64z1.md).

## Method

* **Bank.** The R39 leak's playout sample: release `playout-input-r39-20261013`, 2,423 positions
  (1,352 large-regret, 771 others, 300 agreement controls), each with its population weight and
  inclusion probability.
* **Search and playouts.** `tools/search_reliability.py` (`search_reliability.yml`), CI
  `38384383842`:
  * every position searched by each searcher with two seeds;
  * each position's raw move and every pick played out in 512 paired continuations by R39 greedy on
    both sides, one redeal and one set of dice per pair shared by every move.
  * Searchers on the soup (`@2:`) search the same R39 positions with the R34 soup's network.
* **Report.** `report --by-kind` (commits `4c62af1`, `c76edac`): the table per decision kind and on
  the large-regret rows. The full output is [searchers_on_the_bank.md](E7_rollout_root_tuning_R39/searchers_on_the_bank.md).

## What this does not say

* **The judge is R39 greedy**, the same network the rollouts use. A searcher whose rollouts look like
  the judge's continuations is favoured. More worlds is exactly such a searcher, so the tournaments
  are the test.
* **"Recovers the whole leak" is relative to the reference search's move**, not to perfect play.
