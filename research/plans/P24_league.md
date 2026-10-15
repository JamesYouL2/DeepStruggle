# P24 — an AlphaStar-style league, without a supervised starting point

**Status: shelved (2026-10-09).** Run on E5, E6 and E7 through `tools/scripts/league.py`; the
results are below. Nothing touches `engine/` or the observation. A next attempt starts from
*For the next league*.

## The idea

A pool of past selves cannot punish a weakness it shares, so give the main agent opponents trained
to find its weaknesses: a **main agent** (trains against the league), a **main exploiter** (trains
against the main's newest snapshots, published into the main's pool when it beats the main) and,
optionally, a **league exploiter**. AlphaStar resets exploiters to a supervised policy; we have
none (266 human games, and their BC prior washes out), so exploiters are reset to an early or a
plateau snapshot of the main agent. The driver launches each role as an ordinary
`tools/train.py` run sharing one checkpoint tree; the pool rescans the other roles' directories.

## What the runs showed

| run | outcome | log |
|:---|:---|:---|
| stage 1, from E4-61's ~400M plateau, greedy frozen-target exploiter | late mean −6 (seed 43), +65 (seed 44, 57.9% ± 1.2) | [`P24_stage1_exploiter_collapse.md`](../log/P24_stage1_exploiter_collapse.md) |
| E5-17/18, from scratch at flat temperature | exploits found, published and countered; main level with plain (49.3%) | [`E5_17_league_from_scratch.md`](../log/E5_17_league_from_scratch.md) |
| E5-21 (league from 310M + game-result setup credit) and its E6 replicate E6-04 | accepted: 54.4–54.6% over plain, the US opening repaired; the gain is play after the setup and a calibrated critic | [`E5_21_setup_credit_plus_league.md`](../log/E5_21_setup_credit_plus_league.md), [`E6_04_league_seed44.md`](../log/E6_04_league_seed44.md) |
| E6-13/14, the league on the shallow trunk | +10 Elo on snapshots, 0 on SWAs | [`training_length_and_evaluation.md`](../findings/training/training_length_and_evaluation.md) |
| E7-21/22, from the saturated E7 line at 2,800M | level with its plain control; the exploiter published once in 16 generations | [`E7_league_from_2800M.md`](../log/E7_league_from_2800M.md) |
| `E7-A8-R1-S44@6400M+R19` and the per-seat exploiters R25/R26/R27 | the league published nothing; an alternating exploiter does not exploit; **a US-locked exploiter does** (+11.6 points at 120M over the main's own US self-play rate), a third of it general US skill; distilling it moved the raw network, not the SWA | [`E7_league_exploiters_by_seat.md`](../log/E7_league_exploiters_by_seat.md) |

**Reading.** From a plateau the main agent has little an exploiter can find in a 50M generation,
so the pool stays empty and the league costs ~40% of the main's throughput for nothing.

## For the next league (owner, 2026-10-09)

* **One exploiter per seat**, trained with `--opponent-lock-side` and played in the main's pool
  only in that seat (its other seat is a drifted copy of the main).
* **Publish on live games:** the exploiter's rate in its seat against the main, minus the main's
  own self-play rate there; no separate greedy tournament against 50%.
* Generations of ~100M: the US exploit appeared at 60–80M and held to 160M.
