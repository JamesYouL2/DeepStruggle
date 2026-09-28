# E5-13 — a setup entropy floor keeps the opening moving, and it moves to worse places

**Question.** E5-11 locks its setup by 40M: 0.01–0.04 nats per US placement and about 0.001 per
USSR placement. E5-11-43's locked US opening, Greece 2, costs it about 5 points as US against a
sane opening ([`E5_11_setup_lock_and_critic_views.md`](E5_11_setup_lock_and_critic_views.md)).
Does keeping setup placements resampled, through an adaptive entropy floor on setup decisions only
(`--setup-entropy-floor 0.3`), let training find a better opening without costing strength?

**Arms.** E5-13-43/44 is E5-11-SEED's configuration exactly plus `--setup-entropy-floor 0.3`, from
scratch to 240M. `launch_flags.py --diff` prints only that flag and `--train-steps`. Directories are
`E5-13-43_20260927_222255` and `E5-13-44_20260927_222315`. The runs survived a 7.5 h host suspend
without a restart.

**Pre-registered rule** ([`../runs.md`](../runs.md)): adopt if, on both seeds, the opening check's
gap to a sane opening is within 2 points **and** head to head against E5-11 at 160–240M is at
least 47%.

## The floor did what it was built to do

Setup entropy held at 0.2–0.4 nats per placement from about 15M to the end, against E5-11's
0.01–0.04. The coefficient needed for that was small, at most 0.05. Setup entropy was also logged
where the floor never bound, so E5-11-style locking can now be read from any run.

## Strength: rejected

Round robin per seed: E5-13 and E5-11 at 80/120/160/200/240M, E4-61@560M and `HeuristicBot`, 100
games per seat per pair, greedy. Reports are in `data/reports/e5_13_rr_{43,44}.{md,json}`.

| E5-13 vs E5-11 | seed 43 | seed 44 |
|:---|---:|---:|
| 80M | 41.0% | 49.5% |
| 120M | 46.5% | 36.0% |
| 160M | 44.0% | 38.0% |
| 200M | 47.5% | 47.0% |
| 240M | 42.5% | 43.0% |
| **late block, 160–240 × 160–240 (n = 1,800)** | **42.9% ± 1.2 (−50)** | **41.1% ± 1.2 (−62)** |
| at 240M, as US vs E5-11's self-play US rate | −8 | −4 |
| at 240M, as USSR vs E5-11's self-play USSR rate | −7 | −10 |

Both seeds are below the 47% bar and weaker in both seats.

## Openings: exploration drifted, and the USSR lost Poland

Openings at 240M (setup probe at temperature 0.1, `data/reports/e5_13_openings.txt`):

| | USSR | US |
|:---|:---|:---|
| E5-11-43 | Poland 3, Hungary 3 | France 4, Italy 2, Greece 2, Iran 1 |
| **E5-13-43** | **Finland 4, Hungary 2** | Finland 2, France 3, Italy 2, WG 2 (+ Iran or the Philippines) |
| E5-11-44 | Poland 3, Yugoslavia 3 | Canada 2, Spain/Portugal 2, Italy 2, South Korea 2, France 1 |
| **E5-13-44** | **Yugoslavia 4, Finland 2** | WG 2, France 3, Italy 2, Iran 1, Philippines 1 |

The setup series (`data/reports/e5_13_setup_series.json`) shows the USSR dropping Poland ≥ 3 at
120M on seed 43 and at 160M on seed 44, and never returning. E5-11 holds Poland at every snapshot.
The US openings kept changing snapshot to snapshot, but they did not settle on West Germany 4.

The opening check (`tools/scripts/setup_oracle.py`, 2,000 paired deals, `data/reports/e5_13_setup_oracle.txt`):

| best alternative − own opening, US | E5-13 | E5-11 |
|:---|---:|---:|
| seed 43 | +3.9 ± 1.4 (WG 4, Italy 3, France 2) | +4.4 ± 1.5 (human, WG 4, Italy 3, Iran 2) |
| seed 44 | +0.6 ± 1.4 | +1.2 ± 1.4 |

Seed 43's gap is not closed, so that criterion also fails there.

## Reading

* **Exploration was not what setup learning lacked.** With alternatives sampled in about half of
  training games, the setup did not converge on a better opening. It drifted, and on both seeds
  the USSR moved off Poland, which humans play in 99% of games and E5-11 always played. The
  signal that should rank openings, the critic's advantage on about 15 decisions per game whose
  consequences arrive tens of turns later, is too weak and too noisy. This is P4's diagnosis of
  the E4 setup ("the setup follows a critic whose preference oscillates"), now with exploration
  guaranteed: guaranteeing exploration turns a locked opening into a wandering one.
* **The opening matters less to this model's play than to a human's.** Its own playouts value a
  bizarre opening (E5-11-44's Canada 2, South Korea 2, Spain/Portugal 2) within 1–2 points of a
  human opening. At this strength the setup's learning signal is small by construction.
* **Exploration has a cost.** Keeping the opening noisy costs 50–60 Elo at 160–240M on both seeds.
  The likely cause is the noisier data, since each placement individually changes little.

The floor stays in the code, off by default. What is left for the setup is to stop learning it
from the critic. Either take it from somewhere that can rank openings (a playout search over
candidate openings, distilled into the setup), or fix it: script a known-good opening in training
and play, so the setup is not learned at all.
