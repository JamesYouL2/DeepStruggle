# E5-17 — the P24 league from scratch at flat temperature: seed 43 is level with E5-11

**Question.** Does a main exploiter, whose published snapshots join the main agent's pool, make
the main agent stronger than the same recipe without it? The motivation was that weak setups must
stay punishable: otherwise the US drifts back to them. Because a main agent plus its exploiter
fill both GPU slots, seeds run one at a time. The owner asked for seed 43's result before
deciding on seed 44 and on E5-19.

**Arm.** E5-17-43, driven by `tools/scripts/league.py`:
* **Main agent:** E5-11-43's recipe exactly, from scratch to 560M, with the league group in its
  pool (`--league-frac 0.5`, four members, about 15% of games). `launch_flags --diff` against
  E5-11 shows only the league flags and `--train-steps`.
* **Exploiter (E5-18-43-g):** stage 1's recipe, 50M per generation, reset to the main agent's
  newest tagged resume state, first generation from 60M.

Directory `E5-17-43_20260928_130951`; league events in `data/league/E5-17-43/events.jsonl`.

## The arms race

| generation | exploiter start → target | best against its target | published | outcome |
|:---|:---|---:|---:|:---|
| 1 | 60M → 60M | 71% (USSR 87%, US 55%) | 4 | reset at 110M |
| 2 | 110M → 120M | 76% (USSR 93%, US 58%) | 3 | reset at 150M |
| 3 | 160M → 160M | 67% (USSR 70%, US 65%) | 4 | reset at 200M |
| 4 | 210M → 210M | 66% (USSR 72%, US 59%) | 3 | reset at 260M |
| 5 | 260M → 270M | 53% | 0 | budget spent |
| 6 | 310M → 330M | 49% | 0 | budget spent |
| 7 | 360M → 390M | < 55% | 0 | budget spent |
| 8 | 410M → 450M | 62% | 4 | budget spent |
| 9 | 510M → 510M | 55% | 1 | stopped at the main agent's end |

* **The start → target gap was not intended.** Resume states are tagged every 50M while
  generations launch whenever the previous one ends. Budget-spent generations run about 60M of
  main-agent time, so each started 10M further behind its target: 10, 20, 30, 40M, then wrapping
  back to 0. Generations 5–7 were handicapped by it. Generation 8 exploited anyway from 40M
  behind; generation 9, starting level, published within 10M. The fix is
  `--resume-every-steps 10000000` on the main agent.
* **The main agent counters the exploits.** Every published snapshot was played against the
  main agent's 540M snapshot, greedy, 100 games per seat:
  * generations 1–4's snapshots now score 12–40%, against 55–76% against their targets;
  * generation 8's score 40–49%, against 56–62%, so the main agent closed that gap in about 90M;
  * generation 9's newest scores 57%: an exploit not yet countered.
  The early rows mix countering with general improvement; generation 8 is the clean case.
* **Exploits are found mostly as USSR**, against the main agent's US play, in every generation.

## Strength: level with E5-11 (seed 43)

Round robin (`data/reports/e5_17_rr_43.{md,json}`): E5-17 and E5-11 at 480/520/560M, E4-61@560M and
`HeuristicBot`, 100 games per seat per pair, greedy. The bars are E5-11's twin self-play at each
step.

| late block, 9 pairings | result |
|:---|---:|
| overall head to head | **49.3%** (n = 1,800) |
| as USSR | 54.0% against a bar of 58.2% → **−4.2 ± 2.6** |
| as US | 45.0% against a bar of 40.7% → **+4.3 ± 2.6** |
| Elo at 480 / 520 / 560M | E5-17 2251 / 2265 / 2274; E5-11 2265 / 2267 / 2268 |
| against E4-61@560M, as US / as USSR | E5-17 68.7% / 80.7%; E5-11 71.3% / 78.3% |

**Not accepted:** the USSR seat misses its bar.

**The per-seat split repeats a pattern.** E5-15, E5-16 and now E5-17 are three different changes
(a fixed human opening, game-result setup credit, a league). Each came out about +4 to +8 as US
and −1 to −5 as USSR against E5-11-43. Against the neutral E4-61, E5-17's split is the other way
round: −2.6 as US, +2.4 as USSR. Part of what the rule reads on this seed is E5-11-43's own
profile. Its USSR has trained for 500M+ against its own weak US (Greece-2) and punishes that
kind of US well, so any arm whose US differs looks better as US and worse as USSR against it.

## Openings: the league did not fix them

`data/reports/e5_17_openings_43.txt` and `e5_17_setup_oracle_{us,ussr}_43.txt`, 2,000 paired deals:

| side | E5-17-43@560M's own opening | best alternative − own (playouts) | critic |
|:---|:---|---:|---:|
| US | WG 3, France 2, **Greece 2**, South Korea 2 | +5.5 ± 1.5 (human WG 4, Italy 3, France 1, Iran 1) | −4.0 |
| USSR | Poland 4, Hungary 2 | +5.9 ± 1.4 (Poland 3, Yugoslavia 3) | −26.3 |

Greece 2 is back on the US side. The league cannot repair a setup: setup credit still comes only
from the critic's λ-return. Now the critic *under*-rates the better alternatives on both sides,
where E5-11's critic over-rated them.

## Reading

On seed 43 the league runs as designed: exploits found, published and countered. But it leaves
the main agent level in strength with E5-11 and does not touch the opening. The two mechanisms
that worked each did one thing: game-result setup credit (E5-16) repaired the opening, and the
league keeps weaknesses punished. They are complementary. Seed 44 and E5-19 are on hold for the
owner's decision.
