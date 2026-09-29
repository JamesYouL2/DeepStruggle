# E6-03/E6-04 — E5-21's recipe on E6, seed 44: the result replicates

**Question (owner, 2026-09-28/29).** A new baseline for E6 and E5-21's second seed in one arm:
does game-result setup credit plus the league from 310M beat the plain run by the owner's
per-seat rule, on a new seed and the corrected engine?

**Arms.** All three used E6, the era fix plus the Kitchen Debates fix.

* **E6-03-44:** E5-11's recipe at seed 44, from scratch to 560M, no league. It is the control, and
  the source of the league run's 310M state. Its flag diff against E5-11-43 shows only the seed and
  the budget.
* **E6-04-44 (main agent) + E6-05-44-g (exploiter):** E5-21's recipe exactly, from E6-03-44's
  `resume_310050816steps.pt` to 560M. Its flag diff against E5-21-43 shows only the seed and the
  league directory.
* **Commands:** `research/checkpoints.md`, *Replication recipe*.

## League

The exploiter ran six generations. The sixth was stopped when the main agent finished.

| generation | trained against | published (vs its frozen target) | outcome |
|:---|:---|:---|:---|
| 1 | main@320M | 58 / 62 / **69%** (330–350M) | reset at 65%; against the main's newest snapshot only 52–55% |
| 2 | main@350M | 59 / 63 / 57 / 59% (370–400M) | budget spent; USSR-sided (64–72% as USSR) |
| 3 | main@400M | 56% (430M) | budget spent; balanced |
| 4 | main@450M | 56% (460M) | budget spent |
| 5 | main@500M | 55 / 61 / 56% (530–550M) | budget spent |

This league was far more active than E5-21's, where only generation 2 published anything. The
exploits it found shrank from generation 1 (69%) to 55–61%. Over the run, the main agent's win rate
against its 16-member pool fell from 0.69 to 0.65.

**Setup entropy.** The main agent inherited a locked setup, at 0.004 nats per placement. The floor
re-opened it to 0.74 by 330M. The game-result credit then narrowed it to 0.25–0.30.

## Strength

Round robin on E6 (`data/reports/e6_04_rr_44.{md,json}`): both arms at 480/520/560M, plus
E5-21-43@560M and `HeuristicBot`. Games were greedy, 1,000 per seat per pairing. The bars come from
E6-03-44's twin self-play at each step, 1,000 per seat:

| step | US bar | USSR bar |
|:---|---:|---:|
| 480M | 43.2 | 55.6 |
| 520M | 44.0 | 55.1 |
| 560M | 45.3 | 54.1 |

Results against those bars:

| | E6-04-44 (seed 44, E6) | *E5-21-43 (seed 43, E5)* |
|:---|---:|---:|
| overall vs the plain run, late block (9 pairings) | **54.4%** | *54.6%* |
| as US vs bar | **+10.0 ± 1.0** | *+11.8 ± 2.6* |
| as USSR vs bar | **−0.4 ± 1.0** | *−0.8 ± 2.6* |
| Elo at 480 / 520 / 560M | 2409 / 2432 / 2422 | *2315 / 2340 / 2344 (another field)* |
| plain run at 480 / 520 / 560M | 2386 / 2369 / 2408 | *E5-11: 2293 / 2303 / 2297* |

The ± values are 95% intervals.

**Against the previous best, E5-21-43@560M, on E6** (1,000 per seat, ± 2.2):

| model | 480M | 520M | 560M |
|:---|---:|---:|---:|
| E6-04-44 | 48.7% | 50.9% | 48.5% |
| E6-03-44 | 43.5% | 41.5% | 45.4% |

Elo: E6-04-44@520M 2432, E5-21-43@560M 2427.

## Openings at 560M

The probe ran 500 games at temperature 0.1 (`data/reports/e6_0{3,4}_44_560m_openings.txt`). Both
models play one opening, 500 of 500:

* **US:** West Germany 4, France 3, Italy 2.
* **USSR:** Poland 3, Hungary 3.

Opening check, 2,000 paired deals (`e6_0{3,4}_44_560m_setup_oracle_{us,ussr}.txt`):

| side, alternative | E6-04-44 (league + credit): playouts − own | critic | E6-03-44 (plain): playouts − own | critic |
|:---|---:|---:|---:|---:|
| US, human WG4 It3 Iran2 | **+2.1 ± 1.4** | −4.7 | **+3.6 ± 1.5** | −11.3 |
| US, E5-21's WG3 Fr3 It2 Iran1 | +0.1 ± 1.1 | −1.7 | −0.8 ± 1.1 | −10.1 |
| USSR, Pol3 Yug3 | −1.7 ± 1.4 | −3.3 | −0.2 ± 1.4 | −12.5 |
| USSR, human EG1 Pol4 Yug1 | −3.8 ± 1.4 | −2.1 | −1.2 ± 1.5 | −11.6 |

## Reading

* **E5-21's result replicates on a second seed and the corrected engine.**
  * The overall result is the same: 54.4% against 54.6%.
  * So is the per-seat shape: a large gain in the US seat (+10.0 against +11.8) and the USSR seat
    level with its bar (−0.4 against −0.8).
  * By the letter of the rule it is again **not accepted**. The USSR seat is 0.4 below its bar,
    within one standard error; the US seat is about 20 standard errors above.
  * It is the most consistent arm result in the registry.
* **The mechanism is the same, even though the opening did not move.**
  * The plain run's US opening is not locked on a bad choice this time: it plays West Germany 4 /
    France 3 / Italy 2, not Greece. Its critic is still blind to setups, rating the human opening
    11 points worse than the playouts show.
  * The league run keeps the same opening. It closes the playout gap to the best alternative from
    +3.6 to +2.1, which misses the pre-registered 2 points by 0.1, as E5-21 did at +2.2. It also
    brings the critic most of the way to calibrated: −4.7 against −11.3.
  * The US gain therefore comes from play after the setup, and from a critic that reads setups
    correctly, rather than from a different opening.
* **USSR: the own opening is best in both arms.** This passes.
* **The E6 baseline (E6-03-44) is weaker than the E5 best model:** 41–45% against E5-21-43@560M.
  This is seed 44 plain against seed 43 with the league.
* **E6-04-44 is level with E5-21-43@560M:** 48.5–50.9%, with Elo 2422–2432 against 2427. It is
  the best E6-trained model. Seed 44's plain run is also stronger relative to its bars than seed
  43's was.
* **Still open:** seed 43 on E6 (E6-03/04-43) would complete the pair, and would say whether the
  E6-trained model beats E5-21-43 on its own seed.
