# E5-16 — setup credited by the game result: the US opening is repaired, and seed 43's USSR pays for it

**Question.** E5-11's setup learns only from the critic, and the critic over-rates unfamiliar
openings. With γ = 1 and λ = 0.98, a setup placement's λ-return weights the game result by ≤ 0.01
([`E5_11_setup_lock_and_critic_views.md`](E5_11_setup_lock_and_critic_views.md)). Setup placements
are never resampled (p ≈ 1), so seed 43's US has kept a Greece-2 opening that is 5 points worse
than a sane one. Does crediting the setup with the game result itself (`--setup-mc-credit`,
advantage = result − V, as Ataraxos trains its setup), with `--setup-entropy-floor 0.3` to re-open
the locked setup, repair the opening, and does the model come out stronger?

**Arms.** E5-16-SEED is E5-11-SEED resumed at its best (43: 550M → 700M; 44: 650M → 800M) with
those two flags. `launch_flags.py --diff` against E5-11 shows only them, plus the step target on
seed 43. The control is E5-11's own continuation at the same steps. Both runs completed; the
directories are `E5-16-43_20260928_101018` and `E5-16-44_20260928_101038`.

**Pre-registered** ([`../runs.md`](../runs.md)):
* **Mechanism:** seed 43's US opening ends within 2 points of the best alternative in
  `setup_oracle.py`.
* **Strength**, the owner's per-seat rule: against E5-11 at the matched last 40M, above E5-11's own
  self-play USSR rate as USSR **and** above its US rate as US, on both seeds.

## Dynamics

The locked setup (0.01–0.03 nats per placement) held until the floor's coefficient had climbed to
0.28–0.34. Then it broke free and overshot, to 1.1 nats on seed 43 and 1.45 on seed 44, before the
coefficient returned to 0. From there the game-result credit alone brought the setup down: seed 43
reached 0.51–0.56 by 600–680M and 0.37 at 700M; seed 44 was at 0.64 at 800M. The Monte Carlo step
trained on about 1,700–2,900 finished placements per iteration, with a clip fraction under 3% at
first and 15–20% later. The floor's integrator overshooting a locked start is a tuning point for
any rerun: a lower `--setup-entropy-lr`, or a cap on the coefficient.

## Mechanism: the opening is repaired, and the critic is calibrated on it

Setup series (`data/reports/e5_16_setup_series.json`): seed 43 meets all four human setup targets
(USSR Poland ≥ 3; US West Germany ≥ 4, Italy ≥ 2, Iran ≥ 2) at every snapshot from 610M on. E5-11
meets none of them in any snapshot, and never gets West Germany to 4.

| | USSR | US |
|:---|:---|:---|
| E5-11-43@700M | Poland 3, Hungary 3 | WG 3–4, Greece 2, Italy 2, France 1 (+ Iran) |
| **E5-16-43@700M** | Poland 3, Yugoslavia 3 | **WG 4, France 2, Italy 2, Iran 1** (500 of 500) |
| E5-11-44@800M | Poland 3, Yugoslavia 3 | Canada 2, WG 3, Italy 2, France 1–2 (+ Iran) |
| **E5-16-44@800M** | Poland 3, Yugoslavia 3 | WG 2, France 3, Italy 2, Iran 1, Panama 1 |

Opening check, 2,000 paired deals (`data/reports/e5_16_setup_oracle.txt`):

| best alternative − own opening | playouts | critic |
|:---|---:|---:|
| E5-11-43@700M (control) | +7.3 ± 1.5 | +29.2 |
| **E5-16-43@700M** | **+3.4 ± 1.5** (WG 4, Italy 3, Iran 2; the other two +0.2 / +0.4) | −3.5 |
| E5-11-44@800M (control) | −0.6 | +4.8 |
| **E5-16-44@800M** | +0.8 ± 1.4 | −8.3 |

The seed-43 gap fell from 7.3 to 3.4 points. That misses the pre-registered 2 by 1 standard error,
so **the mechanism criterion narrowly fails on seed 43** and passes on seed 44. The critic stopped
over-rating the alternatives: the control's +26 to +29 became −1 to −8, now slightly the other way.
Sampling openings and crediting them with results appears to calibrate the critic on them too.

## Strength: the US seat gains, and seed 43's USSR loses as much

Round robins per seed (`data/reports/e5_16_rr_{43,44}.md`): E5-16 and E5-11 at the last three
snapshots, E5-11's best, E4-61@560M and `HeuristicBot`. 100 games per seat per pair, greedy. The
bars are E5-11's twin self-play at each late step (200 distinct games each).

| late block (9 pairings × 100 per seat) | seed 43 (660/680/700M) | seed 44 (760/780/800M) |
|:---|---:|---:|
| as USSR vs E5-11 | 50.8% | 58.3% |
| bar: E5-11 self-play USSR | 55.8% (61.0 / 55.5 / 51.0) | 59.3% (62.5 / 58.5 / 57.0) |
| **USSR difference** | **−5.1 ± 2.6** | −1.0 ± 2.6 |
| as US vs E5-11 | 51.3% | 41.9% |
| bar: E5-11 self-play US | 43.8% (39.0 / 44.0 / 48.5) | 40.2% (37.5 / 41.0 / 42.0) |
| **US difference** | **+7.5 ± 2.6** | +1.7 ± 2.6 |
| overall head to head | 50.7% | 50.1% |

**Not accepted:** the USSR seat misses its bar on both seeds.

The USSR test has a confound. E5-16's USSR meets E5-11's US Greece opening, which it no longer
trains against. The common reference, E4-61@560M, playing its own openings, separates this:

| vs E4-61@560M, late block (n = 300 per seat) | as US | as USSR |
|:---|---:|---:|
| E5-16-43 | 85.7% | 76.7% |
| E5-11-43 | 78.0% | 81.3% |
| E5-16-44 | 70.7% | 78.3% |
| E5-11-44 | 70.3% | 79.7% |

Seed 43's USSR is weaker against a neutral opponent too (−4.6 ± 3.3), so the drop is not only the
confound. Its US gain is the same size against both opponents (+7.5 and +7.7).

## Reading

* **Game-result credit does what the critic's λ-return could not.** It moved a US opening locked at
  p ≈ 1 to a sane one (WG 4 / France 2 / Italy 2 / Iran 1). It halved the gap to the best
  alternative. It turned the critic from +29 points of optimism about other openings to calibrated.
* **The US seat's gain is real and repeats.** It is +7.5 here against E5-11 and +7.7 against E4-61,
  after E5-15's +6.8 from the scripted human opening on the same seed. Seed 43's US was
  handicapped by its opening by about 7 points.
* **Seed 43's USSR lost about 5 points, and why is open.** Two candidates:
  * the USSR's own setup moved from Hungary 3 to Yugoslavia 3 (`setup_oracle.py` is US-only, so
    this cannot yet be measured);
  * collateral from the floor's overshoot, near-random openings on both sides for several million
    steps.
  Seed 44's USSR, with the same Poland 3 / Yugoslavia 3 at both ends, is level.
* **Open next:** a USSR opening check (setup_oracle for the USSR side); the floor with a damped
  integrator; and seed-43 repeats. Credit per seat, the US only, would test whether the US gain
  can be had without touching the USSR.

## Follow-up: the USSR opening check (2026-09-28)

`tools/scripts/setup_oracle.py --side USSR`: the USSR's six placements are forced at the start of
the game, the model then plays the US setup in reply, and each branch is played out (2,000 paired
deals, `data/reports/setup_oracle_ussr.txt`). A regression run on the US side reproduces the
earlier numbers exactly (42.6% / 48.0% / +5.4 / critic +16.2). Each figure is the alternative minus
the model's own opening, as USSR win rate:

| USSR opening | E5-16-43@700M (own: Pol 3, Yug 3) | E5-11-43@700M (own: Pol 3, Hun 3) | E5-16-44@800M (own: Pol 3, Yug 3) | E5-11-44@800M (own: Pol 3, Yug 3) |
|:---|---:|---:|---:|---:|
| Poland 3, Hungary 3 | −0.9 ± 1.4 | own | −0.8 | −0.8 |
| Poland 3, Yugoslavia 3 | own | **−22.6 ± 1.5** | own | own |
| Poland 3, Romania 3 | −3.7 | −9.4 | −2.4 | −3.7 |
| human (EG 1, Poland 4, Yug 1) | −0.2 | −8.0 | −1.4 | −0.4 |
| critic's range for the alternatives | −16 to −2 | −42 to −30 | −1 to +4 | −24 to −20 |

* **Seed 43's USSR did not lose strength by moving from Hungary to Yugoslavia.** For E5-16-43 the
  two are within a point of each other (−0.9 ± 1.4), and the human opening is level too. The −5
  in the per-seat test comes from its play, not its opening.
* **E5-11-43 is brittle off its own opening, and E5-16 is not.** Forced into Yugoslavia, E5-11-43
  loses 22.6 points; the human opening costs it 8, Romania 9. E5-16-43, which trained on sampled
  openings for 150M, loses at most 3.7 to any of them. E5-11-44, whose own opening is already
  Yugoslavia, is not brittle either. What E5-11-43 cannot handle is any opening it does not play.
* **The critic's error has a direction.** Read from the US side, which is the mover's view after
  the USSR's setup, E5-11's critic rates every unfamiliar USSR opening as far better for the US
  than it is (−20 to −42 against −1 to −23 in playouts). On the US side, it rated every unfamiliar
  US opening as far better for the US (+15 to +29 against +2 to +7). So it is not "optimistic about
  alternatives": it favours the US in positions it has not seen. E5-16's critic has mostly lost the
  bias; seed 43's Hungary at −16 is the exception.
* **What is left of seed 43's USSR deficit** (−5.1 against E5-11, −4.6 against E4-61) is in its
  play. The candidates are adaptation to the stronger US opening it now trains against (so it plays
  worse against the Greece-2 US and E4-61's openings), collateral from the floor's overshoot, and
  seed noise, since seed 44 is −1.0. Only a replicate on another seed separates these.

## Follow-up: is seed 43's USSR worse, or adapted? (2026-09-28)

**The two readings.** By ~670M, E5-16's opponent pool holds only E5-16 snapshots, whose US plays the
repaired opening, so its USSR has not met a Greece-2 US for a long time. Its USSR deficit against
E5-11 and E4-61 could therefore be a general decline, or specialisation away from a weak US opening
it no longer meets. The test holds the US policy fixed (E5-11's) and varies only its opening: its
own (Greece 2 / Canada 2), or E5-16's repaired one, scripted through a one-sided evaluation opening
(`us_e516_43` / `us_e516_44` in `tools/lib/openings.py`; the USSR sets up for itself). E5-16's
USSR and E5-11's USSR each face it over the late block (3 × 3 pairings × 100 games,
`data/reports/e5_16_adaptation_{43,44}.md`).

| USSR win rate | vs E5-11 US, its own opening | vs E5-11 US, E5-16's opening |
|:---|---:|---:|
| seed 43: E5-16 | 50.8% | 50.3% |
| seed 43: E5-11 | 54.7% | 47.7% |
| seed 43: E5-16 − E5-11 | −3.9 | **+2.7 ± 2.4** |
| seed 44: E5-16 | 58.3% | 59.7% |
| seed 44: E5-11 | 62.3% | 58.1% |
| seed 44: E5-16 − E5-11 | −4.0 | **+1.6 ± 2.3** |

* **E5-11's USSR depends on its own US's weak opening. E5-16's does not.** When the same US
  policy switches to the repaired opening, E5-11's USSR loses 7.0 points (seed 43) and 4.2 (seed
  44). E5-16's USSR is unchanged (−0.5, +1.4).
* **Against the repaired opening, E5-16's USSR is level or better:** +2.7 and +1.6, pooled
  +2.1 ± 1.7. The deficit appears only against the weak openings E5-11's USSR has trained to exploit
  for hundreds of millions of steps.
* So the USSR "loss" is E5-16 no longer exploiting a weakness that its own training population no
  longer has. It is not a general decline. It is real against those opponents: E5-11's US, and
  probably E4-61's. That is the per-seat rule's bar measuring the control's USSR against the
  control's own US.
