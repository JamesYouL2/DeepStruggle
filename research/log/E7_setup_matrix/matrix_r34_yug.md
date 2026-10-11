# Setup matrix: 2 US × 2 USSR setups

Player for both sides after the setup: `models/base.pt`. 400,000 games, 100,000 deals; every cell is played on the same deals. Scores are the US's (a draw is half), in percent; ± is one standard error.

## Setups

| | US setup (influence added) | | USSR setup (influence added) |
|:---|:---|:---|:---|
| U0 | `us_r34`: West Germany 4, France 2, Italy 2, Iran 1 | S0 | `ussr_r34`: Yugoslavia 3, Poland 3 |
| U1 | `us_wg3_fr3_it2`: West Germany 3, France 3, Italy 2, Iran 1 | S1 | `ussr_pol4_yug`: Poland 4, East Germany 1, Yugoslavia 1 |

## Do the setups matter at all?

* **Joint test** that all 4 cells are equal, blocked by deal: Hotelling T² = 15.4 on 3 contrasts, p = 0.002. Read the single comparisons below against it: with this many of them, a few at 2 SE are expected by chance.
* **Each setup averaged over the other side's setups**, against the networks' own (paired by deal):
  * U1-U0 (US points): +0.17 ± 0.11
  * S1-S0 (USSR points): -0.61 ± 0.17

## The matrix (US score)

| US \ USSR | S0 | S1 | mean |
|:---|---:|---:|---:|
| U0 | 50.46 ± 0.16 | 51.13 ± 0.16 | 50.8 |
| U1 | 50.69 ± 0.16 | 51.24 ± 0.16 | 51.0 |
| mean | 50.6 | 51.2 | 50.9 |

## Against the networks' setups (paired by deal)

Each US setup minus U0 against the same USSR setup, in points of the US score:

| US setup | vs S0 | vs S1 |
|:---|---:|---:|
| U1 − U0 | +0.23 ± 0.16 | +0.11 ± 0.16 |

Each USSR setup minus S0 against the same US setup, in points of the *USSR* score:

| USSR setup | vs U0 | vs U1 |
|:---|---:|---:|
| S1 − S0 | -0.67 ± 0.20 | -0.55 ± 0.20 |

## The solved game

* **Value:** US 50.7 [95% 50.4, 51.0] with both sides mixing optimally.
* **US mix:** U0 0% (in the support of 8% of resamples), U1 100% (in the support of 92% of resamples).
* **USSR mix:** S0 100% (in the support of 100% of resamples).
* The networks' own pairing (U0 vs S0) scores 50.5 for the US.

Bootstrap: 1000 resamples of the deals; a setup is "in the support" of a resample when its equilibrium weight there is above 1%.
