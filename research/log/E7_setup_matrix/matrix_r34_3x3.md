# Setup matrix: 3 US × 3 USSR setups

Player for both sides after the setup: `models/base.pt`. 1,000,000 games, 100,000 deals; every cell is played on the same deals. Scores are the US's (a draw is half), in percent; ± is one standard error.

## Setups

| | US setup (influence added) | | USSR setup (influence added) |
|:---|:---|:---|:---|
| U0 | `us_r34`: West Germany 4, France 2, Italy 2, Iran 1 | S0 | `ussr_r34`: Yugoslavia 3, Poland 3 |
| U1 | `us_wg3_fr3_it2`: West Germany 3, France 3, Italy 2, Iran 1 | S1 | `ussr_pol4_aus`: Poland 4, East Germany 1, Austria 1 |
| U2 | `us_e7`: France 6, Italy 2, Iran 1 | S2 | `ussr_e7`: Poland 3, Hungary 2, Austria 1 |

## Do the setups matter at all?

* **Joint test** that all 9 cells are equal, blocked by deal: Hotelling T² = 570.3 on 8 contrasts, p = 0.000. Read the single comparisons below against it: with this many of them, a few at 2 SE are expected by chance.
* **Each setup averaged over the other side's setups**, against the networks' own (paired by deal):
  * U1-U0 (US points): +0.08 ± 0.09
  * U2-U0 (US points): -2.22 ± 0.13
  * S1-S0 (USSR points): -0.72 ± 0.13
  * S2-S0 (USSR points): -1.77 ± 0.13

## The matrix (US score)

| US \ USSR | S0 | S1 | S2 | mean |
|:---|---:|---:|---:|---:|
| U0 | 50.46 ± 0.16 | 51.38 ± 0.16 | 52.58 ± 0.16 | 51.5 |
| U1 | 50.69 ± 0.16 | 51.51 ± 0.16 | 52.48 ± 0.16 | 51.6 |
| U2 | 48.64 ± 0.16 | 49.07 ± 0.16 | 50.05 ± 0.11 | 49.3 |
| mean | 49.9 | 50.7 | 51.7 | 50.8 |

## Against the networks' setups (paired by deal)

Each US setup minus U0 against the same USSR setup, in points of the US score:

| US setup | vs S0 | vs S1 | vs S2 |
|:---|---:|---:|---:|
| U1 − U0 | +0.23 ± 0.16 | +0.13 ± 0.16 | -0.11 ± 0.16 |
| U2 − U0 | -1.82 ± 0.21 | -2.32 ± 0.21 | -2.53 ± 0.21 |

Each USSR setup minus S0 against the same US setup, in points of the *USSR* score:

| USSR setup | vs U0 | vs U1 | vs U2 |
|:---|---:|---:|---:|
| S1 − S0 | -0.92 ± 0.21 | -0.82 ± 0.21 | -0.43 ± 0.21 |
| S2 − S0 | -2.12 ± 0.20 | -1.79 ± 0.20 | -1.41 ± 0.20 |

## The solved game

* **Value:** US 50.7 [95% 50.4, 51.0] with both sides mixing optimally.
* **US mix:** U0 0% (in the support of 8% of resamples), U1 100% (in the support of 92% of resamples).
* **USSR mix:** S0 100% (in the support of 100% of resamples).
* The networks' own pairing (U0 vs S0) scores 50.5 for the US.

Bootstrap: 1000 resamples of the deals; a setup is "in the support" of a resample when its equilibrium weight there is above 1%.
