# Setup matrix: 2 US × 2 USSR setups

Player for both sides after the setup: `models/base.pt`. 400,000 games, 100,000 deals; every cell is played on the same deals. Scores are the US's (a draw is half), in percent; ± is one standard error.

## Setups

| | US setup (influence added) | | USSR setup (influence added) |
|:---|:---|:---|:---|
| U0 | `us_e7`: France 6, Italy 2, Iran 1 | S0 | `ussr_e7`: Poland 3, Hungary 2, Austria 1 |
| U1 | `us_wg3_fr3_it2`: West Germany 3, France 3, Italy 2, Iran 1 | S1 | `ussr_pol4_aus`: Poland 4, East Germany 1, Austria 1 |

## Do the setups matter at all?

* **Joint test** that all 4 cells are equal, blocked by deal: Hotelling T² = 304.9 on 3 contrasts, p = 0.000. Read the single comparisons below against it: with this many of them, a few at 2 SE are expected by chance.
* **Each setup averaged over the other side's setups**, against the networks' own (paired by deal):
  * U1-U0 (US points): +2.43 ± 0.15
  * S1-S0 (USSR points): +0.97 ± 0.14

## The matrix (US score)

| US \ USSR | S0 | S1 | mean |
|:---|---:|---:|---:|
| U0 | 50.05 ± 0.16 | 49.07 ± 0.16 | 49.6 |
| U1 | 52.48 ± 0.16 | 51.51 ± 0.16 | 52.0 |
| mean | 51.3 | 50.3 | 50.8 |

## Against the networks' setups (paired by deal)

Each US setup minus U0 against the same USSR setup, in points of the US score:

| US setup | vs S0 | vs S1 |
|:---|---:|---:|
| U1 − U0 | +2.42 ± 0.21 | +2.44 ± 0.21 |

Each USSR setup minus S0 against the same US setup, in points of the *USSR* score:

| USSR setup | vs U0 | vs U1 |
|:---|---:|---:|
| S1 − S0 | +0.98 ± 0.20 | +0.97 ± 0.20 |

## The solved game

* **Value:** US 51.5 [95% 51.2, 51.8] with both sides mixing optimally.
* **US mix:** U1 100% (in the support of 100% of resamples).
* **USSR mix:** S1 100% (in the support of 100% of resamples).
* The networks' own pairing (U0 vs S0) scores 50.1 for the US.

Bootstrap: 1000 resamples of the deals; a setup is "in the support" of a resample when its equilibrium weight there is above 1%.
