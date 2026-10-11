# Setup matrix by hand: 2 US × 2 USSR setups

Player after the setup: `models/base.pt`. 100,000 deals, each played in every cell; every number is a paired difference of setups on the same deal, in points of the chooser's score (percent of a game); ± is one standard error.

The USSR sets up first with only its own hand to go on, the US second with its hand and the USSR's setup. **Own hand** is what the chooser can condition on; **opponent's hand** is shown to explain an effect, not to choose with.

Setups: U0 `us_e7` (France 6, Italy 2, Iran 1); U1 `us_wg3_fr3_it2` (West Germany 3, France 3, Italy 2, Iran 1); S0 `ussr_e7` (Poland 3, Hungary 2, Austria 1); S1 `ussr_pol4_aus` (Poland 4, East Germany 1, Austria 1).

## Does the hand matter at all?

A least-squares fit of each contrast on which cards the chooser holds; the Wald test is that no card moves it (robust covariance).

| contrast | mean | Wald | dof | p |
|:---|---:|---:|---:|---:|
| U1-U0 vs S0 (US points, US hand) | +2.42 ± 0.21 | 45.0 | 37 | 0.17 |
| U1-U0 vs S1 (US points, US hand) | +2.44 ± 0.21 | 61.1 | 37 | 0.00763 |
| S1-S0 over the US setups (USSR points, USSR hand) | +0.97 ± 0.14 | 72.1 | 37 | 0.0005 |

## A hand-conditioned setup, out of fold

The rule is a fit made on half the deals and applied to the other half (both ways), compared with the best fixed setup of the half it was fitted on. **All cards** fits every card; **sparse** only the cards whose effect reaches |z| ≥ 3.4 on the fitting half (chosen there, so the other half still scores it honestly). A gain within noise means the hand does not tell this continuation which setup to play.

| choice | rule | cards (per half) | departs from the fixed setup | rule − best fixed | rule − reference |
|:---|:---|:---|---:|---:|---:|
| US setup vs S0 | all cards | — | 9.8% | -0.19 ± 0.07 | +2.23 ± 0.20 |
| US setup vs S0 | sparse | none / none | 0.0% | +0.00 ± 0.00 | +2.42 ± 0.21 |
| US setup vs S1 | all cards | — | 12.4% | -0.17 ± 0.07 | +2.27 ± 0.19 |
| US setup vs S1 | sparse | none / none | 0.0% | +0.00 ± 0.00 | +2.44 ± 0.21 |
| USSR setup over the US setups | all cards | — | 24.2% | -0.05 ± 0.07 | +0.93 ± 0.12 |
| USSR setup over the US setups | sparse | none / none | 0.0% | +0.00 ± 0.00 | +0.97 ± 0.14 |

## Cards

Each card: the contrast on deals where the side holds it minus where it does not. 38 cards per table, so |z| ≥ 3.4 is the bar for one of them not being chance (Bonferroni at 5%); the largest by |z| are shown.

### U1-U0 vs S0 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Middle East Scoring | 21% | -1.2 ± 0.5 | -2.3 |
| Defectors | 21% | +1.1 ± 0.5 | +2.2 |
| NATO | 21% | +1.1 ± 0.5 | +2.1 |
| Decolonization | 21% | -1.0 ± 0.5 | -2.1 |
| Red Scare/Purge | 21% | +1.0 ± 0.5 | +2.0 |
| Europe Scoring | 21% | +0.9 ± 0.5 | +1.8 |
| Vietnam Revolts | 21% | +0.9 ± 0.5 | +1.8 |
| Asia Scoring | 21% | -0.8 ± 0.5 | -1.6 |

### U1-U0 vs S0 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| NATO | 21% | -1.4 ± 0.5 | -2.8 |
| Middle East Scoring | 21% | +1.1 ± 0.5 | +2.2 |
| Warsaw Pact Formed | 21% | +1.0 ± 0.5 | +2.0 |
| De Gaulle Leads France | 21% | -0.9 ± 0.5 | -1.8 |
| Five Year Plan | 21% | -0.9 ± 0.5 | -1.8 |
| Formosan Resolution | 21% | +0.9 ± 0.5 | +1.8 |
| Captured Nazi Scientist | 21% | +0.8 ± 0.5 | +1.7 |
| US/Japan Mutual Defense Pact | 21% | -0.8 ± 0.5 | -1.6 |

### U1-U0 vs S1 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Asia Scoring | 21% | -1.5 ± 0.5 | -3.1 |
| Arab-Israeli War | 21% | -1.1 ± 0.5 | -2.1 |
| Defectors | 21% | +1.0 ± 0.5 | +2.0 |
| Containment | 21% | +1.0 ± 0.5 | +2.0 |
| Olympic Games | 21% | -0.9 ± 0.5 | -1.9 |
| Red Scare/Purge | 21% | -0.9 ± 0.5 | -1.8 |
| Marshall Plan | 21% | +0.9 ± 0.5 | +1.8 |
| NORAD | 21% | +0.9 ± 0.5 | +1.8 |

### U1-U0 vs S1 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Suez Crisis | 21% | -1.5 ± 0.5 | -3.0 |
| Captured Nazi Scientist | 21% | +1.2 ± 0.5 | +2.3 |
| Vietnam Revolts | 21% | +1.0 ± 0.5 | +2.0 |
| Romanian Abdication | 21% | -0.9 ± 0.5 | -1.7 |
| De-Stalinization | 21% | +0.9 ± 0.5 | +1.7 |
| East European Unrest | 21% | -0.9 ± 0.5 | -1.7 |
| Nasser | 21% | +0.8 ± 0.5 | +1.6 |
| Middle East Scoring | 21% | +0.8 ± 0.5 | +1.6 |

### S1-S0 over the US setups (USSR points, USSR hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| De-Stalinization ** | 21% | -1.42 ± 0.35 | -4.0 |
| Warsaw Pact Formed | 21% | +1.18 ± 0.35 | +3.4 |
| UN Intervention | 21% | -0.84 ± 0.35 | -2.4 |
| The Cambridge Five | 21% | -0.84 ± 0.35 | -2.4 |
| Red Scare/Purge | 21% | +0.75 ± 0.35 | +2.1 |
| Nasser | 21% | +0.65 ± 0.35 | +1.9 |
| Truman Doctrine | 21% | +0.63 ± 0.35 | +1.8 |
| NATO | 21% | -0.59 ± 0.35 | -1.7 |

### S1-S0 over the US setups (USSR points, USSR hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Comecon ** | 21% | -2.21 ± 0.35 | -6.3 |
| Warsaw Pact Formed ** | 21% | -1.60 ± 0.35 | -4.6 |
| East European Unrest ** | 21% | +1.24 ± 0.36 | +3.5 |
| Red Scare/Purge | 21% | -0.96 ± 0.35 | -2.8 |
| Europe Scoring | 21% | +0.84 ± 0.35 | +2.4 |
| Vietnam Revolts | 21% | +0.74 ± 0.35 | +2.1 |
| NATO | 21% | +0.72 ± 0.35 | +2.1 |
| Formosan Resolution | 21% | +0.61 ± 0.35 | +1.8 |
