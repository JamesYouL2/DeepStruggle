# Setup matrix by hand: 2 US × 2 USSR setups

Player after the setup: `models/base.pt`. 100,000 deals, each played in every cell; every number is a paired difference of setups on the same deal, in points of the chooser's score (percent of a game); ± is one standard error.

The USSR sets up first with only its own hand to go on, the US second with its hand and the USSR's setup. **Own hand** is what the chooser can condition on; **opponent's hand** is shown to explain an effect, not to choose with.

Setups: U0 `us_e7` (France 6, Italy 2, Iran 1); U1 `us_wg3_fr3_it2` (West Germany 3, France 3, Italy 2, Iran 1); S0 `ussr_e7` (Poland 3, Hungary 2, Austria 1); S1 `ussr_pol4_aus` (Poland 4, East Germany 1, Austria 1).

## Does the hand matter at all?

A least-squares fit of each contrast on which cards the chooser holds; the Wald test is that no card moves it (robust covariance).

| contrast | mean | Wald | dof | p |
|:---|---:|---:|---:|---:|
| U1-U0 vs S0 (US points, US hand) | +1.23 ± 0.21 | 140.5 | 37 | 1.39e-13 |
| U1-U0 vs S1 (US points, US hand) | +0.58 ± 0.21 | 109.5 | 37 | 6e-09 |
| S1-S0 over the US setups (USSR points, USSR hand) | +0.31 ± 0.15 | 78.0 | 37 | 9.9e-05 |

## A hand-conditioned setup, out of fold

The rule is a fit made on half the deals and applied to the other half (both ways), compared with the best fixed setup of the half it was fitted on. **All cards** fits every card; **sparse** only the cards whose effect reaches |z| ≥ 3.4 on the fitting half (chosen there, so the other half still scores it honestly). A gain within noise means the hand does not tell this continuation which setup to play.

| choice | rule | cards (per half) | departs from the fixed setup | rule − best fixed | rule − reference |
|:---|:---|:---|---:|---:|---:|
| US setup vs S0 | all cards | — | 32.2% | +0.23 ± 0.12 | +1.46 ± 0.17 |
| US setup vs S0 | sparse | Blockade, De Gaulle Leads France / Blockade, De Gaulle Leads France | 38.2% | +0.32 ± 0.13 | +1.55 ± 0.16 |
| US setup vs S1 | all cards | — | 40.5% | +0.24 ± 0.13 | +0.81 ± 0.16 |
| US setup vs S1 | sparse | Blockade, De Gaulle Leads France / none | 19.1% | +0.13 ± 0.09 | +0.70 ± 0.19 |
| USSR setup over the US setups | all cards | — | 42.3% | +0.07 ± 0.09 | +0.38 ± 0.11 |
| USSR setup over the US setups | sparse | none / Independent Reds | 39.4% | -0.08 ± 0.09 | +0.23 ± 0.11 |

## Cards

Each card: the contrast on deals where the side holds it minus where it does not. 38 cards per table, so |z| ≥ 3.4 is the bar for one of them not being chance (Bonferroni at 5%); the largest by |z| are shown.

### U1-U0 vs S0 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| De Gaulle Leads France ** | 21% | -3.7 ± 0.5 | -7.3 |
| Blockade ** | 21% | -2.6 ± 0.5 | -5.1 |
| Captured Nazi Scientist | 21% | -1.6 ± 0.5 | -3.1 |
| Socialist Governments | 21% | +1.4 ± 0.5 | +2.9 |
| Europe Scoring | 21% | +1.4 ± 0.5 | +2.8 |
| Marshall Plan | 21% | +1.2 ± 0.5 | +2.4 |
| Five Year Plan | 21% | +1.2 ± 0.5 | +2.3 |
| The Cambridge Five | 21% | -1.0 ± 0.5 | -1.9 |

### U1-U0 vs S0 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Defectors | 21% | -1.6 ± 0.5 | -3.1 |
| Socialist Governments | 21% | -1.5 ± 0.5 | -3.0 |
| Formosan Resolution | 21% | +1.3 ± 0.5 | +2.6 |
| Blockade | 21% | +1.0 ± 0.5 | +2.0 |
| Vietnam Revolts | 21% | -1.0 ± 0.5 | -2.0 |
| Middle East Scoring | 21% | +1.0 ± 0.5 | +1.9 |
| De Gaulle Leads France | 21% | +0.9 ± 0.5 | +1.9 |
| CIA Created | 21% | +0.9 ± 0.5 | +1.8 |

### U1-U0 vs S1 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Blockade ** | 21% | -2.4 ± 0.5 | -4.8 |
| De Gaulle Leads France ** | 21% | -2.1 ± 0.5 | -4.1 |
| Europe Scoring ** | 21% | +1.9 ± 0.5 | +3.8 |
| Socialist Governments | 21% | +1.4 ± 0.5 | +2.8 |
| Five Year Plan | 21% | +1.0 ± 0.5 | +2.1 |
| Defectors | 21% | +1.0 ± 0.5 | +2.0 |
| Special Relationship | 21% | -1.0 ± 0.5 | -2.0 |
| Suez Crisis | 21% | -1.0 ± 0.5 | -1.9 |

### U1-U0 vs S1 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Europe Scoring | 21% | -1.6 ± 0.5 | -3.2 |
| Middle East Scoring | 21% | +1.3 ± 0.5 | +2.7 |
| Defectors | 21% | -1.2 ± 0.5 | -2.3 |
| Socialist Governments | 21% | -1.1 ± 0.5 | -2.3 |
| Special Relationship | 21% | +1.1 ± 0.5 | +2.1 |
| CIA Created | 21% | +1.0 ± 0.5 | +2.1 |
| Decolonization | 21% | +1.0 ± 0.5 | +1.9 |
| Nasser | 21% | +0.9 ± 0.5 | +1.8 |

### S1-S0 over the US setups (USSR points, USSR hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Independent Reds ** | 21% | +1.64 ± 0.36 | +4.6 |
| Middle East Scoring | 21% | +0.94 ± 0.35 | +2.6 |
| Indo-Pakistani War | 21% | -0.88 ± 0.36 | -2.5 |
| Romanian Abdication | 21% | +0.82 ± 0.36 | +2.3 |
| Decolonization | 21% | +0.79 ± 0.35 | +2.2 |
| Vietnam Revolts | 21% | +0.79 ± 0.36 | +2.2 |
| US/Japan Mutual Defense Pact | 21% | -0.76 ± 0.36 | -2.1 |
| Suez Crisis | 21% | -0.74 ± 0.36 | -2.1 |

### S1-S0 over the US setups (USSR points, USSR hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| East European Unrest ** | 21% | +1.64 ± 0.36 | +4.5 |
| UN Intervention ** | 21% | +1.60 ± 0.36 | +4.5 |
| Defectors | 21% | -1.08 ± 0.36 | -3.0 |
| Middle East Scoring | 21% | -1.08 ± 0.36 | -3.0 |
| Socialist Governments | 21% | +0.86 ± 0.36 | +2.4 |
| Decolonization | 21% | -0.84 ± 0.36 | -2.4 |
| Captured Nazi Scientist | 21% | -0.75 ± 0.35 | -2.1 |
| Independent Reds | 21% | -0.74 ± 0.36 | -2.1 |
