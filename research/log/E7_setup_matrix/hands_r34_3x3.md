# Setup matrix by hand: 3 US × 3 USSR setups

Player after the setup: `models/base.pt`. 100,000 deals, each played in every cell; every number is a paired difference of setups on the same deal, in points of the chooser's score (percent of a game); ± is one standard error.

The USSR sets up first with only its own hand to go on, the US second with its hand and the USSR's setup. **Own hand** is what the chooser can condition on; **opponent's hand** is shown to explain an effect, not to choose with.

Setups: U0 `us_r34` (West Germany 4, France 2, Italy 2, Iran 1); U1 `us_wg3_fr3_it2` (West Germany 3, France 3, Italy 2, Iran 1); U2 `us_e7` (France 6, Italy 2, Iran 1); S0 `ussr_r34` (Yugoslavia 3, Poland 3); S1 `ussr_pol4_aus` (Poland 4, East Germany 1, Austria 1); S2 `ussr_e7` (Poland 3, Hungary 2, Austria 1).

## Does the hand matter at all?

A least-squares fit of each contrast on which cards the chooser holds; the Wald test is that no card moves it (robust covariance).

| contrast | mean | Wald | dof | p |
|:---|---:|---:|---:|---:|
| U1-U0 vs S0 (US points, US hand) | +0.23 ± 0.16 | 23.8 | 37 | 0.954 |
| U2-U0 vs S0 (US points, US hand) | -1.82 ± 0.21 | 70.2 | 37 | 0.000812 |
| U1-U0 vs S1 (US points, US hand) | +0.13 ± 0.16 | 60.8 | 37 | 0.00827 |
| U2-U0 vs S1 (US points, US hand) | -2.32 ± 0.21 | 66.4 | 37 | 0.00215 |
| U1-U0 vs S2 (US points, US hand) | -0.11 ± 0.16 | 44.1 | 37 | 0.195 |
| U2-U0 vs S2 (US points, US hand) | -2.53 ± 0.21 | 58.3 | 37 | 0.0143 |
| S1-S0 over the US setups (USSR points, USSR hand) | -0.72 ± 0.13 | 90.8 | 37 | 2.36e-06 |
| S2-S0 over the US setups (USSR points, USSR hand) | -1.77 ± 0.13 | 103.4 | 37 | 4.35e-08 |

## A hand-conditioned setup, out of fold

The rule is a fit made on half the deals and applied to the other half (both ways), compared with the best fixed setup of the half it was fitted on. **All cards** fits every card; **sparse** only the cards whose effect reaches |z| ≥ 3.4 on the fitting half (chosen there, so the other half still scores it honestly). A gain within noise means the hand does not tell this continuation which setup to play.

| choice | rule | cards (per half) | departs from the fixed setup | rule − best fixed | rule − reference |
|:---|:---|:---|---:|---:|---:|
| US setup vs S0 | all cards | — | 50.1% | -0.19 ± 0.12 | +0.03 ± 0.14 |
| US setup vs S0 | sparse | Marshall Plan / Defectors | 50.2% | -0.02 ± 0.11 | +0.21 ± 0.12 |
| US setup vs S1 | all cards | — | 49.6% | -0.08 ± 0.11 | +0.05 ± 0.12 |
| US setup vs S1 | sparse | none / none | 0.0% | +0.00 ± 0.00 | +0.13 ± 0.16 |
| US setup vs S2 | all cards | — | 48.1% | -0.02 ± 0.11 | -0.02 ± 0.11 |
| US setup vs S2 | sparse | none / Asia Scoring | 10.5% | +0.03 ± 0.05 | +0.03 ± 0.05 |
| USSR setup over the US setups | all cards | — | 35.0% | +0.02 ± 0.08 | +0.02 ± 0.08 |
| USSR setup over the US setups | sparse | Europe Scoring / Red Scare/Purge | 0.0% | +0.00 ± 0.00 | +0.00 ± 0.00 |

## Cards

Each card: the contrast on deals where the side holds it minus where it does not. 38 cards per table, so |z| ≥ 3.4 is the bar for one of them not being chance (Bonferroni at 5%); the largest by |z| are shown.

### U1-U0 vs S0 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Marshall Plan | 21% | +1.42 ± 0.42 | +3.4 |
| Defectors | 21% | -0.53 ± 0.37 | -1.4 |
| Special Relationship | 21% | -0.56 ± 0.40 | -1.4 |
| Fidel | 21% | +0.46 ± 0.40 | +1.2 |
| East European Unrest | 21% | -0.41 ± 0.41 | -1.0 |
| Independent Reds | 21% | -0.33 ± 0.40 | -0.8 |
| Vietnam Revolts | 21% | +0.32 ± 0.40 | +0.8 |
| Asia Scoring | 21% | +0.31 ± 0.40 | +0.8 |

### U1-U0 vs S0 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Asia Scoring | 21% | -1.01 ± 0.39 | -2.6 |
| Europe Scoring | 21% | -1.01 ± 0.41 | -2.5 |
| Independent Reds | 21% | -0.89 ± 0.40 | -2.3 |
| Nuclear Test Ban | 21% | +0.83 ± 0.40 | +2.1 |
| Fidel | 21% | -0.74 ± 0.40 | -1.9 |
| Warsaw Pact Formed | 21% | +0.68 ± 0.40 | +1.7 |
| Romanian Abdication | 21% | +0.65 ± 0.40 | +1.6 |
| Red Scare/Purge | 21% | +0.62 ± 0.39 | +1.6 |

### U2-U0 vs S0 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Defectors ** | 21% | -2.1 ± 0.5 | -4.1 |
| Europe Scoring | 21% | -1.6 ± 0.5 | -3.1 |
| NATO | 21% | -1.5 ± 0.5 | -3.0 |
| Decolonization | 21% | +1.3 ± 0.5 | +2.6 |
| Asia Scoring | 21% | +1.1 ± 0.5 | +2.2 |
| Duck and Cover | 21% | -1.0 ± 0.5 | -2.0 |
| Fidel | 21% | +0.9 ± 0.5 | +1.8 |
| CIA Created | 21% | +0.9 ± 0.5 | +1.7 |

### U2-U0 vs S0 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Asia Scoring ** | 21% | -1.9 ± 0.5 | -3.7 |
| Suez Crisis | 21% | +1.6 ± 0.5 | +3.1 |
| De-Stalinization | 21% | -1.3 ± 0.5 | -2.6 |
| Five Year Plan | 21% | +1.3 ± 0.5 | +2.5 |
| Marshall Plan | 21% | +1.2 ± 0.5 | +2.4 |
| Nasser | 21% | -1.2 ± 0.5 | -2.4 |
| East European Unrest | 21% | -1.2 ± 0.5 | -2.4 |
| De Gaulle Leads France | 21% | +1.2 ± 0.5 | +2.3 |

### U1-U0 vs S1 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| De-Stalinization | 21% | +1.14 ± 0.38 | +3.0 |
| Europe Scoring | 21% | -0.89 ± 0.35 | -2.5 |
| Special Relationship | 21% | -0.88 ± 0.39 | -2.3 |
| Marshall Plan | 21% | +0.90 ± 0.41 | +2.2 |
| Asia Scoring | 21% | -0.84 ± 0.39 | -2.2 |
| Korean War | 21% | -0.78 ± 0.39 | -2.0 |
| Warsaw Pact Formed | 21% | +0.71 ± 0.38 | +1.9 |
| UN Intervention | 21% | -0.71 ± 0.39 | -1.8 |

### U1-U0 vs S1 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Blockade | 21% | +0.97 ± 0.38 | +2.5 |
| Asia Scoring | 21% | -0.91 ± 0.37 | -2.4 |
| Suez Crisis | 21% | -0.92 ± 0.40 | -2.3 |
| Special Relationship | 21% | +0.77 ± 0.38 | +2.0 |
| Nuclear Test Ban | 21% | -0.70 ± 0.38 | -1.8 |
| De-Stalinization | 21% | -0.56 ± 0.38 | -1.5 |
| Duck and Cover | 21% | -0.56 ± 0.39 | -1.4 |
| UN Intervention | 21% | +0.48 ± 0.38 | +1.3 |

### U2-U0 vs S1 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| De-Stalinization ** | 21% | +1.7 ± 0.5 | +3.4 |
| Defectors | 21% | -1.6 ± 0.5 | -3.0 |
| Europe Scoring | 21% | -1.1 ± 0.5 | -2.2 |
| Arab-Israeli War | 21% | +1.1 ± 0.5 | +2.1 |
| Special Relationship | 21% | -1.1 ± 0.5 | -2.1 |
| Olympic Games | 21% | +0.9 ± 0.5 | +1.7 |
| Containment | 21% | -0.8 ± 0.5 | -1.6 |
| Indo-Pakistani War | 21% | -0.8 ± 0.5 | -1.6 |

### U2-U0 vs S1 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Blockade | 21% | +1.6 ± 0.5 | +3.1 |
| Captured Nazi Scientist | 21% | -1.5 ± 0.5 | -3.0 |
| De-Stalinization | 21% | -1.4 ± 0.5 | -2.8 |
| Vietnam Revolts | 21% | -1.3 ± 0.5 | -2.5 |
| Arab-Israeli War | 21% | -1.2 ± 0.5 | -2.4 |
| Romanian Abdication | 21% | +1.0 ± 0.5 | +2.1 |
| Special Relationship | 21% | +0.9 ± 0.5 | +1.8 |
| NATO | 21% | +0.9 ± 0.5 | +1.8 |

### U1-U0 vs S2 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Romanian Abdication | 21% | +1.02 ± 0.39 | +2.6 |
| Marshall Plan | 21% | +1.01 ± 0.41 | +2.4 |
| Asia Scoring | 21% | +0.88 ± 0.39 | +2.2 |
| Vietnam Revolts | 21% | +0.85 ± 0.39 | +2.2 |
| East European Unrest | 21% | -0.69 ± 0.41 | -1.7 |
| Truman Doctrine | 21% | +0.60 ± 0.39 | +1.5 |
| Warsaw Pact Formed | 21% | -0.59 ± 0.39 | -1.5 |
| UN Intervention | 21% | -0.59 ± 0.39 | -1.5 |

### U1-U0 vs S2 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Europe Scoring | 21% | -1.08 ± 0.40 | -2.7 |
| Warsaw Pact Formed | 21% | +0.80 ± 0.39 | +2.1 |
| Special Relationship | 21% | -0.79 ± 0.39 | -2.0 |
| Suez Crisis | 21% | +0.70 ± 0.41 | +1.7 |
| Red Scare/Purge | 21% | +0.54 ± 0.39 | +1.4 |
| Comecon | 21% | -0.52 ± 0.39 | -1.3 |
| Truman Doctrine | 21% | +0.53 ± 0.39 | +1.3 |
| Asia Scoring | 21% | -0.50 ± 0.38 | -1.3 |

### U2-U0 vs S2 (US points, US hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Asia Scoring | 21% | +1.7 ± 0.5 | +3.3 |
| Defectors | 21% | -1.5 ± 0.5 | -2.9 |
| Middle East Scoring | 21% | +1.2 ± 0.5 | +2.4 |
| Red Scare/Purge | 21% | -1.2 ± 0.5 | -2.3 |
| Decolonization | 21% | +0.9 ± 0.5 | +1.8 |
| NATO | 21% | -0.8 ± 0.5 | -1.5 |
| Europe Scoring | 21% | -0.8 ± 0.5 | -1.5 |
| East European Unrest | 21% | -0.7 ± 0.5 | -1.4 |

### U2-U0 vs S2 (US points, US hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| De Gaulle Leads France | 21% | +1.2 ± 0.5 | +2.3 |
| Comecon | 21% | -1.2 ± 0.5 | -2.3 |
| Blockade | 21% | +1.2 ± 0.5 | +2.3 |
| Suez Crisis | 21% | +1.1 ± 0.5 | +2.2 |
| De-Stalinization | 21% | -1.1 ± 0.5 | -2.1 |
| Middle East Scoring | 21% | -1.0 ± 0.5 | -2.1 |
| NATO | 21% | +0.9 ± 0.5 | +1.9 |
| Truman Doctrine | 21% | +0.9 ± 0.5 | +1.9 |

### S1-S0 over the US setups (USSR points, USSR hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Europe Scoring ** | 21% | -1.40 ± 0.33 | -4.3 |
| Nasser ** | 21% | +1.32 ± 0.33 | +4.0 |
| NATO | 21% | -1.02 ± 0.33 | -3.1 |
| Middle East Scoring | 21% | +0.94 ± 0.33 | +2.9 |
| Socialist Governments | 21% | -0.87 ± 0.35 | -2.5 |
| Romanian Abdication | 21% | +0.82 ± 0.33 | +2.5 |
| Decolonization | 21% | +0.66 ± 0.33 | +2.0 |
| UN Intervention | 21% | -0.59 ± 0.33 | -1.8 |

### S1-S0 over the US setups (USSR points, USSR hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| East European Unrest ** | 21% | +1.98 ± 0.33 | +6.0 |
| Middle East Scoring | 21% | -1.00 ± 0.33 | -3.0 |
| Defectors | 21% | -0.98 ± 0.34 | -2.9 |
| Independent Reds | 21% | +0.75 ± 0.33 | +2.3 |
| NATO | 21% | +0.73 ± 0.33 | +2.2 |
| Decolonization | 21% | -0.67 ± 0.33 | -2.0 |
| The Cambridge Five | 21% | +0.65 ± 0.33 | +2.0 |
| Five Year Plan | 21% | +0.63 ± 0.33 | +1.9 |

### S2-S0 over the US setups (USSR points, USSR hand): own's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Europe Scoring ** | 21% | -1.50 ± 0.32 | -4.7 |
| De-Stalinization ** | 21% | +1.43 ± 0.31 | +4.5 |
| Middle East Scoring | 21% | +0.99 ± 0.32 | +3.1 |
| Warsaw Pact Formed | 21% | -0.84 ± 0.32 | -2.6 |
| Asia Scoring | 21% | +0.77 ± 0.33 | +2.4 |
| Comecon | 21% | -0.69 ± 0.32 | -2.2 |
| De Gaulle Leads France | 21% | -0.66 ± 0.32 | -2.1 |
| East European Unrest | 21% | +0.66 ± 0.32 | +2.0 |

### S2-S0 over the US setups (USSR points, USSR hand): opponent's hand

| card | held | held − not | z |
|:---|---:|---:|---:|
| Comecon ** | 21% | +1.43 ± 0.32 | +4.4 |
| Warsaw Pact Formed | 21% | +1.01 ± 0.32 | +3.1 |
| Middle East Scoring | 21% | -1.02 ± 0.33 | -3.1 |
| Asia Scoring | 21% | -0.75 ± 0.32 | -2.3 |
| Formosan Resolution | 21% | -0.61 ± 0.32 | -1.9 |
| Europe Scoring | 21% | -0.56 ± 0.33 | -1.7 |
| Special Relationship | 21% | -0.56 ± 0.33 | -1.7 |
| Fidel | 21% | -0.55 ± 0.32 | -1.7 |
