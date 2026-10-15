# Where raw play loses: paired playouts by decision kind

Points of win probability a game, both sides' decisions summed (444.2 decisions with two or more legal moves a game): each playout row's paired gain of the best move over raw's, weighted back to the decisions raw play meets. 'k=8 @256 over raw' is what that searcher's own move recovers.

Screened census: the best move is chosen on the screen's pairs, and its gain measured on fresh pairs where the screen's lead held (escalated); other rows count as no loss, so the total is a lower bound. 'raw' is the move under study (annotate --player: the player's).

## total

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| escalated | 1073 | 11.26% | +96.9 ± 3.4 | +52.2 ± 2.3 |
| not escalated | 6927 | 88.74% | +0.0 ± 0.0 | -142.7 ± 5.2 |

## all positions: decision kind, stratified bootstrap (2000 resamples)

Positions resampled within their (bank stratum, playout class) cell: the ± counts the sampling of positions as well as the playout noise the tables below print alone.

| kind | best over raw | share | k=8 @256 over raw | leaks the most in |
|:---|---:|---:|---:|---:|
| AR card choice | +50.4 ± 11.1 | 52% | -37.6 ± 10.4 | 100% |
| play mode | +16.9 ± 3.3 | 17% | -10.9 ± 2.2 | 0% |
| target (influence) | +16.3 ± 5.1 | 17% | -30.1 ± 6.1 | 0% |
| headline | +7.6 ± 1.1 | 8% | -5.4 ± 1.0 | 0% |
| target (coup) | +2.8 ± 1.3 | 3% | -1.4 ± 1.9 | 0% |
| op mode | +2.7 ± 0.6 | 3% | -3.4 ± 0.6 | 0% |
| choose_branch | +0.2 ± 0.1 | 0% | -0.4 ± 0.1 | 0% |
| target (realign) | +0.0 ± 0.1 | 0% | -1.3 ± 0.8 | 0% |
| all | +96.9 ± 12.9 |  | -90.5 ± 12.5 |  |

## all positions: decision kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| AR card choice | 1716 | 23.25% | +50.4 ± 1.4 | -37.6 ± 3.2 |
| play mode | 1704 | 20.14% | +16.9 ± 1.0 | -10.9 ± 1.1 |
| target (influence) | 1429 | 41.17% | +16.3 ± 2.7 | -30.1 ± 4.3 |
| headline | 1367 | 3.94% | +7.6 ± 0.3 | -5.4 ± 0.4 |
| target (coup) | 154 | 4.55% | +2.8 ± 0.9 | -1.4 ± 1.3 |
| op mode | 1268 | 4.32% | +2.7 ± 0.2 | -3.4 ± 0.3 |
| choose_branch | 238 | 0.29% | +0.2 ± 0.0 | -0.4 ± 0.1 |
| target (realign) | 124 | 2.34% | +0.0 ± 0.1 | -1.3 ± 0.7 |

## all positions: side x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| US AR card choice | 895 | 11.94% | +28.4 ± 1.0 | -16.4 ± 2.2 |
| USSR AR card choice | 821 | 11.31% | +22.0 ± 1.0 | -21.2 ± 2.3 |
| US target (influence) | 747 | 21.15% | +11.7 ± 2.0 | -19.3 ± 3.2 |
| US play mode | 857 | 10.11% | +11.5 ± 0.9 | -6.4 ± 0.8 |
| USSR play mode | 847 | 10.03% | +5.4 ± 0.4 | -4.5 ± 0.8 |
| USSR target (influence) | 682 | 20.02% | +4.6 ± 1.8 | -10.8 ± 2.9 |
| US headline | 746 | 2.22% | +4.5 ± 0.2 | -2.9 ± 0.3 |
| USSR headline | 621 | 1.72% | +3.1 ± 0.2 | -2.5 ± 0.3 |
| USSR target (coup) | 86 | 2.53% | +2.0 ± 0.6 | -0.8 ± 0.9 |
| USSR op mode | 613 | 2.22% | +1.5 ± 0.1 | -1.4 ± 0.2 |

## all positions: era x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| T4-7 AR card choice | 572 | 10.73% | +24.9 ± 1.1 | -10.4 ± 2.4 |
| T8-10 AR card choice | 572 | 4.26% | +15.4 ± 0.4 | -9.6 ± 0.7 |
| T1-3 AR card choice | 572 | 8.26% | +10.1 ± 0.8 | -17.7 ± 2.0 |
| T8-10 target (influence) | 439 | 6.17% | +7.5 ± 0.6 | -5.2 ± 0.9 |
| T1-3 play mode | 568 | 7.36% | +6.6 ± 0.7 | -2.6 ± 0.6 |
| T8-10 play mode | 568 | 3.85% | +5.2 ± 0.2 | -4.2 ± 0.3 |
| T4-7 play mode | 568 | 8.93% | +5.1 ± 0.7 | -4.2 ± 0.9 |
| T4-7 target (influence) | 458 | 15.05% | +4.9 ± 1.4 | -16.2 ± 2.6 |
| T1-3 target (influence) | 532 | 19.95% | +3.9 ± 2.2 | -8.8 ± 3.3 |
| T4-7 headline | 519 | 1.85% | +2.9 ± 0.2 | -1.4 ± 0.3 |
| T1-3 headline | 477 | 1.34% | +2.3 ± 0.2 | -2.1 ± 0.3 |
| T8-10 headline | 371 | 0.75% | +2.3 ± 0.1 | -1.9 ± 0.1 |

## all positions: DEFCON x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| D2 AR card choice | 1357 | 18.43% | +30.1 ± 1.2 | -33.7 ± 2.9 |
| D2 other | 2108 | 36.53% | +18.7 ± 2.4 | -24.6 ± 3.8 |
| D3 AR card choice | 196 | 2.62% | +17.2 ± 0.4 | +0.6 ± 1.0 |
| D2 play mode | 1254 | 15.25% | +11.0 ± 0.8 | -8.5 ± 1.0 |
| D3 other | 1729 | 10.64% | +10.1 ± 1.2 | -12.1 ± 1.7 |
| D4+ play mode | 180 | 1.62% | +3.7 ± 0.6 | -0.9 ± 0.3 |
| D4+ AR card choice | 163 | 2.21% | +3.2 ± 0.6 | -4.5 ± 1.0 |
| D3 play mode | 270 | 3.27% | +2.2 ± 0.3 | -1.4 ± 0.4 |
| D4+ other | 743 | 9.44% | +0.8 ± 1.0 | -5.4 ± 1.9 |

## all positions, AR card choice: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| opp 1 -> opp 3 | 5 | 0.08% | +10.6 ± 0.2 | +9.0 ± 0.2 |
| opp 2 -> opp 2 | 228 | 3.06% | +4.2 ± 0.2 | -7.4 ± 1.2 |
| own 2 -> opp 3 | 9 | 0.09% | +2.6 ± 0.3 | +0.4 ± 0.1 |
| own 2 -> China 4 | 7 | 0.10% | +2.6 ± 0.2 | +0.2 ± 0.1 |
| own 3 -> neutral 2 | 6 | 0.14% | +2.1 ± 0.4 | -0.0 ± 0.1 |
| opp 3 -> opp 2 | 9 | 0.14% | +1.6 ± 0.3 | +0.5 ± 0.2 |
| own 2 -> opp 2 | 7 | 0.13% | +1.1 ± 0.3 | +0.3 ± 0.2 |
| neutral 2 -> opp 3 | 6 | 0.06% | +1.0 ± 0.1 | +0.2 ± 0.2 |
| neutral 4 -> own 2 | 5 | 0.07% | +0.9 ± 0.2 | -0.1 ± 0.1 |
| opp 2 -> China 4 | 6 | 0.11% | +0.9 ± 0.3 | +0.3 ± 0.2 |
| opp 2 -> opp 3 | 7 | 0.11% | +0.8 ± 0.2 | +0.6 ± 0.2 |
| scoring 0 -> opp 2 | 7 | 0.10% | +0.7 ± 0.3 | +0.3 ± 0.2 |

## all positions, AR card choice: the card raw plays instead (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Blockade | 21 | 0.30% | +9.3 ± 0.1 | +7.8 ± 0.5 |
| Star Wars | 9 | 0.06% | +3.8 ± 0.1 | -0.2 ± 0.1 |
| Nuclear Test Ban | 20 | 0.37% | +2.3 ± 0.3 | -0.5 ± 0.3 |
| South African Unrest | 14 | 0.17% | +1.8 ± 0.1 | -0.3 ± 0.2 |
| NATO | 19 | 0.44% | +1.6 ± 0.4 | -0.5 ± 0.3 |
| Olympic Games | 54 | 0.61% | +1.3 ± 0.2 | -1.5 ± 0.6 |
| The Voice of America | 13 | 0.22% | +1.3 ± 0.4 | +0.2 ± 0.3 |
| Kitchen Debates | 14 | 0.21% | +1.1 ± 0.2 | -0.2 ± 0.5 |
| Defectors | 22 | 0.30% | +1.1 ± 0.2 | -0.7 ± 0.4 |
| John Paul II Elected Pope | 16 | 0.28% | +1.0 ± 0.1 | -0.2 ± 0.2 |
| Arab-Israeli War | 40 | 0.55% | +1.0 ± 0.3 | -0.2 ± 0.5 |
| SALT Negotiations | 11 | 0.22% | +0.9 ± 0.2 | +0.1 ± 0.5 |

## all positions, AR card choice: the card the best move plays (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| OPEC | 17 | 0.21% | +9.4 ± 0.1 | +9.0 ± 0.5 |
| The China Card | 63 | 0.92% | +6.0 ± 0.6 | -3.1 ± 0.7 |
| Grain Sales to Soviets | 14 | 0.14% | +3.8 ± 0.1 | -0.2 ± 0.3 |
| Duck and Cover | 34 | 0.48% | +1.8 ± 0.2 | -0.6 ± 0.5 |
| SALT Negotiations | 9 | 0.21% | +1.7 ± 0.2 | -0.1 ± 0.4 |
| Socialist Governments | 26 | 0.34% | +1.4 ± 0.1 | -0.0 ± 0.3 |
| Liberation Theology | 17 | 0.32% | +1.2 ± 0.3 | -0.5 ± 0.4 |
| Decolonization | 20 | 0.26% | +1.2 ± 0.2 | -0.4 ± 0.3 |
| Latin American Death Squads | 27 | 0.41% | +1.2 ± 0.2 | -0.1 ± 0.4 |
| Quagmire | 8 | 0.15% | +1.2 ± 0.2 | +0.0 ± 0.1 |
| Our Man in Tehran | 21 | 0.31% | +1.2 ± 0.3 | +0.1 ± 0.2 |
| Indo-Pakistani War | 18 | 0.21% | +1.0 ± 0.3 | +0.7 ± 0.3 |

## all positions, play mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| neutral card: INFL -> EVENT | 4 | 0.12% | +2.2 ± 0.3 | +0.1 ± 0.0 |
| own card: INFL -> EVENT | 6 | 0.07% | +1.8 ± 0.3 | +0.2 ± 0.1 |
| opp card: SPACE -> EVENT | 7 | 0.06% | +1.5 ± 0.2 | +0.7 ± 0.1 |
| own card: EVENT -> INFL | 17 | 0.07% | +1.2 ± 0.1 | +0.7 ± 0.1 |
| opp card: EVENT -> INFL | 11 | 0.06% | +1.0 ± 0.1 | +0.1 ± 0.0 |
| opp card: EVENT -> SPACE | 6 | 0.07% | +0.8 ± 0.3 | +0.2 ± 0.1 |
| neutral card: EVENT -> INFL | 10 | 0.09% | +0.8 ± 0.2 | +0.6 ± 0.1 |
| own card: INFL -> REALIGN | 8 | 0.06% | +0.6 ± 0.3 | +0.1 ± 0.1 |
| own card: EVENT -> COUP | 10 | 0.03% | +0.5 ± 0.1 | +0.2 ± 0.1 |
| opp card: SPACE -> INFL | 6 | 0.02% | +0.5 ± 0.0 | +0.5 ± 0.1 |

## all positions, play mode: the card (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| SALT Negotiations | 8 | 0.18% | +1.9 ± 0.2 | -0.1 ± 0.1 |
| Containment | 12 | 0.19% | +1.7 ± 0.3 | -0.2 ± 0.1 |
| ABM Treaty | 9 | 0.04% | +1.0 ± 0.1 | -0.3 ± 0.1 |
| Decolonization | 21 | 0.49% | +0.9 ± 0.2 | -0.2 ± 0.1 |
| Grain Sales to Soviets | 18 | 0.27% | +0.9 ± 0.1 | +0.0 ± 0.1 |
| East European Unrest | 31 | 0.36% | +0.8 ± 0.3 | -0.1 ± 0.1 |
| Vietnam Revolts | 24 | 0.14% | +0.7 ± 0.3 | -0.1 ± 0.1 |
| How I Learned to Stop Worrying | 14 | 0.23% | +0.6 ± 0.3 | -0.4 ± 0.2 |
| The Iron Lady | 12 | 0.08% | +0.6 ± 0.1 | +0.0 ± 0.1 |
| Willy Brandt | 20 | 0.18% | +0.5 ± 0.1 | -0.7 ± 0.2 |

## all positions, op mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| COUP -> INFL | 27 | 0.06% | +1.1 ± 0.1 | +0.3 ± 0.1 |
| COUP -> REALIGN | 14 | 0.04% | +0.6 ± 0.1 | +0.1 ± 0.0 |
| INFL -> COUP | 26 | 0.05% | +0.5 ± 0.1 | +0.7 ± 0.1 |
| INFL -> REALIGN | 20 | 0.05% | +0.3 ± 0.1 | +0.3 ± 0.0 |
| REALIGN -> COUP | 6 | 0.01% | +0.2 ± 0.0 | +0.1 ± 0.0 |
| REALIGN -> INFL | 5 | 0.01% | +0.0 ± 0.0 | +0.1 ± 0.0 |

## all positions, influence targets: raw's region -> the best region

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Middle East -> Asia | 9 | 0.20% | +2.5 ± 0.5 | +1.1 ± 0.3 |
| Asia -> Asia | 272 | 7.86% | +2.3 ± 0.5 | -9.8 ± 2.1 |
| Central America -> Asia | 6 | 0.22% | +2.0 ± 0.8 | +1.1 ± 0.3 |
| Africa -> Middle East | 5 | 0.13% | +1.5 ± 0.5 | +0.7 ± 0.0 |
| Africa -> Europe | 4 | 0.13% | +1.4 ± 0.6 | +0.1 ± 0.1 |
| Europe -> Middle East | 5 | 0.08% | +1.2 ± 0.3 | +1.0 ± 0.5 |
| Central America -> Central America | 144 | 3.50% | +1.0 ± 0.1 | -5.1 ± 1.4 |
| Asia -> Africa | 5 | 0.22% | +0.9 ± 0.6 | -0.1 ± 0.1 |
| South America -> Asia | 5 | 0.10% | +0.9 ± 0.2 | +0.3 ± 0.3 |
| Asia -> Middle East | 9 | 0.28% | +0.8 ± 0.7 | +1.3 ± 0.6 |

## all positions, headline: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 2 -> scoring 0 | 5 | 0.02% | +1.0 ± 0.1 | +0.0 ± 0.0 |
| own 3 -> scoring 0 | 8 | 0.03% | +0.5 ± 0.1 | +0.0 ± 0.0 |
| own 3 -> own 2 | 13 | 0.04% | +0.5 ± 0.1 | +0.1 ± 0.1 |
| own 2 -> own 1 | 8 | 0.02% | +0.4 ± 0.1 | +0.1 ± 0.1 |
| own 3 -> neutral 4 | 5 | 0.01% | +0.2 ± 0.0 | +0.1 ± 0.0 |
| own 3 -> own 1 | 4 | 0.02% | +0.2 ± 0.1 | +0.0 ± 0.0 |
| neutral 4 -> neutral 2 | 6 | 0.02% | +0.2 ± 0.1 | +0.1 ± 0.0 |
| own 2 -> own 3 | 7 | 0.02% | +0.2 ± 0.0 | +0.1 ± 0.0 |

## The 20 positions that cost raw the most a game (all positions)

| turn/AR | side | DEFCON | kind | raw | best | gain a decision | a game |
|:---|:---|---:|:---|:---|:---|---:|---:|
| T5 AR1 | US | 3 | AR card choice | SelectCard #10 (Blockade) | SelectCard #61 (OPEC) | +83.3 ± 0.8 | +9.26 |
| T10 AR1 | USSR | 3 | AR card choice | SelectCard #85 (Star Wars) | SelectCard #67 (Grain Sales to Soviets) | +59.8 ± 1.1 | +3.74 |
| T7 AR4 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: EVENT | +8.2 ± 1.0 | +1.89 |
| T8 AR4 | USSR | 2 | AR card choice | SelectCard #53 (South African Unrest) | SelectCard #6 (The China Card) | +29.1 ± 1.3 | +1.81 |
| T6 AR1 | US | 2 | AR card choice | SelectCard #21 (NATO) | SelectCard #43 (SALT Negotiations) | +10.6 ± 1.5 | +1.70 |
| T2 AR1 | US | 4 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: EVENT | +8.0 ± 1.3 | +1.66 |
| T3 AR6 | USSR | 2 | target (influence) | PointNode #57 (Zaire) | PointNode #10 (Italy) | +2.8 ± 1.3 | +1.26 |
| T3 AR6 | US | 2 | target (influence) | PointNode #71 (Cuba) | PointNode #39 (Indonesia) | +2.8 ± 1.4 | +1.23 |
| T5 AR3 | USSR | 5 | AR card choice | SelectCard #48 (Kitchen Debates) | SelectCard #4 (Duck and Cover) | +6.8 ± 1.3 | +1.09 |
| T4 AR0 | US | 3 | target (influence) | PointNode #79 (Brazil) | PointNode #64 (Mexico) | +6.0 ± 1.5 | +1.07 |
| T7 AR2 | USSR | 2 | target (influence) | PointNode #32 (Pakistan) | PointNode #44 (South Korea) | +3.0 ± 1.1 | +1.06 |
| T4 AR3 | USSR | 2 | target (influence) | PointNode #10 (Italy) | PointNode #25 (Iran) | +5.6 ± 1.5 | +1.00 |
| T4 AR1 | US | 2 | target (influence) | PointNode #47 (Algeria) | PointNode #26 (Jordan) | +2.7 ± 1.4 | +0.95 |
| T3 AR2 | US | 2 | play mode | Resolution: SPACE | Resolution: EVENT | +4.4 ± 1.0 | +0.91 |
| T2 AR1 | USSR | 2 | target (coup) | PointNode #18 (Yugoslavia) | PointNode #17 (Hungary) | +2.0 ± 1.1 | +0.89 |
| T7 AR5 | US | 2 | AR card choice | SelectCard #68 (John Paul II Elected Pope) | SelectCard #42 (Quagmire) | +7.8 ± 1.2 | +0.86 |
| T4 AR1 | USSR | 2 | target (influence) | PointNode #79 (Brazil) | PointNode #36 (Thailand) | +4.8 ± 0.7 | +0.85 |
| T6 AR1 | US | 2 | AR card choice | SelectCard #56 (Muslim Revolution) | SelectCard #15 (Nasser) | +7.6 ± 1.5 | +0.85 |
| T2 AR6 | USSR | 2 | target (influence) | PointNode #25 (Iran) | PointNode #36 (Thailand) | +4.6 ± 1.3 | +0.83 |
| T3 AR3 | US | 2 | AR card choice | SelectCard #103 (Defectors) | SelectCard #7 (Socialist Governments) | +18.1 ± 1.4 | +0.80 |
