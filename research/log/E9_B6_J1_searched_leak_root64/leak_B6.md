# Where raw play loses: paired playouts by decision kind

Points of win probability a game, both sides' decisions summed (441.4 decisions with two or more legal moves a game): each playout row's paired gain of the best move over raw's, weighted back to the decisions raw play meets. 'k=8 @256 over raw' is what that searcher's own move recovers.

Screened census: the best move is chosen on the screen's pairs, and its gain measured on fresh pairs where the screen's lead held (escalated); other rows count as no loss, so the total is a lower bound. 'raw' is the move under study (annotate --player: the player's).

## total

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| escalated | 783 | 9.12% | +43.7 ± 2.8 | +34.5 ± 2.5 |
| not escalated | 7217 | 90.88% | +0.0 ± 0.0 | -150.0 ± 6.3 |

## all positions: decision kind, stratified bootstrap (2000 resamples)

Positions resampled within their (bank stratum, playout class) cell: the ± counts the sampling of positions as well as the playout noise the tables below print alone.

| kind | best over raw | share | k=8 @256 over raw | leaks the most in |
|:---|---:|---:|---:|---:|
| AR card choice | +20.8 ± 3.3 | 48% | -57.4 ± 5.4 | 98% |
| target (influence) | +11.0 ± 3.2 | 25% | -23.1 ± 6.2 | 2% |
| play mode | +6.2 ± 2.1 | 14% | -20.0 ± 2.4 | 0% |
| headline | +3.7 ± 0.5 | 8% | -7.3 ± 0.7 | 0% |
| op mode | +1.2 ± 0.3 | 3% | -4.8 ± 0.6 | 0% |
| target (coup) | +0.6 ± 0.4 | 1% | -0.0 ± 1.2 | 0% |
| choose_branch | +0.1 ± 0.0 | 0% | -0.1 ± 0.1 | 0% |
| target (realign) | +0.1 ± 0.4 | 0% | -2.8 ± 1.2 | 0% |
| all | +43.7 ± 5.1 |  | -115.6 ± 8.8 |  |

## all positions: decision kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| AR card choice | 1707 | 23.38% | +20.8 ± 1.1 | -57.4 ± 3.4 |
| target (influence) | 1438 | 41.32% | +11.0 ± 2.4 | -23.1 ± 5.6 |
| play mode | 1695 | 20.73% | +6.2 ± 0.9 | -20.0 ± 1.3 |
| headline | 1429 | 4.04% | +3.7 ± 0.2 | -7.3 ± 0.4 |
| op mode | 1175 | 3.14% | +1.2 ± 0.1 | -4.8 ± 0.3 |
| target (coup) | 142 | 4.59% | +0.6 ± 0.4 | -0.0 ± 1.1 |
| choose_branch | 287 | 0.35% | +0.1 ± 0.0 | -0.1 ± 0.1 |
| target (realign) | 127 | 2.46% | +0.1 ± 0.2 | -2.8 ± 0.9 |

## all positions: side x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| USSR AR card choice | 811 | 10.90% | +11.8 ± 0.8 | -28.6 ± 2.3 |
| US AR card choice | 896 | 12.47% | +9.1 ± 0.8 | -28.7 ± 2.5 |
| USSR target (influence) | 700 | 20.91% | +5.9 ± 1.7 | -9.3 ± 3.7 |
| US target (influence) | 738 | 20.41% | +5.1 ± 1.6 | -13.8 ± 4.2 |
| USSR play mode | 849 | 10.43% | +3.5 ± 0.6 | -8.9 ± 1.0 |
| US play mode | 846 | 10.30% | +2.7 ± 0.6 | -11.1 ± 0.9 |
| USSR headline | 707 | 1.99% | +2.2 ± 0.2 | -3.7 ± 0.3 |
| US headline | 722 | 2.05% | +1.5 ± 0.1 | -3.7 ± 0.3 |
| USSR op mode | 583 | 1.56% | +0.6 ± 0.1 | -3.0 ± 0.2 |
| US op mode | 592 | 1.59% | +0.6 ± 0.1 | -1.8 ± 0.2 |

## all positions: era x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| T4-7 AR card choice | 569 | 10.76% | +8.8 ± 0.8 | -26.2 ± 2.5 |
| T8-10 target (influence) | 450 | 6.52% | +8.0 ± 0.5 | -2.9 ± 1.0 |
| T8-10 AR card choice | 569 | 4.50% | +6.3 ± 0.3 | -8.4 ± 0.6 |
| T1-3 AR card choice | 569 | 8.12% | +5.8 ± 0.7 | -22.7 ± 2.1 |
| T1-3 target (influence) | 529 | 20.06% | +2.7 ± 1.8 | -18.0 ± 4.6 |
| T8-10 play mode | 565 | 4.02% | +2.7 ± 0.3 | -5.8 ± 0.4 |
| T1-3 play mode | 565 | 7.48% | +2.1 ± 0.7 | -4.9 ± 0.8 |
| T4-7 headline | 543 | 1.86% | +1.8 ± 0.2 | -3.0 ± 0.3 |
| T4-7 play mode | 565 | 9.22% | +1.5 ± 0.5 | -9.2 ± 1.0 |
| T8-10 headline | 386 | 0.80% | +1.3 ± 0.0 | -1.1 ± 0.1 |
| T8-10 op mode | 363 | 0.66% | +0.7 ± 0.0 | -1.5 ± 0.1 |
| T1-3 headline | 500 | 1.39% | +0.5 ± 0.1 | -3.2 ± 0.3 |

## all positions: DEFCON x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| D2 AR card choice | 1260 | 17.64% | +12.0 ± 0.9 | -37.3 ± 3.0 |
| D2 other | 1993 | 35.02% | +9.8 ± 2.1 | -19.3 ± 4.6 |
| D3 AR card choice | 251 | 3.21% | +7.1 ± 0.5 | -12.8 ± 1.2 |
| D3 other | 1779 | 11.55% | +5.4 ± 0.9 | -12.6 ± 2.2 |
| D2 play mode | 1203 | 15.79% | +2.3 ± 0.7 | -12.7 ± 1.0 |
| D3 play mode | 238 | 2.52% | +2.2 ± 0.3 | -4.2 ± 0.5 |
| D4+ AR card choice | 196 | 2.52% | +1.8 ± 0.4 | -7.2 ± 1.0 |
| D4+ play mode | 254 | 2.42% | +1.7 ± 0.3 | -3.0 ± 0.6 |
| D4+ other | 826 | 9.32% | +1.4 ± 0.8 | -6.2 ± 2.8 |

## all positions, AR card choice: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 3 -> opp 2 | 5 | 0.10% | +1.3 ± 0.3 | -0.0 ± 0.0 |
| own 2 -> own 2 | 196 | 2.66% | +0.7 ± 0.2 | -8.8 ± 1.2 |
| neutral 2 -> opp 2 | 7 | 0.11% | +0.7 ± 0.2 | +0.2 ± 0.1 |
| own 2 -> opp 3 | 5 | 0.05% | +0.4 ± 0.1 | +0.2 ± 0.1 |
| own 3 -> own 2 | 8 | 0.07% | +0.4 ± 0.1 | -0.1 ± 0.2 |
| opp 3 -> opp 3 | 228 | 3.23% | +0.3 ± 0.1 | -6.2 ± 1.0 |
| opp 2 -> opp 2 | 242 | 3.15% | +0.2 ± 0.2 | -10.7 ± 1.2 |
| scoring 0 -> scoring 0 | 125 | 1.59% | +0.1 ± 0.0 | -5.9 ± 0.7 |
| opp 3 -> opp 2 | 5 | 0.03% | +0.1 ± 0.1 | +0.2 ± 0.2 |
| neutral 2 -> own 2 | 5 | 0.06% | +0.0 ± 0.2 | +0.2 ± 0.2 |
| neutral 2 -> neutral 2 | 135 | 1.58% | +0.0 ± 0.0 | -7.6 ± 1.0 |
| own 3 -> own 3 | 166 | 2.55% | +0.0 ± 0.1 | -5.8 ± 0.9 |

## all positions, AR card choice: the card raw plays instead (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Nuclear Test Ban | 27 | 0.57% | +1.9 ± 0.2 | -0.2 ± 0.1 |
| Red Scare/Purge | 11 | 0.15% | +1.5 ± 0.3 | -0.5 ± 0.2 |
| ABM Treaty | 11 | 0.19% | +1.2 ± 0.2 | +0.1 ± 0.1 |
| Latin American Death Squads | 22 | 0.39% | +1.1 ± 0.2 | -3.7 ± 0.6 |
| Five Year Plan | 28 | 0.55% | +1.0 ± 0.2 | +0.0 ± 0.5 |
| Special Relationship | 33 | 0.42% | +0.9 ± 0.2 | -0.9 ± 0.6 |
| Suez Crisis | 9 | 0.12% | +0.8 ± 0.2 | -0.0 ± 0.3 |
| Duck and Cover | 24 | 0.32% | +0.7 ± 0.3 | -0.7 ± 0.3 |
| East European Unrest | 31 | 0.49% | +0.7 ± 0.2 | -1.0 ± 0.5 |
| UN Intervention | 19 | 0.33% | +0.6 ± 0.2 | -1.2 ± 0.4 |
| How I Learned to Stop Worrying | 15 | 0.16% | +0.6 ± 0.2 | -0.7 ± 0.3 |
| Iranian Hostage Crisis | 14 | 0.12% | +0.5 ± 0.1 | -0.7 ± 0.1 |

## all positions, AR card choice: the card the best move plays (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| The China Card | 58 | 0.76% | +3.6 ± 0.2 | -2.0 ± 0.8 |
| Arms Race | 10 | 0.16% | +1.9 ± 0.1 | -0.0 ± 0.2 |
| US/Japan Mutual Defense Pact | 22 | 0.38% | +1.4 ± 0.2 | -0.2 ± 0.3 |
| Arab-Israeli War | 33 | 0.51% | +0.9 ± 0.3 | -1.4 ± 0.5 |
| How I Learned to Stop Worrying | 17 | 0.20% | +0.8 ± 0.2 | -0.3 ± 0.3 |
| John Paul II Elected Pope | 19 | 0.27% | +0.8 ± 0.2 | +0.0 ± 0.5 |
| Sadat Expels Soviets | 16 | 0.20% | +0.7 ± 0.2 | -0.6 ± 0.2 |
| Vietnam Revolts | 10 | 0.12% | +0.7 ± 0.2 | -0.6 ± 0.3 |
| Iran-Contra Scandal | 11 | 0.11% | +0.5 ± 0.1 | -0.1 ± 0.2 |
| Grain Sales to Soviets | 7 | 0.12% | +0.5 ± 0.2 | -0.3 ± 0.2 |
| Southeast Asia Scoring | 12 | 0.22% | +0.5 ± 0.2 | +0.3 ± 0.2 |
| CIA Created | 17 | 0.25% | +0.5 ± 0.1 | -0.2 ± 0.3 |

## all positions, play mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| neutral card: COUP -> INFL | 5 | 0.06% | +1.2 ± 0.3 | +0.2 ± 0.1 |
| own card: INFL -> COUP | 7 | 0.13% | +1.1 ± 0.4 | +0.1 ± 0.1 |
| own card: INFL -> EVENT | 9 | 0.09% | +0.9 ± 0.2 | +0.2 ± 0.1 |
| opp card: EVENT -> INFL | 5 | 0.06% | +0.5 ± 0.2 | +0.3 ± 0.1 |
| neutral card: EVENT -> INFL | 5 | 0.04% | +0.4 ± 0.1 | +0.2 ± 0.1 |
| neutral card: INFL -> COUP | 5 | 0.07% | +0.3 ± 0.3 | +0.3 ± 0.1 |
| own card: COUP -> INFL | 7 | 0.04% | +0.3 ± 0.1 | +0.5 ± 0.1 |
| own card: INFL -> REALIGN | 9 | 0.07% | +0.2 ± 0.2 | +0.4 ± 0.1 |
| neutral card: INFL -> EVENT | 4 | 0.02% | +0.2 ± 0.0 | -0.0 ± 0.0 |
| opp card: INFL -> REALIGN | 6 | 0.03% | +0.2 ± 0.1 | +0.1 ± 0.0 |

## all positions, play mode: the card (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Nuclear Test Ban | 19 | 0.37% | +1.4 ± 0.4 | -0.3 ± 0.1 |
| Five Year Plan | 27 | 0.35% | +1.3 ± 0.1 | -0.2 ± 0.1 |
| Flower Power | 13 | 0.22% | +0.6 ± 0.3 | -0.0 ± 0.1 |
| De Gaulle Leads France | 17 | 0.34% | +0.4 ± 0.3 | +0.1 ± 0.1 |
| AWACS Sale to Saudis | 14 | 0.11% | +0.4 ± 0.1 | -0.1 ± 0.0 |
| Romanian Abdication | 29 | 0.42% | +0.4 ± 0.2 | +0.0 ± 0.1 |
| De-Stalinization | 18 | 0.23% | +0.3 ± 0.1 | -0.3 ± 0.1 |
| Comecon | 32 | 0.36% | +0.2 ± 0.1 | -0.1 ± 0.2 |
| Junta | 5 | 0.11% | +0.2 ± 0.1 | -0.0 ± 0.0 |
| Puppet Governments | 12 | 0.16% | +0.2 ± 0.1 | -0.4 ± 0.1 |

## all positions, op mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| COUP -> REALIGN | 14 | 0.03% | +0.5 ± 0.1 | +0.2 ± 0.0 |
| INFL -> REALIGN | 18 | 0.03% | +0.2 ± 0.0 | +0.3 ± 0.0 |
| REALIGN -> INFL | 4 | 0.01% | +0.2 ± 0.0 | +0.3 ± 0.0 |
| COUP -> INFL | 22 | 0.04% | +0.2 ± 0.1 | +0.4 ± 0.1 |
| INFL -> COUP | 10 | 0.02% | +0.1 ± 0.1 | +0.1 ± 0.0 |
| 208 -> REALIGN | 1 | 0.00% | +0.0 ± 0.0 | +0.0 ± 0.0 |

## all positions, influence targets: raw's region -> the best region

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Middle East -> Middle East | 118 | 3.60% | +1.4 ± 0.5 | -2.5 ± 1.9 |
| Africa -> Africa | 190 | 4.34% | +1.1 ± 0.4 | -3.0 ± 1.8 |
| Europe -> Europe | 401 | 12.65% | +0.8 ± 1.0 | -8.5 ± 3.0 |
| South America -> Central America | 4 | 0.10% | +0.7 ± 0.5 | +0.3 ± 0.2 |
| Asia -> Europe | 6 | 0.05% | +0.6 ± 0.1 | -0.0 ± 0.2 |
| South America -> South America | 167 | 4.55% | +0.6 ± 0.4 | -3.5 ± 1.8 |
| Africa -> South America | 4 | 0.07% | +0.5 ± 0.2 | +1.0 ± 0.3 |
| Asia -> Central America | 6 | 0.15% | +0.4 ± 0.5 | +1.0 ± 0.5 |
| South America -> Middle East | 6 | 0.12% | +0.3 ± 0.3 | +0.1 ± 0.1 |
| Europe -> Middle East | 4 | 0.03% | +0.2 ± 0.1 | +0.6 ± 0.2 |

## all positions, headline: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 3 -> own 1 | 5 | 0.02% | +0.3 ± 0.1 | +0.0 ± 0.0 |
| own 3 -> scoring 0 | 5 | 0.02% | +0.3 ± 0.1 | -0.0 ± 0.0 |
| own 2 -> scoring 0 | 8 | 0.01% | +0.2 ± 0.0 | +0.0 ± 0.0 |
| own 2 -> own 3 | 6 | 0.02% | +0.2 ± 0.1 | +0.1 ± 0.0 |
| own 3 -> own 2 | 7 | 0.02% | +0.2 ± 0.1 | +0.1 ± 0.0 |
| own 3 -> neutral 2 | 7 | 0.01% | +0.2 ± 0.0 | +0.1 ± 0.0 |
| neutral 2 -> own 3 | 5 | 0.01% | +0.1 ± 0.0 | +0.0 ± 0.0 |
| own 2 -> own 2 | 292 | 0.90% | +0.1 ± 0.1 | -1.7 ± 0.2 |

## The 20 positions that cost raw the most a game (all positions)

| turn/AR | side | DEFCON | kind | raw | best | gain a decision | a game |
|:---|:---|---:|:---|:---|:---|---:|---:|
| T4 AR1 | USSR | 3 | AR card choice | SelectCard #34 (Nuclear Test Ban) | SelectCard #39 (Arms Race) | +12.6 ± 0.8 | +1.91 |
| T10 AR4 | US | 3 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: EVENT | +16.5 ± 0.8 | +1.33 |
| T3 AR1 | USSR | 3 | AR card choice | SelectCard #31 (Red Scare/Purge) | SelectCard #27 (US/Japan Mutual Defense Pact) | +11.0 ± 1.5 | +1.33 |
| T2 AR1 | USSR | 4 | play mode | Resolution: OPS_COUP / OpMode: COUP | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | +5.6 ± 1.5 | +1.10 |
| T10 AR2 | USSR | 2 | target (influence) | PointNode #70 (Panama) | PointNode #14 (East Germany) | +7.6 ± 1.2 | +1.10 |
| T10 AR6 | US | 2 | target (influence) | PointNode #53 (Nigeria) | PointNode #72 (Haiti) | +6.3 ± 0.5 | +0.92 |
| T8 AR2 | USSR | 2 | target (influence) | PointNode #59 (Angola) | PointNode #82 (Argentina) | +9.1 ± 1.4 | +0.87 |
| T4 AR3 | US | 3 | target (influence) | PointNode #79 (Brazil) | PointNode #82 (Argentina) | +4.5 ± 1.5 | +0.84 |
| T7 AR2 | US | 2 | AR card choice | SelectCard #69 (Latin American Death Squads) | SelectCard #6 (The China Card) | +13.5 ± 1.4 | +0.74 |
| T2 AR3 | US | 2 | AR card choice | SelectCard #4 (Duck and Cover) | SelectCard #9 (Vietnam Revolts) | +5.5 ± 1.4 | +0.66 |
| T7 AR3 | US | 2 | target (influence) | PointNode #79 (Brazil) | PointNode #64 (Mexico) | +1.9 ± 1.4 | +0.65 |
| T10 AR0 | USSR | 2 | target (influence) | PointNode #8 (France) | PointNode #9 (Spain/Portugal) | +6.4 ± 1.0 | +0.61 |
| T10 AR1 | USSR | 2 | target (influence) | PointNode #19 (Romania) | PointNode #59 (Angola) | +5.9 ± 1.1 | +0.56 |
| T6 AR7 | US | 2 | AR card choice | SelectCard #76 (Ussuri River Skirmish) | SelectCard #72 (Sadat Expels Soviets) | +4.7 ± 1.0 | +0.56 |
| T6 AR1 | US | 2 | AR card choice | SelectCard #57 (ABM Treaty) | SelectCard #38 (Southeast Asia Scoring) | +4.7 ± 1.2 | +0.56 |
| T7 AR2 | USSR | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_COUP / OpMode: COUP | +2.5 ± 1.5 | +0.55 |
| T5 AR2 | US | 2 | AR card choice | SelectCard #105 (Special Relationship) | SelectCard #67 (Grain Sales to Soviets) | +4.6 ± 1.4 | +0.54 |
| T2 AR6 | USSR | 2 | AR card choice | SelectCard #5 (Five Year Plan) | SelectCard #6 (The China Card) | +7.2 ± 1.4 | +0.53 |
| T9 AR2 | USSR | 4 | AR card choice | SelectCard #63 (Colonial Rear Guards) | SelectCard #93 (Iran-Contra Scandal) | +9.0 ± 1.2 | +0.50 |
| T5 AR1 | USSR | 3 | AR card choice | SelectCard #50 (“We Will Bury You”) | SelectCard #68 (John Paul II Elected Pope) | +4.1 ± 1.1 | +0.48 |
