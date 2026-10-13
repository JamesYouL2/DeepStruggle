# Where raw play loses: paired playouts by decision kind

Points of win probability a game, both sides' decisions summed (455.1 decisions with two or more legal moves a game): each playout row's paired gain of the best move over raw's, weighted back to the decisions raw play meets. 'k=8 @256 over raw' is what that searcher's own move recovers. 'Large' is a search-valued raw regret of 2+ points, where the gain is (research/log/E7_search_disagreement_bank.md); the rest sits at its noise.

## total

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| large regret (2+ points) | 1352 | 6.73% | +18.5 ± 2.7 | +18.1 ± 1.4 |
| the rest | 1071 | 88.77% | -21.8 ± 13.8 | -30.6 ± 7.1 |

## large regret: decision kind, stratified bootstrap (2000 resamples)

Positions resampled within their (bank stratum, playout class) cell: the ± counts the sampling of positions as well as the playout noise the tables below print alone.

| kind | best over raw | share | k=8 @256 over raw | leaks the most in |
|:---|---:|---:|---:|---:|
| target (influence) | +11.2 ± 3.3 | 61% | +7.6 ± 2.1 | 88% |
| headline | +4.1 ± 1.1 | 22% | +1.6 ± 0.4 | 1% |
| AR card choice | +1.6 ± 7.2 | 9% | +3.2 ± 2.0 | 11% |
| target (coup) | +1.3 ± 0.6 | 7% | +0.5 ± 0.6 | 0% |
| op mode | +0.5 ± 0.9 | 3% | +1.0 ± 0.6 | 0% |
| target (realign) | +0.0 ± 0.1 | 0% | +0.1 ± 0.1 | 0% |
| choose_branch | +0.0 ± 0.1 | 0% | -0.0 ± 0.1 | 0% |
| play mode | -0.4 ± 3.4 | -2% | +4.2 ± 1.2 | 0% |
| all | +18.5 ± 8.9 |  | +18.1 ± 3.2 |  |

## large regret: decision kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| target (influence) | 106 | 1.03% | +11.2 ± 1.3 | +7.6 ± 0.8 |
| headline | 263 | 0.45% | +4.1 ± 0.3 | +1.6 ± 0.1 |
| AR card choice | 386 | 2.83% | +1.6 ± 1.7 | +3.2 ± 0.9 |
| target (coup) | 12 | 0.12% | +1.3 ± 0.3 | +0.5 ± 0.3 |
| op mode | 202 | 0.43% | +0.5 ± 0.4 | +1.0 ± 0.1 |
| target (realign) | 4 | 0.05% | +0.0 ± 0.2 | +0.1 ± 0.1 |
| choose_branch | 8 | 0.01% | +0.0 ± 0.0 | -0.0 ± 0.0 |
| play mode | 371 | 1.80% | -0.4 ± 1.6 | +4.2 ± 0.4 |

## large regret: side x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| US target (influence) | 53 | 0.46% | +6.8 ± 0.8 | +5.6 ± 0.4 |
| US AR card choice | 176 | 1.28% | +4.8 ± 1.1 | +0.6 ± 0.6 |
| USSR target (influence) | 53 | 0.57% | +4.4 ± 1.0 | +2.0 ± 0.7 |
| US headline | 163 | 0.28% | +3.6 ± 0.3 | +1.5 ± 0.1 |
| US play mode | 186 | 0.82% | +2.2 ± 0.8 | +2.1 ± 0.3 |
| USSR target (coup) | 8 | 0.06% | +0.9 ± 0.2 | +0.6 ± 0.2 |
| USSR op mode | 97 | 0.22% | +0.8 ± 0.3 | +0.2 ± 0.1 |
| USSR headline | 100 | 0.17% | +0.6 ± 0.2 | +0.1 ± 0.1 |
| US target (coup) | 4 | 0.06% | +0.4 ± 0.3 | -0.1 ± 0.2 |
| USSR target (realign) | 1 | 0.01% | +0.1 ± 0.1 | +0.1 ± 0.1 |

## large regret: era x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| T4-7 target (influence) | 32 | 0.48% | +4.7 ± 1.1 | +2.7 ± 0.5 |
| T8-10 AR card choice | 158 | 0.71% | +3.9 ± 0.5 | +1.0 ± 0.3 |
| T8-10 target (influence) | 50 | 0.24% | +3.8 ± 0.2 | +3.1 ± 0.1 |
| T1-3 target (influence) | 24 | 0.31% | +2.7 ± 0.6 | +1.7 ± 0.6 |
| T4-7 headline | 97 | 0.20% | +2.4 ± 0.3 | +0.5 ± 0.1 |
| T8-10 play mode | 156 | 0.44% | +2.2 ± 0.3 | +2.5 ± 0.2 |
| T1-3 play mode | 90 | 0.46% | +1.9 ± 0.8 | +0.2 ± 0.2 |
| T8-10 headline | 64 | 0.09% | +1.3 ± 0.1 | +0.8 ± 0.1 |
| T8-10 target (coup) | 6 | 0.07% | +1.2 ± 0.3 | +0.7 ± 0.3 |
| T4-7 AR card choice | 112 | 1.27% | +1.0 ± 1.4 | +1.0 ± 0.7 |
| T8-10 op mode | 69 | 0.12% | +0.7 ± 0.1 | +0.6 ± 0.1 |
| T1-3 headline | 102 | 0.15% | +0.4 ± 0.2 | +0.3 ± 0.1 |

## large regret: DEFCON x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| D2 other | 217 | 1.24% | +9.4 ± 1.3 | +6.4 ± 0.8 |
| D3 other | 275 | 0.59% | +7.9 ± 0.5 | +4.6 ± 0.4 |
| D3 play mode | 39 | 0.18% | +1.5 ± 0.5 | -0.3 ± 0.2 |
| D4+ AR card choice | 51 | 0.34% | +1.1 ± 0.5 | -0.1 ± 0.4 |
| D3 AR card choice | 43 | 0.31% | +1.0 ± 0.5 | +1.5 ± 0.3 |
| D2 play mode | 288 | 1.41% | +0.5 ± 1.4 | +4.1 ± 0.4 |
| D4+ other | 103 | 0.26% | +0.0 ± 0.4 | -0.3 ± 0.3 |
| D2 AR card choice | 292 | 2.18% | -0.4 ± 1.5 | +1.9 ± 0.7 |
| D4+ play mode | 44 | 0.22% | -2.4 ± 0.7 | +0.4 ± 0.1 |

## large regret, AR card choice: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 2 -> opp 3 | 12 | 0.12% | +1.8 ± 0.4 | -0.5 ± 0.2 |
| own 2 -> opp 2 | 7 | 0.05% | +1.4 ± 0.2 | +0.1 ± 0.0 |
| own 2 -> neutral 3 | 5 | 0.06% | +1.1 ± 0.4 | +0.0 ± 0.1 |
| own 3 -> opp 3 | 7 | 0.06% | +0.9 ± 0.2 | +0.1 ± 0.1 |
| opp 3 -> opp 2 | 9 | 0.08% | +0.7 ± 0.4 | -0.2 ± 0.1 |
| own 2 -> own 3 | 7 | 0.08% | +0.6 ± 0.4 | +0.5 ± 0.1 |
| opp 2 -> China 4 | 7 | 0.04% | +0.6 ± 0.1 | +0.6 ± 0.1 |
| scoring 0 -> China 4 | 5 | 0.05% | +0.3 ± 0.2 | -0.3 ± 0.2 |
| opp 2 -> opp 2 | 8 | 0.04% | +0.2 ± 0.1 | +0.2 ± 0.1 |
| own 2 -> neutral 4 | 5 | 0.02% | +0.2 ± 0.1 | +0.2 ± 0.1 |
| own 1 -> opp 2 | 5 | 0.02% | +0.2 ± 0.1 | +0.2 ± 0.1 |
| scoring 0 -> opp 3 | 21 | 0.10% | +0.2 ± 0.2 | +0.3 ± 0.2 |

## large regret, AR card choice: the card raw plays instead (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Arab-Israeli War | 6 | 0.04% | +1.7 ± 0.2 | +0.2 ± 0.1 |
| Colonial Rear Guards | 4 | 0.08% | +1.1 ± 0.5 | +0.3 ± 0.0 |
| Formosan Resolution | 4 | 0.06% | +1.0 ± 0.3 | +0.2 ± 0.1 |
| Puppet Governments | 4 | 0.04% | +0.9 ± 0.4 | +0.0 ± 0.0 |
| pass | 7 | 0.04% | +0.8 ± 0.2 | +0.2 ± 0.1 |
| The Voice of America | 5 | 0.05% | +0.7 ± 0.2 | +0.1 ± 0.1 |
| Star Wars | 7 | 0.03% | +0.6 ± 0.1 | -0.0 ± 0.1 |
| “We Will Bury You” | 5 | 0.03% | +0.5 ± 0.1 | +0.4 ± 0.1 |
| The China Card | 14 | 0.08% | +0.4 ± 0.2 | +0.5 ± 0.2 |
| NATO | 4 | 0.04% | +0.4 ± 0.3 | +0.1 ± 0.0 |
| Southeast Asia Scoring | 6 | 0.06% | +0.4 ± 0.3 | +0.2 ± 0.3 |
| OPEC | 4 | 0.01% | +0.3 ± 0.1 | +0.3 ± 0.1 |

## large regret, AR card choice: the card the best move plays (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| The China Card | 32 | 0.24% | +2.6 ± 0.4 | +0.6 ± 0.3 |
| East European Unrest | 9 | 0.07% | +1.3 ± 0.2 | +0.4 ± 0.2 |
| Duck and Cover | 5 | 0.04% | +1.0 ± 0.2 | -0.6 ± 0.2 |
| Blockade | 4 | 0.08% | +1.0 ± 0.4 | -0.1 ± 0.1 |
| Our Man in Tehran | 7 | 0.07% | +0.9 ± 0.4 | +0.0 ± 0.1 |
| Korean War | 8 | 0.07% | +0.8 ± 0.3 | +0.1 ± 0.1 |
| Arms Race | 6 | 0.06% | +0.7 ± 0.4 | -0.1 ± 0.1 |
| U-2 Incident | 4 | 0.08% | +0.6 ± 0.4 | -0.0 ± 0.1 |
| Star Wars | 6 | 0.02% | +0.6 ± 0.0 | +0.3 ± 0.0 |
| Decolonization | 10 | 0.04% | +0.4 ± 0.1 | +0.4 ± 0.1 |
| Special Relationship | 9 | 0.04% | +0.3 ± 0.1 | +0.1 ± 0.1 |
| NORAD | 6 | 0.03% | +0.3 ± 0.1 | +0.3 ± 0.1 |

## large regret, play mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own card: INFL -> COUP | 6 | 0.06% | +1.8 ± 0.4 | -0.1 ± 0.1 |
| own card: INFL -> EVENT | 7 | 0.08% | +0.9 ± 0.4 | +0.5 ± 0.1 |
| opp card: SPACE -> INFL | 17 | 0.06% | +0.8 ± 0.2 | +0.4 ± 0.1 |
| own card: EVENT -> INFL | 29 | 0.10% | +0.7 ± 0.2 | +0.5 ± 0.1 |
| own card: COUP -> EVENT | 7 | 0.02% | +0.4 ± 0.1 | +0.5 ± 0.1 |
| own card: EVENT -> COUP | 6 | 0.02% | +0.4 ± 0.1 | +0.2 ± 0.1 |
| neutral card: INFL -> EVENT | 6 | 0.01% | +0.3 ± 0.1 | +0.3 ± 0.1 |
| opp card: COUP -> INFL | 9 | 0.03% | +0.2 ± 0.1 | +0.3 ± 0.1 |
| opp card: SPACE -> EVENT | 15 | 0.05% | +0.1 ± 0.1 | +0.1 ± 0.1 |
| own card: COUP -> REALIGN | 10 | 0.03% | +0.1 ± 0.1 | +0.2 ± 0.1 |

## large regret, play mode: the card (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| De Gaulle Leads France | 6 | 0.07% | +2.1 ± 0.4 | +0.1 ± 0.0 |
| Special Relationship | 10 | 0.08% | +1.7 ± 0.4 | +0.4 ± 0.1 |
| John Paul II Elected Pope | 9 | 0.05% | +0.7 ± 0.2 | +0.4 ± 0.1 |
| The Cambridge Five | 4 | 0.02% | +0.5 ± 0.1 | +0.1 ± 0.0 |
| Arab-Israeli War | 11 | 0.04% | +0.5 ± 0.1 | +0.6 ± 0.1 |
| East European Unrest | 11 | 0.04% | +0.3 ± 0.1 | +0.2 ± 0.1 |
| “One Small Step…” | 5 | 0.02% | +0.3 ± 0.1 | +0.3 ± 0.1 |
| OPEC | 13 | 0.05% | +0.3 ± 0.1 | +0.6 ± 0.1 |
| “An Evil Empire” | 5 | 0.02% | +0.2 ± 0.0 | +0.2 ± 0.0 |
| Solidarity | 4 | 0.01% | +0.2 ± 0.1 | +0.0 ± 0.0 |

## large regret, op mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| COUP -> INFL | 46 | 0.08% | +0.6 ± 0.2 | +0.3 ± 0.1 |
| COUP -> REALIGN | 56 | 0.10% | +0.5 ± 0.2 | +0.6 ± 0.1 |
| INFL -> COUP | 13 | 0.02% | +0.3 ± 0.0 | +0.2 ± 0.0 |
| REALIGN -> INFL | 1 | 0.00% | -0.0 ± 0.0 | -0.0 ± 0.0 |
| INFL -> REALIGN | 86 | 0.24% | -0.8 ± 0.4 | +0.0 ± 0.1 |

## large regret, influence targets: raw's region -> the best region

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Africa -> Asia | 5 | 0.13% | +3.2 ± 0.7 | +0.4 ± 0.1 |
| Africa -> Europe | 5 | 0.02% | +0.7 ± 0.0 | +0.7 ± 0.0 |
| Central America -> Central America | 6 | 0.04% | +0.7 ± 0.2 | +0.7 ± 0.2 |
| Asia -> Africa | 7 | 0.08% | +0.7 ± 0.3 | +0.5 ± 0.3 |
| Asia -> Asia | 8 | 0.19% | +0.6 ± 0.7 | +0.8 ± 0.4 |
| Middle East -> Asia | 4 | 0.04% | +0.4 ± 0.2 | +0.4 ± 0.2 |
| South America -> Europe | 6 | 0.04% | +0.4 ± 0.2 | +0.3 ± 0.1 |
| Asia -> Europe | 5 | 0.05% | +0.3 ± 0.1 | +0.2 ± 0.1 |
| Europe -> Middle East | 4 | 0.03% | +0.2 ± 0.1 | +0.3 ± 0.2 |
| Africa -> Middle East | 4 | 0.02% | +0.1 ± 0.1 | +0.1 ± 0.1 |

## large regret, headline: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 3 -> opp 1 | 4 | 0.01% | +0.6 ± 0.1 | +0.1 ± 0.0 |
| own 4 -> own 3 | 6 | 0.01% | +0.5 ± 0.1 | +0.2 ± 0.0 |
| neutral 4 -> own 2 | 5 | 0.01% | +0.4 ± 0.1 | +0.1 ± 0.0 |
| own 3 -> scoring 0 | 5 | 0.01% | +0.3 ± 0.0 | +0.3 ± 0.0 |
| neutral 4 -> own 3 | 5 | 0.01% | +0.2 ± 0.0 | +0.2 ± 0.0 |
| own 2 -> own 2 | 6 | 0.01% | +0.2 ± 0.0 | +0.1 ± 0.0 |
| neutral 2 -> own 2 | 6 | 0.02% | +0.1 ± 0.1 | +0.0 ± 0.0 |
| own 1 -> own 3 | 6 | 0.01% | +0.1 ± 0.0 | +0.1 ± 0.0 |

## The 20 positions that cost raw the most a game (large regret)

| turn/AR | side | DEFCON | kind | raw | best | gain a decision | a game |
|:---|:---|---:|:---|:---|:---|---:|---:|
| T5 AR4 | USSR | 2 | target (influence) | PointNode #53 (Nigeria) | PointNode #36 (Thailand) | +6.0 ± 1.7 | +2.15 |
| T2 AR3 | USSR | 3 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_COUP / OpMode: COUP | +9.1 ± 2.1 | +1.83 |
| T9 AR5 | USSR | 2 | AR card choice | SelectCard #13 (Arab-Israeli War) | SelectCard #4 (Duck and Cover) | +19.8 ± 1.7 | +1.46 |
| T10 AR5 | US | 2 | target (influence) | PointNode #79 (Brazil) | PointNode #32 (Pakistan) | +100.0 ± 0.0 | +1.28 |
| T3 AR5 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: EVENT | +5.3 ± 2.0 | +1.06 |
| T10 AR6 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN | +12.0 ± 0.9 | +1.05 |
| T2 AR4 | US | 2 | target (influence) | PointNode #32 (Pakistan) | PointNode #44 (South Korea) | +14.2 ± 2.0 | +0.98 |
| T8 AR3 | USSR | 2 | AR card choice | SelectCard #7 (Socialist Governments) | SelectCard #29 (East European Unrest) | +12.1 ± 2.0 | +0.89 |
| T6 AR3 | USSR | 2 | AR card choice | SelectCard #44 (Bear Trap) | SelectCard #108 (Our Man in Tehran) | +5.0 ± 2.1 | +0.85 |
| T4 AR1 | US | 2 | AR card choice | SelectCard #66 (Puppet Governments) | SelectCard #39 (Arms Race) | +4.9 ± 2.1 | +0.84 |
| T2 AR5 | US | 3 | AR card choice | SelectCard #35 (Formosan Resolution) | SelectCard #10 (Blockade) | +5.4 ± 1.8 | +0.78 |
| T3 AR3 | USSR | 2 | target (influence) | PointNode #21 (Lebanon) | PointNode #59 (Angola) | +3.4 ± 1.0 | +0.72 |
| T7 AR6 | US | 4 | AR card choice | SelectCard #74 (The Voice of America) | SelectCard #75 (Liberation Theology) | +6.2 ± 1.4 | +0.69 |
| T5 AR2 | US | 2 | target (influence) | PointNode #47 (Algeria) | PointNode #37 (Vietnam) | +3.5 ± 2.0 | +0.65 |
| T10 AR7 | US | 2 | target (influence) | PointNode #63 (South Africa) | PointNode #19 (Romania) | +50.0 ± 0.0 | +0.64 |
| T6 AR5 | USSR | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN | +12.1 ± 2.1 | +0.64 |
| T10 AR5 | US | 2 | target (influence) | PointNode #47 (Algeria) | PointNode #51 (Sudan) | +28.2 ± 1.5 | +0.60 |
| T7 AR0 | US | 3 | headline | SelectCard #31 (Red Scare/Purge) | SelectCard #70 (OAS Founded) | +14.7 ± 1.7 | +0.59 |
| T9 AR4 | US | 2 | AR card choice | SelectCard #85 (Star Wars) | SelectCard #11 (Korean War) | +10.5 ± 1.8 | +0.59 |
| T10 AR6 | US | 2 | target (influence) | PointNode #26 (Jordan) | PointNode #32 (Pakistan) | +26.9 ± 1.4 | +0.57 |
