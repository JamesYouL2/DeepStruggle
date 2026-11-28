# Where raw play loses: paired playouts by decision kind

Points of win probability a game, both sides' decisions summed (433.7 decisions with two or more legal moves a game): each playout row's paired gain of the best move over raw's, weighted back to the decisions raw play meets. 'k=8 @256 over raw' is what that searcher's own move recovers.

Screened census: the best move is chosen on the screen's pairs, and its gain measured on fresh pairs where the screen's lead held (escalated); other rows count as no loss, so the total is a lower bound. 'raw' is the move under study (annotate --player: the player's).

## total

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| escalated | 902 | 10.94% | +42.0 ± 3.2 | +47.8 ± 2.8 |
| not escalated | 7098 | 89.06% | +0.0 ± 0.0 | -146.1 ± 5.9 |

## all positions: decision kind, stratified bootstrap (2000 resamples)

Positions resampled within their (bank stratum, playout class) cell: the ± counts the sampling of positions as well as the playout noise the tables below print alone.

| kind | best over raw | share | k=8 @256 over raw | leaks the most in |
|:---|---:|---:|---:|---:|
| AR card choice | +20.1 ± 3.1 | 48% | -43.1 ± 6.0 | 93% |
| target (influence) | +11.9 ± 4.5 | 28% | -20.6 ± 7.0 | 7% |
| headline | +4.4 ± 0.7 | 10% | -8.3 ± 0.8 | 0% |
| play mode | +4.2 ± 1.6 | 10% | -17.4 ± 2.1 | 0% |
| op mode | +0.9 ± 0.2 | 2% | -4.2 ± 0.4 | 0% |
| target (coup) | +0.6 ± 0.6 | 1% | -2.2 ± 1.6 | 0% |
| choose_branch | +0.1 ± 0.0 | 0% | -0.4 ± 0.1 | 0% |
| target (realign) | +0.0 ± 0.3 | 0% | -2.0 ± 0.8 | 0% |
| all | +42.0 ± 5.8 |  | -98.3 ± 9.8 |  |

## all positions: decision kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| AR card choice | 1729 | 23.49% | +20.1 ± 1.2 | -43.1 ± 3.5 |
| target (influence) | 1474 | 42.36% | +11.9 ± 2.8 | -20.6 ± 5.1 |
| headline | 1435 | 4.18% | +4.4 ± 0.3 | -8.3 ± 0.4 |
| play mode | 1720 | 20.39% | +4.2 ± 0.8 | -17.4 ± 1.3 |
| op mode | 1134 | 3.30% | +0.9 ± 0.1 | -4.2 ± 0.3 |
| target (coup) | 135 | 4.11% | +0.6 ± 0.6 | -2.2 ± 1.5 |
| choose_branch | 260 | 0.32% | +0.1 ± 0.0 | -0.4 ± 0.1 |
| target (realign) | 113 | 1.85% | +0.0 ± 0.3 | -2.0 ± 0.7 |

## all positions: side x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| US AR card choice | 901 | 12.21% | +11.1 ± 0.8 | -26.0 ± 2.5 |
| USSR AR card choice | 828 | 11.28% | +9.0 ± 0.8 | -17.1 ± 2.5 |
| US target (influence) | 780 | 22.37% | +7.9 ± 2.1 | -18.5 ± 3.6 |
| USSR target (influence) | 694 | 19.99% | +4.0 ± 1.8 | -2.1 ± 3.5 |
| US headline | 731 | 2.15% | +2.6 ± 0.2 | -4.6 ± 0.3 |
| USSR play mode | 859 | 10.83% | +2.2 ± 0.6 | -6.6 ± 0.9 |
| US play mode | 861 | 9.55% | +2.0 ± 0.5 | -10.9 ± 1.0 |
| USSR headline | 704 | 2.03% | +1.8 ± 0.2 | -3.6 ± 0.3 |
| USSR target (coup) | 76 | 2.54% | +0.8 ± 0.5 | -1.9 ± 1.1 |
| USSR op mode | 539 | 1.59% | +0.4 ± 0.1 | -1.8 ± 0.2 |

## all positions: era x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| T8-10 AR card choice | 577 | 4.59% | +10.4 ± 0.4 | -7.5 ± 0.8 |
| T8-10 target (influence) | 471 | 6.70% | +5.5 ± 0.5 | -3.9 ± 1.0 |
| T1-3 AR card choice | 577 | 8.35% | +5.3 ± 0.7 | -15.5 ± 2.2 |
| T1-3 target (influence) | 539 | 20.85% | +4.5 ± 2.2 | -11.4 ± 4.0 |
| T4-7 AR card choice | 575 | 10.56% | +4.3 ± 0.8 | -20.1 ± 2.6 |
| T8-10 play mode | 573 | 4.08% | +2.6 ± 0.2 | -4.5 ± 0.4 |
| T4-7 play mode | 573 | 8.88% | +2.6 ± 0.6 | -7.2 ± 1.0 |
| T4-7 headline | 533 | 1.98% | +2.2 ± 0.2 | -3.7 ± 0.3 |
| T4-7 target (influence) | 464 | 14.81% | +1.9 ± 1.6 | -5.2 ± 2.9 |
| T8-10 headline | 390 | 0.78% | +1.5 ± 0.1 | -1.5 ± 0.1 |
| T1-3 headline | 512 | 1.41% | +0.6 ± 0.1 | -3.1 ± 0.3 |
| T1-3 target (coup) | 30 | 1.13% | +0.5 ± 0.2 | -1.7 ± 1.1 |

## all positions: DEFCON x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| D2 other | 2074 | 35.38% | +13.5 ± 2.4 | -13.0 ± 4.4 |
| D2 AR card choice | 1299 | 17.56% | +12.9 ± 0.9 | -32.4 ± 3.1 |
| D3 other | 1700 | 10.90% | +5.9 ± 0.9 | -18.8 ± 2.1 |
| D3 AR card choice | 225 | 2.93% | +5.3 ± 0.6 | -5.6 ± 1.3 |
| D2 play mode | 1221 | 15.38% | +3.8 ± 0.6 | -12.1 ± 1.1 |
| D4+ AR card choice | 205 | 3.00% | +2.0 ± 0.4 | -5.0 ± 1.2 |
| D3 play mode | 271 | 2.85% | +0.6 ± 0.3 | -3.1 ± 0.6 |
| D4+ play mode | 228 | 2.16% | -0.3 ± 0.5 | -2.2 ± 0.5 |
| D4+ other | 777 | 9.84% | -1.6 ± 1.2 | -5.9 ± 2.1 |

## all positions, AR card choice: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| opp 3 -> opp 2 | 9 | 0.15% | +1.7 ± 0.2 | +0.7 ± 0.4 |
| neutral 3 -> opp 3 | 5 | 0.06% | +1.4 ± 0.1 | +0.4 ± 0.1 |
| opp 2 -> opp 2 | 229 | 2.82% | +1.4 ± 0.2 | -10.4 ± 1.4 |
| opp 3 -> opp 3 | 199 | 2.76% | +0.8 ± 0.1 | -6.3 ± 1.2 |
| opp 3 -> own 3 | 7 | 0.09% | +0.5 ± 0.1 | +0.5 ± 0.2 |
| own 3 -> opp 2 | 7 | 0.06% | +0.5 ± 0.1 | -0.1 ± 0.2 |
| own 3 -> own 3 | 173 | 2.73% | +0.4 ± 0.1 | -4.1 ± 1.1 |
| opp 2 -> own 2 | 5 | 0.06% | +0.4 ± 0.1 | +0.5 ± 0.2 |
| opp 2 -> opp 3 | 5 | 0.04% | +0.3 ± 0.1 | +0.6 ± 0.2 |
| neutral 2 -> own 2 | 5 | 0.02% | +0.3 ± 0.0 | +0.1 ± 0.1 |
| neutral 2 -> opp 3 | 5 | 0.02% | +0.3 ± 0.1 | +0.1 ± 0.1 |
| own 2 -> opp 3 | 5 | 0.05% | +0.3 ± 0.2 | +0.7 ± 0.3 |

## all positions, AR card choice: the card raw plays instead (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Africa Scoring | 18 | 0.29% | +1.4 ± 0.1 | +0.3 ± 0.2 |
| Marshall Plan | 17 | 0.28% | +1.1 ± 0.3 | -0.7 ± 0.4 |
| The China Card | 43 | 0.56% | +1.0 ± 0.2 | -0.3 ± 0.8 |
| The Reformer | 14 | 0.11% | +0.8 ± 0.1 | -0.3 ± 0.2 |
| NATO | 23 | 0.46% | +0.8 ± 0.3 | +0.4 ± 0.5 |
| East European Unrest | 31 | 0.45% | +0.8 ± 0.2 | -0.4 ± 0.7 |
| UN Intervention | 22 | 0.34% | +0.8 ± 0.2 | -0.2 ± 0.3 |
| Summit | 24 | 0.28% | +0.7 ± 0.1 | -0.3 ± 0.4 |
| SALT Negotiations | 13 | 0.28% | +0.7 ± 0.2 | -1.0 ± 0.4 |
| Arms Race | 12 | 0.16% | +0.6 ± 0.1 | -0.2 ± 0.3 |
| The Iron Lady | 13 | 0.12% | +0.6 ± 0.1 | +0.1 ± 0.1 |
| U-2 Incident | 18 | 0.37% | +0.6 ± 0.1 | +0.3 ± 0.1 |

## all positions, AR card choice: the card the best move plays (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Tear Down this Wall | 13 | 0.11% | +1.6 ± 0.0 | +1.6 ± 0.2 |
| Che | 15 | 0.18% | +1.2 ± 0.0 | +0.8 ± 0.2 |
| Arab-Israeli War | 26 | 0.36% | +1.1 ± 0.1 | +0.8 ± 0.5 |
| Blockade | 21 | 0.24% | +1.0 ± 0.2 | -0.7 ± 0.3 |
| Special Relationship | 36 | 0.51% | +1.0 ± 0.2 | -1.1 ± 0.7 |
| The China Card | 49 | 0.65% | +0.9 ± 0.3 | -1.4 ± 0.8 |
| Our Man in Tehran | 16 | 0.24% | +0.9 ± 0.2 | -1.8 ± 0.6 |
| South African Unrest | 22 | 0.22% | +0.8 ± 0.2 | -0.0 ± 0.3 |
| Puppet Governments | 9 | 0.12% | +0.7 ± 0.1 | +0.2 ± 0.4 |
| Warsaw Pact Formed | 27 | 0.34% | +0.7 ± 0.2 | -0.2 ± 0.4 |
| Fidel | 25 | 0.32% | +0.6 ± 0.1 | -0.4 ± 0.5 |
| Decolonization | 29 | 0.44% | +0.5 ± 0.1 | -0.7 ± 0.4 |

## all positions, play mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own card: INFL -> EVENT | 4 | 0.03% | +0.4 ± 0.1 | +0.2 ± 0.1 |
| own card: EVENT -> COUP | 6 | 0.05% | +0.4 ± 0.2 | +0.4 ± 0.1 |
| own card: COUP -> REALIGN | 5 | 0.06% | +0.3 ± 0.3 | +0.2 ± 0.1 |
| neutral card: EVENT -> COUP | 4 | 0.03% | +0.2 ± 0.1 | +0.0 ± 0.0 |
| opp card: EVENT -> SPACE | 4 | 0.01% | +0.2 ± 0.0 | +0.1 ± 0.1 |
| opp card: INFL -> EVENT | 6 | 0.04% | +0.1 ± 0.1 | +0.2 ± 0.1 |
| own card: COUP -> INFL | 6 | 0.03% | +0.1 ± 0.1 | +0.4 ± 0.2 |
| opp card: EVENT -> COUP | 4 | 0.03% | +0.0 ± 0.1 | +0.3 ± 0.1 |
| opp card: EVENT -> INFL | 5 | 0.03% | +0.0 ± 0.1 | +0.4 ± 0.2 |
| opp card: COUP -> INFL | 7 | 0.05% | +0.0 ± 0.1 | +0.1 ± 0.0 |

## all positions, play mode: the card (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Muslim Revolution | 18 | 0.32% | +0.7 ± 0.3 | -0.1 ± 0.1 |
| How I Learned to Stop Worrying | 24 | 0.27% | +0.6 ± 0.1 | -0.1 ± 0.2 |
| Sadat Expels Soviets | 23 | 0.27% | +0.6 ± 0.3 | -0.2 ± 0.1 |
| Pershing II Deployed | 5 | 0.07% | +0.5 ± 0.1 | -0.0 ± 0.0 |
| East European Unrest | 29 | 0.39% | +0.4 ± 0.3 | -0.1 ± 0.1 |
| Grain Sales to Soviets | 14 | 0.20% | +0.3 ± 0.0 | +0.3 ± 0.1 |
| Quagmire | 8 | 0.05% | +0.3 ± 0.0 | -0.0 ± 0.1 |
| Our Man in Tehran | 24 | 0.23% | +0.3 ± 0.1 | -0.4 ± 0.3 |
| Marshall Plan | 13 | 0.11% | +0.3 ± 0.1 | -0.5 ± 0.2 |
| “An Evil Empire” | 9 | 0.05% | +0.2 ± 0.0 | -0.1 ± 0.0 |

## all positions, op mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| COUP -> INFL | 24 | 0.05% | +0.5 ± 0.1 | +0.2 ± 0.1 |
| COUP -> REALIGN | 17 | 0.04% | +0.1 ± 0.1 | +0.2 ± 0.0 |
| INFL -> COUP | 14 | 0.02% | +0.1 ± 0.0 | +0.2 ± 0.0 |
| INFL -> REALIGN | 19 | 0.04% | +0.1 ± 0.1 | +0.2 ± 0.0 |
| REALIGN -> INFL | 2 | 0.00% | +0.0 ± 0.0 | +0.0 ± 0.0 |
| COUP -> COUP | 459 | 1.08% | +0.0 ± 0.0 | -2.7 ± 0.2 |

## all positions, influence targets: raw's region -> the best region

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Asia -> Asia | 282 | 8.73% | +2.8 ± 0.7 | -12.0 ± 2.0 |
| Asia -> Middle East | 10 | 0.28% | +1.2 ± 0.4 | +2.0 ± 0.7 |
| Africa -> Asia | 13 | 0.21% | +1.2 ± 0.4 | +0.3 ± 0.2 |
| Africa -> Africa | 179 | 4.77% | +1.0 ± 0.3 | -5.4 ± 1.9 |
| Asia -> Africa | 4 | 0.11% | +0.8 ± 0.5 | +0.9 ± 0.3 |
| Europe -> Europe | 390 | 12.60% | +0.8 ± 1.3 | -11.8 ± 2.6 |
| Middle East -> Europe | 10 | 0.18% | +0.7 ± 0.4 | +0.6 ± 0.4 |
| Central America -> South America | 4 | 0.16% | +0.7 ± 0.4 | +1.3 ± 0.7 |
| South America -> Africa | 6 | 0.19% | +0.6 ± 0.6 | +0.7 ± 0.3 |
| Africa -> Central America | 4 | 0.09% | +0.4 ± 0.2 | +1.1 ± 0.4 |

## all positions, headline: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 3 -> own 2 | 13 | 0.03% | +0.5 ± 0.1 | +0.3 ± 0.0 |
| neutral 3 -> scoring 0 | 4 | 0.01% | +0.4 ± 0.1 | -0.0 ± 0.0 |
| own 3 -> scoring 0 | 5 | 0.03% | +0.4 ± 0.1 | +0.0 ± 0.0 |
| own 1 -> own 3 | 5 | 0.01% | +0.2 ± 0.0 | +0.0 ± 0.0 |
| own 3 -> own 1 | 7 | 0.03% | +0.2 ± 0.1 | +0.0 ± 0.0 |
| own 2 -> scoring 0 | 4 | 0.02% | +0.2 ± 0.0 | +0.0 ± 0.0 |
| neutral 4 -> neutral 4 | 109 | 0.38% | +0.1 ± 0.0 | -0.2 ± 0.1 |
| own 3 -> own 3 | 298 | 0.93% | +0.1 ± 0.0 | -1.5 ± 0.2 |

## The 20 positions that cost raw the most a game (all positions)

| turn/AR | side | DEFCON | kind | raw | best | gain a decision | a game |
|:---|:---|---:|:---|:---|:---|---:|---:|
| T2 AR1 | US | 2 | target (influence) | PointNode #39 (Indonesia) | PointNode #36 (Thailand) | +8.5 ± 1.3 | +1.91 |
| T10 AR2 | USSR | 2 | AR card choice | SelectCard #79 (Africa Scoring) | SelectCard #96 (Tear Down this Wall) | +71.1 ± 1.0 | +1.32 |
| T10 AR5 | US | 2 | target (influence) | PointNode #30 (Libya) | PointNode #74 (Colombia) | +6.5 ± 0.7 | +0.89 |
| T1 AR0 | US | 5 | target (influence) | PointNode #10 (Italy) | PointNode #8 (France) | +2.1 ± 1.5 | +0.84 |
| T10 AR0 | US | 2 | target (influence) | PointNode #74 (Colombia) | PointNode #75 (Ecuador) | +8.7 ± 0.9 | +0.78 |
| T4 AR2 | US | 2 | target (influence) | PointNode #30 (Libya) | PointNode #25 (Iran) | +2.3 ± 1.0 | +0.78 |
| T6 AR6 | US | 2 | target (influence) | PointNode #7 (West Germany) | PointNode #6 (Benelux) | +2.2 ± 1.0 | +0.75 |
| T10 AR4 | US | 2 | AR card choice | SelectCard #87 (The Reformer) | SelectCard #13 (Arab-Israeli War) | +14.1 ± 0.9 | +0.73 |
| T2 AR2 | US | 4 | AR card choice | SelectCard #32 (UN Intervention) | SelectCard #10 (Blockade) | +9.4 ± 1.5 | +0.72 |
| T6 AR5 | US | 2 | AR card choice | SelectCard #60 (U-2 Incident) | SelectCard #107 (Che) | +28.4 ± 1.0 | +0.71 |
| T6 AR6 | USSR | 2 | target (influence) | PointNode #53 (Nigeria) | PointNode #61 (SE African States) | +3.8 ± 1.2 | +0.68 |
| T6 AR3 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: SPACE | +3.3 ± 1.5 | +0.66 |
| T2 AR3 | USSR | 2 | target (influence) | PointNode #44 (South Korea) | PointNode #30 (Libya) | +2.9 ± 1.4 | +0.65 |
| T2 AR2 | US | 2 | target (influence) | PointNode #44 (South Korea) | PointNode #47 (Algeria) | +10.3 ± 1.5 | +0.64 |
| T10 AR4 | US | 3 | AR card choice | SelectCard #39 (Arms Race) | SelectCard #61 (OPEC) | +12.2 ± 1.0 | +0.64 |
| T3 AR2 | US | 2 | target (influence) | PointNode #77 (Venezuela) | PointNode #59 (Angola) | +1.6 ± 1.5 | +0.63 |
| T9 AR7 | USSR | 2 | AR card choice | SelectCard #6 (The China Card) | SelectCard #66 (Puppet Governments) | +11.5 ± 1.1 | +0.60 |
| T5 AR1 | US | 3 | target (influence) | PointNode #39 (Indonesia) | PointNode #36 (Thailand) | +3.3 ± 0.9 | +0.59 |
| T4 AR3 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: EVENT | +10.1 ± 1.5 | +0.57 |
| T7 AR2 | USSR | 3 | AR card choice | SelectCard #29 (East European Unrest) | SelectCard #108 (Our Man in Tehran) | +4.9 ± 1.5 | +0.56 |
