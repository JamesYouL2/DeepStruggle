# Where raw play loses: paired playouts by decision kind

Points of win probability a game, both sides' decisions summed (450.5 decisions with two or more legal moves a game): each playout row's paired gain of the best move over raw's, weighted back to the decisions raw play meets. 'k=8 @256 over raw' is what that searcher's own move recovers.

Screened census: the best move is chosen on the screen's pairs, and its gain measured on fresh pairs where the screen's lead held (escalated); other rows count as no loss, so the total is a lower bound. 'raw' is the move under study (annotate --player: the player's).

## total

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| escalated | 1122 | 12.02% | +86.6 ± 3.3 | +57.9 ± 2.5 |
| not escalated | 6878 | 87.98% | +0.0 ± 0.0 | -125.0 ± 5.7 |

## all positions: decision kind, stratified bootstrap (2000 resamples)

Positions resampled within their (bank stratum, playout class) cell: the ± counts the sampling of positions as well as the playout noise the tables below print alone.

| kind | best over raw | share | k=8 @256 over raw | leaks the most in |
|:---|---:|---:|---:|---:|
| AR card choice | +36.6 ± 4.8 | 42% | -27.4 ± 5.4 | 93% |
| target (influence) | +25.7 ± 5.8 | 30% | -21.1 ± 7.4 | 7% |
| play mode | +10.7 ± 2.7 | 12% | -6.3 ± 2.3 | 0% |
| headline | +8.7 ± 1.1 | 10% | -3.6 ± 1.1 | 0% |
| op mode | +2.6 ± 0.9 | 3% | -4.6 ± 0.7 | 0% |
| target (coup) | +1.1 ± 1.2 | 1% | -2.6 ± 1.1 | 0% |
| target (realign) | +0.9 ± 0.4 | 1% | -1.5 ± 1.1 | 0% |
| choose_branch | +0.3 ± 0.2 | 0% | +0.0 ± 0.1 | 0% |
| all | +86.6 ± 8.4 |  | -67.1 ± 9.6 |  |

## all positions: decision kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| AR card choice | 1719 | 23.14% | +36.6 ± 1.4 | -27.4 ± 3.3 |
| target (influence) | 1462 | 41.32% | +25.7 ± 2.6 | -21.1 ± 4.8 |
| play mode | 1711 | 20.30% | +10.7 ± 1.2 | -6.3 ± 1.4 |
| headline | 1404 | 4.01% | +8.7 ± 0.3 | -3.6 ± 0.4 |
| op mode | 1189 | 4.20% | +2.6 ± 0.2 | -4.6 ± 0.3 |
| target (coup) | 116 | 4.02% | +1.1 ± 0.5 | -2.6 ± 1.1 |
| target (realign) | 140 | 2.69% | +0.9 ± 0.4 | -1.5 ± 1.1 |
| choose_branch | 259 | 0.31% | +0.3 ± 0.0 | +0.0 ± 0.1 |

## all positions: side x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| US AR card choice | 882 | 11.92% | +20.3 ± 1.0 | -13.0 ± 2.4 |
| USSR AR card choice | 837 | 11.23% | +16.3 ± 1.0 | -14.4 ± 2.3 |
| US target (influence) | 790 | 21.86% | +14.3 ± 1.8 | -10.2 ± 3.6 |
| USSR target (influence) | 672 | 19.46% | +11.4 ± 1.9 | -11.0 ± 3.2 |
| USSR play mode | 879 | 10.29% | +6.2 ± 0.9 | -3.0 ± 1.0 |
| US headline | 755 | 2.23% | +4.7 ± 0.2 | -2.9 ± 0.3 |
| US play mode | 832 | 10.01% | +4.5 ± 0.7 | -3.4 ± 1.0 |
| USSR headline | 649 | 1.78% | +3.9 ± 0.2 | -0.7 ± 0.3 |
| US op mode | 587 | 2.05% | +1.8 ± 0.1 | -2.7 ± 0.2 |
| USSR target (coup) | 74 | 2.82% | +1.1 ± 0.5 | -1.2 ± 0.7 |

## all positions: era x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| T4-7 AR card choice | 573 | 10.69% | +14.4 ± 1.0 | -7.2 ± 2.6 |
| T8-10 AR card choice | 573 | 4.42% | +12.3 ± 0.3 | -8.1 ± 0.7 |
| T8-10 target (influence) | 450 | 6.74% | +11.5 ± 0.6 | -5.6 ± 0.9 |
| T1-3 AR card choice | 573 | 8.04% | +9.9 ± 0.9 | -12.1 ± 2.0 |
| T1-3 target (influence) | 536 | 19.45% | +7.5 ± 2.0 | -10.0 ± 3.8 |
| T4-7 target (influence) | 476 | 15.13% | +6.8 ± 1.5 | -5.5 ± 2.7 |
| T8-10 play mode | 570 | 4.06% | +4.0 ± 0.2 | -2.9 ± 0.4 |
| T4-7 play mode | 569 | 9.04% | +3.7 ± 1.0 | -1.7 ± 1.0 |
| T8-10 headline | 380 | 0.86% | +3.1 ± 0.1 | -0.6 ± 0.1 |
| T1-3 play mode | 572 | 7.21% | +3.0 ± 0.6 | -1.8 ± 0.9 |
| T4-7 headline | 521 | 1.79% | +2.8 ± 0.2 | -1.5 ± 0.3 |
| T1-3 headline | 503 | 1.35% | +2.8 ± 0.2 | -1.5 ± 0.3 |

## all positions: DEFCON x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| D2 AR card choice | 1277 | 17.21% | +26.9 ± 1.1 | -18.4 ± 2.9 |
| D2 other | 2011 | 33.99% | +20.8 ± 2.1 | -18.9 ± 4.2 |
| D3 other | 1743 | 11.48% | +15.5 ± 1.1 | -6.6 ± 1.8 |
| D2 play mode | 1189 | 15.32% | +6.6 ± 1.0 | -3.8 ± 1.2 |
| D3 AR card choice | 252 | 3.25% | +5.1 ± 0.5 | -4.9 ± 1.2 |
| D4+ AR card choice | 190 | 2.69% | +4.6 ± 0.6 | -4.0 ± 1.0 |
| D3 play mode | 294 | 2.79% | +3.3 ± 0.5 | -1.4 ± 0.6 |
| D4+ other | 816 | 11.08% | +3.1 ± 1.3 | -8.0 ± 2.1 |
| D4+ play mode | 228 | 2.20% | +0.8 ± 0.3 | -1.2 ± 0.6 |

## all positions, AR card choice: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 3 -> own 3 | 161 | 2.60% | +1.9 ± 0.1 | -3.3 ± 0.8 |
| opp 2 -> opp 2 | 216 | 2.92% | +1.8 ± 0.3 | -6.4 ± 1.2 |
| own 2 -> neutral 2 | 6 | 0.05% | +1.5 ± 0.1 | +0.3 ± 0.1 |
| own 2 -> opp 3 | 6 | 0.06% | +1.2 ± 0.2 | +1.4 ± 0.3 |
| neutral 2 -> opp 2 | 6 | 0.08% | +0.7 ± 0.2 | +0.2 ± 0.1 |
| own 3 -> neutral 2 | 5 | 0.04% | +0.7 ± 0.1 | +0.1 ± 0.1 |
| scoring 0 -> opp 3 | 6 | 0.04% | +0.7 ± 0.1 | +0.4 ± 0.2 |
| opp 3 -> opp 3 | 218 | 2.91% | +0.6 ± 0.2 | -3.7 ± 1.3 |
| opp 3 -> opp 1 | 5 | 0.06% | +0.6 ± 0.1 | +0.7 ± 0.2 |
| own 3 -> opp 2 | 5 | 0.08% | +0.4 ± 0.3 | -0.0 ± 0.0 |
| opp 2 -> scoring 0 | 5 | 0.03% | +0.4 ± 0.1 | +0.3 ± 0.2 |
| scoring 0 -> opp 2 | 8 | 0.13% | +0.4 ± 0.3 | +0.3 ± 0.1 |

## all positions, AR card choice: the card raw plays instead (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Shuttle Diplomacy | 15 | 0.28% | +2.2 ± 0.2 | +0.2 ± 0.4 |
| Alliance for Progress | 17 | 0.30% | +2.0 ± 0.2 | -0.4 ± 0.3 |
| Europe Scoring | 31 | 0.38% | +1.8 ± 0.3 | -0.6 ± 0.3 |
| Arab-Israeli War | 28 | 0.27% | +1.6 ± 0.1 | +0.5 ± 0.5 |
| Solidarity | 17 | 0.11% | +1.4 ± 0.1 | -0.3 ± 0.1 |
| Five Year Plan | 26 | 0.46% | +1.2 ± 0.2 | +0.3 ± 0.6 |
| Sadat Expels Soviets | 14 | 0.24% | +1.1 ± 0.2 | -0.3 ± 0.3 |
| Special Relationship | 33 | 0.41% | +1.0 ± 0.1 | +0.0 ± 0.4 |
| NATO | 23 | 0.47% | +1.0 ± 0.3 | -0.4 ± 0.2 |
| Olympic Games | 30 | 0.40% | +0.9 ± 0.3 | -0.7 ± 0.5 |
| Che | 13 | 0.23% | +0.9 ± 0.2 | -0.1 ± 0.3 |
| The China Card | 40 | 0.54% | +0.9 ± 0.2 | -3.3 ± 0.6 |

## all positions, AR card choice: the card the best move plays (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| The China Card | 49 | 0.69% | +4.7 ± 0.3 | -3.2 ± 0.7 |
| Central America Scoring | 26 | 0.42% | +2.3 ± 0.1 | +0.6 ± 0.4 |
| Ussuri River Skirmish | 9 | 0.13% | +2.1 ± 0.2 | +0.3 ± 0.4 |
| Che | 15 | 0.23% | +1.7 ± 0.2 | +1.1 ± 0.4 |
| Romanian Abdication | 25 | 0.34% | +1.7 ± 0.3 | -0.4 ± 0.5 |
| Duck and Cover | 31 | 0.44% | +1.6 ± 0.3 | -1.0 ± 0.5 |
| Missile Envy | 8 | 0.11% | +1.5 ± 0.1 | +0.0 ± 0.2 |
| Africa Scoring | 14 | 0.21% | +1.4 ± 0.1 | -0.1 ± 0.1 |
| Decolonization | 17 | 0.24% | +1.1 ± 0.3 | +0.6 ± 0.3 |
| Panama Canal Returned | 11 | 0.21% | +1.0 ± 0.3 | -0.2 ± 0.5 |
| East European Unrest | 34 | 0.47% | +1.0 ± 0.2 | -0.8 ± 0.5 |
| Vietnam Revolts | 16 | 0.26% | +1.0 ± 0.3 | +0.2 ± 0.3 |

## all positions, play mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own card: EVENT -> INFL | 18 | 0.08% | +1.2 ± 0.1 | +1.7 ± 0.2 |
| opp card: SPACE -> INFL | 11 | 0.10% | +1.1 ± 0.3 | +0.5 ± 0.1 |
| own card: EVENT -> COUP | 8 | 0.07% | +1.0 ± 0.3 | +0.5 ± 0.1 |
| neutral card: EVENT -> COUP | 10 | 0.08% | +0.9 ± 0.2 | +0.7 ± 0.1 |
| opp card: SPACE -> EVENT | 13 | 0.04% | +0.8 ± 0.1 | +1.2 ± 0.2 |
| neutral card: COUP -> EVENT | 4 | 0.06% | +0.7 ± 0.3 | +0.1 ± 0.0 |
| own card: INFL -> COUP | 8 | 0.04% | +0.7 ± 0.1 | +0.8 ± 0.2 |
| opp card: REALIGN -> INFL | 5 | 0.04% | +0.5 ± 0.1 | +0.1 ± 0.0 |
| own card: EVENT -> REALIGN | 9 | 0.08% | +0.5 ± 0.3 | +0.4 ± 0.1 |
| opp card: EVENT -> INFL | 11 | 0.14% | +0.4 ± 0.4 | +0.1 ± 0.1 |

## all positions, play mode: the card (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| “One Small Step…” | 16 | 0.24% | +1.2 ± 0.3 | -0.2 ± 0.1 |
| Liberation Theology | 5 | 0.17% | +0.9 ± 0.5 | +0.0 ± 0.0 |
| Defectors | 24 | 0.27% | +0.7 ± 0.3 | +0.0 ± 0.2 |
| Wargames | 8 | 0.12% | +0.6 ± 0.1 | -0.4 ± 0.0 |
| Quagmire | 8 | 0.05% | +0.5 ± 0.1 | +0.1 ± 0.1 |
| Suez Crisis | 14 | 0.12% | +0.5 ± 0.1 | -0.0 ± 0.2 |
| The Cambridge Five | 45 | 0.34% | +0.4 ± 0.1 | +0.3 ± 0.2 |
| Cuban Missile Crisis | 13 | 0.04% | +0.4 ± 0.0 | +0.4 ± 0.1 |
| U-2 Incident | 14 | 0.20% | +0.4 ± 0.1 | +0.3 ± 0.1 |
| Colonial Rear Guards | 16 | 0.27% | +0.4 ± 0.0 | -0.1 ± 0.1 |

## all positions, op mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| INFL -> REALIGN | 16 | 0.05% | +1.0 ± 0.1 | +0.2 ± 0.0 |
| COUP -> INFL | 33 | 0.06% | +0.7 ± 0.1 | +0.7 ± 0.1 |
| REALIGN -> COUP | 7 | 0.01% | +0.4 ± 0.0 | -0.0 ± 0.0 |
| INFL -> COUP | 22 | 0.04% | +0.4 ± 0.1 | +0.3 ± 0.0 |
| COUP -> REALIGN | 11 | 0.03% | +0.1 ± 0.1 | +0.1 ± 0.0 |
| 208 -> COUP | 1 | 0.00% | +0.0 ± 0.0 | -0.0 ± 0.0 |

## all positions, influence targets: raw's region -> the best region

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Asia -> Asia | 269 | 8.21% | +4.1 ± 1.1 | -11.3 ± 2.5 |
| Africa -> Africa | 165 | 3.60% | +3.5 ± 0.6 | -3.5 ± 1.6 |
| Europe -> Europe | 449 | 14.82% | +3.3 ± 1.3 | -7.0 ± 2.5 |
| Middle East -> Europe | 6 | 0.13% | +2.1 ± 0.4 | +0.3 ± 0.2 |
| Africa -> Asia | 8 | 0.09% | +1.9 ± 0.2 | +0.3 ± 0.2 |
| South America -> Central America | 6 | 0.16% | +1.7 ± 0.5 | +0.7 ± 0.2 |
| Europe -> Africa | 8 | 0.20% | +1.6 ± 0.6 | +0.7 ± 0.3 |
| South America -> Middle East | 5 | 0.14% | +1.5 ± 0.4 | +1.2 ± 0.6 |
| Africa -> Europe | 8 | 0.15% | +1.3 ± 0.5 | +0.6 ± 0.4 |
| Middle East -> Middle East | 118 | 3.06% | +1.2 ± 0.3 | +0.2 ± 1.6 |

## all positions, headline: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 2 -> own 3 | 11 | 0.03% | +0.5 ± 0.1 | +0.1 ± 0.0 |
| own 3 -> neutral 2 | 5 | 0.02% | +0.5 ± 0.0 | +0.1 ± 0.0 |
| own 2 -> neutral 2 | 11 | 0.02% | +0.5 ± 0.0 | +0.3 ± 0.0 |
| own 2 -> scoring 0 | 8 | 0.02% | +0.4 ± 0.0 | +0.1 ± 0.0 |
| own 3 -> scoring 0 | 5 | 0.02% | +0.3 ± 0.1 | +0.0 ± 0.0 |
| own 1 -> scoring 0 | 9 | 0.01% | +0.3 ± 0.0 | +0.2 ± 0.0 |
| neutral 2 -> own 2 | 16 | 0.05% | +0.3 ± 0.1 | +0.2 ± 0.0 |
| own 3 -> own 2 | 8 | 0.02% | +0.3 ± 0.0 | +0.2 ± 0.0 |

## The 20 positions that cost raw the most a game (all positions)

| turn/AR | side | DEFCON | kind | raw | best | gain a decision | a game |
|:---|:---|---:|:---|:---|:---|---:|---:|
| T7 AR3 | USSR | 2 | AR card choice | SelectCard #73 (Shuttle Diplomacy) | SelectCard #6 (The China Card) | +18.9 ± 1.2 | +2.20 |
| T3 AR1 | US | 3 | target (influence) | PointNode #33 (India) | PointNode #38 (Malaysia) | +4.9 ± 1.3 | +1.95 |
| T10 AR4 | US | 2 | target (influence) | PointNode #53 (Nigeria) | PointNode #38 (Malaysia) | +11.8 ± 0.9 | +1.95 |
| T10 AR1 | US | 4 | AR card choice | SelectCard #78 (Alliance for Progress) | SelectCard #76 (Ussuri River Skirmish) | +31.9 ± 1.0 | +1.78 |
| T1 AR2 | USSR | 4 | target (influence) | PointNode #24 (Iraq) | PointNode #18 (Yugoslavia) | +7.9 ± 1.5 | +1.67 |
| T7 AR3 | US | 2 | target (influence) | PointNode #81 (Chile) | PointNode #71 (Cuba) | +4.7 ± 1.3 | +1.67 |
| T7 AR2 | USSR | 2 | target (influence) | PointNode #57 (Zaire) | PointNode #63 (South Africa) | +4.2 ± 1.3 | +1.51 |
| T10 AR3 | US | 2 | AR card choice | SelectCard #101 (Solidarity) | SelectCard #49 (Missile Envy) | +20.7 ± 1.3 | +1.35 |
| T2 AR1 | USSR | 3 | target (coup) | PointNode #25 (Iran) | PointNode #70 (Panama) | +3.1 ± 1.3 | +1.25 |
| T10 AR2 | USSR | 2 | AR card choice | SelectCard #13 (Arab-Israeli War) | SelectCard #37 (Central America Scoring) | +18.9 ± 0.7 | +1.06 |
| T1 AR2 | US | 3 | AR card choice | SelectCard #2 (Europe Scoring) | SelectCard #4 (Duck and Cover) | +8.1 ± 1.4 | +1.04 |
| T10 AR2 | US | 2 | target (influence) | PointNode #15 (Poland) | PointNode #11 (Greece) | +50.0 ± 0.7 | +0.99 |
| T8 AR4 | US | 2 | target (influence) | PointNode #81 (Chile) | PointNode #28 (Saudi Arabia) | +5.8 ± 1.3 | +0.96 |
| T4 AR4 | USSR | 2 | AR card choice | SelectCard #107 (Che) | SelectCard #12 (Romanian Abdication) | +5.8 ± 1.2 | +0.91 |
| T7 AR6 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN | +4.1 ± 1.3 | +0.91 |
| T10 AR4 | US | 2 | AR card choice | SelectCard #5 (Five Year Plan) | SelectCard #79 (Africa Scoring) | +15.5 ± 1.2 | +0.87 |
| T7 AR0 | USSR | 3 | target (influence) | PointNode #44 (South Korea) | PointNode #82 (Argentina) | +14.8 ± 1.2 | +0.81 |
| T7 AR1 | US | 4 | target (influence) | PointNode #57 (Zaire) | PointNode #51 (Sudan) | +4.2 ± 1.4 | +0.79 |
| T6 AR1 | USSR | 3 | play mode | Resolution: OPS_COUP / OpMode: COUP | Resolution: EVENT | +3.6 ± 1.3 | +0.79 |
| T7 AR0 | US | 2 | op mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN | +16.2 ± 1.2 | +0.79 |
