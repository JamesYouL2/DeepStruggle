# Where raw play loses: paired playouts by decision kind

Points of win probability a game, both sides' decisions summed (459.9 decisions with two or more legal moves a game): each playout row's paired gain of the best move over raw's, weighted back to the decisions raw play meets. 'k=8 @256 over raw' is what that searcher's own move recovers. 'Large' is a search-valued raw regret of 2+ points, where the gain is (research/log/E7_search_disagreement_bank.md); the rest sits at its noise.

## total

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| large regret (2+ points) | 1310 | 7.27% | +39.0 ± 3.3 | +22.9 ± 1.2 |
| the rest | 1071 | 92.82% | -17.4 ± 15.2 | -6.1 ± 8.3 |

## large regret: decision kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| target (influence) | 126 | 1.56% | +17.8 ± 2.0 | +7.9 ± 0.7 |
| AR card choice | 348 | 2.70% | +8.9 ± 1.7 | +6.2 ± 0.8 |
| play mode | 357 | 2.05% | +6.7 ± 1.8 | +5.6 ± 0.4 |
| target (realign) | 1 | 0.02% | +2.9 ± 0.1 | +0.0 ± 0.0 |
| headline | 268 | 0.43% | +1.3 ± 0.3 | +1.6 ± 0.1 |
| op mode | 191 | 0.39% | +0.9 ± 0.4 | +1.1 ± 0.1 |
| target (coup) | 13 | 0.11% | +0.3 ± 0.3 | +0.3 ± 0.2 |
| choose_branch | 6 | 0.01% | +0.1 ± 0.0 | +0.1 ± 0.0 |

## large regret: side x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| USSR target (influence) | 52 | 0.59% | +10.8 ± 1.2 | +3.2 ± 0.5 |
| US AR card choice | 165 | 1.23% | +7.2 ± 1.1 | +6.6 ± 0.6 |
| US target (influence) | 74 | 0.97% | +7.0 ± 1.6 | +4.8 ± 0.6 |
| US play mode | 198 | 1.17% | +5.4 ± 1.4 | +2.1 ± 0.3 |
| USSR target (realign) | 1 | 0.02% | +2.9 ± 0.1 | +0.0 ± 0.0 |
| USSR AR card choice | 183 | 1.46% | +1.7 ± 1.3 | -0.4 ± 0.6 |
| USSR play mode | 159 | 0.88% | +1.3 ± 1.2 | +3.5 ± 0.2 |
| US headline | 164 | 0.28% | +1.1 ± 0.3 | +1.2 ± 0.1 |
| US op mode | 93 | 0.21% | +0.6 ± 0.3 | +0.9 ± 0.1 |
| US target (coup) | 7 | 0.07% | +0.3 ± 0.3 | +0.3 ± 0.2 |

## large regret: era x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| T8-10 target (influence) | 59 | 0.42% | +7.0 ± 0.5 | +3.6 ± 0.2 |
| T4-7 target (influence) | 40 | 0.64% | +6.0 ± 1.3 | +1.1 ± 0.5 |
| T4-7 AR card choice | 116 | 1.33% | +5.4 ± 1.3 | +4.0 ± 0.6 |
| T1-3 target (influence) | 27 | 0.51% | +4.8 ± 1.4 | +3.2 ± 0.5 |
| T4-7 play mode | 120 | 1.07% | +4.0 ± 1.5 | +3.3 ± 0.3 |
| T8-10 AR card choice | 139 | 0.63% | +3.4 ± 0.5 | +0.4 ± 0.2 |
| T8-10 target (realign) | 1 | 0.02% | +2.9 ± 0.1 | +0.0 ± 0.0 |
| T1-3 play mode | 67 | 0.34% | +1.8 ± 0.8 | +0.7 ± 0.1 |
| T8-10 headline | 71 | 0.10% | +1.1 ± 0.1 | +0.8 ± 0.1 |
| T8-10 play mode | 170 | 0.64% | +0.9 ± 0.6 | +1.6 ± 0.2 |
| T1-3 op mode | 32 | 0.06% | +0.7 ± 0.2 | +0.3 ± 0.1 |
| T4-7 headline | 92 | 0.16% | +0.2 ± 0.2 | +0.4 ± 0.1 |

## large regret: DEFCON x kind

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| D2 other | 223 | 1.48% | +22.0 ± 1.7 | +7.0 ± 0.6 |
| D2 AR card choice | 268 | 2.00% | +5.8 ± 1.4 | +4.5 ± 0.7 |
| D2 play mode | 274 | 1.74% | +3.2 ± 1.7 | +4.1 ± 0.3 |
| D3 play mode | 41 | 0.15% | +2.7 ± 0.5 | +0.9 ± 0.1 |
| D4+ AR card choice | 40 | 0.37% | +2.0 ± 0.7 | +1.5 ± 0.3 |
| D3 AR card choice | 40 | 0.33% | +1.1 ± 0.6 | +0.3 ± 0.3 |
| D3 other | 280 | 0.73% | +0.9 ± 1.1 | +3.6 ± 0.5 |
| D4+ play mode | 42 | 0.16% | +0.8 ± 0.5 | +0.6 ± 0.1 |
| D4+ other | 102 | 0.31% | +0.4 ± 0.5 | +0.5 ± 0.3 |

## large regret, AR card choice: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 2 -> opp 4 | 8 | 0.10% | +1.8 ± 0.3 | -0.0 ± 0.1 |
| opp 2 -> opp 3 | 19 | 0.13% | +1.0 ± 0.3 | +0.5 ± 0.1 |
| own 3 -> own 3 | 5 | 0.03% | +0.7 ± 0.1 | -0.1 ± 0.1 |
| scoring 0 -> China 4 | 5 | 0.03% | +0.7 ± 0.1 | -0.8 ± 0.1 |
| scoring 0 -> own 3 | 8 | 0.07% | +0.6 ± 0.4 | -0.2 ± 0.1 |
| neutral 2 -> opp 2 | 5 | 0.04% | +0.6 ± 0.2 | +0.2 ± 0.0 |
| opp 3 -> China 4 | 5 | 0.02% | +0.5 ± 0.1 | +0.5 ± 0.1 |
| own 2 -> China 4 | 5 | 0.04% | +0.3 ± 0.2 | +0.0 ± 0.1 |
| opp 2 -> opp 2 | 12 | 0.10% | +0.1 ± 0.4 | -0.1 ± 0.1 |
| own 2 -> opp 3 | 11 | 0.08% | +0.1 ± 0.2 | +0.3 ± 0.2 |
| neutral 2 -> own 3 | 5 | 0.02% | +0.1 ± 0.1 | +0.2 ± 0.1 |
| neutral 1 -> own 3 | 6 | 0.04% | +0.1 ± 0.1 | +0.1 ± 0.1 |

## large regret, AR card choice: the card raw plays instead (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| The China Card | 11 | 0.07% | +2.5 ± 0.2 | +1.3 ± 0.2 |
| Central America Scoring | 11 | 0.12% | +1.4 ± 0.5 | +1.2 ± 0.2 |
| Liberation Theology | 5 | 0.07% | +0.7 ± 0.3 | +0.4 ± 0.1 |
| UN Intervention | 6 | 0.04% | +0.5 ± 0.2 | +0.5 ± 0.2 |
| OPEC | 4 | 0.02% | +0.5 ± 0.1 | +0.1 ± 0.1 |
| Truman Doctrine | 5 | 0.03% | +0.4 ± 0.1 | +0.5 ± 0.1 |
| Latin American Death Squads | 4 | 0.04% | +0.3 ± 0.2 | +0.1 ± 0.1 |
| Tear Down this Wall | 4 | 0.02% | +0.3 ± 0.1 | +0.3 ± 0.1 |
| Star Wars | 4 | 0.03% | +0.3 ± 0.2 | -0.1 ± 0.0 |
| Summit | 4 | 0.01% | +0.3 ± 0.1 | +0.3 ± 0.1 |
| Independent Reds | 5 | 0.04% | +0.3 ± 0.2 | +0.2 ± 0.1 |
| Olympic Games | 4 | 0.03% | +0.3 ± 0.1 | +0.0 ± 0.1 |

## large regret, AR card choice: the card the best move plays (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Fidel | 8 | 0.05% | +1.6 ± 0.2 | +1.5 ± 0.2 |
| Muslim Revolution | 8 | 0.06% | +1.4 ± 0.2 | +0.0 ± 0.1 |
| “Ask Not What Your Country Can Do For You…” | 9 | 0.05% | +1.1 ± 0.2 | +0.7 ± 0.2 |
| Cultural Revolution | 4 | 0.09% | +1.0 ± 0.5 | -0.1 ± 0.0 |
| Marshall Plan | 9 | 0.07% | +1.0 ± 0.2 | +0.0 ± 0.1 |
| Warsaw Pact Formed | 4 | 0.03% | +0.9 ± 0.2 | +0.1 ± 0.1 |
| The China Card | 32 | 0.33% | +0.7 ± 0.6 | +1.1 ± 0.3 |
| Indo-Pakistani War | 7 | 0.05% | +0.6 ± 0.2 | -0.3 ± 0.1 |
| Colonial Rear Guards | 9 | 0.14% | +0.5 ± 0.4 | +0.5 ± 0.1 |
| US/Japan Mutual Defense Pact | 5 | 0.05% | +0.5 ± 0.3 | -0.1 ± 0.1 |
| UN Intervention | 4 | 0.03% | +0.4 ± 0.1 | +0.4 ± 0.1 |
| Socialist Governments | 8 | 0.04% | +0.4 ± 0.1 | +0.3 ± 0.1 |

## large regret, play mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| neutral card: INFL -> EVENT | 6 | 0.07% | +2.3 ± 0.5 | +0.4 ± 0.0 |
| opp card: EVENT -> COUP | 6 | 0.11% | +1.6 ± 0.6 | +0.0 ± 0.0 |
| own card: INFL -> REALIGN | 27 | 0.31% | +1.5 ± 0.9 | -0.2 ± 0.1 |
| opp card: SPACE -> EVENT | 22 | 0.11% | +1.2 ± 0.5 | +0.5 ± 0.1 |
| own card: INFL -> EVENT | 13 | 0.03% | +0.8 ± 0.1 | +0.8 ± 0.1 |
| own card: COUP -> INFL | 14 | 0.06% | +0.7 ± 0.2 | +0.2 ± 0.1 |
| own card: EVENT -> INFL | 20 | 0.05% | +0.6 ± 0.1 | +0.6 ± 0.1 |
| neutral card: EVENT -> INFL | 8 | 0.02% | +0.4 ± 0.1 | +0.4 ± 0.1 |
| opp card: EVENT -> SPACE | 9 | 0.05% | +0.4 ± 0.2 | +0.1 ± 0.0 |
| China card: INFL -> REALIGN | 5 | 0.01% | +0.2 ± 0.0 | +0.2 ± 0.0 |

## large regret, play mode: the card (n >= 4)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| “One Small Step…” | 6 | 0.08% | +1.9 ± 0.5 | -0.1 ± 0.0 |
| NATO | 5 | 0.07% | +1.8 ± 0.5 | +0.0 ± 0.0 |
| CIA Created | 4 | 0.05% | +1.3 ± 0.5 | +0.0 ± 0.0 |
| South African Unrest | 5 | 0.06% | +0.6 ± 0.5 | +0.0 ± 0.0 |
| De-Stalinization | 7 | 0.02% | +0.4 ± 0.1 | +0.2 ± 0.1 |
| Suez Crisis | 9 | 0.03% | +0.3 ± 0.1 | +0.6 ± 0.1 |
| Five Year Plan | 6 | 0.02% | +0.3 ± 0.1 | +0.2 ± 0.1 |
| Latin American Death Squads | 7 | 0.04% | +0.3 ± 0.2 | +0.3 ± 0.1 |
| OPEC | 5 | 0.02% | +0.3 ± 0.1 | +0.2 ± 0.0 |
| U-2 Incident | 4 | 0.08% | +0.2 ± 0.5 | +0.2 ± 0.0 |

## large regret, op mode: raw -> best

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| COUP -> INFL | 32 | 0.04% | +0.5 ± 0.1 | +0.5 ± 0.1 |
| REALIGN -> INFL | 4 | 0.00% | +0.4 ± 0.0 | +0.4 ± 0.0 |
| INFL -> COUP | 17 | 0.03% | +0.2 ± 0.1 | +0.1 ± 0.0 |
| REALIGN -> COUP | 2 | 0.00% | +0.0 ± 0.0 | +0.0 ± 0.0 |
| INFL -> REALIGN | 70 | 0.20% | -0.1 ± 0.3 | -0.0 ± 0.1 |
| COUP -> REALIGN | 66 | 0.12% | -0.1 ± 0.2 | +0.2 ± 0.1 |

## large regret, influence targets: raw's region -> the best region

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| Africa -> Africa | 6 | 0.19% | +4.2 ± 1.0 | +0.2 ± 0.2 |
| Europe -> Europe | 15 | 0.24% | +2.7 ± 0.5 | +2.2 ± 0.4 |
| Asia -> Asia | 9 | 0.06% | +1.8 ± 0.2 | +0.7 ± 0.1 |
| Europe -> Asia | 9 | 0.08% | +1.4 ± 0.3 | +1.1 ± 0.3 |
| Africa -> Asia | 9 | 0.05% | +1.0 ± 0.1 | +0.3 ± 0.1 |
| Europe -> Central America | 5 | 0.12% | +0.9 ± 0.7 | +0.1 ± 0.1 |
| Africa -> Europe | 8 | 0.03% | +0.6 ± 0.1 | +0.3 ± 0.1 |
| Central America -> South America | 4 | 0.02% | +0.4 ± 0.1 | +0.3 ± 0.1 |
| Europe -> Middle East | 4 | 0.06% | +0.2 ± 0.3 | -0.3 ± 0.2 |
| South America -> Central America | 4 | 0.05% | +0.2 ± 0.3 | +0.5 ± 0.3 |

## large regret, headline: raw's card -> the best card (kind, Ops)

| group | n | share of decisions | best over raw | k=8 @256 over raw |
|:---|---:|---:|---:|---:|
| own 2 -> own 2 | 15 | 0.03% | +0.5 ± 0.1 | +0.0 ± 0.0 |
| scoring 0 -> own 2 | 6 | 0.01% | +0.2 ± 0.0 | +0.2 ± 0.0 |
| neutral 2 -> own 2 | 9 | 0.01% | +0.2 ± 0.0 | +0.2 ± 0.0 |
| own 3 -> own 2 | 19 | 0.03% | +0.1 ± 0.1 | +0.3 ± 0.0 |
| own 2 -> neutral 2 | 8 | 0.01% | +0.1 ± 0.0 | +0.1 ± 0.0 |
| neutral 2 -> neutral 2 | 6 | 0.01% | +0.1 ± 0.0 | +0.0 ± 0.0 |
| own 1 -> own 3 | 8 | 0.01% | +0.0 ± 0.0 | -0.1 ± 0.0 |
| own 3 -> own 4 | 10 | 0.01% | +0.0 ± 0.0 | +0.1 ± 0.0 |

## The 20 positions that cost raw the most a game (large regret)

| turn/AR | side | DEFCON | kind | raw | best | gain a decision | a game |
|:---|:---|---:|:---|:---|:---|---:|---:|
| T7 AR4 | USSR | 2 | target (influence) | PointNode #47 (Algeria) | PointNode #57 (Zaire) | +9.7 ± 1.5 | +3.62 |
| T2 AR2 | USSR | 2 | target (influence) | PointNode #24 (Iraq) | PointNode #36 (Thailand) | +7.3 ± 2.0 | +3.20 |
| T10 AR7 | USSR | 2 | target (realign) | PointNode #82 (Argentina) | PointNode #63 (South Africa) | +30.4 ± 0.8 | +2.91 |
| T5 AR4 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: EVENT | +7.3 ± 1.8 | +1.85 |
| T7 AR2 | US | 2 | play mode | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN | +7.2 ± 2.0 | +1.82 |
| T8 AR5 | US | 2 | AR card choice | SelectCard #79 (Africa Scoring) | SelectCard #6 (The China Card) | +28.2 ± 1.5 | +1.45 |
| T8 AR5 | USSR | 2 | AR card choice | SelectCard #82 (Iranian Hostage Crisis) | SelectCard #56 (Muslim Revolution) | +16.7 ± 1.9 | +1.31 |
| T3 AR1 | USSR | 3 | play mode | Resolution: EVENT | Resolution: OPS_COUP / OpMode: COUP | +5.8 ± 2.1 | +1.24 |
| T7 AR3 | USSR | 3 | AR card choice | SelectCard #6 (The China Card) | SelectCard #77 (“Ask Not What Your Country Can Do For You…”) | +24.1 ± 2.1 | +1.21 |
| T4 AR2 | USSR | 2 | AR card choice | SelectCard #37 (Central America Scoring) | SelectCard #58 (Cultural Revolution) | +6.2 ± 2.1 | +1.18 |
| T7 AR3 | USSR | 2 | AR card choice | SelectCard #9 (Vietnam Revolts) | SelectCard #23 (Marshall Plan) | +11.2 ± 1.4 | +1.18 |
| T2 AR3 | US | 3 | target (influence) | PointNode #8 (France) | PointNode #0 (Canada) | +5.1 ± 1.8 | +0.97 |
| T3 AR3 | US | 2 | target (influence) | PointNode #33 (India) | 148 | +13.7 ± 2.1 | +0.93 |
| T3 AR3 | US | 2 | target (influence) | PointNode #1 (United Kingdom) | PointNode #40 (Philippines) | +13.2 ± 2.1 | +0.90 |
| T3 AR5 | US | 2 | target (influence) | PointNode #30 (Libya) | 190 | +4.7 ± 2.0 | +0.90 |
| T2 AR3 | USSR | 2 | AR card choice | SelectCard #11 (Korean War) | SelectCard #27 (US/Japan Mutual Defense Pact) | +5.0 ± 1.7 | +0.79 |
| T1 AR1 | USSR | 5 | AR card choice | SelectCard #28 (Suez Crisis) | SelectCard #16 (Warsaw Pact Formed) | +12.6 ± 2.0 | +0.75 |
| T5 AR7 | US | 2 | target (influence) | PointNode #10 (Italy) | 186 | +2.0 ± 1.8 | +0.74 |
| T10 AR5 | US | 2 | target (influence) | PointNode #31 (Afghanistan) | PointNode #51 (Sudan) | +50.0 ± 0.0 | +0.73 |
| T10 AR2 | US | 4 | AR card choice | SelectCard #89 (Soviets Shoot Down KAL-007) | SelectCard #50 (“We Will Bury You”) | +13.6 ± 1.0 | +0.70 |
