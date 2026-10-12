# Control breaks: humans against the model

Humans: {'complete': 119, 'partial': 146, 'skipped': 9} games of the ts-replayer corpus. R34 own games: 4096 greedy self-play games of `R34soup_7120-7200M.pt`

Shares are of that side's Ops influence points (action rounds, not events), ± a standard error clustered by game. *Same positions* asks the model at each human point's position.

### Points placed into a country the opponent controls (2 Ops each)

| side | turns | humans | R34 own games | same positions: human | same positions: model's pick | model − human |
|:---|:---|---:|---:|---:|---:|---:|
| US | T1-3 | 2.6 ± 0.2 | 4.2 ± 0.1 | 2.6 ± 0.2 | 3.3 ± 0.3 | 0.7 ± 0.3 |
| US | T4-7 | 4.9 ± 0.3 | 8.9 ± 0.1 | 4.9 ± 0.3 | 10.6 ± 0.6 | 5.6 ± 0.6 |
| US | T8-10 | 7.2 ± 0.6 | 11.2 ± 0.2 | 7.2 ± 0.6 | 15.9 ± 1.2 | 8.8 ± 1.2 |
| US | all | 4.2 ± 0.2 | 7.2 ± 0.1 | 4.2 ± 0.2 | 8.0 ± 0.4 | 3.8 ± 0.4 |
| USSR | T1-3 | 3.7 ± 0.3 | 7.8 ± 0.1 | 3.7 ± 0.3 | 7.8 ± 0.6 | 4.1 ± 0.6 |
| USSR | T4-7 | 5.8 ± 0.4 | 11.7 ± 0.1 | 5.8 ± 0.4 | 15.3 ± 0.8 | 9.4 ± 0.8 |
| USSR | T8-10 | 6.9 ± 0.6 | 14.3 ± 0.2 | 6.9 ± 0.6 | 19.0 ± 1.4 | 12.1 ± 1.5 |
| USSR | all | 5.1 ± 0.2 | 10.8 ± 0.1 | 5.1 ± 0.2 | 12.8 ± 0.5 | 7.6 ± 0.5 |

### ... of which the point that breaks control

| side | turns | humans | R34 own games | same positions: human | same positions: model's pick | model − human |
|:---|:---|---:|---:|---:|---:|---:|
| US | T1-3 | 2.4 ± 0.2 | 4.0 ± 0.1 | 2.4 ± 0.2 | 2.7 ± 0.3 | 0.3 ± 0.3 |
| US | T4-7 | 4.6 ± 0.3 | 8.5 ± 0.1 | 4.6 ± 0.3 | 8.3 ± 0.6 | 3.7 ± 0.6 |
| US | T8-10 | 6.6 ± 0.6 | 10.3 ± 0.2 | 6.6 ± 0.6 | 12.2 ± 1.0 | 5.6 ± 1.0 |
| US | all | 3.9 ± 0.2 | 6.8 ± 0.1 | 3.9 ± 0.2 | 6.3 ± 0.3 | 2.4 ± 0.3 |
| USSR | T1-3 | 3.5 ± 0.3 | 7.5 ± 0.1 | 3.5 ± 0.3 | 6.9 ± 0.6 | 3.4 ± 0.6 |
| USSR | T4-7 | 5.7 ± 0.4 | 11.3 ± 0.1 | 5.7 ± 0.4 | 13.6 ± 0.7 | 7.9 ± 0.8 |
| USSR | T8-10 | 6.8 ± 0.6 | 13.6 ± 0.2 | 6.8 ± 0.6 | 14.5 ± 1.2 | 7.7 ± 1.3 |
| USSR | all | 5.0 ± 0.2 | 10.3 ± 0.1 | 5.0 ± 0.2 | 11.0 ± 0.5 | 6.0 ± 0.5 |

### US points into Cuba

| side | turns | humans | R34 own games | same positions: human | same positions: model's pick | model − human |
|:---|:---|---:|---:|---:|---:|---:|
| US | T1-3 | 0.2 ± 0.1 | 0.3 ± 0.0 | 0.2 ± 0.1 | 0.0 ± 0.0 | -0.2 ± 0.1 |
| US | T4-7 | 3.5 ± 0.4 | 6.0 ± 0.1 | 3.5 ± 0.4 | 5.0 ± 0.5 | 1.5 ± 0.5 |
| US | T8-10 | 3.3 ± 0.6 | 3.9 ± 0.2 | 3.3 ± 0.6 | 3.8 ± 0.8 | 0.5 ± 1.0 |
| US | all | 1.9 ± 0.2 | 3.1 ± 0.0 | 1.9 ± 0.2 | 2.4 ± 0.2 | 0.5 ± 0.3 |

### US points into Cuba while Fidel is still to come (not yet removed)

* humans: 299 of 15,659 US points into Cuba (1.9%); 85 of them with Fidel to come (0.54% of US points)
* R34 own games: 8,319 of 269,768 US points into Cuba (3.1%); 4,110 of them with Fidel to come (1.52% of US points)

### US points into Cuba, turns 4-10, by where Fidel is

| Fidel | humans | R34 own games | same positions: human | model's pick | model − human |
|:---|---:|---:|---:|---:|---:|
| unseen | 2.1 ± 0.7 (915) | 6.0 ± 0.2 (26,254) | 2.1 ± 0.7 | 9.0 ± 1.8 | 6.9 ± 1.8 |
| discard | 4.6 ± 1.0 (1,332) | 7.5 ± 0.2 (23,671) | 4.6 ± 1.0 | 8.4 ± 1.2 | 3.8 ± 1.5 |
| US hand | 1.8 ± 1.4 (277) | 6.5 ± 0.5 (5,736) | 1.8 ± 1.4 | 9.0 ± 3.4 | 7.2 ± 3.4 |
| gone | 3.5 ± 0.4 (5,688) | 4.5 ± 0.1 (93,009) | 3.5 ± 0.4 | 2.8 ± 0.4 | -0.7 ± 0.4 |

### US Cuba plays while Fidel is to come: does Fidel fire, and wipe the US out

* humans (complete games): 22 Cuba plays (discard 14, unseen 6, US hand 2); Fidel fired later in 36%, with US influence still in Cuba in 36% (mean 3.6 influence removed)
* R34 own games (complete games): 2,602 Cuba plays (discard 1204, unseen 1150, US hand 248); Fidel fired later in 34%, with US influence still in Cuba in 34% (mean 3.6 influence removed)

### Break plays: what happens to the countries entered

| | humans | R34 own games |
|:---|---:|---:|
| **US** countries entered under the opponent's control | 640 | 19195 |
| US: flipped (mover controls after the round) | 27.3 ± 1.9 | 23.6 ± 0.3 |
| US: broken (nobody controls) | 68.0 ± 2.1 | 71.8 ± 0.4 |
| US: held (still the opponent's) | 4.7 ± 1.0 | 4.6 ± 0.2 |
| US: of flipped or broken: retaken by the mover's next round | 34.5 ± 2.0 | 43.8 ± 0.4 |
| US: of flipped or broken: the opponent's at the turn's end | 25.5 ± 2.1 | 33.2 ± 0.5 |
| US: Ops spent per country entered | 3.04 | 3.00 |
| **USSR** countries entered under the opponent's control | 732 | 25619 |
| USSR: flipped (mover controls after the round) | 31.0 ± 2.0 | 20.5 ± 0.3 |
| USSR: broken (nobody controls) | 66.7 ± 2.0 | 76.4 ± 0.3 |
| USSR: held (still the opponent's) | 2.3 ± 0.6 | 3.0 ± 0.1 |
| USSR: of flipped or broken: retaken by the mover's next round | 35.7 ± 1.9 | 46.0 ± 0.3 |
| USSR: of flipped or broken: the opponent's at the turn's end | 35.9 ± 2.2 | 44.4 ± 0.4 |
| USSR: Ops spent per country entered | 3.14 | 2.98 |

### By country

| side | country | humans: entered per 1,000 points / retaken | R34 own games: entered per 1,000 points / retaken |
|:---|:---|---:|---:|
| US | Italy | 1.4 / 38% of 16 | 10.4 / 66% of 2561 |
| US | Thailand | 3.4 / 42% of 45 | 8.5 / 36% of 2209 |
| US | Zaire | 1.8 / 15% of 26 | 5.3 / 33% of 1266 |
| US | Angola | 2.6 / 15% of 39 | 4.8 / 34% of 1131 |
| US | Pakistan | 1.9 / 30% of 23 | 3.8 / 35% of 944 |
| US | Lebanon | 0.2 / 0% of 3 | 3.7 / 29% of 952 |
| US | Brazil | 2.2 / 52% of 29 | 3.4 / 56% of 879 |
| US | Nigeria | 1.0 / 25% of 16 | 3.2 / 31% of 770 |
| US | North Korea | 3.0 / 26% of 43 | 2.4 / 30% of 616 |
| US | Algeria | 0.9 / 83% of 12 | 2.7 / 48% of 711 |
| US | Venezuela | 2.7 / 53% of 38 | 2.1 / 56% of 543 |
| US | Poland | 1.0 / 31% of 13 | 2.5 / 58% of 576 |
| USSR | Italy | 0.8 / 55% of 11 | 11.9 / 69% of 2695 |
| USSR | Pakistan | 3.0 / 34% of 38 | 10.8 / 39% of 2526 |
| USSR | Thailand | 3.7 / 16% of 49 | 9.5 / 35% of 2206 |
| USSR | Mexico | 1.1 / 87% of 15 | 8.6 / 59% of 1988 |
| USSR | Algeria | 1.9 / 37% of 27 | 7.6 / 46% of 1801 |
| USSR | Iran | 2.4 / 47% of 32 | 7.1 / 48% of 1673 |
| USSR | Zaire | 3.0 / 14% of 42 | 6.3 / 33% of 1375 |
| USSR | Angola | 1.8 / 23% of 26 | 5.0 / 29% of 1059 |
| USSR | Laos/Cambodia | 4.7 / 14% of 65 | 1.7 / 19% of 408 |
| USSR | Lebanon | 1.0 / 46% of 13 | 4.6 / 26% of 1072 |
| USSR | Panama | 1.6 / 55% of 20 | 4.1 / 53% of 925 |
| USSR | Brazil | 1.8 / 52% of 25 | 3.8 / 61% of 906 |

## What a break costs

Paired playouts of 4,368 break plays from 400 games, 512 pairs each, continuation `raw`. Gain = keeping the card's Ops out of opponent control minus the model's own play, in points of win probability: per play, and summed per game (both sides). The ± is over plays (positions), which includes the playout noise.

| group | plays | per game | mean gain a play | per game, points | share of plays where no-break is better at 2 SE |
|:---|---:|---:|---:|---:|---:|
| all | 4368 | 10.92 | -1.04 ± 0.10 | -11.41 ± 1.11 | 7% |
| US | 1821 | 4.55 | -0.82 ± 0.19 | -3.74 ± 0.88 | 10% |
| USSR | 2547 | 6.37 | -1.20 ± 0.11 | -7.67 ± 0.67 | 5% |
| T1-3 | 1158 | 2.90 | -0.70 ± 0.14 | -2.01 ± 0.41 | 8% |
| T4-7 | 2052 | 5.13 | -0.84 ± 0.09 | -4.31 ± 0.45 | 5% |
| T8-10 | 1158 | 2.90 | -1.76 ± 0.32 | -5.09 ± 0.92 | 8% |
| after the round: retaken | 1811 | 4.53 | -0.11 ± 0.14 | -0.48 ± 0.62 | 9% |
| after the round: lasted | 2306 | 5.76 | -1.66 ± 0.11 | -9.58 ± 0.64 | 5% |
| after the round: held | 147 | 0.37 | +0.61 ± 0.64 | +0.22 ± 0.23 | 14% |
| after the round: unknown | 104 | 0.26 | -6.05 ± 2.29 | -1.57 ± 0.59 | 3% |
| held the scoring card of the first country's region | 1000 | 2.50 | -1.49 ± 0.17 | -3.71 ± 0.43 | 7% |
| did not | 3368 | 8.42 | -0.91 ± 0.12 | -7.69 ± 1.02 | 7% |

By the first country entered (10+ plays):

| side | country | plays | mean gain a play | per game, points |
|:---|:---|---:|---:|---:|
| US | Poland | 41 | +3.23 ± 2.10 | +0.33 |
| US | Lebanon | 107 | +0.71 ± 0.49 | +0.19 |
| US | Algeria | 64 | +1.17 ± 0.44 | +0.19 |
| US | Cuba | 19 | +1.78 ± 1.10 | +0.08 |
| US | France | 62 | +0.53 ± 0.72 | +0.08 |
| US | Laos/Cambodia | 29 | +0.76 ± 0.60 | +0.06 |
| US | Brazil | 101 | +0.20 ± 0.35 | +0.05 |
| US | West Germany | 11 | +1.26 ± 0.53 | +0.03 |
| USSR | Algeria | 205 | +0.03 ± 0.20 | +0.02 |
| US | Egypt | 12 | -0.03 ± 1.17 | -0.00 |
| USSR | Cuba | 20 | -0.27 ± 0.85 | -0.01 |
| US | Chile | 10 | -0.98 ± 0.75 | -0.02 |
| US | Argentina | 40 | -0.34 ± 0.73 | -0.03 |
| USSR | Indonesia | 38 | -0.42 ± 0.43 | -0.04 |
| USSR | France | 86 | -0.19 ± 0.31 | -0.04 |
| USSR | Laos/Cambodia | 44 | -0.41 ± 0.42 | -0.05 |
| USSR | Egypt | 16 | -1.21 ± 0.60 | -0.05 |
| USSR | South Africa | 14 | -1.58 ± 1.45 | -0.06 |
| US | Indonesia | 44 | -0.52 ± 0.51 | -0.06 |
| USSR | Venezuela | 42 | -0.66 ± 0.50 | -0.07 |
| US | Iran | 23 | -1.25 ± 1.22 | -0.07 |
| USSR | India | 12 | -2.54 ± 1.26 | -0.08 |
| US | Libya | 41 | -0.83 ± 0.59 | -0.08 |
| US | Venezuela | 42 | -0.97 ± 0.63 | -0.10 |
| USSR | Mexico | 192 | -0.25 ± 0.28 | -0.12 |
| USSR | Panama | 110 | -0.43 ± 0.35 | -0.12 |
| USSR | South Korea | 31 | -1.57 ± 0.62 | -0.12 |
| USSR | Lebanon | 111 | -0.44 ± 0.33 | -0.12 |
| US | Nigeria | 77 | -0.67 ± 0.52 | -0.13 |
| USSR | Libya | 41 | -1.32 ± 0.69 | -0.14 |
| USSR | East Germany | 10 | -5.44 ± 0.80 | -0.14 |
| USSR | Argentina | 72 | -0.87 ± 0.36 | -0.16 |
| USSR | West Germany | 74 | -0.86 ± 0.35 | -0.16 |
| US | North Korea | 42 | -1.55 ± 1.35 | -0.16 |
| US | South Korea | 35 | -1.92 ± 0.70 | -0.17 |
| US | Italy | 300 | -0.26 ± 0.53 | -0.19 |
| US | Panama | 66 | -1.45 ± 1.58 | -0.24 |
| US | Angola | 113 | -0.92 ± 0.84 | -0.26 |
| US | Mexico | 56 | -1.94 ± 1.91 | -0.27 |
| USSR | Brazil | 115 | -1.16 ± 0.34 | -0.33 |
| USSR | Zaire | 152 | -0.98 ± 0.56 | -0.37 |
| USSR | Nigeria | 97 | -1.67 ± 1.03 | -0.40 |
| US | East Germany | 31 | -5.55 ± 3.34 | -0.43 |
| USSR | Iran | 178 | -1.06 ± 0.29 | -0.47 |
| US | Zaire | 131 | -1.44 ± 0.91 | -0.47 |
| US | Pakistan | 89 | -2.96 ± 1.03 | -0.66 |
| USSR | Italy | 293 | -0.96 ± 0.23 | -0.71 |
| USSR | Angola | 113 | -2.72 ± 1.12 | -0.77 |
| US | Thailand | 219 | -2.53 ± 0.39 | -1.38 |
| USSR | Pakistan | 236 | -2.39 ± 0.40 | -1.41 |
| USSR | Thailand | 211 | -3.31 ± 0.36 | -1.74 |

The plays where keeping out of opponent control gains most:

| side | turn/AR | countries (end, retaken) | gain ± SE | id |
|:---|:---|:---|---:|:---|
| US | T10 AR6 | Poland (broken, True), Italy (broken, True) | +82.9 ± 1.6 | [71fdacfb66cc3641](https://mihaild.github.io/DeepStruggle/?pos=eJztV1uP4iAU_i88d5NCb1P_ysYQbFHJIjRAnTGT-e97uFjp6MNkNtmHXUM0nI9z_06VviM2OKEVNXpWI9q0BRqYGanUA_O4RZufZYFhtWG9wKdbVpt9d6AT9212TpJdxNtMK1rg7DSe-R3O7K5YvbKrM7ubDxwsuyXa1QNeoq-zAcstFHwUitFQ9lHLkRu0wSt0kuzCdpJH3L1R69jwCzrzjpiU-pVyZuQFUD2hTVmgkQ_C-qZ6w-RuwdxlAkdNgfiZK0cPhinHR6onG2wjCgEOPMgn9kYnbugA_DhzCZjSI48A3QkXKYIFpeiJnuAwaIFHn4BQaFMXaOJqFOoAcUJNgekbZumZyTnmdUWNljI4MvwEXjxkHfdpNh6zWp49Fr15NdCnME3a3ETHzIG7IENJxlGh9hBHDZz6Inzqfd9UuCE9brseNxXpK1xUJUAV1GPnaYJQ9po2De0J_pw4-fA76N9wRBvSQFZnaLFv5tW3b8pH8X2Wyr_KUvmApfIhS-VXWCrvWcJl9W2eYupfJ6T8UzrKf56O7z80TzKeZDzJeJLxn5Cxza5ddOSTi7rDbIzv6HRkFuojnon9oFW2paPR0wT-nKYkGO21GUAOiYjxDspoBfjEIE_kLLXszOmZoAIdORulUJzuhbEutYj0dwf6VaVLxXJgOWQ0Lv_F3f1RMvqxsorzUmXQbB_Fna01uXPJII1R8BvbAYF-PECj9I7yvqS5DGK4X8H84Ti5eozdVBxCcOnYMhd42ZFllxpq52GAOUCbPZOWA_vxcfnwc2msgAkEMvl-zwc_-LjGXVO2_QtcBkEDOPajkl3WTJha5vy9nNRtjQnpCVi1XdOUFTTAzUb5XhQIGrZMJwyVf3sol9UGicDCQcY4OwzAJ_m6qqIJhv6lo06adcDvbT1Wr-zi6WffVfCYdtuQ-UnI-BNSBdFODB4xZ8IrSO0hoD2vbh08VnULjetwUAU4xq9SxDK1gaRsc9xr1cWtonr1jVPWZKXpV5ci5T1J0jblvtTXJOBzhWfhf3kudNJCxcn4-A33qb0l) |
| USSR | T10 AR7 | Nigeria (flipped, None) | +65.6 ± 2.1 | [5dff5e989312404b](https://mihaild.github.io/DeepStruggle/?pos=eJztV-2OoyAUfRd-OwkIWu2rbCaEUdqapWAAO9NM5t33Atbi1h-7ye4mmzSkFs79vudq9BOJzg9Gc2sm3aP9rkCdsD1XphMBd2j_DRc1LBKvTbzu5nXfhUVgrdF6Xo-7m2bwWmaSu9V6t4sW9crvOgaDVUe93JaB990GvmCvUPBp0ILHsk9G9dKiPVmhoxJX8aZkwv0Hd15036Ezn0goZd65FFZdATUj2uMC9bIbXGhqMAzuXkgG-usInqoCyYvUnh-t0F723IwuGicUIhxlPJ_FBx-l5R0Q5O01Ytr0MgH8bfCJI1hQixn5GYRRCzyGDAaN9rRAo9T9oI8QJxaF9i3LQccvQk1ypWqNUtGTlWdwEyDnZciTBswZdQlYchfUQJ_DPBl7P3phj9LHM9RkPR_0AeLoTvJQRci9pbu6ZlVTlzVjmBHcNAWjrGQECnLTOEIod8ubx_5Ef344h_Bv0MDuFJELdDj08uY59OSr2GKJbLD0SBL5pyThDZLwJkf4Vziq_iRHKfG_RMbWLYMfyMD_Mxn4ScaTjCcZTzKeZPw2Ga_ZSxfv5eiTbjdZGzo6noSD-srAxKEzOtvy3ppxBH_e8DIaHYzt4BwTGfoHKKMV4LOAPJF33ImL5JcSFegkRa8GLflhsM7PLWraB4F51_MrxSJwEjLqZxO6IZltXlaiNC40gya3FXZyzma-lYAk-kHeuY4IdGMDTadPlHdlnsp4LCt4l4DpI2luTZ96qSWEkMqLZSrIsiuX3dxON3UdTAHaH4RyErhPN8tXmErrBpg_oFIeDrILY9_ihrYMU7wjNAwuUBwmJX-ftnFqhQcv8NpY47ZiDW5bSpuy3kG__GRhFAhYQ8OW4YSZYkVV4HkR2BP4r-LHCy4oLJxJ0-92pfAxAR8UbBY38CsjHBA2_5crg6TZLnjQqzL364C3Vc43PWR-HlR6gtB4dKOAO8zb-P1RBQhoz6tbOyIxpVuIezC2yOmSWDqXdVSCduDbJkp2P1kuacbu5R7ygnJJ8ldGlM3lQfJLgeUMrEok4ZkQnjxXPppBh9mgXz8ALEu84A) |
| US | T10 AR6 | Italy (broken, True) | +45.3 ± 1.8 | [b9ff742a9382d80e](https://mihaild.github.io/DeepStruggle/?pos=eJztV-tu2yAUfhd-exLge15lqhC1SYJGjAXYbVT13XcA2yFNfqzbtEpTZDmBj3P_Dha8Id45qQdm9DT0aFdlqOOmZ0p33OMW7b7jrMoIvP5p4K23Jx3XQeoarZInWrhIXVbr5bfe9KsriXrTqxJPRSKTWox2Lj5pwOk2W71GvH6ChI9y4CykfdSqFwbtvpEreFT8zJ-VQDsMuHtl1vHuB5TmDXGl9AsT3KgzoHoMIr3opPVV9YreHkkwdx7BUJkhMYvBsYPhgxM906MNuhEFB4fo7sRf2SgM64AgZ84BG3QvIsCepYscwQO56JGdYDFIgUUfgBzQjmZoFEMvhwP4CTmhXVumoGUzV5O4EjVaqWDJiBOY8ZB1wsdJPWa1mj0WzXkxkGfQT9pcpo6bg3BhDjkZx-SwBz9DJ5jPwsdOMKZNU1KCm5qWFSnrps7ytihKn5GdxhF82TVwFgoUDDp58v6foYLdMSAzlNgXczXti_KefQVL5DdYwndYwndJwr9CEv6bJMXA_4iMluZFS2gJPLctDGpcfIoefEMP_qeb6D-n53N75UHGg4wHGQ8yvoaMp-QYxnoxuijbTcb4io5HbuNRphf7TkNR8nXIeqPHEew5zWhQ2mvTwTwEIvsbKKEV4BOHOJGzzPJZsJmiDB0F75UcBNtLY91SoiK_WdAvw3q03FasgJD6Raeqb1cWpSud2C6pg8neMzFZa5JwFIcgeikuXAcEqnEHjbM3lFZl6cowpSUcjaD7SOxb3cdaDgJcCOX41hVkG9FttJTTTl0HXYB2e66sAO7jZnn3XWmshP4DKsV-Lzrf9nnd0LKtMWlIAxLAsG-U5OhmQs9y52tTFriCMyXJaZPXua-Kmwy0AQG3UKytMaGfiqyE_USWt_B7KyP-CkHCJSNstixf_nFAMVwe4ut_aVglcDXCH7SKoBlN55sGWeSjTLmNo_ty08SJzyLsG4j8JFX8euRhakcOu8uZcBspPASUp9mlRuBdkm1XzF_t4lIevMYYqivv61Mt8a8VoVv8a4XSdQq26FKfNIj2g1VCF-GYIIS_pVguwMckZ-m_O2c2ajn4zmjefwJXM8Bd) |
| US | T10 AR6 | Angola (broken, False) | +39.1 ± 1.6 | [3e7209946897ebb6](https://mihaild.github.io/DeepStruggle/?pos=eJztV-tumzAUfhd-Uwkbg0leZaosF5wEzcHINmmjqu--40sMNEiLNm2apsqC4M_n_h0T_J7x1vZqYFpNQ5ft6zxrue6YVC13uMn234q8zhFc7t7Ana6uedRxjhMStOq45vRnKZrQ5RpdSczaa391TqIGTbo0-UH-wov1W_w06gVJ5MYzJHzqB8582iclO6GzPVqho-RX_iJFti8At2_MWN5-h8q8Z1xK9coE1_IKqBq9SCfa3riiOsVoLmH2OoKhKs_ERQyWHTUfrOiYGo3XDSg4OAZ3Z_7GRqFZC_xYffXYoDoRAPbS20ARDEhFjewMi14KLLoA-iHbl3k2iqHrhyP48Tlle4KXoGEXLiexEtVKSm9JizOYcZCxwsVZOswoeXFYMOfEQJ5BOyk9Ty3XR2H9HHLSlvXDAfwMrWAuCxf7rmkwxqisEUVFVTaoKfOyIGVJISEzjSO4Mre4ma-Pt2f7s3P_AgVsT77GF6iwq-XNsqvJR_4oSU_3LBV_laVig6Vik6TiEZKKe5Ic57_IUoj8cTqK36WjuGODbrKBHmADPcQGWrGBt_YM3aQDf6YDbdCBfrpn0Cc2avJP0_G_bY5ff4N9kfFFxhcZf5KM58V3F-vEaINsO2ntKjqeuAnv4U4cWjUsHlmn1TiCPasY9koHpVuY-0D67g5a0ArwmUOcmTXM8ItgF5zl2UnwTvaDYIdeGxtLVNO7BfU6xG-_tGAERNTd_kx29ytR52mlFNqlXECT2XI7GaMXtiWHILpezFx7BKqxgYbZe7asSuxKP8UVfLFC96HQt6oLtRwEuBDS8tQVKD3h9BTLaaa2hS7I9gcujQDuw2b5cF2pTQ_9B1SKw0G0ru0xKkhDyQ7X1PUtMOwaZfFFrX3PcutqQytCK0R3qCKoqUvSQP520tAICJShXKk1oaNIXsGXv99V8FvBSQF-w3HAQSQu4SRSpFEm1OFlOELAIHGVJH0Ukfm5jnpwLCIL03UUIumOb2vPPvRzL8MLhPipGTlsMKv9CcRDwPoyvTnaW4LF7LGCE1wIrFwEuVup4IhXK7RclQWtSlT6tMK8mp05oFwZqZNCHdOD4FOCRQQ-p3jp3YvnykbVD641nsjHD8PCvYE) |
| US | T10 AR5 | Italy (held, None) | +37.7 ± 2.1 | [97a30e13ef2310a3](https://mihaild.github.io/DeepStruggle/?pos=eJztV-tu2yAUfhf_9iTAF-y8ylQhapMEjRgLSNqo6rvvcLAdp8mPdZtWaYqQHfNx7t8hgrdMdkHbQTh7HPpsU-VZJ10vjO1kxH22-U7yOqfwxNHAw5ex_uYodY3Wq0HzCldmqcsqn95s0a-vJPiiV688lSuZtcX4SxefNVjl-J5ns9eE8ydIeK8HKTDtvTW9ctnmG72CRyPP8tmobEMAD6_CB9n9gNK8ZdIY-yKUdOYMqB1RpFed9rGqUTHaoyssnEeFhVYnNQSxc3IIqhd29KibUHCwS-4O8lWMyokOCArujNhge5UA8axD4ggG5GJHcYBFlAKLMQA9ZBuWZ6Maej3swA_mBBFcgV6cpDmqK1FnjUFLTh3ATIR8UDFOFjFvzSliyVwUA3kB_WTdZRqk26mAc8jJBaGHLfgZOiViFjF2SghrmopR0nBW1bTiDc9LxhltICN_HEfw5efABRYIDQZ9iP6foYLdHpETlDgWczYdi_KefwVL9DdYIndYIndJIr9CEvmbJKXA_4iMlhVlS1kFPLctfHBSfooeckMP-aeb6D-n53N75UHGg4wHGQ8yvoaMp9UxTPRqDEm2OzoXKzrupU9HmV5tOwtFKeZP0Ts7jmAvWMFQaWtdB3MMRPc30IpWgA8S4syCF16elDixLM_2SvZGD0pstfNhKlFZ3CzYl2E-Wi4rXkFI_aRT89uVSelKJ7XL2sHR3zNx9N6twjESgui1unCNCFTjDppmb9m6KlNX4pRVcDSC7qOpb22fajkocKFMkEtX0OWLLV9TOf2x66ALss1WGq-A-7RZ3mNXOq-h_4BKtd2qLrZ9wRtWtZzQhjYgAQzHRlkd3Rz2rAxgg5OGlkVVlXDYaGlLyhZUwtFBI1BwDOVaWhM6qoSbCYHbQXrKuLtyGq8LFK8ZuN3yYvoliBK4PqQnvhmuUrgckQ9aJWom08WiQSf5JFMt38l9tWimwXGlxJ0DkR-0Sf8fBU79KGF_BYf3kTJCQPo6O5JfAodnSradsXi5S0sFek0x1Dm5M-op_rkibIl_rhC7kqdLfdZY-8FqhTJ0yg-iXzJkE_Axx5OOfzxnMVo9xNZo338CQzTAiQ) |
| US | T10 AR3 | Lebanon (held, None) | +26.9 ± 2.3 | [2bed60bc869212ae](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tuozAQ_ReeWckXMGl-ZVVZLjgJWsdGtkkbVf33nTEOkMtD1ZX2YbeyCJ7jmfHMnIGY90K1sXdWejfartjysmiV76RxrUI8FNufpBQwKAxRbuBq5oFYky4cIs-m36pkMEQaE8bSTMC9mS2arMGSjsj2Is9Evm7nVdZY7BFrcoyLz0t01ZVPkXUxXvEMCR96q2RK--BMp32xpVfoYNRZvRg94fFNhqjaX1CZ90IZ416lVt6cAXVDsSVl0em2D1hUNMzuZiyeB3BUl4U-aRvl3isbdSfdEJLthMIGe53ko3qTg_ayBX6iPyfMuk5PgHzp40QRDEjFDfIIi0kLPGIAvU2sDtp2vd3DPimnYsue1mCQJ2VGfaXqnTHJk9dHcINQiBrj5IgFZ06ITe5QDfQltJPzixiV3-uYZMjJR9nbHexjWy0xC4yd0qauG9ZUvN4wVm84yGVFaM04ZBTGYYC9wiVwmQqUHMb-iPu_QAXbAyRUQ1FPUGQs58U5luWj_CxPP-6JIn-VKPKAKPKQJ_IZnsg9T5RUX2ZqCv3zjJA_5YP883R8_bH5JuObjG8yvsn4T8h4Xp28ZKeHOOm2o_dY0eGgAuTHkIld6-xqKjvvhgH8RSdZMto534KcAum7O2hFK8BHBXEWMcigTlqeWFEWB60601std70Pcf5j5Xcr7tXm49-8EDSE1C1lvV3IJj-ubKZ2Wfsfw8NtxxD84tsoiKHr9cJ1QqAaD9BJei_WVcldmcR0vILuo1Pfum6qpdWwhTZRzV1B5xmbZ7mcYWxb6IJiu1MmaOB-elg-sCt96KH_gEq92-kW255XzRN5EoKIDZCJDGOjrM7UPvWsingwr-tqQ9mGC85JTajYCLCJo7dYorKAes29CS1VpcM_yQM_DEjJM0KTRFbrbJ5d5CrPKE8iyxBNAx0uDuo8ry8mCVs2ozfuL9vilxY-OhD5sTfzCwTEMCh4wKJP3yAcISB9nd1ttGupyjIt-bzG53xYQqbaQGj1SglNq1VG6wKR7Ob6frHiNzroj1J09pyDnxNkGbhKkeIrAV88Zzm43mJrUPrxGzg6vWQ) |
| USSR | T10 AR5 | Poland (broken, True) | +25.1 ± 2.4 | [6761802b8a8ee77e](https://mihaild.github.io/DeepStruggle/?pos=eJztV9uO2yAQ_ReevRJgbKf5lWqFWJskVglYgLMbrfbfOwyOQ5o8VFupD-0K2YEzw9zO2MHvRPVxdFZ6N9uBbJuK9MoP0rheJTyQ7XdatThExapNxasOR4tXvrMVuf6KRX4ZpXaa8dUGW3fyRXbZwVad69XdaCQdgTbYEsdFjxU-yvguOzmi4hkSPoxWSUz74MygPdk-sRt4MuqsXowmWwp4fJMhqv4HlOadKGPcq9TKmzOgbkKVQfdjSFVNGy_2VjCeJ42V1idto9x7ZaMepJsCbs4oeNhnf0f1JiftZQ8MRX9GzLpBZ0C-jDGTBAOScZM8ghC1wGKKYLRkKyoyaTuMdg9-MCmy5V0JBnlSZtY3qt4Zg5a8PoKZBIWoU5wiYcGZU8KyuaQG-hIayvnrMiq_1xHXkJOPcrQ78GN7LVMWKXZe02bTUbb5JmrRNHXX0mrTdjWkE-ZpAkfhErXE6qC1OB6T8xcoX39A5AT1TZW82E0V-aj-gCP6VzmiDziiDymiv0MRvaeIUfFpknLof5EP-s_T8fkn5ouMLzK-yPgi4z8h47k4dclBTzHr9rP3qaLTQQXIjycmdr2zxVQO3k0T2ItOcty0c76HNQYyDndQQSvARwVxkhhkUCctT5xU5KDVYEar5W70IS4lars7gXu1yQ4rBEFDRMP6X8zvRcump5tduV_qAprDI79zCL40bhSEMYz6yjYiUI8HaF69k7IuS1_ikjdwYIX-Y7lz3ZCraTW40CaqtS_YOuPrbClomPse-oBsd8oEDeznx-Uj9aUPI3QgkKl3O92nxq8b0XFOOaUtHAlBB1hOzVKe1zw2ropghnWiEUxsYNem6zrR0QYOO3H2NpWjIlCztUGhrwR-ztCbwfgyqeE7J_3yq6hQEyDnMGrUaPHO8Dska7J1J1v085yv0uShXmYsO64Lh_U64_gAQejH0eTXCMdlmBQ8ZtHjV0idIKC-TK9ICq4GBsy6BWpxeRFyvNdrsnyVprlA_FoY8UvN7keZ-lIQit9mj2qZ04PgbxME4CZFkV4M6fVzlpMbbWqPJ9Z-_ATPQL45) |
| US | T10 AR3 | Angola (flipped, True) | +24.9 ± 2.8 | [d1b1fc0412edf3fe](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReevRIXX5L8yqpC1CaJtQ5YgNNGVf99ZzB2SJOHqivtw26F7MBhOMzMGTv4jag29NZIZyfTkZ0oSKtcJwfbKsQ92f2kRR1bCdcGLgZXE9vyezu6oiytbNIv9jigy3yZrOuE1em-2JZx14ULW7laLCw8463TzHLn0XbekSXLDx49QcDH3igZwz7aodOO7H6wG3gc1EU9D5rsEA-v0gfV_oLUvBE1DPZFauWGC6B2JDtakE63vces4kLkYxkWLiMQVQXRZ22CPDhlgu6kHX1cO6OwwUHH8Um9ylE72YJAwV0iZmynZ0A-92HWCBrEYkd5gsloBYzoQG-irKM2XW8OsE-MCXyiNEe9PKth0je2zg5DpHL6BDwI-aDRUYGYt8MZsZkPzcBeQkFZdx0G5Q46xDEE5YLszR72Ma2WGAY6v-W82nAmWFPXvGnqhjaFoKUQW4jIT-MIW_nFcRkTFPlCf8LtnyGD7ZHseAVJPUOSMZ0LN6blvfi6TvSv6kQf6PRYJfoZlei9SozWX9Zpdv3zgtA_lYP-83J8_aH5FuNbjG8xvsX4T8R4yg5estNjmG3byTnM6HhUHuLjqMS-tSbrys7ZcQS-YCWPi_bWtTCOjvTdHZTJCvBJgZ8keOnVWcszJwU5atUNvdFy3zsfUooEu5uwL2Y5TK4zXoNL3fJnfD-R1tzMzNUiMmjyDxgm713mzKDAha7XV6UjArl4gM6jN5LnJNVkHMazFdQem6vWdnMmjYYt9BDUWhNs7fG1l5Lpp7aFGiC7vRq8BuXnR-Uda9L5HqoPhNT7vW6x6Dkrm7piJedwGiSoL5ZJdlBzsWJVAI5SbCndsk0laN2IktENJCtMzuB5pyCQrbUwoZ5EURV0bRV8BeAdvy6uqFh7PEN5vHicxW8KxEoYl6lXRny2zdnyht8h1WohEotY7zNXHR8bcPzUD_PLg8WhHxU8XMHFz48IgeZ5cB83Y1kI9Rq4WB3Mk8GzXhUdFas9yyzY4mLsL1xsdf7KVmaJnJOdQn5Krq_hVQm4CbDElwG-ci5ytL3BsqjffwMX17wa) |
| US | T10 AR5 | Nigeria (broken, True) | +23.7 ± 2.1 | [ae1d1427a840550a](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReevRL4lia_sooQtUmCloAFOG1U9d93GIhDmjxUXWkfdivUhDnM_YxT_EbEEJQ13NnZjGTTVWQQbuTaDiLinmx-0orB6nE9wd9qWX3xuQKdtO-L8zrbJbwvtJIFK07TWdyxwu6CtTd2bWF38dGiv3rJ4uqBLdHLbOpouYWCD8oIjmUfrB6lIxt2g05anMWzlgkPr9wHMfyCzrwRobV94VI4fQbUTmRDKzLKQfnY1GiY3S1YOE8S-yxP0gS-d8IEOXI7ebRNKATYS5SP4pVP0vEB-AnujJixo0wAf1YhUQQLSrETP8IhaoHHmIAyZFNXZJJmVGYPcbAmyIk-lajnJ6FneaPrrNboyskj-ImQDzImWkfMW32KWPIX1UCfwzxZdxWDcHsZUIaiXODK7CCOGSSPZcTk1-uuYV29Zv1qzbqmXjesaihADVTk52mCUP6SOMcGob-gjjH8M3RwOEBOHTT1BE2O7bz4jm15r77OE_2rPNEHPNGHLNHPsETvWWK0-TJPKfXPE0L_lA76z9Px9Yfmm4xvMr7J-CbjPyFjW1y8-CinkHSH2bnY0ekgfLq7jHI3WFNs-ejsNIG_YHmNRjvrBpAxETXeQQWtAB8F5EmC516cJD_VpCIHKUatjOQ75XzILarXdwf2xeRLxXLgJWQ0Lv-LV_dH2ejHjVWal6aAZv8o7uy9K51rAWmMSl7ZRgT68QBN0hsp-5LnEkW8X8H8sTS5dkzdNBJCSB3EMhds2dXLLjfUz8MAc0A2O6G9BPbT4_Ie59J5BRMIZMrdTg5x8FnLVh3t109wGQQN4DiOSnFZczi1IsTmrLp21bF4g2wb2rdNbEyYnYmtqAj0axlOmKn4-kCX1aNU4wtBlBkrDhH4IF9WU3VoGN862qzZIl7qXL_bG7uk_9F3gx7zbouZH5VOvyANin4S8IQFh-8gbYSA9bK628RTVdfQrMWDBuEUv8kRaW5DnbMt8ajVLk4vDi-fLGdd32jGtcqRyp5kaZtzX-rrMvCxwpOKPzxnPlllcDCa99-g4b1W) |
| US | T10 AR6 | Zaire (broken, True) | +23.6 ± 1.9 | [81bed5e3e04ec439](https://mihaild.github.io/DeepStruggle/?pos=eJztV-uK4yAUfhd_Z0Fzb15lKeIkpg1rNajpTBnm3fd4SWppF4ay7C7LIE08x3P_jqm-I9bbSUmq1SIH1NUZ6pkeqFA9c3yDuu84q7fRZiRr_HDvOs7DqJPfKhO0Vv712dzQYUayMuEH7Wbzl1pONa6xkbiew4x4epXM4299Jn73kPBxkoz6tI9KDFyjjtxwZ8Eu7EXwwLdv1FjW_4DKvCMmhHqlnGlxAa6aUYczNPB-Mq6oTjGa23j2MoOhKkP8zKWlB82k5QNVs_G6gQsODtzTJ_ZGZ65pD_hYffE8qQYeGPRlsgEiGJCKmukJFr0UWHQBTBJ1RYZmLodJHsCPzwl1uyZlGnpmYuE3oloJ4S1pfgIzjmUsd3EWjmeUODteMOfEQJ5COyl9JS3TB249DTlpSyc5gh_Zc-qycLHvyjLHOK_znBRlUTVFW2TAqRvIxyzzDJ7MGjb15fHm7HRy3l-gfv0RdXkFJT1DiV0xV9OuKB_Z8yjhP4oSfoASfggS_gxI-B6ktnoapRD55_HAvx-N5_dM_is0CClxW-G6bf8mLM_vnT-OCv7v98gXGF9gfIHxb4KxT45fdOCzDbL9orWr6Hxkhvuv_cDHXslkSget5hnsWUVzrzQq3QPtA5mGO1YCK7BPDOJE1lDDzpyec5ShI2eDmCSn46SNjSUqi7sF9Srj39m2YDhENEQVUt-vRJ1vN0qhXVIHi3nkdjFGJ7YFgyCGiV-x9hyoxgNuoN5RWpXYlZ70pyzoPhL6Vg2hlpKDCy4s27qCbLN8m8VymqXvoQtQNzJhOGAfNsuH60ptJug_gJKPI-9d2xP4d27caLE7sALCrlGSQ4L2Pcss2KhxBSfIosnbCo6ROwxPML9oaAQCjqFcW2tCR1X-ioDj2AGFswouCPhuOF7hZyTSYRA_r6JEDpcRd1Uq_buIulcP5Y3t0l9SVoki8RSem97eB36aRPh85J40M4PtZbW_hngWYJ4mRxLHLi3noNlcl_FdbI6Ku8SDTHMTTBFtrDo4sVr45MNa6jtNz402yvlS72PoW3pVZNwk6E_27qNzobOapGuL8uMn3FG9_g) |
| US | T3 AR3 | France (broken, True) | +22.1 ± 2.1 | [1c98ffd78be087d8](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_Rc_uxLgS9L8yipCxCaJtQQswGmjqv--w0Adp3G20aparbQR8mWOYW5nsJi3TDS-M5pbM-g2WxV51gjbcmUaEXCXrX6QnMGo8hruS3gu0qCALCaDpifDewUyhfcan2Et-SdG8CbEkOQ1BLzvtOAY9t6oVtps9UQv4F6Jk9goma0I4P6VOy-an5Cat0woZV64FFadADU9Tmll07mQ1bAw6KMTzJ96UFTlmTxK7fnOCu1ly03vcG1EwcAumjuIV95LyxsgyNsTYtq0MgJ80_nIUYzF9PwAH3EWaAwOdBpp7aVuO70DOxgTenDGHD8KNciLmdYohYqsPICWADkvg5tFwJxRx4BFbWEazOdQTsaeRS_sTnqUISTreae3YEc3kocgguvPjBVVRZ8XyyWtirosl4u8LiEWN_Q9mHEfLnNMDery3SGY3kDumn22YhUEc4T0hkR-6A0Jec_vZYhcMbT4Y4boXQzRC4boDEOsnqWIfqaIzlBEv6SIfqKoXNzmKLp-PyHk--kgf3XDkBk6yCwb5J4NQ75zwzzIeJDxIONBxn9Cxnpy5OKt7H2c2wzWhoz2e-EgPhaY2DZGT155a03fgz5vOMNFW2MbkNGRrr2CJrQCfBDgZ-Ydd-Io-ZFlebaXolWdlnzbWedTigp69cG86HTsGz84CR61aQktrr-kNU8Xi2K5TGcPbs7s4Jyd6FYCnGg7eeYaEcjGDBqlt2yalVSVKOLpCqqPxro1bcyllmBCKi_GqqDjGxvfUjrd0DRQBdlqK5STwH3cLO-hKq3roP6ASrndyiaUfQHHQfZcM8KKBYFjSeA4lMrkOG2xaoUP56GKFkW1LAtGS1IyWpMSDAw2nnshX2NtQkmV2JB8NAMVvpfYFJxbhBLvNTYL801Ema7Q4JRjY0Hy4svmYyr9piVao-OHTsX_B0XR9QL2l7fYeyAEpE-D-2yonJisUWLpqpK7bHSFTdwpLlwrko5bybg1biZjnVwfw2MJuAgQ_wfhr3Pivel0qIun-v0XQgG51Q) |
| USSR | T10 AR3 | Panama (broken, False) | +19.3 ± 1.7 | [c2f74f75fdc9e8ad](https://mihaild.github.io/DeepStruggle/?pos=eJztV8Fu4yAQ_RefvRIYxyH5lVWFqE0SawlYQNJGVf99hwE7pMmh6kp72K2QE3gMM2_mjS37rZJ9GK0Rzp7MUG1ZXfXSDULbXkbcV9ufpGZ1lweHa42DwjXP03peXdE2n5qt2-V3tkynkk2bT3YZu14JZcu8zTbrhVe77CeuDDxT9E8XZjTbdSX3J0j4MBopMO2D1YNy1ZbeoJOWF_msVbUlgIdX4YPsf0Fl3iqptX0RSjp9AdROaDKofvSxqPFgdPeDFmC4TOBpVVfqrEwQeydNUIOwk8fDCYUI-xTvKF_FpJzoQaDgLogZO6gEiOcxJI1gQC52EkfYRCvwGBmMBmWdlBlGs4c4mFS15U0JenGW-qRuTJ3VGj05dQQ3EfJBRZ5NxLzV54gld9EM7AX0k3XXZZBurwKuIScXxGh2EMf0SsQsIvcN6_iaU7JabRilpOs4r2nTckjHn6YJAvmZtcDqoLcwHmPwZyhffwBGK6joGSocazl7jjV5r_9AJfJXVSIPVCIPRSKfEYnci7TiX1YpMf-8IORP5SD_vBpfv2W-xfgW41uMbzH-EzGeitcuMagpJNv-5Fys6HSQXuFLyaB2vTXFVAzOThP4C1Y0eGhnXQ9rJDIOd1AhK8BHCTyr4IWXZyXOTVVXByUHPRoldqPzIZdow-827IuZXyqWHa-A0pDPtJv7nXzo5kxqF1ZAJ__Ixcl7V9DREkgMo7pqjQhU4wGaVm9VWZXclbjEtyvoPpr61g6plkZBCKWDXLqCLrNmmeVy-lPfQxdU253UXoH26WZ5j13p_Aj9B1Kq3U71se35Zk1owwhnnEGWUeLYKeW7msOulQG80I5TzglvecdWrGOkpdAC4eSgFyjEhoot3QlNxeoVfAOQPGDewV9XNwuU4HnW4BUPMBhNgXf4G0eLCMV_mucET3z8JWDTLvYEP1XaxSfL8TDOEzI_jjo9Qlpc-knCLRYcfoI0EQLdy-zKJGIIGn3yvOwwFoM1QyYlrzkrnllwZEIXZu1SDVKU76OHcid94N3bz8805L7kNydzk2EbnwnxyXMRkx1N7A32_hu0lb1U) |
| US | T7 AR1 | Mexico (broken, True) | +18.2 ± 2.2 | [aa88cc5eab3e08ec](https://mihaild.github.io/DeepStruggle/?pos=eJztV-GK4yAQfhd_Z0FjUtu-ylHETWwbzqqo6W5Z9t131CRNtj3ownFwUEKp8804zsw3Cc4HEk3ojObO9LpFW1KgRriWK9OIiHu0_YWLEh6SnjX8WPqxJLNJYoPFd_mKkeRnNSHZ52qyYZOGFPWwqgdNndZ00FaTph7244efnEuOCOQdJHzstOAp7aNRrXS5DFfUKnERr0pmPLxzH0TzGyrzgYRS5o1L4dQFUGPRFheolU3nY1HjxsHdhIWLBUd1geRZ6sAPTuggW26sT3szCgccZJJP4p1b6XgD_AR3SZg2rcwAf-1CpiinYiw_gTJZgccYQKfRlhbISt12-gDnpJzQtqrmoOdnoXq5MHVGqeTJyRO4iZAPMsZJI-aNOkcsu4tmYM-hnYy7ikG4gwxJhpxc4J3ewzm6kTxmEWMnZLUua8IoZmVF15uSYFpQXJV4Axn53lo4y4-B81Sg5DB0p3j-K1SwOaJtWUNRz1DkWM7ReSzLZ3GPJ_IQT-Sf8oTv8ITv0oQfoWn9N2nKgT9OB_4JGfdeGnxDBv6fycBPMp5kPMl4kvEk48dk7GbXLt5KG7Jt0zsXK2qPwkN-ZWRi3xg9W_LWGWvBXzC8TJv2xjUgp0C69gaa0QrwSUCcKHjuxVnyc4kKdJSiVZ2WfN85H8bLFLtRmDcd_byQmcZLCKkd9rDNrWbYtNiT24XOoN7fc9F772bhKAFBtJ28cp0QqMYdNEsfaF6VoSuTmO5W0H0k961pcy21hCOkCmLqCjKtymk1lNP3TQNdgLZ7obwE7vPL8hm70vkO-g-olPu9bGLbE8qqumaUsBquMZHh2Cizi5pLPStCvKStKKVVTdarNWEbui7ZKvZW76AToBJQrqk1oaOqNDKM40AFIw1OI8HtqECSZlzhhQ2ZsDiOjL4w-MYz3_mfLfyVM4nO1tmeVCOwS5GfOjV9P0D0VsD7FVyaP0iEgPR5dt8zIGQRUDXMQFeUTgFd56O4kSYbMlnRRUrfnzGR6o8W2dc1nN0Q_DJBABYpVvGLEL87F25Np2NnvFSfX-Tyu3k) |
| US | T10 AR4 | Italy (broken, True) | +17.9 ± 2.1 | [bfe95ca7ea25ce0a](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReevRJgfIl_ZVUhapMELTEW4LRR1X_fAWyHNHmoutI-7FaWE-YwzO0MFrwh0XtlRm7NPA6oYwXqhR24Nr0IuEPdT1zUBYG3LljRwm-zPfm4iVq3aJ09pKjizKp1nW2WX7qtrzMNtlhMMVw9sUwntxj-yeazBqvBMt2k1StNdp8g4aMaBY9pH40epEXdD3IDT1pcxLOWqMOA-1fuvOh_QWnekNDavHAprL4AaqaoMsheuVDVsDDYIxnmLxMYqgokz3L0_GDF6OXAzeTi2oSCg0NydxKvfJKW90CQt5eIjWaQCeDPyieO4IFczMRPMBm1wGIIQI2oKws0yXFQ4wH8xJxQt2ty0PGz0LO8UbVG62jJyhOYCZDzMsRZBswZfQ5YMhfUQJ9DPxl7Fb2wB-mjDDlZz9W4Bz9jL3nIIsROMKZt1eC6LBvW7nBFSqCXNpS0kJGbpwl8uTVwHgsUDXp1Cv6foYL9EXW0gqKeocihnKvxUJb34us84b_KE37AE35IE_4MTfieprb8Mk8p8s_zgf-UDfzPs_H1TfNNxjcZ32R8k_GfkPGUnbv4ICefdPvZ2lDR6Sgc5EcDE_vepBNPGvLBmmkCe95wGhftje1BjoGo4Q7KaAX4JCBO5B134iz5maICHaUYtBol3yvr_FIiVt5NmJdxPUtuM05CSMOypm7uZ5ZFN2tSu-QOZvfIxOyczcLRAoIYlLxyHRGoxgM0SW8or8rSlVGMhyvoPpL61gyplqMEF1J7sXUF2UZ0Gy3ldHPfQxegbi-0k8B92izvoSutU9B_QKXc72Uf2r5sWlrtGkxa0oIGMBwaJTup2dizwoMNwghtWM2aXQ3nx4qVNd2B_dlCJxDwDPXaehNaisFdBMN9IL0sbK-CMBBIvFjE_VaUyz-OKIYLQ3pxvE6UEW-X2esqtq0kUZss43IbJ50VD1K1oelp4gyLWwciPymdPiBlFN0kYIN5G28gLEDAep4dLq6Bx7xisvWKhetcmiqj1xRDXeAHT71kvFYkSVVWIXqjT7b65Njug9Uq6pAlP4h-y5AuwMcczyp8eS58MmoMvbF7_w1Bdb2e) |
| US | T6 AR4 | Italy (broken, True) | +17.6 ± 2.5 | [4f92664e5d1a7124](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tq4zAQ_Rc_u6Bb7Dq_shSh2kpiVpGMJKcNpf_e0SWOQvKQLmUXliASS2cumpkzMvJHJXo_Gs2tmfVQrVld9cIOXJleBNxV61-obpbxDD9St3ngZRZGk_-bPMeFXZMl7GKNwVepQUCe9HBhzRbv4ceyhOTdkw66e5S7w_oFEt6NWvCY9s6oQdpq_YQv4EmJo3hVsloH3L9z50X_G0rzUQmlzBuXwqojoGaq1qiuBtmPLlQ1GAZ_uMD8cQJHq7qSB6k931qhvRy4mVy0TShssJVxvRfvfJKW90CQt8eIaTPIBPDX0SeOUi5m4nsQRi3wGAIYdaR1knoY9Rb2iTlBBE0JOn4QapYXqtYoFT1ZuQc3AXJehjhZwJxRh4Ald0EN9Dn0k7HnpRd2K31cQ07W81FvYB_dSx6yCLF33Yp0HWspRl3X4uf2GTqhIQR1kJCbpwm2cqe4eaxP9OfHfdj-FQrY72KND1DhUMuT51CTz_pekp5-kiX8ByyhGyyhmyShe0hCP0lSCvx-NtA1G7jBqMGkpc23aEFXrKC_enb-c1a-d0YeZDzIeJDxIOPfkPFS3L74ICefdPvZ2lDRaScc5EcCE5ve6GLKB2umCfx5w0k02hjbwzoGMg5XUEErwHsBcVbecScOkh9IVVc7KQY1ask3o3X-dKdCVwLzpk-Xi0XiJIQ0ZBuM6LUoW10YpX4plWd308fsnD1zpgREMYzyTHZEoBw30LT6qMqy5LaMS7KCKxG0H06Na4ZUTC1hC6m8WNoCLzOyzHI93dz30AbVeiOUk0B-Oi2foS2tG6EBgUu52cg-9D1eIcpQQ2kTtg4Uh04pLtY2Nq3w4KNDjLWMUcRaRLqWtitoAD9b6AS47EK1ltaEjiL1qvgswDQ-2BnIT5qfJP5wISfLE-dxsmaF49IHy35K-7P3W4PGkwOR70eV3h8kLt0k4Hx5Gz9DWICA8jK7S9cExxhO30lNjCTES5es6FXUt-OhRbz4hjzJ7vGFAgEvOfYlP5qBiwxpeCGE186RT2bUoS-eus8vdM69iA) |

## What a Cuba placement with Fidel to come costs

Paired playouts of 1,264 US Cuba plays from 2,000 games, 512 pairs each: the same card's Ops kept out of Cuba minus the model's own play, points of win probability.

| group | plays | per game | mean gain a play | per game, points |
|:---|---:|---:|---:|---:|
| all | 1264 | 0.63 | -0.29 ± 0.09 | -0.18 ± 0.06 |
| Fidel unseen | 574 | 0.29 | +0.09 ± 0.15 | +0.03 ± 0.04 |
| Fidel discard | 562 | 0.28 | -0.56 ± 0.12 | -0.16 ± 0.03 |
| Fidel US hand | 128 | 0.06 | -0.81 ± 0.26 | -0.05 ± 0.02 |
| T1-3 | 122 | 0.06 | -0.14 ± 0.27 | -0.01 ± 0.02 |
| T4-7 | 1025 | 0.51 | -0.26 ± 0.10 | -0.13 ± 0.05 |
| T8-10 | 117 | 0.06 | -0.69 ± 0.30 | -0.04 ± 0.02 |
| Fidel fired later in the game | 438 | 0.22 | -0.36 ± 0.15 | -0.08 ± 0.03 |
| Fidel never fired | 826 | 0.41 | -0.25 ± 0.12 | -0.10 ± 0.05 |

