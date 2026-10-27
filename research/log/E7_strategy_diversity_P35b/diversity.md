# Strategy diversity: self-play, searched player and humans (P35b a)

| source | games | whole games | opening (T1-3) complete |
|:---|---:|---:|---:|
| greedy | 16,384 | 16,384 | 15,682 |
| searched | 2,048 | 0 | 1,951 |
| humans | 265 | 118 | 230 |

Entropy in bits; *eff.* = 2^H, the number of equally likely choices it amounts to. *H@humans* is the entropy of a sample the size of humans's (mean of 200 draws without replacement). *humans unplayed* is the share of humans's games whose choice this source makes in under 1% of its games; *TVD* the total variation distance to humans.

## setup -- US

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 16,384 | 2 | 0.02 | 1.0 | 0.02 | 1.0 | 99.8% | 1.00 | 100.0% |
| searched | 2,048 | 11 | 1.10 | 2.1 | 1.09 | 2.1 | 74.1% | 1.00 | 100.0% |
| humans | 254 | 12 | 2.06 | 4.2 | 2.06 | 4.2 | 48.0% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| West Germany 4, Italy 3, Iran 1, Panama 1 | 99.8% | 74.1% | 0.0% |
| Italy 4, West Germany 4, Iran 1 | 0.0% | 0.0% | 48.0% |
| France 3, West Germany 3, Italy 2, Iran 1 | 0.0% | 0.0% | 30.7% |
| West Germany 4, Italy 2, France 1, Iran 1, Panama 1 | 0.2% | 21.6% | 0.0% |
| Italy 3, West Germany 3, France 2, Iran 1 | 0.0% | 0.0% | 7.1% |
| West Germany 4, Italy 3, Iran 2 | 0.0% | 0.0% | 4.7% |
| West Germany 4, Italy 3, France 1, Iran 1 | 0.0% | 0.0% | 3.1% |
| West Germany 4, France 3, Italy 2 | 0.0% | 0.0% | 3.1% |

## setup -- USSR

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 16,384 | 1 | -0.00 | 1.0 | 0.00 | 1.0 | 100.0% | 1.00 | 100.0% |
| searched | 2,048 | 6 | 0.27 | 1.2 | 0.26 | 1.2 | 96.2% | 1.00 | 100.0% |
| humans | 254 | 5 | 1.62 | 3.1 | 1.62 | 3.1 | 45.3% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| Poland 3, Yugoslavia 3 | 100.0% | 96.2% | 0.0% |
| Poland 4, Austria 1, East Germany 1 | 0.0% | 0.0% | 45.3% |
| Poland 4, East Germany 1, Yugoslavia 1 | 0.0% | 0.0% | 29.9% |
| Poland 5, East Germany 1 | 0.0% | 0.0% | 23.6% |
| Poland 3, Yugoslavia 2, East Germany 1 | 0.0% | 3.1% | 0.0% |
| Austria 4, Poland 2 | 0.0% | 0.0% | 0.8% |
| Finland 3, Poland 3 | 0.0% | 0.0% | 0.4% |
| Yugoslavia 3, Poland 2, East Germany 1 | 0.0% | 0.3% | 0.0% |

## AR1 -- US

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 16,384 | 32 | 3.27 | 9.6 | 3.20 | 9.2 | 33.7% | 0.33 | 13.0% |
| searched | 2,048 | 28 | 3.41 | 10.6 | 3.34 | 10.2 | 29.2% | 0.28 | 13.0% |
| humans | 254 | 27 | 3.65 | 12.5 | 3.65 | 12.5 | 21.7% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| influence Europe+Asia | 33.7% | 29.2% | 11.4% |
| influence Asia+Middle East | 11.7% | 14.0% | 21.7% |
| influence Asia | 11.7% | 13.4% | 15.4% |
| coup Iran | 11.3% | 10.6% | 9.4% |
| influence Europe | 4.0% | 3.7% | 9.1% |
| influence Europe+Asia+Middle East | 5.1% | 5.7% | 7.5% |
| influence Europe+Middle East | 2.6% | 2.6% | 3.9% |
| coup Laos/Cambodia | 3.9% | 3.6% | 0.8% |

## AR1 -- USSR

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 16,384 | 10 | 1.75 | 3.4 | 1.74 | 3.3 | 50.9% | 0.38 | 16.9% |
| searched | 2,048 | 15 | 1.92 | 3.8 | 1.88 | 3.7 | 49.3% | 0.36 | 16.9% |
| humans | 254 | 13 | 2.43 | 5.4 | 2.43 | 5.4 | 50.8% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| coup Italy | 50.9% | 49.3% | 17.3% |
| coup Iran | 32.8% | 32.7% | 50.8% |
| event | 7.7% | 7.7% | 6.3% |
| influence Asia | 0.0% | 0.4% | 6.3% |
| influence Europe | 2.8% | 2.7% | 5.9% |
| influence Asia+Middle East | 0.0% | 0.1% | 4.3% |
| Asia Scoring | 3.9% | 3.3% | 1.2% |
| Europe Scoring | 1.8% | 2.0% | 1.6% |

## main region, T1-3 -- US

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 15,682 | 6 | 1.18 | 2.3 | 1.16 | 2.2 | 55.1% | 0.18 | 1.3% |
| searched | 1,951 | 6 | 1.19 | 2.3 | 1.18 | 2.3 | 57.6% | 0.15 | 1.3% |
| humans | 230 | 5 | 1.10 | 2.1 | 1.10 | 2.1 | 70.9% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| Asia | 55.1% | 57.6% | 70.9% |
| Europe | 42.4% | 39.5% | 24.8% |
| Middle East | 1.2% | 1.4% | 3.0% |
| Africa | 0.6% | 0.5% | 0.9% |
| South America | 0.7% | 0.8% | 0.4% |
| Central America | 0.1% | 0.3% | 0.0% |

## main region, T1-3 -- USSR

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 15,682 | 6 | 1.41 | 2.7 | 1.39 | 2.6 | 48.7% | 0.21 | 0.9% |
| searched | 1,951 | 6 | 1.40 | 2.6 | 1.39 | 2.6 | 52.0% | 0.18 | 0.4% |
| humans | 230 | 5 | 1.28 | 2.4 | 1.28 | 2.4 | 66.5% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| Asia | 48.7% | 52.0% | 66.5% |
| Europe | 42.5% | 39.4% | 22.2% |
| Middle East | 7.4% | 7.0% | 10.4% |
| South America | 0.9% | 1.2% | 0.4% |
| Africa | 0.3% | 0.3% | 0.4% |
| Central America | 0.1% | 0.1% | 0.0% |

## top two regions (as a set), T1-3 -- US

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 15,682 | 15 | 1.42 | 2.7 | 1.38 | 2.6 | 77.3% | 0.17 | 0.9% |
| searched | 1,951 | 11 | 1.50 | 2.8 | 1.48 | 2.8 | 75.3% | 0.16 | 0.9% |
| humans | 230 | 8 | 1.64 | 3.1 | 1.64 | 3.1 | 66.1% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| Europe + Asia | 77.3% | 75.3% | 66.1% |
| Asia + Middle East | 6.4% | 7.1% | 15.2% |
| Asia + Africa | 2.1% | 2.3% | 9.1% |
| Europe + Middle East | 4.4% | 4.4% | 3.9% |
| Asia + South America | 2.8% | 3.8% | 1.3% |
| Europe + Africa | 2.8% | 2.9% | 2.6% |
| Europe + South America | 1.9% | 1.6% | 0.9% |
| Asia + Central America | 1.3% | 1.7% | 0.0% |

## top two regions (as a set), T1-3 -- USSR

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 15,682 | 15 | 1.97 | 3.9 | 1.93 | 3.8 | 53.9% | 0.19 | 2.6% |
| searched | 1,951 | 14 | 1.93 | 3.8 | 1.90 | 3.7 | 55.2% | 0.18 | 2.6% |
| humans | 230 | 9 | 1.94 | 3.8 | 1.94 | 3.8 | 45.2% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| Europe + Asia | 53.9% | 55.2% | 45.2% |
| Asia + Middle East | 22.9% | 22.3% | 35.2% |
| Europe + Middle East | 12.8% | 11.8% | 7.0% |
| Asia + South America | 3.1% | 4.4% | 6.5% |
| Asia + Africa | 1.6% | 1.4% | 3.0% |
| Europe + South America | 2.5% | 2.2% | 0.4% |
| Middle East + Africa | 0.2% | 0.1% | 1.3% |
| Middle East + South America | 0.8% | 0.6% | 0.9% |

## coup targets, T1-3 (each coup) -- US

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 31,184 | 33 | 3.89 | 14.9 | 3.85 | 14.5 | 17.2% | 0.23 | 15.0% |
| searched | 4,195 | 30 | 3.94 | 15.4 | 3.91 | 15.0 | 15.4% | 0.24 | 14.6% |
| humans | 526 | 34 | 4.13 | 17.5 | 4.13 | 17.5 | 17.9% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| Iran | 17.2% | 15.4% | 17.9% |
| Egypt | 11.6% | 11.2% | 12.4% |
| Colombia | 10.2% | 11.0% | 7.2% |
| Syria | 8.1% | 9.4% | 1.1% |
| Vietnam | 5.3% | 5.9% | 8.7% |
| Iraq | 8.6% | 8.1% | 7.2% |
| Venezuela | 7.2% | 6.5% | 1.7% |
| Thailand | 4.8% | 4.5% | 6.1% |

## coup targets, T1-3 (each coup) -- USSR

| source | n | distinct | H | eff. | H@humans | eff.@humans | top share | TVD | humans unplayed |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 60,462 | 30 | 3.13 | 8.8 | 3.11 | 8.6 | 31.3% | 0.20 | 17.4% |
| searched | 7,689 | 26 | 3.27 | 9.6 | 3.25 | 9.5 | 29.4% | 0.21 | 11.8% |
| humans | 798 | 28 | 3.47 | 11.1 | 3.47 | 11.1 | 32.7% | 0.00 | -- |

| choice | greedy | searched | humans |
|:---|---:|---:|---:|
| Iran | 31.3% | 29.4% | 32.7% |
| Panama | 15.0% | 14.1% | 16.9% |
| Italy | 15.1% | 14.4% | 6.3% |
| Pakistan | 9.3% | 8.7% | 6.3% |
| Colombia | 7.9% | 8.8% | 5.8% |
| Lebanon | 5.7% | 6.1% | 3.9% |
| Egypt | 1.0% | 1.5% | 3.6% |
| Libya | 2.6% | 2.5% | 3.5% |

## Counts per game (mean ± SD across games)

Opening counts over games whose turns 1-3 are complete; whole-game counts over games played to the end (a searched player cut at turn 3 has none).

| feature | side | greedy | searched | humans |
|:---|:---|---:|---:|---:|
| coups, T1-3 | US | 1.99 ± 1.26 | 2.15 ± 1.36 | 2.29 ± 1.35 |
| coups, T1-3 | USSR | 3.86 ± 1.28 | 3.94 ± 1.37 | 3.47 ± 1.36 |
| battleground coups, T1-3 | US | 1.29 ± 0.85 | 1.32 ± 0.86 | 1.56 ± 0.80 |
| battleground coups, T1-3 | USSR | 3.20 ± 0.92 | 3.18 ± 0.93 | 2.80 ± 0.89 |
| realignment rolls, T1-3 | US | 1.10 ± 1.86 | 1.21 ± 1.91 | 0.47 ± 1.25 |
| realignment rolls, T1-3 | USSR | 0.04 ± 0.36 | 0.19 ± 0.72 | 0.18 ± 0.77 |
| space plays, T1-3 | US | 1.34 ± 0.91 | 1.30 ± 0.92 | 1.59 ± 0.93 |
| space plays, T1-3 | USSR | 0.69 ± 0.72 | 0.66 ± 0.72 | 0.87 ± 0.85 |
| influence added, T1-3 | US | 31.34 ± 6.22 | 31.39 ± 6.43 | 31.95 ± 6.08 |
| influence added, T1-3 | USSR | 37.19 ± 6.41 | 37.08 ± 6.39 | 38.90 ± 6.62 |
| coups, game | US | 7.11 ± 3.69 | -- | 10.61 ± 3.45 |
| coups, game | USSR | 9.89 ± 4.09 | -- | 12.17 ± 3.78 |
| realignment rolls, game | US | 6.27 ± 5.67 | -- | 5.71 ± 4.86 |
| realignment rolls, game | USSR | 2.69 ± 4.12 | -- | 3.69 ± 3.86 |
| space plays, game | US | 4.06 ± 2.07 | -- | 5.42 ± 1.96 |
| space plays, game | USSR | 3.79 ± 2.04 | -- | 5.00 ± 1.75 |

## Influence added in turns 1-3, share by region (mean ± SD across games)

| side | region | greedy | searched | humans |
|:---|:---|---:|---:|---:|
| US | Europe | 31.7% ± 13.6% | 30.1% ± 13.2% | 26.3% ± 11.6% |
| US | Asia | 36.8% ± 14.7% | 37.1% ± 14.3% | 39.6% ± 13.1% |
| US | Middle East | 11.0% ± 7.7% | 11.1% ± 7.8% | 15.0% ± 8.6% |
| US | Africa | 8.1% ± 6.7% | 8.4% ± 6.5% | 11.5% ± 7.9% |
| US | Central America | 7.4% ± 5.2% | 7.6% ± 5.4% | 5.0% ± 4.5% |
| US | South America | 5.1% ± 7.6% | 5.7% ± 7.8% | 2.5% ± 5.8% |
| USSR | Europe | 29.2% ± 15.2% | 28.4% ± 14.6% | 21.8% ± 12.6% |
| USSR | Asia | 31.6% ± 12.7% | 32.2% ± 12.2% | 35.9% ± 11.5% |
| USSR | Middle East | 19.4% ± 8.7% | 19.2% ± 8.7% | 20.9% ± 9.4% |
| USSR | Africa | 6.4% ± 6.1% | 6.5% ± 5.8% | 7.6% ± 7.3% |
| USSR | Central America | 5.8% ± 5.5% | 5.4% ± 5.4% | 6.6% ± 5.5% |
| USSR | South America | 7.6% ± 8.1% | 8.2% ± 8.3% | 7.1% ± 8.4% |
