# Where Operations go, humans against bots, country by country

Humans: the ts-replayer corpus, 254 games with Ops targets ({"complete": 119, "partial": 146, "skipped": 9}); 30,080 influence, 4,466 coup, 1,729 realign. Bots' own games: **soup** 4,096 greedy self-play games of E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt (91a43a8b089c…); **r32** 4,096 greedy self-play games of R32_swa_6720-6800M.pt (1d4f65f04068…). Same positions: **soup** E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt (91a43a8b089c…); **r32** R32_swa_6720-6800M.pt (1d4f65f04068…), asked at every human target's position.

A share is the side's targets in that mode that went to the country. *Same positions* is the bot's probability for the country, averaged over the humans' decisions (its own pick in brackets). The gap is humans − bot on those decisions, ± one standard error clustered by game; bold where it is 2+ SE.

## US influence

### Regions by era

| region | era | humans | soup own games | r32 own games | soup same positions | r32 same positions | humans − soup, same positions | humans − r32, same positions | human points |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Europe | Early War | 23 | 20 | 21 | 24 | 24 | -1 ± 1 | -1 ± 1 | 7,447 |
| Europe | Mid War | 12 | 16 | 17 | 14 | 14 | -1 ± 1 | -1 ± 1 | 5,571 |
| Europe | Late War | 25 | 34 | 33 | 29 | 29 | **-4 ± 2** | **-4 ± 2** | 2,641 |
| Asia | Early War | 39 | 41 | 41 | 50 | 51 | **-11 ± 1** | **-12 ± 1** | 7,447 |
| Asia | Mid War | 18 | 18 | 19 | 19 | 19 | -1 ± 1 | -1 ± 1 | 5,571 |
| Asia | Late War | 15 | 14 | 13 | 11 | 13 | **+4 ± 1** | **+3 ± 1** | 2,641 |
| Middle East | Early War | 19 | 15 | 14 | 9 | 9 | **+10 ± 1** | **+10 ± 1** | 7,447 |
| Middle East | Mid War | 10 | 7 | 6 | 6 | 6 | **+4 ± 1** | **+4 ± 1** | 5,571 |
| Middle East | Late War | 19 | 12 | 12 | 14 | 13 | **+5 ± 2** | **+6 ± 2** | 2,641 |
| Africa | Early War | 11 | 6 | 6 | 2 | 3 | **+9 ± 1** | **+8 ± 1** | 7,447 |
| Africa | Mid War | 22 | 17 | 17 | 15 | 16 | **+6 ± 1** | **+6 ± 1** | 5,571 |
| Africa | Late War | 16 | 16 | 17 | 16 | 16 | -0 ± 2 | -0 ± 1 | 2,641 |
| Central America | Early War | 5 | 9 | 9 | 7 | 7 | **-1 ± 1** | **-2 ± 1** | 7,447 |
| Central America | Mid War | 16 | 21 | 21 | 28 | 28 | **-12 ± 1** | **-12 ± 1** | 5,571 |
| Central America | Late War | 11 | 13 | 14 | 18 | 18 | **-8 ± 2** | **-7 ± 2** | 2,641 |
| South America | Early War | 3 | 8 | 8 | 8 | 7 | **-5 ± 1** | **-4 ± 1** | 7,447 |
| South America | Mid War | 21 | 20 | 20 | 17 | 17 | **+4 ± 1** | **+5 ± 1** | 5,571 |
| South America | Late War | 15 | 11 | 12 | 12 | 12 | **+3 ± 1** | **+3 ± 1** | 2,641 |

### Countries, Early War, turns 1-3 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Malaysia | Asia | 4.9 | 6.8 | 6.9 | 24.2 (24.5) | 24.2 (25.0) | **-19.3 ± 1.0** | **-19.3 ± 0.9** |
| France | Europe | 12.0 | 2.1 | 2.2 | 17.1 (17.2) | 16.6 (17.6) | **-5.1 ± 1.0** | **-4.6 ± 0.9** |
| Colombia | South America | 0.8 | 2.0 | 1.9 | 6.3 (6.3) | 5.1 (5.3) | **-5.4 ± 0.7** | **-4.2 ± 0.5** |
| Afghanistan | Asia | 1.2 | 3.6 | 3.7 | 4.3 (4.3) | 4.6 (4.4) | **-3.1 ± 0.5** | **-3.4 ± 0.5** |
| Egypt | Middle East | 4.7 | 4.3 | 4.1 | 1.6 (1.5) | 1.7 (1.5) | **+3.1 ± 0.4** | **+3.1 ± 0.4** |
| Pakistan | Asia | 5.9 | 6.1 | 6.2 | 3.1 (3.2) | 3.0 (3.3) | **+2.8 ± 0.4** | **+2.9 ± 0.4** |
| India | Asia | 4.7 | 4.0 | 4.0 | 2.2 (2.2) | 2.4 (2.3) | **+2.4 ± 0.4** | **+2.3 ± 0.4** |
| Angola | Africa | 2.3 | 0.2 | 0.2 | 0.0 (0.0) | 0.0 (0.0) | **+2.3 ± 0.2** | **+2.2 ± 0.2** |
| West Germany | Europe | 3.7 | 9.2 | 9.6 | 1.6 (1.6) | 1.7 (1.7) | **+2.2 ± 0.3** | **+2.0 ± 0.3** |
| Burma | Asia | 2.0 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+2.0 ± 0.2** | **+2.0 ± 0.2** |
| South Africa | Africa | 1.9 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+1.9 ± 0.2** | **+1.9 ± 0.2** |
| Thailand | Asia | 7.3 | 9.3 | 9.2 | 5.5 (5.6) | 5.2 (5.5) | **+1.9 ± 0.4** | **+2.2 ± 0.4** |

### Countries, Mid and Late War, turns 4-10 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Mexico | Central America | 5.6 | 6.7 | 6.7 | 10.1 (10.5) | 9.9 (10.8) | **-4.5 ± 0.8** | **-4.2 ± 0.8** |
| Israel | Middle East | 3.8 | 1.3 | 1.2 | 0.6 (0.6) | 0.7 (0.5) | **+3.2 ± 0.3** | **+3.2 ± 0.3** |
| South Africa | Africa | 4.6 | 2.9 | 2.7 | 1.6 (1.8) | 1.7 (1.9) | **+3.0 ± 0.4** | **+3.0 ± 0.4** |
| Cuba | Central America | 3.5 | 6.0 | 6.0 | 6.3 (6.5) | 6.6 (6.6) | **-2.8 ± 0.6** | **-3.1 ± 0.6** |
| Guatemala | Central America | 0.7 | 1.2 | 1.1 | 3.7 (3.9) | 3.4 (3.3) | **-3.1 ± 0.4** | **-2.7 ± 0.3** |
| Uruguay | South America | 1.7 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+1.7 ± 0.2** | **+1.7 ± 0.2** |
| Afghanistan | Asia | 0.6 | 0.6 | 0.7 | 2.3 (2.3) | 2.4 (2.5) | **-1.7 ± 0.4** | **-1.8 ± 0.4** |
| Malaysia | Asia | 0.6 | 0.2 | 0.2 | 2.4 (2.2) | 2.3 (2.1) | **-1.7 ± 0.3** | **-1.7 ± 0.3** |
| Saudi Arabia | Middle East | 1.9 | 0.6 | 0.6 | 0.3 (0.3) | 0.3 (0.3) | **+1.6 ± 0.2** | **+1.6 ± 0.2** |
| United Kingdom | Europe | 1.0 | 1.6 | 1.7 | 2.5 (2.5) | 2.7 (2.8) | **-1.4 ± 0.4** | **-1.7 ± 0.4** |
| Angola | Africa | 3.7 | 2.5 | 2.5 | 2.3 (2.4) | 2.3 (2.4) | **+1.4 ± 0.4** | **+1.4 ± 0.4** |
| West Germany | Europe | 4.6 | 3.3 | 3.2 | 3.3 (3.3) | 3.2 (3.5) | **+1.4 ± 0.4** | **+1.4 ± 0.4** |

## USSR influence

### Regions by era

| region | era | humans | soup own games | r32 own games | soup same positions | r32 same positions | humans − soup, same positions | humans − r32, same positions | human points |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Europe | Early War | 15 | 23 | 24 | 16 | 17 | -1 ± 1 | -1 ± 1 | 6,032 |
| Europe | Mid War | 11 | 17 | 18 | 12 | 13 | -1 ± 1 | -2 ± 1 | 6,024 |
| Europe | Late War | 25 | 33 | 33 | 23 | 24 | +2 ± 2 | +1 ± 2 | 2,365 |
| Asia | Early War | 46 | 38 | 39 | 39 | 40 | **+7 ± 1** | **+6 ± 1** | 6,032 |
| Asia | Mid War | 20 | 19 | 19 | 18 | 17 | **+2 ± 1** | **+3 ± 1** | 6,024 |
| Asia | Late War | 15 | 15 | 14 | 10 | 10 | **+5 ± 1** | **+5 ± 1** | 2,365 |
| Middle East | Early War | 24 | 22 | 21 | 32 | 30 | **-9 ± 1** | **-7 ± 1** | 6,032 |
| Middle East | Mid War | 13 | 9 | 9 | 14 | 13 | -0 ± 1 | -0 ± 1 | 6,024 |
| Middle East | Late War | 18 | 9 | 8 | 11 | 11 | **+7 ± 2** | **+7 ± 1** | 2,365 |
| Africa | Early War | 6 | 4 | 4 | 5 | 4 | **+1 ± 1** | **+2 ± 1** | 6,032 |
| Africa | Mid War | 20 | 14 | 14 | 15 | 16 | **+4 ± 1** | **+4 ± 1** | 6,024 |
| Africa | Late War | 16 | 16 | 17 | 20 | 19 | **-5 ± 1** | **-4 ± 1** | 2,365 |
| Central America | Early War | 1 | 4 | 4 | 1 | 1 | +0 ± 0 | -0 ± 0 | 6,032 |
| Central America | Mid War | 10 | 17 | 17 | 14 | 14 | **-4 ± 1** | **-4 ± 1** | 6,024 |
| Central America | Late War | 13 | 14 | 14 | 20 | 19 | **-7 ± 2** | **-6 ± 1** | 2,365 |
| South America | Early War | 7 | 9 | 9 | 7 | 7 | +1 ± 1 | -0 ± 1 | 6,032 |
| South America | Mid War | 26 | 24 | 24 | 27 | 26 | -1 ± 1 | -0 ± 1 | 6,024 |
| South America | Late War | 14 | 14 | 14 | 16 | 17 | -2 ± 2 | -3 ± 1 | 2,365 |

### Countries, Early War, turns 1-3 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Lebanon | Middle East | 1.3 | 3.4 | 2.8 | 11.0 (11.2) | 10.0 (10.9) | **-9.8 ± 1.0** | **-8.8 ± 0.8** |
| South Korea | Asia | 12.9 | 10.0 | 9.8 | 7.8 (7.7) | 8.0 (7.8) | **+5.0 ± 0.7** | **+4.9 ± 0.7** |
| Iraq | Middle East | 6.5 | 7.4 | 7.2 | 11.8 (11.9) | 10.8 (11.0) | **-5.4 ± 0.8** | **-4.4 ± 0.7** |
| Laos/Cambodia | Asia | 3.2 | 2.6 | 2.7 | 7.9 (8.2) | 7.5 (8.2) | **-4.7 ± 0.7** | **-4.3 ± 0.6** |
| Saudi Arabia | Middle East | 4.1 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+4.1 ± 0.3** | **+4.1 ± 0.3** |
| Burma | Asia | 3.2 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+3.2 ± 0.3** | **+3.2 ± 0.3** |
| India | Asia | 6.5 | 3.5 | 3.5 | 3.0 (2.9) | 3.4 (3.3) | **+3.5 ± 0.4** | **+3.0 ± 0.4** |
| Hungary | Europe | 0.0 | 1.7 | 1.8 | 2.5 (2.3) | 2.4 (1.7) | **-2.4 ± 0.4** | **-2.4 ± 0.3** |
| Syria | Middle East | 2.3 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+2.3 ± 0.1** | **+2.3 ± 0.1** |
| Yugoslavia | Europe | 0.1 | 1.1 | 1.3 | 2.3 (2.4) | 2.7 (2.7) | **-2.2 ± 0.3** | **-2.5 ± 0.3** |
| Vietnam | Asia | 0.3 | 0.7 | 0.6 | 2.1 (2.0) | 2.4 (2.2) | **-1.8 ± 0.4** | **-2.1 ± 0.4** |
| Spain/Portugal | Europe | 1.6 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+1.6 ± 0.2** | **+1.6 ± 0.2** |

### Countries, Mid and Late War, turns 4-10 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Colombia | South America | 1.0 | 1.6 | 1.5 | 5.8 (5.9) | 6.1 (6.4) | **-4.8 ± 0.6** | **-5.1 ± 0.6** |
| Saudi Arabia | Middle East | 4.3 | 0.1 | 0.1 | 0.2 (0.2) | 0.2 (0.2) | **+4.1 ± 0.3** | **+4.1 ± 0.3** |
| South Africa | Africa | 3.8 | 1.7 | 1.9 | 0.9 (0.8) | 0.9 (0.9) | **+2.9 ± 0.4** | **+2.8 ± 0.4** |
| Israel | Middle East | 2.8 | 0.1 | 0.0 | 0.1 (0.1) | 0.1 (0.1) | **+2.7 ± 0.3** | **+2.7 ± 0.3** |
| Lebanon | Middle East | 0.3 | 0.7 | 0.6 | 3.0 (2.9) | 2.7 (2.8) | **-2.7 ± 0.5** | **-2.4 ± 0.5** |
| Iraq | Middle East | 1.5 | 0.9 | 0.9 | 3.6 (3.6) | 3.5 (3.6) | **-2.1 ± 0.5** | **-2.0 ± 0.4** |
| Uruguay | South America | 1.6 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+1.5 ± 0.2** | **+1.6 ± 0.2** |
| Cameroon | Africa | 0.9 | 1.2 | 1.1 | 2.7 (2.7) | 2.4 (2.4) | **-1.8 ± 0.3** | **-1.5 ± 0.2** |
| Yugoslavia | Europe | 0.3 | 1.0 | 1.0 | 1.8 (1.8) | 1.9 (1.8) | **-1.5 ± 0.2** | **-1.6 ± 0.2** |
| Japan | Asia | 1.6 | 0.0 | 0.0 | 0.1 (0.1) | 0.2 (0.1) | **+1.4 ± 0.3** | **+1.4 ± 0.3** |
| Botswana | Africa | 1.4 | 0.1 | 0.1 | 0.1 (0.0) | 0.1 (0.0) | **+1.3 ± 0.2** | **+1.3 ± 0.2** |
| Venezuela | South America | 5.0 | 5.2 | 5.1 | 3.7 (3.8) | 3.6 (3.6) | **+1.3 ± 0.3** | **+1.4 ± 0.3** |

## US coup

### Regions by era

| region | era | humans | soup own games | r32 own games | soup same positions | r32 same positions | humans − soup, same positions | humans − r32, same positions | human points |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Europe | Early War | 1 | 3 | 3 | 1 | 1 | +0 ± 0 | -0 ± 0 | 566 |
| Europe | Mid War | 0 | 0 | 0 | 0 | 0 | -0 ± 0 | -0 ± 0 | 1,030 |
| Europe | Late War | 2 | 1 | 1 | 2 | 2 | +0 ± 0 | +0 ± 0 | 459 |
| Asia | Early War | 25 | 24 | 26 | 30 | 29 | **-5 ± 1** | **-5 ± 1** | 566 |
| Asia | Mid War | 3 | 3 | 3 | 3 | 3 | +0 ± 1 | +0 ± 1 | 1,030 |
| Asia | Late War | 1 | 1 | 1 | 1 | 1 | +1 ± 0 | +1 ± 0 | 459 |
| Middle East | Early War | 49 | 49 | 48 | 46 | 46 | **+3 ± 1** | **+3 ± 1** | 566 |
| Middle East | Mid War | 2 | 3 | 3 | 2 | 3 | +0 ± 1 | -0 ± 1 | 1,030 |
| Middle East | Late War | 4 | 3 | 4 | 3 | 3 | +1 ± 1 | +1 ± 1 | 459 |
| Africa | Early War | 12 | 4 | 4 | 9 | 9 | **+3 ± 1** | **+3 ± 1** | 566 |
| Africa | Mid War | 51 | 36 | 37 | 44 | 44 | **+7 ± 2** | **+7 ± 2** | 1,030 |
| Africa | Late War | 47 | 36 | 36 | 40 | 39 | **+7 ± 3** | **+8 ± 3** | 459 |
| Central America | Early War | 3 | 1 | 1 | 3 | 3 | -0 ± 1 | -0 ± 1 | 566 |
| Central America | Mid War | 20 | 22 | 22 | 24 | 25 | **-5 ± 2** | **-5 ± 1** | 1,030 |
| Central America | Late War | 29 | 27 | 27 | 29 | 30 | +1 ± 3 | -0 ± 3 | 459 |
| South America | Early War | 11 | 19 | 17 | 12 | 12 | -1 ± 1 | -1 ± 1 | 566 |
| South America | Mid War | 23 | 36 | 34 | 26 | 25 | -2 ± 2 | -1 ± 2 | 1,030 |
| South America | Late War | 17 | 32 | 32 | 26 | 26 | **-9 ± 3** | **-10 ± 3** | 459 |

### Countries, Early War, turns 1-3 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Egypt | Middle East | 12.4 | 5.8 | 5.8 | 8.5 (8.3) | 8.8 (8.5) | **+3.8 ± 0.9** | **+3.6 ± 0.8** |
| Iraq | Middle East | 8.1 | 5.5 | 5.7 | 5.3 (5.3) | 5.4 (5.1) | **+2.8 ± 0.8** | **+2.7 ± 0.7** |
| Angola | Africa | 5.1 | 0.1 | 0.0 | 2.3 (2.1) | 2.4 (1.9) | **+2.9 ± 0.7** | **+2.7 ± 0.7** |
| Lebanon | Middle East | 4.6 | 14.8 | 15.6 | 7.2 (7.1) | 7.5 (7.8) | **-2.6 ± 0.8** | **-2.9 ± 0.9** |
| North Korea | Asia | 0.2 | 2.6 | 2.2 | 2.1 (1.9) | 2.1 (2.3) | **-1.9 ± 0.6** | **-1.9 ± 0.6** |
| Thailand | Asia | 5.7 | 8.0 | 8.8 | 7.3 (7.1) | 7.3 (7.2) | **-1.6 ± 0.8** | **-1.7 ± 0.8** |
| Vietnam | Asia | 8.8 | 4.6 | 5.1 | 7.4 (7.4) | 7.2 (7.2) | +1.4 ± 0.7 | **+1.6 ± 0.7** |
| Libya | Middle East | 4.2 | 2.1 | 2.6 | 5.6 (5.8) | 5.6 (6.0) | -1.4 ± 0.7 | -1.4 ± 0.7 |
| Algeria | Africa | 2.5 | 0.8 | 0.7 | 1.1 (1.1) | 1.2 (1.1) | **+1.4 ± 0.5** | **+1.3 ± 0.5** |
| Syria | Middle East | 1.2 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | **+1.2 ± 0.5** | **+1.2 ± 0.5** |
| Laos/Cambodia | Asia | 4.8 | 6.0 | 6.3 | 6.3 (6.5) | 5.9 (6.0) | **-1.5 ± 0.6** | **-1.2 ± 0.5** |
| Nigeria | Africa | 1.2 | 0.5 | 0.6 | 2.3 (2.5) | 2.2 (2.7) | **-1.0 ± 0.5** | **-1.0 ± 0.5** |

### Countries, Mid and Late War, turns 4-10 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Colombia | South America | 10.1 | 28.9 | 27.9 | 16.3 (16.3) | 16.4 (16.6) | **-6.1 ± 1.2** | **-6.3 ± 1.1** |
| Zaire | Africa | 10.3 | 5.0 | 5.0 | 5.7 (5.5) | 5.5 (5.3) | **+4.6 ± 0.7** | **+4.8 ± 0.7** |
| Nicaragua | Central America | 8.5 | 6.4 | 6.6 | 11.2 (11.4) | 11.4 (11.8) | **-2.7 ± 0.8** | **-2.9 ± 0.8** |
| Guatemala | Central America | 4.0 | 9.7 | 10.0 | 6.1 (6.4) | 6.1 (6.6) | **-2.1 ± 0.7** | **-2.2 ± 0.7** |
| Argentina | South America | 3.4 | 0.9 | 0.9 | 1.8 (1.7) | 1.7 (1.7) | **+1.7 ± 0.5** | **+1.7 ± 0.5** |
| Cameroon | Africa | 8.9 | 8.6 | 8.4 | 7.5 (7.7) | 7.1 (7.1) | +1.4 ± 0.8 | **+1.8 ± 0.7** |
| Haiti | Central America | 6.2 | 4.1 | 3.9 | 4.6 (4.2) | 5.0 (4.4) | **+1.6 ± 0.6** | **+1.3 ± 0.6** |
| Venezuela | South America | 2.8 | 1.1 | 0.9 | 1.6 (1.4) | 1.6 (1.5) | **+1.2 ± 0.5** | **+1.2 ± 0.5** |
| Brazil | South America | 2.4 | 3.6 | 3.7 | 3.8 (4.0) | 3.3 (3.6) | **-1.4 ± 0.6** | -0.9 ± 0.5 |
| Angola | Africa | 7.4 | 5.0 | 5.3 | 6.2 (6.4) | 6.6 (6.8) | +1.2 ± 0.7 | +0.8 ± 0.7 |
| Nigeria | Africa | 6.1 | 5.8 | 5.8 | 6.8 (6.6) | 6.8 (6.7) | -0.7 ± 0.8 | -0.7 ± 0.8 |
| Panama | Central America | 2.1 | 1.4 | 1.5 | 1.4 (1.3) | 1.4 (1.5) | +0.7 ± 0.5 | +0.6 ± 0.5 |

## USSR coup

### Regions by era

| region | era | humans | soup own games | r32 own games | soup same positions | r32 same positions | humans − soup, same positions | humans − r32, same positions | human points |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Europe | Early War | 7 | 6 | 6 | 15 | 16 | **-8 ± 1** | **-9 ± 1** | 870 |
| Europe | Mid War | 0 | 0 | 0 | 0 | 0 | +0 ± 0 | +0 ± 0 | 1,045 |
| Europe | Late War | 0 | 0 | 0 | 0 | 0 | -0 ± 0 | -0 ± 0 | 496 |
| Asia | Early War | 17 | 13 | 15 | 14 | 14 | **+3 ± 1** | **+3 ± 1** | 870 |
| Asia | Mid War | 3 | 3 | 4 | 3 | 3 | +0 ± 1 | -0 ± 1 | 1,045 |
| Asia | Late War | 2 | 1 | 1 | 1 | 1 | +1 ± 1 | +1 ± 1 | 496 |
| Middle East | Early War | 45 | 48 | 47 | 48 | 46 | **-4 ± 2** | -1 ± 2 | 870 |
| Middle East | Mid War | 4 | 5 | 4 | 6 | 5 | **-2 ± 1** | -1 ± 1 | 1,045 |
| Middle East | Late War | 3 | 6 | 5 | 2 | 2 | +1 ± 1 | +1 ± 1 | 496 |
| Africa | Early War | 7 | 1 | 1 | 5 | 5 | **+2 ± 1** | **+2 ± 1** | 870 |
| Africa | Mid War | 45 | 34 | 35 | 36 | 37 | **+9 ± 2** | **+8 ± 2** | 1,045 |
| Africa | Late War | 53 | 40 | 40 | 51 | 52 | +2 ± 3 | +2 ± 3 | 496 |
| Central America | Early War | 17 | 12 | 12 | 11 | 13 | **+6 ± 1** | **+5 ± 1** | 870 |
| Central America | Mid War | 26 | 25 | 25 | 29 | 29 | -2 ± 2 | -3 ± 2 | 1,045 |
| Central America | Late War | 27 | 25 | 25 | 25 | 26 | +3 ± 2 | +2 ± 2 | 496 |
| South America | Early War | 7 | 21 | 19 | 6 | 7 | +0 ± 0 | +0 ± 0 | 870 |
| South America | Mid War | 22 | 33 | 32 | 26 | 26 | **-4 ± 1** | **-4 ± 1** | 1,045 |
| South America | Late War | 14 | 28 | 28 | 21 | 20 | **-7 ± 2** | **-6 ± 2** | 496 |

### Countries, Early War, turns 1-3 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Lebanon | Middle East | 3.9 | 13.7 | 14.8 | 12.6 (13.0) | 12.4 (12.6) | **-8.7 ± 1.3** | **-8.5 ± 1.2** |
| Italy | Europe | 6.6 | 5.6 | 6.0 | 15.0 (15.3) | 15.7 (16.2) | **-8.5 ± 0.9** | **-9.2 ± 0.9** |
| Panama | Central America | 16.8 | 11.4 | 11.3 | 10.9 (10.9) | 12.1 (11.7) | **+5.9 ± 1.1** | **+4.6 ± 1.1** |
| Iran | Middle East | 32.9 | 29.4 | 28.1 | 28.9 (28.7) | 26.4 (26.4) | **+4.0 ± 1.4** | **+6.4 ± 1.3** |
| Egypt | Middle East | 3.8 | 1.3 | 1.1 | 1.8 (1.7) | 1.6 (1.4) | **+2.0 ± 0.7** | **+2.2 ± 0.7** |
| Pakistan | Asia | 6.2 | 1.6 | 1.7 | 4.3 (4.1) | 4.1 (4.1) | **+1.9 ± 0.8** | **+2.1 ± 0.7** |
| Angola | Africa | 2.4 | 0.0 | 0.0 | 0.9 (0.9) | 1.0 (1.0) | **+1.5 ± 0.4** | **+1.4 ± 0.4** |
| Malaysia | Asia | 2.9 | 1.0 | 1.3 | 1.2 (1.3) | 1.6 (1.7) | **+1.6 ± 0.5** | **+1.3 ± 0.5** |
| Libya | Middle East | 3.8 | 3.2 | 3.1 | 5.1 (4.9) | 5.0 (4.5) | -1.3 ± 0.8 | -1.2 ± 0.8 |
| South Africa | Africa | 1.3 | 0.1 | 0.1 | 0.2 (0.2) | 0.2 (0.2) | **+1.0 ± 0.3** | **+1.0 ± 0.3** |
| Zaire | Africa | 2.1 | 0.1 | 0.1 | 2.5 (2.6) | 2.5 (2.9) | -0.4 ± 0.3 | -0.4 ± 0.3 |
| Thailand | Asia | 3.2 | 5.4 | 6.2 | 3.6 (3.6) | 3.6 (3.6) | -0.4 ± 0.4 | -0.4 ± 0.4 |

### Countries, Mid and Late War, turns 4-10 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Colombia | South America | 9.3 | 20.7 | 19.3 | 14.1 (14.1) | 14.3 (14.5) | **-4.9 ± 1.0** | **-5.0 ± 1.0** |
| Angola | Africa | 7.5 | 3.7 | 3.8 | 3.0 (2.9) | 3.5 (3.5) | **+4.5 ± 0.6** | **+3.9 ± 0.5** |
| Guatemala | Central America | 4.9 | 8.4 | 8.9 | 7.6 (7.6) | 7.7 (7.9) | **-2.7 ± 0.7** | **-2.8 ± 0.7** |
| Haiti | Central America | 4.7 | 2.6 | 2.3 | 2.5 (2.5) | 2.7 (2.5) | **+2.2 ± 0.5** | **+2.0 ± 0.5** |
| Mexico | Central America | 3.3 | 5.8 | 6.1 | 5.1 (4.9) | 5.1 (4.9) | **-1.8 ± 0.5** | **-1.8 ± 0.5** |
| Brazil | South America | 3.2 | 6.0 | 6.5 | 5.1 (5.3) | 4.7 (5.3) | **-1.9 ± 0.5** | **-1.6 ± 0.5** |
| Nigeria | Africa | 7.2 | 8.8 | 9.3 | 8.7 (8.4) | 9.1 (9.7) | **-1.5 ± 0.7** | **-1.9 ± 0.6** |
| Lebanon | Middle East | 0.3 | 1.5 | 1.4 | 1.9 (1.9) | 1.6 (1.7) | **-1.6 ± 0.4** | **-1.3 ± 0.3** |
| Saharan States | Africa | 7.4 | 5.4 | 5.5 | 6.0 (5.6) | 6.1 (6.0) | **+1.4 ± 0.6** | **+1.3 ± 0.6** |
| Nicaragua | Central America | 7.9 | 2.8 | 2.9 | 6.6 (6.7) | 6.2 (6.2) | **+1.3 ± 0.6** | **+1.6 ± 0.6** |
| Cameroon | Africa | 8.1 | 4.9 | 4.9 | 7.3 (7.5) | 6.8 (6.9) | +0.8 ± 0.6 | **+1.3 ± 0.5** |
| Botswana | Africa | 1.1 | 0.0 | 0.0 | 0.3 (0.3) | 0.3 (0.3) | **+0.8 ± 0.3** | **+0.8 ± 0.3** |

## US realign

### Regions by era

| region | era | humans | soup own games | r32 own games | soup same positions | r32 same positions | humans − soup, same positions | humans − r32, same positions | human points |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Europe | Early War | 6 | 0 | 0 | 7 | 7 | -1 ± 4 | -1 ± 4 | 112 |
| Europe | Mid War | 2 | 0 | 0 | 1 | 2 | +0 ± 0 | +0 ± 0 | 612 |
| Europe | Late War | 5 | 15 | 15 | 5 | 5 | +0 ± 0 | +0 ± 0 | 315 |
| Asia | Early War | 4 | 0 | 0 | 6 | 6 | -3 ± 3 | -3 ± 2 | 112 |
| Asia | Mid War | 1 | 0 | 0 | 1 | 1 | +0 ± 0 | +0 ± 0 | 612 |
| Asia | Late War | 1 | 0 | 0 | 1 | 1 | +0 ± 0 | +0 ± 0 | 315 |
| Middle East | Early War | 5 | 0 | 0 | 1 | 1 | +4 ± 3 | +4 ± 3 | 112 |
| Middle East | Mid War | 0 | 0 | 0 | 0 | 0 | -0 ± 0 | **-0 ± 0** | 612 |
| Middle East | Late War | 1 | 1 | 0 | 1 | 1 | +1 ± 1 | +1 ± 1 | 315 |
| Africa | Early War | 23 | 4 | 4 | 15 | 15 | +8 ± 6 | +8 ± 5 | 112 |
| Africa | Mid War | 23 | 12 | 13 | 22 | 22 | +2 ± 3 | +2 ± 2 | 612 |
| Africa | Late War | 30 | 21 | 22 | 27 | 26 | +3 ± 2 | +4 ± 2 | 315 |
| Central America | Early War | 43 | 80 | 82 | 55 | 52 | **-12 ± 5** | -9 ± 5 | 112 |
| Central America | Mid War | 52 | 69 | 69 | 55 | 54 | -3 ± 3 | -3 ± 2 | 612 |
| Central America | Late War | 43 | 47 | 43 | 49 | 48 | -6 ± 3 | -5 ± 3 | 315 |
| South America | Early War | 19 | 15 | 14 | 14 | 17 | +4 ± 5 | +1 ± 5 | 112 |
| South America | Mid War | 22 | 18 | 18 | 20 | 21 | +2 ± 2 | +1 ± 2 | 612 |
| South America | Late War | 19 | 17 | 19 | 17 | 19 | +2 ± 3 | -1 ± 3 | 315 |

### Countries, Early War, turns 1-3 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Cuba | Central America | 31.2 | 48.0 | 49.2 | 44.9 (44.6) | 42.9 (43.8) | **-13.6 ± 5.3** | **-11.6 ± 5.1** |
| Angola | Africa | 7.1 | 0.0 | 0.1 | 0.0 (0.0) | 0.2 (0.0) | **+7.1 ± 3.3** | **+7.0 ± 3.2** |
| Algeria | Africa | 6.2 | 4.0 | 3.7 | 11.3 (11.6) | 10.0 (10.7) | -5.1 ± 4.9 | -3.8 ± 3.7 |
| Argentina | South America | 5.4 | 1.9 | 1.4 | 0.4 (0.0) | 1.9 (0.9) | +4.9 ± 2.6 | +3.4 ± 2.4 |
| Chile | South America | 5.4 | 0.0 | 0.1 | 1.3 (0.9) | 2.0 (1.8) | **+4.1 ± 1.8** | +3.3 ± 1.7 |
| East Germany | Europe | 0.0 | 0.0 | 0.0 | 3.5 (3.6) | 3.3 (3.6) | -3.5 ± 2.5 | -3.3 ± 2.3 |
| Zaire | Africa | 7.1 | 0.1 | 0.2 | 3.9 (4.5) | 4.2 (4.5) | +3.3 ± 2.9 | +2.9 ± 3.3 |
| Iraq | Middle East | 3.6 | 0.0 | 0.0 | 0.9 (0.9) | 1.0 (0.9) | +2.7 ± 2.0 | +2.6 ± 1.9 |
| Italy | Europe | 4.5 | 0.0 | 0.0 | 1.8 (1.8) | 1.9 (0.9) | +2.7 ± 2.5 | +2.6 ± 2.4 |
| Thailand | Asia | 3.6 | 0.0 | 0.0 | 6.3 (6.2) | 6.1 (7.1) | -2.7 ± 2.5 | -2.6 ± 2.4 |
| Venezuela | South America | 8.0 | 10.4 | 9.6 | 10.9 (10.7) | 10.4 (10.7) | -2.9 ± 3.2 | -2.4 ± 3.5 |
| Mexico | Central America | 8.0 | 29.3 | 29.6 | 5.9 (6.2) | 6.0 (7.1) | +2.1 ± 1.4 | +2.1 ± 1.3 |

### Countries, Mid and Late War, turns 4-10 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Cuba | Central America | 25.8 | 28.9 | 27.1 | 30.4 (30.0) | 29.7 (30.4) | **-4.6 ± 1.8** | **-3.9 ± 1.7** |
| Panama | Central America | 7.0 | 4.6 | 5.4 | 4.5 (4.5) | 4.8 (4.6) | **+2.5 ± 1.1** | +2.2 ± 1.1 |
| Algeria | Africa | 5.4 | 6.1 | 5.7 | 7.6 (7.7) | 7.4 (7.2) | -2.2 ± 1.2 | -2.0 ± 1.2 |
| Zaire | Africa | 4.2 | 3.1 | 3.6 | 2.5 (2.4) | 2.6 (2.4) | **+1.7 ± 0.8** | **+1.6 ± 0.8** |
| Venezuela | South America | 6.7 | 8.4 | 8.5 | 5.3 (5.1) | 5.1 (5.4) | **+1.4 ± 0.7** | **+1.6 ± 0.7** |
| Mexico | Central America | 16.1 | 26.7 | 26.2 | 17.8 (18.0) | 17.4 (18.0) | -1.8 ± 1.2 | -1.4 ± 1.0 |
| Angola | Africa | 7.2 | 1.0 | 1.1 | 6.4 (6.6) | 6.3 (6.5) | +0.9 ± 0.7 | +0.9 ± 0.8 |
| South Africa | Africa | 2.2 | 0.7 | 1.0 | 1.8 (1.8) | 1.8 (1.7) | +0.4 ± 0.4 | +0.3 ± 0.4 |
| France | Europe | 2.4 | 0.6 | 0.6 | 2.1 (2.0) | 2.1 (2.2) | +0.3 ± 0.2 | +0.3 ± 0.2 |
| Chile | South America | 8.2 | 3.2 | 3.2 | 8.5 (8.6) | 9.0 (9.1) | -0.3 ± 1.0 | -0.8 ± 1.0 |
| Nigeria | Africa | 4.4 | 3.7 | 3.6 | 4.2 (4.3) | 4.0 (4.2) | +0.2 ± 0.6 | +0.4 ± 0.6 |
| Argentina | South America | 2.5 | 2.1 | 2.2 | 2.3 (2.3) | 2.7 (2.4) | +0.2 ± 0.6 | -0.2 ± 0.5 |

## USSR realign

### Regions by era

| region | era | humans | soup own games | r32 own games | soup same positions | r32 same positions | humans − soup, same positions | humans − r32, same positions | human points |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Europe | Early War | 30 | 0 | 0 | 32 | 33 | -2 ± 2 | -2 ± 2 | 46 |
| Europe | Mid War | 3 | 0 | 0 | 3 | 3 | +0 ± 1 | -0 ± 1 | 409 |
| Europe | Late War | 2 | 0 | 0 | 2 | 2 | +0 ± 0 | +0 ± 0 | 235 |
| Asia | Early War | 13 | 0 | 0 | 10 | 10 | +3 ± 2 | +3 ± 2 | 46 |
| Asia | Mid War | 5 | 1 | 1 | 4 | 4 | +1 ± 1 | +1 ± 1 | 409 |
| Asia | Late War | 3 | 0 | 1 | 3 | 2 | +0 ± 0 | +1 ± 1 | 235 |
| Middle East | Early War | 17 | 0 | 0 | 14 | 13 | +4 ± 5 | +4 ± 5 | 46 |
| Middle East | Mid War | 1 | 0 | 0 | 1 | 1 | +1 ± 1 | +0 ± 0 | 409 |
| Middle East | Late War | 2 | 0 | 0 | 3 | 3 | -1 ± 1 | -1 ± 1 | 235 |
| Africa | Early War | 15 | 2 | 2 | 2 | 3 | +13 ± 9 | +13 ± 9 | 46 |
| Africa | Mid War | 39 | 16 | 15 | 31 | 31 | **+7 ± 3** | **+7 ± 3** | 409 |
| Africa | Late War | 40 | 33 | 31 | 43 | 42 | -3 ± 5 | -2 ± 5 | 235 |
| Central America | Early War | 4 | 26 | 31 | 18 | 19 | -14 ± 7 | **-15 ± 7** | 46 |
| Central America | Mid War | 10 | 25 | 26 | 14 | 15 | **-5 ± 2** | **-5 ± 2** | 409 |
| Central America | Late War | 10 | 21 | 23 | 10 | 11 | -0 ± 4 | -1 ± 3 | 235 |
| South America | Early War | 20 | 72 | 67 | 21 | 21 | -2 ± 10 | -2 ± 10 | 46 |
| South America | Mid War | 43 | 58 | 58 | 46 | 45 | -3 ± 3 | -2 ± 3 | 409 |
| South America | Late War | 43 | 45 | 45 | 39 | 40 | +4 ± 6 | +3 ± 5 | 235 |

### Countries, Early War, turns 1-3 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Panama | Central America | 4.3 | 23.8 | 28.4 | 15.6 (15.2) | 17.0 (17.4) | -11.2 ± 6.6 | **-12.7 ± 6.3** |
| Angola | Africa | 8.7 | 0.4 | 0.0 | 0.0 (0.0) | 0.4 (0.0) | +8.7 ± 6.4 | +8.3 ± 6.4 |
| Brazil | South America | 17.4 | 3.1 | 0.4 | 10.6 (10.9) | 10.7 (10.9) | +6.8 ± 4.4 | +6.7 ± 4.8 |
| Argentina | South America | 0.0 | 0.0 | 0.0 | 5.4 (6.5) | 5.8 (6.5) | -5.4 ± 5.2 | -5.8 ± 5.5 |
| South Africa | Africa | 6.5 | 0.4 | 0.0 | 2.0 (2.2) | 1.9 (2.2) | +4.5 ± 2.9 | +4.7 ± 2.9 |
| Israel | Middle East | 4.3 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | +4.3 ± 4.5 | +4.3 ± 4.5 |
| West Germany | Europe | 30.4 | 0.0 | 0.0 | 22.6 (26.1) | 26.3 (28.3) | +7.9 ± 4.7 | +4.1 ± 3.4 |
| United Kingdom | Europe | 0.0 | 0.0 | 0.0 | 3.6 (4.3) | 4.2 (4.3) | -3.6 ± 3.5 | -4.2 ± 3.0 |
| Venezuela | South America | 2.2 | 69.3 | 66.7 | 5.4 (4.3) | 4.6 (4.3) | -3.2 ± 3.1 | -2.4 ± 2.3 |
| Mexico | Central America | 0.0 | 2.3 | 1.9 | 2.4 (2.2) | 2.4 (2.2) | -2.4 ± 2.4 | -2.4 ± 2.4 |
| Laos/Cambodia | Asia | 0.0 | 0.0 | 0.0 | 2.2 (2.2) | 2.3 (2.2) | -2.2 ± 2.1 | -2.3 ± 2.3 |
| Afghanistan | Asia | 2.2 | 0.0 | 0.0 | 0.0 (0.0) | 0.0 (0.0) | +2.2 ± 2.1 | +2.2 ± 2.1 |

### Countries, Mid and Late War, turns 4-10 (the 12 largest same-position gaps; countries with 1%+ somewhere)

| country | region | humans | soup own games | r32 own games | soup same positions (pick) | r32 same positions (pick) | humans − soup, same positions | humans − r32, same positions |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Algeria | Africa | 4.5 | 4.5 | 4.3 | 7.5 (7.5) | 7.5 (7.1) | -3.0 ± 1.7 | -3.0 ± 1.6 |
| Chile | South America | 6.1 | 2.7 | 2.9 | 3.1 (3.0) | 3.3 (3.1) | **+2.9 ± 1.1** | **+2.7 ± 1.1** |
| Zaire | Africa | 10.2 | 4.7 | 4.7 | 7.2 (7.5) | 7.6 (7.6) | **+3.0 ± 1.2** | **+2.7 ± 1.2** |
| South Africa | Africa | 11.3 | 5.1 | 5.1 | 8.4 (8.1) | 8.9 (9.3) | +2.9 ± 1.5 | +2.4 ± 1.4 |
| Panama | Central America | 6.8 | 16.2 | 17.2 | 9.0 (8.7) | 9.3 (8.9) | -2.2 ± 1.7 | -2.5 ± 1.6 |
| Brazil | South America | 10.6 | 6.9 | 7.3 | 11.6 (11.6) | 11.7 (11.5) | -1.0 ± 1.3 | -1.2 ± 1.2 |
| Mexico | Central America | 0.9 | 2.0 | 2.1 | 1.8 (1.6) | 1.7 (1.4) | -0.9 ± 0.7 | -0.8 ± 0.5 |
| Venezuela | South America | 12.3 | 34.5 | 32.9 | 13.7 (14.0) | 13.0 (13.5) | -1.4 ± 1.4 | -0.7 ± 1.2 |
| Costa Rica | Central America | 0.5 | 1.2 | 1.0 | 1.1 (1.1) | 1.1 (1.2) | -0.6 ± 0.7 | -0.6 ± 0.6 |
| Argentina | South America | 13.0 | 7.4 | 8.2 | 13.5 (14.1) | 13.6 (13.8) | -0.5 ± 1.8 | -0.5 ± 1.6 |
| Nigeria | Africa | 7.9 | 6.8 | 5.9 | 7.5 (7.3) | 6.9 (6.5) | +0.5 ± 1.5 | +1.1 ± 1.3 |
| Cuba | Central America | 1.1 | 3.2 | 3.7 | 0.7 (0.8) | 0.7 (0.8) | +0.4 ± 0.5 | +0.4 ± 0.5 |

## How often the bot picks the human's target

| side | mode | era | human targets | soup: pick = human's | r32: pick = human's | soup: p(human's) | r32: p(human's) |
|:---|:---|:---|---:|---:|---:|---:|---:|
| US | influence | Early War | 7,447 | 28% ± 1 | 29% ± 1 | 28% | 27% |
| US | influence | Mid War | 5,571 | 33% ± 1 | 35% ± 1 | 32% | 32% |
| US | influence | Late War | 2,641 | 33% ± 2 | 35% ± 2 | 31% | 31% |
| US | coup | Early War | 566 | 74% ± 2 | 73% ± 2 | 74% | 74% |
| US | coup | Mid War | 1,030 | 55% ± 2 | 55% ± 2 | 55% | 54% |
| US | coup | Late War | 459 | 49% ± 3 | 47% ± 3 | 48% | 47% |
| US | realign | Early War | 112 | 56% ± 6 | 59% ± 6 | 56% | 55% |
| US | realign | Mid War | 612 | 72% ± 3 | 72% ± 3 | 71% | 69% |
| US | realign | Late War | 315 | 70% ± 3 | 71% ± 3 | 70% | 68% |
| USSR | influence | Early War | 6,032 | 31% ± 1 | 32% ± 1 | 30% | 30% |
| USSR | influence | Mid War | 6,024 | 33% ± 1 | 34% ± 1 | 32% | 32% |
| USSR | influence | Late War | 2,365 | 33% ± 1 | 35% ± 1 | 31% | 31% |
| USSR | coup | Early War | 870 | 63% ± 2 | 62% ± 2 | 62% | 61% |
| USSR | coup | Mid War | 1,045 | 55% ± 2 | 54% ± 2 | 54% | 54% |
| USSR | coup | Late War | 496 | 52% ± 3 | 52% ± 2 | 51% | 51% |
| USSR | realign | Early War | 46 | 65% ± 9 | 67% ± 10 | 60% | 64% |
| USSR | realign | Mid War | 409 | 64% ± 3 | 66% ± 3 | 63% | 63% |
| USSR | realign | Late War | 235 | 57% ± 5 | 57% ± 5 | 56% | 55% |

