# Event census of the shallow soup (2026-10-03)

Owner: for the shallow_E7-02+03+04+05_1200M soup, how often each card in its owner's hand is headlined or played as the event in a round (neutral cards: overall / by US / by USSR), sorted; the opponent's events and chained plays (Grain Sales, Star Wars) excluded; only holdings where playing the event would be legal.

`tools/scripts/event_play_census.py` (8 × 512 greedy self-play games). Legality is the engine's: at each of the owner's action-round card choices each held card is selected on a copy and the play-mode mask read; a headline or a scoring card counts as legal. The engine offers EVENT for some cards whose event would do nothing (One Small Step when not behind in space, Wargames above DEFCON 2), so those still count; NATO's prerequisite is enforced (4,240 holdings -> 1,923).

**Unplayed scoring cards** (8.3% of 5,912 scoring holdings in a 512-game trace): 2.4% discarded by the US with Ask Not What Your Country Can Do For You; 2.7% in hand when the game ended (mostly 20 VP); 0.05% lost to holding a scoring card; ~0.8% taken or discarded by opponent events (Grain Sales, Aldrich Ames, Five Year Plan, Terrorism); ~2.1% discarded at the start of a turn just after the headline choices -- not yet explained.

How often a card in its owner's hand is used for its event -- headlined, or played in an action round as the event -- 4,096 greedy self-play games. One count per holding (the card entering the hand until it leaves), counting only holdings in which the event was legal for the owner at some point (checked at each of the owner's action-round card choices; a headline or a scoring card counts as legal). US/USSR cards: the owner's holdings only. Neutral cards: whoever holds it, with the split by side.

| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |
|:---|:---|---:|---:|---:|---:|---:|
| Defectors | us | **100%** (3,697) | 100% | 0% |  |  |
| The Reformer | ussr | **99%** (1,018) | 73% | 26% |  |  |
| The Voice of America | us | **99%** (2,578) | 34% | 65% |  |  |
| Junta | neutral | **98%** (5,302) | 70% | 28% | 99% (2,752) | 98% (2,550) |
| Pershing II Deployed | ussr | **98%** (1,007) | 50% | 49% |  |  |
| Brush War | neutral | **98%** (5,359) | 32% | 66% | 99% (2,787) | 98% (2,572) |
| UN Intervention | neutral | **98%** (8,458) | 0% | 98% | 98% (4,590) | 98% (3,868) |
| Decolonization | ussr | **97%** (4,096) | 33% | 64% |  |  |
| Colonial Rear Guards | us | **97%** (2,602) | 27% | 70% |  |  |
| Asia Scoring | neutral | **97%** (8,970) | 22% | 75% | 97% (4,755) | 97% (4,215) |
| Liberation Theology | ussr | **96%** (2,515) | 46% | 50% |  |  |
| Europe Scoring | neutral | **95%** (8,832) | 16% | 79% | 95% (4,620) | 95% (4,212) |
| Middle East Scoring | neutral | **95%** (8,766) | 19% | 76% | 95% (4,700) | 94% (4,066) |
| Southeast Asia Scoring | neutral | **94%** (3,645) | 10% | 84% | 92% (1,695) | 97% (1,950) |
| Central America Scoring | neutral | **94%** (5,196) | 8% | 86% | 94% (2,585) | 95% (2,611) |
| Africa Scoring | neutral | **94%** (5,178) | 13% | 81% | 94% (2,547) | 94% (2,631) |
| South America Scoring | neutral | **94%** (5,188) | 9% | 85% | 92% (2,635) | 95% (2,553) |
| OAS Founded | us | **93%** (1,756) | 15% | 78% |  |  |
| Tear Down this Wall | us | **93%** (1,019) | 41% | 52% |  |  |
| Quagmire | ussr | **91%** (2,134) | 71% | 20% |  |  |
| Containment | us | **91%** (1,880) | 91% | 0% |  |  |
| ABM Treaty | neutral | **89%** (5,385) | 60% | 30% | 95% (2,858) | 82% (2,527) |
| Marine Barracks Bombing | ussr | **87%** (1,054) | 21% | 66% |  |  |
| Grain Sales to Soviets | us | **86%** (2,627) | 86% | 0% |  |  |
| De-Stalinization | ussr | **86%** (2,934) | 12% | 74% |  |  |
| Captured Nazi Scientist | neutral | **85%** (4,657) | 33% | 52% | 83% (2,083) | 86% (2,574) |
| Allende | ussr | **84%** (1,962) | 27% | 57% |  |  |
| Panama Canal Returned | us | **83%** (1,770) | 14% | 69% |  |  |
| Sadat Expels Soviets | us | **79%** (1,781) | 10% | 70% |  |  |
| Red Scare/Purge | neutral | **77%** (9,079) | 77% | 0% | 69% (4,916) | 86% (4,163) |
| Brezhnev Doctrine | ussr | **75%** (2,011) | 74% | 0% |  |  |
| Aldrich Ames Remix | ussr | **74%** (1,034) | 74% | 0% |  |  |
| Indo-Pakistani War | neutral | **73%** (8,678) | 25% | 48% | 69% (4,621) | 77% (4,057) |
| Nixon Plays the China Card | us | **71%** (2,025) | 7% | 64% |  |  |
| “Ask Not What Your Country Can Do For You…” | us | **70%** (2,083) | 37% | 33% |  |  |
| Camp David Accords | us | **68%** (2,134) | 10% | 58% |  |  |
| Iran-Iraq War | neutral | **68%** (2,029) | 17% | 51% | 85% (1,015) | 51% (1,014) |
| Solidarity | us | **68%** (509) | 7% | 61% |  |  |
| Puppet Governments | us | **67%** (2,074) | 20% | 47% |  |  |
| South African Unrest | ussr | **65%** (2,472) | 26% | 38% |  |  |
| Korean War | ussr | **62%** (3,034) | 34% | 28% |  |  |
| Nasser | ussr | **61%** (2,664) | 40% | 21% |  |  |
| Socialist Governments | ussr | **61%** (3,864) | 37% | 24% |  |  |
| Missile Envy | neutral | **59%** (5,188) | 59% | 0% | 74% (2,690) | 42% (2,498) |
| Vietnam Revolts | ussr | **56%** (2,730) | 54% | 2% |  |  |
| Soviets Shoot Down KAL-007 | us | **52%** (1,017) | 52% | 0% |  |  |
| Ussuri River Skirmish | us | **50%** (2,195) | 16% | 34% |  |  |
| Fidel | ussr | **48%** (3,061) | 11% | 37% |  |  |
| De Gaulle Leads France | ussr | **43%** (3,107) | 36% | 7% |  |  |
| Suez Crisis | ussr | **38%** (3,103) | 26% | 13% |  |  |
| Cultural Revolution | ussr | **32%** (2,152) | 15% | 17% |  |  |
| CIA Created | us | **30%** (2,598) | 30% | 0% |  |  |
| OPEC | ussr | **27%** (2,419) | 17% | 11% |  |  |
| Blockade | ussr | **25%** (3,017) | 1% | 24% |  |  |
| Truman Doctrine | us | **25%** (2,477) | 13% | 12% |  |  |
| Star Wars | us | **25%** (586) | 25% | 0% |  |  |
| Portuguese Empire Crumbles | ussr | **24%** (2,271) | 12% | 12% |  |  |
| Bear Trap | us | **24%** (2,026) | 24% | 0% |  |  |
| “Lone Gunman” | ussr | **22%** (2,357) | 21% | 0% |  |  |
| Latin American Debt Crisis | ussr | **21%** (1,031) | 7% | 14% |  |  |
| Duck and Cover | us | **20%** (4,581) | 20% | 0% |  |  |
| Alliance for Progress | us | **20%** (2,290) | 12% | 8% |  |  |
| SALT Negotiations | neutral | **17%** (4,883) | 17% | 0% | 25% (2,507) | 8% (2,376) |
| Willy Brandt | ussr | **17%** (2,180) | 8% | 8% |  |  |
| Muslim Revolution | ussr | **16%** (2,268) | 16% | 1% |  |  |
| NORAD | us | **15%** (3,018) | 15% | 0% |  |  |
| Marshall Plan | us | **13%** (2,978) | 13% | 0% |  |  |
| “We Will Bury You” | ussr | **13%** (2,340) | 13% | 0% |  |  |
| Glasnost | ussr | **10%** (1,000) | 10% | 0% |  |  |
| Che | ussr | **10%** (2,450) | 9% | 1% |  |  |
| Terrorism | neutral | **10%** (2,007) | 9% | 1% | 5% (980) | 14% (1,027) |
| John Paul II Elected Pope | us | **7%** (2,206) | 5% | 2% |  |  |
| Kitchen Debates | us | **7%** (2,362) | 4% | 3% |  |  |
| The Cambridge Five | ussr | **7%** (3,490) | 6% | 0% |  |  |
| Arab-Israeli War | ussr | **6%** (3,451) | 5% | 2% |  |  |
| US/Japan Mutual Defense Pact | us | **6%** (3,095) | 6% | 0% |  |  |
| Olympic Games | neutral | **5%** (8,751) | 5% | 0% | 9% (4,673) | 0% (4,078) |
| East European Unrest | us | **4%** (4,568) | 4% | 0% |  |  |
| Iranian Hostage Crisis | ussr | **4%** (1,010) | 4% | 0% |  |  |
| How I Learned to Stop Worrying | neutral | **4%** (5,023) | 4% | 0% | 6% (2,488) | 2% (2,535) |
| “An Evil Empire” | us | **3%** (981) | 3% | 0% |  |  |
| Romanian Abdication | ussr | **3%** (3,109) | 3% | 0% |  |  |
| Our Man in Tehran | us | **3%** (2,261) | 3% | 0% |  |  |
| Cuban Missile Crisis | neutral | **3%** (5,070) | 2% | 0% | 1% (2,610) | 5% (2,460) |
| Special Relationship | us | **2%** (4,718) | 2% | 0% |  |  |
| Flower Power | ussr | **2%** (1,939) | 2% | 0% |  |  |
| Yuri and Samantha | ussr | **2%** (986) | 2% | 0% |  |  |
| AWACS Sale to Saudis | us | **2%** (992) | 1% | 0% |  |  |
| Five Year Plan | us | **1%** (4,458) | 1% | 0% |  |  |
| Iran-Contra Scandal | ussr | **1%** (1,012) | 0% | 0% |  |  |
| Reagan Bombs Libya | us | **1%** (1,008) | 1% | 0% |  |  |
| Independent Reds | us | **1%** (2,856) | 1% | 0% |  |  |
| Shuttle Diplomacy | us | **1%** (2,289) | 1% | 0% |  |  |
| Nuclear Test Ban | neutral | **0%** (8,634) | 0% | 0% | 1% (4,617) | 0% (4,017) |
| U-2 Incident | ussr | **0%** (2,169) | 0% | 0% |  |  |
| Latin American Death Squads | neutral | **0%** (5,104) | 0% | 0% | 1% (2,611) | 0% (2,493) |
| Ortega Elected in Nicaragua | ussr | **0%** (1,013) | 0% | 0% |  |  |
| The Iron Lady | us | **0%** (942) | 0% | 0% |  |  |
| Wargames | neutral | **0%** (1,841) | 0% | 0% | 0% (881) | 0% (960) |
| Arms Race | neutral | **0%** (4,951) | 0% | 0% | 0% (2,537) | 0% (2,414) |
| Warsaw Pact Formed | ussr | **0%** (3,164) | 0% | 0% |  |  |
| “One Small Step…” | neutral | **0%** (5,010) | 0% | 0% | 0% (2,606) | 0% (2,404) |
| Chernobyl | us | **0%** (972) | 0% | 0% |  |  |
| Summit | neutral | **0%** (5,090) | 0% | 0% | 0% (2,553) | 0% (2,537) |
| Formosan Resolution | us | **0%** (2,950) | 0% | 0% |  |  |
| Comecon | ussr | **0%** (3,149) | 0% | 0% |  |  |
| Nuclear Subs | us | **0%** (2,144) | 0% | 0% |  |  |
| NATO | us | **0%** (1,923) | 0% | 0% |  |  |
| North Sea Oil | us | **0%** (970) | 0% | 0% |  |  |
