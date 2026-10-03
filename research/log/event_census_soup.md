# Event census of the shallow soup (2026-10-03)

Owner: "for the shallow_E7-02+03+04+05_1200M soup, create a table for each card: how often it is played as event when in hand (neutral cards: general / by USSR / by US), sorted by frequency" -- then: count only the owner headlining it or playing it as the event in a round; ignore the opponent's events and chained plays (Grain Sales, Star Wars).

`tools/scripts/event_play_census.py` (8 × 512 greedy self-play games). A holding = the card entering the owner's hand until it leaves. Caveats: the denominator includes holdings where the event was not legal (NATO before Marshall Plan / Warsaw Pact, Wargames above DEFCON 2, One Small Step when not behind in space); scoring cards fall short of 100% where the card left the hand otherwise (game over, discards).

How often a card in its owner's hand is used for its event -- headlined, or played in an action round as the event -- 4,096 greedy self-play games. One count per holding (the card entering the hand until it leaves). US/USSR cards: the owner's holdings only. Neutral cards: whoever holds it, with the split by side.

| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |
|:---|:---|---:|---:|---:|---:|---:|
| Junta | neutral | **97%** (5,378) | 69% | 28% | 97% (2,802) | 97% (2,576) |
| The Reformer | ussr | **97%** (1,038) | 71% | 26% |  |  |
| Pershing II Deployed | ussr | **97%** (1,024) | 49% | 48% |  |  |
| Decolonization | ussr | **97%** (4,128) | 33% | 64% |  |  |
| Brush War | neutral | **96%** (5,505) | 32% | 64% | 96% (2,878) | 96% (2,627) |
| UN Intervention | neutral | **95%** (8,714) | 0% | 95% | 94% (4,776) | 96% (3,938) |
| Asia Scoring | neutral | **95%** (9,138) | 21% | 74% | 95% (4,869) | 95% (4,269) |
| Liberation Theology | ussr | **95%** (2,557) | 45% | 50% |  |  |
| The Voice of America | us | **94%** (2,696) | 32% | 62% |  |  |
| Colonial Rear Guards | us | **94%** (2,697) | 26% | 67% |  |  |
| Middle East Scoring | neutral | **93%** (8,929) | 19% | 75% | 93% (4,820) | 93% (4,109) |
| Europe Scoring | neutral | **93%** (9,024) | 16% | 78% | 92% (4,768) | 94% (4,256) |
| Containment | us | **90%** (1,882) | 90% | 0% |  |  |
| Africa Scoring | neutral | **90%** (5,433) | 12% | 77% | 88% (2,722) | 91% (2,711) |
| Central America Scoring | neutral | **90%** (5,475) | 8% | 82% | 87% (2,780) | 92% (2,695) |
| Southeast Asia Scoring | neutral | **90%** (3,843) | 9% | 80% | 84% (1,847) | 94% (1,996) |
| Tear Down this Wall | us | **90%** (1,054) | 40% | 50% |  |  |
| Quagmire | ussr | **89%** (2,169) | 70% | 20% |  |  |
| South America Scoring | neutral | **89%** (5,453) | 8% | 81% | 86% (2,832) | 93% (2,621) |
| OAS Founded | us | **89%** (1,839) | 14% | 75% |  |  |
| ABM Treaty | neutral | **86%** (5,553) | 58% | 29% | 92% (2,953) | 80% (2,600) |
| Grain Sales to Soviets | us | **85%** (2,641) | 85% | 0% |  |  |
| De-Stalinization | ussr | **85%** (2,952) | 12% | 73% |  |  |
| Marine Barracks Bombing | ussr | **85%** (1,083) | 20% | 64% |  |  |
| Captured Nazi Scientist | neutral | **84%** (4,687) | 33% | 52% | 83% (2,106) | 86% (2,581) |
| Allende | ussr | **82%** (2,003) | 26% | 56% |  |  |
| Defectors | us | **80%** (4,645) | 80% | 0% |  |  |
| Panama Canal Returned | us | **78%** (1,892) | 13% | 65% |  |  |
| Red Scare/Purge | neutral | **76%** (9,123) | 76% | 0% | 68% (4,952) | 86% (4,171) |
| Sadat Expels Soviets | us | **75%** (1,883) | 9% | 66% |  |  |
| Brezhnev Doctrine | ussr | **74%** (2,029) | 73% | 0% |  |  |
| Aldrich Ames Remix | ussr | **73%** (1,047) | 73% | 0% |  |  |
| Indo-Pakistani War | neutral | **72%** (8,861) | 25% | 47% | 67% (4,758) | 76% (4,103) |
| Nixon Plays the China Card | us | **68%** (2,116) | 6% | 61% |  |  |
| “Ask Not What Your Country Can Do For You…” | us | **67%** (2,153) | 35% | 32% |  |  |
| Iran-Iraq War | neutral | **65%** (2,111) | 17% | 49% | 81% (1,061) | 49% (1,050) |
| Camp David Accords | us | **65%** (2,225) | 9% | 56% |  |  |
| Puppet Governments | us | **64%** (2,155) | 19% | 45% |  |  |
| South African Unrest | ussr | **63%** (2,546) | 26% | 37% |  |  |
| Korean War | ussr | **61%** (3,055) | 34% | 28% |  |  |
| Nasser | ussr | **61%** (2,673) | 39% | 21% |  |  |
| Socialist Governments | ussr | **59%** (3,982) | 36% | 23% |  |  |
| Vietnam Revolts | ussr | **56%** (2,745) | 54% | 2% |  |  |
| Soviets Shoot Down KAL-007 | us | **51%** (1,048) | 51% | 0% |  |  |
| Fidel | ussr | **48%** (3,083) | 11% | 37% |  |  |
| Ussuri River Skirmish | us | **47%** (2,296) | 15% | 32% |  |  |
| De Gaulle Leads France | ussr | **43%** (3,135) | 36% | 7% |  |  |
| Suez Crisis | ussr | **38%** (3,136) | 26% | 12% |  |  |
| Missile Envy | neutral | **37%** (8,168) | 37% | 0% | 54% (3,684) | 24% (4,484) |
| Solidarity | us | **33%** (1,041) | 3% | 30% |  |  |
| Cultural Revolution | ussr | **31%** (2,217) | 15% | 17% |  |  |
| CIA Created | us | **29%** (2,632) | 29% | 0% |  |  |
| OPEC | ussr | **26%** (2,544) | 16% | 10% |  |  |
| Blockade | ussr | **25%** (3,040) | 1% | 24% |  |  |
| Truman Doctrine | us | **24%** (2,522) | 13% | 12% |  |  |
| Portuguese Empire Crumbles | ussr | **24%** (2,329) | 12% | 12% |  |  |
| Bear Trap | us | **23%** (2,114) | 23% | 0% |  |  |
| “Lone Gunman” | ussr | **21%** (2,426) | 21% | 0% |  |  |
| Latin American Debt Crisis | ussr | **21%** (1,067) | 7% | 14% |  |  |
| Duck and Cover | us | **20%** (4,656) | 20% | 0% |  |  |
| Alliance for Progress | us | **19%** (2,372) | 11% | 8% |  |  |
| SALT Negotiations | neutral | **16%** (5,081) | 16% | 0% | 24% (2,627) | 8% (2,454) |
| Willy Brandt | ussr | **16%** (2,313) | 8% | 8% |  |  |
| Muslim Revolution | ussr | **15%** (2,540) | 14% | 1% |  |  |
| NORAD | us | **14%** (3,068) | 14% | 0% |  |  |
| Star Wars | us | **13%** (1,087) | 13% | 0% |  |  |
| Marshall Plan | us | **13%** (3,048) | 13% | 0% |  |  |
| “We Will Bury You” | ussr | **12%** (2,520) | 12% | 0% |  |  |
| Che | ussr | **10%** (2,536) | 9% | 1% |  |  |
| Glasnost | ussr | **10%** (1,058) | 10% | 0% |  |  |
| Terrorism | neutral | **9%** (2,104) | 8% | 1% | 5% (1,050) | 14% (1,054) |
| John Paul II Elected Pope | us | **7%** (2,318) | 4% | 2% |  |  |
| Kitchen Debates | us | **6%** (2,569) | 4% | 2% |  |  |
| US/Japan Mutual Defense Pact | us | **6%** (3,174) | 6% | 0% |  |  |
| The Cambridge Five | ussr | **5%** (4,215) | 5% | 0% |  |  |
| Arab-Israeli War | ussr | **5%** (4,124) | 4% | 1% |  |  |
| Olympic Games | neutral | **5%** (8,960) | 5% | 0% | 8% (4,825) | 0% (4,135) |
| East European Unrest | us | **4%** (4,663) | 4% | 0% |  |  |
| Iranian Hostage Crisis | ussr | **4%** (1,056) | 4% | 0% |  |  |
| How I Learned to Stop Worrying | neutral | **4%** (5,245) | 3% | 0% | 5% (2,630) | 2% (2,615) |
| Romanian Abdication | ussr | **3%** (3,140) | 3% | 0% |  |  |
| “An Evil Empire” | us | **3%** (1,051) | 3% | 0% |  |  |
| Our Man in Tehran | us | **3%** (2,402) | 2% | 0% |  |  |
| Cuban Missile Crisis | neutral | **2%** (5,300) | 2% | 0% | 1% (2,733) | 4% (2,567) |
| Special Relationship | us | **2%** (4,826) | 2% | 0% |  |  |
| Flower Power | ussr | **2%** (2,200) | 2% | 0% |  |  |
| Yuri and Samantha | ussr | **2%** (1,027) | 2% | 0% |  |  |
| AWACS Sale to Saudis | us | **1%** (1,057) | 1% | 0% |  |  |
| Five Year Plan | us | **1%** (4,569) | 1% | 0% |  |  |
| Iran-Contra Scandal | ussr | **1%** (1,051) | 0% | 0% |  |  |
| Reagan Bombs Libya | us | **1%** (1,068) | 1% | 0% |  |  |
| Independent Reds | us | **1%** (2,911) | 1% | 0% |  |  |
| Shuttle Diplomacy | us | **1%** (2,404) | 1% | 0% |  |  |
| Nuclear Test Ban | neutral | **0%** (9,048) | 0% | 0% | 1% (4,844) | 0% (4,204) |
| U-2 Incident | ussr | **0%** (2,246) | 0% | 0% |  |  |
| Latin American Death Squads | neutral | **0%** (5,336) | 0% | 0% | 1% (2,744) | 0% (2,592) |
| Ortega Elected in Nicaragua | ussr | **0%** (1,050) | 0% | 0% |  |  |
| The Iron Lady | us | **0%** (1,057) | 0% | 0% |  |  |
| Warsaw Pact Formed | ussr | **0%** (3,204) | 0% | 0% |  |  |
| Arms Race | neutral | **0%** (5,310) | 0% | 0% | 0% (2,713) | 0% (2,597) |
| Wargames | neutral | **0%** (2,093) | 0% | 0% | 0% (1,039) | 0% (1,054) |
| “One Small Step…” | neutral | **0%** (5,341) | 0% | 0% | 0% (2,850) | 0% (2,491) |
| Chernobyl | us | **0%** (1,033) | 0% | 0% |  |  |
| Summit | neutral | **0%** (5,358) | 0% | 0% | 0% (2,747) | 0% (2,611) |
| Formosan Resolution | us | **0%** (2,995) | 0% | 0% |  |  |
| Comecon | ussr | **0%** (3,185) | 0% | 0% |  |  |
| NATO | us | **0%** (4,240) | 0% | 0% |  |  |
| Nuclear Subs | us | **0%** (2,295) | 0% | 0% |  |  |
| North Sea Oil | us | **0%** (1,048) | 0% | 0% |  |  |
