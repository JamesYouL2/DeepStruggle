# Event census of the shallow soup (2026-10-03)

Owner: for the shallow_E7-02+03+04+05_1200M soup, how often each card in its owner's hand is headlined or played as the event in a round (neutral cards: overall / by US / by USSR), sorted; the opponent's events and chained plays (Grain Sales, Star Wars) excluded; only holdings where playing the event would be legal.

`tools/scripts/event_play_census.py` (8 × 512 greedy self-play games). A holding counts when the owner could have played the event at one of their decisions: a headline choice where the card may be headlined (headlining fires it -- Defectors is only ever playable this way), or an action-round card choice where selecting it offers the event (checked on a copy). An earlier version counted headline legality only when the card was headlined, which put Defectors at 100%; corrected (81%). The engine offers EVENT for some cards whose event would do nothing (One Small Step when not behind in space, Wargames above DEFCON 2), so those still count; NATO's prerequisite is enforced (4,240 holdings -> 1,923).

**Unplayed scoring cards** (8.3% of 5,912 scoring holdings in a 512-game trace): 2.4% discarded by the US with Ask Not What Your Country Can Do For You; 2.7% in hand when the game ended (mostly 20 VP); 0.05% lost to holding a scoring card; ~0.8% taken or discarded by opponent events (Grain Sales, Aldrich Ames, Five Year Plan, Terrorism); ~2.1% discarded at the start of a turn just after the headline choices -- not yet explained.

How often a card in its owner's hand is used for its event -- headlined, or played in an action round as the event -- 4,096 greedy self-play games. One count per holding (the card entering the hand until it leaves), counting only holdings in which the owner could have played the event at one of their decisions: a headline choice where the card may be headlined, or an action-round card choice where selecting it offers the event. US/USSR cards: the owner's holdings only. Neutral cards: whoever holds it, with the split by side.

| card | side | evented (holdings) | headlined | event in a round | held by US | held by USSR |
|:---|:---|---:|---:|---:|---:|---:|
| UN Intervention | neutral | **98%** (8,458) | 0% | 98% | 98% (4,590) | 98% (3,868) |
| Junta | neutral | **97%** (5,372) | 69% | 28% | 97% (2,796) | 97% (2,576) |
| The Reformer | ussr | **97%** (1,038) | 71% | 26% |  |  |
| Pershing II Deployed | ussr | **97%** (1,024) | 49% | 48% |  |  |
| Decolonization | ussr | **97%** (4,128) | 33% | 64% |  |  |
| Brush War | neutral | **96%** (5,477) | 32% | 64% | 96% (2,850) | 96% (2,627) |
| The Voice of America | us | **96%** (2,647) | 33% | 63% |  |  |
| Asia Scoring | neutral | **95%** (9,107) | 21% | 74% | 95% (4,839) | 95% (4,268) |
| Colonial Rear Guards | us | **95%** (2,656) | 27% | 68% |  |  |
| Liberation Theology | ussr | **95%** (2,557) | 45% | 50% |  |  |
| Middle East Scoring | neutral | **93%** (8,910) | 19% | 75% | 93% (4,801) | 93% (4,109) |
| Europe Scoring | neutral | **93%** (9,008) | 16% | 78% | 92% (4,752) | 94% (4,256) |
| OAS Founded | us | **91%** (1,789) | 15% | 77% |  |  |
| Tear Down this Wall | us | **91%** (1,037) | 40% | 51% |  |  |
| Containment | us | **90%** (1,882) | 90% | 0% |  |  |
| Africa Scoring | neutral | **90%** (5,392) | 13% | 78% | 89% (2,685) | 92% (2,707) |
| Central America Scoring | neutral | **90%** (5,433) | 8% | 82% | 88% (2,740) | 92% (2,693) |
| Southeast Asia Scoring | neutral | **90%** (3,816) | 9% | 81% | 85% (1,820) | 94% (1,996) |
| South America Scoring | neutral | **90%** (5,417) | 8% | 82% | 87% (2,797) | 93% (2,620) |
| Quagmire | ussr | **89%** (2,169) | 70% | 20% |  |  |
| ABM Treaty | neutral | **87%** (5,535) | 58% | 29% | 93% (2,935) | 80% (2,600) |
| Grain Sales to Soviets | us | **85%** (2,636) | 85% | 0% |  |  |
| De-Stalinization | ussr | **85%** (2,952) | 12% | 73% |  |  |
| Marine Barracks Bombing | ussr | **85%** (1,083) | 20% | 64% |  |  |
| Captured Nazi Scientist | neutral | **84%** (4,680) | 33% | 52% | 83% (2,099) | 86% (2,581) |
| Allende | ussr | **82%** (2,003) | 26% | 56% |  |  |
| Defectors | us | **81%** (4,579) | 81% | 0% |  |  |
| Panama Canal Returned | us | **79%** (1,849) | 13% | 66% |  |  |
| Sadat Expels Soviets | us | **77%** (1,835) | 9% | 68% |  |  |
| Red Scare/Purge | neutral | **76%** (9,116) | 76% | 0% | 69% (4,946) | 86% (4,170) |
| Brezhnev Doctrine | ussr | **74%** (2,029) | 73% | 0% |  |  |
| Aldrich Ames Remix | ussr | **73%** (1,047) | 73% | 0% |  |  |
| Indo-Pakistani War | neutral | **72%** (8,840) | 25% | 47% | 68% (4,737) | 76% (4,103) |
| Nixon Plays the China Card | us | **69%** (2,073) | 6% | 63% |  |  |
| “Ask Not What Your Country Can Do For You…” | us | **68%** (2,120) | 36% | 32% |  |  |
| Camp David Accords | us | **67%** (2,183) | 10% | 57% |  |  |
| Iran-Iraq War | neutral | **66%** (2,103) | 17% | 49% | 82% (1,053) | 49% (1,050) |
| Puppet Governments | us | **65%** (2,109) | 19% | 46% |  |  |
| South African Unrest | ussr | **63%** (2,546) | 26% | 37% |  |  |
| Korean War | ussr | **61%** (3,055) | 34% | 28% |  |  |
| Nasser | ussr | **61%** (2,673) | 39% | 21% |  |  |
| Socialist Governments | ussr | **59%** (3,982) | 36% | 23% |  |  |
| Missile Envy | neutral | **57%** (5,287) | 57% | 0% | 72% (2,743) | 42% (2,544) |
| Vietnam Revolts | ussr | **56%** (2,745) | 54% | 2% |  |  |
| Soviets Shoot Down KAL-007 | us | **51%** (1,037) | 51% | 0% |  |  |
| Ussuri River Skirmish | us | **48%** (2,248) | 16% | 33% |  |  |
| Fidel | ussr | **48%** (3,083) | 11% | 37% |  |  |
| De Gaulle Leads France | ussr | **43%** (3,135) | 36% | 7% |  |  |
| Suez Crisis | ussr | **38%** (3,136) | 26% | 12% |  |  |
| Solidarity | us | **33%** (1,033) | 3% | 30% |  |  |
| Cultural Revolution | ussr | **31%** (2,217) | 15% | 17% |  |  |
| CIA Created | us | **30%** (2,612) | 30% | 0% |  |  |
| OPEC | ussr | **26%** (2,544) | 16% | 10% |  |  |
| Blockade | ussr | **25%** (3,039) | 1% | 24% |  |  |
| Truman Doctrine | us | **24%** (2,515) | 13% | 12% |  |  |
| Portuguese Empire Crumbles | ussr | **24%** (2,329) | 12% | 12% |  |  |
| Bear Trap | us | **23%** (2,081) | 23% | 0% |  |  |
| “Lone Gunman” | ussr | **21%** (2,425) | 21% | 0% |  |  |
| Latin American Debt Crisis | ussr | **21%** (1,067) | 7% | 14% |  |  |
| Duck and Cover | us | **20%** (4,627) | 20% | 0% |  |  |
| Alliance for Progress | us | **19%** (2,343) | 11% | 8% |  |  |
| SALT Negotiations | neutral | **16%** (5,050) | 16% | 0% | 25% (2,596) | 8% (2,454) |
| Willy Brandt | ussr | **16%** (2,313) | 8% | 8% |  |  |
| Muslim Revolution | ussr | **15%** (2,539) | 14% | 1% |  |  |
| NORAD | us | **14%** (3,048) | 14% | 0% |  |  |
| Star Wars | us | **14%** (1,065) | 14% | 0% |  |  |
| Marshall Plan | us | **13%** (3,029) | 13% | 0% |  |  |
| “We Will Bury You” | ussr | **12%** (2,520) | 12% | 0% |  |  |
| Che | ussr | **10%** (2,536) | 9% | 1% |  |  |
| Glasnost | ussr | **10%** (1,058) | 10% | 0% |  |  |
| Terrorism | neutral | **9%** (2,096) | 8% | 1% | 5% (1,042) | 14% (1,054) |
| John Paul II Elected Pope | us | **7%** (2,278) | 4% | 2% |  |  |
| Kitchen Debates | us | **6%** (2,523) | 4% | 2% |  |  |
| US/Japan Mutual Defense Pact | us | **6%** (3,157) | 6% | 0% |  |  |
| The Cambridge Five | ussr | **5%** (4,214) | 5% | 0% |  |  |
| Arab-Israeli War | ussr | **5%** (4,117) | 4% | 1% |  |  |
| Olympic Games | neutral | **5%** (8,939) | 5% | 0% | 8% (4,804) | 0% (4,135) |
| East European Unrest | us | **4%** (4,638) | 4% | 0% |  |  |
| Iranian Hostage Crisis | ussr | **4%** (1,056) | 4% | 0% |  |  |
| How I Learned to Stop Worrying | neutral | **4%** (5,215) | 3% | 0% | 5% (2,600) | 2% (2,615) |
| Romanian Abdication | ussr | **3%** (3,140) | 3% | 0% |  |  |
| “An Evil Empire” | us | **3%** (1,041) | 3% | 0% |  |  |
| Our Man in Tehran | us | **3%** (2,352) | 3% | 0% |  |  |
| Cuban Missile Crisis | neutral | **2%** (5,272) | 2% | 0% | 1% (2,705) | 4% (2,567) |
| Special Relationship | us | **2%** (4,797) | 2% | 0% |  |  |
| Flower Power | ussr | **2%** (2,200) | 2% | 0% |  |  |
| Yuri and Samantha | ussr | **2%** (1,027) | 2% | 0% |  |  |
| AWACS Sale to Saudis | us | **1%** (1,047) | 1% | 0% |  |  |
| Five Year Plan | us | **1%** (4,543) | 1% | 0% |  |  |
| Iran-Contra Scandal | ussr | **1%** (1,051) | 0% | 0% |  |  |
| Reagan Bombs Libya | us | **1%** (1,062) | 1% | 0% |  |  |
| Independent Reds | us | **1%** (2,899) | 1% | 0% |  |  |
| Shuttle Diplomacy | us | **1%** (2,355) | 1% | 0% |  |  |
| Nuclear Test Ban | neutral | **0%** (9,016) | 0% | 0% | 1% (4,812) | 0% (4,204) |
| U-2 Incident | ussr | **0%** (2,246) | 0% | 0% |  |  |
| Latin American Death Squads | neutral | **0%** (5,290) | 0% | 0% | 1% (2,698) | 0% (2,592) |
| Ortega Elected in Nicaragua | ussr | **0%** (1,050) | 0% | 0% |  |  |
| The Iron Lady | us | **0%** (1,046) | 0% | 0% |  |  |
| Warsaw Pact Formed | ussr | **0%** (3,204) | 0% | 0% |  |  |
| Arms Race | neutral | **0%** (5,271) | 0% | 0% | 0% (2,674) | 0% (2,597) |
| Wargames | neutral | **0%** (2,082) | 0% | 0% | 0% (1,028) | 0% (1,054) |
| “One Small Step…” | neutral | **0%** (5,299) | 0% | 0% | 0% (2,808) | 0% (2,491) |
| Chernobyl | us | **0%** (1,026) | 0% | 0% |  |  |
| Summit | neutral | **0%** (5,314) | 0% | 0% | 0% (2,703) | 0% (2,611) |
| Formosan Resolution | us | **0%** (2,984) | 0% | 0% |  |  |
| Comecon | ussr | **0%** (3,185) | 0% | 0% |  |  |
| NATO | us | **0%** (4,191) | 0% | 0% |  |  |
| Nuclear Subs | us | **0%** (2,262) | 0% | 0% |  |  |
| North Sea Oil | us | **0%** (1,039) | 0% | 0% |  |  |
