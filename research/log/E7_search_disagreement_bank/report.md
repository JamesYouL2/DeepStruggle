# Search disagreement bank: 8000 positions

Population-weighted (each bank row stands for `weight` raw-play decisions). Values and regret are in the value head's units, the mover's [-1, 1]: 0.02 is one point of win probability.

Confidence of the reference: high 2896, medium 2942, low 2162.

## Reference-best recall by raw top-k (high-confidence positions)

| positions | n | top-1 | top-2 | top-4 | top-8 | top-16 | raw regret outside top-4 | outside top-8 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 2896 | 76.8% | 91.2% | 98.1% | 99.6% | 100.0% | 6.9% | 1.1% |
| raw wrong | 1321 | 0.0% | 62.1% | 91.8% | 98.5% | 100.0% | 6.9% | 1.1% |

## Each method against the reference

| method | = reference | = raw | mean regret | median | p90 | p95 | p99 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| raw | 52.3% | 100.0% | 0.0080 | 0.0000 | 0.0284 | 0.0481 | 0.1028 |
| g16 | 54.0% | 85.9% | 0.0071 | 0.0000 | 0.0270 | 0.0451 | 0.0973 |
| g64 | 55.2% | 82.0% | 0.0066 | 0.0000 | 0.0263 | 0.0424 | 0.0934 |
| g256 | 56.8% | 79.5% | 0.0062 | 0.0000 | 0.0256 | 0.0418 | 0.0899 |

**Calibration.** Over the decisions a raw game makes (504 with two or more legal moves), the measured regret adds up to raw 201.0, g16 178.8, g64 166.8, g256 154.9 points of win probability a game. A tournament measured Gumbel k=8 @256 at about +9 points over the raw network; a total far from that says the regret measures search's opinion of itself more than strength.

## Concentration of raw regret (positive part)

| worst share of decisions | share of raw regret |
|:---|---:|
| 0.1% | 4.6% |
| 1.0% | 18.9% |
| 5.0% | 47.8% |
| 10.0% | 68.0% |
| 20.0% | 89.8% |
| 50.0% | 100.0% |

## Raw regret by turn

| turn | n | of decisions | regret raw | regret g16 | regret g64 | regret g256 | raw ≠ best |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 962 | 14.7% | 0.0051 | 0.0046 | 0.0046 | 0.0043 | 41.9% |
| 10 | 741 | 7.0% | 0.0122 | 0.0092 | 0.0086 | 0.0077 | 53.7% |
| 2 | 848 | 11.2% | 0.0087 | 0.0078 | 0.0073 | 0.0067 | 48.3% |
| 3 | 803 | 10.7% | 0.0083 | 0.0078 | 0.0071 | 0.0068 | 54.3% |
| 4 | 705 | 11.4% | 0.0078 | 0.0066 | 0.0062 | 0.0064 | 48.0% |
| 5 | 689 | 10.7% | 0.0071 | 0.0060 | 0.0056 | 0.0061 | 42.4% |
| 6 | 659 | 9.4% | 0.0077 | 0.0071 | 0.0066 | 0.0058 | 43.6% |
| 7 | 719 | 9.3% | 0.0110 | 0.0104 | 0.0090 | 0.0081 | 51.3% |
| 8 | 1009 | 8.7% | 0.0069 | 0.0067 | 0.0059 | 0.0051 | 49.5% |
| 9 | 865 | 6.8% | 0.0076 | 0.0068 | 0.0073 | 0.0055 | 49.0% |

## Raw regret by action round

| action round | n | of decisions | regret raw | regret g16 | regret g64 | regret g256 | raw ≠ best |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1591 | 10.9% | 0.0080 | 0.0069 | 0.0068 | 0.0065 | 48.0% |
| 1 | 844 | 14.2% | 0.0064 | 0.0057 | 0.0057 | 0.0054 | 41.2% |
| 2 | 935 | 15.8% | 0.0076 | 0.0070 | 0.0062 | 0.0058 | 53.5% |
| 3 | 937 | 13.1% | 0.0084 | 0.0073 | 0.0069 | 0.0062 | 46.9% |
| 4 | 888 | 14.1% | 0.0084 | 0.0076 | 0.0065 | 0.0061 | 48.4% |
| 5 | 947 | 11.7% | 0.0081 | 0.0078 | 0.0065 | 0.0054 | 50.2% |
| 6 | 1111 | 13.1% | 0.0080 | 0.0067 | 0.0064 | 0.0066 | 46.1% |
| 7 | 737 | 7.1% | 0.0101 | 0.0088 | 0.0094 | 0.0087 | 45.6% |
| 8 | 10 | 0.1% | -0.0058 | -0.0066 | -0.0058 | -0.0032 | 66.3% |

## Raw regret by decision

| decision | n | of decisions | regret raw | regret g16 | regret g64 | regret g256 | raw ≠ best |
|:---|---:|---:|---:|---:|---:|---:|---:|
| card | 1776 | 22.6% | 0.0112 | 0.0096 | 0.0087 | 0.0075 | 52.0% |
| event_choice | 234 | 0.3% | 0.0038 | 0.0040 | 0.0022 | 0.0010 | 18.1% |
| headline | 1242 | 3.7% | 0.0110 | 0.0095 | 0.0089 | 0.0075 | 38.0% |
| point_node | 1775 | 49.3% | 0.0052 | 0.0045 | 0.0042 | 0.0041 | 51.1% |
| select_op_mode | 1200 | 4.5% | 0.0084 | 0.0081 | 0.0079 | 0.0073 | 32.4% |
| select_play_mode | 1773 | 19.7% | 0.0107 | 0.0102 | 0.0097 | 0.0092 | 40.0% |

## Raw regret by raw rank of the best move

| raw rank of the best move | n | of decisions | regret raw | regret g16 | regret g64 | regret g256 | raw ≠ best |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3255 | 52.3% | -0.0017 | -0.0003 | -0.0000 | 0.0003 | 0.0% |
| 2 | 2609 | 23.7% | 0.0187 | 0.0137 | 0.0124 | 0.0112 | 100.0% |
| 3 | 1288 | 13.3% | 0.0187 | 0.0164 | 0.0150 | 0.0139 | 100.0% |
| 4 | 656 | 8.4% | 0.0171 | 0.0165 | 0.0151 | 0.0142 | 100.0% |
| 5 | 103 | 1.1% | 0.0239 | 0.0171 | 0.0169 | 0.0118 | 100.0% |
| 6 | 34 | 0.3% | 0.0268 | 0.0232 | 0.0257 | 0.0164 | 100.0% |
| 7 | 23 | 0.3% | 0.0243 | 0.0319 | 0.0333 | 0.0177 | 100.0% |
| 8 | 12 | 0.3% | 0.0193 | 0.0189 | 0.0194 | 0.0194 | 100.0% |
| 9+ | 20 | 0.4% | 0.0133 | 0.0121 | 0.0115 | 0.0110 | 100.0% |

## Raw regret by confidence

| confidence | n | of decisions | regret raw | regret g16 | regret g64 | regret g256 | raw ≠ best |
|:---|---:|---:|---:|---:|---:|---:|---:|
| high | 2896 | 36.6% | 0.0084 | 0.0073 | 0.0065 | 0.0054 | 23.2% |
| low | 2162 | 26.9% | 0.0062 | 0.0054 | 0.0051 | 0.0051 | 82.4% |
| medium | 2942 | 36.5% | 0.0089 | 0.0082 | 0.0078 | 0.0077 | 46.7% |

## Raw regret by agreement

| agreement | n | of decisions | regret raw | regret g16 | regret g64 | regret g256 | raw ≠ best |
|:---|---:|---:|---:|---:|---:|---:|---:|
| agree | 2000 | 69.2% | 0.0053 | 0.0053 | 0.0053 | 0.0053 | 38.0% |
| disagree | 6000 | 30.8% | 0.0139 | 0.0111 | 0.0095 | 0.0080 | 69.3% |

## Raw regret by card

| card | n | of decisions | regret raw | regret g16 | regret g64 | regret g256 | raw ≠ best |
|:---|---:|---:|---:|---:|---:|---:|---:|
| - | 2842 | 27.3% | 0.0116 | 0.0101 | 0.0092 | 0.0082 | 51.8% |
| ABM Treaty | 77 | 0.5% | 0.0141 | 0.0114 | 0.0108 | 0.0121 | 65.9% |
| AWACS Sale to Saudis | 23 | 0.3% | 0.0090 | 0.0029 | 0.0064 | 0.0041 | 43.2% |
| Aldrich Ames Remix | 39 | 0.3% | 0.0009 | 0.0016 | 0.0011 | 0.0009 | 19.7% |
| Allende | 27 | 0.2% | 0.0068 | 0.0057 | 0.0032 | 0.0061 | 62.4% |
| Alliance for Progress | 27 | 0.6% | 0.0067 | 0.0071 | 0.0048 | 0.0059 | 18.7% |
| Arab-Israeli War | 104 | 0.8% | 0.0113 | 0.0137 | 0.0117 | 0.0087 | 61.7% |
| Arms Race | 41 | 1.1% | 0.0074 | 0.0058 | 0.0070 | 0.0063 | 55.7% |
| Bear Trap | 42 | 0.8% | 0.0018 | 0.0013 | 0.0010 | 0.0001 | 42.0% |
| Blockade | 65 | 0.7% | 0.0101 | 0.0099 | 0.0090 | 0.0105 | 63.9% |
| Brezhnev Doctrine | 36 | 0.4% | 0.0070 | 0.0054 | 0.0062 | 0.0097 | 48.8% |
| Brush War | 20 | 0.7% | 0.0021 | 0.0032 | 0.0025 | 0.0013 | 7.8% |
| CIA Created | 99 | 0.8% | 0.0037 | 0.0033 | 0.0038 | 0.0033 | 24.9% |
| Camp David Accords | 31 | 0.3% | 0.0058 | 0.0027 | 0.0016 | 0.0032 | 46.0% |
| Captured Nazi Scientist | 13 | 0.1% | 0.0139 | 0.0134 | 0.0092 | 0.0110 | 93.6% |
| Che | 69 | 1.2% | 0.0093 | 0.0083 | 0.0084 | 0.0085 | 57.6% |
| Chernobyl | 88 | 0.4% | 0.0125 | 0.0123 | 0.0084 | 0.0096 | 56.4% |
| Colonial Rear Guards | 63 | 1.0% | 0.0088 | 0.0068 | 0.0066 | 0.0059 | 48.0% |
| Comecon | 72 | 1.5% | 0.0076 | 0.0074 | 0.0076 | 0.0068 | 59.6% |
| Containment | 39 | 0.6% | 0.0040 | 0.0017 | 0.0019 | 0.0016 | 26.1% |
| Cuban Missile Crisis | 41 | 1.1% | 0.0161 | 0.0162 | 0.0155 | 0.0160 | 58.1% |
| Cultural Revolution | 34 | 0.6% | 0.0047 | 0.0033 | 0.0017 | 0.0025 | 26.7% |
| De Gaulle Leads France | 55 | 0.5% | 0.0097 | 0.0092 | 0.0095 | 0.0075 | 43.1% |
| De-Stalinization | 57 | 1.0% | 0.0011 | 0.0025 | -0.0011 | -0.0024 | 48.2% |
| Decolonization | 67 | 1.3% | 0.0023 | 0.0006 | 0.0016 | 0.0011 | 41.4% |
| Defectors | 56 | 0.3% | 0.0120 | 0.0118 | 0.0127 | 0.0099 | 49.8% |
| Duck and Cover | 70 | 1.4% | 0.0053 | 0.0040 | 0.0036 | 0.0030 | 50.4% |
| East European Unrest | 106 | 1.9% | 0.0090 | 0.0084 | 0.0075 | 0.0071 | 48.6% |
| Fidel | 49 | 0.6% | 0.0081 | 0.0077 | 0.0067 | 0.0068 | 38.8% |
| Five Year Plan | 111 | 1.9% | 0.0057 | 0.0058 | 0.0046 | 0.0052 | 49.4% |
| Flower Power | 58 | 1.1% | 0.0063 | 0.0048 | 0.0052 | 0.0046 | 50.2% |
| Formosan Resolution | 56 | 0.7% | 0.0087 | 0.0076 | 0.0071 | 0.0077 | 33.8% |
| Glasnost | 28 | 0.3% | 0.0077 | 0.0059 | 0.0062 | 0.0042 | 38.1% |
| Grain Sales to Soviets | 28 | 0.4% | 0.0045 | 0.0013 | 0.0007 | 0.0019 | 10.4% |
| How I Learned to Stop Worrying | 57 | 0.6% | 0.0014 | 0.0015 | -0.0005 | 0.0005 | 30.7% |
| Independent Reds | 88 | 1.3% | 0.0044 | 0.0032 | 0.0038 | 0.0046 | 50.3% |
| Indo-Pakistani War | 53 | 0.7% | 0.0078 | 0.0082 | 0.0088 | 0.0093 | 34.8% |
| Iran-Contra Scandal | 33 | 0.3% | 0.0028 | 0.0026 | 0.0027 | 0.0029 | 59.6% |
| Iran-Iraq War | 19 | 0.2% | 0.0023 | 0.0020 | 0.0022 | 0.0025 | 59.9% |
| Iranian Hostage Crisis | 33 | 0.3% | 0.0096 | 0.0098 | 0.0087 | 0.0087 | 50.7% |
| John Paul II Elected Pope | 15 | 0.2% | 0.0052 | 0.0044 | 0.0050 | 0.0036 | 51.9% |
| Junta | 85 | 0.7% | 0.0054 | 0.0051 | 0.0046 | 0.0060 | 33.7% |
| Kitchen Debates | 33 | 0.2% | 0.0031 | 0.0038 | 0.0062 | 0.0035 | 25.3% |
| Korean War | 64 | 0.4% | 0.0189 | 0.0206 | 0.0150 | 0.0145 | 40.5% |
| Latin American Death Squads | 36 | 0.7% | 0.0073 | 0.0034 | 0.0064 | 0.0078 | 47.3% |
| Latin American Debt Crisis | 39 | 0.3% | 0.0020 | 0.0028 | 0.0025 | 0.0004 | 48.4% |
| Liberation Theology | 36 | 0.5% | 0.0049 | 0.0010 | 0.0003 | 0.0093 | 39.3% |
| Marine Barracks Bombing | 25 | 0.3% | 0.0033 | 0.0045 | 0.0022 | 0.0009 | 60.6% |
| Marshall Plan | 81 | 2.7% | 0.0036 | 0.0025 | 0.0027 | 0.0028 | 52.4% |
| Missile Envy | 56 | 0.5% | 0.0146 | 0.0135 | 0.0148 | 0.0124 | 48.3% |
| Muslim Revolution | 54 | 1.1% | 0.0036 | 0.0038 | 0.0041 | 0.0027 | 57.7% |
| NATO | 56 | 1.7% | 0.0067 | 0.0067 | 0.0060 | 0.0051 | 36.5% |
| NORAD | 61 | 1.2% | 0.0081 | 0.0079 | 0.0075 | 0.0067 | 46.8% |
| Nasser | 49 | 0.3% | 0.0176 | 0.0164 | 0.0156 | 0.0154 | 50.7% |
| Nixon Plays the China Card | 29 | 0.3% | 0.0078 | 0.0080 | 0.0080 | 0.0099 | 35.6% |
| North Sea Oil | 30 | 0.2% | 0.0034 | 0.0031 | 0.0024 | 0.0023 | 61.6% |
| Nuclear Subs | 43 | 0.5% | 0.0042 | 0.0042 | 0.0046 | 0.0044 | 25.1% |
| Nuclear Test Ban | 70 | 1.9% | 0.0070 | 0.0069 | 0.0060 | 0.0054 | 39.5% |
| OAS Founded | 55 | 0.5% | 0.0042 | 0.0035 | 0.0030 | 0.0029 | 27.3% |
| OPEC | 39 | 0.8% | 0.0077 | 0.0069 | 0.0073 | 0.0054 | 76.9% |
| Olympic Games | 70 | 1.4% | 0.0052 | 0.0043 | 0.0037 | 0.0043 | 37.6% |
| Ortega Elected in Nicaragua | 19 | 0.4% | 0.0102 | 0.0097 | 0.0103 | 0.0095 | 50.6% |
| Our Man in Tehran | 78 | 1.1% | 0.0166 | 0.0169 | 0.0175 | 0.0163 | 46.5% |
| Panama Canal Returned | 45 | 0.3% | 0.0026 | 0.0038 | 0.0065 | 0.0028 | 41.5% |
| Pershing II Deployed | 21 | 0.3% | 0.0040 | 0.0027 | 0.0094 | 0.0036 | 49.3% |
| Portuguese Empire Crumbles | 37 | 0.3% | 0.0020 | -0.0027 | 0.0011 | 0.0017 | 23.9% |
| Puppet Governments | 46 | 0.7% | 0.0060 | 0.0060 | 0.0052 | 0.0043 | 31.5% |
| Quagmire | 17 | 0.3% | 0.0040 | 0.0045 | 0.0037 | 0.0028 | 18.1% |
| Reagan Bombs Libya | 29 | 0.3% | 0.0053 | 0.0043 | 0.0053 | 0.0040 | 48.2% |
| Red Scare/Purge | 24 | 0.5% | 0.0092 | 0.0075 | 0.0077 | 0.0072 | 37.4% |
| Romanian Abdication | 66 | 0.6% | 0.0053 | 0.0012 | 0.0019 | 0.0050 | 41.4% |
| SALT Negotiations | 39 | 0.3% | 0.0088 | 0.0095 | 0.0096 | 0.0075 | 44.3% |
| Sadat Expels Soviets | 30 | 0.2% | 0.0066 | 0.0056 | 0.0049 | 0.0032 | 78.1% |
| Shuttle Diplomacy | 52 | 0.9% | 0.0061 | 0.0063 | 0.0058 | 0.0054 | 43.2% |
| Socialist Governments | 78 | 1.3% | 0.0071 | 0.0070 | 0.0061 | 0.0056 | 77.9% |
| Solidarity | 27 | 0.2% | 0.0013 | -0.0024 | -0.0003 | -0.0016 | 21.0% |
| South African Unrest | 45 | 0.6% | 0.0082 | 0.0075 | 0.0084 | 0.0065 | 37.8% |
| Soviets Shoot Down KAL-007 | 18 | 0.2% | 0.0047 | 0.0035 | 0.0034 | 0.0024 | 42.7% |
| Special Relationship | 124 | 1.6% | 0.0066 | 0.0051 | 0.0055 | 0.0053 | 61.1% |
| Star Wars | 24 | 0.2% | 0.0137 | 0.0063 | 0.0133 | 0.0088 | 25.1% |
| Suez Crisis | 76 | 1.5% | 0.0053 | 0.0079 | 0.0045 | 0.0034 | 50.2% |
| Summit | 20 | 0.5% | 0.0042 | 0.0043 | 0.0032 | 0.0081 | 51.4% |
| Tear Down this Wall | 27 | 0.2% | 0.0119 | 0.0106 | 0.0096 | 0.0053 | 56.1% |
| Terrorism | 16 | 0.2% | 0.0048 | 0.0022 | 0.0033 | 0.0019 | 68.9% |
| The Cambridge Five | 91 | 1.1% | 0.0069 | 0.0059 | 0.0060 | 0.0062 | 60.1% |
| The China Card | 51 | 1.5% | 0.0057 | 0.0042 | 0.0032 | 0.0029 | 30.6% |
| The Iron Lady | 32 | 0.5% | 0.0074 | 0.0076 | 0.0074 | 0.0070 | 56.6% |
| The Reformer | 23 | 0.4% | 0.0012 | -0.0004 | -0.0014 | -0.0015 | 58.2% |
| The Voice of America | 51 | 1.1% | 0.0060 | 0.0054 | 0.0057 | 0.0042 | 52.3% |
| Truman Doctrine | 64 | 0.8% | -0.0003 | -0.0010 | -0.0004 | -0.0011 | 30.2% |
| U-2 Incident | 48 | 1.0% | 0.0045 | 0.0031 | 0.0025 | 0.0042 | 59.1% |
| UN Intervention | 34 | 0.8% | -0.0013 | -0.0012 | -0.0016 | -0.0018 | 8.4% |
| US/Japan Mutual Defense Pact | 54 | 1.0% | 0.0061 | 0.0052 | 0.0050 | 0.0045 | 44.1% |
| Ussuri River Skirmish | 36 | 0.8% | 0.0312 | 0.0305 | 0.0301 | 0.0298 | 46.7% |
| Vietnam Revolts | 39 | 0.2% | 0.0121 | 0.0133 | 0.0050 | 0.0094 | 67.0% |
| Wargames | 25 | 0.3% | 0.0058 | 0.0044 | 0.0023 | 0.0009 | 66.5% |
| Warsaw Pact Formed | 217 | 2.1% | 0.0036 | 0.0035 | 0.0031 | 0.0033 | 44.3% |
| Willy Brandt | 45 | 0.5% | 0.0066 | 0.0052 | 0.0028 | 0.0044 | 52.4% |
| Yuri and Samantha | 21 | 0.2% | 0.0106 | 0.0025 | 0.0094 | 0.0017 | 21.6% |
| “An Evil Empire” | 29 | 0.4% | 0.0058 | 0.0046 | 0.0046 | 0.0034 | 36.6% |
| “Ask Not What Your Country Can Do For You…” | 138 | 1.0% | 0.0019 | 0.0016 | -0.0001 | 0.0011 | 47.6% |
| “Lone Gunman” | 50 | 0.2% | 0.0080 | 0.0073 | 0.0082 | 0.0086 | 55.3% |
| “One Small Step…” | 31 | 0.6% | 0.0109 | 0.0099 | 0.0103 | 0.0071 | 64.3% |
| “We Will Bury You” | 21 | 0.4% | 0.0019 | 0.0011 | 0.0031 | 0.0017 | 35.9% |

## Why raw is wrong (high-confidence positions where raw is not the best)

| searcher | best outside its candidates (errors) | (regret) | picks best when inside | bare critic ranks best first when inside |
|:---|---:|---:|---:|---:|
| g16 | 8.2% | 6.9% | 26.8% | 43.5% |
| g64 | 8.2% | 6.9% | 34.6% | 43.5% |
| g256 | 1.5% | 1.1% | 46.1% | 45.1% |

## Oracle candidates: the best move forced into the candidate set

| searcher | positions | picks the forced best move |
|:---|---:|---:|
| g256 | 15 | 42.2% |
| g64 | 136 | 27.5% |

## Playout check: outcomes, independent of the value head

**the reference best over the raw move**, per raw-play decision: playouts +0.087 ± 0.109 points of win probability; search-valued +0.430. Per game (504 decisions): playouts +43.6 ± 54.8, search-valued +216.5.

**Gumbel k=8 @256's move over the raw move**, per raw-play decision: playouts +0.035 ± 0.031 points of win probability; search-valued +0.076. Per game (504 decisions): playouts +17.4 ± 15.7, search-valued +38.2.

Where the playout-measured gain sits (population-weighted; gain per game if --decisions-per-game is given, else per decision):

| decisions | move | share of decisions | playout gain over raw (points) |
|:---|---:|---:|---:|
| search-valued regret < 2 points | best | 85.5% | -4.4 ± 49.8 |
| search-valued regret < 2 points | g256 | 85.5% | -12.8 ± 14.3 |
| search-valued regret >= 2 points | best | 6.8% | +44.6 ± 8.3 |
| search-valued regret >= 2 points | g256 | 6.8% | +28.9 ± 2.0 |

Recall where the playouts confirm a search-valued regret of 2+ points (205 positions): top-2 49.8%, top-4 94.6%, top-8 100.0%.

Where the reference best differs from raw, by the search-valued regret (unweighted):

| search-valued regret | positions | search says (points) | playouts say (points) | playouts favour best |
|:---|---:|---:|---:|---:|
| < 1 point | 294 | +0.01 | -0.33 ± 0.16 | 38.1% |
| 1-2 points | 242 | +1.14 | +0.24 ± 0.20 | 50.4% |
| 2-5 points | 901 | +3.12 | +1.46 ± 0.12 | 58.5% |
| 5+ points | 328 | +8.64 | +4.87 ± 0.19 | 68.0% |

## Highest-regret raw decisions

| id | turn/AR | side | decision | card | raw | best (raw rank) | regret ± SE |
|:---|---:|---:|---:|---:|---:|---:|---:|
| bb6711986ec102e4 | T10 AR7 | US | POINT_NODE | Yuri and Samantha | PointNode #79 (Brazil) | PointNode #30 (Libya) (4) | 0.695 ± 0.040 |
| c1d8d06e570183a3 | T10 AR7 | USSR | SELECT_OP_MODE | Camp David Accords | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN (2) | 0.565 ± 0.107 |
| adf6e91446a8294f | T5 AR0 | USSR | SELECT_CARD | - | SelectCard #75 (Liberation Theology) | SelectCard #34 (Nuclear Test Ban) (4) | 0.450 ± 0.037 |
| a91ec6b48e1bd107 | T4 AR3 | US | SELECT_PLAY_MODE | Brush War | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: EVENT (2) | 0.438 ± 0.071 |
| 6ba58d5b1bb067ff | T10 AR7 | USSR | SELECT_OP_MODE | Puppet Governments | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN (2) | 0.402 ± 0.096 |
| af8828ca9cbc177c | T10 AR4 | USSR | SELECT_CARD | - | SelectCard #101 (Solidarity) | SelectCard #11 (Korean War) (2) | 0.383 ± 0.055 |
| 71af05ee0938bd25 | T6 AR7 | US | SELECT_PLAY_MODE | De-Stalinization | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: SPACE (2) | 0.373 ± 0.086 |
| 46faae92fa3ca758 | T10 AR3 | USSR | SELECT_PLAY_MODE | Liberation Theology | Resolution: EVENT | Resolution: OPS_REALIGN / OpMode: REALIGN (2) | 0.368 ± 0.034 |
| df577f7b923469e9 | T9 AR0 | USSR | SELECT_CARD | - | SelectCard #12 (Romanian Abdication) | SelectCard #90 (Glasnost) (2) | 0.358 ± 0.042 |
| ad9106c11687bde5 | T9 AR7 | USSR | SELECT_PLAY_MODE | Colonial Rear Guards | Resolution: SPACE | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE (3) | 0.357 ± 0.057 |
| d82d5e9ef2140452 | T10 AR1 | USSR | SELECT_PLAY_MODE | Muslim Revolution | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN (2) | 0.345 ± 0.061 |
| 0c78109af60dcf6b | T9 AR7 | US | SELECT_CARD | - | CONFIRM_DONE / PASS | SelectCard #6 (The China Card) (2) | 0.344 ± 0.077 |
| c22cee4331bb2c3d | T10 AR4 | USSR | SELECT_PLAY_MODE | OPEC | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN (2) | 0.340 ± 0.041 |
| 6a801d9e7c7d130f | T4 AR7 | US | SELECT_PLAY_MODE | De Gaulle Leads France | Resolution: SPACE | Resolution: EVENT (3) | 0.326 ± 0.055 |
| f91c1e15c4427bfd | T9 AR0 | US | SELECT_CARD | - | SelectCard #108 (Our Man in Tehran) | SelectCard #29 (East European Unrest) (2) | 0.319 ± 0.030 |
| 45c6dbeb176def6f | T10 AR5 | USSR | SELECT_PLAY_MODE | Suez Crisis | Resolution: EVENT | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE (2) | 0.316 ± 0.027 |
| 9e6ec452af6a2f54 | T10 AR4 | US | SELECT_PLAY_MODE | “Ask Not What Your Country Can Do For You…” | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE | Resolution: OPS_REALIGN / OpMode: REALIGN (2) | 0.310 ± 0.053 |
| 7e7f39c28150b077 | T10 AR0 | US | SELECT_CARD | - | SelectCard #85 (Star Wars) | SelectCard #101 (Solidarity) (2) | 0.310 ± 0.033 |
| e146c8af13c208da | T5 AR0 | US | SELECT_CARD | - | SelectCard #67 (Grain Sales to Soviets) | SelectCard #77 (“Ask Not What Your Country Can Do For You…”) (2) | 0.305 ± 0.038 |
| 3db9cf2f5f6f86e7 | T10 AR0 | US | SELECT_CARD | Missile Envy | SelectCard #99 (Pershing II Deployed) | SelectCard #33 (De-Stalinization) (3) | 0.298 ± 0.031 |
| 54de8161b80ca913 | T10 AR6 | US | SELECT_PLAY_MODE | Reagan Bombs Libya | Resolution: OPS_COUP / OpMode: COUP | Resolution: OPS_REALIGN / OpMode: REALIGN (2) | 0.294 ± 0.069 |
| a78cfbeb5f4b7231 | T10 AR6 | US | SELECT_CARD | - | SelectCard #99 (Pershing II Deployed) | SelectCard #95 (Latin American Debt Crisis) (2) | 0.293 ± 0.036 |
| db3fd7f01c16d7eb | T10 AR6 | USSR | SELECT_OP_MODE | Chernobyl | Resolution: OPS_REALIGN / OpMode: REALIGN | Resolution: OPS_INFLUENCE / OpMode: INFLUENCE (2) | 0.288 ± 0.017 |
| 66d2e15613dcc501 | T3 AR0 | US | SELECT_CARD | - | SelectCard #106 (NORAD) | SelectCard #103 (Defectors) (2) | 0.286 ± 0.015 |
| ec3a618d888ebc84 | T5 AR4 | US | SELECT_CARD | - | SelectCard #63 (Colonial Rear Guards) | SelectCard #29 (East European Unrest) (2) | 0.284 ± 0.019 |

