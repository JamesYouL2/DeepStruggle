# Humans against bots, card by card and region by region (2026-10-06)

**Question (owner).** Where do strong humans and the current bot differ most, in what a card in hand
is used for (headline, event, Ops, space) and where Ops influence goes, so that the spots can be
opened in the workbench?

**Answer.** On cards, **30 of 109 differ robustly** between the humans and every one of four bots.
In 27 of the 30 the humans event more, and the largest gaps are the events whose value is off the
board or depends on timing:
* Terrorism +53, Che +52, Marshall Plan +48, Bear Trap +44, AWACS +44, Our Man in Tehran +43;
* Aldrich Ames 95% against 50–70%;
* Arms Race, One Small Step, Cuban Missile Crisis, Special Relationship and Wargames, at 19–28%
  against 0–2%.

The bots play scoring cards in the turn's last action round 2–4× as often as humans. Humans space
opponent cards the bots almost never space (Bear Trap 43%, Puppet Governments 44%, Muslim Revolution
26%, Nuclear Subs 18%). **Search does not move any of it.** Gumbel k = 4 at 32 simulations on the newest
model changes no card's event rate by 10 points; the mean shift is 1.9 points, against 5.9 between two
seeds of one recipe. Placement differs less than the event tables suggest. The steady differences:
* humans put 2× the bots' share into the Middle East from the Mid War on;
* humans put a third of the bots' share into Italy;
* humans place in Israel and Japan, which no bot does;
* the bots are 5–10 points more concentrated on battlegrounds.

## Method

* **Census.** `tools/scripts/event_play_census.py`, the [soup census](event_census_soup.md)'s
  definition: one record per holding (the card entering a hand until it leaves), event rates over
  holdings where the owner could have played the event. It now also records Ops / space / kept and
  the turn and round of the play. Its holding logic runs as a tracker fed one decision at a time, so
  the same code counts self-play and the human corpus (through the converter's `on_decision`).
* **Humans.** The 265 distinct convertible games of the ts-replayer corpus: 119 complete and 146
  partial. In a partial game, holdings still open where the record stops are dropped, and so is
  everything spent in that turn, whose late rounds are not recorded. Dropping that turn changed 2
  of 32 robust gaps and moved timing rows by a point or two.
* **Bots.** Greedy self-play, 4,096 games each, seed 55,000:
  * **SWA**: `E7line_swa_4720-4800M`, the newest published model (ONNX);
  * **E7-20-44@2800M**: its line's `snapshot_final`;
  * **soup**: `shallow_E7-02+03+04+05_1200M`;
  * **gumbel32-k4**: the SWA under honest Gumbel-root search at every decision, k = 4, 32
    simulations, first-play urgency 0.2 (`gumbel:<pt>:32:4`, [E7_gumbel_headroom](E7_gumbel_headroom.md)'s
    configuration). It ran on CI, run `37487687364`, 20 runners. The torch network the searcher
    needs was rebuilt from the ONNX by `tools/onnx_to_checkpoint.py`, which reproduces the export to
    2e-4 in logits and 2e-6 in values, with the same favourite at all 512 test positions.
* **Robust gap** (`event_census_compare.py`). Every bot is on the same side of the humans, each by at
  least 10 points and more than 3 standard errors, with 20+ holdings per column. One census differs
  from another seed of its recipe on about one card in five
  (`research/log/event_census_E7-02-44_E7-08-43.md` on the `event-investigation` branch), so
  a single bot's gap is not evidence.
* **Placement** (`placement_census.py`). Every point of Influence placed with Ops in an action
  round, with region, era, and the scoring cards the mover held. Points within one play are not
  independent, so no standard errors are given.
* **Engines.** Humans and greedy bots ran on `d81e54f…` (main, the export's engine). The search run
  ran on `ee07d966…`, which is main plus the Gumbel root's search-tree change in
  `bindings/batched_search.hpp`: search code only, not rules or the observation.

Full tables: [cards](human_census_E7line/cards.md), [placement](human_census_E7line/placement.md).

## Cards

### Events

Robust gaps (▲ humans higher, ▼ lower); evented = headlined or played for the event in a round.

| card | side | humans | gumbel32-k4 | SWA | E7-20-44@2800M | soup | humans − bots |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Terrorism | neutral | 69% (81) | 19% | 18% | 17% | 9% | +53 ▲ |
| Che | ussr | 61% (146) | 8% | 8% | 12% | 8% | +52 ▲ |
| Marshall Plan | us | 60% (155) | 2% | 1% | 30% | 14% | +48 ▲ |
| Bear Trap | us | 68% (133) | 28% | 30% | 16% | 24% | +44 ▲ |
| AWACS Sale to Saudis | us | 49% (47) | 10% | 7% | 3% | 1% | +44 ▲ |
| Our Man in Tehran | us | 45% (108) | 5% | 2% | 1% | 2% | +43 ▲ |
| Chernobyl | us | 46% (39) | 15% | 12% | 2% | 0% | +39 ▲ |
| Muslim Revolution | ussr | 47% (118) | 11% | 12% | 5% | 16% | +36 ▲ |
| Nuclear Subs | us | 36% (119) | 0% | 0% | 1% | 0% | +36 ▲ |
| Aldrich Ames Remix | ussr | 95% (42) | 60% | 62% | 50% | 70% | +35 ▲ |
| Korean War | ussr | 19% (215) | 29% | 35% | 89% | 61% | −34 ▼ |
| Yuri and Samantha | ussr | 37% (46) | 8% | 9% | 4% | 3% | +31 ▲ |
| Iran-Contra Scandal | ussr | 32% (37) | 3% | 3% | 2% | 1% | +30 ▲ |
| Arms Race | neutral | 28% (273) | 1% | 1% | 1% | 0% | +27 ▲ |
| “One Small Step…” | neutral | 26% (275) | 1% | 1% | 1% | 0% | +25 ▲ |
| Truman Doctrine | us | 12% (194) | 34% | 35% | 58% | 23% | −25 ▼ |
| North Sea Oil | us | 26% (34) | 2% | 2% | 1% | 0% | +25 ▲ |
| Puppet Governments | us | 83% (119) | 59% | 51% | 61% | 66% | +24 ▲ |
| Wargames | neutral | 22% (90) | 2% | 0% | 0% | 0% | +22 ▲ |
| Special Relationship | us | 23% (255) | 2% | 2% | 5% | 2% | +20 ▲ |
| The Cambridge Five | ussr | 26% (204) | 8% | 7% | 6% | 7% | +19 ▲ |
| Five Year Plan | us | 24% (241) | 7% | 6% | 3% | 1% | +19 ▲ |
| How I Learned to Stop Worrying | neutral | 25% (251) | 9% | 7% | 6% | 3% | +19 ▲ |
| Romanian Abdication | ussr | 25% (168) | 9% | 7% | 5% | 4% | +19 ▲ |
| SALT Negotiations | neutral | 47% (232) | 37% | 36% | 31% | 16% | +17 ▲ |
| Arab-Israeli War | ussr | 28% (211) | 13% | 13% | 12% | 7% | +17 ▲ |
| Cuban Missile Crisis | neutral | 19% (272) | 2% | 2% | 2% | 2% | +17 ▲ |
| De-Stalinization | ussr | 96% (160) | 83% | 83% | 77% | 85% | +15 ▲ |
| Grain Sales to Soviets | us | 98% (133) | 84% | 87% | 81% | 88% | +13 ▲ |
| UN Intervention | neutral | 85% (464) | 97% | 98% | 97% | 98% | −12 ▼ |

Large gaps that one bot breaks, so not counted robust:
* NORAD: humans 6%; the two newest bots 60–64%, the soup 14%;
* OPEC: humans 69%; the SWA and search ~60%, older bots 27–29%;
* Blockade: humans 39%; the SWA and search 2–3%, older bots 26–37%;
* Tear Down this Wall: humans 81%; the SWA and search 30%, older bots 90–95%.

In each, the newest line moved away from the older bots. The SWA moved toward the humans on OPEC
and away from them on the other three.

What the event rate does not say: the bots are greedy self-play and the humans are humans, so the
positions differ. Wargames is a clear case. The human events are game-winning spots (DEFCON 2 with
a lead; the SWA, asked at those positions, gives them 13% on average). The bots almost never reach
such a spot holding it, and when they do they decline ([expert review](expert_review_E7.md)). For a
cost, open the spots: `census_spots.py --card <name> --checkpoint <model>` lists every human play
with a workbench link and the model's probability of the same move there, least likely first.

### What humans do almost always

One card that humans settle ≥85% one way is broken by the newest bots: **NORAD, played for Ops 91%
by humans, 34–38% by the SWA with or without search** (71% E7-20-44@2800M, 84% soup). The other
near-unanimous human uses all match the bots (all used for Ops: Comecon, Formosan Resolution,
Warsaw Pact, NATO, Nuclear Test Ban, US/Japan, Independent Reds).

### Neutral cards by side

Humans event the neutral non-board cards from both seats: Arms Race 17% US / 38% USSR, One Small
Step 27% / 25%, How I Learned 26% / 24%, Cuban Missile Crisis 23% / 14%. Every bot is at 0–11%.
The bots' side asymmetries that humans do not share:
* SALT, USSR seat: bots 9–24%, humans 48% (the US seat matches: 47% against 23–50%);
* Terrorism: bots 5–27% from either seat, humans 60% / 78%.

Missile Envy also leans that way (USSR seat: bots 38–52%, humans 61%) but is not robust.

### The opponent's cards

Humans space opponent cards the bots almost never space:

| card | owner | humans space | bots space |
|:---|:---|---:|---:|
| Puppet Governments | us | 44% | 21–24% |
| Bear Trap | us | 43% | 1–2% |
| Iranian Hostage Crisis | ussr | 33% | 3–6% |
| Muslim Revolution | ussr | 26% | 4–7% |
| North Sea Oil | us | 20% | 1% |
| Nuclear Subs | us | 18% | 0–1% |
| Arab-Israeli War | ussr | 14% | 2–4% |
| Marshall Plan | us | 14% | 0% |
| Vietnam Revolts | ussr | 14% | 1% |

### When in the turn

31 robust differences in the share of a card's action-round plays made in the turn's last round.
Two patterns:
* **Scoring cards.** Bots play them in the last round 2–4× as often as humans: e.g. South America
  Scoring (USSR) 8% vs 21–47%, Southeast Asia Scoring (USSR) 6% vs 20–37%, Middle East Scoring
  (USSR) 9% vs 25–39%. Every region's card on at least one seat; Asia Scoring's gap is smaller and
  not robust on either.
* **Opponent cards and dead cards.** Humans hold some to the last round where the bots dump them
  early: Nuclear Subs held by the USSR 49% vs 3–6%, Chernobyl held by the USSR 69% vs 21–35%,
  Bear Trap (US) 42% vs 2–4%. Others go the other way: CIA Created (US) 14% vs 36–56%.

Mean action round moves with the share; the full table has both.

## Placement

| | humans | the four bots |
|:---|---:|---:|
| Middle East, Mid War | 12% | 5–6% |
| Middle East, Late War | 18% | 7–9% |
| Europe, Mid War | 12% | 13–20% |
| Europe, Late War | 25% | 26–35% |
| Central America, Mid War | 13% | 18–20% |
| battleground share, Mid War | 79% | 83–89% |

Holding a region's scoring card raises that region's share for humans and bots alike. The bots
swing more on Europe and Central America (e.g. Central America 40–47% holding vs 11–12% not;
humans 31% vs 8%). The humans swing more on the Middle East (36% vs 16%, bots 24–28% vs 8–10%).

By country, per side, every bot on one side of the humans:
* **Italy**: humans 1.4% (US) / 1.5% (USSR) of their influence, bots 4.4–6.9%;
* **Israel**: humans 2.8% / 2.3%, bots ≤ 0.4%;
* **Japan**: humans 1.3% from each side, bots ≤ 0.2%;
* Mexico for the USSR: humans 1.3%, bots 3.7–4.2%;
* North Korea for the US: humans 1.8%, bots 3.0–4.9%.

These repeat the [expert review](expert_review_E7.md)'s corpus comparison, where the owner judged
the bot right on regional allocation. So they are leads to check, not errors.

## Caveats

* The corpus is "strong" only by repute. It records no players or ratings, so strong and average
  humans cannot be separated.
* Partial games tilt the human sample toward the Early War (13,599 placement points in turns 1–3,
  5,284 in 8–10). Rates are per era or per card, so this changes weights, not rates.
* A gap is a difference, not a cost. Frequency comes from here; cost needs paired playouts (the
  owner's rule, [expert review](expert_review_E7.md)).
