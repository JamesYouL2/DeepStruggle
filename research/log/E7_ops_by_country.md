# Where Operations go, humans against the bots, country by country (2026-10-10)

**Question (owner).** Where do humans put their Ops, country by country, and where do the current
best bots put them? The earlier census (`research/log/human_census_E7line.md` on the fork's
`exp/human-census` branch) compared regions in each side's own games, for older bots. A bot that
keeps a region quiet places there less, whatever it would choose in the same spot. So this note
asks the bots at the humans' own positions too, and adds coups and realignment rolls to influence.

**Answer.**

* **The two bots are indistinguishable.** The heads soup and R32's SWA agree with each other on
  every region and country to within a standard error, in their own games and on the humans'
  positions. R32's training (Gumbel targets inside RL) did not move where Ops go.
* **On the same position the bot takes the human's target about a third of the time.** That is 28–35%
  of influence points, 47–74% of coups and 56–72% of realignment rolls. Its probability for the
  human's target is about the same figure, so these are confident disagreements, not near-ties.
* **The Early War gaps are mostly about openings.** The humans' setups are not the bots':
  * US, average influence at the first turn-1 placement: West Germany 3.8, Italy 2.7, France 1.8
    for humans, against France 5.7 and West Germany 0.6 for the bot;
  * USSR: humans hold Poland 4.3 and East Germany 4.1, where the bot spreads into Hungary 2.1 and
    Austria 0.9.

  From a human opening, the bot's US puts its first turn-1 point in Malaysia 43% of the time.
  Humans do so 11% of the time, and the bot in its own games 12% of its turn-1 points. It goes to
  France 25% of the time. The bots' openings here are from 256 of the soup's self-play games.
  Read the Early War rows as the bot playing someone else's opening.
* **From turn 4 on, the gap is stability.** Humans spend Ops on stable countries and those deep
  inside a region; the bots almost never take them. The bots take cheap stability-1 or 2 targets
  instead. Mid and Late War, same positions, share of that side's influence points in percent
  (bot = the soup; R32 is within a few tenths on every row):

| side | humans more (humans / bot) | bot more (humans / bot) |
|:---|:---|:---|
| US | Israel 3.8 / 0.6, South Africa 4.6 / 1.6, Uruguay 1.7 / 0.0, Saudi Arabia 1.9 / 0.3, Angola 3.7 / 2.3, West Germany 4.6 / 3.3 | Mexico 5.6 / 10.1, Guatemala 0.7 / 3.7, Cuba 3.5 / 6.3, Malaysia 0.6 / 2.4, Afghanistan 0.6 / 2.3, United Kingdom 1.0 / 2.5 |
| USSR | Saudi Arabia 4.3 / 0.2, South Africa 3.8 / 0.9, Israel 2.8 / 0.1, Uruguay 1.6 / 0.0, Japan 1.6 / 0.1, Botswana 1.4 / 0.1, Venezuela 5.0 / 3.7 | Colombia 1.0 / 5.8, Lebanon 0.3 / 3.0, Iraq 1.5 / 3.6, Cameroon 0.9 / 2.7, Yugoslavia 0.3 / 1.8 |

  Each gap is 3.5+ standard errors, clustered by game, against both bots.
* **Coups: the bots coup Colombia; humans coup Africa's battlegrounds.** Colombia takes 16% of the
  US bot's Mid/Late War coups on the humans' positions (29% in its own games) and 10% of the
  humans'. For the USSR it is 14% (21%) against 9%. Humans coup Zaire more as the US (10.3% against
  5.7%) and Angola more as the USSR (7.5% against 3.0%), and Haiti more from both seats. By region:
  * US: Africa is 51% of the humans' Mid War coups and 44% of the bot's;
  * USSR: 45% against 36%.
* **By region**, on the same positions:
  * the bots put 8–12 points more of the US's Mid and Late War influence into Central America;
  * humans put 4–7 points more into the Middle East from either seat in the Late War;
  * humans put 4–6 points more into Africa in the Mid War.

These repeat the earlier census in direction (humans: more Middle East, less Central America), now
on equal positions. A gap is a difference, not a cost. The owner has judged the bot right on
regional and battleground allocation before ([expert review](expert_review_E7.md)), so the rows
above are leads to price, not errors.

## Method

Tool `tools/scripts/ops_census.py` (`human`, `selfplay`, `onhuman`, `report`).

* **Records.** One per target chosen with Operations in an action round:
  * a point of influence;
  * a coup's target;
  * a realignment roll's target.

  Event placements, coups and realignments are excluded, as is the setup. A placement costing
  two Ops in an enemy-controlled country is one point. The influence records are exactly
  `placement_census.py`'s (a test pins this).
* **Humans.** The ts-replayer corpus through the converter (CONV-1 fixed), each game's untrusted turn
  dropped. There are 265 games (119 complete, 146 partial, 9 not convertible), and 254 of them
  have Ops targets: 30,080 influence points, 4,466 coups and 1,729 realignment rolls, each with
  its position.
* **Bots.** 4,096 greedy self-play games each, seed 55,000:
  * **soup**: `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt` (sha256 `91a43a8b…`), the heads soup;
  * **r32**: `E7-A8-R1-S44@6400M+R32_20261010_005623/swa_6720-6800M.pt` (sha256 `1d4f65f0…`), R32's
    last SWA, the leaderboard's top network.
* **Same positions.** Each bot's policy at every human record's position: its probability for each
  legal target country and its argmax. The gap for a country is the mean over the humans'
  decisions of (human chose it) − (bot's probability for it). Its standard error is clustered by
  game, because the points of one play and the plays of one game are not independent.
* **Engine.** Built from upstream `2b6c759` (`check_engine_fresh` 6c036593…). Local, 135 s for
  the corpus and ~3 min per 4,096 self-play games.

Full tables, every side and mode, by region and era and by country: [tables](E7_ops_by_country/tables.md).

## Caveats

* The corpus records no players or ratings, so the humans are strong only by repute, and of mixed
  strength.
* "Same positions" are the humans' positions, so later turns still descend from human openings and
  human play. The bot is asked about one point at a time, given the human's earlier points in the
  same play.
* Partial games tilt the human sample toward the Early War. The per-era and per-period tables
  hold that fixed; the pooled agreement does not.
* Nothing here says who is right. Paired playouts at the positions where the two disagree would
  price it, but with the bot as the continuation they undervalue a target whose value lies in a
  follow-up the bot does not know.

## Replicate

```bash
export PYTHONPATH=.:build/release
tools/scripts/check_engine_fresh.sh
python tools/scripts/ops_census.py human --out human_ops.jsonl.gz
for m in soup r32; do
  python tools/scripts/ops_census.py selfplay --checkpoint $m.pt --games 4096 --out ${m}_selfplay.json.gz
  python tools/scripts/ops_census.py onhuman --human human_ops.jsonl.gz --checkpoint $m.pt --out ${m}_onhuman.json.gz
done
python tools/scripts/ops_census.py report --human human_ops.jsonl.gz \
  --selfplay soup=soup_selfplay.json.gz r32=r32_selfplay.json.gz \
  --onhuman soup=soup_onhuman.json.gz r32=r32_onhuman.json.gz --top 12 --out tables.md
```
