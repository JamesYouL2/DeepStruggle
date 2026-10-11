# Turn 1 on the R34 soup (2026-10-11): the owner's six questions

**Network.** All runs use the R34 soup, `E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M`
(leaderboard 1650, the strongest network). Every value is to this network's greedy play after the
move in question.

**Runs.**
* Setup matrix 38166064972: 400,000 games over 100,000 deals.
* Full turn-1 pricing 38166070718: 500 games, 96 worlds each, with the 60 clearest departures
  replayed to the end of the game.
* Three censuses on the same 10,000 deals (`--worlds 2`, frequencies only), read by
  `tools/turn1_census.py`:
  * the network's own setup, 38166076520;
  * 4222 forced, 38166082352;
  * 3322 forced, 38166088993.

**Setups named.**
* **4222** is West Germany 4, France 2, Italy 2, Iran 2. That is `us_r34`, the network's own US
  setup on 70% of deals; Iran 2 is the fixed point plus a bonus point.
* **3322** is West Germany 3, France 3, Italy 2, Iran 2 (`us_wg3_fr3_it2`).
* The network's USSR setup is **Yugoslavia 3, Poland 3** (`ussr_r34`) on every deal.

## 1. Is Yugoslavia 3 better than East Germany 4, Poland 4, Yugoslavia 1?

**Under this network's play, yes: by +0.61 ± 0.17 points of the USSR's score**, averaged over
4222 and 3322 (+0.67 against 4222, +0.55 against 3322). The joint test gives p = 0.002.
[Matrix](E7_setup_matrix/matrix_r34_yug.md).

Earlier runs agree:
* Against East Germany 4, Poland 4, Austria 1 the margin was +0.72 ± 0.13
  ([3 × 3](E7_setup_by_hand.md)).
* The cells shared with that 3 × 3 reproduce it to the hundredth.

The network trained only from its own setup, so the comparison favours it. A searched
continuation is the test of whether the edge survives better play.

## 2. The top USSR headlines

[Census](E7_setup_matrix/turn1_census_r34_own.md) and
[pricing](E7_setup_matrix/turn1_r34_own_setup.md):

| card | headlined (all games) | when held | priced vs the network's pick |
|:---|---:|---:|---:|
| Socialist Governments | 20.3% | 98% | 0.0 (always the pick) |
| Suez Crisis | 15.6% | 74% | −0.6 ± 0.1 |
| De Gaulle Leads France | 11.6% | 56% | −0.3 ± 0.2 |
| Vietnam Revolts | 9.4% | 47% | −0.5 ± 0.2 |
| Nasser | 7.4% | 36% | −0.1 ± 0.2 |
| Decolonization | 7.2% | 35% | −1.0 ± 0.2 |
| Captured Nazi Scientist | 6.9% | 33% | −1.1 ± 0.3 |
| Red Scare/Purge | 6.8% | 32% | −1.1 ± 0.2 |

The network's headline is at the priced best: the honest gain from re-picking is −0.24 ± 0.08 a
game. The headline barely depends on the US setup; it is the same against 4222 and 3322 on 97% of
deals.

## 3. When does the USSR not coup Iran on turn 1, and what does it do instead?

**This network coups Iran in AR1 in only 15.2% of games, so not couping Iran is the norm.** The
15.2% is 4.8% as straight Operations plus 10.4% after playing Duck and Cover's event first. Its
main line is **a coup in Italy: 44.9%** (37.1% as Operations, 7.8% after an event, mostly UN
Intervention).

What it plays in AR1:

| | share |
|:---|---:|
| coup in Italy (NATO, Nuclear Test Ban, Warsaw Pact, Comecon, NORAD; UN Intervention then a coup) | 44.9% |
| an event only (mostly Red Scare/Purge) | 14.5% |
| influence, without an event (Italy 3 is the most common spread) | 10.9% |
| coup in Iran | 15.2% |
| an event, then influence | 7.5% |
| a scoring card (Europe 3.7%, Asia 3.3%) | 6.7% |

**Priced** (96 worlds, at the decisions where both were legal):
* a coup in Iran instead of the network's coup target is −1.4 ± 0.2;
* Italy is −0.3 ± 0.1;
* every other target priced at more than 3 decisions is −6.7 or worse.

**What goes with an Iran coup:**
* **The card.** When the USSR plays Duck and Cover, 65% of those plays are an Iran coup
  (z = −43): the event goes first and its 3 Operations then coup Iran. Holding Duck and Cover at
  all makes an Iran coup 41% likely, against 8% without it.
* **A Middle East hand.** Holding Middle East Scoring, Nasser, Decolonization or De-Stalinization
  raises it, as do the USSR headlines Nasser and Decolonization.
* **A US Europe Scoring headline** raises it to 65%.

**What goes against it:** holding or headlining Socialist Governments (95.8% no Iran coup), Red
Scare/Purge, De Gaulle Leads France or NATO, and a US Middle East Scoring headline.

## 4. The US's standard AR1

Census of the network's own setup, 10,000 games. Influence is 71% of AR1s (63.5% straight, 7.3%
after an event). Coups are about 21%: Italy 7%, Iran 6%, Thailand 3%, Vietnam 2%.

**The most common spreads:**

| spread | share of influence AR1s |
|:---|---:|
| Pakistan 2, France 1, Malaysia 1 | 12.3% |
| Pakistan 2, Malaysia 1 | 6.3% |
| France 1, Malaysia 1, Pakistan 1 | 5.9% |
| Italy 3 | 4.5% |
| France 1, Italy 1, Malaysia 1, Pakistan 1 | 4.2% |

**The countries that get a point:**

| country | share | country | share |
|:---|---:|:---|---:|
| Pakistan | 68% | Iran | 10% |
| Malaysia | 59% | Egypt | 9% |
| France | 52% | West Germany | 5% |
| Italy | 37% | Lebanon | 4% |
| | | Afghanistan | 3% |

So Egypt is a minor option, and the core is Pakistan and Malaysia plus a Europe battleground.
Lebanon and Afghanistan are the other places, and the US also coups Thailand and Vietnam. The
pricing has its first point Pakistan −0.2, Malaysia −0.4, France −0.6, Lebanon and Afghanistan
−1.0, Italy −1.1, Egypt −1.9 and Iran −1.9, each against the network's pick.

**What shifts the spread:**
* **Italy** follows the USSR's AR1. The US never goes there after an Iran coup (0% against 39%)
  and goes more after an Italy coup (52%). It also goes more after a Socialist Governments
  headline (56%), and less after a De Gaulle (10%) or Suez (24%) headline.
* **France** depends on the Europe-scoring threat and the headlines. It is almost never a target
  once Europe Scoring is played on either side (4%), or after a Marshall Plan headline (24%). It
  is more likely after a Suez headline (72%) or when the US holds Defectors.
* **Pakistan** largely disappears after a USSR Asia Scoring headline (15%). It is almost certain
  after the USSR plays Europe Scoring in AR1 (96%), and more likely after a Nasser headline (85%).
* **Malaysia** is likelier when the US holds Nuclear Test Ban or NATO, and less likely when it
  holds Europe Scoring or Middle East Scoring.

**Two confirmed departures in the US's AR1:**
* a coup in Egypt instead of Iran, with NATO: +11.7 ± 2.9 to the end of the game;
* a first point in the UK instead of Italy, with Nuclear Test Ban: +15.0 ± 3.0.

## 5. The top US headlines

| card | headlined (all games) | when held | priced vs the network's pick |
|:---|---:|---:|---:|
| Defectors | 17.8% | 85% | −0.0 ± 0.1 |
| Red Scare/Purge | 16.4% | 78% | −1.0 ± 0.2 |
| Containment | 15.6% | 77% | −0.7 ± 0.2 |
| Marshall Plan | 11.2% | 56% | −1.2 ± 0.2 |
| Middle East Scoring | 9.0% | 43% | −0.3 ± 0.2 |
| East European Unrest | 7.7% | 38% | −1.5 ± 0.3 |
| Captured Nazi Scientist | 5.0% | 25% | −1.4 ± 0.3 |
| Europe Scoring | 4.5% | 21% | −3.4 ± 0.4 |

As for the USSR, the network's US headline is at the priced best (honest gain −0.33 ± 0.07).

## 6. 4222 against 3322

**Overall they tie:** 3322 − 4222 is +0.08 ± 0.09 for the US, pooled over the USSR's three
setups, and +0.17 ± 0.11 in this run.

**When each is better** (100,000 deals, pooled; held-minus-not, in US points):
* **The US holds Marshall Plan:** 3322 is better by +1.1 ± 0.2.
* **The USSR holds Asia Scoring** (−0.8 ± 0.2) **or Europe Scoring** (−0.7 ± 0.2): 4222 is
  better.
* **The USSR holds Blockade:** 3322 is better by +0.6 ± 0.2. West Germany 4 has more to lose.

**Is 4222 especially weak to De Gaulle or Suez? No.**
* De Gaulle in the USSR's hand: +0.3 ± 0.2.
* Suez in the USSR's hand: −0.2 ± 0.2.

**What the USSR does against each: nothing different.** It plays the same headline on 97% of the
same deals and the same AR1 action on 87.5%.
* The Italy coup is 46.7% against 4222 and 44.2% against 3322.
* The Iran coup is 13.5% and 12.7%.
* The only clear shift is Europe Scoring in AR1: 3.1% against 4222, 6.1% against 3322.

[Comparison](E7_setup_matrix/turn1_4222_vs_3322.md).

**What differs is the US's own AR1.** Each setup leaves one Europe battleground a point short of
control, and the US fills it.
* **4222:** France 2 is one short of its stability of 3. The US adds to France in 47% of games.
* **3322:** West Germany 3 is one short of its stability of 4. The US adds to West Germany in
  28%, and France falls to 15%.

The rest of the spread (Pakistan, Malaysia, Italy) is the same.

## Limitations

* **Greedy continuation.** Moves whose value lies in follow-ups the network does not play are
  undervalued.
* **The census is frequencies, not values.** The "what goes with" tables are single conditions,
  and they overlap: Duck and Cover in hand and a Middle East hand, for example.
* **Priced values are to the turn's end.** They are averaged over the decisions where an option
  was legal, against the network's pick. Only the 60 clearest departures were replayed to the end
  of the game.

## Replicate

```bash
M="_models/E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M.pt"
gh workflow run setup_matrix.yml --ref exp/setup-matrix -f model="$M" -f us_setups="us_r34 us_wg3_fr3_it2" \
  -f ussr_setups="ussr_r34 ussr_pol4_yug" -f games_per_cell=100000 -f shard_pairs=250
gh workflow run turn1_pricing.yml --ref exp/setup-matrix -f model="$M" -f games=500 -f worlds=96
gh workflow run turn1_pricing.yml --ref exp/setup-matrix -f model="$M" -f games=10000 -f worlds=2 \
  -f seed=200000 -f bank_top=1 -f pairs=8 [-f opening=us_r34+ussr_r34 | us_wg3_fr3_it2+ussr_r34]
python tools/turn1_census.py --rows rows/*.jsonl.gz --out census.md --json games.json
```

## 7. Yugoslavia 3: three European battlegrounds, domination, and East European Unrest / Independent Reds

The owner's case against Yugoslavia 3 is that it turns on East European Unrest and Independent
Reds. The case for it is that a controlled non-battleground saves two Ops toward Europe
domination once the USSR has three battlegrounds.

**Runs.** Three censuses on the same 10,000 deals, R34 playing its 4222 US setup and each USSR
setup:
* Yugoslavia 3, Poland 3: 38166082352;
* Poland 4, East Germany +1, Austria 1: 38173675553;
* Poland 4, East Germany +1, Yugoslavia 1: 38173682877.

Script: [turn1_europe.py](E7_setup_matrix/turn1_europe.py). "At the end" means at turn 1's last
recorded decision.

| | Yugoslavia 3, Poland 3 | Pol4/EG+1/Yug1 | Pol4/EG+1/Aus1 |
|:---|---:|---:|---:|
| USSR controls 3+ European battlegrounds, at the end | 40.6% | 38.8% | 36.9% |
| the same, at some point in turn 1 | 46.5% | 43.8% | 41.1% |
| USSR has Europe domination (or control), at the end | **33.7%** | 19.5% | 13.1% |
| the same, at some point in turn 1 | 66.0% | 23.5% | 15.1% |
| USSR controls Yugoslavia, at the end | 76.1% | 24.5% | 11.1% |
| Independent Reds' event fired on turn 1 | 9.4% | 10.6% | 11.0% |
| US influence in Yugoslavia, at the end | 9.0% | 8.1% | 1.2% |

* **The third battleground** is Italy about 80% of the time and France most of the rest.
* **Domination.** The extra non-battleground is what Yugoslavia 3 buys. Domination at the end is
  2.6 times as common as with Austria 1.
* **Independent Reds.** It fires about as often under every setup (9-11%), and almost always into
  Yugoslavia. Against Yugoslavia 3 it hands the US 3 influence there instead of 1.

**How each side plays the two cards** (games where that side holds the card on turn 1, Yugoslavia
3 setup; the human setups are within a few points):

| | Space | event first | Ops first (the event fires) | dumped with UN Intervention | kept into turn 2 |
|:---|---:|---:|---:|---:|---:|
| USSR, East European Unrest | 34% | 9% | 7% | 12% | 39% |
| USSR, Independent Reds | 28% | 11% | 11% | 5% | 45% |

* **The USSR.** Under Yugoslavia 3 it keeps Independent Reds into turn 2 more often (45% against
  32-35%) and fires it less (22% against 29-32%).
* **The US.** It mostly plays these cards for Operations without the event. East European Unrest
  is played for Ops 46-53% of the time, headlined 31-39%, and evented in an action round 3-4%.
  Independent Reds is played for Ops 60% of the time, headlined 19%, used to coup 17%, and almost
  never evented in an action round.
