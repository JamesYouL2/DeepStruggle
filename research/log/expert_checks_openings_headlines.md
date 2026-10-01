# Expert checks on E6: the USSR opening, the US setup and a headline (2026-10-01)

**Who and why.** James You (fork `JamesYouL2/deepstruggle`) plays Twilight Struggle at a high
level. This log records where his reading of the game and the E6 net disagree, and what paired-deal
playouts said about each disagreement. Where a strong player's reading and the critic diverge, the
critic is the suspect, and the playouts are the judge.

**Model.** E6-06-44@soup_680-760, the `newest` export on Hugging Face, on both sides unless stated.

**Method.** Temperature 0, with the same deal seeds in every arm, so arms pair deal by deal. At
temperature 0 a net against itself plays a deal identically from either seat, so a mirror of N
games a side is N distinct deals; this was checked, 512 of 512 identical. Scores count a draw as ½,
and ± is one standard error of the paired difference. Fixed openings are scripted placements
(`opening:<name>:` entrants), and everything ran on the fork's CI. The tooling is on the fork's
branch [`exp/hungary-openings`](https://github.com/JamesYouL2/deepstruggle/tree/exp/hungary-openings):
`tools/lib/openings.py`, `ai/eval/opening_search.py`, `ai/eval/branch_oracle.py`,
`ai/eval/determinization_targets.py`, and a board snapshot per logged game in `batch_tournament.py`.
Engine fingerprint `5419a582ec07…`.

## 1. The USSR's Poland 3 / Hungary 3 opening

Every E6 model opens the USSR with Poland 3 / Hungary 3, leaving East Germany at its starting 3, so
three countries sit exactly at control. **The expert reading: Hungary is clearly suboptimal, not a
trap a stronger US exploits.** Poland 3 / Yugoslavia 3 has the same shape and dominates it in every
respect, because Yugoslavia also borders Italy (and Greece) where Hungary borders nothing the USSR
needs. Almost all strong players open East Germany 4 / Poland 4 plus 1 in Yugoslavia or Austria,
which also buffers both battlegrounds:

* East European Unrest and Independent Reds hit an unbuffered 3 / 3 / 3.
* John Paul II and Solidarity are worth more against Poland 3.
* Hungary gives up the Italy border that Yugoslavia or Austria would give.

So the question for the net is not whether Hungary is worse, but why it never learns that.

In all of these runs the US setup was fixed at WG 3 / France 3 / Italy 2 / Iran 2.

### The board effects are real

From the 4,096-deal mirrors (fork runs `36799548667`, `36799553935`, `36799559256`, `36807182739`,
`36807187343`):

| USSR opening | USSR score | − Hungary, paired | USSR controls Italy, T4 | lost EG or Poland by T2 | holds Poland, T8 |
|:---|---:|---:|---:|---:|---:|
| Pol 3 / Hun 3 | 51.5% | — | 11.9% | 8.5% | 74.4% |
| Pol 3 / Yug 3 | 52.7% | +1.2 ± 1.0 | 22.6% | 8.2% | 74.2% |
| EG 4 / Pol 4 / Yug 1 (the human opening) | 52.2% | +0.8 ± 1.0 | 22.5% | 0.0% | 82.3% |
| EG 4 / Pol 4 / Aut 1 | 51.6% | +0.2 ± 1.0 | 22.5% | 0.0% | 80.8% |
| EG 4 / Pol 5 | 51.2% | −0.3 ± 1.0 | 11.6% | 0.0% | 88.9% |

* **An Italy-adjacent opening doubles USSR control of Italy by turn 4**, and the East Germany 4
  buffer removes the turn-1 loss of a battleground, 8.5% → 0.
* **The result moves by about a point.** At E6 strength the five openings are within 1.5 points.
  Yugoslavia, the dominating opening, leads Hungary by 1.2 ± 1.0, the right sign at a size the
  board differences do not explain. **The net, on either side, does not convert the advantages a
  strong player plays these openings for.**

### The gap grows with the opponent's strength

Search-64 (`search:…:64:determinize`) plays the US and the net plays the USSR, 512 deals per
opening, compared deal by deal with the net playing the US. The US headlines first, so these
games were never affected by the determinization leak fixed in #4.

| USSR opening | vs search-64 US | vs net US | cost of the stronger US |
|:---|---:|---:|---:|
| Pol 3 / Hun 3 | 41.1% | 47.9% | −6.7 ± 2.6 |
| Pol 3 / Yug 3 | 44.0% | 49.3% | −5.3 ± 2.8 |
| EG 4 / Pol 4 / Aut 1 | 43.1% | 48.0% | −5.0 ± 2.4 |
| Pol 3 / Cz 3 | 43.2% | 45.4% | −2.2 ± 2.5 |

Hungary is punished 1.5 ± 3.8 points more than Yugoslavia, 1.8 ± 3.6 more than Austria, and
4.5 ± 3.6 more than Czechoslovakia. **Every comparison has the expert's sign; none is
significant on its own.** A better player converts more of a structural edge, so a widening gap is
what a dominated opening should show. The net is simply not yet strong enough for the difference to
be large.

### The critic carries a fixed bonus for Hungary

A determinized search was run from the first decision after setup: 400 deals, 4 worlds per budget
(fork run `36807155441`). USSR values are in win-probability points; the alternatives are a paired
difference from Hungary (± ≈ 0.2).

| reading | Pol 3 / Hun 3 | Pol 3 / Yug 3 | EG 4 / Pol 4 / Aut 1 | EG 4 / Pol 4 / Yug 1 | EG 4 / Pol 5 |
|:---|---:|---:|---:|---:|---:|
| critic (0 sims) | +2.5 | −3.8 | −5.4 | −2.6 | −2.1 |
| search 1,024 | +5.0 | −2.9 | −3.4 | −1.5 | −1.0 |
| search 8,192 | +5.7 | −2.8 | −3.0 | −1.4 | −0.4 |
| playouts (4,096 deals) | 51.5% | **+1.2** | +0.2 | +0.8 | −0.3 |

* **The critic rates Hungary 2–5 points above every alternative, with zero spread across deals.**
  That is a fixed per-opening offset, not a reading of the position: the trained opening's habit.
* **8,192 simulations do not remove it.** Search inherits the critic's leaves, so it is precise
  and biased here. The playouts put Yugoslavia ahead; search at 8,192 still puts it 2.8 behind.
* The same offset appears with the net's own US setup (fork run `36807180884`). That US answers
  WG 4 / France 3 / Italy 2 / Iran 1 in all 2,000 setups, whatever the USSR opened: **the US setup
  never reacts to the USSR's.**
* Setup credit learns from this critic, so the offset is a candidate mechanism for the lock in
  [`E5_11_setup_lock_and_critic_views.md`](E5_11_setup_lock_and_critic_views.md). E6-04 re-opened
  setup entropy, and seed 44 came back to Hungary
  ([`E6_04_league_seed44.md`](E6_04_league_seed44.md)).

## 2. The US setup: Iran 2

[`E4_goal_probes.md`](E4_goal_probes.md) found humans take US Iran ≥ 2 96% of the time, and that
checkpoints fail that target most often. E6's US setup is WG 4 / France 3 / Italy 2, with Iran at 1.
Three setups a strong player uses were tried against it: 4,096 deals each, the net on both sides,
and the USSR setting up freely (it opened Pol 3 / Hun 3). Fork runs `36850912031` (control),
`36850892013`, `36850898583`, `36850905576`.

| US setup | US score | − E6's own, paired | US holds WG, T4 | France, T4 | Italy, T4 |
|:---|---:|---:|---:|---:|---:|
| E6's own: WG 4 / Fr 3 / It 2 / Ir 1 | 47.4% | — | 86% | 72% | 86% |
| WG 4 / It 4 / Ir 2 | 48.8% | +1.4 ± 1.0 | 85% | 62% | 98% |
| WG 3 / Fr 3 / It 2 / Ir 2 | 48.5% | +1.2 ± 1.0 | 83% | 75% | 85% |
| **WG 4 / Fr 2 / It 2 / Ir 2** | **49.4%** | **+2.1 ± 1.0** | 86% | 68% | 82% |

* **All three expert setups beat E6's own**, by 1.2–2.1 points. The best is at about 2 standard
  errors, and what they share is the second point in Iran.
* Among the three, the differences are within noise. Each trades France against Italy differently,
  as the turn-4 columns show.

## 3. A headline the net gets wrong: the turn-10 USSR headline

This position was found by the blind-spot probe: Dirichlet root noise in determinized searches over
10,000 sampled decisions. A candidate is a move search prefers whose prior is under 5%. One of
these was confirmed in 10,000 decisions.

The position: the USSR headline on turn 10, VP −4, DEFCON 3. The net headlines **Missile Envy at
0.997**. The expert reading was that Missile Envy is strong, but South African Unrest and Marine
Barracks Bombing are stronger here. Each headline was checked with 256 paired playouts, with the
hidden cards redealt per pair:

| USSR headline | − Missile Envy |
|:---|---:|
| South African Unrest | **+19.9 ± 3.6** |
| Marine Barracks Bombing | +15.6 ± 3.6 |
| The Iron Lady | +9.0 ± 3.9 |
| Reagan Bombs Libya | +3.1 |
| Sadat | −1.0 |
| Chernobyl | −3.1 |
| Arms Race | −5.1 |
| Voice of America | −13.9 |

Missile Envy itself scores 49.2%.

* **The expert's ranking is the playouts' ranking** for the top two.
* The first measurement read +34. That was the determinization leak fixed in #4: every sampled world
  resolved the US's real headline, Soviets Shoot Down KAL-007. The leak was found because the +34
  contradicted the expert reading.
* **Headlines are a guessing game**: the best headline depends on what the opponent headlines at the
  same time, so the right answer can be a mix. The net plays one card at 0.997. A determinized
  search cannot find a mix either: it assumes it may play differently in each sampled world. Most
  of the game is calculation; headlines are the place where a near-deterministic policy is most
  exposed.

## 4. Search mostly hands the net back its own preference

The same probe, with no root noise and the training searcher's settings, 4,000 positions (fork run
`36797382596`):

| net's top probability | positions | search changes the top move |
|:---|---:|---:|
| < 0.5 | 312 | 8.3% |
| 0.5–0.8 | 958 | 2.4% |
| 0.8–0.95 | 764 | 0.0% |
| ≥ 0.95 | 1,966 | 0.0% |

* **A 64-simulation search target never overturns a move the net rates above 0.8**, which is 68% of
  decisions. Distilling it can therefore not break a locked decision such as the Hungary opening.
* Together with §1's fixed critic bonus, this supports the order in
  [`../plans/P29_big_bets_from_scratch.md`](../plans/P29_big_bets_from_scratch.md), where bet 5
  (search-driven training) waits for a better critic. Deeper search reached the critic's error, not
  the playouts. And at 64 simulations the target is the net's own preference.
* Blind spots that search can fix exist but are rare: one confirmed in 10,000 decisions (§3).

## Open expert hypotheses, not yet tested

* **Realignment targeting.** The expert rule is to realign either with several +1 targets or with a
  single +2 target, and often well below the modifiers the net waits for.
  [`E5_11_selfplay_review.md`](E5_11_selfplay_review.md) found forced realignments lose 3–5 points,
  but its forced branch let the policy choose its own targets. A census of the net modifier at
  each realignment roll, and an oracle branch that follows the expert rule, would separate the
  decision to realign from the targeting.
* **Mixed headline strategies.** Whether a US that headlines the obvious card (KAL-007 when held)
  can be exploited by the USSR's choice, measured as exploitability, not as a single best response.

## What this does not say

* Every forced opening is played by a net that practised Hungary, so a structural gap is if anything
  understated. Forcing the opening into training would remove that.
* One model, an SWA soup. These are model-against-model results: they measure what the net does
  with an opening, not what a strong human would.
* §1's stronger-US rung is search-64 on the same net, about +36 Elo over the net, not an
  independent strong player.
