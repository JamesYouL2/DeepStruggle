# An expert review of the E7 soup: strengths, card-play leaks, and a forced-win fix (2026-10-02)

**Who and why.** James You (fork `JamesYouL2/DeepStruggle`) is a top-20-to-50 Twilight Struggle
player who played the current best model and turned those observations into measurements:

* the **doctrine census** -- how often the model departs from a strong player's rule;
* **paired playouts by the model itself** -- what each departure costs. The cards the mover cannot
  see are redealt for each pair, every branch gets the same dice, and the model plays both sides;
* comparisons with the human ts-replayer corpus.

**Model.** `shallow_E7-02+03+04+05_1200M.onnx`, the `newest` export. Engine fingerprint
`5419a582ec07…`. Everything ran on the fork's CI, and the run ids are at the end. The tools are on
the fork branch
[`exp/hungary-openings`](https://github.com/JamesYouL2/DeepStruggle/tree/exp/hungary-openings),
under `ai/eval/`.

**The standing caveat.** A playout verdict values a move given how *this* model plays afterwards.
A move whose payoff lies in a follow-up the model does not know reads worse than it is. That makes
the weaknesses below lower bounds.

## Summary

* **Board judgment is the model's strength, and it beats strong humans.** Its Ops placement holds
  up against its own playouts at every level tested:
  * 24,000 sampled decisions show no regret;
  * forcing an influence play into any one region costs 1.7-3.7 points;
  * declining a live scoring battleground is right.

  Where the human corpus allocates differently -- about twice the Middle East, little Italy or
  Asian battlegrounds, more scattered non-battlegrounds -- the reviewer's verdict is that the
  model is right and the humans are wrong.
* **Replies and defence are fine.** Search-64 plays the same action round as the net 91% of the
  time and gains +0.1 ± 0.2 where they differ.
* **The critic is calibrated.** Fitted temperature 1.02 over 1.9M readings. But **the two seats
  disagree** by about 4 points at every hand-over, which is noise for search.
* **The weakness is card play, not the board:**
  * **decisive wins not taken.** Wargames is declined 429 of 430 times at a winning lead, about 1.5
    points a game, and `safety.py` did not see it -- fixed in this PR. OPEC and Alliance for Progress
    are declined about half the time when the event would reach 20 VP, which `safety.py` does see;
  * **immediate VP and space from events undervalued, at specific spots.** One Small Step one box
    behind, Soviets Shoot Down KAL-007 with South Korea US-controlled, Star Wars ahead in space,
    Glasnost with The Reformer in play and Special Relationship with NATO in effect: the model events
    each 0-1% of the time at those spots (Star Wars 26% overall), and its own playouts value the event
    at +2 to +6 points a play. Humans make 1.6-3.3 more VP
    a game from events. Generic rules ("event at k VP", "event when behind in space") do *not* hold;
    the spots do;
  * **event timing against the hand and the rounds left.** Five Year Plan and Aldrich Ames; the net
    is never told how many rounds it has left;
  * **rarely reached decisions decided poorly.** The choices inside events, such as Aldrich Ames'
    discard.
* **What it suggests for training.** Card play is the target, and self-play Elo cannot see these
  leaks, because both seats share them. See the last section.

## The fix in this PR: wins that take more than one decision

`classify_legal_actions` never saw a Wargames win at all. Its forced probe stops at the first
decision with more than one option, so it stopped at the event's branch; and the hand-written rule
for that branch compared flat action indices against 0 while the branch sits at
`ActionEncoder.BRANCH_OFFSET`, so it never fired -- from the day it was written (`6669cf8`). At the
decision that matters, play the card for its event or for Ops, the event read "normal". Two things
followed:

* the decisive probe never counted the model declining this win, so
  [`agent_deficiencies_and_decisiveness.md`](agent_deficiencies_and_decisiveness.md)'s "declining a
  forced win is usually free" never covered it;
* any safety layer built on the classifier never took it.

Wargames is one case of a general gap, so the fix is general. An action is now a win when a line
from it ends the game for the mover using only the mover's own choices and single-option steps --
no opponent decision, no die, no hidden draw, nothing past the current card's play. That covers,
with no card list:

* Wargames at DEFCON 2 with a lead over 6, at the card in hand, its event and its branch;
* an event from hand whose VP reaches 20;
* Star Wars eventing Wargames or such an event from the discard -- the mover's own Star Wars, or the
  pick when the opponent plays it for Ops;
* the card Grain Sales drew, whoever played Grain Sales. Choosing Grain Sales is not a win, since
  what it draws is chance -- unless the USSR holds a single card, which the engine draws without
  the RNG.

The card Grain Sales drew is also judged as itself now for losses: the classifier read the card at a
play-mode decision as Grain Sales, so a drawn DEFCON card's loss went unseen, and the DEFCON rules
assumed the decider was phasing, when in the USSR's round taking DEFCON to 1 wins for the US.

`tests/training/test_safety_decisive.py` pins each case for both sides, that following the labels
reaches the win, and the game's other instant endings, which were already right: a battleground coup
at DEFCON 2, by Ops or by an event's free coup, and not under Nuclear Subs; a scoring card reaching
20 VP; Europe Scoring with Europe controlled.

This changes what the decisive probe reports from now on. Forced wins it never saw are counted, and
each chance to win counts once however many decisions it takes -- Wargames from hand was three
"win" decisions, one chance -- where a decision at which every action wins is no chance at all. The
metrics carry `decisive_probe_version` 2 so the rate is not read against earlier runs.

## The weaknesses, with the playout evidence

| leak | how often | cost (the model's own paired playouts) | evidence |
|:---|:---|:---|:---|
| **Wargames declined at a winning lead** | 429 of 430 plays; ~11% of games | 14.3 ± 2.6 pts per decline (it still wins 85.7%): **~1.5 pts a game** | census `36943954436`; 60 positions × 32 pairs |
| **Game-winning OPEC / Alliance for Progress declined** | declined 31 of 58 / 19 of 27 where `safety.py` labels the event a win | +12.8 / +6.7 pts per decline (the model still wins 87% / 93%); this is most of what OPEC and Alliance for Progress gain at 5+ VP | `37034959708` |
| **Star Wars not evented** | evented 0% (census, ahead in space; card-rules bank), 26% overall | +3.8 ± 1.2 over 127 positions (+9.4 ± 2.9 in close games); +1.6 ± 0.6 ahead in space | `37026844506`, `36945382157`, `36964635572` |
| **Special Relationship not evented with NATO in effect** | evented 0% of 320 positions | **+2.1 ± 0.5** where it scores its 2 VP (301 positions); every Ops mode is worse | 320 positions × 64 pairs, `37034959708` |
| **Soviets Shoot Down KAL-007 not evented with South Korea US-controlled** | evented 0% of 305 positions at DEFCON 3+ | **+4.2 ± 0.6** (+7.3 ± 1.2 in close games): DEFCON 3 +4.7 ± 0.8, DEFCON 4 +4.2 ± 1.0, DEFCON 5 +1.3 ± 1.2; its Ops play is not even its best Ops mode (influence +0.9) | 305 positions × 64 pairs, `37045332746` |
| **Glasnost not evented with The Reformer in play** | evented 1% of 320 positions | **+2.3 ± 0.5** (+5.9 ± 1.3 in close games): DEFCON 2 +1.5 ± 0.7, DEFCON 3 +4.4 ± 0.9 | 320 positions × 64 pairs, `37045332746` |
| **One Small Step not evented one box behind** | evented 0% of 1,489 positions | US: +2.8 ± 0.4 at 1 vs 2, **+5.7 ± 0.5** at 3 vs 4; USSR: −0.2 ± 0.5 at 1 vs 2, +2.9 ± 0.5 at 3 vs 4 | 320-496 positions a side and spot × 64 pairs, `37037507768`, `37037511438` |
| **Five Year Plan timing (USSR)** | last round 55%; two or more other cards in hand 38% | -- | census `36953663726` |
| **Choices inside events** | -- | Aldrich Ames: best discard −0.8 against Ops, the model's own discard −5.5 (40 positions) | local, 32 pairs |

### Notes on each

**Immediate VP.** A VP ledger credits every VP change to the step that made it: 4,001 self-play
games against 175 complete corpus games (`36966918630`). Humans earn more from events, the model
more from the board:

| VP per game from | model | humans |
|:---|---:|---:|
| events, US | 7.8 | 11.1 |
| events, USSR | 7.3 | 8.9 |
| scoring cards | about 1 more for each side | -- |

The largest event gaps are OPEC, Arms Race, Wargames, Duck and Cover and Alliance for Progress.
Forcing the events humans favour, but whose value is delayed (John Paul II, Bear Trap, Missile
Envy), reads *negative* by the model's playouts. That is the standing caveat: the playouts see only
the immediate cases.

**One Small Step.** One box behind, the event jumps the mover two boxes, past the opponent and into
the next VP box: from 1 vs 2 into Man in Space, from 3 vs 4 into Lunar Orbit. Positions were taken
from the model's own self-play wherever the spot arose, and every mode was played out:

| spot | side | positions | model events | event − model | close games (model 25-75%) | best other mode − model |
|:---|:---|---:|---:|---:|---:|---:|
| 1 vs 2 | US | 320 | 0% | +2.8 ± 0.4 | +3.4 ± 0.8 | −0.6 (influence) |
| 1 vs 2 | USSR | 320 | 0% | −0.2 ± 0.5 | −1.0 ± 1.0 | −0.4 (influence) |
| 3 vs 4 | US | 353 | 0% | **+5.7 ± 0.5** | **+12.6 ± 1.4** | −0.4 (influence) |
| 3 vs 4 | USSR | 496 | 0% | **+2.9 ± 0.5** | +5.9 ± 1.4 | 0.0 (influence) |

The value is the VP swing of jumping past the opponent, not being behind as such: over all of the
model's One Small Step plays while behind in space, the event reads −0.5 ± 1.1 for the US and
−3.5 ± 1.0 for the USSR (`37026844506`). A rule that names the spot is worth a lot, while "event
when behind" is not -- the same lesson as every other card here.

**Timing.**

* The misses fall one round early, where the count of action rounds changes. In turns 1-3, Five
  Year Plan clusters on round 6, which is right. From turn 4, 788 plays fall in round 6 against
  1,316 in round 7.
* The net sees the turn, the action round and both hand sizes. It is never told the rounds left
  for each side, or whether the opponent replies before the turn ends.
* A two- or three-float "rounds left / surplus cards" block, built with the view-spec mechanism of
  C4, would cost nothing to existing checkpoints.

**Choices inside events** were the weakest decision type everywhere:

* the Aldrich Ames discard;
* where De-Stalinization removes influence -- humans take it off Finland or Romania, the model off
  Syria;
* Marshall Plan's placements;
* a Star Wars retrieval of a DEFCON card at DEFCON 2, which the playouts above guard against.

On turn 1, the humans' choices inside events beat the model's by +0.6 ± 0.3 over 754 disagreements
(`36963933063`). These are decisions self-play rarely reaches -- the model never events Aldrich
Ames in an action round.

**Checked and not a leak.** The US setup bonus into South Korea (23% of games): a 4,096-deal
bake-off measures +0.2 ± 2.1 (`36970757999`, `36970763934`). Kitchen Debates where its event
scores 2 VP: the model events it 5% of the time and the event ties its choice, +0.1 ± 0.3 over 400
positions, whether or not the US still needs Military Ops (`37041009744`). OPEC and Alliance for
Progress at 5+ VP, short of a game-winning haul, are break-even too (+0.4 ± 0.5, +0.0 ± 0.5; at
exactly 5 VP +0.7 ± 0.5, −0.2 ± 0.6): the +2.4 and +1.1 the choice oracle measured at 5+ VP
(`36945382157`) come from the larger and the game-winning hauls (`37034959708`). Captured Nazi
Scientist into a VP box: the model already events it 75% (US) and 86% (USSR) of the time, and the
event ties its choice (−0.2 ± 0.2, −0.2 ± 0.1; `37045332746`).

## Why training may plateau, and what to try

The playouts see 2-15 point gaps at specific decisions. RL credits one sampled move, from one
game's ±1, against a critic whose seats disagree by about 4 points. A move the policy gives
p ≈ 0.002 (Wargames for the win) is almost never sampled at all.

Candidates, cheapest first. All but item 3 are recipe changes that touch neither the engine nor
the observation; item 3's rounds-left block is a new observation block and needs the owner's
approval:

1. **More exploration at play-mode decisions and choices inside events only.** A temperature or
   entropy bonus there, so events the policy dislikes still get tried.
2. **A small potential-based VP shaping**, Φ = c × VP lead. It does not change the optimal policy,
   and it makes immediate VP visible at the step it happens, beside `--decisiveness-turns`.
3. **The rounds-left block** above, and **C2** (the card-event target), which teaches the trunk
   what events do.
4. **Paired-branch advantages at play-mode decisions.** Play each legal mode out with common dice
   and redeals, and use the paired differences as the policy-gradient signal for those actions.
   This gives a signal for moves the policy would never sample.
5. **Distil the rules a position bank confirms**, as a small auxiliary loss refreshed every ~200M
   steps: the card-specific spots above and decisive moves. Not generic thresholds -- "event when
   its immediate VP is k or more" loses by the bank (−3.4 ± 0.2 over 2,394 decisions).

Judge each on the census card-play rates and the events-VP gap, not on Elo alone.

**Request.** Could you share `_soups/shallow_E7-02+03+04+05_1200M.pt`? It is the checkpoint the
ONNX metadata names, sha256 `55a26630519e78991a90317ea071f395e61d53e01e641cf99224ef3426a733fc`.
James would like to test items 4-5 as a fine-tune from the soup. The net is 1.27M parameters, so a
CPU is enough, and the labels come from free CI.

## Reproduction

All runs are on the fork's CI with the model and engine above. Every report carries a provenance
block.

* **Doctrine census.** `36943954436`; hand size at the discard events `36953663726`.
* **Choice oracle.** Event and headline questions `36941086943`, `36945382157`; the cards humans
  event most, forced `36964635572`.
* **Playout audit.** 24,000 decisions: `36953614691`, with parts redone after a fix in
  `36955589646` and `36958736889`.
* **Reply probe.** `36951383335`.
* **Live scoring battlegrounds.** `36956833709`.
* **Region weight.** `37002909075`.
* **Human disagreements.** All turns `36958653207`; turn 1 `36963933063`.
* **VP ledger and calibration.** `36966918630`.
* **US setup bake-off.** `36970757999`, `36970763934`, `36970769645`.
* **Position banks** (every legal mode played out from the model's own positions). Card event
  rules over the 36 cards humans event most against the model `37026844506`; One Small Step one box
  behind `37037507768` (1 vs 2) and `37037511438` (3 vs 4); Special Relationship with NATO, OPEC and
  Alliance for Progress at 5+ VP `37034959708`; Kitchen Debates where it scores `37041009744`;
  Captured Nazi Scientist into a VP box, KAL-007 with South Korea US-controlled and Glasnost with The
  Reformer `37045332746`.
* **Local measurements, the same paired method.** The Wargames cost (60 positions × 32 pairs) and
  the Aldrich Ames discard (40 positions × 32 pairs).
