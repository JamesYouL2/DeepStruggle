# An expert review of the E7 soup: strengths, card-play leaks, and a forced-win fix (2026-10-02)

**Who and why.** James You (fork `JamesYouL2/DeepStruggle`) is a top-20-to-50 Twilight Struggle
player. He played the current best model and turned his observations into measurements:

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
    points a game, and `safety.py` did not see it -- fixed in this PR;
  * **immediate VP from events undervalued.** OPEC and Alliance for Progress at 5+ VP, Star Wars
    when ahead. Humans make 1.6-3.3 more VP a game from events;
  * **event timing against the hand and the rounds left.** Five Year Plan and Aldrich Ames; the net
    is never told how many rounds it has left;
  * **rarely reached decisions decided poorly.** The choices inside events, such as Aldrich Ames'
    discard.
* **What it suggests for training.** Card play is the target, and self-play Elo cannot see these
  leaks, because both seats share them. See the last section.

## The fix in this PR: Wargames at the play-mode decision

`classify_legal_actions` recognised Wargames' win only at its `CHOOSE_BRANCH` node -- "give 6 VP
and end the game", the *second* step. At the decision that matters, play the card for its event or
for Ops, the event read "normal". Two things followed:

* the decisive probe never counted the model declining this win, so
  [`agent_deficiencies_and_decisiveness.md`](agent_deficiencies_and_decisiveness.md)'s "declining a
  forced win is usually free" never covered it;
* any safety layer built on the classifier never took it.

Now, at DEFCON 2 with a lead over 6, the event is a win. At 6 or less it is not a loss, since the
branch can still decline to end the game. `tests/training/test_safety_decisive.py` pins both, beside the game's other instant endings (a battleground coup at DEFCON 2 by Ops or by an event's free coup, and not under Nuclear Subs; a scoring card reaching 20 VP; Europe Scoring with Europe controlled), which were already right. It
also confirms the engine side: the event, then branch 0, ends the game for the mover.

This changes what the decisive probe reports from now on: forced wins it never saw are counted.

## The weaknesses, with the playout evidence

| leak | how often | cost (the model's own paired playouts) | evidence |
|:---|:---|:---|:---|
| **Wargames declined at a winning lead** | 429 of 430 plays; ~11% of games | 14.3 ± 2.6 pts per decline (it still wins 85.7%): **~1.5 pts a game** | census `36943954436`; 60 positions × 32 pairs |
| **OPEC not evented at 5+ VP** | evented 24% | event +2.4 ± 0.3 | 363 positions × 128 pairs, `36945382157` |
| **Star Wars not evented when ahead in space** | evented 0% (census), 26% overall | +1.6 ± 0.6; +2.0 ± 0.7 where it declines | `36945382157`, `36964635572` |
| **Alliance for Progress not evented at 5+ VP** | evented 22% | +1.1 ± 0.3 | 512 positions × 128 pairs, `36945382157` |
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
bake-off measures +0.2 ± 2.1 (`36970757999`, `36970763934`).

## Why training may plateau, and what to try

The playouts see 2-15 point gaps at specific decisions. RL credits one sampled move, from one
game's ±1, against a critic whose seats disagree by about 4 points. A move the policy gives
p ≈ 0.002 (Wargames for the win) is almost never sampled at all.

Candidates, cheapest first. All are recipe changes; none touches the engine or the observation:

1. **More exploration at play-mode decisions and choices inside events only.** A temperature or
   entropy bonus there, so events the policy dislikes still get tried.
2. **A small potential-based VP shaping**, Φ = c × VP lead. It does not change the optimal policy,
   and it makes immediate VP visible at the step it happens, beside `--decisiveness-turns`.
3. **The rounds-left block** above, and **C2** (the card-event target), which teaches the trunk
   what events do.
4. **Paired-branch advantages at play-mode decisions.** Play each legal mode out with common dice
   and redeals, and use the paired differences as the policy-gradient signal for those actions.
   This gives a signal for moves the policy would never sample.
5. **Distil the pooled rules a position bank confirms**, as a small auxiliary loss refreshed every
   ~200M steps. Event when its immediate VP is k or more; decisive moves.

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
* **Local measurements, the same paired method.** The Wargames cost (60 positions × 32 pairs) and
  the Aldrich Ames discard (40 positions × 32 pairs).
