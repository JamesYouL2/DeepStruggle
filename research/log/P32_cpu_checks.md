# P32 CPU checks -- the forced-exit trap (4a), Wargames at a lead of 6 (2), Chernobyl off Europe (3)

**Plan:** [`../plans/P32_teacher_and_paired_credit.md`](../plans/P32_teacher_and_paired_credit.md).
**Network:** the heads soup `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M`. Probes and outputs in
`data/reports/` (`trap_probe`, `wargames_lead6`, `chernobyl_regions`, `.py` / `.out`).

## 4a. The forced-exit DEFCON trap: the critic sees it; it explains a fifth of DEFCON-1 losses

2,000 self-play games at temperature 1. A side is **trapped** at its action-round card choice at
DEFCON 2 when it has more rounds left to fill than safe plays: non-suicide cards (a playable China
Card counts), UN Intervention paired with a suicide card, and space attempts left for suicide
cards with the Ops for the next box. Suicide cards are safety.py's no-choice degraders of the
opponent (We Will Bury You for the US; Duck and Cover and KAL-007 for the USSR) plus Lone Gunman
for the US. The critic is the decider's own `v_win`.

| side | trap entries | then lost by DEFCON 1 | outcome mean | critic at entry | at the previous card choice | critic − outcome |
|:---|---:|---:|---:|---:|---:|---:|
| US | 56 | 22 (39%) | −0.32 | −0.40 | −0.28 | −0.08 |
| USSR | 21 | 6 (29%) | −0.14 | −0.14 | −0.06 | 0.00 |

* **The critic sees the trap.** It drops at entry (US −0.28 → −0.40) to where the outcomes land,
  if anything slightly pessimistic. P32's decision rule -- an error at entry above 0.1 sends 4b,
  the forced-exit auxiliary head -- is not met: the critic knows, and what fails is the policy.
* **It is a minority of the DEFCON-1 losses.** DEFCON 1 ends 142 of the 2,000 games (7.1%); 29 of
  them follow a trap entry of the loser (28 provoked, 1 own), 86 are provoked with no trap and 27
  are self-inflicted. The provoked-without-trap majority is where the next probe goes: the coups
  handed over by Lone Gunman / Ortega / CIA Created at DEFCON 2 (the Cuban Missile Crisis probe
  found the same policy, saturated at the coup) are one candidate.

**The attacker's side** (`trap_attacker`, 6,000 games at temperature 1): at every Aldrich Ames
Remix discard at DEFCON 2 -- the USSR picks one of the US's cards -- each option is applied and the US
tested for the trap. Of 436 such discards only **8** offered a trapping option; the USSR took one in
**all 8** (trapping options were 73% of the choices there), and its critic values the trapping
discards **+0.948** against **+0.759** for the rest (+0.19). A small sample, but the attacker's
critic sees the trap as the defender's does: with both sides seeing it, section 4 of P32 closes, and
what remains of it is the policy's use of what the critic knows (4c, inside B4').

## 2. Wargames at a lead of 6: taking the draw is right

Positions from 12,000 self-play games at temperature 1 where the Wargames branch is reached at
DEFCON 2 with the brancher 6 ahead -- ending the game is a draw. Paired playouts, 64 pairs, the
model's greedy play on both sides:

* 46 positions: ending scores 0.500, playing on **0.288** -- playing on minus ending
  **−0.212 ± 0.045**. Playing on is better in 10 of 46.
* The network ends the game in 36 of 46; its choice agrees with the playouts in 32.

By P32's criterion -- the draw is right when the position's win probability is below 50% -- this is
not a leak. (3,000 games, 16 positions, gave −0.131 ± 0.078, the same answer.)

## 3. Chernobyl's region off Europe: not a Europe-only habit, no leak found

Positions from 1,200 self-play games at temperature 1: the US's region choice whenever Chernobyl's
event fires, every legal region scored by paired playouts (32 pairs).

| fired during | choices | the network chooses | Europe minus the chosen region, where it chose another |
|:---|---:|:---|---:|
| a US play | 69 | Europe 62, Central America 4, Asia 2, South America 1 | −0.011 ± 0.032 (7) |
| a USSR play | 240 | Central America 87, Asia 66, South America 47, Europe 20, Africa 17, Middle East 3 | **−0.019 ± 0.008** (220) |

Forcing Europe where the network chose another region is *worse*, by about 2 points. Ranking each
position's six regions by their 32-pair means names Europe best in 106 of the 240 -- a maximum over
six noisy estimates, biased upward; the paired comparison against one fixed alternative is not.
Choices beaten by more than 2 SE: 19 of 240, about what chance gives with five alternatives each.
