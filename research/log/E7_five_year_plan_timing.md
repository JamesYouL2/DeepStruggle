# Five Year Plan timing as the USSR (2026-10-13): early plays are not a leak

**Question** (owner): the network "mostly learns to postpone Five Year Plan till AR7 as USSR, but sometimes still
plays it early" -- a sign the network lacks a signal like "USSR, early AR, Five Year Plan → hold"?

**Setup.** The R34 soup (`E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M`) in 4,000 greedy self-play games;
every USSR card choice with Five Year Plan selectable logged (33,684 decisions in 3,016 games). Early plays (before
the turn's last action round: AR6 in turns 1–3, AR7 after) priced by paired rollouts (`ai.search.turn_pricing.price`:
shared worlds from the USSR's view, the network greedy after the first move) against the network's best other card;
as a contrast, early *holds* priced against playing Five Year Plan instead. Turn end: the critic at the next turn,
128 worlds per option; game end: a greedy playout, 64 worlds per option. Δ is Five Year Plan now minus the other
card, in win probability, ± the standard error over positions. Scripts and outputs in the job's scratch directory
(`fyp/`).

## Frequency

Never headlined. Per turn with Five Year Plan in hand: played at the last AR 55.3%, held through the turn 31.9%,
played earlier 12.8%. Of all plays 18.8% come early (0.15 a game), almost all exactly one AR before the last. Early
plays track the hand: 11.1% of decisions when no other card is a US event against 0.8–2.6% with one or more, and
27.6% with one other card left.

## Pricing

| sample | turn end Δ | game end Δ |
|:---|---:|---:|
| early plays (619) | +0.39 ± 0.21 pp | 0.00 ± 0.36 pp |
| early holds (600), playing it instead | −3.34 ± 0.23 pp | −3.89 ± 0.37 pp |

* **The network separates the two kinds of position.** Where it plays early the play is worth as much as its best
  alternative; where it holds, playing would cost ~3.5 pp. Early plays with no US event among the other cards are
  slightly *better* than the alternative (+1.0 / +1.3 pp); confident ones (p > 0.9) clearly so (+1.9 / +2.2 pp).
* **The leak is at most ~0.15 pp a game.** Position by position the pricing calls ~25% of early plays worse and
  ~27% better; a switching rule chosen on half the worlds and measured on the other half gains +0.98 ± 0.13 pp
  (turn end) / +0.33 ± 0.13 pp (game end) per early play, times 0.155 early plays a game.
* The heads run's 6,400M snapshot gives the same picture (plays +0.52 / +0.47 pp, holds −3.30 / −4.10 pp).

**Reading.** This case is not evidence of a missing signal: the timing rule is learned, and its exceptions are
mostly correct. The values are "this network plays on from here"; a plan it would not follow is not credited.
