# The USSR's Poland 3 / Hungary 3 opening (2026-10-01, interim)

**Question.** Every E6 model opens the USSR with Poland 3 / Hungary 3: East Germany left at its
starting 3, so all three sit exactly at control. The owner's case against it: East European Unrest
and Independent Reds hit an unbuffered 3/3/3, John Paul II and Solidarity are worth more against
Poland 3, and Hungary gives up the Italy border that Yugoslavia (or Austria) would give. And the
hypothesis: Hungary was chosen against weak opponents early in training and never revisited.

**Status: interim.** The net-only results below are complete. The search-64 rung (a US stronger
than the model), an unforced search run and three 4,096-deal mirrors are still running on
branch `exp/hungary-openings`; this note is updated when they land.

## Setup

* Openings in [`../../tools/lib/openings.py`](../../tools/lib/openings.py). US always
  **WG 3 / France 3 / Italy 2 / Iran 2** (the E6 US setup). USSR one of: `wfi_ph` Poland 3 /
  Hungary 3, `wfi_py` Poland 3 / Yugoslavia 3, `wfi_pc` Poland 3 / Czechoslovakia 3,
  `wfi_eg4_pol4_aut1` East Germany 4 / Poland 4 / Austria 1, `wfi_eg4_pol4_yug1` East Germany 4 /
  Poland 4 / Yugoslavia 1 (the human opening).
* USSR player: **E6-06-44@soup_680-760** (`newest` on Hugging Face at the time). Temperature 0,
  512 deals per cell, same deal seeds in every opening, so cells pair by deal.
* Each logged game now carries board snapshots (after setup, start of turns 2/4/8, end) in
  `games.jsonl` ([`../../tools/lib/batch_tournament.py`](../../tools/lib/batch_tournament.py),
  `_board_snapshot`).
* Engine fingerprint `5419a582ec07…`, commits `e462335` (mirrors) and `05fac04` (ladder) on `exp/hungary-openings`.
* At temperature 0 a net against itself plays each deal identically from either seat, so a mirror
  of 512 games a side is 512 distinct deals, not 1,024 (checked: 512 of 512 identical).

## Result 1: the board effects are real

E6 against itself (runs `36795604942`, `36795610102`, `36795615081`, `36795620011`):

| USSR opening | USSR wins | ≥3 Europe BGs, start T2 | …start T4 | USSR controls Italy, T4 | lost EG or Poland by T2 | holds Poland, T8 |
|:---|---:|---:|---:|---:|---:|---:|
| Pol 3 / Hun 3 | 46.9% | 6.4% | 25.1% | 11.0% | ~7% | 75.7% |
| Pol 3 / Yug 3 | 48.4% | 10.9% | 34.9% | 21.0% | ~7% | 75.1% |
| Pol 3 / Cze 3 | 44.9% | 7.0% | 24.8% | 12.6% | ~8% | 70.5% |
| EG 4 / Pol 4 / Aut 1 | 47.1% | 13.1% | 38.1% | 22.8% | 0% | 80.4% |

* An Italy-adjacent opening (Yugoslavia, Austria) **doubles** USSR control of Italy by turn 4.
* The buffer removes the turn-1 loss of a battleground (~7% → 0) and keeps Poland more often
  through John Paul II (fired by turn 8 in ~65% of games in every opening).
* With its own setups (run `36795625066`) the USSR holds ≥3 European battlegrounds at the start of
  turn 2 in **8.6%** of games, 26.4% by turn 4; Italy is almost never the third.

## Result 2: the win rate barely moves, at any US strength

USSR win %, USSR = E6 under the forced opening, US = the rung (runs `36796912701`, `36796918391`,
`36796924090`, `36796929604`, plus the mirrors above):

| USSR opening | vs E4-08-36@240M | vs E5-11-43@560M | vs E6-06-44 soup |
|:---|---:|---:|---:|
| Pol 3 / Hun 3 | 90.2 | 69.1 | 46.9 |
| Pol 3 / Yug 3 | 93.6 | 70.1 | 48.4 |
| Pol 3 / Cze 3 | 92.2 | 69.3 | 44.9 |
| EG 4 / Pol 4 / Aut 1 | 90.4 | 66.6 | 47.1 |

Paired by deal, pooled over the three rungs (inverse-variance):

| alternative − Hungary | pooled | 95% interval |
|:---|---:|:---|
| Yugoslavia | **+2.2 ± 1.2** | −0.2 to +4.5 |
| Czechoslovakia | +0.5 ± 1.2 | −1.8 to +2.9 |
| EG 4 / Pol 4 / Aut 1 | −0.6 ± 1.3 | −3.1 to +1.9 |

Pairing barely helps above the E4 rung: a deal has the same result under two openings only ~60%
of the time against E5/E6 (87% against E4).

## Reading

* **The "weak opponents only" hypothesis is not supported up to E6 strength.** Hungary − buffered
  is −0.2 / +2.5 / −0.2 across the rungs: no trend. The rung that would test it properly is a US
  stronger than the model (search-64, pending).
* **Hungary is plausibly a small strict loss to Yugoslavia, 2–3 points** — suggestive, not
  settled. The buffered Austria opening is not better at this strength.
* **Why the lock survives:** the setup locks early (E5-11 by 40M,
  [`../log/E5_11_setup_lock_and_critic_views.md`](../log/E5_11_setup_lock_and_critic_views.md)),
  and a 2-point difference is below what setup credit, a 64-simulation teacher
  ([`determinization_targets.md`](determinization_targets.md)) or a 512-game tournament resolves.
  E6-04 re-opened setup entropy and seed 44 came back to Hungary
  ([`../log/E6_04_league_seed44.md`](../log/E6_04_league_seed44.md)), consistent with
  near-indifference rather than a reasoned preference.
* **Deep search agrees on the size.** Plain search on the E6 opening first tries Poland as the
  first placement near 9,400 simulations and prefers it near 30,000; at 64 simulations the values
  are +0.080 / +0.079, and the gap at 65k is ~0.02 (about a point of win probability). Source:
  commit `b249e6c` on `feat/mcts-gumbel` ([`gumbel_root.md`](gumbel_root.md)).

## What this does not say

* Every alternative is played by a net that practised Hungary; a structural gap is if anything
  understated. Forcing the opening into training (a `--forced-opening` arm) would remove that.
* The US setup is fixed throughout; whether the answer depends on it is untested. The unforced
  mirror (51.0% USSR) against the forced one (46.9%) hints the fixed US setup is ~4 points better
  for the US than E6's own, at 1.3 standard errors.
* E4 and E5 were trained on older engines, so the ladder confounds strength with engine fit.
