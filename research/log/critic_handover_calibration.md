# Critic calibration across a hand-over (2026-10-02)

Owner: "check our current best soup for critic calibration. Particularly ... if difference between
US and USSR critic on action boundary averages to 0 (check that it should)".

## Should it?

At a hand-over -- one side's last decision s_t, the other side's first decision s_{t+1} -- each
critic is read from its own perspective (the side it is trained on). Each predicts its own return,
so by the law of total expectation, averaged over positions,

    E[v_A(s_t) + v_B(s_{t+1})] = E[G_A] + E[G_B] = 0     if the targets are zero-sum.

No nesting of information is needed for the average: that the USSR knows its hand and the US does
not breaks equality per position, not on average. It holds within any class of positions both sides
can see (turn, phase), not within a class defined by one side's hand. The recipe's targets are
zero-sum except in the turn of a blunder (`blunder_aware`: the blunderer's target is −1, the
winner's pinned to its own value), which would pull the mean slightly *negative*. The value loss
covers every row, the learner's and a pool opponent's alike, so the pool does not bias it.

## Measured (`python -m ai.eval.critic_boundary`; self-play at temperature 1, 2,048 games, SEs clustered by game)

| | shallow soup (E7-02/03/04/05@1,200M) | E7-02-44@1,200M |
|:---|---:|---:|
| mean v_prev + v_next over hand-overs | **+0.012 to +0.015 ± 0.0008** (two runs) | **+0.0123 ± 0.0008** |
| US → USSR / USSR → US | +0.015 / +0.010 | +0.013 / +0.011 |
| games without a blunder ending / ending in DEFCON 1 | +0.013 / +0.009 | +0.013 / +0.011 |
| same mover, consecutive decisions, v_next − v_prev | +0.0027 ± 0.0001 | |
| same mover, last − first decision of a run | +0.0087 ± 0.0002 | |

By turn the sum varies (soup: −0.012 on turn 1, +0.01 to +0.025 later; E7-02-44: +0.003 to +0.029),
always small.

## Reading

* **Not zero, but small: both critics are slightly optimistic.** A sum of +0.012–0.015 on the
  [−1, 1] scale is each side over-rating its win probability by about 0.3–0.4 points.
* **Not the blunder effect** (that would be negative, and blunder-ending games are no different),
  **and not the soup**: the single trained model shows exactly the same.
* **Where it sits: each side grows more optimistic through its own run of decisions** -- its own
  critic should be a martingale along its own moves, but it rises +0.0087 from the first decision of
  a run to the last -- and the hand-over resets it. Consistent with a critic that lags the policy
  (it values a state as if play will continue at a slightly weaker level than the current policy's,
  so the current policy's own moves look like gains, for whichever side is moving).
* **Effect on training:** in GAE the hand-over TD error has mean −(sum) ≈ −0.015 and each
  within-run step +0.0027; against a raw advantage std of ~0.2 that is a structured but small
  term (≈ 0.07 SD on the last decision of each run, partly cancelled by λ = 0.98 over the run).
* Per-side error against the realised result is too noisy to decompose at this size (SE 0.017).

## Overall calibration of the soup's critic (2026-10-02)

Owner: "How well calibrated is critic overall?" Every decision's predicted win probability
(v + 1)/2, from the mover's side, against whether the mover won; 2,030 self-play games at
temperature 1 (~1.04M positions), SEs clustered by game (`python -m ai.eval.critic_boundary`).

| | bias (predicted − realised) | ECE | Brier | Brier skill | AUC |
|:---|---:|---:|---:|---:|---:|
| all positions | **+0.002 ± 0.0004** | **0.006** | 0.175 | +0.30 | 0.816 |
| as US | −0.020 ± 0.009 | 0.020 | 0.173 | +0.31 | 0.820 |
| as USSR | +0.026 ± 0.009 | 0.026 | 0.177 | +0.29 | 0.813 |

Reliability: every bin within ±0.010 of the diagonal except 0.6–0.7 (predicted 0.649, realised
0.626 ± 0.008). By turn, calibration in the large holds throughout (|bias| ≤ 0.008), while
discrimination grows from AUC 0.62 / skill +0.04 on turn 1 to AUC 0.96 / skill +0.70 on turn 10.
By ending: normal games AUC 0.83, held-scoring endings 0.77, **DEFCON-1 endings AUC 0.64 and skill
0.00** (ECE 0.12).

* **Well calibrated overall:** the diagonal is met to about a point everywhere, ECE 0.6%.
* **A small side tilt:** both critics rate the USSR's chances ~2 points too high (USSR optimistic
  +0.026, US pessimistic −0.020, each ~2–3 SE).
* **Blind to DEFCON-1 endings:** in games that end at DEFCON 1 its predictions say nothing about who
  wins -- the known scoreboard-shaped critic (`ai/eval/critic_calibration.py`), and those targets are
  also the ones the blunder window alters.
