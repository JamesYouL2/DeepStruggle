# P5 — Oracle critic (and the belief head that rides with it)

**Status:** **sized offline and shelved (2026-09-28)**: the opponent's hand explains +0.3 points of
the result's variance beyond the public critic (0.283 → 0.286), at most +0.8 at setup. There is too
little hidden information for an oracle critic to reduce much variance; see *Sizing* below.
**Gate:** after P2, so a deal-side variance reducer is not confounded with the deal-side
bootstrapping change. If P2 is demoted, run this in its slot.
**Needs approval:** none.

> **The code no longer exists in the tree, and that is deliberate.** The oracle critic and
> belief head lived on `ColdWarNetV4`, which was removed along with `ColdWarNetV3`. Read the
> old implementation out of the history when this step runs:
>
> ```bash
> git show 9c74b98:ai/models/coldwar_net_v4.py     # belief_head, oracle head, forward_all,
> git show 9c74b98:ai/training/nash_pg.py          #   predict_belief, evaluate_oracle
> git show 9c74b98:ai/training/rollout_buffer.py   # OracleGuidedNashPGTrainer, the losses
> git show 9c74b98:bindings/ts_env.py              # opp_hands columns, get_batches_with_oracle
> ```                                              # info["opponent_hands"] plumbing
>
> `VectorizedBatchRunner::get_opponent_hands` was **kept** in the bindings: it is a read-only
> accessor, it costs nothing unless called, and it is the one piece of this that never needs
> migrating when the observation changes.
>
> Treat what comes back as a reference for the idea, not as a port. It was written against the
> legacy observation, the pre-starred-card engine, and *scalar* value heads — and P1 replaces
> those with a categorical VP distribution, so `mse(oracle_val, ret_win)` and the distillation
> `mse(v_win, oracle_val.detach())` both need redesigning against a distribution anyway. Keeping
> it in the tree would not have saved that work; it would only have kept a stale design in view,
> and made every observation change migrate three architectures instead of one.

## Goal

Measure the privileged critic that is already implemented and has never appeared in the
experiment log: `oracle_loss_coef = 0.25` and `belief_loss_coef = 0.10` in
`ai/training/nash_pg.py` (`get_batches_with_oracle`, `forward_all(obs, mask, opp_hands)`,
`evaluate_oracle`, `predict_belief`; oracle loss `mse(oracle_val, ret_win)`, distillation
`mse(v_win, oracle_val.detach())`, belief loss `BCE(pred_belief, opp_hands)`).

## Why

- In a game with weak hidden information and heavy chance, the oracle critic is mainly a
  **deal-side variance reducer**: a critic that sees both hands has the hand luck removed from
  its target, and the public critic distils from it (Suphx, `paper_suphx_mahjong.md`). That is a
  different mechanism from the one Suphx needed it for (Mahjong's hidden information is strong),
  and it is the mechanism that fits here.
- The belief head is the cheap version of what the VOA-awareness goal needs: P(VOA in the
  opponent's hand) is one of its 110 outputs. Whether it is any good has never been measured.
- It is no longer the free arm it was described as. Running it on V4 would have compared a v4
  backbone *plus* an oracle critic against a v2 baseline, which confounds the two and cannot
  attribute the result — so the heads have to be rebuilt on the v2 backbone as optional heads
  either way. That is the implementation cost of this step, and it buys a one-factor arm.

## Change

Rebuild the oracle critic and belief head as **optional heads on the v2 backbone**, off by
default, so the arm differs from its control in one thing. The old implementation is the
reference (see the header). What has to be decided when it runs is how the oracle target and
the distillation work against a categorical critic, which is a question the scalar version did
not have to answer.

The paragraph below is what the step said when V4 still existed, kept because the mechanism
description is still right:

None to code, if the paths still run on the current layout (the observation has changed twice
since these heads were written — v2.1 and then v2.2, `experiments.md` §24; the oracle path
consumes `opp_hands` separately from the observation, so check it, and check that `forward_all`
exists on the categorical head from P1). *Decide before running:* the two coefficients — keep the defaults
for the screen; do not tune on the screen.

Everything else fixed at the current baseline.

## Procedure

1 arm × 2 seeds × 80M: baseline + oracle + belief, against the baseline. If it helps, a second
pair with the belief loss off, to attribute the effect. Confirm the winner at 240M.

## Measure

Pre-deal calibration and the ordinary calibration curve (the oracle's own curve and the public
critic's); belief head quality — AUC of `pred_belief` against the true opponent hand, and
specifically P(VOA in hand) when it is; the VOA-exposure probe; then Elo.

## Decision rule

- Adopt if the public critic's calibration improves with Elo not worse than −20.
- If the belief AUC is near chance, the head is not learning from the BCE alone — log it and
  queue "belief head with a real target weight" in reserve rather than tuning here.
- If Elo drops by more than 20: the distillation term is pulling the public critic toward
  values it cannot know from its own inputs (the classic oracle failure); halve
  `oracle_loss_coef` once, and if that does not recover it, kill.

## Follow-ups

- VOA-exposure probe still at baseline after adoption → the belief head is learned but the value
  does not use it; that is a P6-style architecture question (does the fused vector carry the
  belief?), not a loss-weight question.
- Oracle adopted + P2 adopted → both deal-side reducers are in; the remaining chance is the
  policy's own sampling, which Q-boosting proper (an action-value critic, `references.md` §3)
  would address. Reserve.

## Runs

(none yet)

## Sizing (2026-09-28)

`tools/scripts/oracle_sizing.py` ran before anything was built. It used 29,924 self-play games of
E5-11-43@560M at temperature 1 and recorded 2% of decisions (281,538 records), scored on 5,984
held-out games. The opponent's hand (110 bits) was the only extra input: no deck order and no
future rolls, which are chance, not hidden information. Two fits compare the result's explained
variance (1 − Brier / Brier of the base rate) without and with it. The first is an MLP on the
network's trunk features plus its own v_win, early-stopped on held-out games. The second is a
linear fit on the critic's own logit. Report: `data/reports/oracle_sizing_e5_11_43.md`.

| slice | checkpoint's critic | MLP: + opponent's hand | linear: + opponent's hand |
|:---|---:|---:|---:|
| all | 0.283 | +0.003 | +0.003 |
| setup | 0.049 | +0.008 | +0.008 |
| headline | 0.276 | +0.001 | +0.004 |
| action round | 0.292 | +0.003 | +0.003 |
| turn 1 / 2–3 / 4–5 / 6–7 / 8–10 | 0.067 / 0.174 / 0.295 / 0.338 / 0.505 | +0.004 / +0.001 / +0.001 / +0.002 / +0.005 | +0.005 / +0.003 / +0.002 / +0.003 / +0.005 |

**The hidden hand carries almost no information about the result that the public critic lacks.**
About 72% of the result's variance is unexplained by either, which is future chance (dice,
draws) and future play. An oracle critic's value is bounded by the extra variance it explains, so
this one would reduce the policy's advantage variance by well under 1%. The earlier finding that
the critic cannot rank headline choices position by position is therefore not a hidden-information
problem: at the headline the hand adds +0.1 to +0.4 points.

Caveats: a first run on 6,000 games overfit (heads memorise games, since the result is one label
per game) and was discarded. And the MLP sees hand × board interactions only through a 128-wide
layer. Neither moves the conclusion much: the linear fit gives the same numbers.

If variance is the target, the chance side is where it is: P2's chance-aware targets, which
average over the next roll instead of sampling it.
