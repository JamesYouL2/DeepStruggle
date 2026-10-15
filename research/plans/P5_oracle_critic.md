# P5 — Oracle critic (and the belief head that rides with it)

**Status: sized offline and shelved (2026-09-28).** The opponent's hand explains **+0.3 points** of
the result's variance beyond the public critic (0.283 → 0.286), at most +0.8 at setup: too little
hidden information for an oracle critic to reduce much variance.

## The idea

A critic that also sees the opponent's hand, used to reduce the variance of the public critic's
targets (and a belief head predicting that hand). The old implementation lived on `ColdWarNetV4`,
removed from the tree; read it from history as a reference, not a port (`git show
9c74b98:ai/models/coldwar_net_v4.py`, and `nash_pg.py`, `rollout_buffer.py`, `bindings/ts_env.py`
at the same commit). `VectorizedBatchRunner::get_opponent_hands` is kept in the bindings.

## Sizing (2026-09-28)

`tools/scripts/oracle_sizing.py`: 29,924 self-play games of E5-11-43@560M at temperature 1, 2% of
decisions recorded (281,538), scored on 5,984 held-out games. The only extra input is the
opponent's hand (110 bits); explained variance = 1 − Brier / Brier of the base rate. Report:
`data/reports/oracle_sizing_e5_11_43.md`.

| slice | checkpoint's critic | MLP: + opponent's hand | linear: + opponent's hand |
|:---|---:|---:|---:|
| all | 0.283 | +0.003 | +0.003 |
| setup | 0.049 | +0.008 | +0.008 |
| headline | 0.276 | +0.001 | +0.004 |
| action round | 0.292 | +0.003 | +0.003 |
| turn 1 / 2–3 / 4–5 / 6–7 / 8–10 | 0.067 / 0.174 / 0.295 / 0.338 / 0.505 | +0.004 / +0.001 / +0.001 / +0.002 / +0.005 | +0.005 / +0.003 / +0.002 / +0.003 / +0.005 |

**Reading.** ~72% of the result's variance is future chance and future play, unexplained either
way. The critic's failure to rank headline choices is therefore not a hidden-information problem
(the hand adds +0.1 to +0.4 points there). A first run on 6,000 games overfit and was discarded;
the linear fit gives the same numbers as the MLP.
