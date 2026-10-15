# P2 — Chance-aware value targets

**Status: sized offline and shelved (2026-09-28).** Averaging over dice and deals would remove
about **7%** of the critic's one-step error variance. Low priority: it needs a bindings helper and
the owner's approval for that gain.
**Needs approval:** yes — a batched forced-roll enumeration helper in `bindings/` (not a rules
change: `ROLL_DIE` already accepts a forced die). Put the signature in front of the owner first.

## The idea

Take the expectation over chance in the value target where the engine makes it exact:

1. **Dice.** At a step that lands on a `ROLL_DIE` node, bootstrap from the critic's mean over all
   outcomes (6 or 36) stepped on clones, not from the value after the sampled roll.
2. **Deal.** End the λ-return at the `TURN_CLEANUP` afterstate and bootstrap from the critic there,
   so nothing before the deal is credited with the hand that came after it (Stochastic MuZero's
   afterstates, Suphx's per-round boundary value).

Sign discipline: every value in the clone average is from the mover's perspective at the decision
step, and the mover can change before the roll — write the test first
([`../log/measurement_bugs.md`](../log/measurement_bugs.md), *a policy was choosing its own dice*).

## Sizing (2026-09-28)

`tools/scripts/chance_sizing.py` on E5-11-43@560M self-play at temperature 1: 49,869 decisions from
2,000 games, the chance after each chosen action resolved 16 times; the critic's Var_chance V(s′)
is the part of δ = V(s′) − V(s) that averaging removes. Report:
`data/reports/chance_sizing_e5_11_43.md`.

| | decisions followed by it | share of Var(δ) removed |
|:---|---:|---:|
| dice (same turn) | 8.3% | 4.3% |
| deal (the turn advances) | 1.7% | 2.8% |
| **both** | 9.8% | **7.1%** |

By turn: 16.6% (turn 1), 9.3% (2–3), 8.0% (4–5), 6.0% (6–7), 4.5% (8–10).

**Reading.** The rest of the one-step variance is the sampled actions (the signal the policy
gradient needs) and the critic's own inconsistency. With P5's sizing (the hidden hand adds +0.3
points of explained variance), variance reduction in the targets is not where this model's limits
are. Judge any revival on training outcome, never on offline target quality: E3-22's better
offline return estimate lost 520 Elo.
