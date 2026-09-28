# P27 — Short-horizon planning inside an ops play: capacity, learning or data?

**Status:** stages 1 and 2 done (2026-09-26): [`../log/P27_ops_block_stage1.md`](../log/P27_ops_block_stage1.md). Stage 1: half of takeable contested battlegrounds are missed, and the critic's favourite skips them too. Stage 2, the rollout oracle: taking one instead is worth −0.8 ± 0.6 pp when this policy plays the rest, so the miss is not a defect by outcome and stage 3 has nothing to teach. The open caveat is a stronger playout policy.

## The problem

The owner's observation: the models show nothing like a plan even over a few micro-steps.
* With enough ops to take a contested battleground, they first go for immediate control of an
  uncontested country and leave the contested one open.
* Or they waste a point on overcontrol.

This is easy to eyeball and hard to formalise. The question is which kind of problem it is:
* **capacity:** the network cannot represent the plan;
* **learning:** the signal to learn it is there, but the policy does not pick it up;
* **data:** the situations are too rare to learn from.

## The lever: an influence play is solvable exactly

An ops play spent on influence is a run of POINT_NODE decisions by one player with no chance node
between them. From the position where it starts, every legal allocation can be enumerated with the
engine judging legality at each point, and each distinct end board scored. Canonical-order search
with a transposition set gives 36–24,000 end boards for 1–5 ops in about 0.5 s.
`tests/training/test_ops_block.py` checks it against an unrestricted search.

## Stages

1. **Measure** (`tools/scripts/ops_block_probe.py`), on positions from each checkpoint's own
   self-play:
   * **missed contested battleground:** some allocation takes a battleground the opponent has
     influence in, and the policy's does not;
   * **uncontested instead:** the policy gains an uncontested country in those positions;
   * **reinforcing points;**
   * **battleground shortfall** against the best allocation.

   **Against the network's own critic:** the policy's allocation against the critic's favourite,
   the regret, and — the diagnosis — in takeable positions, whether the critic's favourite takes
   the contested battleground.

   | critic's favourite | policy | reading |
   |:---|:---|:---|
   | takes it | skips it | a **learning problem in the policy**: it cannot execute what its critic knows |
   | skips it | skips it | a **value problem**: the critic does not price it |

2. **Ground truth for the value question:** on a sample of takeable positions, play each
   candidate allocation out many times with a fixed strong policy (a rollout oracle). That says
   whether taking the contested battleground actually wins more. It checks that the value-problem
   reading is a critic error, not a correct judgement that the rule is too crude.
3. **Frequency and a yardstick:**
   * how often takeable positions occur per game — rare means data;
   * the same probe on human play from the corpus, which needs positions rebuilt by the replayer.
4. **Capacity:** supervised training of the same architecture on the oracle-best allocations, on
   held-out positions. If it fits, capacity is not the limit. If it cannot, capacity or
   representation is, and a change there needs approval.

## What each outcome leads to

* **Learning** → distil the exact block-best allocation into the policy. This is cheap, exact and
  needs no MCTS. Or use λ = 1 inside ops-placement blocks only, much narrower than P4's A2.
* **Value** → targets that price board position: search or rollout values for the critic on these
  states, or auxiliary battleground targets.
* **Data** → oversample such positions, for example through the start pool.
* **Capacity** → an architecture change, proposed with its cost.

## Runs

* Stage 1: E4-61-44@720M (strongest rated) and E4-08-36@240M (best battleground coverage on the
  goal probes), 800 positions each.
