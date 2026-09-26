# P27 stage 1: how the models spend an influence ops play, 2026-09-26

**Plan:** [`../plans/P27_ops_block_planning.md`](../plans/P27_ops_block_planning.md).

**Probe:** `tools/scripts/ops_block_probe.py` (exact enumeration, `ai/eval/ops_block.py`), 800
start positions per checkpoint from its own self-play at temperature 0.1, both seats,
`--max-allocations 150000` (6 and 4 positions skipped as larger). Reports:
`data/reports/p27_ops_block_v2.{md,json}`.

**Engine:** measured on the engine *before* the rule change below, where an influence play could
stop with points unspent. That makes no difference here: the policy stopped early in 2 of 1,590
positions.

**Definitions, as the owner set them:**
* A **contested battleground** is a battleground controlled by neither side, and reachable by both.
  Reachable means own influence there or in a neighbour, or adjacent to one's superpower.
* A **takeable** position is one where some allocation of the play takes one.
* The probe does not score *which* contested battleground is taken. Taking any one counts.

## The result

| measure | E4-61-44@720M | E4-08-36@240M |
|:---|---:|---:|
| positions | 794 | 796 |
| mean Ops | 2.65 | 2.51 |
| a contested battleground is takeable | 78.8% | 69.8% |
| policy takes one \| takeable | 48.9% | 50.4% |
| policy misses \| takeable | 51.1% | 49.6% |
| points spent on overcontrol (all positions) | 1.0% | 0.9% |
| battleground shortfall against the best allocation, mean | 1.14 | 1.03 |

**What the policy does instead, when it misses.** An outcome can hold together with another, so
the rows do not sum to 100%.

| outcome | E4-61-44@720M | E4-08-36@240M |
|:---|---:|---:|
| puts points into a contested battleground without taking it | 35.3% | 32.6% |
| takes another battleground | 24.4% | 37.7% |
| takes a non-battleground | 43.8% | 29.3% |
| breaks an opponent's control | 19.1% | 19.2% |
| no control change at all | 25.9% | 26.1% |
| stops with points unspent | 0% | 0% |

**Where its points go, when it misses** (share of the points placed):

| class | E4-61-44@720M | E4-08-36@240M |
|:---|---:|---:|
| contested battleground | 27.4% | 24.0% |
| other uncontrolled battleground | 21.5% | 31.9% |
| opponent-controlled battleground | 13.3% | 17.4% |
| own-controlled country (overcontrol) | 0.7% | 1.3% |
| uncontrolled non-battleground | 34.5% | 24.6% |
| opponent-controlled non-battleground | 2.5% | 0.7% |

**Against the network's own critic**, in takeable positions:

| critic's favourite | policy | E4-61-44@720M | E4-08-36@240M |
|:---|:---|---:|---:|
| takes it | takes it | 24.1% | 20.5% |
| takes it | skips it | 7.2% | 3.8% |
| skips it | takes it | 24.8% | 29.9% |
| skips it | skips it | 43.9% | 45.9% |

Over all positions, the policy's allocation is the critic's favourite in 21.2% and 18.2% of them.
The mean regret on the critic's own scale is 0.06 and 0.04.

## Reading

* **The failure is common and specific.** About half of the takeable positions are missed, on
  both checkpoints.
  * The owner's example — control of an uncontested country instead — accounts for most of the
    misses: 44% take a non-battleground, 24–38% take another battleground.
  * A third of misses put points into the contested battleground without taking it.
  * A quarter change no control at all.
* **Overcontrol is rare in this setting.** It gets about 1% of the points placed, overall and
  within misses, so it is not the main failure in these positions.
* **Mostly not a learning problem, as the critic sees it.** "Critic takes it, policy skips it" is
  the policy failing to carry out what its own critic prefers, and that is only 7.2% and 3.8% of
  takeable positions. In 69–76% of them the critic's favourite skips the contested battleground
  too. So the value function does not favour taking it either.
* **That is not yet a verdict of "value problem".** Two caveats:
  * the critic's favourite is an argmax over hundreds to thousands of end boards, so it is biased
    toward the critic's own errors;
  * the rule "take the contested battleground" may itself be too crude.
  Stage 2, the rollout oracle, decides between a critic that misprices the board and a rule that
  overstates the move.

## Rule change found on the way: influence Ops must be spent in full

The owner's ruling: an influence ops play may not end with points unspent unless no country can
take the next point. The engine used to open every influence play with `allow_early_stop = 1`.

**Before the change, measured on self-play at temperature 0.1**, at every influence-play decision
where a placement was legal (`data/logs/p27/early_stop_rate.{py,log}`):

| | E4-61-44@720M | E4-08-36@240M |
|:---|---:|---:|
| decisions | 190,665 | 223,027 |
| stopped (sampled) | 0.058% | 0 |
| stopped (greedy) | 0.047% | 0 |
| mean probability of stopping | 1.3% | 0.0 |

* Stops were rare in evaluation: 0.06 per game, 90 of the 110 with one point left.
* At training temperature, 1.3% per decision is about 1.4 points thrown away per game.
* E4-08-36's exact zero is logit drift: its uncentred country logits sit far above the stop.

**The change:**
* `begin_op_mode` sets `allow_early_stop = 0` for influence. Realignment keeps its stop.
* The mask offers `CONFIRM_DONE` in an influence play only when no country can take the next
  point. That is a legitimate end and raises no anomaly.
* `rules/rules.md` §5.1 states the rule.

**The observation keeps the old value.** Every checkpoint so far was trained seeing
`ALLOW_EARLY_STOP = 1` throughout an influence play, so the slot still reads 1 there, while the
engine's flag is truthful.
* A patch on the model side was not possible. From the observation alone, an ops play and some
  event placements (Marshall Plan, Decolonization, De-Stalinization and others) are identical:
  same decision type, op mode, Ops value, active card and max-per-country.
* **Check:** random games, never stopping early, recorded on a build of the previous commit and
  replayed on the new engine. One first game per env, 1,024 games, 1.43M decisions.
  * The mask differs only by the removed stop.
  * The observation is bit-identical everywhere except three decision kinds, one decision each,
    in 129 of the 1,024 games. All three come from one pre-existing leak, described next.

**The leak (not fixed; reported).** When a card is played for Ops first and the opponent's event
then fires (`state_machine.cpp` `advance_after_ops`, action round and headline), the event runs in
the Ops frame and inherits `allow_early_stop`. Under the old rule, after an influence play:
* Warsaw Pact Formed's mandatory branch choice offered "done", so it could be declined;
* the free Op from CIA Created or Lone Gunman showed a stale 1 in the slot.

The new rule removes this after influence. It remains after a realignment, which still opens with
the stop.

**The whole-corpus conversion found an Ops-value bug the old stop had hidden.** At turn 2 AR6 of
ts-replayer game 219 (also recorded as 220), the USSR plays Truman Doctrine (1 Op) for Influence
under both Red Scare/Purge and Vietnam Revolts. The log says "1 Ops": one point, into Burma.
* The engine floored Purge's result at 1 before adding Vietnam Revolts' +1, so it made the play
  worth 2. The old engine let the converter stop after Burma, which hid the error.
* The owner's ruling: each modifier has its own limit.
  * Containment and Brezhnev Doctrine may not raise a card above 4, or 5 for the China Card in
    Asia.
  * Red Scare/Purge may not lower it below 1, and that floor applies to the final value.
  * Vietnam Revolts adds on top of both, so a 4-Op card becomes 5 and the China Card 6.
* `Operations::combine_ops` (`engine/src/ops.cpp`) holds this one rule. The coup path, which added
  the bonuses to an already-floored value, now calls the same function
  (`get_effective_ops_in`).
* Every existing modifier test still passes. Only the 1-Op card under both Purge and Vietnam
  Revolts changes.
* The rule is written into `rules/rules.md`, beside the Ops modes.

**Tests:**
* the C++ suite, 382 tests, including:
  * three new ones under `InfluenceOpsMustBeSpent`;
  * `OpsModifiers_VietnamRevolts_BeyondTheCaps_PurgeFloorOnTheResult`;
* `ts_fuzz`, 10,000 games with no anomaly;
* the backend Python suites;
* the WebAssembly engine test;
* the whole-corpus conversion (`corpus_full`): all 300 games, with nothing added to the
  invalid-play list.

Two expected changes:
* Replay 59's invalid-play entry now takes 3 answering decisions instead of 1.
* One bot replay recorded under the old rule, which stopped early at step 172, was moved to
  `data/replays/archive_pre_influence_spent/`.
