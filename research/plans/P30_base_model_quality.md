# P30 — base-model quality

**Status:** proposed (2026-10-01). Supersedes the open part of
[P29](P29_big_bets_from_scratch.md): bets 1–3 are closed, bet 2's idea is carried forward as C2,
and bets 4–5 continue here.

**Goal (owner, 2026-10-01): the quality of the base model** -- what one training run produces.
SWA and model soups are evaluation devices that read a recipe with less noise. They are not the
product, and averaging does not enter training.

## What is known

From [`../findings/training/trunk_depth_and_state.md`](../findings/training/trunk_depth_and_state.md)
and [`../findings/training/training_length_and_evaluation.md`](../findings/training/training_length_and_evaluation.md):

* **Settled at saturation (2026-10-01, E7-06-44):** the deep trunk plateaus at about 560M and the shallow one ends about +120 Elo above it at 1,200M. At the end the shallow line wins 64.8% on snapshots and 57.2% on SWAs, passing the owner's rule ([`../log/E7_06_44_deep_to_saturation.md`](../log/E7_06_44_deep_to_saturation.md)). The shallow trunk is the base for every candidate below (seed 44; the seed-43 control will cover the second seed).
* **The base recipe is now the shallow trunk:** E7-01/02-44, `--ladder-res-blocks 0`, no league.
  It beats M2d on two seeds at 560M, saturates at about 900M–1B steps, and its SWA at 1,200M was
  level with the old best model.
* **Depth and width after the fusion do not help** (M2d against shallow; bet 1's 768 × 8).
* **The trunk drops state at the input projections.** Training adds outcome information to it,
  not state.
* **Card × board interactions:** the grouped trunk can partly represent them (0.80 within-card
  R²), explicit attention represents them better (0.90), and RL leaves the trunk at the untrained
  level (0.66).
* **Recipes have been compared before saturation** (at 560M).

## The evaluation protocol

* **Arm and control from scratch on the same seed.** Each is traced in one field every 80M to
  about 1B steps.
* **Read at the end:** the late snapshots, and the last-80M SWA, by the owner's rule: per seat
  against the panel, and head to head.
* **Adopt on the same rule** (better in one seat, not worse in the other) **at saturation, on two
  seeds.** A gain at 560M only is recorded as "faster", not "better".
* **Cost:** about 3 h per arm solo. Shallow arms run two at a time for +20% throughput, so one
  arm-and-replicate pair takes about 5–6 h.
* **The seed-44 control exists:** E7-01-44 to 560M and E7-02-44 to 1,200M. **A seed-43 control
  to 1B is the first item**, so every candidate below has both of its controls.

## Candidates, in the order proposed

### C1 — Card and country tokens (the input representation)

**Change.** Every country and card becomes a token: its raw row plus a learned identity, projected
to d = 128. One or two transformer layers run over the 194 tokens plus a global token. The
country and card heads read their own tokens; the pooled trunk reads the global token alongside
the grouped path, which stays. This is the `board_mode`/`card_mode` work that P21's M3–M5 rungs
are blocked on.

**Why.** The state is lost at the input projections. Explicit attention represents the
card × board targets at 0.90 against 0.80, with 7× fewer parameters. The country head proved the
per-entity read-out (+282).

**First:** measure its training throughput on the shallow recipe. Attention over 195 tokens per
position may cost a large share of steps/s. If it is more than about 2× slower, judge it at
matched wall-clock as well as matched steps.

**Engineering:** about a day. A new input mode in `LadderNet`, the head wiring, and tests.

#### C1b (owner, 2026-10-02) -- cheaper, sharper tokens: concatenate, don't project

Queued behind C1's 400M readout. In C1 a card token is `W·row + b + identity`, with `W` a learned
128 × 14 matrix. But of a card's 14 slots only the location one-hot (8) and the active-card flag
are dynamic; ops, side, era, one-time and is-scoring are constant per card, so the identity
already carries them, and `W` spends parameters re-encoding a handful of places. The owner's
variant: **card token = [location one-hot ; active flag ; a slightly smaller learned identity]**,
concatenated, no `W`. Countries likewise: their dynamic slots concatenated with a country
identity. Also worth fixing in the same arm: C1 initialises the identities at 0.02 (‖id‖ 0.23
against ‖W·row‖ 2.4–3.7 at init), the scale the P22 lookup found invisible.

### C2 — The card-event auxiliary target (`--aux-card-events`, built)

**Change.** A per-card head on the trunk is trained on what each held card's event would do and
what its Ops could reach. The engine labels it (Ops modifiers exact; choice events played out),
and a replay buffer gives many steps per run. The code, tests and docs are in place
(`ai/training/card_event_targets.py`), postponed by the owner on 2026-09-30.

**Why.** The architecture can hold more of this than RL puts there. This tests whether teaching it
helps the base model. Unlike bet 2's ownership head, its signal is dense, per card, and immediate.

**Check at 200M, before the full run:** `card_board_probe.py --frozen`. The trunk must move from
the untrained 0.66 towards 0.80 within-card, or the target has not reached it (bet 2's failure
mode).

### C3 — A learning-rate step-down in the base recipe

**Change.** E7-02-44's recipe with `--lr-schedule step` over the last third (as E7-04-44: 1e-4,
then 3e-5).

**Why.** E7-04-44 was the strongest single branch of the soup (53.5% against its constant-rate
sibling). The deep line's version (P28 step 2a) lost on its SWA. So the evidence conflicts, and it
is a cheap recipe change to settle at saturation.

### C4 — The ops budget in the observation (P29 bet 4, owner-held)

**Approved and running (owner, 2026-10-01) as E7-07-44.** The owner's framing: an observation
addition is not a decision-stream change, so it is built as a configurable *view spec* -- optional
blocks appended after the base layout, recorded in the weights, built per decider -- and models of
different views play each other directly. Three floats (`OPS_BUDGET`, `engine/AGENTS.md` §6a): the
Ops the card at a play-mode decision grants after every modifier, and each side's per-card
modifier. The base layout and existing checkpoints are untouched.

### C5 — Search-driven training (P29 bet 5, gated)

Unchanged: it needs a critic that search can use, and +5 pp of honest search headroom. Re-measure
the headroom on the base recipe at saturation before considering it.

## Sequencing (two arms at a time)

| slot | arms |
|:---|:---|
| 1 | the seed-43 control to 1B ‖ C2 on seed 44 (no build needed) |
| 2 | C1, seed 44, after its throughput check ‖ C3, seed 44 |
| 3 | seed-43 replicates of whatever leads |

C4 (E7-07-44) runs alongside the seed-43 control (E7-08-43) in slot 1; C1 is built meanwhile. C5 waits for its gate.
