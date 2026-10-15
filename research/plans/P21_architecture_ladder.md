# P21 — the architecture ladder (`--arch ladder`)

**Status: closed (2026-09-22); the backbone is the current architecture.** Requested by the owner
2026-09-19: build up from an MLP one mechanism at a time. The M2 family closed with **M2d** on top;
M3–M5 (attention, pooling) were never run as rungs, and the token form of that idea became P30 C1
(stopped at 400M, [`../log/P30_C1_tokens.md`](../log/P30_C1_tokens.md)). The record is
[`../log/P21_ladder_status.md`](../log/P21_ladder_status.md) and the `P21_*` files beside it. The
code (`ai/training/train.py`, `ai/models/ladder_net.py`) cites this file for the backbone's
contract and the rung definitions.

## The backbone

`ai/models/ladder_net.py`: `LadderNet` subclasses `ColdWarNetV2`, so the heads, the value contract
and the action space are untouched and only the encoder differs. `create_like` rebuilds a copy
from the model's own `ladder_config()`.

**Every structural axis is required, at construction and at the CLI.** `--arch ladder` refuses to
run unless each is named, and a ladder flag under another `--arch` is refused rather than ignored:
a defaulted variant argument was the mechanism behind five instances of one bug here. The ladder is
cold-start only (`--mode warmup` refuses it). `tests/training/test_ladder_net.py` pins the rungs
and the refusals. Refused combinations: per-entity heads, attention or `identity_dim > 0` with
`flat`/`grouped` input (no tokens to read; position already identifies the entity), and
`--drop-static` with `entity` input (the static slots are how a shared encoder knows what it reads).

## The rungs

| rung | flags (beside `--ladder-entity-proj-dim 256 --ladder-hidden-dim 384 --ladder-res-blocks 4`) | result at 80M |
|:---|:---|:---|
| **M0** flat MLP | `--ladder-input-mode flat --ladder-aggregation flatten --ladder-entity-dim 16 --drop-static` | the floor |
| **M1** grouped projections | `--ladder-input-mode grouped --ladder-aggregation flatten --drop-static` | **+109** over M0 |
| **M2** both per-entity heads | M1 + `--per-entity-heads 64 --ladder-head-entities both` | the full mechanism |
| **M2d** country head only | M2 with `--ladder-head-entities country` | **+282** over M1, the largest mechanism |
| **M2e** card head only | M2 with `--ladder-head-entities card` | −7: nothing |
| **M2a / M2b / M2c** | M2d without head context / without per-type constants / both | −84 to −146, −40 to −73, −87 to −189 against M2d |
| **M2.5, M2.5b, M2.5c** | identity vectors added to the head | all lose to M2d |
| M3 / M4 / M5 | `entity` input + `--ladder-card-self-attention` / + `--ladder-cross-attention` / `--ladder-aggregation pool` | never run |

The base recipe has since moved on (the shallow trunk, `--ladder-res-blocks 0`; the
card-conditioned play-mode and branch heads, A8): the current codes and their flags are
[`../architectures_and_recipes.md`](../architectures_and_recipes.md).
