# P30 C1 (2026-10-02): card and country tokens with attention -- readouts

**Arm.** E7-09-44: E7-01-44's shallow recipe, seed 44, plus `--ladder-token-layers 2 --ladder-token-dim 128`
(195 tokens: 84 countries, 110 cards, one global; each `W·row + b + identity`; two pre-norm
transformer layers, bf16) and `--compile-update default` ([`../runs.md`](../runs.md)). ~10.4k steps/s
compiled (the shallow recipe alone: ~96k). To 400M first.

## Identity vectors grow slowly

Initialised at 0.02 per coordinate. Mean norms, against the projected state part of the token:

| step | card ‖W·row+b‖ / ‖id‖ | country ‖W·row+b‖ / ‖id‖ |
|---:|---:|---:|
| init | 3.73 / 0.23 | 2.44 / 0.23 |
| 20M | 3.21 / 0.64 | 2.17 / 0.57 |
| 40M | 3.31 / 0.88 | 2.30 / 0.76 |

## 80M (owner: "a small tournament on 80M against 80M and 160M snapshots" of the shallow arm)

One field, 1,000 games per side per pairing, with the panel (E6-03-44@80/240/550M):

| model | Elo |
|:---|---:|
| E7-01-44@160M (shallow, seed 44) | 1563 |
| E7-01-44@80M (shallow, seed 44 -- the control) | 1476 |
| E7-07-44@80M (shallow + C4, seed 44) | 1411 |
| E7-08-43@80M (shallow, seed 43) | 1410 |
| **E7-09-44@80M (C1)** | **1404** |

| C1@80M against | US (panel) | USSR (panel) | head to head |
|:---|---:|---:|---:|
| E7-01-44@80M (paired control) | −8.1 ± 1.1 | −6.0 ± 1.1 | 38.7% |
| E7-01-44@160M | | | 29.0% |
| E7-07-44@80M | +1.1 ± 1.0 | −4.5 ± 1.1 | 47.9% |
| E7-08-43@80M | +3.3 ± 1.0 | −2.7 ± 1.1 | 48.9% |
| *reference: E7-07-44@80M against E7-01-44@80M* | −9.2 | −1.5 | 36.2% |
| *reference: E7-08-43@80M against E7-01-44@80M* | −11.4 | −3.3 | 40.0% |

* **C1 is behind its paired control at 80M (38.7%), but E7-01-44@80M is an outlier:** the other two
  shallow runs at 80M (C4 and the seed-43 control) lose to it by as much (36.2%, 40.0%). Against
  those two C1 is level (48–49%).
* **So at 80M C1 is an ordinary shallow-recipe run** -- neither the attention gain nor a cost from
  its slow start is visible. 80M is early (the recipe saturates at ~1B); next look at 160M.
* Reports: `data/reports/e7_09_44_rr_80{,b}.{md,json}`.
