# P30 (2026-10-03): C2's card-event head reaches the trunk; soups of the two branch arms

Follow-up to [`P30_branch_arms_c2_play_mode_temp.md`](P30_branch_arms_c2_play_mode_temp.md). The
owner asked for the card probe on C2 and for a soup of the two arms against the old soup.

## 1. Card probe: did the card-event target reach the trunk?

The P30 check ([`P30_card_board_targets.md`](P30_card_board_targets.md)): a fresh per-card head fitted
on the *frozen* hidden vector, scored within card on held-out games. The old dataset was made on the
E6 engine, so a new one was made on the current engine (1d11c2f2): self-play of E7-02-44@1200M and
E7-03-44@1200M, 1,500 games each, 159k positions (`data/datasets/card_board/e7_02_03_1200M.npz`;
report `data/reports/p30_c2_card_board_probe.{md,json}`). The positions come from the control's
own lineage, not from C2's games.

| trunk | T1 within (scoring VP) | T2 within (ops arithmetic) | **T4 within (event outcome)** |
|:---|---:|---:|---:|
| untrained (E7-01-44 at step 0) | 0.645 | 0.746 | 0.594 |
| E7-02-44@820M, the branch point | 0.870 | 0.826 | 0.620 |
| E7-02-44@1200M, the control | 0.872 | 0.827 | 0.619 |
| **C2 E7-13-44@1200M** | **0.905** | **0.914** | **0.656** |
| play-mode temp E7-14-44@1200M | 0.874 | 0.833 | 0.620 |
| *the same trunk trained on the targets (supervised; stopped at 13 epochs)* | *0.864* | *0.957* | *0.678* |

* **The target reached the trunk.** The control learns nothing about card × board between 820M and
  1,200M (0.620 → 0.619 on T4, as on E6). C2, over the same 380M, moves T4 to 0.656 and T2 from
  0.827 to 0.914 -- about two thirds of the way from the control to the supervised trunk on both.
  Its scoring-card VP (T1) is above the supervised fit. This is the opposite of bet 2's ownership
  head, which never moved the trunk.
* **The play-mode temperature does not move it** (0.620), as expected: it changes exploration,
  not the targets.
* The supervised ceiling here (0.678) is far below the E6 dataset's 0.80. The datasets differ, and
  this fit stopped early at 13 epochs, so 0.678 may understate the ceiling.

**Behaviour at card selections** (within-card correlation of a card's relative logit with its
event outcome):

| net | VP | regional margins | battleground balance | influence balance |
|:---|---:|---:|---:|---:|
| E7-02-44@1200M | +0.095 | +0.184 | +0.188 | +0.222 |
| C2 E7-13-44@1200M | +0.101 | +0.213 | +0.216 | +0.247 |
| play-mode temp E7-14-44@1200M | +0.099 | +0.206 | +0.207 | +0.247 |

Both arms respond slightly more to what a card's event would do than the control does (+0.01 to
+0.03). C2 is not clearly ahead of the play-mode temperature, so having the information in the
trunk has, so far, changed the policy only a little.

## 2. Soups of the two arms against the old soup

E7-13-44 and E7-14-44 both branch from E7-02-44@820M, so they average. C2's card-event head is
training-only and the other arm has none, so it was dropped before averaging. Panel rule against
the old soup `shallow_E7-02+03+04+05_1200M`, 1,000 games per side, temperature 0
(`data/reports/e7_13_14_soup.{md,json}`, `data/reports/e7_two_way_soups.{md,json}`):

| soup (all at 1,200M unless stated) | US | USSR | head to head |
|:---|---:|---:|---:|
| **C2 + play-mode temp** | −1.5 ± 0.8 | −3.0 ± 0.8 | 47.1% ± 1.1 |
| C2 + play-mode temp, 1,120–1,200M SWAs (against the old SWA soup) | −2.2 ± 0.8 | −3.3 ± 0.8 | 46.4% |
| *control: E7-02 + E7-03* | −1.9 | −3.6 | 46.8% |
| *control: E7-04 + E7-05* | −1.1 | −0.4 | 48.0% |
| E7-02 + C2 | −1.1 | −2.2 | 45.0% |
| E7-02 + play-mode temp | −1.6 | −1.8 | 46.3% |
| **all six** (old four + C2 + play-mode temp) | +0.2 ± 0.8 | −0.2 ± 0.8 | 51.6% ± 1.1 |

* **Every two-way soup is 45–48% against the four-way soup.** The shortfall comes from the number
  of ingredients, not from the new arms. Among two-way soups, C2 + play-mode temp is level with
  E7-02 + E7-03 (51.8%) and below E7-04 + E7-05 (47.4%, E7-04 being the strongest old branch).
* **The six-way soup is level with the four-way one.** Adding the two arms neither helps nor
  hurts, so the best model is unchanged ([`../checkpoints.md`](../checkpoints.md)).
* The old soup's numbers reproduce its first measurement exactly (90.0 / 89.2 against the panel,
  58.9% against E7-02-44's SWA): temperature-0 tournaments are deterministic.

## Reading

C2 passes the probe check that the plan set before a full run: the card-event target, unlike the
ownership target, puts its information into the trunk. Strength on the branch protocol is a small
positive signal (the previous log). As a soup ingredient it is no better and no worse than a plain
branch. **The next test is C2 from scratch against E7-02-44** -- the probe says the mechanism
works; only a full run can say whether it makes the base model stronger.
