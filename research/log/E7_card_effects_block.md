# What each held card does, as an observation block (2026-10-04): used, and no effect

**Question.** The expert review ([`expert_review_E7.md`](expert_review_E7.md)) found card-play leaks
at specific spots, and P30's probes found that RL never puts what a card does on this board into the
trunk ([`P30_card_board_targets.md`](P30_card_board_targets.md)). If the network is handed those
effects directly, does card play improve?

**Verdict.** No. With the block alone, or the block plus a head that feeds it straight into the
card and play-mode scores, strength is level with the control after 80M steps, and the census
spots do not move. The head is used (it changes card play), but its changes do not follow the
event values it is given.

Fork experiment, branch
[`feat/card-effects-obs`](https://github.com/JamesYouL2/DeepStruggle/tree/feat/card-effects-obs)
(`8f1e8c5`). It adds an observation block, so it is **not proposed for main**.

## The block

`obs_features` bit 2, `CARD_EFFECTS`, appended after the 3,824 base floats: 110 cards × 18 floats
= 1,980. Filled only at the decider's card decisions (choosing a card outside an event, and
choosing how to play it), and only for cards in the decider's hand; zero elsewhere. Per card:

* whether the event can fire;
* Ops reach: effective Ops, countries and battlegrounds the Ops could bring under control, best
  coup chance overall and on a battleground;
* what the event does here: change in VP and DEFCON, the six regional scoring margins,
  battlegrounds controlled by each side, total influence of each side.

These are exactly the P30 C2 labels (`ai/training/card_event_targets.py`), ported to C++ and checked
equal to the Python labeller on the same states and dice. Events are played out on a copy of the
state in which the cards the decider cannot see are redealt, so the block never reveals the
opponent's hand, and averaged over 4 dice seeds. The C++ labeller takes 0.27 ms per position
against 5.9 ms for the Python one.

**The head** (`--ladder-card-effects-head 64`): a per-card MLP over [the card's 18 effects, its 14
base card features, a projection of the trunk]. It outputs a correction to the card's score and to
the 5 play-mode scores (the latter only for the card being played). Its output layer starts at
zero, so a warm start plays exactly as its parent.

## Arms

All warm-start from E7-04-44@1200M with E7-75-45's flags (lr 3e-5 constant, seed 45, opponent pool
0.3; [`E7_playout_pg_finetune.md`](E7_playout_pg_finetune.md)), 80M steps, one RunPod RTX 4090,
engine `26686521`. Finals in release
[`e7-90-92-45`](https://github.com/JamesYouL2/DeepStruggle/releases/tag/e7-90-92-45).

| arm | change | steps/s |
|:---|:---|---:|
| E7-90-45 | block + card-effects head 64 | 27-31k |
| E7-91-45 | block only (read by the trunk) | 27-31k |
| E7-92-45 | control on the same commit | 33k |

E7-92-45 reproduced E7-75-45's weights bit for bit, so the two controls are one network rated
twice, which gives the noise of the rating itself.

## Strength

Round robin on CI (`37178016663`), 2,000 games per side per pairing, temperature 0.1:

| model | Elo | overall |
|:---|---:|---:|
| E7-90-45 block + head | 1504 | 50.3% |
| E7-75-45 control | 1501 | 49.8% |
| E7-91-45 block only | 1499 | 49.2% |
| E7-92-45 control (same weights as E7-75-45) | 1496 | 48.9% |

The identical controls finish 5 Elo apart; every gap in the table is that size or smaller. Pooling
the two controls (8,000 games each way): block + head about 50.6%, block only about 49.2%, each
about one standard error from 50%.

## Card play

Doctrine census, greedy self-play (CI `37178029849` control, `37178021132`, `37178025152`):

| rule | control | block + head | block only |
|:---|---:|---:|---:|
| US events Star Wars when ahead in space | 0% | 0% | 0% |
| USSR events OPEC when it scores 5+ VP | 21% | 15% | 16% |
| US events Alliance for Progress when it scores 5+ VP | 15% | 14% | 23% |
| USSR events Che well over half the time | 4% | 1% | 2% |
| Terrorism evented when behind or after Iranian Hostage Crisis | 6% | 2% | 4% |

**Is the head used?** On 5,808 card decisions from the control's own games: E7-90-45's card choice
is at KL 0.21 from the control and its play-mode choice at 0.28, against 0.19 and 0.18 for the
block-only arm (top choice the same 88.8% / 90.7% of the time). The head shifts legal cards' scores
by about 1.1 logits. But where an event can fire, E7-90-45's change in event probability is
uncorrelated with the block's event-VP column (r = +0.006) and DEFCON column (r = −0.008). There is
no seed-matched baseline for how far two runs drift in 80M, so the KL figures compare the two arms
only.

## Reading

* **The information was there, was used, and did not help.** The card mistakes are not a matter of
  not knowing what a card does now: the block states it exactly. They are a matter of valuing it,
  which depends on what follows. A one-step description cannot carry that.
* This agrees with C2: its card-event head teaches the trunk the same quantities and moved strength
  only slightly ([`P30_branch_arms_c2_play_mode_temp.md`](P30_branch_arms_c2_play_mode_temp.md)).

## Reproduction

`--obs-features card_effects [--ladder-card-effects-head 64]` on the branch above; E7-75-45's flags
otherwise, recorded in each run's `metadata.json` (in the release). Census and tournament run on
branch `exp/card-effects-eval`, entrants `hf:relpt:<tag>/<run>@80M.pt`, which keeps a torch
checkpoint (the ONNX export reads only the base observation).
