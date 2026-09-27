# E5-11: a locked US opening that costs 5 points, and a critic read from the side it was never trained on

The owner, playing E5-11 in the workbench, reported two things on 2026-09-27: the critic moves a lot
during a setup that is itself nearly deterministic, and the US always opens with Greece, with
probability 1.

## The critic during setup is read from the untrained side

Training records `v_win` only from the **acting** player's perspective (`ai/training/nash_pg.py`,
around line 840). The value head has never been trained on a position from the view of the player
who is not moving. The replay trace (`ai/eval/policy_readout.read_critic`) and the workbench
(`web/ui/src/analysis/readout.ts`) read it from both views. The workbench's value curve and its
Δv chip plot the **US** view (`web/ui/src/trace_view.ts:98`), so while the USSR moves the curve shows
the untrained reading.

In seed 1 of the E5-11-43@560M self-plays, during the USSR's six setup placements, the US view reads
+0.45, +0.38, −0.42, −0.49, −0.57 and −0.18, while the USSR view holds at 0.15–0.18. The reverse
happens during the US setup. Over the four replays, for consecutive steps by one mover:

| mean \|Δv\| per step | mover's view | other side's view |
|:---|---:|---:|
| setup | 0.037 | **0.165** |
| rest of the game | 0.050 | 0.099 |

The swings the owner saw are extrapolation by a head that was never trained on these inputs.
The mover's own reading is stable through the setup.

## The US opening is locked, and the lock costs games

E5-11-43@560M opens the US setup with **Greece 2** at p = 1.000. The full opening is Greece 2,
Italy 2, France 1 and West Germany 4 in 1,731 of 2,000 deals, and the rest differ only by one point
from West Germany to Iran. The human corpus essentially never places setup influence in Greece: its
means are West Germany 3.58, Italy 3.00, France 1.28 and Iran 1.03. At p = 1, sampling at temperature
1 never tries anything else, so training cannot find out whether something else is better.

`tools/scripts/setup_oracle.py` plays each deal up to the US setup with the checkpoint, forces an
opening, and plays out once per branch with the same seed in every branch (2,000 deals):

| US win rate, forced opening − own | E5-11-43@560M (own: Greece 2, Italy 2, France 1, WG 4) | E5-11-44@640M (own: Canada 2, Italy 2, WG 4, Iran +1) |
|:---|---:|---:|
| own | 42.6% | 47.0% |
| WG 4, Italy 3, France 2 (seed 43's opening with Greece's points moved) | **+5.4 ± 1.5** | +1.3 ± 1.4 |
| WG 4, Italy 3, France 1, Iran +1 (human) | **+4.5 ± 1.5** | +1.3 ± 1.4 |
| WG 4, Italy 3, Iran +2 (human) | +2.0 ± 1.5 | +1.0 ± 1.4 |

The model, playing both sides, wins about 5 points more as US from a sensible opening than from
its own. Seed 44's lock (Canada 2) costs little or nothing by this measure. The Denmark 7 opening of
E5-12-43 ([`E5_12_setup_credit_at_flat_temperature.md`](E5_12_setup_credit_at_flat_temperature.md))
is the same mechanism in a worse place.

## Reading

* **Setup exploration collapses.** The setup is about 15 decisions played once per game, and it is
  pushed to p ≈ 1 early. After that no gradient reaches the alternatives. Whatever opening the
  policy locked into early stays, good or bad.
* **The critic is single-perspective by construction.** Anything that reads it from a fixed side
  (the workbench curve, a search from the opponent's view) sees an untrained output half the time.
