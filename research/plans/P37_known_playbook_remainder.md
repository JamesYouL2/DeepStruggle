# P37 — what the known playbook still holds: lost pieces, plasticity, representation (2026-10-22)

**Question (owner).** With the scaled flagship running and search in the loop closed, what known
techniques are untried here, and is it time for research mode?

**Answer.** The playbook for "better targets for the same loop" is exhausted, by controlled tests
(P34, P36). What remains falls into three groups: validated pieces that fell out of the recipe
(cheap, low ceiling); training-dynamics techniques aimed at the plateau's *measured* mechanism, loss
of plasticity, never tried here (cheap, moderate prior); and representation and function-class
changes that are known in the field and new here, each a from-scratch run (the owner's call on
input changes). Research mode starts only if the scaled flagship reads level, and then with memory.

## The evidence the second group rests on: what a plateau network cannot fit

P36c trained three students on the same 200k entry-decision targets -- the keyed root's improved policy,
`softmax(logit + σ(completed Q))` over the legal moves, at 128 keyed worlds -- with the policy heads
trained and the value path frozen, 60 epochs, to convergence. 41.4% of the positions were
**departures**: the root's top move differs from the network's own argmax, so the target asks the
student to switch its top move. On the *training* departures:

| student | switches its argmax to the root's move (training set) | held out |
|:---|---:|---:|
| the plateau network (A13+R48's SWA) fine-tuned, lr 3e-4 | **21.9%** | 15.2% |
| the same, lr 1e-3 | **40.9%** | 20.2% |
| the same shape from fresh weights | **96.4%** | 24.7% (chance 26.6%) |
| a deep 8.7× network from fresh weights | **98.9%** | 23.6% |

Same architecture, same data, same optimiser, same epochs: fresh weights memorise the targets, the
plateau weights cannot even after 60 epochs at a learning rate three times the trainer's. What the
fine-tuned network did instead on held-out departures: kept its own move 64%, the root's 20%, a third
move 16%, and it left the right move on 17% of the positions where it had agreed with the root. The
soft target's gradient is not small at a saturated logit (cross-entropy's gradient is p − target), so
this is not the PPO (1 − P) effect; it is the trunk's features no longer able to express a changed
ranking -- the signature the continual-learning literature calls **loss of plasticity** (Nikishin et
al. 2022; Lyle et al. 2023; Dohare et al. 2024). Two older readings agree: the plateau is partly
optimiser noise (the 4× minibatch and the LR step-down each gave +15–25 once, R46/R47), and grown-in-
place capacity (A14, A18) is active but only sharpens. None of the plateau's remedies in that
literature has been tried here.

## Group 1 -- validated pieces that fell out, and never-run flags

* **Advantage filtering** (`--adv-filter-quantile`): settled positive on E3, +25 / +24 on two seeds
  (E3-07), then absent from every E4-onward recipe with no recorded decision. One 400M branch from
  the A13+R48 plateau against the unchanged control, and a late-switch cell in the flagships if it
  passes. The categorical value head from the same E3 plan settled negative (−28) and stays out.
* Closed by the record, listed so they are not re-proposed: EMA weights as the acting policy (R6,
  lost to the SWA), the mid-game start pool (harmful as configured on E3), PFSP and the KL knobs
  (level, P35), the logit caps, the league, human injection.

## Group 2 -- plasticity: the plateau's mechanism, never targeted (`P37a`, then branches)

* **`P37a`, the plasticity probe (CPU, hours).** Lyle's measure: the plateau trunk against a fresh
  trunk of the same shape, each with a fresh head, fitting (i) random labels on real positions and
  (ii) P36c's targets, frozen-trunk and full fine-tune; the fraction of dormant units in the trunk
  (Sokar et al. 2023) alongside. **Read:** if the plateau trunk fits random labels as well as a fresh
  one, P36c's result was target-specific and this group is closed; if it fits markedly worse, the
  loss is measured and the branches below are aimed at a known cause.
* **The branches** (each 400M from A13+R48's plateau, against the unchanged control, raw and under
  the corrected keyed root; the stability panel; R codes at launch). They are meant to act at a
  plateau, so the branch form is the right one for once:
  - **last-layer resets**: the policy and value heads re-initialised (the trunk kept) every N steps
    with a short re-warm (Nikishin et al.);
  - **shrink-and-perturb**: weights scaled toward zero and perturbed with fresh-init noise once at the
    branch point (Ash & Adams), a cheap single cell;
  - **regularisation toward the initial weights** (L2-init, Kumar et al. 2023), a standing term;
  - **dormant-unit recycling** (ReDo, Sokar et al.), if `P37a` finds dormant units.
* **Decision rule:** adopt by the per-seat rule; a plateau that *moves again* (a positive slope over
  the 400M windows where the control is flat) counts even if the end point is level, because the
  question is whether training can continue, not the 400M gain. Any pass goes into the flagships as
  a late switch and into the from-scratch recipe as a standing term.

## Group 3 -- representation and function class (from scratch; the owner's call on inputs)

In priority order, each a from-scratch run competing with the flagships for a slot, so none before
the scaled flagship's plateau read:

1. **Memory** -- an LSTM or transformer core over the game's history (AlphaStar's design). The belief
   probe says the information is in the opponent's choices (+12% log loss over the prior), the oracle
   root says it is worth +53 Elo to the search, and only a memoryful policy gets it into the network
   itself. The largest untested lever; an architecture and input decision the owner holds.
2. **An opponent-prediction auxiliary** -- the trunk predicts the opponent's next move from the
   observation (self-supervised, no new input). Cheap; pushes the trunk toward the opponent model
   the belief probe found absent. Low-to-moderate prior.
3. **Per-side heads or trunks** (P35's note): heads cheap; trunks double parameters at no cost per
   step, worth it only if the flagship shows capacity pays.
4. **Strategy conditioning for diversity**, after P35's `P35b` measurement.
5. **The standard AlphaZero / Gumbel form** -- CE-only policy loss, every move searched by a 16-
   simulation Gumbel tree, self-play from the search policy, value from outcomes; the one search
   configuration with literature standing, never run (P36's note). Two days to a 1,000M matched read;
   low-to-moderate prior, and the hidden-information literature points away from it.

## Sequencing

| when | GPU | CPU |
|:---|:---|:---|
| now (scaled flagship alone, ~31 h to 12,000M) | -- | `P37a` plasticity probe |
| the flagship's plateau read | the encoder flagship resumes in slot 1; **slot 2**: group 2's branches two at a time (shrink-and-perturb and L2-init first, resets and ReDo by `P37a`), then advantage filtering | |
| after the branches | group 3 by the flagship's verdict: capacity pays → per-side trunks and the opponent auxiliary as from-scratch cells on the scaled shape; level → **research mode: memory first** | |

Group 2 and the advantage-filtering branch need one slot for about two days and do not delay the
encoder. Group 3 waits for the capacity verdict because its cells inherit the winning shape.

## What is not here

Anything that changes targets for the current loop (P34, P36: closed); more search compute; the
belief root (not adopted); the belief input until pure search is settled (owner).

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

## `P37a` result (2026-10-22): the plasticity loss is in the policy head, not the trunk

`ai/eval/plasticity_probe.py`, CPU; `data/reports/p37a_plasticity.json`. 20,000 positions from P36c's training games
(decoded on this build after checking: every state's bytes round-trip, and the labelling network picks the recorded
argmax at 100.00%). Adam 1e-3, batch 512, to convergence (under 0.3 points in 5 epochs, cap 40). "Fit" is the
training-set accuracy; for P36c's 128-world targets, "departures" is the share of positions where the target's top move
differs from the network's own argmax on which the fitted policy now picks the target's.

| trunk | head | random labels, trunk frozen | random labels, full fine-tune | P36c targets, trunk frozen (departures) | P36c targets, full fine-tune (departures) |
|:---|:---|---:|---:|---:|---:|
| plateau (A13+R48 SWA 12,730–12,800M) | **its own trained head** | **33.3%** | **80.0%** | 74.0% (**44.9%**) | 97.7% (96.4%) |
| plateau | a fresh head | **98.9%** | 99.7% | **97.8% (97.0%)** | 97.9% (97.1%) |
| fresh, same shape | a fresh head | 91.1% | 99.2% | 95.6% (93.9%) | 96.1% (94.9%) |

* **The trunk keeps its plasticity.** With a fresh head the plateau's features fit random labels at 98.9% frozen --
  better than a fresh trunk's 91.1% -- and P36c's targets at 97.8% (departures 97.0%). The features are rich, not stuck.
* **The trained head has lost it.** The same features under the network's own head fit random labels at 33.3% with the
  trunk frozen and 80.0% fine-tuned, and switch only 44.9% of the departures with the trunk frozen -- P36c's failure
  (22–41% of departures on 180k positions) reproduced at the head. With the trunk free and only 20,000 positions the
  own head does reach 97.7%, so on P36c's 180k the fine-tune's ceiling was the head's, compounded by the data.
* **The head's signature** (4,096 positions): weight norms 20–50x a fresh head's (`policy_head` 390 vs 20,
  `branch_head_net` 220 vs 6.5, `play_mode_head_net` 251 vs 6.6), **13.5% dormant units at τ = 0.1** (6.6% at 0.025;
  a fresh head 0%), and saturated outputs (median top-1 minus top-2 logit 1.82 against 0.20; mean top probability 0.77).
  The trunk's dormant fraction is small: 2.5% at the plateau, 1.1–1.2% along the heads run (400M 0.7%), 0.1% fresh.
* **Read for group 2.** Not target-specific -- the loss is real and measured -- but it sits in the last layers, which is
  what **last-layer resets** (Nikishin et al.: re-initialise the heads, keep the trunk, re-warm) target directly; it is
  promoted from conditional to first. **Shrink-and-perturb** and **L2-init** act on the measured weight-norm growth;
  **ReDo** has something to act on in the head (13.5%) though little in the trunk, so if run it is aimed at the heads.

## `P37a` on the E8 recipe search (2026-10-23): the scaled network's trunk loses rank

`/workspace/data/reports/p37a_e8_2400M.json`; the same probe (20,000 of P36c's positions, trunk frozen, Adam 1e-3, batch
512, cap 40) on 80M-window SWAs: the E8 scaled network `E8-A25-R47-S44` (A25: A22 with the trunk norm, 16k minibatch)
and the E8 control `E8-A8-R1-S44-2` at 2,400M, beside the E7 heads run at 2,000M and 6,400M.

| network | random labels, own head | random labels, fresh head | P36c targets, own head (departures) | head dormant τ 0.1 | head norm / fresh (`policy_head`, `play_mode_head_net`) |
|:---|---:|---:|---:|---:|:---|
| E7 heads run, 2,000M | 47.2% | 98.4% | 74.6% (59.2%) | 3.8% | 9.2x, 21.3x |
| E8 control, 2,400M | 44.3% | 98.7% | 71.1% (53.9%) | 4.7% | 10.0x, 22.7x |
| E7 heads run, 6,400M | 34.6% | 98.2% | 67.4% (46.0%) | 5.7% | 15.5x, 29.9x |
| A13+R48 plateau (P37a) | 33.3% | 98.9% | 74.0% (44.9%) | 13.5% | 20x, 38x |
| **E8 scaled, 2,400M** | **37.3%** | **59.6%** (71.7% at 80 epochs) | **59.8% (40.3%)** | 7.8% | 7.3x, 16.0x |

* **A8's head loses plasticity gradually** along the heads line (own-head random-label fit 47 → 44 → 35 → 33%), its
  trunk does not (a fresh head fits its features at 98% throughout).
* **The scaled network's trunk loses it too.** A fresh head on its frozen 2,400M trunk fits random labels at 71.7% after
  80 epochs, where a fresh, untrained trunk of its shape reaches 88.1% (68 epochs) and a fresh A8 trunk 93.5% (39) --
  A8's trained trunk improves on its fresh self (98.7%), the scaled one falls below it.
* **The mechanism: the residual stream still grows, in a few dimensions, and the norm compresses the rest.** Before
  the trunk norm its RMS is 1.3 / 8.2 / 25 / 37 at 0 / 400 / 1,600 / 2,400M, with its largest dimension 46x the median
  one; after the norm the features' effective rank (exp of the singular values' entropy, 4,096 positions) falls 64 →
  24 → 21 → 22 of 768 (top-10 components 51 → 70% of the variance), where A8's rises 51 → 94 of 480 (6,400M: 95). The
  trunk norm stopped the logit blow-up, not the growth: the heads read a 768-wide vector with ~22 effective dimensions,
  a quarter of A8's ~94, so the extra capacity is not used.

## The last-layer reset on the E8 control (2026-10-25)

`E8-A8-R1-S44-2@10010M+R87`: the A8 control on E8 (ENG-5), plateaued at 36–40% against A13+R48's SWA from 5,600M to 10,000M, branched at 10,010M with the policy head re-initialised once (`--resume-reset-policy-head`: `policy_head`, `pe_trunk`, `pe_country`, `branch_head_net`, `play_mode_head_net` get a fresh network's weights and lose their Adam moments; the trunk, the value heads and pi_ref are kept), 800M on the control's recipe. The plateau is the control's own band, so the read is whether it moves again: reads at 10,200 / 10,400 / 10,800M against A13+R48's SWA, the head-norm trajectory in the training log (`head_norm_over_fresh_*`). Value heads are not reset: P37a measured the loss in the policy head, and a fresh critic would void the advantages the new head learns from.

**10,200M: 31.4%** (US 25.4 / USSR 36.3; the control 38.4% at 10,000M, its band 36–40%): most of the strength back within 190M of a fresh head; the head's norm regrows fast (policy head 1.65x → 3.2x a fresh head's, play-mode head 3.6x → 7.7x over 10,010–10,210M), entropy back to 0.49, the kept critic steady (explained variance 0.85–0.87).
**10,400M: 34.6%** (US 28.9 / USSR 39.5) -- recovering (10,200M 31.4%), not yet back in the control's 36–40% band.
**10,800M: 36.5%** (US 31.2 / USSR 41.0) -- back inside the control's 36–40% band, not above it. **Verdict: a single last-layer reset does not move the plateau**: a fresh policy head on the plateau trunk recovers to the same level in ~800M (31.4 / 34.6 / 36.5% at 10,200 / 10,400 / 10,800M), its weights regrowing as before -- the ceiling is not the head's plasticity alone.
