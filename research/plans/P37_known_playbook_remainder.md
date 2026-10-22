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
