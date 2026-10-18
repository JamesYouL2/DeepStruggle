# P35 — seven checks toward a much stronger base model: capacity, belief, training dynamics (2026-10-18)

**Goal.** After P34 closed (the critic is at the greedy-value ceiling; the leaks the searched player
was thought to keep were the judge's own follow-up), the owner's priorities for a significantly
stronger base model, in order: a transformer encoder at scale, capacity at saturation, a belief over
the hidden state, the KL regularisation checks, search-computed advantages, prioritised opponent
sampling, strategy conditioning for diversity. This plan gives each a concrete arm, says what runs in
parallel on one GPU with two training slots, and names the interactions that decide the order.

**What the record fixes before any of it.** The policy's argmax loses ~180 Elo to its own critic over
sixteen playouts of its own top four, and that gap is thousands of two-to-four-point corrections with
no rule in them. A judge built from the network can no longer find systematic errors. Every capacity
read so far was taken before any line's plateau (A3 at 560M, tokens at 400M, lines plateau at
4,000–12,000M), and width was the only branch still rising at 8,000M. The KL term is a 2%-of-loss
trust region on a three-iteration-old reference (the reference refreshes every 200k steps, an
iteration is 65,536), never ablated on E7. None of these seven has been measured where it matters.

**Labels.** `P35a`… for measurements that train nothing; every arm is an A or R code assigned at
launch by `run_codes.py`; cells are named here. Observation changes (items 3 and 7) are the owner's
decision and are marked as such.

## The seven items

### 1. The transformer encoder at scale (A code; from scratch)

* **What.** Countries and cards as tokens with attention between them (A6's `--ladder-token-layers`,
  d = 128, 2 layers to start), on top of A8's card-conditioned heads, the current recipe (pool, 16k
  minibatch, the gap penalty switched on late as the ladder does), trained **to its own plateau**
  by the plain line's rule (the mean of the last two 400M windows within +1.5 of the two before).
* **Why.** AlphaStar, Ataraxos and DeepNash all attend every entity to every other; here the token
  variant was judged at 400M on a line that plateaus at 4,000M, so it was never tested.
* **Cost.** ~9× the compute per step of A8 at 2 layers; ~15–20k steps/s with `--compile-update`;
  a 12,000M plateau is 8–10 days of one slot. Checkpoints every 400M, so partial reads exist and
  the run can be stopped by the rule.
* **Cheap early read (grown in place, 400M).** Token layers added at identity onto A13+R48's
  plateau (the A13 / Cell A pattern: zero-initialised projection, day-one play unchanged). Known
  false-negative risk: Cell A's region attention was active but only sharpened a saturated policy.
  A positive here is a strong signal; a level result does not cancel the from-scratch run.
* **Read.** Matched-step SWAs against the heads line (`E7-A8-R1-S44`) at 400M windows raw, then the
  plateau's SWA against A13+R48's under the keyed root. Adopt if its plateau beats A13+R48's by the
  per-seat rule raw and searched.

### 2. Capacity at saturation (A code; from scratch)

* **What.** The *current* architecture scaled, no tokens: the 512-wide projections (A13), fusion
  width 480 → 768 and 4 residual blocks (A3's shape halfway), the same heads and recipe, to its own
  plateau. The control for item 1's encoder: it separates "more parameters" from "attention".
* **Cost.** ~3× A8's compute per step, ~35k steps/s; a 12,000M plateau ≈ 4–5 days of one slot.
* **Cheap early read.** Residual blocks grown at identity onto A13+R48's plateau, 400M (the A14
  value-block pattern applied to the trunk). Same caveat as item 1's.
* **Read.** As item 1, and head to head with item 1's plateau.

### 3. A belief over the hidden state (`P35a`, then a search change, then the owner's call)

* **`P35a`, the probe (CPU, a day).** A belief network on logged self-play histories of A13+R48
  (the public record: cards each side played and when, hand sizes held over, discards, removed cards,
  the board) predicting the opponent's hidden hand (sigmoids over the hidden pool). Scored on held-out
  games against the uniform prior (log loss 0.3914) and the public-information probe (0.3780, the
  trunk's level; `ai/eval/belief_probe.py`). **Decision:** a gain at least three times the public
  probe's (≤ ~0.35) opens 3b and 3c; less closes the item for both the search and the policy.
* **3b, belief-weighted worlds for the search (no observation change).** The rollout root deals its
  worlds from the belief net's posterior instead of uniformly (importance-weighted or sampled;
  `ts.determinize` takes a per-card weight). Measured as a player: the keyed root with belief
  worlds against the keyed root, strong tier (1810). Adopt by the per-seat rule.
* **3c, a belief input for the policy (the owner's decision).** The compact form is the belief net's
  posterior as an appended `obs_features` block (110 floats, the per-card probability the opponent
  holds it; the base 3,824 unchanged, every checkpoint keeps running). Cost: 110 floats and one
  small forward per step. Precedent: the v2.1 turn-history slice was unread at 80–240M on the old
  architecture, so a *raw* history slice is not the form; a learned posterior is. If approved it goes
  into items 1 and 2's from-scratch runs rather than into a branch, since a saturated policy will
  not learn to read a new input (the Cell A lesson).

### 4. The KL regularisation checks (R codes; 400M branches from A13+R48's plateau)

* **4a, η = 0.** Does the term still earn its place at the plateau? On E3 at 80M removing it cost
  169 Elo as a stabiliser (no cycles found either way); on the current line it is 0.0006 of a
  0.035 policy loss while the gap penalty takes 46% of the gradient.
* **4b, the slow reference, `--ref-update-freq 5000000`** (75 iterations instead of 3) at η 0.1.
  Harmful from scratch on E4 (−168 / −68 at 80M) and the best late arm in the E4 record when
  switched on after 160M (+78 to +108 at 240M); never tried on E7 or from a plateau. E3-31-28
  collapsed under it by KL domination, so `kl_term` and entropy are watched.
* **Read.** Raw and under the keyed root at 80M windows, per seat, against the unchanged control;
  the stability panel. **Decision:** 4a level → the term leaves the recipe and the from-scratch
  runs use Ataraxos's annealed reverse-KL damping instead; 4b positive → adopted as a *late switch*
  in every from-scratch run (on after the first plateau), never from step 0.

### 5. Search-computed advantages for the policy gradient (R code; a build, then one branch)

* **What.** At one in eight of the learner's *entry* decisions (card choice, headline; ~3.4% of
  decisions) the keyed root (k = 4, 16 worlds, `earlier`+own) on a **slow frozen snapshot** of the
  learner (refreshed every 50M steps, not with π_ref, to keep the R29 self-referential loop out)
  gives Q(s, a) for the four candidates. The policy gradient at those decisions uses the root's
  advantage, Q(s, a_taken) − mean_a Q, when the taken action is a candidate, and adds an
  expected-advantage term Σ_a π(a|s)(Q(a) − V) over the candidates. **Not** a cross-entropy toward
  the root's argmax: that form failed seven times on near-ties; this one tells PPO which of its own
  moves is better and by how much.
* **Cost.** ~30× the trainer's inference, ~5–10× slower; 400M ≈ a day of one slot.
* **Read.** Raw and searched as usual; the own-search gain (the root over the raw network) is the
  number of interest, expected to shrink if the policy absorbs the root's ranking.

### 6. Prioritised fictitious self-play (R code; one 400M branch)

* **What.** `--opponent-pfsp` (built, `var` weighting, uniform floor), never measured. The pool's
  own note predicts it should hurt in pure self-play by starving the trailing seat of winnable
  games; `opp_pfsp_entropy` is the guard.
* **Read.** Per seat raw and searched; the seat balance. Adopt only if both seats hold.

### 7. Strategy conditioning for diversity (`P35b` first; the build is the owner's decision)

* **`P35b`, the measurement.** How diverse is the policy, and does the lack of it cost strength? The
  setup is one placement in 100% of games; the USSR seat is exploitable by +11.6. Measure the
  strategy entropy of self-play (setups, the regions invested by turn 3, coup and space usage by
  side) against the searched player's and the human corpus's distributions, and the exploit
  margin of a fresh seat-locked exploiter against the current line.
* **The build, if `P35b` says diversity is the missing piece.** AlphaStar's z: a strategy summary
  sampled per game (from self-play's own distribution plus designed openings), given to the policy
  as an appended observation block (~20–30 floats) with a pseudo-reward for following it. An
  observation change; the owner's call; into a from-scratch run only.

## What runs in parallel

One GPU, two training slots, evaluation matches and CPU probes beside them.

| days | slot 1 | slot 2 | CPU / CI | GPU evaluation |
|:---|:---|:---|:---|:---|
| 1–2 | 4a η = 0 (400M) | 4b slow reference (400M) | `P35a` belief probe; `P35b` diversity read; builds: item 5's trainer, 3b's weighted determinize, the token and scaled configs | 80M-window reads of 4a / 4b |
| 2–3 | 6 PFSP (400M) | 1's grown token layers (400M) | builds continue | reads |
| 3–4 | 5 search advantages (400M, ~a day) | 2's grown blocks (400M) | 3b's belief-world root, if `P35a` passed | reads; 3b's strong-tier match |
| 5 → ~19 | **1, the encoder from scratch** | **2, capacity from scratch** | | 400M-window reads of both |

The two from-scratch runs take the GPU for about two weeks, so everything that can change their
recipe or their input runs first: items 4 and 6 set the recipe, `P35a` with the owner sets the input.
Two arms at once give +20% total throughput over one, so the pair costs little over running them in
sequence. Nothing else trains during the flagship fortnight; branches of the flagships' own plateaus
come after.

## Interactions

* **1 ↔ 2.** Both are capacity; they share the control, the recipe and the launch day so their
  plateaus are comparable, and item 2 is item 1's control for "attention" against "parameters".
  Their value depends on the recipe (4, 6) and on the input (3c): a flagship launched before those
  read runs with a recipe known to be improvable, which is why the branches go first.
* **3 → 1, 2.** A belief input is an architecture decision that must be made before the flagships
  launch; it cannot be added to them later without a second fortnight. `P35a` is therefore on the
  critical path even though it trains nothing.
* **3b ↔ 5.** Search-computed advantages use whatever root is deployed; if belief worlds pass, item
  5's root should carry them, so 3b's read precedes 5's launch where the schedule allows.
* **4 ↔ 6.** Both change the update's dynamics (the anchor, the opponents). Never combined in one
  arm; if both pass, one combined 400M arm before either enters the recipe.
* **4 ↔ the gap penalty.** The penalty is 46% of the gradient; a slow anchor pulls against its
  spreading of the policy, and η = 0 removes the only other pull. Entropy and `logit_gap_frac` are
  read beside the usual panel in 4a and 4b.
* **4 ↔ 5.** Item 5's root reads a snapshot of the learner; with the reference refreshed every three
  iterations the R29 loop returns, which is why its snapshot refresh is its own, slow, cadence.
* **5 ↔ the critic.** The root's Q is the critic over playouts; item 5 inherits the critic's
  resolution (P34), so its gain is bounded by what sixteen worlds can rank. It cannot teach what the
  root cannot see.
* **6 ↔ 7.** PFSP concentrates the opponent draw; strategy conditioning widens the population.
  Opposite pressures on diversity; `P35b` is read before either is adopted.
* **Branch gains against from-scratch gains.** The gap penalty helped from a plateau and cost
  150–240 Elo from step 0; the slow anchor the reverse. Every recipe item that passes as a branch
  enters a from-scratch run as a *late switch* unless a from-scratch cell shows it safe early.
* **Grown-in-place reads against from-scratch runs.** Cell A and A14 were active and level: a
  saturated policy sharpens rather than corrects with new capacity. The grown reads of 1 and 2 are
  screens with a known false-negative risk and do not cancel the flagships; a positive grown read
  raises the flagships' prior, a level one changes nothing.

## Decision rules, collected

* Item 1 / 2: adopt a plateau that beats A13+R48's by the per-seat rule raw **and** under the keyed
  root; between them, the better plateau head to head.
* `P35a`: belief gain ≥ 3× the public probe's → 3b runs and 3c goes to the owner; else closed.
* 3b: adopt by the per-seat rule against the keyed root on the strong tier.
* 4a level → the term leaves the recipe; 4b positive by the rule → late switch in the flagships.
* 5: adopt by the rule with the own-search gain not larger than the control's.
* 6: adopt only with both seats holding and `opp_pfsp_entropy` not collapsed.
* `P35b`: a measured strength cost of low diversity (an exploiter margin above the current +11.6,
  or a searched-player distribution the policy cannot cover) → the z build goes to the owner.

## What is not in this plan

Evaluator or critic training on playout labels (P34, closed); policy distillation of search in any
form; the league; human cloning from the uncurated corpus; more search compute as a result.

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

**Pair 1, item 4 (2026-10-18): both level.** 400M from A13+R48's 12,400M state against the control's unchanged leg (12,400 → 12,800M), 80M-window SWAs, 1,000 deals a side:

| window | 4a η = 0 (R55) | 4b slow reference (R56) |
|:---|---:|---:|
| 12,400–12,480M | 51.7% | 48.4% |
| 12,480–12,560M | 49.0% | 50.4% |
| 12,560–12,640M | 48.3% | 50.4% |
| 12,640–12,720M | 48.8% | 50.1% |
| 12,720–12,800M | 49.9% (US 46.6 / USSR 53.1) | 49.5% (US 46.1 / USSR 53.0) |
| same, keyed root (16 worlds) | 48.4% (US 46.3 / USSR 50.6) | 49.1% (US 47.0 / USSR 51.2) |

Against the control's self-play bar (US 41.5 / USSR 58.0) each arm is about +5 as US and −5 as USSR: level, not a seat shift. Neither arm was unstable (entropy 0.74 / 0.78; 4b's KL 0.016, no domination). By the decision rule, 4a level means the KL term is not earning its place at the plateau, so the from-scratch runs would use annealed reverse-KL damping instead (owner's call before it changes a recipe); 4b level means the slow reference is not adopted as a late switch.

**Pair 2.** Item 6 (PFSP, R57) runs under MPS; item 1's grown token layers (A18) need the whole GPU (~9× A8's compute; 4.7k steps/s beside R57, which fell from 69k to 10–20k), so A18 runs alone after R57 instead of beside it.
