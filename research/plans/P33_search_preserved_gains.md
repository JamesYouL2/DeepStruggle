# P33 — base-checkpoint gains that survive search (2026-10-15)

**Status (2026-10-16).** The leaf fix is the night's result: playouts read at the mover's own next
decision (`leaf=own`, the owner's design) with `stop=earlier` beat today's root 53.05% over 4,000 games
and rate 1804 on the E7 leaderboard (today's root on the soup: 1791). R52's approach (code R49) is null
at 1 in 128 targets. k = 6 is level with k = 4 in play (49.7%): the candidate channel is closed, R51
dropped. Next: the pre-deal critic (below).

**Status (2026-10-15).** Diagnostics done ([`../log/P33_diagnostics.md`](../log/P33_diagnostics.md)):
the critic is the channel the search keeps (P33a), it is calibrated on greedy play (P33b) but at
chance on the decisive leak pairs (P33c), and only a lookahead to the turn's end sees them (card
choice and play mode, not placement). R49, R50, R55 dropped. Next: R52 (turn-end value targets) and
its control, proposed to the owner; a turn-end rollout root is level to slightly better in play (51.3%).

**Goal.** Improve the raw checkpoint in ways the rollout root keeps, so that the deployed player
(network + `rollout(k=4,worlds=16,horizon=4,rule=z2)`) gets stronger when the network does. The
owner's priority: "improvements of basic checkpoint that also are preserved under search."

**Labels.** Per the nomenclature (2026-10-15): `P33a`–`P33f` are this plan's measurements that train
nothing; `R49`–`R55` are training approaches and `A14` an architecture, reserved here and registered in
[`../architectures_and_recipes.md`](../architectures_and_recipes.md) at launch, where a run's name
carries them (`…+A13@12000M+R49`). `E` is the engine only.

**What this plan is not.** Not another way to distil the search into the policy (five online and
two offline forms are null, see below), not a search-side tuning plan (that is the fork's rollout
tuning), and not a product recipe (the product is the searched soup).

**The owner's standard (2026-10-15): a model that goes *much* further, not a one-off.** This
section says what the record allows on that question before the arms below.

* **Every single big lever has been pulled once and did not move the plateau on its own:**
  capacity (E6-10-44, 11.2M parameters against 3.2M, behind at matched games and at matched
  wall-clock), the token architecture (E7-09-44, level at 400M at ~9× the compute), human data
  (a 157–236 Elo handicap as injection; the BC prior washes out in 2% of training), the league
  (level), search as a policy teacher (seven forms), counterfactual credit (B4′). What did move it
  were readouts the policy could not express (the card-conditioned heads, ~3,000M of climbing)
  and optimiser noise (the minibatch, the step-down, souping).
* **The headroom is nevertheless large and it is measurable.** The rollout root's gain over the
  raw network keeps rising with worlds (+90 / +177 / +214 at 4 / 16 / 64) with no ceiling in
  sight, and the searched player still leaks ≥ 87 points a game by the paired-playout judge. A
  network near the search's ceiling would show neither. What the search adds is *evaluation
  depth*: four action rounds of consequences over sampled worlds, which a forward pass does not
  compute. So the model that goes much further is one that has **internalised deeper
  evaluation**, and the only training signal that carries it is a **value** signal -- the
  policy-side forms fail because a one-hot on a near-tie teaches "be less sure", while a value
  regression averages noisy labels across positions instead of fitting them one by one.
* **Why capacity did not pay, and when it should.** A larger trunk has nothing to fit when the
  policy is saturated (logits 19–33 nats, the PPO gradient ∝ 1 − P ≈ 0) and the critic's target is
  the λ-return of sampled games. In AlphaZero-style training the capacity is spent on the search's
  value and visit targets, not on the policy gradient. The order this plan follows is therefore:
  first make the training signal deep (R49/R50, R52, R53, R54), **then** re-test capacity under it (A14,
  and the token architecture from scratch) -- a capacity re-test on today's signal would repeat
  E6-10-44.
* **The full AlphaZero loop is out of reach at this scale, a sparse one is not.** The root costs
  ~910 network rows a decision against one for the trainer's own forward (~95k decisions/s), so
  searching every training move is ~900× the inference. Searching 1–2% of them is 10–20×, the
  regime R42 already ran in (9k steps/s). The two halves of that loop that have *not* been run
  here are value targets from the deployed search (R49) and the searched move being **played** in
  the training games (R54), not only used as a target. Those are the arms with a step-change
  shape; the diagnostics decide which goes first.
* **The one lever that stays with the owner: hidden information.** The observation carries no
  play history, so the network infers the opponent's hand from the board alone, and the search
  samples worlds uniformly from the unknown pool. Belief-state methods (ReBeL, DeepStack) are the
  literature's step change for hidden-information games, but any form of them is an observation or
  architecture decision, and the cheapest probe -- a hand-prediction head on the frozen trunk
  against the "deck or hidden" prior -- would first show whether the trunk already carries a belief.
  Listed as P33f, a probe only; the representation decision is the owner's.

## Where the record stands (main, 2026-10-15)

The three results the owner named, with the numbers behind them
([`../runs.md`](../runs.md), [`../checkpoints.md`](../checkpoints.md), the leaderboard):

* **A bigger minibatch helps.** R47 (A8 plain, 16,384) +22 Elo on the SWA and +40–48 on the raw
  network over the plain leg; R48 (the gap-penalty line, 16,384) climbs from 7,200M to a plateau
  at 9,200M, its SWA 1667 against the R34 soup's 1650. R46 (the LR step-down) gains the same: the
  A8 plateau was partly the iterate's noise.
* **A bit of width helps, not by a lot.** A13 (input projections 256 → 512, grown in place) alone
  is level with R47 at 8,000M; A13 on top of R48 climbs to 11,200M and rates **1679 raw, the
  top no-search model**, 12 over R48 and 29 over the soup.
* **The gain disappears under rollout search.** Under the same root the order reverses:

  | network | raw | + rollout root | the root adds |
  |:---|---:|---:|---:|
  | R34 soup `(R34,R34-S45,R34-S46)@7120..7200M` | 1650 | **1791** | +141 |
  | R39 SWA (CE toward the soup, through the rollout root) | 1676 | 1779 | +103 |
  | A13+R48 SWA @11,920..12,000M | **1679** | 1767 | +88 |

  Raw gains of +29 over the soup become −24 under search. What training added is, to within the
  measurement, what the search would have found anyway.
* **Rollout search does not distil.** Online: R39 absorbs ~25 of the root's ~140 Elo and the
  rest is the soup's raw policy (R45, the soup's *raw* policy as the target, is level with R39);
  R40–R42 (raw-snapshot teacher, 4× targets and weight, 64-world targets at weight 1.0) absorb
  nothing. Offline: one round of the root's picks is −13 Elo, soft steps are level
  ([`../log/E7_search_depth_and_value.md`](../log/E7_search_depth_and_value.md) §5–6). The
  diagnosis is in the targets: half the root's departures at 16 worlds are seed noise, and the
  signal is spread thin — +2–4 points of win probability each, over every kind of decision, no
  card dominant.
* **The searched player still leaks at least +86.6 points a game** at 12% of its decisions
  ([`../log/E7_searched_leak_R39.md`](../log/E7_searched_leak_R39.md)): card choice +36.6 (of
  which DEFCON 2 +26.9), influence placement +25.7 (the wrong country in the right region, late),
  play mode +10.7, headline +8.7. The better move is in the network's top 4 for 73% of it and at
  rank 5–6 for the rest; in two thirds of it the **search kept the network's move** with a better
  one among its own candidates. So the root's own judgement — 16 worlds, 4 action rounds of
  greedy play, then the critic — is what fails there, not the candidate set.
* **The game is far from solved for this network.** Play-time compute still buys a lot: the root
  at 64 worlds is +214 over raw against +177 at 16 and +90 at 4, with no sign of a ceiling; the
  gap over raw did not shrink from the heads soup to A13+R48. A network near the search's ceiling
  would show both.
* **Closed, do not retry:** RL continued from a soup's weights (R44 loses the soup's edge in
  10M steps); the gap penalty from scratch (−150–240 Elo; it is a late-branch recipe); every
  logit cap (A9–A12); B4′ as built; T2 with Gumbel k=4 @32 at 1 in 8 (R35, null, Gumbel gap not
  widened — but Gumbel is no longer the deployed search, so T2 is re-asked below with the root);
  more or cleaner policy targets from the root (R41, R42).

## The reading: what a rollout root keeps, and what it replaces

The root takes the network's top-k moves, plays each out with the network's **greedy** policy on
both sides through 4 action-round boundaries in 16 sampled worlds, reads the **critic** at the
stop, and keeps the network's move unless another leads by 2 SE. So a change to the network
reaches the searched player through exactly four channels:

1. **The candidate set** — whether the best move is in the top k. Recall, not ranking.
2. **The rollout policy** — the quality of greedy play inside the horizon, on both sides. Paired
   worlds cancel what both candidates' continuations share, so only line-specific play counts.
3. **The leaf critic** — the value read 4 action rounds ahead, and above all its *differences*
   between the candidates' leaves in the same world (common-mode error cancels under z2).
4. **The default** — the network's own move, which the root keeps at ~70–85% of decisions. A
   better argmax is kept only where the root does not find a 2-SE lead against it.

Everything the recent arms improved sits in channel 4 and, to a lesser degree, 2: the minibatch,
the width, the CE toward the soup and the gap penalty act on the policy's ranking among the top
few moves at the network's own resolution — exactly what the root re-decides. None of them was
aimed at the critic, and no arm since the heads line has measured its critic. Two facts say the
critic is the channel that matters: the root's leak is where its own judgement fails with the
right candidate in hand, and the only model whose searched rating stands clearly above the rest
is an **average of three seeds** — the one operation that reduces a critic's per-position noise
without changing what the policy ranks. R32, trained on Gumbel targets, is the counter-example in
the other direction: its raw rating rose and its search *lost* (−11), the critic having fallen
behind its prior.

A second, specific reason to distrust the critic at the leaves: **it is trained on the returns of
a different policy from the one the search plays.** Training rollouts sample at temperature 1
(`--rollout-temps 1.0 ×4`), the λ-return (λ = 0.98, γ = 1) mixes those sampled games' outcomes
into the target, and on the gap-penalty line the sampled policy is a much weaker player than the
greedy one (R34 at T = 1 scores 36.6% against the plain SWA; greedy 55.5%). The search's leaves
are reached by greedy play and ask "what happens from here under greedy play". Where sampled play
is riskier than greedy play — DEFCON 2 with a coup in the mix, a scoring card that must not be
held — the critic prices the sampled policy's accidents into the position. DEFCON 2 card choice is
the single largest item of the searched leak. This is a hypothesis, and §P33b measures it before
any arm is built on it.

**The measurement rule that follows.** From here every arm is read **under the root as well as
raw**: last-80M SWA, `rollout(k=4,worlds=16,horizon=4,rule=z2)` on both, 1,000 deals a side
against the searched R34 soup (1791) and head to head against the control's searched SWA. The
search-preserved gain is the searched head-to-head by the per-seat rule (P32's: above the
control's own self-play bar in one seat, not significantly below it in the other, rates read from
the JSON, [`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)). The raw
reading is kept so a raw loss is
not hidden by a searched gain. Cost: ~11 min per 1,000 rollout games on one GPU, ~25 min a read.

## P33a–P33d — diagnostics first (no training; ~4 GPU-hours in all)

Each one separates two explanations and decides which arms below run. Pre-registered reads.

### P33a — the leaf-critic swap

A two-network root: candidates and rollouts from network A, the leaf read by network B
(`RolloutRoot` gains a `leaf_model`; `rollout_root.py:_net` is one forward, so B is a second
forward on the same leaf observations; ~40 lines plus a `leaf=` field in the player grammar). All
A8/A13 models read the same 3,824-float observation, so any pair works.

| player | what it tests |
|:---|:---|
| rollouts A13+R48, leaf soup | does the soup's critic carry its searched lead? |
| rollouts soup, leaf A13+R48 | does A13+R48's critic cost it? |
| rollouts R39, leaf soup / rollouts soup, leaf R39 | the same on the 256-wide pair |
| the four single-network roots | the baselines |

1,000 deals a side each against the searched soup, plus the two swapped pairs head to head.
**Read:** if swapping the leaf moves the rating by more than swapping the rollouts (> 3 SE), the
critic is the search-preserved channel and the critic track is the route; if the rollouts carry it, channel 2
(greedy play inside the horizon) is, and R51 moves ahead of the critic track; if neither moves it, the searched
ratings are at a common ceiling set by the horizon and R52 (the longer-horizon value target) is
the route.

### P33b — the critic's calibration on greedy against sampled play

For R39's SWA, A13+R48's SWA and the R34 soup: 2,000 self-play games greedy and 2,000 at
temperature 1, the critic's value at every decision against the game's result. Report log loss
and the signed error by DEFCON, turn and side, on each kind of game (extend
`ai.eval.critic_boundary`, which already walks self-play and reads the critic). **Read:** the
hypothesis above predicts the critic is well calibrated on temperature-1 games and biased on
greedy ones, with the bias concentrated at DEFCON 2 and in the late turns, and larger on the
gap-penalty line than on the plain SWA. If the greedy-game error is not larger (< 2 SE of the
sampled-game error), drop R50 and R55 and keep R49 only.

### P33c — the critic on the confirmed leak positions

The searched-leak bank holds 1,122 confirmed positions with 2,048-pair playout labels for the
player's move and the best move. For each, the critic's value after each move (one forward each,
the mover's frame) against the playout difference, for the three networks. **Read:** the share
of pairs the critic ranks wrong and the correlation of its difference with the labelled one. If
the critic ranks the labelled pairs no better than chance on the DEFCON-2 card choices, that is
the leak's location and the critic track's target; if it ranks them right and the root still kept the wrong
move, the horizon or the world noise is the fault and R52 is.

### P33d — candidate recall

The root at k = 8 on 2,000 positions from R39's searched self-play, recording the rank under the
network of the move it picks. **Read:** the share of departures at rank 5–8. Above ~25% (the leak
census's 27% at rank 5–6), k = 6 at play time is a product change worth its +50% cost, and R51's
recall objective is worth an arm; below 10%, channel 1 is closed.

## R49 and R50 — the critic as the search's leaf (the main track)

Three forms, from cheapest to most faithful. Each branches from A13+R48's plateau state
(`resume_11210…` or its last resume) with its recipe (`--logit-gap-coef 0.001 --batch-size
16384`) and runs 400M with the control being the same continuation without the change, read at
80M windows under the rule above. The policy loss is untouched in all three, so none can flatten
the policy (R35's design).

### R49 — greedy-continuation value targets (T2 with the root)

`--search-value-beta β` with `--search-rollout-horizon 4`: at 1 in S learner decisions the
position is played out greedily by the **learner's own frozen snapshot** (not a teacher: the
target is "greedy play from here", which any snapshot of the line gives; the teacher question is
separate) through 4 boundaries in 16 worlds and the critic read at the stop; v_win's target
becomes (1 − β) × the λ-return + β × that mean. Build: the assertion at
`ai/training/nash_pg.py:416` refuses this pairing; lift it and route the value through
`RolloutTeacher` with a single candidate (the state, not k moves — a quarter of the root's cost).
Cost at S = 64: about R39's (which priced 4 candidates at 1 in 256), ~25k steps/s, 400M ≈ 4.5 h.
Cells: β = 0.5, S = 64 first; β = 1.0 at the same S if the first is positive and raw not worse.

**Why this and not R35 again.** R35 mixed in Gumbel k=4 @32 values at 1 in 8: a determinized
tree's root value, one world per halving phase, read by a critic one or two moves ahead — the
search that §1 of the depth log showed cannot choose and §4 showed loses ground past 256. The
target here is the deployed search's own leaf estimate, averaged over 16 worlds, 4 action rounds
ahead.

### R50 — a greedy value head, fed by greedy-only environments

A second value head `v_greedy` on the trunk, trained only on the returns of environments that
play **both sides greedily** (a `--greedy-env-frac f` of the envs, f = 0.25; those envs contribute
to the value losses only, never to the policy loss, so PPO's log-probabilities stay the ones the
actions came from). The root reads `v_greedy` at the leaf when the checkpoint has one, `v_win`
otherwise. No search cost; the price is a quarter of the rollout games carrying no policy
gradient (≈ −25% policy throughput; the exploration of the other 75% is unchanged). Build: an
env flag in the batch runner's temperature path (`--rollout-temps` is already per-env), a head, a
masked loss, the root's reader, `check_checkpoint_layout` unchanged (the observation does not
move). If P33b finds no greedy/sampled gap, skip.

### R55 — the single-head variant

As R50 but one head, mixed targets (f = 0.25 greedy envs feed `v_win` with everyone else). Fewer
parts; the critic then predicts a 3:1 mixture. Run only if R50 is positive and the owner prefers
one head; otherwise skip.

### Decision rule for the critic track

Adopt the first form whose searched SWA beats the control's searched SWA by the per-seat rule
with raw not lower by more than 1 SE, **and** whose own-search gain (its searched self over its
raw self) is not smaller than the control's. A positive P33c with a null critic track means the critic is
signal-limited in a way 400M of these targets does not fix; then the value-capacity arm (A14).

## A14 — value capacity, only if the critic track moves the critic's labels but not the games

A value-only residual block after the fusion (policy heads untouched), grown in place at identity
so day-one play is unchanged (the `--resume-widen` pattern, `ai/models/grow.py`): A14. The depth
log's value probe argued the critic is signal-limited, not capacity-limited (a small head on the
frozen trunk doubled the sibling correlation), so this waits for the critic track's evidence. One cell, 400M,
same reads.

## R51 — candidate recall (only if P33d says channel 1 is open)

A recall objective that search preserves by construction: the frozen-teacher CE already pushes
mass toward the teacher's full distribution; the specific form here is a **top-k hinge** — at a
searched decision, if the root's pick is outside the policy's top k, a loss on the margin between
the k-th logit and the pick's (nothing when it is inside). It cannot broaden the policy where the
root agrees (no gradient there) and does not ask the argmax to move, which is where R39–R42
failed. Cheap on R39's targets at 1 in 256. One cell, read under the root at k = 4 and k = 6.

## R52 — the horizon: value targets to the turn's end

*Since the diagnostics:* P33c's turn-end read is R52's premise, and it holds for card choice and play
mode (37.8% of confirmed leak pairs ranked wrong against ~47% for the critic). The stop is a setting
of the rollout batch (`stop = boundaries | turn | later`, `--search-rollout-stop`), not a horizon
sentinel. The text below is the plan as written.

If P33a finds a common ceiling and P33c says the critic ranks the leak pairs correctly, the root's
4-boundary horizon is what hides the value (scoring-card and China-card timing resolve at the
turn's end or later). The base-checkpoint version of a longer horizon is a critic trained on
**turn-end** greedy continuations: R49 with the stop at the turn boundary (`turn_pricing.price`
already stops there; 1 in 128 decisions to hold the cost). Judged under the 4-boundary root — a
critic that has learned what the turn's end looks like should make the 4-boundary leaves worth
more — and, as a product read, under a 6-boundary root.

## R53 — exact endgame values as critic targets (the one place noise-free labels exist)

The engine can enumerate the rest of the game where it ends in the current turn
(`tools/scripts/blunder_bank.py prove`: dice averaged, the opponent's replies searched; 400k–2M
nodes a position), and 23.5% of A13+R48's games reach final scoring, 22 of the 35 confirmed
blunders are in turn 10, and the US's last placement is the one recurring engine-proven leak.
Those positions have **exact** values -- the only noise-free value labels in the game -- and a
critic that is exact there bootstraps the turns before it through the λ-return. Build: a labelling
pass over the trainer's own turn-10 positions (a `--exact-endgame-frac` of the last-round
positions solved on CPU workers beside the rollout, the budget capped, unsolved ones skipped),
their v_win target replaced by the exact value, in the mover's frame, averaged over the mover's
unseen cards as `prove` does. Cost: the solver's node budget sets it; at 1 in 16 last-round
positions and a 200k-node cap it is a CPU-side cost only. **Read:** the critic's error on an
exact-labelled held-out set of turn-10 positions (the bank's 7 solvable rows plus fresh ones);
then, under the root, the share of games lost at final scoring and the US last-placement leak
(the census's 1.8 points). A product-side last-placement solver is the cheaper way to the same
points at play time; this arm is for the *critic*, whose exactness the search then reads at every
turn-10 leaf.

## R54 — act with the search in training (expert iteration, sparse)

R39–R42 used the root's improved policy as a **target** and left the rollouts untouched: the
training games were still the network's own, and their returns the network's own. In expert
iteration the searched move is *played*: the games get stronger, the λ-returns are the searched
player's, and the critic learns the searched player's value function -- the one the search
needs. Build: at 1 in S learner decisions (S = 64, ~1.5%) the rollout root chooses and its move is
forced as environment, exactly as a forced opening is (`learner = 0`, no policy gradient on it,
`ai/training/forced_openings.py`'s mechanism), with the improved policy as a CE target there as in
R39. The search's network is the learner's own frozen snapshot, refreshed with π_ref. Cost ~R39's.
**Read:** raw and searched as for the critic track; the own-search gain is the number of interest -- a network
whose games are played by the search should see the root add *less* on top of it than the
control's does (the gap closing from below) while its searched rating rises. **Caution, from
R29/R31:** the targets follow the learner here; the frozen snapshot and the refresh cadence keep
them still between refreshes, and KL(target ‖ policy) = search CE − target entropy is watched as
in P32 -- rising past 0.5 inside a refresh period stops the arm. A positive R54 is the first arm
with a from-scratch cell (S = 256, the heads recipe), since it changes the training distribution
itself.

## The pre-deal critic: both sides at the turn's end, and a search that stops there (planned, 2026-10-16)

**The owner's arm.** Train the critic for *both* sides at the turn's end -- after the last action, before
the next deal -- and search with playouts that stop exactly there.

**The point exists.** After a turn's last action the engine parks the game at a chance node
(`StateMachine::begin_turn_cleanup`: no decider, `pending_roll = TURN_CLEANUP`); everything after it is
deterministic (military-ops check, held-scoring check, China card, turn + 1, era cards, DEFCON + 1)
except **the deal**. Auto-advance drains it, so today it is neither trained nor read. The observation
already marks it -- decision type `ROLL_DIE` (7) has its one-hot slot (`observation.cpp`, globals
`DECISION_TYPE + 7`) -- so no observation change is needed.

**Why a leaf there should be better than anything measured.**
* No decider, so neither side's reading is off-side once both are trained -- the problem `leaf=own` works
  around by playing on.
* The mover's reading keeps its own hand (held cards) and hides the opponent's, as at a decision.
* **The deal is integrated by the critic, not sampled by the world.** Every leaf after the deal (today's
  turn-end, `own` at a headline) carries the deal's randomness in each world; at the pre-deal state the
  critic averages over it, removing that variance from the root's paired differences.
* It is the turn boundary P33c found the critic needs: the turn-end read ranked the leak pairs 37.8% wrong
  against ~47% at 4 boundaries.

**Build (engine untouched; a bindings change, needs the owner).**
1. *Runner:* during auto-advance, when an env reaches the `TURN_CLEANUP` park, record that state's
   observation from **both** sides (and the env and step) into a side buffer the trainer reads each step,
   then continue as today. The decision stream is unchanged.
2. *Rollout batch:* `stop = predeal` -- a playout stops at the first `TURN_CLEANUP` park of the decision's
   turn (or the game's end), read from the root mover's side -- and `stop = earlier_predeal`, the earlier
   of the horizon's boundaries and that park. Both builds, parity cases, the agent string and
   `--search-rollout-stop`.
3. *Trainer:* `--predeal-value-coef w` (0 = off): each pre-deal row joins the value loss with the
   λ-return of **that side's next decision** as its target (the opponent's row: the opponent's next
   decision), the terminal result if the game ends at the turn's end (a held scoring card, final
   scoring); a row whose next decision falls outside the rollout buffer is dropped. About 2 rows per env
   per turn: +3% value rows.

**Checks before the arm.** (a) The untrained critic at pre-deal states: log loss against the result for
both sides and their sum (`critic_consistency`, a pre-deal section) -- the baseline; (b) the pre-deal
leaf with today's critic in games (`stop=predeal` against `earlier`+own), which says what training must add.

**The arm.** A13+R48 from 12,400M (the control's end) with R48's recipe plus `--predeal-value-coef 1.0`,
400M; control: A13+R48 continued unchanged 12,400 → 12,800M. Two arms at once.

**Launched 2026-10-16** (owner): bindings `2ff1f00`, trainer `11d9aae`, run `…+A13@12400M+R50` (recipe
code R50) from 12,400M, ~1,350 park rows a rollout; control A13+R48 leg 4. **Baseline** (the untrained
critic, A13+R48 SWA 12,320–12,400M, 512 greedy games, 3,728 parks; `data/logs/temp/predeal_baseline_a13.out`):
log loss 0.491 (US side) / 0.500 (USSR side) against 0.496 for the decider's reading over all decisions;
v_US + v_USSR −0.043, mean |sum| 0.079 (0.138 at decisions) -- already about as good as at a decision.
**In games the untrained critic is a poor pre-deal leaf** (the R34 soup, 1,000 deals a side, against the
soup under `earlier`+own; `data/reports/predeal_base_*`): `stop=predeal` **41.6%** (40.3 / 43.0),
`stop=earlier_predeal` **38.9%** (38.5 / 39.3). Calibrated on average, it does not resolve the differences
between candidates' parks, a position it has never trained on; mixing park and 4-boundary leaves in one
comparison is worse still. The trained arm has ~8 points (~60 Elo) to recover before the pre-deal root
breaks even.

**Results (2026-10-16).**
* **The critic at the park** (final SWAs 12,720–12,800M, 512 greedy games each): R50 log loss 0.449 / 0.449
  (US / USSR side) against the control's 0.513 / 0.502; v_US + v_USSR +0.004 (|sum| 0.031) against −0.038
  (0.088). R50's critic reads the park far better, and the two sides nearly agree there.
* **The decision match** (R50 under `stop=predeal` vs the control under `earlier`+own, 2 × 2,000 games):
  49.2% and 50.8%, **50.0% pooled -- level**. Training at the park closed the untrained critic's gap (41.6%)
  but does not beat the best root.
* **Is the training needed?** A separate head on R50's frozen trunk does not beat its main head at the park
  (`park_head_probe`: 0.4588 against 0.4511; a correction gains 0.0035, < 1 SE) -- the trunk, not the head,
  is the limit there. And the control with only its value head refitted at the park (frozen trunk, 117,884
  rows, park log loss 0.4767 → 0.4664; `park_head_fit`) under `stop=predeal` scores **39.9%** (33.0 / 46.8)
  against the control under `earlier`+own, against R50's 49.2% on the same deals. A head fitted afterwards
  does not give the pre-deal search what joint training does: R50's training is needed to make the
  pre-deal root viable, and with it the root is level with `earlier`+own, not better.

**Reads.** (1) The critic at pre-deal states: both sides' log loss and sum, arm against control. (2) P33c's
leak pairs read at the pre-deal state. (3) Searched: the arm's SWA under `stop=predeal` and under
`earlier`+own against the control's SWA under its best root (`earlier`+own), per seat; raw head to head;
the best on the leaderboard with `--strong`. **Adopt** if the arm's pre-deal root beats the control's
best searched player by the per-seat rule with raw not lower by more than 1 SE.

## Cell A: region attention before the per-country head (2026-10-16/17)

The owner's suspicion: much of the placement leak ("the wrong country in the right region") is the network
unable to compare countries within a region -- each country's head reads only its own row and one shared
64-number context. **Build** (`06c40e1`, architecture code A14): each country's head input joined with its
map neighbours' mean, attending within its region (country ids are contiguous by region, so six slices),
added through a zero-initialised projection; grown in place on A13+R48 at 12,400M, R48's recipe, to 12,800M
(~31k steps/s against ~41k). **Result:** level with A13+R48's leg 4 -- SWA 12,720–12,800M raw 50.2%,
searched under `earlier`+own 49.4%. Not measured: whether the layer moved the placement decisions at all.

## The critic on both sides (design, 2026-10-15; not built)

**Why.** The trainer reads the critic only from a decision's decider; the rollout root reads leaves from
the root mover's side, so 15–55% of leaves (by stop rule) were read off the training distribution, where
the reading is optimistic for its side by ~0.08 and worse calibrated (log loss 0.51 against 0.49;
[`../log/P33_diagnostics.md`](../log/P33_diagnostics.md), "Is the critic self-consistent?"). A constant
shift does not repair it and averaging both sides gains nothing. The play-time fixes (`leaf=decider`,
`leaf=own`) avoid the off-side reading; this arm would make it trustworthy, so a leaf can be read from
the mover's side anywhere -- no extension plies, no discarded hand.

**What.** At each decision, also train v_win on the *non-decider's* observation of the same state, with
the negated λ-return as its target. Each side then predicts its own win probability from its own
information; their per-position disagreement stays allowed (different hidden cards), but their average
over hidden information is pulled into agreement -- the owner's consistency condition.

**Build.**
* The batch runner returns the other side's observation per env beside the decider's (a bindings
  change: one more `extract_observation_features` per env and step, written to a second buffer).
* The rollout buffer stores it for a subsample `--critic-both-frac f` of steps (f = 0.25: +25%
  observation memory) and the value loss adds those rows with negated targets at the same weight.
* The GAE recursion is unchanged (it bootstraps across a change of mover with two decider readings);
  a same-perspective bootstrap could be re-tested once the off-side reading is trained (the E3 test
  that found it worse bootstrapped off untrained readings).

**Cost.** One extra observation and, at f = 0.25, ~+25% of value-side trunk forwards and backwards;
policy throughput roughly unchanged.

**Reads.** (1) `critic_consistency` section 1: the off-side log loss should approach the decider's
without the decider's getting worse; (2) searched games with `leaf=mover` against the control's
searched SWA, per seat; (3) raw head to head, so a raw loss is not hidden.

**Risk.** Value capacity shared with the decider critic (A14's value block is the fallback); an arm on
A13+R48's plateau against the unchanged control, 400M.

## P33f — does the trunk carry a belief about the opponent's hand? (probe, CPU)

A head on the frozen trunk of A13+R48's SWA, trained on 50k self-play positions to predict the
opponent's hidden hand (110 sigmoids), against the "deck or hidden" prior (the pool's uniform
probability) and against a probe that sees the discards and the known cards only. **Read:** log
loss and the AUC per card. If the trunk beats the public-information probe, a belief is already
represented and a belief *input* would add little; if it does not, the gap is the ceiling that a
history or belief feature could lift, and the number goes to the owner as the cost/benefit of
that representation decision. No training arm follows from this plan.

## P33e — the setup under searched continuation (a check, ~40 min)

The network's setup (USSR Poland +3, Yugoslavia +3 in 100% of games; US West Germany +4, Italy
+3, Iran +1, Panama +1) looks wrong to a human (USSR Poland 4, East Germany 4, +1 Yugoslavia or
Austria; US West Germany 4, Italy 3, Iran 2). Two reads said the alternatives are level **by the
network's own continuation**: the CPU check (+0.5 ± 1.5 for the human US opening) and R43 (forced
openings as environment, no gain). Neither used a searched continuation. Here: the three setups
(the network's, the human one, the owner's forced-openings set) as forced openings, both sides
played by the rollout root, 500 paired deals each. **Read:** a setup that beats the network's by
> 3 SE under searched play is direct evidence that the critic misprices positions that resolve
over several turns — the same fault R49 and R52 target — and R43 is re-run after the critic track lands. Level again
→ the setup is a style, not a leak, and it leaves the record.

## Sequencing (two arms at a time)

1. **Now, no GPU conflict:** P33a–P33d and P33f (eval-only; P33a needs the small two-network build).
   One day of CPU/GPU-eval time in all.
2. **Then two arms:** R49 (β 0.5, S 64) and R54 (S 64) -- the two halves of the sparse loop, one
   value-side and one acting-side, from A13+R48's plateau, 400M, read at 80M windows raw and
   searched. R50 takes R54's slot if P33b finds the greedy/sampled gap and R54's build is not ready.
3. **P33e beside them** (eval only); R53's labelling pass built meanwhile (CPU).
4. **Then by the rules:** R53, R52, A14 or R51, one cell each, never more than two arms live. A
   positive critic track or R54 moves A14 (capacity under the deeper signal) ahead of the rest.
5. **Adoption, and the from-scratch question.** A positive form is carried to the next
   from-scratch run only as a *late-phase* recipe unless a from-scratch cell shows it is safe
   from step 0 -- the gap penalty's lesson. The from-scratch run that this plan aims at is the
   heads recipe plus whichever of R49/R50/R53/R54 passed, **then** the capacity re-test on that signal
   (A14 and the tokens), judged at saturation and under the root. That run is the candidate for
   "much further"; nothing before it is.

## Measure

* Raw: last-80M SWA, greedy, 1,000 deals a side, against the control's SWA and the R34 soup.
* Searched: the same SWA under `rollout(k=4,worlds=16,horizon=4,rule=z2)`, against the control's
  searched SWA and the searched R34 soup (1791); per-seat rates from the JSON.
* Own-search gain: the searched SWA against its raw self.
* Critic: P33b's calibration tables and P33c's leak-pair ranking, re-run on each arm's SWA.
* Stability: entropy, KL to π_ref, pool win rate, value loss — the usual.

## Decision rules, collected

* P33a: leaf swap > rollout swap by 3 SE → the critic track is the route; the reverse → R51 first; neither → R52.
* P33b: greedy-game critic error larger than sampled-game error by > 2 SE, concentrated at DEFCON
  2 / late turns → R50 runs; otherwise R49 only.
* P33c: critic at chance on the DEFCON-2 leak pairs → the leak is the critic's; ranks them right →
  the horizon's (R52).
* P33d: departures at rank 5–8 > 25% → R51 and k = 6; < 10% → channel 1 closed.
* R49/R50: adopt per the rule above; null with a positive P33c → A14.
* P33e: > 3 SE for another setup under searched play → re-run R43 after the critic track; level → closed.
* R53: critic error on exact-labelled turn-10 positions halves and the final-scoring loss share
  falls under the root → adopt; exact labels learned but no game effect → the endgame is not where
  the searched player's points are, closed.
* R54: own-search gain smaller than the control's **and** searched rating higher → adopt and open
  the from-scratch cell; KL(target ‖ policy) past 0.5 inside a refresh period → stop (the R29 loop).
* P33f: trunk beats the public-information probe → no belief input proposed; trunk at the prior →
  the gap is reported to the owner as the case for a history/belief representation.

## Follow-ups

* A positive critic track re-opens the teacher loop with a repaired critic: re-freeze to the arm's SWA only
  if it *searched* beats the current teacher searched (P32's corrected rule).
* A positive R50 makes `v_greedy` the leaf for every searched product player.
* If P33a says the soup's lead is the critic's noise averaging, a cheap product option is a leaf
  ensemble (the model's critic averaged with the soup's); it is not a base-checkpoint gain and is
  recorded as a product note only.

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))

* **P33b, P33c, P33d, P33f done (2026-10-15)** -- [`../log/P33_diagnostics.md`](../log/P33_diagnostics.md):
  the critic is calibrated on greedy play (R50/R55 dropped, R49 stays), ranks ordinary move pairs well
  but is at chance where the searched player errs (the leak is the critic's), candidate recall is open
  at 28% of departures (k = 6 and R51 qualify), and the trunk holds exactly the public card
  information about the opponent's hand. P33a running.
* **R49 built and skipped** (`--search-value-beta` with `--search-rollout-horizon`, no CE): its own
  target -- the 4-round greedy continuation -- ranks the leak pairs at chance at 16 and at 64 worlds, so
  it cannot teach what the critic misses ([`../log/P33_diagnostics.md`](../log/P33_diagnostics.md)).
* **P33a done**: A13+R48's critic costs it under search -- its playouts with the soup's critic are
  level with the searched soup, the soup's with its critic lose in both seats, the two swaps head to
  head 58.2%. R39's critic is slightly behind the soup's (−1.4 / −2.9 in the soup's root; swaps 50.3%). The critic is the
  channel; arms are read by their critic under the root.
* **The rollout root with a longer stop, in play (eval only):** `stop=later` (the later of 4
  boundaries and the turn's end) against today's root on the R34 soup, 16 worlds: 47.8%, −1.8 / −2.7
  against the seat bars; `stop=turn` (the turn's end, whether sooner or later than 4 boundaries):
  **51.3%, +1.2 / +1.4** (replicate on other deals 51.1%; pooled 51.2%, +0.5 / +2.0); `stop=earlier`
  (the turn's end when it comes within 4 boundaries): **51.9%, +2.5 / +1.3**, replicate 52.0%; pooled
  **51.95%, ~2.5 SE, ~+14 Elo** -- the best of the four, a play-time candidate (designation is the owner's). A turn-end leaf holds up in
  play without training; R52 trains the critic toward it with the noise averaged out.
* **R52 launched 2026-10-16 as recipe code R49** (`run_codes` gives the next free code; this plan's R49
  approach was never launched): `…+A13@12010M+R49`, `--search-value-beta 0.5 --search-rollout-horizon 4
  --search-rollout-stop turn --search-rollout-leaf-side own --search-subsample 0.0078125`, from the
  12,010M state (the 12,000M one was overwritten by the control's first save), 400M; control: A13+R48
  continued unchanged (leg 3, 12,000 → 12,400M). Approved by the owner. **Null as built** (2026-10-16):
  raw level, searched level to slightly positive, the critic's leak-pair ranking unchanged -- at 1 in
  128 decisions the target is under 1% of the value signal; a denser cell (1 in 16) is the next step.
