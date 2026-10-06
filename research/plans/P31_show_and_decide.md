# P31 — Show, then let training decide: the rare-spot card leaks

**Status:** proposed (2026-10-06).
**Root:** E7-20-44@2,800M (`E7-20-44_20261004_173440/resume_state.pt`), with its plain control
leg (2,800 → 3,600M, same name, second directory) as the no-change branch. The root is chosen
for its control, not its strength: the 3,600M end state is a little stronger but has no sibling.
**Gate:** step 0 (the banks on the root) before any arm is *read*; nothing gates building.
**Needs approval:** none of it touches `engine/` or the observation. Steps 1a–1c are trainer
changes; 1d adds a head (model only) and may need a small *read-only* binding (the non-phasing
player's mask) — if it does, it is proposed before being written. The forced-event seeding (1b)
constructs training situations; it changes no rule.

## The problem this addresses

The expert review ([`../log/expert_review_E7.md`](../log/expert_review_E7.md)) found that the
model's board play beats strong humans and its weakness is **card play at specific spots**:
Wargames declined at a winning lead (429 of 430; ~1.5 points a game), One Small Step one box
behind (0% of 1,489), Special Relationship with NATO (0% of 320), KAL-007 with South Korea
US-controlled (0% of 305), Glasnost with The Reformer (1% of 320), game-winning OPEC and Alliance
for Progress declined about half the time, and the forced-win take rate at 0.45
([`../log/decisive_win_misses_e7.md`](../log/decisive_win_misses_e7.md)).

The per-checkpoint reports on the `event-investigation` branch
(`research/log/per_checkpoint/E7-{02,20,21}-44_*.md`) show what 1,600M more steps and a league
did to these: the **frequent** decisions fixed themselves — Star Wars played for its event while
ahead in space went from 2% to 71–72% of plays, Five Year Plan moved to the last round with a
smaller hand, Aldrich Ames is played event-first — while **every rare-spot leak stayed at its
floor**: Wargames 1% → 0% → 0%, One Small Step 1% → 1% → 0%, Special Relationship 2% → 4% → 6%,
KAL-007 38% → 43% → 40%, Glasnost 13% → 17% → 12%, OPEC and Alliance for Progress flat. The
league's control leg matches the league, so the plain training did all of it.

The mechanism is the one the review named: a move the policy gives p ≈ 0.002 is sampled about
once per 500 games the spot occurs, PPO's ratio clip bounds what one lucky sample can move, and
the early verdict ("this event is bad") was formed in the random contexts where it *is* bad. Two
of the owner's cases are worse than that:

* **Nuclear Subs** is a two-step chain: the payoff of eventing it lives in a later decision (a
  battleground coup at DEFCON 2) that the policy has separately learned never to make. Neither
  half ever gets signal.
* **Chernobyl** is a three-step chain whose value is a counterfactual about the *opponent's*
  options: event it, then place into the forbidden region for the rest of the turn, then collect
  control that sticks. The critic's prior is that a placement into a USSR-held region is wasted
  because a response follows; under Chernobyl that prior is wrong, and the critic has never seen
  a Chernobyl-active state to learn the exception. The forbidden region *is* in the observation
  (`observation.cpp:378`, a 6-way one-hot in the global block), so this is data and credit, not
  representation.

Generic exploration has been tried and does not reach these: temperature 2 at play-mode
decisions was slightly worse ([`../log/P30_branch_arms_c2_play_mode_temp.md`](../log/P30_branch_arms_c2_play_mode_temp.md)),
a VP potential was level ([`../log/P30_win_only_and_vp_potential.md`](../log/P30_win_only_and_vp_potential.md)),
and uniform search targets did not pass (E7-18-44, [`../runs.md`](../runs.md)).

## The approach

Four mechanisms, each a standard answer to "show the model something it would not try, and let
training decide whether it is needed", each generic — no card list enters the trainer:

| | mechanism | literature | what it is for |
|:---|:---|:---|:---|
| 1a | **action floor** in the behaviour policy at play-mode decisions | ε-soft policies; off-policy correction | big-value rare moves (Wargames at a lead, One Small Step 3 vs 4, a coup under Subs): a few hundred outcome samples carry the signal |
| 1b | **scenario seeding**, the precursor forced as environment | KataGo hint positions; Go-Explore; reverse curriculum (Florensa 2017) | skill chains (Subs, Chernobyl): learn the last link where the state exists, then the precursor |
| 1c | **counterfactual mode credit** from paired playouts | CFR counterfactual values; all-actions policy gradient (Mean Actor-Critic); expert iteration | small-value spots (Special Relationship +2.1, Glasnost +2.3): variance reduction is what makes them detectable at all |
| 1d | **opponent-legality auxiliary target** | auxiliary targets (KataGo); P30 C2 | the trunk learns what a restriction effect denies the opponent — Chernobyl, DEFCON's region locks, Quagmire, Red Scare — rather than one card |

Two rules carried from the record. **The forced step is never trained as the policy's own
action**: E7-11-44 showed that a scripted action trained with the policy's log-prob and a biased
credit gets adopted whatever it is worth (USSR Romania 6 at 99.8%). And **a playout verdict
values a move given how this model plays afterwards**, so counterfactual credit is applied where
the follow-up does not depend on a skill the model lacks — at the exploitation step of a chain,
not at its precursor.

## Why from the saturated checkpoint, and why from scratch afterwards

A branch from a shared late state is a far tighter A/B than a from-scratch pair (seeds differ by
55–100 Elo at saturation; a no-change branch of the same state lands at 50.3% against its
sibling), the playout and critic yardsticks are only trustworthy late (the critic's ECE is 0.006
on the soup), and the rest of play does not move, so a bank delta is attributable. So every
mechanism is **screened as a branch from the root**. Adoption is **one from-scratch run**,
because representation changes (1d) shape the trunk in its plastic phase and skill interactions
(a model that knows Wargames early plays the DEFCON track differently) only compound from
scratch — and the standard for base-model quality is quality at saturation
([P30](P30_base_model_quality.md)).

**Measured on the banks, not on Elo.** Each leak is worth 1–2 points a game and both seats share
it, so self-play Elo cannot see it. Panel Elo and the head-to-head against the control are
reported as a floor (the arm must not get worse), never as the criterion.

## Step 0 — spot-conditioned baselines on the root (CPU, ~1 day)

1. **Port the review's bank tools** from the fork (`JamesYouL2/DeepStruggle`,
   `exp/hungary-openings`, `ai/eval/`): the position banks, the choice oracle, the VP ledger. PR
   #6 brought the forced-win classifier; the banks are not in the tree yet.
2. **Run on the root and its control leg** (2,800M; 3,200M and 3,600M of the control): the seven
   spot banks (Wargames at a winning lead; One Small Step 1 vs 2 and 3 vs 4; KAL-007 with South
   Korea US-controlled; Glasnost with The Reformer; Special Relationship with NATO; game-winning
   OPEC and Alliance for Progress), `decisive_cost.py` (take rate, classifier v2), the events-VP
   ledger against the corpus, and `checkpoint_report.py`. The control leg's drift from 2,800M to
   3,600M is the noise floor every arm is read against.
3. **Two new banks by the same paired-playout method**, built once 1b can produce the states:
   **Nuclear Subs** — coup-under-Subs rate and cost, and the Subs event rate; **Chernobyl** —
   placement-into-the-region rate and the gain that sticks while the effect is active, and the
   event rate. Each with 64 pairs per position, ≥300 positions, the model on both sides.

## Step 1 — the builds (trainer and model only)

| | flag | change | effort |
|:---|:---|:---|:---|
| **1a** | `--play-mode-floor ε`, `--floor-scope {all,seeded}` | the *behaviour* policy at `SELECT_PLAY_MODE` and the choices inside events is (1−ε)·π + ε·uniform over the legal options; the mixture log-prob is stored so PPO's ratio corrects it; the target policy is untouched. Default ε 0.03; an `--floor-anneal` to 0 over N steps for the from-scratch run | hours: the `--play-mode-temp` plumbing is most of it |
| **1b** | `--seed-scenarios subs,chernobyl --seed-frac f` | in a fraction f of games, when the US first holds the named card in an action round where its event may trigger, the event play is forced **as environment**: no transition is stored for it, the policy's log-prob is not evaluated on it, and the opponent sees an ordinary event. Chernobyl's region is drawn uniformly from the six. Everything after is the policy's own. A scenario list is data, not knowledge: the trainer treats every entry the same way | ~1 day: the scripted-opening machinery minus the credit, plus a card-play hook |
| **1c** | `--mode-cf-coef c --mode-cf-subsample k` | at 1 in k play-mode decisions (and the choices inside events), each legal option is played out to the end with common dice and redeals, the model on both sides, on the C++ playout path; the per-option values, centred on the policy's own option, enter the policy loss as an all-options term, weighted by c. Nothing is acted on; rollouts stay on the raw policy | ~3 days; built last |
| **1d** | `--aux-opp-legality w` | a per-country head predicting whether the opponent may add influence there, and may coup there, at its next decision; the label is the engine's mask for the non-phasing player at the stored state. If the bindings cannot produce that mask without an engine change, it is proposed first | ~1 day plus the binding check |

Every flag off by default; `launch_flags.py --diff` against the root's own continuation must
show the flag and `--train-steps` and nothing else.

## Step 2 — branch arms from the root, +400M each

Two at a time; ~1.2 h per arm at ~90k steps/s. Each is read against the control leg's matched
window (3,120–3,200M) and against step 0's drift.

| arm | change | reads, in order |
|:---|:---|:---|
| **B1** | floor, ε 0.03, all play-mode decisions | Wargames-at-lead take rate and cost; One Small Step 3 vs 4; KAL-007 + South Korea; `decisive_cost` take rate; the events-VP ledger |
| **B2** | seeding f 0.05 (Subs, Chernobyl), no floor | **reverse-curriculum order:** coup-under-Subs rate → Subs event rate; placement-into-region rate and sticking gain under Chernobyl → Chernobyl event rate. The exploitation step moves first if the mechanism works; the precursor moves only after it |
| **B3** | B2 + floor scoped to seeded games | the same banks; whether the floor shortens the chain |
| **B4** | counterfactual mode credit, c 0.5, k 16 | Special Relationship + NATO; Glasnost + The Reformer; game-winning OPEC / Alliance for Progress — the small-value spots B1 cannot resolve; then the big-value spots, which it should also move |
| **B5** | opponent-legality head, w 0.1 | the head reaches the trunk (frozen-trunk R² against the untrained value, as C2's check); then the Chernobyl placement bank. A signal only — the from-scratch run is its test |

Order: B1 ‖ B2 as soon as 1a and 1b exist; B3 ‖ B5 next; B4 when 1c is built. **Every arm also
reports** the panel per seat, the head-to-head against the control, and the self-inflicted
DEFCON-1 rate (a floor at play-mode decisions samples DEFCON-breaking events too; that cost is
the tax, and it must be seen).

**Decision rule per arm.** *Promote* if at least two of its target banks move by more than 3 SE
*and* the panel per-seat rule is not failed. A bank that moves without strength is logged as a
mechanism result — expected for the +2-point spots, since the whole set is worth a few points a
game. *Kill* if the panel rule fails in either seat, or if the DEFCON-1 rate rises by more than
the floor's share of play-mode decisions would explain.

**Washout check on each promoted arm:** continue 100M with the mechanism off and re-bank. A
skill that persists (Wargames at a lead should — the spot recurs in ~11% of games and the move
wins, so it sustains itself once p is high) needs the mechanism during acquisition only; one that
decays stays in the recipe.

## Step 3 — adoption: one from-scratch run

The E7 shallow recipe plus the promoted mechanisms, **switched on from ~200M** — before that the
critic and the playout policy cannot price the spots, and a floor is a pure tax — seed 44 to
2,800M (~9 h), the E7 line as the control at matched steps; the floor annealed to 0 over the
last 400M if the washout check said it could be. Read at saturation by the panel rule, the
head-to-head, and the full bank set; then seed 43. This run decides adoption; a branch cannot
show the interactions or whether a trunk that grows up with the legality head is better.

## Step 4 — close the loop with humans

The banks are self-play-derived. Re-run the review's doctrine census and the corpus agreement by
card on the from-scratch model, and ask the reviewer for games against it. The leaks were found
against a human; that is where their repair is confirmed.

## Measure

Banks first (step 0's set plus Subs and Chernobyl), each with its playout cost; the events-VP
ledger; `decisive_cost`; then the panel per seat, the head-to-head and the SWA reading, per the
P28 measurement rule. Seed variance does not apply to the branches (shared state); it does to
step 3, which is why it runs on both seeds.

## Decision rules, collected

* 0: no decision; the root's numbers and the control leg's drift are what every arm is read against.
* B1–B5: promote on ≥ 2 target banks moving by > 3 SE with the panel rule not failed; kill on a
  failed seat or an unexplained DEFCON-1 rise.
* washout: a persisting skill → mechanism is acquisition-only (anneal); a decaying one → stays on.
* 3: adopt on the owner's rule at saturation on both seeds *and* the banks holding their branch
  gains; a bank gain without the rule passing is recorded as "a repaired leak that does not show
  in self-play", which the human re-review (step 4) then arbitrates.

## Budget

Step 0 a day of CPU; step 1 about a week of engineering in total (1a hours, 1b and 1d a day
each, 1c three days); step 2 ~6 GPU-hours for the five branches plus ~1.5 h for the washout
checks; step 3 ~18 GPU-hours for two seeds. Standing rules: two arms at a time, every arm
registered in [`../runs.md`](../runs.md) before launch, readouts at 40M marks, `launch_flags.py
--diff` before and after launch.

## Follow-ups

* B2 works → the scenario list grows by the same mechanism: any card whose value is a chain
  (Bear Trap / Quagmire exits, Missile Envy, the space-race jumps) without a trainer change.
* B4 works → the counterfactual term is the natural home for the review's candidate 5
  (bank-confirmed rules) without a rule list: the bank *is* the playout.
* B5 works from scratch → the same head form for "what the opponent can score" (live scoring
  cards by region), the other half of what the critic's prior gets wrong.
* Nothing moves → the spots are not reachable by credit at all at this capacity, and the
  question becomes P30's (representation), not exploration.

## Runs

(none yet)
