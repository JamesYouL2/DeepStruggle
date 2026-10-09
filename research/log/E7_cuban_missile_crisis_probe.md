# E7 — does the heads soup understand Cuban Missile Crisis? It does not

**Question (owner, 2026-10-09):** "does model understands Cuban Missile Crisis. Make model headline
it when it gets it, and check if 1) models cancel it or not; 2) when not cancelled (either by choice
or because can't), model still coup. Especially check cases if US plays Cuban Missile Crisis on USSR,
and then US plays Lone Gunman or Che, and if USSR plays Cuban Missile Crisis on US, US doesn't cancel
it and later USSR plays CIA created."

**Probe:** `data/reports/cmc_probe.py`, output `data/reports/cmc_probe.out`.
**Model:** the heads soup `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M`, greedy self-play, 2,000
games a run (seeds from 77,000). Whoever holds the crisis (card 40) at a headline is made to
headline it. A second run also forces the follow-ups: under its own crisis the US plays Lone Gunman
(62) or Che (107) when it holds one, and the USSR plays CIA Created (26).

**The rules as the engine has them.** The card is neutral: its player is protected for the rest of
the turn, and a coup by the other side (the restricted side) loses the game -- unless the restricted
side can pay (USSR: 2 Influence in Cuba; US: 2 in West Germany or Turkey), in which case the engine
pays inside the coup and the coup stands. Paying voluntarily is offered at the head of the
restricted side's action rounds. The crisis **is** in the observation: persistent-effect bits 11
(`CMC_ACTIVE_US`) and 12 (`CMC_ACTIVE_USSR`) are global features 23 and 24, by seat rather than by
perspective. So a suicidal coup is a learning failure, not missing information.

**What "paid" and "could not pay" mean here.** A coup "that paid" is a coup by the restricted side
while it held the 2 Influence: the engine removed them (Cuba, or West Germany / Turkey) inside the
coup, the crisis ended, and the coup resolved normally -- legal, at a cost of 2 Influence. A coup
that "could not pay" was made without them; there is then no way to pay at all, so the only safe
play is not to coup. No pay prompt is shown during the opponent's round (after Lone Gunman or Che,
for instance) -- the engine offers one only at the head of the payer's own action round -- but none
is needed: a coup pays by itself whenever paying is possible.

## 1. Cancelling: almost never, which is right

Offers at the head of an action round: the US declined 3,193 and paid 43 (1.3%), the USSR declined
2,276 and paid 33 (1.4%). Paying ahead of time buys nothing unless a coup is planned, and a coup
pays automatically when it can; declining is what most human players do too.

## 2. Couping under the opponent's crisis: often, including when it cannot pay

Natural play (only the headline forced), 1,745 games with a headlined crisis:

| restricted side | coups that paid (crisis cancelled) | coups that could not pay: **game lost** | lost games / crises its opponent headlined |
|:---|---:|---:|---:|
| USSR (US headlined) | 33 | **54** | 54 / 837 = **6.5%** |
| US (USSR headlined) | 93 | **28** | 28 / 928 = **3.0%** |

Every one of the 82 ended in a win for the protected side.

## 3. The named cases (restricted side's response when the event fires)

| case | run | coups that could not pay (**lost**) | coups that paid | no coup |
|:---|:---|---:|---:|---:|
| US's crisis, US plays **Lone Gunman** → USSR 1-Op play | natural (52) | 6 | 4 | 28 (26 influence, 2 realign) |
| | forced (175) | **79** | 74 | 18 (influence) |
| US's crisis, US plays **Che** → USSR optional coup | natural (80) | 16 | 6 | 1 decline |
| | forced (101) | **29** | 19 | 2 declines |
| USSR's crisis, US did not cancel, USSR plays **CIA Created** → US 1-Op play | natural (11) | 1 | 6 | 3 |
| | forced (47) | **9** | 36 | 2 (influence) |

(The rest -- Che 53 / 47, Lone Gunman 10 / 4, CIA 1 / 0 -- had no decision to make, mostly Che with
no legal target.)

* **Che:** when the USSR cannot pay, it took the coup every time it had one -- 16 of 16 natural,
  29 of 29 forced -- and declined only twice in 181 Che plays, both times when it could pay.
* **Lone Gunman:** when the USSR cannot pay, forced 79 coups against 12 influence plays (87%
  suicidal); natural 6 against 17 (26%).
* **CIA Created:** the US usually can pay (2 in West Germany), and coups; when it cannot it still
  coups -- 9 of 9 forced, 1 of 2 natural.

**Reading.** The network has not learnt that a coup under the opponent's crisis is a loss when it
cannot pay. It treats a free coup as a free coup. The lost games are a few percent of the games in
which the crisis is played, and the US exploits nothing deliberately: it plays Che and Lone Gunman
under its crisis about as often as it holds them.

## Does "always headline it" pay? No: −1 to −2 points a seat (owner, 2026-10-09)

"Check if simple rule 'always headline cuban missile crisis if you got it' improves winrate (for each
seat separately), on raw model and on soup." The rule player is `headline:40:<checkpoint>` (new: a
scripted headline rule over any agent, `tools/lib/batch_tournament.apply_headline_rule`), playing
the plain network; each of its games is paired with the network's own self-play on the same deal
(`--log-games`, fixed deal seeds), so the two are the same game until the rule changes a headline.
Greedy, 20,000 games a seat; reports `data/reports/cmc_rule_{soup,raw6800}_*.jsonl`.

| network | rule as US vs self-play | rule as USSR vs self-play | games the rule changed |
|:---|---:|---:|---:|
| heads soup | 46.82 vs 47.85: **−1.03 ± 0.28** | 50.56 vs 52.15: **−1.59 ± 0.28** | ~51% |
| raw `E7-A8-R1-S44@6800M` | 46.17 vs 47.15: **−0.98 ± 0.28** | 50.86 vs 52.85: **−1.99 ± 0.29** | ~51% |

The network's own choice -- usually not to headline it -- is better in both seats, on both
networks, though the opponent's crisis suicides (section 2) are a gift whenever it is headlined.
Headlining it spends the headline on an event that drops DEFCON to 2 for both sides for the turn,
and the opponent's suicidal coups come in only a few percent of those games.

## Only when the opponent cannot pay: level (owner, 2026-10-09)

"Lets check rule 'headline Crisis only when opponent can't pay'." `headline:40+oppcantpay:<checkpoint>`
headlines the crisis only when, at that headline, the opponent could not pay it off (USSR: fewer
than 2 in Cuba; US: fewer than 2 in both West Germany and Turkey); otherwise the network chooses.
Same pairing, deals and size as above (`data/reports/cmc_rule_*_rule40_oppcantpay.jsonl`).

| network | rule as US vs self-play | rule as USSR vs self-play | games changed (US / USSR seat) |
|:---|---:|---:|---:|
| heads soup | 47.67 vs 47.85: −0.18 ± 0.22 | 52.02 vs 52.15: −0.12 ± 0.13 | 32% / 11% |
| raw `E7-A8-R1-S44@6800M` | 47.07 vs 47.15: −0.08 ± 0.21 | 52.78 vs 52.85: −0.07 ± 0.12 | 28% / 9% |

Level on both networks, in both seats: the condition removes the cost of headlining the crisis
blind, but adds nothing. The opponent's suicidal coups are too rare to pay for the headline, and it
can still put Influence back into Cuba (or West Germany / Turkey) before it coups.

## Does being under the crisis change the decision to coup? (owner, 2026-10-09)

"What exactly is P(USSR coups | CMC on USSR and US plays Lone Gunman or Ortega and USSR can't pay)?
What is P(US can't pay)? Overall, is model's decision to coup or not coup affected by it being under
CMC and not able to pay?" `data/reports/cmc_probe2.py` (output `cmc_probe2.out`): 4,000 greedy games,
the crisis headlined whenever held, the follow-ups forced (US: Lone Gunman, Ortega, Che; USSR: CIA
Created). At every decision where a side could coup it records the coup probability on the real
position and on a clone with **only the crisis flag cleared** -- DEFCON 2 and everything else kept,
so the difference is what the network reads from the flag.

**Can the opponent pay when the crisis is headlined?** USSR, when the US headlines it: cannot in
1,007 of 1,671 (60.3%). US, when the USSR headlines it: cannot in 386 of 1,902 (20.3%) -- the US
usually holds 2 in West Germany; at the US's coup decisions under the crisis it cannot in ~16-19%.

| side | coup option | under the opponent's crisis | n | chose coup | P(coup) | P(coup), flag cleared |
|:---|:---|:---|---:|---:|---:|---:|
| USSR | Lone Gunman, the US played it | **cannot pay** | 164 | **90.9%** | 0.915 | 0.990 |
| USSR | Lone Gunman, the US played it | can pay | 152 | 91.4% | 0.908 | 0.972 |
| USSR | Che, the US played it | **cannot pay** | 71 | **98.6%** | 0.983 | 1.000 |
| US | CIA Created, the USSR played it | **cannot pay** | 17 | 94.1% | 0.925 | 0.999 |
| USSR | own card, play-mode choice | cannot pay | 4,044 | 0.4% | 0.004 | 0.202 |
| USSR | own card, Ops-mode choice | cannot pay | 722 | 3.0% | 0.030 | 0.471 |
| US | own card, play-mode choice | cannot pay | 1,703 | 1.7% | 0.017 | 0.198 |
| US | own card, Ops-mode choice | cannot pay | 306 | 8.2% | 0.080 | 0.327 |

(Ortega: no case in 4,000 games -- a late-war card and a mid-war crisis in the US hand in the same
turn never came up. Without a crisis the same options: Lone Gunman 96.7%, Che 94.6%, CIA 82.3%.)

* **On its own cards the network reads the flag and stops couping:** the coup probability falls from
  0.20-0.47 with the flag cleared to 0.004-0.08 with it set.
* **On a coup handed to it by the opponent's card it barely reacts:** Lone Gunman 0.99 → 0.92, Che
  1.00 → 0.98, CIA 1.00 → 0.93. These are rare positions, and there the policy plays the event's
  coup almost as it would without a crisis.
* **It does not distinguish "can pay" from "cannot pay" anywhere**: the rates are the same either
  way (own card, Ops mode: 3.0% / 3.0% USSR; Lone Gunman: 91.4% / 90.9%). It has learnt "the
  crisis means no coup" for its own cards, not "no coup unless I can pay".

## The US combo: +3.5 points as US on the soup, +2.9 on the raw network (owner, 2026-10-09)

The owner's rule, `script:cmc-combo:<checkpoint>` (`tools/lib/batch_tournament.SCRIPTS`): the US,
holding Cuban Missile Crisis and Lone Gunman, Che or Ortega, with **no** USSR Influence in Cuba,
headlines the crisis and then plays the other card (Lone Gunman, then Ortega, then Che). Ortega and
Che are played for Influence -- Ops first, then the event -- placed where the event's coup can reach
it: Ortega into Cuba, else Costa Rica; Che into a non-battleground of the Americas or Africa, least
stable first. Lone Gunman's mode is the network's. Everything else is the network's own play.

**In the games** (`data/reports/cmc_combo_probe.py`, 4,000 greedy games, the US with the script):

| | heads soup | raw `@6800M` |
|:---|---:|---:|
| combo fired | 415 (10.4%) | 364 (9.1%) |
| Lone Gunman: USSR coups, cannot pay → **loses** | **175 of 201** | **158 of 187** |
| Che: USSR coups, cannot pay → **loses** | **115 of 129** | **107 of 114** |
| Ortega: USSR coups, cannot pay → **loses** | **25 of 29** | **15 of 15** |
| no follow-up reached | 56 | 48 |
| US wins of the combo games | **356 / 415 (85.8%)** | **313 / 364 (86.0%)** |

**Win rate,** paired with self-play on the same deals, 20,000 a seat:

| network | rule as US vs self-play | as USSR (the script never acts: the method's control) |
|:---|---:|---:|
| heads soup | 51.33 vs 47.85: **+3.48 ± 0.15** | −0.01 ± 0.01 |
| raw `E7-A8-R1-S44@6800M` | 50.01 vs 47.15: **+2.85 ± 0.13** | +0.00 ± 0.00 |

A scripted exploit of the USSR's blind spot is worth about 3 points of the US's win rate overall,
from one game in ten. When the USSR has nothing in Cuba, the event's coup is taken 87-100% of the
time and the game is lost on the spot. (The earlier forced run headlined the crisis whatever Cuba
held, and the USSR could pay in about 40% of its coups; with nothing in Cuba at the headline, it
rarely gets to 2 before the follow-up.)

## Diagnostics for an exploiter and for search (owner, 2026-10-09)

`data/reports/cmc_combo_probe2.py` (output `cmc_diagnostics.out`), the USSR always the frozen
`E7-A8-R1-S44@6400M` (the US exploiter's target), greedy unless stated. A combo position is a US
headline with the crisis and Lone Gunman, Che or Ortega in hand and no USSR Influence in Cuba.

| US | games | combo positions | crisis headlined there | follow-up played under it | USSR suicides |
|:---|---:|---:|---:|---:|---:|
| the main itself | 4,000 | 349 (8.7%) | 6 (1.7%) | 0 | 0 |
| US exploiter R25 @6520M | 4,000 | 323 (8.1%) | 10 (3.1%) | 1 | 0 |
| R25 SWA 6480-6560M | 4,000 | 352 (8.8%) | 11 (3.1%) | 2 | 0 |
| the main + the scripted combo, against **Gumbel k8@256 USSR** | 2,000 | 176 | 176 | 162 | **103 (64%)** |

* **Neither the main nor the US exploiter plays the combo.** R25's +9 to +12 against this target
  came from somewhere else; it found nothing of the crisis line in 200M.
* **Search defends only partly**: against Gumbel the USSR still coups without the means to pay in
  64% of the follow-ups (Lone Gunman 51 of 93, Che 40 of 52, Ortega 12 of 14), against ~87% greedy.

**Why search does not defend** (`data/reports/cmc_search_debug.py`). On 30 USSR Lone Gunman
decisions under the US's crisis with nothing in Cuba, Gumbel coups in 21, batched or one at a time
alike. A hand-played coup there ends the game at once (20 VP, `CMC_SUICIDE_LOSS`), and the search
sees it -- but the network's preference is extreme. In one of the 21:

| | influence | coup | realign |
|:---|---:|---:|---:|
| logit | 5.91 | **34.25** | −2.59 |
| searched value, for the USSR | −0.78 | **−0.95** | −0.81 |
| Gumbel's sigma(completed Q) | 9.2 → 15.6 | 0 | 7.7 → 13.2 |

sigma is (c_visit + max visits) * c_scale * Q rescaled to [0, 1] -- c_visit 50, c_scale 0.1, as in
mctx -- so at 256 evaluations it can add at most ~15 to a logit, and the coup leads by 28. PUCT's
visits follow the same prior (79 of 128 on the coup) and it plays the most visited. The position
was losing anyway, so the searched difference is small (−0.95 against −0.78); where the prior is
less extreme the search does decline (the first position tried: influence).

**For best-response search** -- the opponent modelled as greedy -- the same mechanism bites from the
other side: the main headlines the crisis in 1.7% of combo positions, so its headline logit is low,
and a candidate the prior has written off cannot be rescued by sigma at this budget. A test of the
idea therefore needs the root's prior softened (a temperature on the root logits) or sigma scaled up,
and that changes every searched decision, so it has to be checked for strength as well.

## Can best-response search find the combo? Only its last step (owner, 2026-10-09)

`BatchedMCTSConfig.opponent="greedy"` (new, Python tree): at every node where the side not
searching moves, the tree takes that side's most probable move, so the US plans against the greedy
network. `gumbel_prior_temperature` (new, default 1) divides the Gumbel root's logits before
candidates are taken and ranked. `data/reports/br_search_probe.py` (output `br_search_probe.out`):
positions from 1,000 greedy games of `E7-A8-R1-S44@6400M` -- 109 combo headlines (crisis + Lone
Gunman / Che / Ortega, no USSR Influence in Cuba) and 105 US card choices under its own crisis
holding one of the three -- and what each searcher plays there.

At the combo headlines the crisis is the 2nd-7th most probable legal move (inside Gumbel's 8
candidates) but a median 17 nats behind the top move (max 39).

| searcher (Gumbel k8 @256) | root T | headlines the crisis | plays the follow-up |
|:---|---:|---:|---:|
| the network, greedy | -- | 0 / 109 | 6 / 105 |
| opponent searched | 1 | 1 | 10 |
| opponent greedy | 1 | 2 | 11 |
| opponent searched | 3 | 1 | 26 |
| opponent greedy | 3 | 5 | 33 |
| opponent searched | 10 | 5 | 40 |
| opponent greedy | 10 | 8 | **44** |

* **The follow-up -- one decision from the win -- is found once the prior is softened**: 44 of 105
  at T = 10 against a greedy opponent. Even with the opponent searched it is found 40 times, since
  the USSR's in-tree choice is driven by the same 28-nat prior towards the coup.
* **The headline -- the plan's first step -- is not**: 8 of 109 at best. At 256 evaluations a
  candidate gets ~10 in the first halving phase, and the win lies past the USSR's whole action round
  (one tree node per USSR decision, greedy or not) and the US's card choice. Depth, not only prior.
* So a search-only exploiter at this budget would seldom set this trap. Finding plans of this kind
  needs the opponent's moves collapsed out of the tree (played greedily without spending a
  simulation each), or a much larger budget, as well as the softened prior.

## Sanity check: can training teach the USSR to avoid it? Not with plain RL (owner, 2026-10-09)

`E7-A8-R1-S44@6400M+R28`: the heads run's own flags from 6,400M plus `--seed-scenarios cmc_combo
--seed-frac 1.0` (new): wherever it applies, the US's scripted combo is forced as environment, with
no policy gradient on the forced moves, so the only learner decision in the trap is the USSR's
answer. Run directly, the seeder forces the crisis headline in 31 of 303 greedy games (~10%), so
by 200M the learner had met the trap in tens of thousands of games.

The scripted combo US (the 6,400M main + script) against each snapshot, 2,000 games:

| step | R28: USSR suicides / follow-ups | plain leg |
|:---|---:|---:|
| 6,450M | 163 / 179 (91%) | 172 / 194 (89%) |
| 6,500M | 138 / 163 (85%) | 141 / 170 (83%) |
| 6,550M | 141 / 155 (91%) | 167 / 193 (87%) |
| 6,600M | 153 / 177 (86%) | 137 / 157 (87%) |

**No learning.** The reason is in the logits: on 60 USSR Lone Gunman decisions under the US's crisis
with nothing in Cuba, the coup leads the best alternative by a median 19.4 nats at the start, 23.3 /
19.4 on the plain leg at 6,450 / 6,600M, and **32.9 / 27.4 on R28** -- P(coup) is 1 to float precision
in most of them (1 − P underflows). A policy-gradient step on a sampled coup is scaled by 1 − P, so
an immediate loss produces no gradient, and influence is never sampled, so there is no positive
example either; the lead even grew, pushed by the rest of the game. The network can represent "do
not coup under the crisis" -- it does on its own cards (section above) -- but plain PPO cannot move a
softmax this saturated. That is also why the main never learnt it.

What would let it learn: a cap on the policy logits (c·tanh(z/c)), so no gap saturates and gradients
survive; or a cross-entropy target towards a better policy (gradient target − π does not vanish at
saturation), e.g. a Gumbel improved-policy target softmax(logits/T + sigma(Q)) with a softened root.
An exploration floor alone does not: it reweights by pi_old / mu, which is ~0 for such an action.
