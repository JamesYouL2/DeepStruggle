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
