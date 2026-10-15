# Instant-win and instant-loss decisions

Decisions that end the game at once. They need no answer key: the engine either reaches
`Phase::GAME_OVER` with the acting player losing, or it does not. Each entry was verified by
building the position with `ai/eval/positions.py` and stepping the engine to a terminal state.
`ai/eval/safety.classify_legal_actions` now labels these mechanically for any position (a forced
loss taken, a forced win not taken), and the blunder census and bank use it
(`tools/README.md` §7d); the encoded behavioural claims are in `ai/eval/claims.py`.

Two rules generate most of the catalogue:

* **The DEFCON-1 loser is always the phasing player** (`ts/defcon.hpp`), whoever's event fired.
* **An opponent-associated card can never be played as an event** (`action_mask.cpp` offers EVENT
  only for your own or neutral cards). Playing it for Ops fires the opponent's event
  unavoidably, so the real choice is Ops, Space, or not selecting the card.

## 1. Unconditional loss: a DEFCON-degrading event at DEFCON 2

Precondition: DEFCON 2, card in hand, and for the Ops variants the acting player is phasing.
Space is the safe disposal for an opponent's card.

| card | loses when |
|:---|:---|
| #4 Duck and Cover (US) | USSR plays it for Ops, or US plays it for the event |
| #50 We Will Bury You (USSR) | US plays it for Ops, or USSR plays it for the event |
| #89 Soviets Shoot Down KAL-007 (US) | USSR plays it for Ops, or US plays it for the event |
| #46 How I Learned to Stop Worrying (neutral) | either side sets DEFCON 1 with the event |

## 2. Conditional loss: the opponent is handed a free battleground coup

Weaker than category 1: a losing line exists, but the opponent must take it. HeuristicBot does
not, so playing against it never punished these.

| card | acting player | precondition |
|:---|:---|:---|
| #91 Ortega Elected in Nicaragua | US, for Ops | US influence in Cuba (reproduced greedily, not only in search) |
| #62 Lone Gunman | US, for Ops | US influence in a coup-eligible battleground |
| #67 Grain Sales to Soviets | USSR, for Ops | USSR influence in a coup-eligible battleground |
| #96 Tear Down This Wall | USSR, for Ops | USSR influence in a European battleground |
| #20 Olympic Games | either, for the event | the opponent chooses to boycott (a greedy participate never reaches the loss) |

## 3. Couping a battleground at DEFCON 2

Any battleground coup degrades DEFCON (`ops.cpp`), and at DEFCON 2 coups are still allowed in
Africa and Central/South America. The observation already carries the answer: board slot 7 is
"couping here takes DEFCON to 1" (`observation.cpp`).

## 4. Wargames (#100)

Offered only at DEFCON 2; it gives the **opponent** 6 VP and ends the game. A lead of 7+ wins
(trigger it), exactly 6 draws, 5 or less loses (do not).

## 5. Held scoring card

Holding a scoring card at the end of a turn loses (`Engine::is_held_scoring_loss`), unless
playing it would itself cross the player's own −20, in which case both choices lose and a test
must not assert either.

## 6. VP threshold crossings

Any award reaching +20 (US) or −20 (USSR) ends the game at that award. Not enumerated per card;
it needs each VP-awarding event's amount and bounds.

## Missed forced wins

1. A playable scoring card that reaches your 20 VP: play it.
2. Wargames at DEFCON 2 with a 7+ VP lead: trigger it.
3. An opponent's event or Olympic Games grants you a coup at DEFCON 2 and they hold influence in
   an eligible battleground: take the coup.
4. Missile Envy (#49) at DEFCON 2 handing over a category-1 opponent event: fires under the
   opponent's round. **Not verified** (who is phasing at resolution).

## Corrections and open items

* CIA Created (#26) is a US card; the original claim listing it as a coup-granting opponent event
  was wrong. The reverse framing (USSR plays it for Ops) is unverified.
* Not verified: the Missile Envy hand-over, the full VP-award enumeration, and whether every
  coup-granting event reproduces for both sides.
