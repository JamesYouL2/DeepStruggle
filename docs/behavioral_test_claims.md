# Behavioural test-suite claims

The answer key behind `tools/behavioral_test.py`: strategic claims extracted from a card-by-card
strategy guide (`docs/card_strategies.md`, git-ignored for its licence and regenerated locally by
`tools/scrape_card_strategies.py`) at the level of a player with a handful of games, each
reviewed by a strong human player. Deck-counting, reshuffle timing and multi-turn reads of the
opponent's hand are out of scope; every precondition is checkable from public state plus the
acting player's own hand.

**Encoded so far** (`ai/eval/claims.py`, tiers L = instant loss, S = strategic waste, B =
preference): `a-01`, `a-02`, `a-03`, `a-04`, `a-05`, `a-15`, `b-01`, `c-03`, `n-01`, `n-02`
(as `n-02` and `n-02b`). Everything else below is specified but not implemented. The mechanical
instant-loss cases are catalogued separately in [`instant_decisions.md`](instant_decisions.md).

**Phrasing rule.** An opponent-associated card can never be played *as* an event
(`action_mask.cpp`); playing it for Ops fires the owner's event. So "X should not trigger
Y's card" is always tested as Ops against Space (or holding it). Claims rephrased this way:
`a-01`, `a-08`, `a-09`, `a-11`, `a-15`, `b-12`, `b-19`, `b-20`, `c-02`, `c-07`, `c-19`.

Assertion kinds: **never-argmax** (A, hard invariant: the move is never the policy's favourite),
**prefer** (B: P(one option) > P(another) in one position), **rate** (C: a frequency over
self-play above or below a level). "Review" is the reviewer's verdict; where it adds a condition,
the condition is part of the claim. Removed on review: `a-13` (Iron Lady, US never event),
`a-16` (John Paul II, US always event), `b-03` (US prefers losing West Germany), `b-07`
(Defectors headline).

## A — hard invariants (never-argmax)

| id | precondition → the move to avoid | review |
|:---|:---|:---|
| a-01 | USSR, DEFCON 2, Duck and Cover (#4): never fire its event (DEFCON 1, USSR loses) | include |
| a-02 | either side, DEFCON 2, Olympic Games (#20): never play for the event (a boycott drops DEFCON) | include |
| a-03 | US, NATO (#21) live: never play for the event | hard invariant on turns 1-2 with US VP < 10; rare exceptions later |
| a-04 | US, US/Japan Pact (#27), USSR does not control Japan: never the event | include |
| a-05 | USSR, De Gaulle (#17), USSR already controls France: never the event | include |
| a-06 | USSR, Romanian Abdication (#12): always Ops | include, "when taking Romania will not affect scoring" |
| a-07 | USSR, Comecon (#14), outside the turn-1 Comecon trap: always Ops | early and mid war only; can be useful late |
| a-08 | USSR holds Voice of America (#74): never fire it | include, except when the alternative is DEFCON suicide |
| a-09 | US holds the China Card and Cultural Revolution (#58): never fire CR while the China Card is unplayed | include, with "face-up China Card and USSR lead < 15 VP" |
| a-10 | DEFCON 2, Summit (#45), no commanding regional lead: never the event | include |
| a-11 | USSR, Star Wars (#85): never let it fire (USSR plays it for Ops; the US picks from the discard) | when the discard holds strong US/neutral events; hard at DEFCON 2 with a DEFCON-degrading card in the discard, or with How I Learned (#46) there at any DEFCON. Engine checked 2026-08-30: live only while the US leads the space race, and the US picks |
| a-12 | USSR, Tear Down This Wall (#96): always Space rather than Ops | at DEFCON 2 when the US has influence in a European battleground |
| a-14 | US, AWACS (#110), Muslim Revolution gone, US controls Saudi Arabia: never the event | include |
| a-15 | US, DEFCON 2, US influence in Cuba, Ortega (#91): never let it fire | include |
| a-17 | any event whose own precondition fails (first case: One Small Step (#80) when not behind on space) must not be chosen over Ops | proposed after a review game; not implemented. Generalises `a-14` |

## B — comparative preferences

| id | precondition → preferred over | review |
|:---|:---|:---|
| b-01 | US, Five Year Plan (#5), no discard-forcing plan: Ops over the event | include |
| b-02 | USSR, Blockade (#10): the final AR over a headline or mid-turn play | include |
| b-04 | US at the headline with Containment (#25): headline over Ops that turn | include |
| b-05 | US, Marshall Plan (#23), early war: the event over Ops | include |
| b-06 | USSR, Decolonization (#30), both regions open: most influence to Africa over Southeast Asia | early war |
| b-08 | USSR, Grain Sales (#67) plus another spaceable card: space Grain Sales first | include |
| b-09 | USSR, Cultural Revolution (#58), US holds the China Card: the event over Ops | include |
| b-10 | US, Camp David (#65) and Sadat (#72) in one turn, USSR in Egypt: Camp David first | include |
| b-11 | Brush War (#36): a 2-stability battleground over an easy 1-stability African one | include |
| b-12 | USSR, Alliance for Progress (#78): fire it if the US controls ≤ 2 Central/South American battlegrounds, space it otherwise | include |
| b-13 | USSR, South African Unrest (#53): 1 in South Africa + split to neighbours over both in South Africa | unless 2 breaks US control of South Africa and 1 does not |
| b-14 | turn ≥ 6, own military Ops ≥ opponent's, Ops would not flip a domination: take Arms Race's (#39) VP | include |
| b-15 | USSR, The Reformer (#87): the event over Ops | include |
| b-16 | US, The Reformer (#87): Space over Ops | include |
| b-17 | US fires Tear Down This Wall (#96) on East Germany, France or West Germany: realign over coup | with "US has a positive realignment modifier or no influence in the target" |
| b-18 | USSR, Glasnost (#90): the event once The Reformer has been played, not before | include |
| b-19 | USSR, KAL-007 (#89), DEFCON 3 | tested as P(Ops) < P(Space): the USSR cannot fire a US event directly |
| b-20 | USSR, The Iron Lady (#83) | tested as P(Ops) > P(Space) |
| b-21 | US, AWACS (#110), Muslim Revolution live, USSR controls Saudi Arabia: the event over Ops | include |

## C — rate tendencies

| id | precondition → expected frequency | review |
|:---|:---|:---|
| c-01 | USSR, turn 1, China Card (#6), an alternative card would do: low rate of playing it | include |
| c-02 | US, Korean War (#11): high rate of getting it played promptly | include |
| c-03 | USSR, Warsaw Pact (#16), before turn 7: low event rate | with "no US influence in Poland and East Germany, ≥ 3 USSR influence in each" |
| c-04 | either, Red Scare/Purge (#31): very high event rate | check headline + AR1 only; see `n-05` |
| c-05 | either, Nuclear Test Ban (#34): Ops ~98% | include |
| c-06 | USSR, De-Stalinization (#33): event mostly at DEFCON ≤ 2; rarely headlined | include |
| c-07 | US, Quagmire (#42): spaced, self-fired < 10% | include |
| c-08 | US at the headline with Grain Sales (#67): headlined > 80% | include |
| c-09 | ABM Treaty (#57): event > 90% when played | include |
| c-10 | US, OPEC (#61) worth ≥ 3 VP to the USSR: spaced/mitigated > 70% | include |
| c-11 | USSR, OAS Founded (#70): disposed of > 70% | include |
| c-12 | USSR, Liberation Theology (#75): event > 85% | include |
| c-13 | USSR, Ussuri River Skirmish (#76): spaced > 80% | include |
| c-14 | US ahead on space, Star Wars (#85), weak discard: low immediate-play rate | when the discard holds Voice of America or Grain Sales |
| c-15 | USSR, Terrorism (#92) after Iranian Hostage Crisis: event > 80% | include |
| c-16 | USSR at the headline with Aldrich Ames (#98): high headline rate | include; US variant is `n-06` |
| c-17 | US controls Iran, Iranian Hostage Crisis (#82): Space + Ops > event | include |
| c-18 | USSR, Latin American Debt Crisis (#95), no large South American stacks: low event rate | include |
| c-19 | turn ≥ 7, USSR controls East Germany and Poland: overprotects them more than earlier | include |

## N — reviewer-supplied

| id | type | claim |
|:---|:---|:---|
| n-01 | B | USSR with NATO (#21) and another 4-Ops card, neither Warsaw Pact nor Marshall Plan fired (NATO's event inert): spend NATO first |
| n-02 | B | USSR with live NATO and Marshall Plan or US/Japan: play NATO before them |
| n-03 | A | DEFCON 2, holding Wargames (#100), leading by more than 6: trigger it |
| n-04 | A | US holding only Ask Not (#77) and a scoring card that would take the USSR to 20: play Ask Not and discard the scoring card |
| n-05 | A | Red Scare/Purge (#31) on the final AR of a turn: never the event (the opponent has no AR left for the penalty) |
| n-06 | B | US with Aldrich Ames (#98), a scoring card that hurts the US, and other playable cards: hold Ames until only it and the scoring card remain |

## Excluded

Dropped as needing the opponent's hand, deck-counting or holistic judgement: Missile Envy (#49)
headline timing; Cuban Missile Crisis / Fidel foothold play; Wargames when the opponent may hold
it; Cambridge Five (#104) headline timing; Muslim Revolution (#56); Ask Not's hand-building (one
branch kept as `n-04`); the Truman Doctrine (#19) turn-1 trick; Chernobyl (#94) headline timing
(the guide contradicts itself; only `c-19` kept); Nixon Plays the China Card (#71) for the USSR.
