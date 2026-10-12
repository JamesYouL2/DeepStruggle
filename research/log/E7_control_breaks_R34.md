# Control breaks and Cuba on the R34 soup (2026-10-12): R34 breaks control twice as often as strong humans, but its breaks mostly pay; the clear mistakes are 5% of them, about 3.5 points a game, mostly the US's and turn 10. Cuba only while Fidel is to come, at no measurable cost

**Question (owner, from a game against the bot).** "It broke control way too optimistically at
times. US might also overvalue Cuba specifically, but I am really not sure about that one." Measure
both on the current strongest network, the R34 soup
`E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M` (leaderboard 1650): how often it breaks
control, against the humans of the ts-replayer corpus on their own positions, whether its breaks
last, and what they cost in paired playouts.

**Definitions.** A point of Ops influence into a country the opponent controls costs 2 Ops. A
*break play* is one card's Ops with at least one such point. Each country it entered ends the
action round *flipped* (the mover controls it), *broken* (nobody does) or *held* (still the
opponent's). A flipped or broken country is *retaken* when the opponent controls it again by the
time the mover's next action round starts.

**Answer.**

* **It breaks control twice as often as strong humans.**
  * On the humans' own positions, R34 puts 8.0% of the US's Ops influence points into
    USSR-controlled countries, against the humans' 4.2%. For the USSR it is 12.8% against 5.1%.
  * The gap grows through the game: in turns 8–10 the USSR is at 19.0% against 6.9%.
* **Its breaks are undone more often.** The opponent has retaken 44–46% of the countries R34 broke
  or flipped before R34 moves again, against 35–36% for humans. These are different games.
  * Italy is a tug-of-war in R34's own games. It is broken in 56% of them and retaken 66–69% of the
    time, and the breaker almost never takes control of it.
* **But the average break pays, and a searched continuation agrees.**
  * In paired playouts of 4,368 breaks, keeping the same card's Ops out of opponent control loses
    1.0 point a play, 11 a game.
  * The US's breaks with any card but the China Card are break-even.
  * On a 400-play sample with Gumbel 64:4 on both sides afterwards, keeping out loses 1.34 ± 0.54
    points a play, against 1.51 ± 0.41 greedy. A stronger responder does not punish the breaks.
* **The clearly wrong breaks are 5% of them, about 3.5 points a game.**
  * The US's are twice the USSR's in rate and in cost: 2.5 against 1.0 points a game.
  * A third of the cost is at turn 10: breaks the opponent can still answer before final scoring.
  * That is about a fifth of the placement leak measured on the same network (+17.8 a game,
    [E7_leak_by_kind_R34.md](E7_leak_by_kind_R34.md)), so "at times" is the right word.
* **Cuba: yes, but only while Fidel is to come, and it costs nothing measurable.**
  * On the humans' positions, with Fidel unseen, R34's US puts 9.0% of its points into Cuba, the
    humans 2.1%. Once Fidel has fired the two are equal.
  * Fidel later wipes a third of those Cuba plays.
  * In playouts, keeping the same card's Ops out of Cuba is worth +0.09 ± 0.15 points a play with
    Fidel unseen, and −0.29 ± 0.09 overall.

## How often: the humans' own positions

R34 at each of the 30,080 Ops influence points of 265 corpus games, against the human's own point.
Share of that side's Ops influence points, ± a standard error clustered by game:

| side | turns | humans | R34 on the same positions | R34 − humans | R34 in its own games |
|:---|:---|---:|---:|---:|---:|
| US | 1–3 | 2.6 ± 0.2 | 3.3 ± 0.3 | +0.7 ± 0.3 | 4.2 |
| US | 4–7 | 4.9 ± 0.3 | 10.6 ± 0.6 | **+5.6 ± 0.6** | 8.9 |
| US | 8–10 | 7.2 ± 0.6 | 15.9 ± 1.2 | **+8.8 ± 1.2** | 11.2 |
| USSR | 1–3 | 3.7 ± 0.3 | 7.8 ± 0.6 | +4.1 ± 0.6 | 7.8 |
| USSR | 4–7 | 5.8 ± 0.4 | 15.3 ± 0.8 | **+9.4 ± 0.8** | 11.7 |
| USSR | 8–10 | 6.9 ± 0.6 | 19.0 ± 1.4 | **+12.1 ± 1.5** | 14.3 |
| US | all | 4.2 ± 0.2 | 8.0 ± 0.4 | +3.8 ± 0.4 | 7.2 |
| USSR | all | 5.1 ± 0.2 | 12.8 ± 0.5 | +7.6 ± 0.5 | 10.8 |

* The point that actually breaks control is most of it (US 6.3% against 3.9%, USSR 11.0% against
  5.0%). The rest are points into a country that stays the opponent's.
* R34's probability of an opponent-controlled target is higher still than its pick rate: 14.0%
  (US) and 17.0% (USSR) at turns 4–10, against the humans' 5.6% and 6.1%.
* The excess, turns 4–10, per 1,000 points (R34's pick / the humans'):
  * US: Poland 9.6 / 1.8, Zaire 9.1 / 3.0, Pakistan 7.3 / 2.2, Argentina 6.9 / 1.6, Cuba 6.2 / 1.8,
    East Germany 5.2 / 1.3, Lebanon 5.0 / 0.2, Italy 5.0 / 1.0.
  * USSR: Mexico 20.7 / 1.9, Algeria 20.0 / 2.7, Zaire 13.1 / 4.9, Angola 11.6 / 2.6,
    Pakistan 11.4 / 1.2, West Germany 9.7 / 2.1, Lebanon 8.2 / 1.3, Nigeria 8.0 / 3.2.

The search bank of the same network says the same from the other side. Its 86,241 annotated
positions put 6.9% (US) and 10.7% (USSR) of R34's Ops influence points into opponent control, and
the reference search (Gumbel k=16 @1024, 64 worlds) values those points worse than the rest:

| side | points | mean regret (pts) | regret ≥ 4 pts | Gumbel k=8 @256 keeps R34's point |
|:---|:---|---:|---:|---:|
| US | into opponent control (breaking it) | 2.88 | 23% | 47% |
| US | elsewhere | 0.53 | 4% | 67% |
| USSR | into opponent control (breaking it) | 1.33 | 11% | 55% |
| USSR | elsewhere | 0.37 | 3% | 74% |

Where search's best point and R34's differ on this, over-breaking (R34 into control, search
elsewhere) carries 4–5× the search-valued regret of under-breaking (search into control, R34
elsewhere).

## Do the breaks last

| | humans (own games) | R34 (own games) |
|:---|---:|---:|
| US: countries entered under USSR control | 640 | 19,195 |
| US: flipped / broken / still held | 27 / 68 / 5% | 24 / 72 / 5% |
| US: of flipped or broken, retaken before the US moves again | 34.5 ± 2.0% | **43.8 ± 0.4%** |
| US: of flipped or broken, the USSR's at the turn's end | 25.5 ± 2.1% | 33.2 ± 0.5% |
| USSR: countries entered under US control | 732 | 25,619 |
| USSR: flipped / broken / still held | 31 / 67 / 2% | 21 / 76 / 3% |
| USSR: of flipped or broken, retaken before the USSR moves again | 35.7 ± 1.9% | **46.0 ± 0.3%** |
| USSR: of flipped or broken, the US's at the turn's end | 35.9 ± 2.2% | 44.4 ± 0.4% |

Both spend about 3 Ops per country entered. R34's breaks flip control less often and are undone
more often. This compares different games, though: R34's breaks meet R34's answers and the humans'
breaks meet humans'.

**Italy is the clearest case.**
* R34 enters Italy under the opponent's control 10–12 times per 1,000 Ops influence points, from
  either side. Humans do so 1 time per 1,000 points (34 times in 265 games).
* In 4,096 self-play games, Italy's control is broken in 56% of games, 2.5 times on average when
  it happens, and up to 13 times.
* A break spends 3.15 Ops for 2.1 influence. It flips control once in 2,801 US tries.
* It is retaken before the breaker moves again 66% (US) and 69% (USSR) of the time.
* Half of these breaks are in turns 1–3. The Europe Scoring card is in the breaker's hand in 20%.

## What a break costs

Every break play of 400 R34 self-play games (4,368, 10.9 a game), at its first point into the
opponent's control. Two lines per pair, sharing the redeal and the dice:
* R34's own play;
* the same card's remaining Ops kept out of every country the opponent controls. R34 still chooses,
  but those targets are masked until its action round ends.

Then R34 greedy on both sides, 512 pairs. Gain = the kept-out line minus R34's play, points of win
probability; ± over plays.

| group | plays | mean gain a play | per game (both sides) |
|:---|---:|---:|---:|
| all | 4,368 | −1.04 ± 0.10 | −11.4 ± 1.1 |
| US | 1,821 | −0.82 ± 0.19 | −3.7 ± 0.9 |
| USSR | 2,547 | −1.20 ± 0.11 | −7.7 ± 0.7 |
| US, turns 1–9, the China Card | 204 | −3.67 ± 0.43 | |
| US, turns 1–9, any other card | 1,424 | **+0.00 ± 0.12** | |
| USSR, turns 1–9, the China Card | 265 | −3.74 ± 0.31 | |
| USSR, turns 1–9, any other card | 2,073 | −0.62 ± 0.08 | |
| turn 10 | 402 | −3.83 | −3.9 |

**In R34's own games, breaking pays on average.** Keeping the card out of opponent control loses
a point a play. Most of that comes from two groups:
* the China Card's breaks;
* the USSR's breaks.

By country, breaking pays most in Thailand and Pakistan, from either side.

**The US's other breaks are break-even.** Some are right and some are wrong, and on average they
gain nothing for the 2 Ops a point they cost.

**The clearly wrong breaks.** These are plays where keeping out wins at 2 standard errors. They
are selected on half of the pairs and measured on the other half, so the selection does not
inflate the cost.

| side | turns | break plays | clearly wrong | their cost a game, held out |
|:---|:---|---:|---:|---:|
| US | 1–9 | 1,628 | 6.6% | +1.40 |
| US | 10 | 193 | 13.5% | +1.14 |
| USSR | 1–9 | 2,338 | 3.1% | +0.57 |
| USSR | 10 | 209 | 5.0% | +0.44 |
| all | | 4,368 | 4.9% | **+3.55** |

* The US's wrong breaks are twice the USSR's in rate and cost. The search bank agrees: 23% of US
  break points are large errors against 11% of the USSR's.
* Turn 10 is a third of the cost. These are breaks the opponent can still answer before final
  scoring, at up to 83 points each:
  * the US into Poland and Italy at T10 AR6;
  * the USSR flipping Nigeria at T10 AR7;
  * the US into Italy, and into Angola, at T10 AR6.
* Whose card pays for the break makes no difference: own, opponent's and neutral cards are wrong
  at 6–7% (US) and 3–4% (USSR).

The 56 plays before turn 10 that are wrong in both halves of their pairs are listed below (the top
14), for review. Twelve of the 14 are the US's, and in half of them the opponent retook the
country before the US moved again:

| side | turn/AR | card | VP | DEFCON | countries entered (end, retaken) | no-break gain | position |
|:---|:---|:---|---:|---:|:---|---:|:---|
| US | T3 AR3 | Five Year Plan | -6 | 2 | France (broken, retaken) | +22.1 ± 2.1 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_Rc_uxLgS9L8yipCxCaJtQQswGmjqv--w0Adp3G20aparbQR8mWOYW5nsJi3TDS-M5pbM-g2WxV51gjbcmUaEXCXrX6QnMGo8hruS3gu0qCALCaDpifDewUyhfcan2Et-SdG8CbEkOQ1BLzvtOAY9t6oVtps9UQv4F6Jk9goma0I4P6VOy-an5Cat0woZV64FFadADU9Tmll07mQ1bAw6KMTzJ96UFTlmTxK7fnOCu1ly03vcG1EwcAumjuIV95LyxsgyNsTYtq0MgJ80_nIUYzF9PwAH3EWaAwOdBpp7aVuO70DOxgTenDGHD8KNciLmdYohYqsPICWADkvg5tFwJxRx4BFbWEazOdQTsaeRS_sTnqUISTreae3YEc3kocgguvPjBVVRZ8XyyWtirosl4u8LiEWN_Q9mHEfLnNMDery3SGY3kDumn22YhUEc4T0hkR-6A0Jec_vZYhcMbT4Y4boXQzRC4boDEOsnqWIfqaIzlBEv6SIfqKoXNzmKLp-PyHk--kgf3XDkBk6yCwb5J4NQ75zwzzIeJDxIONBxn9Cxnpy5OKt7H2c2wzWhoz2e-EgPhaY2DZGT155a03fgz5vOMNFW2MbkNGRrr2CJrQCfBDgZ-Ydd-Io-ZFlebaXolWdlnzbWedTigp69cG86HTsGz84CR61aQktrr-kNU8Xi2K5TGcPbs7s4Jyd6FYCnGg7eeYaEcjGDBqlt2yalVSVKOLpCqqPxro1bcyllmBCKi_GqqDjGxvfUjrd0DRQBdlqK5STwH3cLO-hKq3roP6ASrndyiaUfQHHQfZcM8KKBYFjSeA4lMrkOG2xaoUP56GKFkW1LAtGS1IyWpMSDAw2nnshX2NtQkmV2JB8NAMVvpfYFJxbhBLvNTYL801Ema7Q4JRjY0Hy4svmYyr9piVao-OHTsX_B0XR9QL2l7fYeyAEpE-D-2yonJisUWLpqpK7bHSFTdwpLlwrko5bybg1biZjnVwfw2MJuAgQ_wfhr3Pivel0qIun-v0XQgG51Q) |
| US | T7 AR1 | Bear Trap | -4 | 2 | Mexico (broken, retaken) | +18.2 ± 2.2 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV-GK4yAQfhd_Z0FjUtu-ylHETWwbzqqo6W5Z9t131CRNtj3ownFwUEKp8804zsw3Cc4HEk3ojObO9LpFW1KgRriWK9OIiHu0_YWLEh6SnjX8WPqxJLNJYoPFd_mKkeRnNSHZ52qyYZOGFPWwqgdNndZ00FaTph7244efnEuOCOQdJHzstOAp7aNRrXS5DFfUKnERr0pmPLxzH0TzGyrzgYRS5o1L4dQFUGPRFheolU3nY1HjxsHdhIWLBUd1geRZ6sAPTuggW26sT3szCgccZJJP4p1b6XgD_AR3SZg2rcwAf-1CpiinYiw_gTJZgccYQKfRlhbISt12-gDnpJzQtqrmoOdnoXq5MHVGqeTJyRO4iZAPMsZJI-aNOkcsu4tmYM-hnYy7ikG4gwxJhpxc4J3ewzm6kTxmEWMnZLUua8IoZmVF15uSYFpQXJV4Axn53lo4y4-B81Sg5DB0p3j-K1SwOaJtWUNRz1DkWM7ReSzLZ3GPJ_IQT-Sf8oTv8ITv0oQfoWn9N2nKgT9OB_4JGfdeGnxDBv6fycBPMp5kPMl4kvEk48dk7GbXLt5KG7Jt0zsXK2qPwkN-ZWRi3xg9W_LWGWvBXzC8TJv2xjUgp0C69gaa0QrwSUCcKHjuxVnyc4kKdJSiVZ2WfN85H8bLFLtRmDcd_byQmcZLCKkd9rDNrWbYtNiT24XOoN7fc9F772bhKAFBtJ28cp0QqMYdNEsfaF6VoSuTmO5W0H0k961pcy21hCOkCmLqCjKtymk1lNP3TQNdgLZ7obwE7vPL8hm70vkO-g-olPu9bGLbE8qqumaUsBquMZHh2Cizi5pLPStCvKStKKVVTdarNWEbui7ZKvZW76AToBJQrqk1oaOqNDKM40AFIw1OI8HtqECSZlzhhQ2ZsDiOjL4w-MYz3_mfLfyVM4nO1tmeVCOwS5GfOjV9P0D0VsD7FVyaP0iEgPR5dt8zIGQRUDXMQFeUTgFd56O4kSYbMlnRRUrfnzGR6o8W2dc1nN0Q_DJBABYpVvGLEL87F25Np2NnvFSfX-Tyu3k) |
| US | T6 AR4 | Muslim Revolution | -9 | 2 | Italy (broken, retaken) | +17.6 ± 2.5 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tq4zAQ_Rc_u6Bb7Dq_shSh2kpiVpGMJKcNpf_e0SWOQvKQLmUXliASS2cumpkzMvJHJXo_Gs2tmfVQrVld9cIOXJleBNxV61-obpbxDD9St3ngZRZGk_-bPMeFXZMl7GKNwVepQUCe9HBhzRbv4ceyhOTdkw66e5S7w_oFEt6NWvCY9s6oQdpq_YQv4EmJo3hVsloH3L9z50X_G0rzUQmlzBuXwqojoGaq1qiuBtmPLlQ1GAZ_uMD8cQJHq7qSB6k931qhvRy4mVy0TShssJVxvRfvfJKW90CQt8eIaTPIBPDX0SeOUi5m4nsQRi3wGAIYdaR1knoY9Rb2iTlBBE0JOn4QapYXqtYoFT1ZuQc3AXJehjhZwJxRh4Ald0EN9Dn0k7HnpRd2K31cQ07W81FvYB_dSx6yCLF33Yp0HWspRl3X4uf2GTqhIQR1kJCbpwm2cqe4eaxP9OfHfdj-FQrY72KND1DhUMuT51CTz_pekp5-kiX8ByyhGyyhmyShe0hCP0lSCvx-NtA1G7jBqMGkpc23aEFXrKC_enb-c1a-d0YeZDzIeJDxIOPfkPFS3L74ICefdPvZ2lDRaScc5EcCE5ve6GLKB2umCfx5w0k02hjbwzoGMg5XUEErwHsBcVbecScOkh9IVVc7KQY1ask3o3X-dKdCVwLzpk-Xi0XiJIQ0ZBuM6LUoW10YpX4plWd308fsnD1zpgREMYzyTHZEoBw30LT6qMqy5LaMS7KCKxG0H06Na4ZUTC1hC6m8WNoCLzOyzHI93dz30AbVeiOUk0B-Oi2foS2tG6EBgUu52cg-9D1eIcpQQ2kTtg4Uh04pLtY2Nq3w4KNDjLWMUcRaRLqWtitoAD9b6AS47EK1ltaEjiL1qvgswDQ-2BnIT5qfJP5wISfLE-dxsmaF49IHy35K-7P3W4PGkwOR70eV3h8kLt0k4Hx5Gz9DWICA8jK7S9cExxhO30lNjCTES5es6FXUt-OhRbz4hjzJ7vGFAgEvOfYlP5qBiwxpeCGE186RT2bUoS-eus8vdM69iA) |
| US | T4 AR1 | NATO | +1 | 2 | Algeria (broken, retaken) | +16.2 ± 3.3 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV21v4yAM_i98zkmBvPTlr5wqxBLaoqOAgGSrpv33M5CmtMu0TTqddLoKJcUPxth-nNR5RazzQitq9aB6tMUF6pjtqdQdC7hD259l0cLAMJpiDffVNDCgTZy1k0yinKQm7kn72ojheZDpwrNGvnL5beFO4qnNgsZlpLXyyyM_D-QdBHwUitEY9lHLntuUhitqJDuzJ8kT7l-o86z7BZl5RUxK_Uw5s_IMqDZoWxao551wIalh42RuxvzZgKGmQHzkytODZcrznmrj4t6EwgEHHuUTe6GGW9oBP96eI6Z0zxNAn4RPFKVQtKEnWIxaYDE4IBTa1gUyXPVCHeCcGBPaEpyDjo5MDvxG1WopoyXLT2AmQM7z4CcJmNNyDFgyF9RAn0I5aXsVPbMH7qMMMVlPhdrDOarjNEQRfN9s6hrXTbuuV-tVS0CCEmogGDcYA8e4i8805iba8uIUjn6C5HVH8KeBfI6Q35DJi92QkbfiqxSV7yha_VWKygWKykWGynuG6gWGyk8Z-oHvKCLVxxwl179OSPnn6Sj_FTqWHpjP6fjGA_Mg40HGg4wHGf8JGbus56I9Nz7pdoO1IaPmyByPLUnP951W2ZT2VhsD9rymJG7aa9uBHB0R_TsooxXgEwM_kXfUsZHTkaACHTnrpVCc7oV1fkoR9Cv3C_pZ8ek_dl5xHFzqL91X-35l2nSzJ5VLlUGDWzIxOGczdyQDJ3rBr1xHBLKxgCbpFeVZmaoyirG7gurDqW51n3KpOBzBpWdzVeB5RubZlE43dB1UAdrumXQcuE8Py1uoSusE1B9Qyfd73oWyx2W9hhVgNhRI1kXbWKvMh1a8aupNjasKkw30jU3brsBNP9jU70Ka5pKESqpj33_5BmiKOt4_-m7AN3OSSQ1cJI56Wg261YKNOrNT3dggC9rT2EXHT0LOrw0QnWHwWHkbvzlIgIDrPLglUxeXmuhykHF2JzeukCmUVZxdncZzyN8ZVZZCkoe7m1y_DQ-A-wBHEV42Z2q0ULEc3n4DeAG4Ww) |
| US | T3 AR1 | Red Scare/Purge | -3 | 2 | Lebanon (flipped, kept) | +15.8 ± 2.9 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReeXQmwEzf5lVWEqI0TtAQQ4LRR1X_f4VLHaay9qNrV7ipBSpgDDDPnjCPPK-JdkEYzZ0bdoy2pUMddz5TpeMQ92n7BFa1ItarW8P2Y5m0ZBKwWVjKSV1Zpb5u-SVWXs3En_isGKaPYO0j4IDVnKe2DUb1wmYYLahU_8yclMh5emA-8-wrMvCKulHlmgjt1BtRYtMUV6kUnfSQ1HizuJiycLThaVUichA5s77gOomfG-nQ2o3DBXiT7yF-YFY51oE9w54Rp04sMsCcZskQ5FWPZERbTLvAYA5AabZsKWaF7qfdwT8oJbWsyBz07cTWKq63OKJU8OXEENxHyQcQ4m4h5o04Ry-7iNtjPoJyMu5iBu70IyYacXGBSD3CP7gSLWcTYN5TWq_WmbR_bFjcNTDfVuoFk_GgtXOPfY2aJm-QryGO8-gnI6w5oS1fA5wn4jUy--42MvFX_ikRkQSK6XpSI_IxE-DdIROkGk2Zdk0dCflUi_FmB8I1A-I8KhBcEwov64D-uTw78LsZdjLsYdzH-ezF2s7cw1gsb8t5udC4yag_cQ340KjF0Rs-mrHfGWvAXDKPp0GBcB3YKRPY30ExWgI8c4kTBM89Pgp0oqtBB8F5JLdggnQ-FouYGN8-6vGVMC15AQH05QRZWypmHq6VcLfUMGv3CraP3buZacYihl-KidEKAiwU0W69ozkmpyWSmty2oPZKr1vSZSS3gCqECn2qCTDM6zQqZfuw6qAG0HbjyApTPj8pbrEnnJVQfCCmGQXQhV2EUNdbG7H3NpTLlIWq9aki9xrRZE1xT-DQQYhidTlQBQ1MxQg1R6Ecu_UADPQq-9ANl1KVXuF6ZdzE0zXMv0UydBf5hn3N903d271LgR6mmPwwwveXwQAWX-g8SIdB5ntySq3r6rWe_NIVSTyHUs4A-JlLfEPTJsSuhT-nRAnxM8CTj38yZWSN1LIWH-u0bjR26KQ) |
| US | T7 AR3 | Duck and Cover | +1 | 2 | Poland (held, ?) | +15.3 ± 2.0 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9uO4yAM_ReesxKXXNr8ymqEmIS20VKIgHSmGs2_ryE0JdM-VLPSPuxWURJ8wMb2cSL8gUTnB6O5NZPuUcsK1Anbc2U6EXCH2p-4IEUFN4V7AzcpmnRV2bhJY5KufEwydH6zZJMUdXqTzEK9vNmChzusDTObxR5LT_zwRcHKxQ-QXyDgw6AFj2EfjOqlRS1ZoaMSZ_Gq5Iz7d-686H5BZj6QUMq8cSmsOgNqRtTiAvWyG1xIalBM5hbMn0cwVBVInqT2fG-F9rLnZnRRd0Zhg72M8lG881Fa3gE_3p4jpk0vZ4C_Dn6maA7FjPwIk3EVWAwODDqyOkrdD3oP-8SYUFvmmOMnoSa5WmmNUtGQlUewEiDnZXCTBcwZdQrYbC0sg_UcqsnYq-iF3UsfZQjJej7oHeyjO8lDEMH17bZimFC8KWnT0A0hVVUwzBhhEI-bxhG2che3eUxPtOeHY9j-FfLXHVBLQQ2dIMUhmRfbISmfxaMs_bilCf9VmvAdmvBdmvAjNOFbmqrtt3maPX-cEPyndOB_no3vfzRPMp5kPMl4kvGfkPGSHbt4L0c_r-0ma0NGx4NwEB8NTOw6o7Mh760ZR7DnDadRaWdsB3J0ZOhvoIxWgI8C_ETecSdOkp8oKtBBil4NWvLdYJ2_HKaamwnzptPRb5lwEjzqkwrB5e1UUvqx0prrhWXQ5O7tOzlnc-NKgBv9IK9sRwTycQedpQ-U5yXVZRTj-Qrqj8yVa_o5m1rCFlJ5sdQFWUZ0GaWEuqnroA5QuxPKSWB__lw-Q11aN0AFAplyt5NdKHxSYUbrsqk3YevAcSiV7EhtY9UKDza2DW4Io6xmVcUagmvQ8JOFSoAEQbaW0oSKKqGJuLYD0HzQVX8Qu4KsoyCr3oHG_iFfHfqiMumUX3oNkr2vY5aha5067hFGZfxywPPjoOb_RxlFNwr4vryN_QcOEHCeR_fVgS08N0liKfI6ukDvOICTFJzM03JxeZ2qPDH30rWWNsl6tPySXF_Cu8SyCpCG_0H465z5aAYdq-LzNx7SuuE) |
| US | T3 AR1 | Red Scare/Purge | +4 | 2 | Lebanon (flipped, kept) | +15.1 ± 2.9 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV-GOoyAQfhd-uwmi1q2vcmkIq9iSo0AAu9ts9t1vALV67Sa9veRyuWsmWuZjGGbmGy2-I9Z6oRW1elAdavIMtcx2VOqWBdyh5hvO8igVXM9xtIljApJndbzqiOQRSXqyLyJSwb3O8F8hmxhlNek7SPggFKMx7YOWHbeoecpXsJHszF4kT_Xxb9R51n6H0rwjJqV-pZxZeQZUG9TgDHW8FS5UNSwM_vIF5s8GHFUZ4ieuPN1bpjzvqDYurk0obLDnUT-yN2q4pS0Q5O05Ykp3PAH0RfjEUcpFG3qEyWgFHkMAQqGmzJDhqhNqD_vEnFBT5EvQ0ROTA1-ZWi1l9GT5EdwEyHke4iwD5rQ8BSy5C2ZgT6GftL2ontk991GHnKynQvWwj2o5DVmE2LeEFCV5Lqvtc4U3ZFvhOtuUkIwbjIFt3BQzjbWJvrw4hq1foHjtATWkgnqeoL6hkpPfUJGP7F6Knq45wn-UI3yDI3yTInwPRfiaorz-Mkcp8vsJwb9LB_7n2fj6A_Mg40HGg4wHGf8JGbvFmYt23Phk2w7WhoqaA3OQHwlM9K1WiyHtrDYG_HlNSVzUa9uCHgMR3RW0oBXgI4M4kXfUsROnJ4IydOCsk0Jx2gvr_PS_iourGf2qxoPfPOE4hNRdyvrzxLjkabUmtcvS_-Bubjs4Zy--JYMYOsEvXEcEqnEDTdo7WlZl7MqoxtMVdF-e-lZ3qZaKwxZcejZ3RT6PyDway-mGtoUuQE3PpOPAfXpYPkJXWieg_4BK3ve8DW2fY7KBGWA2NMjiFG1jrzIfjuLFtiS4hiNjVZKy2mzLAsL0g1WxXlCmuSWhkxZHfpBtVn76gZCP11omezJblKP1te1tn59pK9nFwI9Czq8NUJ1h8Fh5G785SICA6mVya9d1vJMRKUDSbwo_zU2zl9G1FHel9guyG0Of05tyWSWYh9dAeNmcqdFChXYoP34Aete3jQ) |
| US | T8 AR2 | Nuclear Subs | -2 | 2 | North Korea (broken, kept) | +14.9 ± 2.7 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReevRLgW5xfWVWI2iSxlgACnDaq-u87XOKQJg9VV9qH3Qo5YQ5zP2PLfkN89LNWzOpFTWhLKzRyOzGpRx5wh7Y_cUWqLl5dtYm7Pq-0o6uU5A6uJu9IRglokXgSvNAVTatbdYJliZA1Gsme-6jT5Qh93HVrpOuixT5FbIqTNfYTFHyYFWex7IOWk7BoS25QI_mZP0uRcP_KnOfjL-jMG-JS6hcmuJVnQLVBW1yhSYyzC00NhtndivmzAUdthcRJKM_2lisvJqaNi7YJhQB7EeUjf2VGWDYCP96eI6b0JBLAnmefKIIFpWjDjnAYtcBjSGBWkVUj1DSrPcSJNaFtQ0rQsROXi7hRtVrK6MmKI7gJkPMi5EkD5rQ8BSy5C2qgz2CctL2Kntu98FGGmqxns9pBHDUKFqoIuQ9DW5OW1g3uh03fNz2tK1ITugncuMUYCOUuebPYn-jPz8cQ_hkaOB4gpxZ6eoIeh25efIeuvFdfpwn_VZrwA5rwQ5bwZ1jC9yy1X2YpJf55OvCfkoH_eTK-fst8k_FNxjcZ32T8J2Q8FW9dbBLGJ91xsTZ01By4S28uk9iNWhVbNlltDPjzmtFotNN2BDkmMk93UEErwEcOeSLvmOMnwU4UVegg-CRnJdhuts7nFvV3uH5Rwc0PUpw4ARlN2YQ29yfZ6MYmTUtdQIt75GJxzl6zkRxymGZxZToi0IsHaJLeUNmTPJNRjG9WMHskTa2eUieVgBBCer7OBFl3dN3lZrplHGEG0HbHpRPAfLpV3sNMWjfD9AGRYrcTYxh6QgdMun4gTQcKQG-YkuItzcaB5R5cdO3Q4X4z0KZuKKV4aIB8v1iYgk2FoFfrWMI01fANgPMiVQsXzt8JAbmcNfEUZzT91qAH3w-brNJFwwA3-bhZVXFhdtVMjtvitIaF7xbJtztkfpxlenbQKDrD4d7yNn56RAgYL6v76Kq-2ZM1iUugek0syaSLJqQa8j-5cVO6uxj2BU5uNNrqYyo0Bm9ydZD7Wl-dgZsK48MgPHLOzOhZhbH4Qd9_A6bLuts) |
| US | T3 AR2 | US/Japan Mutual Defense Pact | +0 | 2 | Thailand (broken, retaken) | +14.1 ± 2.2 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV_9u2yAQfhf-diVsx3GdV5kiRM0lQSOAAKeNqr77DnAcZ0nVrpOmaYuQEPfB_fzOFrwS3gdpNHNm0IKsqoL03AmmTM8j7snqGy3KoiqWODfFI84tSjXOLcpxLsdVPtHgbj6TkTyWqEH_ilGmqNqTvMaEd1JzltLeGSXAkVV5gVrFj_xJQcbDC_OB99-xMq-EK2WeGXCnjogaS1a0IAJ66WNRo-JobsLC0aKhpiBwAB3Y1nEdQDBjfdLNKDrYQpL3_IVZcKxHfoI7JkwbARlgTzJkinIqxrI9bqZTaDEGIDVZLQpiQQupt-gn5YRMt3PQswNXA1wcdUapZMnBHs1EyAeIcS4i5o06RCybi8fwPMN2Mu4sBu62EJKMObnApN6gH90Di1nE2LuqqptlV9clLdt20bRdk3Lxg7XoxZ9CZqk0yVSQ--j5CWvX7zCVBst5wPLGQp7MxoK8FZ9l6OGaIvpHKaI3KKI3GaKfYYheM1Q2X6aI_iIh9HfpoP88G1__Xu5k3Mm4k3En4z8hYz27cjEBNuSz_eBcrKjdcQ_p3ipg0xs9WzLhjLVoLxhWJaWNcT3KKRAprqAZrQjvOcZJgmeeH4AdKlKQHXChpAa2kc6H00Wqu9owz3q89k0bHjAiMarU1xujysOFTu6W-enB3_I6eO_OphXHEISEM9MJwVrcQLP0SuY1GXsyieluhb1X5q41IldSA7oAFfjUE-W0qqbVWEw_9D32AFltuPKAzOdP5S32pPMSuw-JhM0G-tj0j81j19WLpotXUWQ3NsnsIu1Sv_IQb-N1t6ho29Bls6iWy452JVYlDE6nOmCtprbEbqouniALfATE-f1nwu01RStlQspRO0sfPW_KD_ansU6B76Wafh0oesvx0wouPTvKCCHh8-R-dhRDW04B1wmv8U1G0-vshJ3SOes2F8HWFzY_P94txnoMfUqvGoGLBNOvIP5wjswaqWNT0Lcf9HK4XQ) |
| US | T2 AR6 | Socialist Governments | -1 | 2 | France (broken, kept) | +14.0 ± 2.5 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV-uK4yAUfhd_p6C5NnmVpYgTbStrNajpTBnm3feoaSbdhqEwy-7-KBKqn-f-naT6jljvpdHUmlFz1NUZ6pnlVJmeBdyh7gfO6ozAU2dVto2_JGtgkHkeZs0k1cSnib_1rBkw_F-M36LZQcJHqRmNaR-N4sKibkNu4EGxC3tRAnUB92_Uedb_hNK8I6aUeaWCWXUB1AyowxniopcuVDUoBntkgfnLAIaqDImz0J4eLNNecGoGF3UTCg4OIq5P7I0OwtIeCPL2EjFtuEgAfZE-cZRyMQM9wWaUAoshAKlRV2ZoEJpLfQA_MSfUNUvM0TNTo7iRtEapaMiKE1gJkPMihFkGzBl1DliyFsRAnkI7Gfu59MwehI9rSMl6KvUe_Ohe0JBECL3N86LMt2XVkLZpirZo26wuIRc3DgO4cdeQaSxNtOXlKbh-gdr1x1jeMxQ3lPFqNZTjI3uUn82fJChfIygnpKiK-mua8ApNeJUm_AhN-E_SlAJ_nBP8XU7wHSX4r74zTzKeZDzJeJLxJOOfk7FbnLooF4NPsv1obajocGROxP9dLva90Ysp5dYMA9jzhuZRaW9sD-sYiOR30IJWgE8M4kTeUcfOgp5zlKGjYFxJLeheWuenEuXV3YZ51dPJb95wAiLikwop7ncmnc2NUmqXpfTo1tyOztmFbcUgCC7FJ9cRgWqsoGn1jpZVmboyLvMKPEH3kdS3hqdaagEuhPJs7goyz_J5NpXTjX0PXYC6PVNOAPfpZfkIXWmdhP4DKsV-L_rQ9ttq27ZF1cCRCQV-Q5ssTtM2dizz4UheFLhuigrjosWkDDM42vrRpj6AYs2NCf1UxivJ9TIQrjM4ywEhq5eFgBbTPIfxuZNPuyQrJ8nygcsH-WJ1M3Yx8JNU6eOR8nADg5fL23j3wAECxpfJ3Zous3ZhsoyJkCmd8ia1q06-Gkyxin5j7KbQ5_TKCbhJMB7qwyfnQgcjdWiKDfn4BXerufE) |
| US | T7 AR5 | De-Stalinization | +1 | 2 | France (held, ?) | +13.7 ± 2.6 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV81u4yAQfhfOrmSwHTd5lVWFqE0SawlYgNNGVd99ZwbHIU0OVQ972I2QE_hgfr-xxXww1cXBWendZHu2aQrWKd9L4zqFeGCbX2XBl_EMTzsPns1xrAi7PPi7olNixnhREbKC0Sw6L5rSr8ikK5p_1fxMEg2MdtZRfnusZmurJPUCAe8HqySFvXem155tnvgVPBp1Uq9Gsw3i8V2GqLrfkJoPpoxxb1Irb06AupFtyoL1uhsCZhUFUR_PsHgaNSVaH7WNcueVjbqXbgwkm1AwsNO0Pqh3OWovOyAo-hNh1vU6AfJ1iImjFIsb5QE26RRoRAcGyzZVwUZt-8HuwA7FBNgVGORRmUlfHfXOGNLk9QHUIBSiRj8rxIIzR8SSOjwG5yXUk_OXZVR-pyOtISYf5WC3YMd2WmIU6Pt6Xde8blrRrETTlBXnUCdlVbVrCChM4wimwtlvSfkhfXE4oPlXSGC3J-QIGcZcnjVjTj6L75L0dMtS-VdZKu-wVN4lqfwOSeUtSY34MUvJ879IR_nPs_HzV-ZBxoOMBxkPMv4TMl6yW5fs9RjT2W7yHjM67lWA-AQyse2czaay924cQV90UpDQ1vkO1uTI0N9AGa0AHxT4yWKQQR21PApWsL1WvRmsltvBhziniJfVzY57s_PVb9kIGlzqL2n9ujGLPF3JpHLJ9U_hrtkpBH_RbRT40A_6wjUhkI07aFp9sDwrc1XSEu5lUK6u56luXZ9yaTWY0CaqpSr4MhPLbE5nmLoOqoBttsoEDdynl-UTq9KHAeoPqNTbre6w7OFKWJXtmjfP7RqjRI6xVLIbtaeqVVFTInhTi7oWbQtXSL5uUSZOHmqhLRgkbClOqKmamo9zQyCgFSipkUjrvJng8yOWs4LambQnCMH_mpCK_kUmfx5NhtfU1uSWcl_O5wS9OuD4YTDpAyJoGUYFL1j01H80CAHpeXDXhttiPZtPvU9D5qrMcFXcupvczBNR3emxzkFXWYB8Ri_h3I5q_qCR60t451iuAhT4QcDPzkmObrBYGPzzDxA7uyg) |
| US | T2 AR5 | Comecon | +6 | 2 | Italy (broken, retaken) | +13.6 ± 2.8 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV-1u4yAQfBd-uxLGn8mrnCJEbZygI4AAp42qvvst4DhOk-raU3U9nSIUhx1g2dlZLPyCWOeFVtTqUfVoXWWoY7anUncs4A6tf-CsznJo4dnCs4l9Av-hT-Kzmcbr2PKsjKOnlvr4n2gpyuZkb4DwTihGI-2dlj23aP2QX8BGsiN7lBytA-6fqfOs-wmpeUFMSv1EObPyCKg2aI0z1PNOuJDVsDD4yxeYPxoeE80PXHm6tUx53lNtXFybUNhgy6O9Z8_UcEs7EMjbY8SU7nkC6KPwSaPERRu6h8E4CzyGAIRC6yJDhqteqC3sEzlBTOUSdPTA5MgvplotZfRk-R7cBMh5HuIsAua0PAQsuQvTYD6FetL2bHpmt9xHGzhZT4UaYB_VcRpYhNhXhBRV3bZVBb26WIGV1SWQcaMxsI07xUxjbqIvL_Zh60dIXreL-T1AdkMeT15DPl6zjwr08JUK5bcUahtS5hUpCP6NVviGVvimVPgjUuGvlCoF_nFd8LUuZUmKuv2UOPhKG_xXT89_rsjnTspdjLsYdzHuYnyPGJvF_Yv23Pg0txutDRk1O-aAHwlKDJ1Wiy7trTYG_HlNSVw0aNuBHQMR_RW0kBXgPYM4kXfUsQOnB4IytOOsl0JxOgjr_JSi8grXT2q6As4DjkNA_bSiuR6YljxcrEnFUiyg0d3YdHTOnj1LBhH0gp91jghk4gaarBe0zMhUkdEkFdyJoPLyVLO6T3lUHLbg0rO5IvK5R-belEo3dh1UAFoPTDoOuqeD8hoq0joBtQcy8mHgXSj5ospXZFUTTIqyamEO6BvKZHGvtrFimQ-X82q1anHT1quiqDAhuKlAfj_aVAeQrrkwoZ5IVi0-C0po730y5NOvWGAE2ttZ5eL_7egtn-9ZF20TA98LmV4eiYczDA6Xt_ErJA8QiL4kd2u7U0iJagjzTKiY6S-J5VM_n2f92TfXct0F1c0U-kwPT8AFwfgyCK-cIzVaqFAY9esvkGG7tQ) |
| USSR | T9 AR2 | Nuclear Test Ban | +8 | 2 | Italy (broken, retaken) | +13.3 ± 1.9 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReevRL4bv_KqkLUJglaAhbgtFHVf9_hEuI0eai60j7sVlZi5swwtzOJ4A2xyQmtqNGrmtFYFmhiZqZST8zjFo0_cdEWdfiURQ_vsujSc11dJQIW_t2GVXw62NulVZv1XfDq7UjQd0m_9RDXJOlIXpPwjlFJ8tKG3GKuJNmQFDnuLDdek8UTFHwQitFQ9kHLmRs0_iA38CLZmT1LjkaPu1dqHZt-QWveEJNSv1DOjDwDqhc04gLNfBLWd9VvvPjLoDsv4KkpED9x5ejeMOX4TPViw-aIQoQ9D_KRvdKFGzoBQ86cA6b0zCNAn4WLJMEDxeiFHkEZrMCjz0AoNNYFWriahdpDnFAUGqsb0NITkyu_MTVayuDJ8CO48ZB13OdZe8xqefJYdOfNwJ7CQGlzFR0ze-6CDDUZR4XaQRw1ceqr8LmTkvTtgPuGlM2Ah6qv2r4gFSGdp8euywKx7CVxGhoUHDpx9PGfoYPTAYa3gaaeoMm-nRfnvi3vxR8Qhf8qUfgBUfghT_gzPOF7nnyTvkhUzPzzhOA_pQP_82x8_VfzTcY3Gd9kfJPxn5DxtDl50ZkvLtpOqzG-o8uBWR7OrzPfTVptlnQ2elnAn9O0DJt22kwgh0TEfAdtaAX4yCBP5Cy17MTpqUQFOnA2S6E43QljXWpRd4frF3U5U2SN5ZDRnLb0zb0mbbrZE6el2kCrfeRitdZcs5EMcpgFvzIdEOjFAzRKb2jbkzSTQQxnK5g9EqdWz7GTikMILh3LM0Hyqsyr1Ey7ThPMABp3TFoOzMefyrufSWMFTB8QyXc7PvmhL7uBdFU1DE0HzfD8-jHZHtRMGFnm_Lm86vu669u673rStG0HbwiwGhiEoUDQrjyZMFAV3Apwfmq4B2C4G0SpCRIuqqTDQSZJ6-8QcOO4qHHaWCaHbTBss_E1yHZd5_AkSNUmgTJ9Gv_9FDI_Cpn_PkC0C4OflzPhBlJ5CEjfVoeL22dba6y8ykEiVmVtSKpMudXBiGTlcOeObL63mV87UBYf07nYxuog91xfmYCPFZ6E_9c500UL5Sejf_8N5Ui85g) |
| USSR | T7 AR1 | Bear Trap | -11 | 2 | Zaire (flipped, kept) | +11.8 ± 3.0 | [open](https://mihaild.github.io/DeepStruggle/?pos=eJztV9tu4yAQ_ReeXQkMdur8yipC1CaJtQQQ4LRR1X_vcInjbLJS1Yc-7EbIMXOAuZ2x43lHog-j0dyZSQ9oTSrUCzdwZXoRcY_Wv3BF5vGcflcwKFx5lkfE23I_Yw3MablIWTn_5lVytdIWzQTWVgWnaecK1poi0fkEnbXgL4-LtSbKGwh4P2rBU9h7owbpchouqFXiJF6UzHh44z6I_jdk5h0Jpcwrl8KpE6DGojWu0CD70cekxoNR3RNZgOFkQVNTIXmUOvCdEzrIgRvr0-GMgoWdTPJBvHErHe-BoOBOCdNmkBngL2PIHOVYjOUHWEy7QGP0YNRoTStkpR5GvQM7KSi0ZmwJen4UapJXW51RKmly8gBqIuSDjH7SiHmjjhHL6uI22M-hnoy7iEG4nQxJhphc4KPegh3dSx6jiL53q67rnhnrKG0ZaTtS44ph0mICAfnJWjDlz37zlJ-kL4yHaP4FEtjvE3KEDMdcnjXHnHxUX2XpliT8oyThOyThuxzhr3CEbzlq2bdJyp7_IBv4n2fj-0_Mg4wHGQ8yHmT8J2RsFl9dfJA25L395FzMqN0LD_HVkYltb_RiygdnrAV9wfA6Hdoa14OcHBmHG2hBK8AHAX6i4LkXR8mPNarQXopBjVry7eh8OP-vrm4WzKsu3xTzgpfg0VCONPR2pZx5ujqUy2W5e_L3zE7eu4VuJcCJYZQXrhMC2biDZukdLbNSqjKJdQOfq1B9JNetGXIutQQTUgUxVwWZZ_U8K-n0U99DFaD1Vigvgfv8sHzEqnR-hPoDKuV2K_tY9oS1uO4YqRluYQcwHAtl-T3tUtGKAEq6FV4RWtOWNrSjGDeQlzA5KASYQLbmyoSCYtA0XJqBJkn0jxaBlkYhX2yxUkMDhGeMpL2snIl3suxEyNxvxIN4PkgW6tmid7kytkmeH0aVXx8sid4KeLyCS91HEyHgfBndtfHS4hTjbZJoGhfvz47VadbOsdSlUbrkA1fdX1oqOuu4zhW5sydZ3BTX5_BwAa4CTK-D-NI5cWtGHcviiZCPT3WCuvQ) |

## A searched continuation

The playouts above continue with R34 greedy on both sides. If R34 answers a break worse than a
strong player would, its breaks look better there than they are. So a uniform 400 of the 4,368
plays were played again with a searched continuation: Gumbel k=4, 64 simulations, both sides, 32
pairs. Gain = kept out minus R34's play, points a play:

| group | plays | greedy, 512 pairs | Gumbel 64:4, 32 pairs | searched − greedy |
|:---|---:|---:|---:|---:|
| all | 400 | −1.51 ± 0.41 | −1.34 ± 0.54 | +0.17 ± 0.48 |
| US | 160 | −1.76 ± 0.91 | −1.98 ± 0.97 | −0.23 ± 0.80 |
| USSR | 240 | −1.35 ± 0.32 | −0.91 ± 0.63 | +0.44 ± 0.60 |
| US, turns 1–9, not the China Card | 118 | −0.24 ± 0.41 | −0.97 ± 0.84 | −0.73 ± 0.84 |
| USSR, turns 1–9, not the China Card | 204 | −0.88 ± 0.24 | −0.78 ± 0.65 | +0.10 ± 0.66 |
| the China Card | 51 | −3.62 ± 1.19 | −1.65 ± 1.61 | +1.96 ± 1.13 |
| turn 10 | 38 | −7.11 ± 3.72 | −6.04 ± 3.09 | +1.07 ± 1.99 |
| Italy first | 51 | +0.67 ± 0.92 | +0.15 ± 1.33 | −0.52 ± 1.49 |

Breaking still pays on average, by the same margin. The per-play gains correlate at 0.52 between
the two continuations. The searcher is built on R34, so this is a stronger R34, not a human; the
one test it does pass is that better answers do not turn R34's breaks into losses.

## Cuba

The US's Ops points into Cuba, turns 4–10, at the humans' positions, by where Fidel is. Fidel
removes all US influence from Cuba and gives the USSR control:

| Fidel is | humans | R34 on the same positions | R34 − humans | R34 in its own games |
|:---|---:|---:|---:|---:|
| unseen (deck or USSR hand) | 2.1 ± 0.7 | 9.0 ± 1.8 | **+6.9 ± 1.8** | 6.0 |
| in the discard pile | 4.6 ± 1.0 | 8.4 ± 1.2 | +3.8 ± 1.5 | 7.5 |
| in the US's own hand | 1.8 ± 1.4 | 9.0 ± 3.4 | +7.2 ± 3.4 | 6.5 |
| gone (evented) | 3.5 ± 0.4 | 2.8 ± 0.4 | −0.7 ± 0.4 | 4.5 |

* **The overvaluation is only while Fidel is still to come.** Once Fidel has fired, R34 puts no
  more into Cuba than humans do.
* **Fidel then wipes a third of those plays.** In R34's own games the US makes 2,602 Cuba plays
  with Fidel to come, 0.6 a game. Fidel fires later in 34% of them, removing 3.6 influence on
  average.
* **The cost is nothing measurable.** Paired playouts of 1,264 of those plays, against the same
  card's Ops kept out of Cuba, 512 pairs each, R34 greedy:
  * all: −0.29 ± 0.09 points a play (keeping out is slightly worse);
  * Fidel unseen: +0.09 ± 0.15;
  * Fidel in the discard pile: −0.56 ± 0.12;
  * Fidel in the US's hand: −0.81 ± 0.26.

  Summed over its 0.6 plays a game, the habit is worth −0.2 points a game.
* **It is not a self-play convention either.** The bots fire Fidel in an action round at least
  as often as humans do (headlines not counted):
  * R34's USSR: 48% of the plays sampled in its search bank;
  * the older E7 network: 48% in the event census;
  * humans: 39%.

## Method

Tool `tools/scripts/control_census.py` (`tools/README.md` §7f), branch `exp/control-breaks`
(commits `3e276de`, `fbfde8f`), engine `b670b258`. It has five stages:
* `human`: the 265 corpus games, 119 complete and 146 partial, with each game's untrusted turns
  left out (`corpus_driver.untrusted_turn`). That gives 30,080 Ops influence points and 1,357 break
  plays.
* `selfplay`: 4,096 greedy R34 games, seed 71000, giving 510,574 points and 44,715 break plays.
  * The break bank is every break play of the first 400 games: 4,368 rows, release
    `control-bank-r34-20261012`, sha256 `4147e50a…`.
  * The Cuba bank is every US Cuba play with Fidel to come in the first 2,000 games: 1,264 rows,
    release `cuba-bank-r34-20261012`, sha256 `8fec7760…`.
  * A second run reproduced the break bank row for row.
* `onhuman`: R34's argmax at each human point.
* `playouts`, on the fork's CI (`search_bank.yml -f tool=tools/scripts/control_census.py`):
  * breaks: `38311296749`, 512 pairs, greedy;
  * Cuba: `38312398463`, 512 pairs, greedy;
  * the searched sample: `38315290091`, 400 rows (release `control-bank-sample-r34-20261012`,
    `random.Random(4242)`), `--continue-with 64:4`, 32 pairs.
* `report`: [report.md](E7_control_breaks_R34/report.md), with every table above and the by-country
  costs.

The search-bank figures are the R34 bank of [E7_leak_by_kind_R34.md](E7_leak_by_kind_R34.md), with
each Ops influence point classed the same way.

**What this does not say.**
* **The kept-out line is R34's own best play outside opponent control,** not the best move. The
  reference search finds a much better point at 23% (US) and 11% (USSR) of break points. That is a
  wider question than "break or not", and the [leak](E7_leak_by_kind_R34.md) measures it.
* **"Retaken" compares R34's games with the humans'.** R34's breaks meet R34's answers and the
  humans' breaks meet humans'. The same-position comparison covers only the decision to break.
* **The searched check is R34-based and small:** 400 plays, 32 pairs, 64 simulations.
* **Turn 10 is the final-scoring arithmetic the blunder census already found**
  (`research/log/E7_blunder_census_R34.md` on `exp/blunder-census-r34`). A deterministic
  last-turn placement solver would cover these breaks too.

```bash
O=data/reports/control_census; M=data/checkpoints/hf/R34soup_7120-7200M.pt
PYTHONPATH=.:build/release .venv/bin/python tools/scripts/control_census.py human --out $O/human.json.gz
PYTHONPATH=.:build/release .venv/bin/python tools/scripts/control_census.py selfplay --checkpoint $M \
  --games 4096 --out $O/selfplay.json.gz --bank $O/bank.jsonl.gz --bank-games 400 \
  --cuba-bank $O/cuba_bank.jsonl.gz --cuba-games 2000
PYTHONPATH=.:build/release .venv/bin/python tools/scripts/control_census.py onhuman --human $O/human.json.gz \
  --checkpoint $M --out $O/onhuman.json.gz
# each bank as a release asset named bank.jsonl.gz, then
gh workflow run search_bank.yml --ref exp/control-breaks -f tool=tools/scripts/control_census.py -f stage=playouts \
  -f release=<release> -f model='_models/E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M.pt' -f args="--pairs 512"
PYTHONPATH=.:build/release .venv/bin/python tools/scripts/control_census.py report --human $O/human.json.gz \
  --selfplay R34=$O/selfplay.json.gz --onhuman R34=$O/onhuman.json.gz --bank $O/bank.jsonl.gz $O/cuba_bank.jsonl.gz \
  --playouts <break playouts> <Cuba playouts> --out report.md
```
