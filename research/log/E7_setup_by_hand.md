# Turn-1 setups by hand, on R32 and the R34 soup (2026-10-11): a fixed setup loses nothing measurable to the newest network

**Question (owner).** Follow-up to [the setup matrix](E7_setup_matrix.md): does the right setup
depend on the deal? Measure it on the existing R32 games, and on the newest network as well.

**Answer.**

* **R32 setup conclusions came from how R32 plays the rest of the game.** On R32's games a hand
  rule does pay:
  * Against the networks' USSR setup, the US should play France 6 instead of West Germany 3 when
    it holds Blockade or De Gaulle Leads France, and West Germany 3 otherwise.
  * Scored out of fold, that rule gains +0.32 ± 0.13 over always West Germany 3.
  * Under the R34 soup the same card effects vanish (Blockade −0.1 ± 0.5, De Gaulle −0.3 ± 0.5,
    against −2.6 and −3.7 under R32), as does the USSR's Independent Reds effect (+0.2 against
    +1.6).

  They measured how R32 plays the cards that follow, not the setup itself. The pricing run had
  already flagged R32's Blockade handling.
* **R34 plays its own setups, not the ones the matrix called "the networks'".** Over 300 fresh
  deals, greedy:
  * The USSR plays **Yugoslavia 3, Poland 3** on every deal.
  * The US plays **West Germany 4, France 2, Italy 2, Iran 1** on 70%, West Germany 4, France 3,
    Italy 2 on 15%, and the human West Germany 3, France 3, Italy 2, Iran 1 on 14%.

  R32 and the heads soup play France 6 and Poland 3, Hungary 2, Austria 1 on every deal. The
  setups `us_r34` and `ussr_r34` were added to `tools/lib/openings.py`.
* **Under R34's play its own setups are best or tied** (3 × 3, 900,000 games over 100,000 deals):
  * The USSR's Yugoslavia 3, Poland 3 beats the human East Germany 4, Poland 4, Austria 1 by +0.72 ±
    0.13. It beats R32's setup by +1.77 ± 0.13.
  * The US's West Germany 4 setup and the human West Germany 3 setup are a tie (+0.08 ± 0.09).
    Both beat France 6 by ~2.2.
  * The solved 3 × 3 is worth 50.7% [50.4, 51.0] to the US, against 50.5% for R34's own pairing.
* **For R34 the hand matters but does not pay.** A joint test rejects "no card moves the contrast"
  for the USSR (p ≈ 1e-6 to 1e-8) and for most US contrasts. But each card moves a contrast by 2
  points at most, on the ~21% of deals that hold it. Every hand rule, scored out of fold, is
  within noise of R34's best fixed setup; the best is +0.03 ± 0.05. The USSR's rule never leaves
  Yugoslavia 3, Poland 3.

**Caveat.** Every value is to the network's own play after the setup. R34 trained from its own
setups and never from the human ones, so this comparison favours its setups. It does not show
that Yugoslavia 3, Poland 3 is a good human setup. A searched continuation (Next) is the test.

## Design

`tools/setup_matrix.py hands` (commits 241acd1, 7d9084e) reads a matrix's per-game log:

* **Hands.** It rebuilds each deal seed's turn-1 hands the way the tournament deals them
  (`reset_game(i, seed)`; the deal comes before the setup).
* **Contrasts.** The USSR sets up first and sees only its own hand (`rules.json`, `setup_order`).
  The US sets up second and sees its hand and the USSR's setup. So:
  * the US contrast is each US setup minus the reference against each USSR setup, conditioned on
    the US hand;
  * the USSR's is averaged over the US setups, conditioned on the USSR hand.
* **Joint test.** A robust Wald test fits each contrast on the cards the chooser holds. Every hand
  holds exactly 8 cards, so the columns are collinear; the least-squares cut-off drops that one
  direction.
* **Card tables.** For each card: the contrast when the chooser holds it minus when it does not.
  The opponent's hand is shown beside it, to explain an effect only. The Bonferroni bar is
  |z| ≥ 3.4 over 38 cards.
* **Hand rule.** The rule is cross-fitted: fitted on half the deals and scored on the other half,
  against the best fixed setup of the fitting half. **All cards** fits every card. **Sparse** fits
  only the cards past the bar on the fitting half, so selection is part of the fit and the other
  half scores it honestly.

## Results

### R32 (2 × 2, 38121015201): [report](E7_setup_matrix/hands_r32_2x2.md)

| choice | rule | cards | rule − best fixed |
|:---|:---|:---|---:|
| US vs networks' USSR setup | sparse | Blockade, De Gaulle (both halves) | +0.32 ± 0.13 |
| US vs Pol4/EG+1/Aus1 | sparse | Blockade, De Gaulle / none | +0.13 ± 0.09 |
| USSR | sparse | none / Independent Reds | −0.08 ± 0.09 |

### R34 soup, R32's and the human setups (2 × 2, 38160045724): [matrix](E7_setup_matrix/matrix_r34_2x2.md), [hands](E7_setup_matrix/hands_r34_2x2.md)

R34's play widens both human setups' leads over R32's setups to US +2.43 ± 0.15 and USSR +0.97 ±
0.14, with no interaction. No card passes the bar for either side, and no rule gains.

### R34 soup, with its own setups (3 × 3, 38161115924): [matrix](E7_setup_matrix/matrix_r34_3x3.md), [hands](E7_setup_matrix/hands_r34_3x3.md)

US score, %, each cell ± 0.16:

| US \ USSR | Yug3/Pol3 (R34) | Pol4/EG+1/Aus1 (human) | Pol3/Hun2/Aus1 (R32) |
|:---|---:|---:|---:|
| WG4/Fr2/It2/Ir1 (R34) | 50.46 | 51.38 | 52.58 |
| WG3/Fr3/It2/Ir1 (human) | 50.69 | 51.51 | 52.48 |
| France 6 (R32) | 48.64 | 49.07 | 50.05 |

The R32 setups' cell reproduces the 2 × 2's to the hundredth (50.05): the same deals give the
same games.

The largest own-hand effects, none of which a rule could use out of fold:
* **USSR holding Europe Scoring:** its own setup gains another ~1.5 over both alternatives
  (z −4.3, −4.7).
* **US holding Defectors:** France 6 is worse by a further 2.1 (z −4.1).

## Next

* **Searched continuation on the R34 3 × 3's top cells.** Does Yugoslavia 3, Poland 3 still beat
  the human USSR setup when neither side is limited to greedy play? This is the one result here
  that an expert would doubt.
* **R34's own hand-dependence.** Its US setup varies by deal (70/15/14). The contrast between its
  two main US setups is ~0, so that switching neither costs nor gains measurably.

## Replicate

```bash
gh workflow run setup_matrix.yml --ref exp/setup-matrix \
  -f model="_models/E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M.pt" \
  -f us_setups="us_r34 us_wg3_fr3_it2 us_e7" -f ussr_setups="ussr_r34 ussr_pol4_aus ussr_e7" \
  -f games_per_cell=100000 -f shard_pairs=250
python tools/setup_matrix.py hands --plan plan.json --games games.jsonl.gz --out hands.md
```
