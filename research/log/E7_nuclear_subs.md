# Nuclear Subs (2026-10-07): the model never follows Subs with battleground coups, and scripted ones do not make the event worth playing

**Question.** Follows [`E7_event_decision_bank.md`](E7_event_decision_bank.md). There, Subs played
for its event lost 4.5 points for the US against the raw network's own play, but the playouts'
continuation is raw. Subs pays only through what comes after it: battleground coups for the rest of
the turn, above all at DEFCON 2, where they would otherwise end the game. So:

1. Does raw make those coups once Subs is in effect?
2. Is the event worth playing when they are made?

Humans play Subs for its event 28% of the time; raw, 0%.

**Answer.**

* **Raw does not follow up.** At DEFCON 2, after the Subs event, raw makes 0.42 coups per pair for
  the rest of the turn and 0.01 in battlegrounds. That is the same as without the event (0.45 and
  0.01). It plays as if Subs were not in effect, although the observation carries the flag.
* **Scripted battleground coups do not rescue the event.** A battleground coup was available in 92%
  of pairs. Forcing one at the next US action round (`next`) makes 0.92 battleground coups per pair
  at DEFCON 2, and forcing one every remaining US action round (`all`) makes 2.25. The event is
  still worse than raw's play:
  * the event alone: -4.5 ± 0.6 points
  * the event and the next coup: -4.9 ± 1.5
  * the event and a coup every round: -6.9 ± 2.0
* **The best branch loses too.** It is chosen on half the pairs and measured on the other half, and
  comes out at -1.1 ± 0.4 points against raw at DEFCON 2.
* **No position is a clear exception.** Only 4 of 93 positions have any branch more than 2 SE above
  raw, fewer than the ~6 that noise alone would give.

**What it implies.** Within what these playouts can test, the 28% human event rate looks like
overvaluation. A turn's Ops spent on Subs, plus the Ops of the card that makes the coup, buys less
than raw gets from playing those Ops normally. Raw's 0% is right here. Not following up costs it
nothing, because following up does not pay. A follow-up better than the script could still be
worth more:
* a card chosen for the coup
* a target chosen for its value, not raw's most probable battleground
* coups timed around scoring

Those would need search to test. Nothing here points to Subs as a training target.

## Method

* **Positions.** Every US action-round play-mode decision for Nuclear Subs with the event legal in
  the search bank's annotated self-play: 1,500 games at temperature 0.1, one decision in eight
  kept. That gives 93 positions, 74 of them at DEFCON 2, about 0.5 a game.
* **Branches.** Four per pair, with the same redeal of the US's unseen cards and the same dice;
  256 pairs:
  * `alt`: raw's most probable non-event play.
  * `event`: Subs for its event, then raw plays on.
  * `next`: the event, then a battleground coup at the first later US play-mode decision of the
    turn where one is possible. Raw chooses the card; the target is raw's most probable
    battleground.
  * `all`: the same at every such decision left in the turn.

  Raw plays both sides throughout, under `play_safe`'s guard against suiciding on the spot.
  Implemented in `ai/eval/subs_followup.py`.
* **Coups.** The US's, from the move to the end of the turn: at its coup-target decision.
  `alt`'s counts include a coup it makes itself.

Full tables, including every position: [`E7_nuclear_subs/report.md`](E7_nuclear_subs/report.md).

## Replicate

```bash
# exp/search-bank 6568470; playouts on CI (run 37724351904, 10 runners)
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py subs-select \
  --annotated data/reports/search_bank/annotate/annotate.jsonl.gz --out subs_bank.jsonl.gz
gh release create subs-bank-20261007 bank.jsonl.gz --prerelease   # subs_bank.jsonl.gz renamed
gh workflow run search_bank.yml --ref exp/search-bank -f stage=subs \
  -f release=subs-bank-20261007 -f args="--pairs 256" -f runners=10
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py subs-report \
  --bank subs_bank.jsonl.gz --subs subs.jsonl.gz --games 1500 --out report.md
```
