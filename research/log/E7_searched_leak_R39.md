# Where the searched player still loses (2026-10-14): R39 under the rollout root, judged by full-game playouts

**Question (owner).** The player we deploy is R39 under the tuned rollout root, `4:32:4:z2`
([tuning](https://github.com/JamesYouL2/DeepStruggle/blob/20968ddc9dc7e170cd799f64cc53e85f846797fa/research/log/E7_rollout_root_tuning_R39.md)).
Where does *that* player still lose, and how much, split by kind of decision? The owner expected
few remaining issues.

The network is R39's last SWA, `E7-A8-R1-S44@6400M+R39@7120..7200M`. On HF it is
`E7-A8-R1-S44@6400M+R39_20261011_145632/swa_7120-7200M.pt`, sha256 `19d3008d…`.

**Answer.**

* **The searched player still loses at least +86.6 ± 8.4 points a game.** This is a lower bound.
  It is found at 12% of its decisions; the other 88% count as no loss.
* **There are few kinds of loss but no few positions.** Two kinds make up 72% of the total, but
  each is spread thin over many positions.

  | decision kind | leak | share | leaks the most in |
  |:---|---:|---:|---:|
  | action-round card choice | **+36.6 ± 4.8** | 42% | 93% of resamples |
  | influence placement | **+25.7 ± 5.8** | 30% | 7% |
  | play mode (event / Ops / space) | +10.7 ± 2.7 | 12% | |
  | headline | +8.7 ± 1.1 | 10% | |
  | Ops mode | +2.6 ± 0.9 | 3% | |
  | coup target | +1.1 ± 1.2 | 1% | |
  | realignment target | +0.9 ± 0.4 | 1% | |
  | branch choice | +0.3 ± 0.2 | 0% | |
  | **all** | **+86.6 ± 8.4** | | |

* **Card choice is the largest leak, and no single card dominates it.**
  * The China Card played at the right moment is the largest single item, at +4.7.
  * Next comes late-game scoring-card timing. For example, at T10 the player plays Arab-Israeli
    War instead of Central America Scoring, or Five Year Plan instead of Africa Scoring.
  * After that the items fall to +1–2 each.
  * Both sides leak alike (US +20.3, USSR +16.3).
  * Most of it comes at DEFCON 2 (+26.9 of +36.6).
* **It is a ranking problem, not a recall problem.**
  * The better move is in the network's top 4 for 73% of the leak, and at rank 5–6 for the other 27%.
  * Gumbel@256 found nothing the top 6 missed (+0.1).
  * In two thirds of the leak (+58.3) the search kept the network's move when a better one was
    among the network's own candidates.
  * In another +9.3 the search departed from a network move that was in fact best.
* **The search already recovers a lot.** In the player's own games, its moves beat the network's
  by **+47.9 ± 4.9** a game, card choice +27.0 of that. Card choice is where the search gains
  most and where most of the loss remains.
* **The earlier census was blind to card choice.**
  * The [R39 raw census](https://github.com/JamesYouL2/DeepStruggle/blob/bb9fa427c16058a464f6741dddb9a704a5e219a1/research/log/E7_leak_by_kind_R39.md)
    took its best move from Gumbel search. It put R39's card-choice leak at +1.6 ± 7.2.
  * The searched player alone gains +27 over the network on card choice and still leaks +36.6.
  * So that census's judge could not see card-choice errors. Its total (+18.5 for R39 raw) is
    not comparable with the +86.6 here.

## Why the judge is playouts and not a search

The census so far measured a player against a reference search's best move. Neither available
reference judges a rollout-root player well:

* **Gumbel@256** scored −19 against the rollout root on the large-regret rows of the
  [tuning grid](https://github.com/JamesYouL2/DeepStruggle/blob/20968ddc9dc7e170cd799f64cc53e85f846797fa/research/log/E7_rollout_root_tuning_R39.md).
  Its "best" moves would often score *below* the player's, and those negative gains would cancel
  real leak.
* **A stronger rollout root** shares the player's network, candidates and leaf critic, so it
  shares its blind spots.

So the judge here is the full-game paired playout itself, which is what the census already
trusted for measuring a gain. Every candidate at every bank position is played out, and the best
is taken from the playouts. Two stages keep the selection honest:

1. **Screen.**
   * Each of the 8,000 bank positions plays out up to 8 moves, 38,778 in all, at 512 paired
     continuations each.
   * The moves are the player's, the network's 6 most probable, and Gumbel k=8 @256's.
   * The move with the best mean is the candidate.
2. **Confirmation.**
   * A position escalates when its candidate's paired lead over the player's move is at least
     1 point and at least 1.5 SE.
   * The two moves are then played out again on 2,048 fresh pairs (another seed).
   * The leak is read only from these fresh pairs, so choosing the best of eight noisy scores
     does not inflate it.
   * 1,122 of the 8,000 positions escalated.

Positions that did not escalate count as zero, which is why the total is a lower bound.

**Units, as in the earlier entries.** Points of win probability a game, both sides' decisions
summed. This is what the player would gain, in its own self-play games, by playing the confirmed
best move wherever one was found. There are 450.5 decisions with two or more legal moves a game.
Each ± in the kind table is a stratified bootstrap, as in the R39 raw entry: 2,000 resamples of
positions within their bank stratum. The smaller tables below print playout noise alone.

## Inside card choice

| | leak |
|:---|---:|
| US | +20.3 |
| USSR | +16.3 |
| T1–3 | +9.9 |
| T4–7 | +14.4 |
| T8–10 | +12.3 |
| DEFCON 2 | +26.9 |
| DEFCON 3 | +5.1 |
| DEFCON 4–5 | +4.6 |

The card the best move plays, and the card the player plays instead (positions with n ≥ 4):

| best plays | leak | | player plays instead | leak |
|:---|---:|:-:|:---|---:|
| The China Card | +4.7 | | Shuttle Diplomacy | +2.2 |
| Central America Scoring | +2.3 | | Alliance for Progress | +2.0 |
| Ussuri River Skirmish | +2.1 | | Europe Scoring | +1.8 |
| Che | +1.7 | | Arab-Israeli War | +1.6 |
| Romanian Abdication | +1.7 | | Solidarity | +1.4 |
| Duck and Cover | +1.6 | | Five Year Plan | +1.2 |
| Missile Envy | +1.5 | | Sadat Expels Soviets | +1.1 |
| Africa Scoring | +1.4 | | Special Relationship | +1.0 |

The largest categories are swaps between cards of the same kind and Ops value: own 3 → own 3
(+1.9) and opponent 2 → opponent 2 (+1.8). Even the costliest single position adds only
+2.2 a game: T7 AR3, USSR at DEFCON 2, playing Shuttle Diplomacy where the China Card is +18.9
points better.

## Inside influence placement

* Most of it is the wrong country within the right region: Asia +4.1, Africa +3.5, Europe +3.3,
  Middle East +1.2.
* By era it is late: T8–10 +11.5, against T1–3 +7.5 and T4–7 +6.8.
* The search kept the network's placement in 22.1 of the 25.7 points.
* The better point is in the network's top 4 for 17.7 of the 25.7, and at rank 5–6 for 9.9.

## The rest

* **Play mode** +10.7. USSR +6.2, US +4.5. It is spread across cards, the largest being
  "One Small Step…" at +1.2.
* **Headline** +8.7. US +4.7, USSR +3.9. Each card swap is worth ≤ 0.5.
* **Ops mode, coups, realignments and branch choices** together come to +4.9.

The full tables, including the 20 costliest positions, are in
[`E7_searched_leak_R39/leak.md`](E7_searched_leak_R39/leak.md). In those tables `raw` means the
player's move, and "k=8 @256 over raw" is negative throughout (−67 in total) because the player
beats Gumbel@256.

**Where the best move ranks, and what the search did there** (confirmation pairs, playout noise only):

| | all | card choice | influence |
|:---|---:|:---:|---:|
| best in the network's top 4 | +63.4 | +26.2 | +17.7 |
| best at the network's rank 5–6 | +23.2 | +10.4 | +9.9 |
| best found only by Gumbel@256 | +0.1 | | +0.1 |
| the search kept the network's move | +58.3 | +22.3 | +22.1 |
| the search departed, a third move best | +19.0 | +11.7 | +3.6 |
| the search departed, the network's move best | +9.3 | +2.6 | +1.9 |

**The search's gain over the network** (screen pairs, every bank position where the two differ,
12.9% of decisions):

| kind | the player over the network |
|:---|---:|
| action-round card choice | +27.0 ± 2.9 |
| influence placement | +9.8 ± 3.8 |
| play mode | +6.3 ± 1.1 |
| headline | +3.3 ± 0.4 |
| Ops mode | +1.5 ± 0.2 |
| **all** | **+47.9 ± 4.9** |

## What this does not say

* **It is a lower bound.** Unescalated positions count as zero. Some hold real losses smaller
  than 1 point or 1.5 SE on the screen.
* **The continuation is the network's own greedy play**, on both sides. A move whose value lies
  in a follow-up the network does not find reads worse than it is. Holding a scoring card or the
  China Card for later can be that kind of move.
* **The candidates are the player's move, the network's top 6 and Gumbel@256's.** A better move
  outside all three is not counted.
* **The leak is a sum of per-decision gains.** It is not the strength a perfect fix would add,
  because the decisions interact.
* **The positions are the player's own self-play games.** An opponent who plays differently meets
  other positions.

## Reproduction

**Code.** Branch `exp/searched-leak-r39` of the fork, commit
[`bb9fa42`](https://github.com/JamesYouL2/DeepStruggle/tree/bb9fa427c16058a464f6741dddb9a704a5e219a1).
The tool is
[`tools/search_bank.py`](https://github.com/JamesYouL2/DeepStruggle/blob/bb9fa427c16058a464f6741dddb9a704a5e219a1/tools/search_bank.py),
whose docstring describes the stages, and the CI workflow is
[`search_bank.yml`](https://github.com/JamesYouL2/DeepStruggle/blob/bb9fa427c16058a464f6741dddb9a704a5e219a1/.github/workflows/search_bank.yml).
Two commits on top of `exp/leak-r39` added what this needed:

* `e9a025c`: `annotate --player SPEC`. The given player plays itself, a row's `raw` is the
  player's move, and the network's argmax is kept in `net`.
* `bb9fa42`: `screen-select`, `confirm-select`, and `leak --screen … --large 0`.

The annotate run used `e9a025c`. Its stage is unchanged at `bb9fa42`, which ran everything after it.

**Data.** Every intermediate is in the release
[`searched-leak-r39-data-20261014`](https://github.com/JamesYouL2/DeepStruggle/releases/tag/searched-leak-r39-data-20261014),
with a `SHA256SUMS` file. The screen and confirmation inputs are the releases
`searched-leak-screen-r39-20261014` and `searched-leak-confirm-r39-20261014`.

| step | where | output | sha256 |
|:---|:---|:---|:---|
| 1. annotate | CI run `38557924000` | 84,470 positions from 1,500 games, player ≠ network at 12.9% | `27a908df…` |
| 2. select | local | `bank.jsonl.gz`, 8,000 positions in 72 strata | `10a16d23…` |
| 3. screen-select | local | 38,778 moves | `d54ab7b8…` |
| 4. playouts (screen) | CI run `38568207462` | 512 pairs, seed 0 | `b6fa05a8…` |
| 5. confirm-select | local | 1,122 of 8,000 escalated | `9ea56d26…` |
| 6. playouts (confirmation) | CI run `38596564880` | 2,048 pairs, seed 1 | `14f49d38…` |
| 7. leak | local | `leak.md` | |

The commands, from a checkout of `bb9fa42` with the engine built. Each local step is
deterministic. `M` is the HF path above; `$D` is a data directory.

```bash
tools/scripts/check_engine_fresh.sh
M="E7-A8-R1-S44@6400M+R39_20261011_145632/swa_7120-7200M.pt"
R=JamesYouL2/DeepStruggle
P="env PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py"

# 1. annotate (CI, 20 runners, ~80 min); the artifact holds annotate.jsonl.gz
gh workflow run search_bank.yml -R $R --ref exp/searched-leak-r39 -f stage=annotate \
  -f args="--games 1500 --player rollout:4:32:4:z2" -f model="$M" -f runners=20

# 2-3. select and screen-select (local)
$P select --annotated $D/annotate/annotate.jsonl.gz --size 8000 --control 2000 --seed 0 --out $D/bank.jsonl.gz
$P screen-select --bank $D/bank.jsonl.gz --out $D/screen_input/bank.jsonl.gz

# 4. screen playouts (CI); the input is published as a release first
gh release create -R $R searched-leak-screen-r39-20261014 $D/screen_input/bank.jsonl.gz --prerelease
gh workflow run search_bank.yml -R $R --ref exp/searched-leak-r39 -f stage=playouts \
  -f args="--pairs 512 --seed 0" -f release=searched-leak-screen-r39-20261014 -f model="$M" -f runners=20

# 5. confirm-select (local)
$P confirm-select --screen-input $D/screen_input/bank.jsonl.gz --playouts $D/screen/playouts.jsonl.gz \
  --z 1.5 --floor 0.01 --out $D/confirm_input/bank.jsonl.gz

# 6. confirmation playouts (CI)
gh release create -R $R searched-leak-confirm-r39-20261014 $D/confirm_input/bank.jsonl.gz --prerelease
gh workflow run search_bank.yml -R $R --ref exp/searched-leak-r39 -f stage=playouts \
  -f args="--pairs 2048 --seed 1" -f release=searched-leak-confirm-r39-20261014 -f model="$M" -f runners=20

# 7. leak (local); 450.5 = 84,470 annotated positions x 8 / 1,500 games
$P leak --bank $D/bank.jsonl.gz --playouts-input $D/confirm_input/bank.jsonl.gz \
  --playouts $D/confirm/playouts.jsonl.gz --screen $D/screen/playouts.jsonl.gz \
  --large 0 --decisions-per-game 450.5 --out $D/leak.md
```

To skip the CI steps, download the release and put its files where the commands expect them:

* `annotate.jsonl.gz` → `$D/annotate/`
* `screen_playouts.jsonl.gz` → `$D/screen/playouts.jsonl.gz`
* `confirm_playouts.jsonl.gz` → `$D/confirm/playouts.jsonl.gz`

Steps 2, 3, 5 and 7 then reproduce the hashes above. Steps 3 and 5 also rebuild the screen and
confirmation inputs that `splits.py` reads.

**The rank, departure and search-gain tables** are not part of the `leak` stage. They come from
`splits.py` in the data release, run in `$D`:

```python
import collections, gzip, json, math

DPG = 450.5  # decisions a game: annotate rows x 8 / 1500


def load(path):
    return [json.loads(line) for line in gzip.open(path, "rt")]


def kind(r):
    if r["decision_type"] == "SELECT_CARD":
        return "headline" if r["phase"] == "HEADLINE" else "AR card"
    return {"SELECT_PLAY_MODE": "play mode", "SELECT_OP_MODE": "op mode",
            "POINT_NODE": "target", "CHOOSE_BRANCH": "branch"}[r["decision_type"]]


def table(title, rows, scores, pair, keys):
    """Population-weighted paired gain a game (playout noise only), per key."""
    w = sum(r["weight"] for r in rows)
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        moves = pair(r)
        if moves is None:
            continue
        a, b = (scores[r["id"]][str(m)] for m in moves)
        d = [x - y for x, y in zip(a, b)]
        m = sum(d) / len(d)
        v = sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)
        c = 100 * DPG * r["weight"] / w
        for k in keys(r):
            acc[k][0] += c * m
            acc[k][1] += c * c * v
    print(f"\n{title}")
    for k, (m, v) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:32s} {m:+6.1f} ± {math.sqrt(v):.1f}")


# 1. The search's gain over the network, in the player's own games (screen pairs, every bank row).
screen = load("screen_input/bank.jsonl.gz")
sc = {r["id"]: r["scores"] for r in load("screen/playouts.jsonl.gz")}
table("player over net (screen)", screen, sc,
      lambda r: None if r["raw"] == r["net"] else (r["raw"], r["net"]),
      lambda r: ("all", kind(r), f"{kind(r)} {r['side']}"))

# 2. The confirmed leak by where the best move ranks under the network, and by whether the player
#    had departed from the network there (confirmation pairs, escalated rows; the rest count zero).
conf = load("confirm_input/bank.jsonl.gz")
cc = {r["id"]: r["scores"] for r in load("confirm/playouts.jsonl.gz")}


def rank(r):
    order = sorted(range(len(r["legal"])), key=lambda j: -r["logits"][j])
    k = [r["legal"][j] for j in order].index(r["best"]) + 1
    return "best in net top 4" if k <= 4 else "best net rank 5-6" if k <= 6 else "best = g256 only"


def dep(r):
    if r["raw"] == r["net"]:
        return "player kept net's move"
    return "departed, net's move best" if r["best"] == r["net"] else "departed, a third move best"


table("leak by candidate rank / departure (confirmation)", conf, cc,
      lambda r: (r["best"], r["raw"]) if r["escalated"] else None,
      lambda r: (f"all | {rank(r)}", f"all | {dep(r)}", f"{kind(r)} | {rank(r)}", f"{kind(r)} | {dep(r)}"))
```
