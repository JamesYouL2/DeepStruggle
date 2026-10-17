# The dataset ledger — labelled position sets, and the labelling recipes (D codes)

One row per dataset instance: the generator model, the labelling recipe, size, rows spent, hashes and
where the files live. A dataset is named as a model under a labelling recipe, the way a player is a
model under a search ([`method/run_nomenclature.md`](method/run_nomenclature.md)):
`<generator model>~D<n>(seed=…,games=…)`. The files are under the git-ignored `data/` tree and are
never committed; the ledger carries their hashes.

## Labelling recipes

A `D` code is a flag set of the labelling tool, like an R code of the trainer. Until
`tools/scripts/run_codes.py` reads the labelling tool's metadata, the codes are defined here.

| code | what it is | flags (intended) | datasets |
|:---|:---|:---|:---|
| **D1** | P34 generation-1 labels: greedy self-play positions at temperature 0.25, one decision in eight, 5% of games held out; 32 worlds from the mover's view, fresh dice each, both sides greedy to the game's end under the safe-playout rule; columns: the mover's score, VP at the end, VP and DEFCON at the turn's end, each region's swing at its next scoring; sibling slice: 50,000 positions x the top 4 candidates x 16 shared worlds, each row's leaf the mover's next decision | `tools/value_labels.py` defaults (`labellings.D1` in [`run_codes.json`](run_codes.json)) | A13+R48 below |
| **D2** | D1's sibling slice relabelled keyed (P34 revision 2): the same 50,000 positions x the top 4 candidates x 16 worlds under keyed chance, each row's leaf the rollout root's own (stop=earlier, leaf=own); no main labels | `label --no-main --keyed --leaf-horizon 4` (`labellings.D2` in [`run_codes.json`](run_codes.json)) | A13+R48 below |

## Datasets

| dataset | positions | rows spent | device | files (sha256) | used by |
|:---|---:|---:|:---|:---|:---|
| `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M~D1(seed=44,games=20000)` | 1,115,626 (1,059,433 train / 56,193 held out; 2,794,624 sibling rows) | 9,444,677,177 (2 h 22 min, RTX 4090) | cuda, engine 040b1617 | `data/datasets/p34/A13R48-D1-s44-g20000/`: train.npz bca65c978a61, holdout.npz 954a7e2b622d, siblings.npz ae9350e568ca | |
| `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M~D2(seed=44,games=20000)` | 43,666 decisions (sibling slice only; 2,794,624 rows) | 677,160,028 (11.5 min) | cuda, engine c436ce3c (positions 040b1617, one epoch) | `data/datasets/p34/A13R48-D2-s44-g20000/`: siblings.npz b631ffe64a75 (positions linked from D1) | |
