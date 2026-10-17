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
| **D1** | P34 generation-1 labels: greedy self-play positions at temperature 0.25, one decision in eight; 32 worlds from the mover's view, both sides greedy to the game's end under the safe-playout rule; columns: the mover's score, VP margin at the end, VP and DEFCON at the turn's end, each region's VP swing at its next scoring | `--temperature 0.25 --keep 1/8 --worlds 32 --stop game --labels score,vp,turn_vp,turn_defcon,region_swing` | (none yet) |

## Datasets

| dataset | positions | rows spent | device | files (sha256) | used by |
|:---|---:|---:|:---|:---|:---|
| (none yet) | | | | | |
