# Search headroom with a Gumbel root (2026-10-05): k = 8 candidates at 256 simulations is the strongest, +79 Elo over the network

**Question.** How much play strength does honest (determinized) search add on top of the best
network, and which root makes the most of a budget?

**Answer.** A noise-free Gumbel root (Danihelka et al., ICLR 2022: the k most probable moves,
narrowed by sequential halving) with first-play urgency 0.2 scores 58-61% against the plain network, and
beats PUCT search of the same network given 8x its simulations. Gains flatten after 64 simulations:
with k = 4 the budget stops helping at 64, and k = 8 is needed to use 128-256; 16 candidates are no
better than 8.

| player (honest search, every decision) | Elo | vs the network | vs k=4 @64 |
|:---|---:|---:|---:|
| **Gumbel k=8 @256** | **1520.5** | **60.1%** | 51.8% |
| Gumbel k=16 @256 | 1518.5 | 61.2% | 50.8% |
| Gumbel k=8 @128 | 1511.7 | 59.8% | 50.8% |
| Gumbel k=4 @128 | 1504.2 | 58.0% | 50.0% |
| Gumbel k=4 @64 | 1503.7 | 58.1% | — |
| the network, greedy | 1441.4 | — | 41.9% |

60,000 games, 2,000 per side per pair, sampling temperature 0.1; a head-to-head is ±0.8. Win
rates count wins only (draws about 1%).

Against PUCT search of the same network (a separate 24,000-game run, same settings): Gumbel k=4
at **16** simulations beats PUCT at 128 (52.1%), and k=4 at 64 beats it 54.5%. PUCT at 128 scores
52.4% against the plain network, the Gumbel root at 64 59.2%.

## Method

* **Network.** `C2_soup+A.pt`, the fork's strongest: the shallow soup
  ([`E7_shallow_soup.md`](E7_shallow_soup.md)) averaged 50/50 with E7-20-44's weights averaged over
  2,720-2,800M. sha256 `5cbb77ef98682b01a550213c0f80646a8c7b55ae4a44a3e49d3ab125744da69b`, fork
  release [`e7-20-44-soups`](https://github.com/JamesYouL2/DeepStruggle/releases/tag/e7-20-44-soups).
* **Search.** Honest: each phase samples one world from the decider's side. At every decision the
  root takes the k most probable moves (Gumbel scale 0), splits the budget over ceil(log2 k)
  halving phases, searches each surviving candidate's position with the ordinary batched search
  (equal shares), and keeps the better half by logit + sigma(completed Q), with sigma and the
  completed Q as in mctx (c_visit 50, c_scale 0.1). Unlike Gumbel MuZero each phase searches a
  candidate afresh rather than growing one tree. First-play urgency 0.2 values an unvisited move at
  its node's value less 0.2 in those searches. Code: `ai/search/gumbel_root.py`;
  `BatchedMCTSConfig.gumbel_k`, `gumbel_scale`, `fpu_reduction` (both trees).
* **Cost.** Gumbel k=4 @32 ran about 8% slower than PUCT @32. The 256-simulation entrants are
  roughly 4x k=4 @64.

## Replicate

The `gumbel:<checkpoint>[:sims[:k]]` agent spec plays this configuration (defaults 256, 8):

```bash
gh release download e7-20-44-soups -R JamesYouL2/DeepStruggle -p 'C2_soup+A.pt' -D data/checkpoints
C=data/checkpoints/C2_soup+A.pt
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/tournament.py \
  --models $C gumbel:$C:64:4 gumbel:$C:128:4 gumbel:$C:128:8 gumbel:$C:256:8 gumbel:$C:256:16 \
  --games-per-side 2000 --temperature 0.1 --output-report gumbel_headroom.md
```

The fork measured it on CI (runs `37327142506` and `37318435430`, 20 runners) with the equivalent
`search:<ckpt>:<sims>:determinize:all` entrants and the three settings as overrides.

## What this does not say

* One network. The plain network's own tournament strength is the reference; nothing here was
  checked against older ladders.
* Search as a training target is a separate question: on the fork, fine-tunes toward the visit
  counts or toward this root's choice both lost to their unsearched control.
