"""Where the rollout root's time goes, stage by stage.

The root's loop per ply: the C++ batch writes every live row's observation and mask (`fill_live`), the
rows go to the device, the network runs, the logits come back, and the batch steps every row (`step`);
at the end the leaves are filled and read by the critic. This times each stage on the network's own
entry decisions, synchronising the device around each so a stage's cost is its own, and reports rows,
bytes and time per stage -- the split the speed plan (the labeller's codes, continuous batching, a CUDA
graph; fewer rows by sequential halving) needs before any of it is built.

    PYTHONPATH=.:build/release python -m ai.eval.search_profile --checkpoint swa.pt --worlds 16 128
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

import ts_engine as ts
from ai.eval.rollout_consistency import collect
from ai.search.rollout_root import RolloutConfig, RolloutRoot, _stop_kw
from tools.lib.player_agent import NeuralAgent


def profile(root: RolloutRoot, states: List[ts.GameState], chunk: int) -> Dict[str, float]:
    t: Dict[str, float] = defaultdict(float)
    dev = root.device
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    model = root.model
    for lo in range(0, len(states), chunk):
        part = states[lo:lo + chunk]
        t0 = time.perf_counter()
        lg, masks, _ = root._forward(part)
        lg = np.ascontiguousarray(lg, dtype=np.float32)
        cands = [ts.rollout_candidates(lg[i], masks[i], root.cfg.k) for i in range(len(part))]
        batch = ts.RolloutBatch(horizon=root.cfg.horizon, shared_dice=True, guard=True, features=root.features,
                                seed=root._rng.getrandbits(64), **_stop_kw(root.cfg))
        for i, st in enumerate(part):
            if len(cands[i]) >= 2:
                batch.add(st, cands[i], root.cfg.worlds)
        t["root and add"] += time.perf_counter() - t0
        while not batch.all_done():
            a = time.perf_counter()
            obs, m = batch.fill_live()
            b = time.perf_counter()
            obs_t = torch.from_numpy(np.ascontiguousarray(obs)).to(dev)
            m_t = torch.from_numpy(np.ascontiguousarray(m)).to(dev)
            sync()
            c = time.perf_counter()
            with torch.no_grad():
                out = model(obs_t, m_t)[0]
            sync()
            d = time.perf_counter()
            lg_np = np.ascontiguousarray(out.float().cpu().numpy(), dtype=np.float32)
            e = time.perf_counter()
            batch.step(lg_np)
            f = time.perf_counter()
            t["fill_live (C++)"] += b - a
            t["upload"] += c - b
            t["forward"] += d - c
            t["download"] += e - d
            t["step (C++)"] += f - e
            t["rows"] += obs.shape[0]
            t["plies"] += 1
            t["bytes up"] += obs.nbytes + m.nbytes
        a = time.perf_counter()
        lobs, lm = batch.fill_leaves()
        if lobs.shape[0]:
            with torch.no_grad():
                v = model(torch.from_numpy(np.ascontiguousarray(lobs)).to(dev), torch.from_numpy(np.ascontiguousarray(lm)).to(dev))[1]
            batch.set_leaf_values(np.ascontiguousarray(v.float().reshape(-1).cpu().numpy(), dtype=np.float32))
        sync()
        t["leaves"] += time.perf_counter() - a
        t["leaf rows"] += lobs.shape[0]
        t["decisions"] += len(part)
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--worlds", type=int, nargs="+", default=[16, 128])
    ap.add_argument("--positions", type=int, default=512)
    a = ap.parse_args()
    model = NeuralAgent.from_checkpoint(a.checkpoint, device="cuda").model.eval()
    dev = next(model.parameters()).device
    states = collect(model, dev, 64, 0.2, np.random.default_rng(1), True, a.positions)
    for w in a.worlds:
        root = RolloutRoot(model, RolloutConfig(k=4, worlds=w, horizon=4, rule="z2", stop="earlier", leaf_side="own",
                                                keyed=True), device=dev)
        profile(root, states[:32], max(32, 16384 // w))           # warm-up
        t = profile(root, states, max(32, 16384 // w))
        stages = ["root and add", "fill_live (C++)", "upload", "forward", "download", "step (C++)", "leaves"]
        total = sum(t[s] for s in stages)
        print(f"\n{w} worlds: {int(t['decisions'])} decisions in {total:.1f}s ({t['decisions'] / total:.1f}/s); "
              f"{int(t['rows']):,} playout rows over {int(t['plies'])} plies ({t['rows'] / max(1, t['plies']):,.0f} a ply), "
              f"{int(t['leaf rows']):,} leaf rows; {t['rows'] / total / 1e3:.0f}k rows/s overall; "
              f"{t['bytes up'] / max(1e-9, t['upload']) / 1e9:.1f} GB/s upload")
        for s in stages:
            per_row = 1e6 * t[s] / max(1, t["rows"])
            print(f"  {s:18s} {t[s]:7.2f}s  {100 * t[s] / total:5.1f}%  {per_row:6.2f} us a playout row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
