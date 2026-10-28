"""Does a critic predict the results of searched games? (P33, R51's sanity check.)

Fresh games are played by the rollout root (k=4, 16 worlds, stop=earlier, leaf=own) on both sides with a fixed
searcher network, in `SearchedGames` worker processes; every decision is labelled with its game's result from
the decider's side -- exactly R51's training rows. Each critic named is then scored on them: mean squared error
against the ±1 result and log loss of (v + 1) / 2 on the decided games. A critic trained on such rows should
predict them better than one that was not.

    PYTHONPATH=.:build/release python -m ai.eval.searched_value_probe --searcher control.pt \\
        --critic r51.pt control.pt [--games 512]
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from ai.search.rollout_root import RolloutConfig
from ai.training.searched_games import SearchedGames
from ai.eval.loading import load_for_eval


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--searcher", required=True)
    ap.add_argument("--critic", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    searcher = load_for_eval(a.searcher, "cpu", merged_ok=False).model
    sg = SearchedGames(a.workers, searcher, RolloutConfig(k=4, worlds=16, horizon=4, rule="z2", stop="earlier",
                                                         leaf_side="own"), dev, envs=32)
    obs: List[np.ndarray] = []
    tgt: List[np.ndarray] = []
    games = 0
    try:
        while games < a.games:
            time.sleep(5)
            got = sg.drain()
            if got is not None:
                obs.append(got[0])
                tgt.append(got[1])
                games += got[2]
    finally:
        sg.close()
    x = np.concatenate(obs)
    y = np.concatenate(tgt)
    decided = y != 0
    print(f"{games} searched games by {a.searcher}: {x.shape[0]} decisions ({int(decided.sum())} in decided games)")
    for ck in a.critic:
        model = load_for_eval(ck, str(dev), merged_ok=False).model
        vs = []
        with torch.no_grad():
            for lo in range(0, x.shape[0], 4096):
                vs.append(model(torch.from_numpy(x[lo:lo + 4096]).to(dev), None)[1].float().squeeze(-1).cpu().numpy())
        v = np.concatenate(vs)
        mse = float(((v - y) ** 2).mean())
        p = np.clip((v[decided] + 1) / 2, 1e-6, 1 - 1e-6)
        yy = (y[decided] > 0).astype(np.float64)
        ll = float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
        print(f"   {ck}\n      MSE against the result {mse:.4f}   log loss {ll:.4f}   mean v {float(v.mean()):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
