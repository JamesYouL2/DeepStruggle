#!/usr/bin/env python3
"""How often does a checkpoint choose each play mode -- above all, how often does it realign?

Realignment is the hardest mode to learn: its value depends on the roll modifiers (adjacent
control, the superpower's own adjacency, who has more influence) and it pays off only by
preventing the opponent's scoring or control later. A policy can learn "never realign" early,
while it understands nothing about realignment, and then stop sampling it often enough to find out
it is sometimes right. This counts, at every node that chooses how a card resolves
(event / space / influence / coup / realign), what the policy picked and how much probability it
put on each mode, from its own self-play, beside the same count over the human corpus.

Only nodes where realignment is legal are counted for the realignment figures, and DEFCON 2 is
reported apart: there coups are barred from most regions and realignment is the remaining way to
attack influence without spending it.

    PYTHONPATH=.:build/release python tools/scripts/play_mode_census.py \\
        --checkpoint a.pt b.pt --games 400 --human-corpus data/datasets/human_corpus_e4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402

#: flat indices of the resolution choice, in ts::Resolution order
MODE_BASE = 110
MODES = ("event", "space", "influence", "coup", "realign")
REALIGN = 4
#: global_features start after the 84x26 board and the 110x14 card block
_GLOBAL = 84 * 26 + 110 * 14
_DEFCON = _GLOBAL + 1
_TURN = _GLOBAL + 6


def _record(rows: List[Dict[str, Any]], legal: np.ndarray, chosen: int, probs: Optional[np.ndarray],
            defcon: int, turn: int, side: str) -> None:
    rows.append({"legal": [int(x) for x in legal], "chosen": int(chosen),
                 "probs": None if probs is None else [round(float(x), 5) for x in probs],
                 "defcon": int(defcon), "turn": int(turn), "side": side})


@torch.no_grad()
def self_play(checkpoint: str, games: int, envs: int, seed: int, temperature: float,
              dev: torch.device, max_steps: int = 200_000) -> Dict[str, Any]:
    agent = NeuralAgent.from_checkpoint(checkpoint, device=str(dev))
    model = agent.model.eval()
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    obs, masks, _info = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(seed)
    rows: List[Dict[str, Any]] = []
    done_games = 0
    for _ in range(max_steps):
        o_np, m_np = np.asarray(obs), np.asarray(masks)
        o = torch.from_numpy(o_np).float().to(dev)
        m = torch.from_numpy(m_np).to(dev)
        logits = model(o, m)[0].float()
        probs = torch.softmax(logits, dim=-1)
        if temperature <= 0.0:
            acts = logits.argmax(-1)
        else:
            acts = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1)
        acts_np = acts.cpu().numpy()
        mode_legal = m_np[:, MODE_BASE:MODE_BASE + len(MODES)].astype(bool)
        pick = np.flatnonzero(mode_legal.sum(1) >= 2)
        if len(pick):
            p_np = probs[torch.from_numpy(pick).to(dev), MODE_BASE:MODE_BASE + len(MODES)].cpu().numpy()
            for j, i in enumerate(pick):
                a = int(acts_np[i])
                if not MODE_BASE <= a < MODE_BASE + len(MODES):
                    continue
                st = env.runner.get_state(int(i))
                side = "US" if int(st.ctx().decision_player) == int(ts.Player.US) else "USSR"
                _record(rows, np.flatnonzero(mode_legal[i]), a - MODE_BASE, p_np[j],
                        int(st.defcon), int(st.turn), side)
        obs, masks, _r, d, _info = env.step(acts_np)
        done_games += int(np.asarray(d).sum())
        if done_games >= games:
            break
    return {"source": agent.name, "games": done_games, "temperature": temperature, "rows": rows}


def human(corpus_dir: str) -> Dict[str, Any]:
    action = np.load(os.path.join(corpus_dir, "action.npy"))
    sel = (action >= MODE_BASE) & (action < MODE_BASE + len(MODES))
    idx = np.flatnonzero(sel)
    mask = np.unpackbits(np.load(os.path.join(corpus_dir, "mask.npy"), mmap_mode="r")[idx], axis=1)
    obs = np.load(os.path.join(corpus_dir, "obs.npy"), mmap_mode="r")
    rows: List[Dict[str, Any]] = []
    for k, i in enumerate(idx):
        legal = np.flatnonzero(mask[k, MODE_BASE:MODE_BASE + len(MODES)])
        if len(legal) < 2:
            continue
        g = obs[i, [_DEFCON, _TURN]].astype(np.float32)
        _record(rows, legal, int(action[i]) - MODE_BASE, None, int(round(g[0] * 5)),
                int(round(g[1] * 10)), "?")
    games = int(len(np.unique(np.load(os.path.join(corpus_dir, "game.npy"))[idx])))
    return {"source": "human corpus", "games": games, "temperature": None, "rows": rows}


def summarise(res: Dict[str, Any]) -> Dict[str, Any]:
    rows = res["rows"]

    def block(sel: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        ra = [r for r in sel if REALIGN in r["legal"]]
        ops = [r for r in ra if r["chosen"] >= 2]
        out: Dict[str, Any] = {
            "nodes": len(sel),
            "nodes_realign_legal": len(ra),
            "realign_chosen": sum(r["chosen"] == REALIGN for r in ra),
            "realign_share_of_nodes": (sum(r["chosen"] == REALIGN for r in ra) / len(ra)) if ra else None,
            "realign_share_of_ops": (sum(r["chosen"] == REALIGN for r in ops) / len(ops)) if ops else None,
            "mode_share": {MODES[k]: sum(r["chosen"] == k for r in sel) / max(1, len(sel))
                           for k in range(len(MODES))},
        }
        pr = [r["probs"][REALIGN] for r in ra if r["probs"] is not None]
        if pr:
            p = np.array(pr)
            out["realign_prob_mean"] = float(p.mean())
            out["realign_prob_median"] = float(np.median(p))
            out["realign_prob_over_5pct"] = float((p > 0.05).mean())
        return out

    return {"source": res["source"], "games": res["games"], "temperature": res["temperature"],
            "all": block(rows),
            "defcon2": block([r for r in rows if r["defcon"] == 2]),
            "defcon3plus": block([r for r in rows if r["defcon"] >= 3])}


def table(summaries: Sequence[Dict[str, Any]]) -> List[str]:
    def pct(x: Optional[float]) -> str:
        return "—" if x is None else f"{100 * x:.2f}%"

    out = ["| source | games | realign nodes | realign chosen | of ops plays | DEFCON 2: of ops plays | "
           "DEFCON 3+: of ops plays | mean P(realign) | P(realign) > 5% |",
           "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        a, d2, d3 = s["all"], s["defcon2"], s["defcon3plus"]
        out.append(f"| {s['source']} | {s['games']} | {a['nodes_realign_legal']} | {a['realign_chosen']} | "
                   f"{pct(a['realign_share_of_ops'])} | {pct(d2['realign_share_of_ops'])} "
                   f"({d2['realign_chosen']}/{d2['nodes_realign_legal']}) | {pct(d3['realign_share_of_ops'])} | "
                   f"{pct(a.get('realign_prob_mean'))} | {pct(a.get('realign_prob_over_5pct'))} |")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Self-play temperature; 0 is greedy, as tournaments rate. The policy's own "
                         "probabilities are recorded at temperature 1 either way.")
    ap.add_argument("--human-corpus", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    results = [self_play(c, a.games, a.envs, a.seed, a.temperature, dev) for c in a.checkpoint]
    if a.human_corpus:
        results.append(human(a.human_corpus))
    sums = [summarise(r) for r in results]
    print("\n".join(table(sums)))
    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump({"summaries": sums, "results": results}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
