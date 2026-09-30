#!/usr/bin/env python3
"""P29 bet 2: does the ownership head know more than a per-country habit?

Plays a checkpoint trained with `--aux-ownership` against itself, samples positions, and when each
game ends compares the head's prediction of every country's final controller with two baselines:

* **constant** -- the country's most common final controller over the probe's games (Australia
  is the US's, North Korea the USSR's). Fitted on the same games, so it is an upper bound for any
  per-country constant.
* **current** -- whoever controls the country at the sampled position keeps it.

Both questions the owner asked are answered per country:

1. is the head better than the constant for that country (accuracy and log-loss against the
   country's own final-controller frequencies)?
2. on the **surprises** -- games where the country did *not* end with its usual controller -- does
   the head call the actual outcome, and does it do better than the current board?

Labels are absolute (US / USSR / neither); the head's mover-frame output is turned around for a
USSR mover. Positions come from self-play at `--temperature` (1.0, the training rollouts' value).

    PYTHONPATH=.:build/release python tools/scripts/aux_ownership_probe.py \\
        --checkpoint <run>/snapshot_...pt --games 2000 --output-json out.json --output-md out.md
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

N_COUNTRIES = 84
US, USSR, NONE = 0, 1, 2                 # absolute classes
CLASS_NAMES = ("US", "USSR", "none")
EPS = 1e-6


def _absolute_class(ctrl: np.ndarray) -> np.ndarray:
    """ts.Player values (+1 US, -1 USSR, 0 neither) -> US / USSR / NONE."""
    return np.where(ctrl == 1, US, np.where(ctrl == -1, USSR, NONE)).astype(np.int64)


def _control(st: "ts.GameState") -> np.ndarray:
    return np.array([int(ts.Scoring.get_country_control(st, c)) for c in range(N_COUNTRIES)], dtype=np.int8)


@torch.no_grad()
def collect(agent: NeuralAgent, games: int, envs: int, seed: int, sample_frac: float,
            temperature: float, dev: torch.device) -> Dict[str, np.ndarray]:
    """Sampled positions with the head's absolute probabilities, the board then, and the final one."""
    model: Any = agent.model
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    env.record_final_control = True
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    pending: List[List[Dict[str, Any]]] = [[] for _ in range(envs)]
    rows: List[Dict[str, Any]] = []
    done_games = 0
    # Stopping at the target would drop the games still running, which are the long ones. So once
    # it is reached no new game is sampled, and stepping goes on until those in flight end.
    sampling = np.ones(envs, dtype=bool)
    while done_games < games or any(pending):
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        o = torch.from_numpy(np.asarray(obs)).float().to(dev)
        m = torch.from_numpy(np.asarray(masks)).to(dev)
        pick = np.flatnonzero(sampling & (dp != 0) & (rng.random(envs) < sample_frac))
        if pick.size:
            own, _vp = model.forward_aux(o[torch.from_numpy(pick).to(dev)])
            p = torch.softmax(own.float(), dim=-1).cpu().numpy()          # (k, 84, mine/theirs/neither)
            for j, i in enumerate(pick):
                mover = int(dp[i])
                pa = p[j][:, [0, 1, 2]] if mover == 1 else p[j][:, [1, 0, 2]]   # -> US / USSR / none
                st = env.runner.get_state(int(i))
                pending[int(i)].append({"p": pa.astype(np.float32), "turn": int(st.turn),
                                        "now": _absolute_class(_control(st))})
        logits = model(o, m)[0].float()
        if temperature <= 0:
            acts = logits.argmax(-1).cpu().numpy()
        else:
            acts = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1).cpu().numpy()
        obs, masks, _r, _d, info = env.step(acts)
        for ep in info.get("completed_episodes", []):
            i = int(ep["env_idx"])
            final = _absolute_class(np.asarray(ep["final_control"]))
            if not sampling[i]:
                continue
            for r in pending[i]:
                rows.append({**r, "final": final, "game": done_games})
            pending[i] = []
            done_games += 1
            if done_games >= games:
                sampling[i] = False
    return {"p": np.stack([r["p"] for r in rows]), "now": np.stack([r["now"] for r in rows]),
            "final": np.stack([r["final"] for r in rows]), "turn": np.array([r["turn"] for r in rows]),
            "game": np.array([r["game"] for r in rows])}


def analyse(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
    p, now, final, turn = d["p"], d["now"], d["final"], d["turn"]
    n = final.shape[0]
    # per-country final-controller frequencies (the constant's distribution), counted once per game
    _, first = np.unique(d["game"], return_index=True)
    freq = np.stack([np.bincount(final[first, c], minlength=3) for c in range(N_COUNTRIES)]).astype(float)
    freq /= freq.sum(1, keepdims=True)
    const = freq.argmax(1)                                              # (84,)
    head = p.argmax(-1)                                                 # (n, 84)
    idx = np.arange(N_COUNTRIES)
    p_true = np.take_along_axis(p, final[..., None], -1)[..., 0]        # head's prob. on the outcome
    f_true = freq[idx[None, :], final]                                  # constant's prob. on it
    hit_head, hit_const, hit_now = head == final, const[None, :] == final, now == final
    surprise = final != const[None, :]
    changed = final != now

    def block(mask: np.ndarray) -> Dict[str, float]:
        k = int(mask.sum())
        if k == 0:
            return {"n": 0}
        return {"n": k, "head": float(hit_head[mask].mean()), "constant": float(hit_const[mask].mean()),
                "current": float(hit_now[mask].mean()),
                "head_logloss": float(-np.log(p_true[mask] + EPS).mean()),
                "constant_logloss": float(-np.log(f_true[mask] + EPS).mean())}

    every = np.ones_like(final, dtype=bool)
    stages = {"turns 1-3": (1, 3), "turns 4-7": (4, 7), "turns 8-10": (8, 10)}
    out: Dict[str, Any] = {
        "positions": n, "games": int(len(first)),
        "all": block(every), "surprises": block(surprise), "changes": block(changed),
        "by_stage": {k: {"all": block(every & ((turn >= a) & (turn <= b))[:, None]),
                         "surprises": block(surprise & ((turn >= a) & (turn <= b))[:, None])}
                     for k, (a, b) in stages.items()},
        "countries": [],
    }
    for c in range(N_COUNTRIES):
        col = np.zeros_like(final, dtype=bool)
        col[:, c] = True
        out["countries"].append({
            "id": c, "name": ts.MapData.get_country_name(c), "usual": CLASS_NAMES[int(const[c])],
            "usual_rate": float(freq[c].max()), "all": block(col), "surprises": block(col & surprise)})
    return out


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def render(res: Dict[str, Any]) -> str:
    L: List[str] = [f"# Ownership head against per-country baselines: {res['checkpoint']}", "",
                    f"{res['games']} self-play games at temperature {res['temperature']}, "
                    f"{res['positions']} sampled positions × 84 countries.", "",
                    "Accuracy in %, log-loss in nats. *constant* = the country's most common final controller "
                    "(fitted on these games); *current* = its controller at the position. A *surprise* is a "
                    "country that did not end with its usual controller; a *change* one whose controller "
                    "changed after the position.", "",
                    "| slice | n | head | constant | current | head log-loss | constant log-loss |",
                    "|:---|---:|---:|---:|---:|---:|---:|"]

    def row(name: str, b: Dict[str, Any]) -> str:
        if not b.get("n"):
            return f"| {name} | 0 | | | | | |"
        return (f"| {name} | {b['n']} | {_pct(b['head'])} | {_pct(b['constant'])} | {_pct(b['current'])} | "
                f"{b['head_logloss']:.3f} | {b['constant_logloss']:.3f} |")

    L.append(row("all", res["all"]))
    L.append(row("surprises", res["surprises"]))
    L.append(row("changes after the position", res["changes"]))
    for k, v in res["by_stage"].items():
        L.append(row(f"{k}, all", v["all"]))
        L.append(row(f"{k}, surprises", v["surprises"]))
    L += ["", "## Per country", "",
          "Sorted by how often the usual controller fails. *surprises* columns are the games where it did.", "",
          "| country | usual | usual rate | head | constant | current | head LL | const LL | surprises n | head | current |",
          "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(res["countries"], key=lambda c: c["usual_rate"]):
        a, s = c["all"], c["surprises"]
        L.append(f"| {c['name']} | {c['usual']} | {_pct(c['usual_rate'])} | {_pct(a['head'])} | "
                 f"{_pct(a['constant'])} | {_pct(a['current'])} | {a['head_logloss']:.3f} | "
                 f"{a['constant_logloss']:.3f} | {s.get('n', 0)} | {_pct(s.get('head'))} | {_pct(s.get('current'))} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--sample-frac", type=float, default=0.05,
                    help="fraction of decisions sampled as positions (training uses 0.1)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    agent = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev))
    if not getattr(agent.model, "aux_heads", False):
        raise SystemExit(f"{a.checkpoint} has no aux heads (trained without --aux-ownership)")
    data = collect(agent, a.games, a.envs, a.seed, a.sample_frac, a.temperature, dev)
    res = {"checkpoint": agent.name, "path": a.checkpoint, "temperature": a.temperature, **analyse(data)}
    md = render(res)
    print(md)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(res, f, indent=1)
    if a.output_md:
        with open(a.output_md, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
