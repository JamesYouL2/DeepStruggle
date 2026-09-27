#!/usr/bin/env python3
"""The realignment oracle: where the checkpoint could realign but does not, would realigning win more?

The play-mode census (`play_mode_census.py`) found the policy realigns at the human rate at DEFCON 2
and about 30x less than humans at DEFCON 3+. A rate does not say who is right. This asks the game.

Positions are the checkpoint's own play-mode nodes (the choice between event, space, influence,
coup and realignment), from its self-play at temperature 0.1, where realignment is legal and the
greedy policy picks something else. From each, two branches:

* **policy** -- the node as it stands; the playout's greedy policy makes its own choice;
* **realign** -- realignment is forced at this node, and from there the checkpoint plays on as it
  would, choosing the targets of every roll itself. This measures "realign here, the way you would
  realign", not "realign optimally", so a gain is a lower bound on what realignment is worth.

Each branch is played to the end `--pairs` times by the checkpoint on both sides, greedily. Pair k
of a position re-seeds the engine RNG identically in both branches, so the difference is the mode's.
Positions are stratified by DEFCON (2 against 3+), and results are reported by DEFCON, side, the
policy's own P(realign) and the mode it chose instead.

    PYTHONPATH=.:build/release python tools/scripts/realign_oracle.py \\
        --checkpoint <snapshot.pt> --positions-per-class 400 --pairs 16 --output-md out.md

**`--scenario locked-bg`** asks the owner's sharper question instead: the textbook realignment
spot. At DEFCON 2, with realignment legal, there is a battleground where the mover has no influence,
the opponent has some, and the mover's net realignment modifier is positive, and the mover can
reach no battleground that nobody controls. There realigning is often right. Every such node is
counted, whatever the policy does, so the policy's realignment rate in the spot is reported. For
the nodes where it declines, a third branch forces realignment with every roll on that battleground
(the one with the best modifier) while it is legal. **`--scenario locked-bg-takeable`** relaxes "no
reachable open battleground" to "none this card's influence points could take control of".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from bindings.settle import SettleMode, settle  # noqa: E402
from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from ai.eval.ops_block import NODE_OFFSET, access, controlled, country_table, influence  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.scripts.ops_block_oracle import _pair_seed, playouts  # noqa: E402

MODE_BASE = 110
MODES = ("event", "space", "influence", "coup", "realign")
REALIGN = MODE_BASE + 4


def collect(agent: NeuralAgent, per_class: int, envs: int, seed: int, dev: torch.device,
            accept: float = 0.08, max_steps: int = 200000) -> List[Dict[str, Any]]:
    """Play-mode nodes where realignment is legal and the greedy policy declines it, up to
    `per_class` at DEFCON 2 and as many at DEFCON 3+, from self-play at temperature 0.1.

    Each candidate is kept with probability `accept`, so the sample spreads over many games and
    every stage of them instead of filling up from the opening of the first batch."""
    rng = np.random.default_rng(seed)
    model = agent.model
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    obs, masks, _info = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(seed)
    got: Dict[str, List[Dict[str, Any]]] = {"defcon2": [], "defcon3+": []}
    for _ in range(max_steps):
        m_np = np.asarray(masks)
        with torch.no_grad():
            o = torch.from_numpy(np.asarray(obs)).float().to(dev)
            m = torch.from_numpy(m_np).to(dev)
            logits = model(o, m)[0].float()
            probs = torch.softmax(logits, -1)
            greedy = logits.argmax(-1).cpu().numpy()
            acts = torch.multinomial(torch.softmax(logits / 0.1, -1), 1, generator=gen).squeeze(-1).cpu().numpy()
        legal = m_np[:, MODE_BASE:MODE_BASE + len(MODES)].astype(bool)
        cand = np.flatnonzero(legal[:, 4] & (legal.sum(1) >= 2) & (greedy != REALIGN)
                              & (greedy >= MODE_BASE) & (greedy < MODE_BASE + len(MODES)))
        if len(cand):
            p_re = probs[torch.from_numpy(cand).to(dev), REALIGN].cpu().numpy()
            for j, i in enumerate(cand):
                st = env.runner.get_state(int(i))
                cls = "defcon2" if int(st.defcon) == 2 else "defcon3+"
                if len(got[cls]) >= per_class or rng.random() >= accept:
                    continue
                got[cls].append({"state": st.clone(), "defcon": int(st.defcon), "turn": int(st.turn),
                                 "side": 0 if int(st.ctx().decision_player) == int(ts.Player.US) else 1,
                                 "p_realign": float(p_re[j]), "chosen": MODES[int(greedy[i]) - MODE_BASE]})
        if all(len(v) >= per_class for v in got.values()):
            break
        obs, masks, _r, _d, _info = env.step(acts)
    return got["defcon2"] + got["defcon3+"]


def run(checkpoint: str, per_class: int, pairs: int, envs: int, seed: int, dev: torch.device,
        chunk_envs: int, accept: float = 0.08) -> Dict[str, Any]:
    agent = NeuralAgent.from_checkpoint(checkpoint, device=str(dev))
    model = agent.model.eval()
    t0 = time.time()
    nodes = collect(agent, per_class, envs, seed, dev, accept)
    t_collect = time.time() - t0
    for n in nodes:
        forced = n["state"].clone()
        ts.Engine.step_flat(forced, REALIGN, False)
        settle(forced, SettleMode.CHANCE)
        n["forced"] = forced
    per = 2 * pairs
    step = max(1, chunk_envs // per)
    for lo in range(0, len(nodes), step):
        chunk = nodes[lo:lo + step]
        states: List["ts.GameState"] = []
        seeds: List[int] = []
        for idx, n in enumerate(chunk):
            for k in range(pairs):
                s = _pair_seed(seed, lo + idx, k)
                states += [n["forced"], n["state"]]
                seeds += [s, s]
        util = playouts(model, dev, states, seeds, 0.0)
        for j, n in enumerate(chunk):
            u = util[j * per:(j + 1) * per].reshape(pairs, 2)
            score = ((1.0 if n["side"] == 0 else -1.0) * u + 1.0) / 2.0
            n["win_realign"] = float(score[:, 0].mean())
            n["win_policy"] = float(score[:, 1].mean())
    rows = [{k: v for k, v in n.items() if k not in ("state", "forced")} for n in nodes]
    return {"checkpoint": agent.name, "positions": len(rows), "pairs": pairs,
            "seconds": round(time.time() - t0, 1), "seconds_collecting": round(t_collect, 1), "rows": rows}


def summarise(res: Dict[str, Any]) -> List[str]:
    rows = res["rows"]
    if not rows:
        return ["no positions"]
    d = np.array([r["win_realign"] - r["win_policy"] for r in rows])

    def line(label: str, sel: np.ndarray) -> str:
        x = d[sel]
        if len(x) < 2:
            return f"| {label} | {len(x)} | — | — | — | — | — |"
        idx = np.flatnonzero(sel)
        wr = np.mean([rows[i]["win_realign"] for i in idx])
        wp = np.mean([rows[i]["win_policy"] for i in idx])
        se = x.std(ddof=1) / np.sqrt(len(x))
        return (f"| {label} | {len(x)} | {100 * wr:.1f}% | {100 * wp:.1f}% | **{100 * x.mean():+.1f}** ± {100 * se:.1f} "
                f"| {100 * np.mean(x >= 0.25):.1f}% | {100 * np.mean(x <= -0.25):.1f}% |")

    dc = np.array([r["defcon"] for r in rows])
    side = np.array([r["side"] for r in rows])
    p = np.array([r["p_realign"] for r in rows])
    chosen = np.array([r["chosen"] for r in rows])
    out = [f"{res['checkpoint']}: {res['positions']} play-mode nodes where realignment is legal and the greedy "
           f"policy declines it; {res['pairs']} paired greedy playouts per branch",
           "",
           "| positions | n | realign | policy | realign − policy (pp) | realign ≥ 25 pp better | ≥ 25 pp worse |",
           "|:---|---:|---:|---:|---:|---:|---:|"]
    out.append(line("all", np.ones(len(rows), dtype=bool)))
    for label, sel in (("DEFCON 2", dc == 2), ("DEFCON 3+", dc >= 3)):
        out.append(line(label, sel))
        out.append(line(f"{label}, mover US", sel & (side == 0)))
        out.append(line(f"{label}, mover USSR", sel & (side == 1)))
    turn = np.array([r["turn"] for r in rows])
    for lo, hi in ((1, 3), (4, 7), (8, 10)):
        out.append(line(f"turns {lo}–{hi}", (turn >= lo) & (turn <= hi)))
    for lo, hi in ((0.0, 0.01), (0.01, 0.05), (0.05, 1.0)):
        out.append(line(f"P(realign) in [{lo:.0%}, {hi:.0%})", (p >= lo) & (p < hi)))
    for m in MODES[:4]:
        out.append(line(f"policy chose {m}", chosen == m))
    return out


_BOARD_W = 26
_NET_REALIGN_SLOT = 2          # board_features[country * 26 + 2]: net realignment modifier / 5, mover's view


INFLUENCE = MODE_BASE + 2


def _influence_points(st: "ts.GameState") -> int:
    """How many influence points this card would place, read off the engine by entering the mode."""
    probe = st.clone()
    ts.Engine.step_flat(probe, INFLUENCE, False)
    settle(probe, SettleMode.CHANCE)
    return int(probe.ctx().remaining_steps)


def locked_bg_target(st: "ts.GameState", takeable_only: bool = False) -> Optional[int]:
    """The battleground this node is the textbook realignment spot for, or None.

    DEFCON 2; a battleground with no mover influence, some opponent influence and a positive net
    modifier for the mover; and no battleground nobody controls within the mover's reach -- or,
    with `takeable_only`, none the mover could take control of with this card's influence points.
    Of several targets, the best modifier, then the most opponent influence."""
    if int(st.defcon) != 2:
        return None
    stab, bg, _ = country_table()
    mover = ts.Player(int(st.ctx().decision_player))
    side = 0 if mover == ts.Player.US else 1
    inf = influence(st)
    open_bg = bg & ~controlled(inf, side) & ~controlled(inf, 1 - side) & access(inf, side)
    if takeable_only and open_bg.any():
        mask = np.asarray(ts.Engine.get_flat_action_mask(st, False))
        pts = _influence_points(st) if mask[INFLUENCE] else 0
        need = inf[1 - side] + stab - inf[side]
        open_bg &= need <= pts
    if open_bg.any():
        return None
    board = np.asarray(ts.extract_observation(st, mover))[:84 * _BOARD_W].reshape(84, _BOARD_W)
    net = board[:, _NET_REALIGN_SLOT]
    cand = np.flatnonzero(bg & (inf[side] == 0) & (inf[1 - side] > 0) & (net > 0))
    if not len(cand):
        return None
    return int(max(cand, key=lambda c: (net[c], inf[1 - side, c])))


def collect_locked(agent: NeuralAgent, n: int, envs: int, seed: int, dev: torch.device,
                   takeable_only: bool = False, max_steps: int = 400000) -> Dict[str, Any]:
    """Every play-mode node of the locked-battleground scenario in self-play at temperature 0.1,
    until `n` of them are declines; the realigning ones are counted but not kept."""
    model = agent.model
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    obs, masks, _info = env.reset_all()
    gen = torch.Generator(device=dev).manual_seed(seed)
    keep: List[Dict[str, Any]] = []
    seen = realigned = games = 0
    p_all: List[float] = []
    for _ in range(max_steps):
        m_np = np.asarray(masks)
        with torch.no_grad():
            o = torch.from_numpy(np.asarray(obs)).float().to(dev)
            logits = model(o, torch.from_numpy(m_np).to(dev))[0].float()
            probs = torch.softmax(logits, -1)
            greedy = logits.argmax(-1).cpu().numpy()
            acts = torch.multinomial(torch.softmax(logits / 0.1, -1), 1, generator=gen).squeeze(-1).cpu().numpy()
        legal = m_np[:, MODE_BASE:MODE_BASE + len(MODES)].astype(bool)
        for i in np.flatnonzero(legal[:, 4] & (legal.sum(1) >= 2)):
            st = env.runner.get_state(int(i))
            tgt = locked_bg_target(st, takeable_only)
            if tgt is None:
                continue
            seen += 1
            p_re = float(probs[int(i), REALIGN].item())
            p_all.append(p_re)
            if int(greedy[i]) == REALIGN:
                realigned += 1
                continue
            if not MODE_BASE <= int(greedy[i]) < MODE_BASE + len(MODES):
                continue
            keep.append({"state": st.clone(), "defcon": 2, "turn": int(st.turn), "target": tgt,
                         "side": 0 if int(st.ctx().decision_player) == int(ts.Player.US) else 1,
                         "p_realign": p_re, "chosen": MODES[int(greedy[i]) - MODE_BASE]})
        if len(keep) >= n:
            break
        obs, masks, _r, d, _info = env.step(acts)
        games += int(np.asarray(d).sum())
    return {"nodes": keep[:n], "seen": seen, "realigned": realigned, "games": games,
            "p_realign_mean": float(np.mean(p_all)) if p_all else None}


@torch.no_grad()
def _drive_realign(model: torch.nn.Module, dev: torch.device, start: "ts.GameState", target: int,
                   rng: int) -> "ts.GameState":
    """Realign at `start`, every roll on `target` while legal; the rest of the mover's action
    round greedily."""
    st = start.clone()
    st.rng_state = int(rng)
    mover = int(st.ctx().decision_player)
    turn, ar = st.turn, st.action_round
    ts.Engine.step_flat(st, REALIGN, False)
    settle(st, SettleMode.CHANCE)
    for _ in range(200):
        if ts.Engine.is_terminal(st) or (st.turn, st.action_round) != (turn, ar) \
                or int(st.ctx().decision_player) != mover:
            return st
        mask = np.asarray(ts.Engine.get_flat_action_mask(st, False))
        if st.ctx().op_mode == ts.OpMode.REALIGN and mask[NODE_OFFSET + target]:
            a = NODE_OFFSET + target
        else:
            o = torch.from_numpy(np.asarray(ts.extract_observation(st, ts.Player(mover)))[None]).float().to(dev)
            a = int(model(o, torch.from_numpy(mask[None]).to(dev))[0].float().argmax(-1).item())
        ts.Engine.step_flat(st, a, False)
        settle(st, SettleMode.CHANCE)
    raise RuntimeError("the forced realignment did not finish its action round")


def run_locked(checkpoint: str, n: int, pairs: int, envs: int, seed: int, dev: torch.device,
               chunk_envs: int, takeable_only: bool = False) -> Dict[str, Any]:
    agent = NeuralAgent.from_checkpoint(checkpoint, device=str(dev))
    model = agent.model.eval()
    t0 = time.time()
    col = collect_locked(agent, n, envs, seed, dev, takeable_only)
    nodes = col["nodes"]
    per = 3 * pairs
    step = max(1, chunk_envs // per)
    for lo in range(0, len(nodes), step):
        chunk = nodes[lo:lo + step]
        states: List["ts.GameState"] = []
        seeds: List[int] = []
        for idx, nd in enumerate(chunk):
            own = nd["state"].clone()
            ts.Engine.step_flat(own, REALIGN, False)
            settle(own, SettleMode.CHANCE)
            for k in range(pairs):
                s = _pair_seed(seed, lo + idx, k)
                states += [_drive_realign(model, dev, nd["state"], nd["target"], s), own, nd["state"]]
                seeds += [s, s, s]
        util = playouts(model, dev, states, seeds, 0.0)
        for j, nd in enumerate(chunk):
            u = util[j * per:(j + 1) * per].reshape(pairs, 3)
            score = ((1.0 if nd["side"] == 0 else -1.0) * u + 1.0) / 2.0
            nd["win_target"] = float(score[:, 0].mean())
            nd["win_realign"] = float(score[:, 1].mean())
            nd["win_policy"] = float(score[:, 2].mean())
    rows = [{k: v for k, v in nd.items() if k != "state"} for nd in nodes]
    return {"checkpoint": agent.name, "scenario": "no takeable open battleground" if takeable_only
            else "no reachable open battleground", "positions": len(rows), "pairs": pairs, "seen": col["seen"],
            "realigned": col["realigned"], "games": col["games"], "p_realign_mean": col["p_realign_mean"],
            "seconds": round(time.time() - t0, 1), "rows": rows}


def summarise_locked(res: Dict[str, Any]) -> List[str]:
    rows = res["rows"]
    _, _, names = country_table()
    out = [f"{res['checkpoint']} ({res['scenario']}): the locked-battleground spot came up {res['seen']} times in about "
           f"{res['games']} games; greedy realigns in {res['realigned']} "
           f"({100 * res['realigned'] / max(1, res['seen']):.1f}%), mean P(realign) "
           f"{100 * (res['p_realign_mean'] or 0):.1f}%. Of the declines, {len(rows)} played out, "
           f"{res['pairs']} paired greedy playouts per branch.", ""]
    if not rows:
        return out
    dt = np.array([r["win_target"] - r["win_policy"] for r in rows])
    dr = np.array([r["win_realign"] - r["win_policy"] for r in rows])

    def line(label: str, sel: np.ndarray) -> str:
        if sel.sum() < 2:
            return f"| {label} | {int(sel.sum())} | — | — | — | — | — |"
        idx = np.flatnonzero(sel)
        wp = np.mean([rows[i]["win_policy"] for i in idx])
        wt = np.mean([rows[i]["win_target"] for i in idx])
        a, b = dt[sel], dr[sel]
        return (f"| {label} | {len(idx)} | {100 * wp:.1f}% | {100 * wt:.1f}% | **{100 * a.mean():+.1f}** ± "
                f"{100 * a.std(ddof=1) / np.sqrt(len(a)):.1f} | {100 * b.mean():+.1f} ± "
                f"{100 * b.std(ddof=1) / np.sqrt(len(b)):.1f} | {100 * np.mean(a >= 0.25):.1f}% |")

    out += ["| positions | n | policy | realign on the battleground | target − policy (pp) | "
            "realign, policy's targets − policy | target ≥ 25 pp better |",
            "|:---|---:|---:|---:|---:|---:|---:|"]
    side = np.array([r["side"] for r in rows])
    turn = np.array([r["turn"] for r in rows])
    chosen = np.array([r["chosen"] for r in rows])
    out.append(line("all", np.ones(len(rows), dtype=bool)))
    out.append(line("mover US", side == 0))
    out.append(line("mover USSR", side == 1))
    for lo, hi in ((1, 3), (4, 7), (8, 10)):
        out.append(line(f"turns {lo}–{hi}", (turn >= lo) & (turn <= hi)))
    for m in MODES[:4]:
        out.append(line(f"policy chose {m}", chosen == m))
    tg = [names[r["target"]] for r in rows]
    for c in sorted(set(tg), key=lambda c: -tg.count(c))[:8]:
        out.append(line(f"target {c}", np.array([t == c for t in tg])))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scenario", choices=["declined", "locked-bg", "locked-bg-takeable"], default="declined")
    ap.add_argument("--positions-per-class", type=int, default=400, help="per DEFCON class (2, 3+)")
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--accept", type=float, default=0.08,
                    help="Probability of keeping each candidate node, to spread the sample over games.")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-envs", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.scenario in ("locked-bg", "locked-bg-takeable"):
        res = run_locked(a.checkpoint, a.positions_per_class, a.pairs, a.envs, a.seed, dev, a.chunk_envs,
                         a.scenario == "locked-bg-takeable")
        lines = summarise_locked(res)
    else:
        res = run(a.checkpoint, a.positions_per_class, a.pairs, a.envs, a.seed, dev, a.chunk_envs, a.accept)
        lines = summarise(res)
    print("\n".join(lines))
    if a.output_md:
        with open(a.output_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
