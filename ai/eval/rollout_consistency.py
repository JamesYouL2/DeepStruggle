"""Why doesn't the rollout root distil? (1) is its verdict consistent across seeds, (2) where do its departures
(and their measured leads) lie. Positions: greedy self-play of the network, every decision with >= 2 legal moves
sampled at rate p (all node types, as the training targets were). Each position searched R times, reseeded.
    PYTHONPATH=.:build/release python -m ai.eval.rollout_consistency <checkpoint> <games> <p> <repeats>

Findings: research/log/E7_search_depth_and_value.md section 6.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

import ts_engine as ts
from ai.search.pimcts import acting_player
from ai.search.rollout_root import RolloutConfig, RolloutTeacher
from bindings.ts_env import model_obs_features
from tools.lib.player_agent import NeuralAgent, load_agent


def collect(model: Any, dev: torch.device, games: int, p: float, rng: np.random.Generator) -> List[Any]:
    feats = model_obs_features(model)
    runner = ts.VectorizedBatchRunner(games, 515)
    if feats:
        runner.set_obs_features([feats] * games, [feats] * games)
    done = np.zeros(games, dtype=bool)
    out: List[Any] = []
    for _ in range(6000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32)).to(dev)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            lg = model(obs, torch.from_numpy(masks).to(dev))[0].float().cpu().numpy()
        acts = np.where(masks > 0, lg, -np.inf).argmax(axis=1)
        acts[done] = 0
        for i in np.flatnonzero(~done & (masks.sum(axis=1) >= 2)):
            if rng.random() < p:
                out.append(runner.get_state(int(i)).clone())
        runner.step_flat_all(acts.tolist(), True)
    return out


def describe(st: Any) -> Dict[str, str]:
    ctx = st.ctx()
    dt = ts.DecisionType(int(ctx.decision_type)).name
    if dt == "POINT_NODE":
        dt += "/COUP" if ctx.op_mode == ts.OpMode.COUP else ("/REALIGN" if ctx.op_mode == ts.OpMode.REALIGN else "/INFLUENCE")
    card = int(ctx.pending_op_card) or int(ctx.resolving_card)
    phase = "HEADLINE" if st.current_phase == ts.Phase.HEADLINE else ("SETUP" if st.current_phase == ts.Phase.SETUP else "AR")
    t = int(st.turn)
    era = "early(1-3)" if t <= 3 else ("mid(4-7)" if t <= 7 else "late(8-10)")
    side = "US" if acting_player(st) == ts.Player.US else "USSR"
    return {"type": dt, "phase": phase, "era": era, "defcon": f"DEFCON {int(st.defcon)}", "side": side,
            "card": (str(ts.CardData.get_card_info(card)["name"]) if card else "-")}


def main() -> None:
    path, games, p, R = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    agent = load_agent(path, device="cuda")
    if not isinstance(agent, NeuralAgent):
        raise SystemExit(f"{path} is not a network")
    model = agent.model.eval()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(7)
    states = collect(model, dev, games, p, rng)
    print(f"{len(states)} positions from {games} greedy games")
    teacher = RolloutTeacher(model, RolloutConfig(k=4, worlds=16, horizon=4, rule="z2"), device=dev)
    runs: List[List[Dict[str, Any]]] = []
    for r in range(R):
        teacher.root.reseed(1000 + 17 * r)
        res: List[Dict[str, Any]] = []
        for lo in range(0, len(states), 512):
            chunk = states[lo:lo + 512]
            tg = teacher.improved_policies(chunk)
            for st, stats, (acts, probs) in zip(chunk, teacher.root.last_stats, tg):
                cands = stats["candidates"]
                base = cands[0] if cands else None
                pick = base
                gain = 0.0
                se = 0.0
                if stats["worlds"] > 0:
                    leads = stats["lead"]
                    best = max(cands, key=lambda a: stats["q"][a])
                    m, s = leads[best]
                    if s > 0 and m / s >= 2.0 and best != base:
                        pick, gain, se = best, m, s
                res.append({"pick": pick, "base": base, "lead": gain, "se": se,
                            "acts": acts, "probs": probs, "searched": stats["worlds"] > 0})
        runs.append(res)
    # (1) consistency
    n = len(states)
    searched = [i for i in range(n) if runs[0][i]["searched"]]
    dep = [[runs[r][i]["pick"] != runs[r][i]["base"] for i in range(n)] for r in range(R)]
    print(f"\n(1) consistency over {R} reseeded searches, {len(searched)} searched positions")
    for r in range(R):
        print(f"  run {r}: departs from the network's move in {100*np.mean([dep[r][i] for i in searched]):.1f}%")
    pairs = [(a, b) for a in range(R) for b in range(R) if a < b]
    for a, b in pairs:
        da = [i for i in searched if dep[a][i]]
        same_pick = np.mean([runs[a][i]["pick"] == runs[b][i]["pick"] for i in da]) if da else float("nan")
        also_dep = np.mean([dep[b][i] for i in da]) if da else float("nan")
        tv: List[float] = []
        for i in searched:
            pa = dict(zip(runs[a][i]["acts"], runs[a][i]["probs"])); pb = dict(zip(runs[b][i]["acts"], runs[b][i]["probs"]))
            keys = set(pa) | set(pb)
            tv.append(0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in keys))
        top = np.mean([max(zip(runs[a][i]["probs"], runs[a][i]["acts"]))[1] == max(zip(runs[b][i]["probs"], runs[b][i]["acts"]))[1]
                       for i in searched])
        print(f"  runs {a} vs {b}: of run {a}'s departures, run {b} departs too in {100*also_dep:.0f}% and picks the same move in "
              f"{100*same_pick:.0f}%; targets' total variation {np.mean(tv):.3f} (median {np.median(tv):.3f}); "
              f"target top move agrees {100*top:.0f}%")
    # (2) where the departures lie (run 0), with leads
    print(f"\n(2) where run 0's departures lie (share of positions, share of departures, mean lead of the departure)")
    d0 = [i for i in searched if dep[0][i]]
    descs = [describe(states[i]) for i in range(n)]
    for key in ("type", "phase", "era", "defcon", "side"):
        allc = Counter(descs[i][key] for i in searched)
        depc = Counter(descs[i][key] for i in d0)
        lead: Dict[str, List[float]] = defaultdict(list)
        for i in d0:
            lead[descs[i][key]].append(runs[0][i]["lead"])
        print(f"  by {key}:")
        for k, c in allc.most_common():
            print(f"    {k:22s} positions {100*c/len(searched):5.1f}%  departs {100*depc[k]/max(1,c):5.1f}% of them  "
                  f"share of departures {100*depc[k]/max(1,len(d0)):5.1f}%  mean lead {np.mean(lead[k]) if lead[k] else 0:+.3f}")
    cardc = Counter(descs[i]["card"] for i in d0)
    print("  departures by card (top 15):", ", ".join(f"{k} {v}" for k, v in cardc.most_common(15)))
    stable = [i for i in d0 if all(runs[r][i]["pick"] == runs[0][i]["pick"] for r in range(R))]
    print(f"\n  departures stable in all {R} runs: {len(stable)} of {len(d0)}; their mean lead "
          f"{np.mean([runs[0][i]['lead'] for i in stable]) if stable else 0:+.3f} vs the unstable "
          f"{np.mean([runs[0][i]['lead'] for i in d0 if i not in set(stable)]) if len(stable) < len(d0) else 0:+.3f}")
    stc = Counter(descs[i]["type"] for i in stable)
    print("  stable departures by decision type:", ", ".join(f"{k} {v}" for k, v in stc.most_common()))


if __name__ == "__main__":
    main()
