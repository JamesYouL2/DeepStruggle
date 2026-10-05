"""The doctrine census: how often the model follows a strong player's card rules (`ai/eval/doctrine_census.py`).

    # one part: 500 greedy self-play games from seed 1
    PYTHONPATH=.:build/release .venv/bin/python tools/doctrine_census.py run \\
        --model data/checkpoints/E6-06-44@soup_680-760.onnx --games 500 --seed 1 --out parts/part-1.json.gz
    # pool the parts
    ... report --parts parts/part-*.json.gz --out-md report.md --out-json report.json

On CI: `.github/workflows/doctrine_census.yml`.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np
import ts_engine as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.eval.doctrine_census import PolicyFn, StatePolicyFn, collect, report  # noqa: E402


def greedy_policy(model_path: str) -> "tuple[PolicyFn, int]":
    """The model's greedy batched policy and its observation feature set. An `.onnx` export reads
    the base observation; a `.pt` checkpoint is run in torch with the features in its weights."""
    if model_path.endswith(".pt"):
        import torch
        from bindings.ts_env import model_obs_features
        from tools.lib.player_agent import NeuralAgent
        agent = NeuralAgent.from_checkpoint(model_path, device="cpu")
        net = agent.model.eval()

        def act_pt(obs, masks):
            with torch.no_grad():
                lg, _v, _vp = net(torch.from_numpy(np.ascontiguousarray(obs[:, :agent.obs_size])),
                                  torch.from_numpy(np.ascontiguousarray(masks)).bool())
            return lg.argmax(dim=1).numpy().astype(np.int32)
        return act_pt, model_obs_features(net)
    from tools.lib.player_agent import OnnxAgent
    onnx = OnnxAgent(model_path)
    return (lambda obs, masks: onnx.act_batch(obs, masks, 0.0, True)), 0


def search_policy(spec: str, search_at: str = "all") -> "tuple[StatePolicyFn, int]":
    """A `search:` agent spec (tools/lib/player_agent.load_agent), deciding from the states: the
    census of what the searcher plays, for comparison with its network's. `search_at="doctrine"`
    searches only the decisions the card rules are about (doctrine_decision) and plays the rest
    with the searcher's own network, greedily -- so a very large budget stays affordable."""
    from ai.eval.doctrine_census import doctrine_decision
    from bindings.action_encoder import ActionEncoder
    from bindings.ts_env import model_obs_features
    from tools.lib.player_agent import BatchSelector, load_agent
    agent = load_agent(spec, device="cpu")
    if not isinstance(agent, BatchSelector):
        raise SystemExit(f"{spec!r} does not decide batches of states")
    feats = model_obs_features(getattr(agent, "mcts").model)
    if search_at == "all":
        return agent.select_actions_batch, feats
    if search_at != "doctrine":
        raise SystemExit(f"--search-at {search_at!r}: 'all' or 'doctrine'")
    plain = getattr(agent, "_policy_actions")

    def act(states: "list[ts.GameState]") -> "list[int]":
        out = [0] * len(states)
        hot = [i for i, st in enumerate(states)
               if doctrine_decision(st, np.asarray(ActionEncoder.get_legal_mask(st)))]
        hot_set = set(hot)
        cold = [i for i in range(len(states)) if i not in hot_set]
        act.searched += len(hot)                    # type: ignore[attr-defined]
        for idx, picks in ((hot, agent.select_actions_batch([states[i] for i in hot]) if hot else []),
                           (cold, plain([states[i] for i in cold]) if cold else [])):
            for i, a in zip(idx, picks):
                out[i] = int(a)
        return out
    act.searched = 0                                # type: ignore[attr-defined]
    return act, feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True,
                   help="an .onnx or .pt file, or a search: agent spec (tools/lib/player_agent.py)")
    r.add_argument("--games", type=int, default=500)
    r.add_argument("--search-at", default="all", choices=["all", "doctrine"],
                   help="with a search: model, which decisions it searches: every one, or only those "
                        "the card rules are about (the rest by its network, greedily)")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--envs", type=int, default=32)
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-json", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        t0 = time.time()
        if a.model.startswith("search:"):
            act_states, feats = search_policy(a.model, a.search_at)
            data = collect(lambda o, m: m.argmax(1), a.games, a.seed, envs=a.envs,
                           obs_features=feats, act_states=act_states)
            print(f"searched decisions: {getattr(act_states, 'searched', 'all')}", flush=True)
            name = a.model.replace("data/checkpoints/", "")
        else:
            act, feats = greedy_policy(a.model)
            data = collect(act, a.games, a.seed, envs=a.envs, obs_features=feats)
            name = os.path.basename(a.model)
        data["meta"] = {"model": name, "seed": a.seed, "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump(data, f)
        print(f"{data['games']} games, {len(data['plays'])} plays, {data['meta']['seconds']}s")
    else:
        pooled = {"plays": [], "headlines": [], "coups": [], "boards": [], "games": 0}
        model = None
        for fn in sorted({f for pat in a.parts for f in glob.glob(pat)}):
            with gzip.open(fn, "rt") as f:
                part = json.load(f)
            if model not in (None, part["meta"]["model"]):
                raise SystemExit(f"{fn} was played by {part['meta']['model']}, not {model}")
            model = part["meta"]["model"]
            seed = part["meta"]["seed"]
            for k in ("plays", "headlines", "coups", "boards"):
                for row in part[k]:
                    row["game"] = f"{seed}:{row['game']}"
                pooled[k] += part[k]
            pooled["games"] += part["games"]
        md, summary = report(pooled, {"model": model})
        with open(a.out_md, "w") as f:
            f.write(md)
        with open(a.out_json, "w") as f:
            json.dump(summary, f)
        print(md)


if __name__ == "__main__":
    main()
