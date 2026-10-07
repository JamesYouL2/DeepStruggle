"""The search disagreement bank (tools/search_bank.py): a stored position reproduces its legal moves
and the raw network's logits when reloaded, every searcher's record is consistent with its own
choice, and the Gumbel root's diagnostic hooks leave its play unchanged.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from bindings.action_encoder import ActionEncoder
from tools.scripts.event_play_census import load_policy, state_from_token
from tools.search_bank import legal_actions, main, mover_of, parse_budgets, raw_rank, read_rows


def _checkpoint(tmp_path: Path) -> str:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    path = tmp_path / "m.pt"
    torch.save(m.state_dict(), str(path))
    return str(path)


def _states(n: int = 16, advance: int = 60) -> List[ts.GameState]:
    runner = ts.VectorizedBatchRunner(n, 4242)
    for _ in range(advance):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    return [runner.get_state(i) for i in range(n)
            if not ts.Engine.is_terminal(runner.get_state(i)) and len(legal_actions(runner.get_state(i))) >= 2]


def test_raw_rank_and_budgets() -> None:
    assert raw_rank([0.1, 2.0, -1.0], [5, 9, 12], 9) == 1
    assert raw_rank([0.1, 2.0, -1.0], [5, 9, 12], 12) == 3
    assert parse_budgets(["g16=16:4", "g256=256:8"]) == [("g16", 16, 4), ("g256", 256, 8)]


def test_annotated_positions_reload_to_the_same_moves_and_logits(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path)
    out = str(tmp_path / "bank.jsonl.gz")
    assert main(["annotate", "--model", ckpt, "--games", "1", "--sample", "16",
                 "--budgets", "g8=8:2", "g16=16:4", "--out", out]) == 0
    rows = read_rows([out])
    assert len(rows) >= 5
    logits_fn, features = load_policy(ckpt)
    meta = json.load(open(out + ".meta.json"))
    assert meta["schema"] == 1 and meta["budgets"]["g16"] == {"simulations": 16, "k": 4}
    for r in rows:
        st = state_from_token(r["pos"])
        assert legal_actions(st) == r["legal"]
        obs = np.asarray(ts.extract_observation_features(st, mover_of(st), features), dtype=np.float32)[None]
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)[None]
        lg = logits_fn(obs, mask)[0]
        assert np.allclose([lg[a] for a in r["legal"]], r["logits"], atol=1e-3)
        assert r["raw"] == r["legal"][int(np.argmax(r["logits"]))]
        for label, m in r["methods"].items():
            assert m["choice"] in r["legal"]
            assert m["choice"] in m["candidates"] and m["dropped"][str(m["choice"])] is None
            assert set(m["candidates"]) <= set(r["legal"])
            assert str(m["choice"]) in r["names"]


def test_resume_skips_finished_games(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path)
    out = str(tmp_path / "bank.jsonl.gz")
    args = ["annotate", "--model", ckpt, "--games", "2", "--sample", "32", "--budgets", "g8=8:2", "--out", out]
    main(args[:4] + ["1"] + args[5:])                       # the first game only
    n1 = sum(1 for _ in gzip.open(out, "rt"))
    main(args + ["--resume"])
    games = [json.loads(line)["game"] for line in open(out + ".games.jsonl")]
    assert games == [0, 1]
    assert sum(1 for _ in gzip.open(out, "rt")) > n1


def test_the_gumbel_hooks_leave_play_unchanged_and_forced_candidates_are_searched() -> None:
    states = _states()
    torch.manual_seed(0)
    model = create_coldwar_net_v2("cpu").eval()
    cfg = BatchedMCTSConfig(simulations=16, determinize=True, gumbel_k=4, gumbel_scale=0.0, seed=7)
    a = BatchedMCTS(model, device="cpu", config=cfg, featurise_capacity=256)
    b = BatchedMCTS(model, device="cpu", config=cfg, featurise_capacity=256)
    plain = a.best_actions(states)
    assert b._gumbel is not None
    same = b._gumbel.choose(states, candidates=[None] * len(states))
    assert plain == same
    stats = b._gumbel.last_stats
    assert all(len(s["candidates"]) == min(4, len(legal_actions(st))) for s, st in zip(stats, states))
    # Forcing each position's least probable legal move into the candidates puts it there.
    forced = []
    for s, st in zip(stats, states):
        worst = [x for x in legal_actions(st) if x not in s["candidates"]]
        forced.append(list(s["candidates"])[:3] + worst[:1] if worst else None)
    b.reseed(7)
    b._gumbel.choose(states, candidates=forced)
    for s, f in zip(b._gumbel.last_stats, forced):
        if f is not None:
            assert sorted(s["candidates"]) == sorted(f)
