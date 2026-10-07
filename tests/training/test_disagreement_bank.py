"""The human-marked disagreement bank (`ai/eval/banks/disagreement_verdicts.jsonl`) holds on the
current engine: every position loads, asks the decision it was reviewed as, and every mark names a
move that is legal there -- so `bank_verdicts.py score` measures a model against what the reviewer
said, not against moves an engine or converter change has made unreachable. And the tools that make
the bank read the network's value on its own scale, refuse an agent in another action view, and
seed and pool playouts so that a position's result does not depend on how a run was split.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.paired_playouts import ar_key, decider, play_safe
from tools.lib.corpus_driver import require_e4_view, state_from_token
from tools.lib.game_step import drain_chance
from tools.scripts.bank_playouts import pool, position_seed
from tools.scripts.bank_verdicts import load, legal_names
from tools.scripts.disagreement_bank import KINDS, PATTERNS, kind_of, row_id, win_probability


def test_the_bank_holds_one_row_per_position_under_its_own_id() -> None:
    rows = load()
    assert len(rows) >= 100
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert row_id(r) == r["id"]
        assert r["kind"] in KINDS and r["pattern"] in PATTERNS


def test_every_mark_is_a_legal_move_of_the_decision_reviewed() -> None:
    marked = 0
    for r in load():
        st = state_from_token(r["pos"])
        assert kind_of(st) == r["kind"], r["id"]
        legal = legal_names(st)
        for move in (r["human"], r["network"], r["search"], *r["marks"]):
            assert move in legal, (r["id"], move)
        assert set(r["marks"].values()) <= {"good", "bad"}
        assert not set(r["marks"]) & set(r["stale_marks"])
        marked += bool(r["marks"])
    assert marked >= 70


def test_the_network_win_chance_is_read_off_the_value_scale() -> None:
    # v_win regresses the mover's result on [-1, +1]; it is neither a probability nor a logit.
    assert np.allclose(win_probability(np.array([-1.0, -0.2, 0.0, 0.5, 1.0, 1.3])),
                       [0.0, 0.4, 0.5, 0.75, 1.0, 1.0])


def test_a_merged_view_agent_is_refused() -> None:
    class _Merged:
        merged_influence = True

    with pytest.raises(ValueError, match="merged-influence"):
        require_e4_view("model.onnx", _Merged())
    require_e4_view("model.onnx", object())


def test_pool_names_the_parts_that_did_not_arrive(tmp_path: Path) -> None:
    parts = []
    for k in (1, 3):
        path = tmp_path / f"playouts-{k}.jsonl.gz"
        with gzip.open(path, "wt") as f:
            f.write(json.dumps({"id": f"{k:016x}"}) + "\n")
        parts.append(str(path))
    out = str(tmp_path / "playouts.jsonl.gz")
    assert pool(parts, out, expect=3) == 0
    with gzip.open(out, "rt") as f:
        assert len(f.readlines()) == 2
    with open(out + ".parts.json") as f:
        assert json.load(f) == {"expected": 3, "found": [1, 3], "missing": [2]}


def test_a_playout_that_has_not_ended_is_not_scored() -> None:
    st = ts.GameState()
    ts.Engine.init_game(st, 1)
    drain_chance(st, context="test")

    def first_legal(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return masks.argmax(axis=1)

    with pytest.raises(RuntimeError, match="had not ended"):
        play_safe([st], [decider(st)], [ar_key(st)], first_legal, 0, max_steps=3)


def test_a_position_is_seeded_by_its_id_and_the_run_seed_alone() -> None:
    rid = "05ff4bf1da8d6ad2"
    assert position_seed(rid, 0) != position_seed(rid, 1)
    assert position_seed(rid, 0) != position_seed("39c132b5205eef58", 0)
    assert 0 <= position_seed("ffffffffffffffff", 7) < 2 ** 63
