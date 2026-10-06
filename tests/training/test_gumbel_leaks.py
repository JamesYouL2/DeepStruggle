"""The leak probe (ai/eval/gumbel_leaks.py): positions from the model's own games in its own
observation view, the Gumbel root's departures priced by paired playouts, and a report whose
leaks open in the workbench at their positions."""

from __future__ import annotations

import base64
import json
import math
import zlib
from typing import Any, Dict

import pytest
import torch

import ts_engine as ts
from ai.eval import gumbel_leaks as L
from ai.eval.target_forms import paired_advantage
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.models.ladder_net import LadderNet
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from ai.search.gumbel_root import GumbelRoot


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _feature_model() -> torch.nn.Module:
    """A small LadderNet reading the OPS_BUDGET block: the base observation alone is refused."""
    torch.manual_seed(1)
    cfg: Dict[str, Any] = dict(hidden_dim=64, num_res_blocks=1, entity_proj_dim=32, num_attn_heads=4,
                               categorical_value=False, input_mode="entity", aggregation="flatten",
                               entity_dim=8, card_self_attention=False, cross_attention=False,
                               per_entity_heads=0, head_context=True, head_static=True,
                               head_entities="both", identity_dim=0, drop_static=False,
                               card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                               card_lookup_identity_dim=0, obs_features=int(ts.OBS_FEATURE_OPS_BUDGET))
    m = LadderNet(**cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def sample():
    return L.collect_positions(_model(), 10, seed=3, games=8)


def test_positions_are_real_choices_from_whole_games(sample) -> None:
    positions, per_game, games = sample
    assert len(positions) == 10 and games > 0 and per_game > 10
    for p in positions:
        assert len(p.legal) > 1
        assert p.state.current_phase != ts.Phase.SETUP
        assert math.isclose(sum(p.prior.values()), 1.0, rel_tol=1e-6)
    assert len({p.turn for p in positions}) > 1


def test_a_model_with_a_feature_block_is_played_in_its_own_view() -> None:
    """The base observation is the wrong width for this model and would raise; sampling, the
    root and the playouts all have to hand it its own feature set."""
    m = _feature_model()
    positions, _per_game, _games = L.collect_positions(m, 4, seed=2, games=4)
    picks, stats = L.gumbel_choices(m, positions, L.RootSpec(sims=8, k=3), seed=1)
    assert all(a in p.legal for a, p in zip(picks, positions)) and len(stats) == 4
    items = [(p, p.legal[0], p.legal[0]) for p in positions[:2]]
    assert all(r == (0.0, 0.0) for r in paired_advantage(m, items, pairs=2, seed=5, batch=16))


def test_the_root_records_its_evidence_without_changing_its_choice(sample) -> None:
    positions, _, _ = sample
    cfg = BatchedMCTSConfig(simulations=12, temperature=0.0, auto_advance=True, advance_root=False,
                            determinize=True, node_filter="all", subsample=1.0, seed=9,
                            fpu_reduction=0.2, gumbel_k=4, gumbel_scale=0.0)
    states = [p.state for p in positions]
    plain = GumbelRoot(BatchedMCTS(_model(), config=cfg)).choose(states)
    stats: list = []
    recorded = GumbelRoot(BatchedMCTS(_model(), config=cfg)).choose(states, stats)
    assert plain == recorded and len(stats) == len(states)
    for p, pick, st in zip(positions, recorded, stats):
        cands = {c["action"]: c for c in st["candidates"]}
        assert len(cands) == min(4, len(p.legal))
        net = max(p.prior, key=lambda a: p.prior[a])
        assert net in cands                                   # noise-free: the top logits
        assert cands[pick]["dropped_in_phase"] is None
        assert sum(c["dropped_in_phase"] is None for c in cands.values()) == 1
        assert math.isclose(st["value_mover"], p.value_mover, abs_tol=1e-5)


def test_one_candidate_never_departs(sample) -> None:
    positions, _, _ = sample
    picks, _ = L.gumbel_choices(_model(), positions, L.RootSpec(sims=8, k=1), seed=1)
    assert picks == [max(p.prior, key=lambda a: p.prior[a]) for p in positions]


def test_the_position_token_opens_the_same_position(sample) -> None:
    """The workbench's `pos=`: base64url without padding of the zlib-deflated save JSON."""
    st = sample[0][0].state
    tok = L.position_token(st)
    raw = zlib.decompress(base64.urlsafe_b64decode(tok + "=" * (-len(tok) % 4)))
    back = ts.state_from_save_json(raw.decode("utf-8"))
    assert back.to_save_json() == st.to_save_json()
    assert "=" not in tok and "+" not in tok and "/" not in tok


def test_a_part_end_to_end_and_its_pooled_report(tmp_path) -> None:
    rows, meta = L.run_part(_model(), 12, seed=4, pairs=2, spec=L.RootSpec(sims=8, k=4),
                            games=8, log=lambda _s: None)
    assert len(rows) == 12 and meta["games_counted"] > 0
    deps = [r for r in rows if r["depart"]]
    for r in deps:
        assert r["gumbel"] != r["net"] and {"adv", "se", "token", "net_name", "gumbel_name"} <= set(r)
        assert -1.0 <= r["adv"] <= 1.0
    path = tmp_path / "part-1.jsonl"
    L.dump(rows, dict(meta, model="m.pt"), str(path))
    back, metas = L.load([str(path), str(path)])
    assert len(back) == 24 and len(metas) == 2
    md = L.report(back, metas, workbench="http://wb/", workbench_model="hf:o/r@main:m.onnx")
    assert "leak per decision" in md and "By decision segment" in md and "Event leaks" in md
    if deps:
        assert "http://wb/?pos=" in md and "model=hf:o/r@main:m.onnx" in md
    json.dumps(back)                                          # rows stay plain JSON


def test_the_report_prices_leaks_per_decision() -> None:
    """Departure rate x mean advantage, in per mille: twenty positions, one departure worth +0.5
    (half a game turned) -> 0.5 / 20 = 25‰ a decision, SE 0.1 / 20 = 5‰."""
    dep = {"segment": "play_mode", "turn": 3, "mover": 1, "n_legal": 3, "net": 1, "net_p": 0.9,
           "depart": True, "gumbel": 2, "gumbel_p": 0.1, "net_name": "a", "gumbel_name": "b",
           "net_mode": "Influence", "gumbel_mode": "Event", "net_card": 5, "card": 5,
           "value_mover": 0.0, "search_gap": 0.1, "adv": 0.5, "se": 0.1, "token": "x"}
    same = {"segment": "ar_card", "turn": 3, "mover": 1, "n_legal": 3, "net": 1, "net_p": 0.9,
            "depart": False}
    rows = [dep] + [dict(same) for _ in range(19)]
    md = L.report(rows, [{"choices_per_game": 4.0, "games_counted": 10, "model": "m",
                          "root": "gumbel32-k4-fpu0.2", "pairs": 2, "temperature": 0.1}])
    assert "**+25.00 ± 5.00‰**" in md                       # leak per decision and its SE
    assert "search plays the event (net: Influence)" in md
    assert "mode: " in md and "Influence → Event" in md
