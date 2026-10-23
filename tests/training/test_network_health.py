"""The early stability checks (`ai.eval.network_health`): the rank and geometry measures read what they claim, the
logit statistics group by decision type, a network with a trunk norm reports its stream, and the training-log mode
summarises windows."""

import json
from pathlib import Path

import torch

from ai.eval.network_health import effective_rank, geometry, log_windows, logit_stats, measure
from ai.models.ladder_net import create_ladder_net

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=2, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True, res_dropout=0.0)


def test_effective_rank_reads_the_number_of_directions_used() -> None:
    torch.manual_seed(0)
    iso = torch.randn(4096, 32)
    assert effective_rank(iso) > 30
    two = torch.randn(4096, 2) @ torch.randn(2, 32)
    assert effective_rank(two) < 2.2
    g = geometry(torch.cat([torch.randn(4096, 31), 50 * torch.randn(4096, 1)], dim=1))
    assert g["max_over_median_dim_rms"] > 40 and g["effective_rank"] < 3


def test_logit_stats_group_by_decision_type_and_ignore_masked_actions() -> None:
    raw = torch.tensor([[0.0, 10.0, -1e9], [0.0, 1000.0, 5.0]])
    masks = torch.tensor([[1, 1, 0], [1, 1, 1]])
    s = logit_stats(raw, masks, ["SELECT_CARD", "SELECT_PLAY_MODE"])
    assert abs(s["SELECT_CARD"]["spread_max"] - 10.0) < 1e-4
    assert abs(s["SELECT_PLAY_MODE"]["spread_max"] - 1000.0) < 1e-3
    assert s["all"]["n"] == 2 and s["all"]["lse_absmax"] >= 1000.0


def test_measure_reports_the_stream_under_a_trunk_norm() -> None:
    torch.manual_seed(1)
    obs, masks = torch.randn(64, 3824), torch.ones(64, 220, dtype=torch.bool)
    normed = measure(create_ladder_net("cpu", **{**CFG, "trunk_norm": True}), obs, masks, ["SELECT_CARD"] * 64)
    assert "stream_before_trunk_norm" in normed and normed["head_input"]["width"] == 64
    plain = measure(create_ladder_net("cpu", **{**CFG, "num_res_blocks": 0}), obs, masks, ["SELECT_CARD"] * 64)
    assert "stream_before_trunk_norm" not in plain
    assert set(plain["head_norm_over_fresh"]) >= {"policy_head", "play_mode_head_net", "branch_head_net"}


def test_log_windows_summarise_each_window(tmp_path: Path) -> None:
    rows = [{"total_steps": s * 10_000_000, "logit_lse_absmax": float(s), "kl_div": 0.01 * s, "entropy": 1.0,
             "explained_variance": 0.8} for s in range(1, 11)]
    (tmp_path / "training_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    w = log_windows(str(tmp_path), 50)
    assert [r["from_m"] for r in w] == [0.0, 50.0]
    assert w[0]["logit_lse_absmax"] == 5.0 and w[1]["logit_lse_absmax"] == 10.0
    assert abs(w[1]["kl_div"] - 0.10) < 1e-9 and w[0]["entropy"] == 1.0
