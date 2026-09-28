"""Packed matchups reproduce one-pairing-at-a-time play: the same deal seed per game, the same
seats, the same accounting. Deterministic agents (the heuristic bot, greedy networks) must give
the same per-pairing results game for game."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from ai.models.ladder_net import create_ladder_net
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.player_agent import HeuristicAgent, NeuralAgent

KEYS = ("a_wins", "b_wins", "draws", "a_wins_as_us", "a_losses_as_us", "a_wins_as_ussr",
        "a_losses_as_ussr", "avg_turn", "avg_ply", "avg_vp_margin_a", "causes_all")

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=32,
    card_self_attention=False, cross_attention=False, per_entity_heads=8, head_context=True,
    head_static=True, head_entities="country", identity_dim=0, card_lookup=False,
    card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0, drop_static=True,
    hidden_dim=32, num_res_blocks=1, num_attn_heads=4, categorical_value=False, head_center=True)


def _net(seed: int, name: str) -> NeuralAgent:
    torch.manual_seed(seed)
    m = create_ladder_net("cpu", **M2D).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g))
    return NeuralAgent(model=m, device=torch.device("cpu"), name=name)


def _compare(pairs: List[Tuple[Any, Any]], gps: int) -> None:
    single = [BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=gps, device="cpu",
                                                     temperature=0.0) for a, b in pairs]
    packed = BatchMatchRunner.play_packed_matchups(pairs, games_per_side=gps, device="cpu",
                                                   temperature=0.0)
    assert len(packed) == len(single)
    for s, p in zip(single, packed):
        assert (s["agent_a"], s["agent_b"]) == (p["agent_a"], p["agent_b"])
        for k in KEYS:
            assert s[k] == p[k], (s["agent_a"], s["agent_b"], k, s[k], p[k])


def test_greedy_networks_reproduce_one_pairing_at_a_time() -> None:
    a, b, c = _net(1, "net-a"), _net(2, "net-b"), _net(3, "net-c")
    _compare([(a, b), (a, c), (b, c)], gps=4)


def test_bots_and_networks_mixed() -> None:
    h = HeuristicAgent()
    a = _net(4, "net-d")
    _compare([(h, a), (a, h)], gps=3)


def test_the_pack_schedule_covers_every_pairing_once_with_few_agents_per_pack() -> None:
    from tools.tournament import _pack_schedule
    for M, pack in ((33, 25), (7, 25), (2, 25), (12, 9)):
        packs = _pack_schedule(M, pack)
        flat = [pr for p in packs for pr in p]
        assert sorted(flat) == [(i, j) for i in range(M) for j in range(i + 1, M)]
        assert all(i < j for i, j in flat)
        g = max(2, int(pack ** 0.5))
        for p in packs:
            agents = {x for pr in p for x in pr}
            assert len(agents) <= 2 * g or len(p) <= pack
