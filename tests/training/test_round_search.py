"""Dense round search: mechanics only, with a stub network so it needs no GPU and no checkpoint."""

from __future__ import annotations

import numpy as np
import torch

import ts_engine as ts
from ai.search.round_search import RoundSearchAgent, RoundSearchConfig
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle


class _StubNet(torch.nn.Module):
    """Uniform policy; value is the (deterministic) sum of the first obs slice, so lines differ."""

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs, mask):
        logits = torch.zeros(obs.shape[0], mask.shape[1])
        v = torch.tanh(obs[:, :512].sum(dim=1, keepdim=True) * 1e-3)
        return logits, v, None


def _opening_round_state() -> ts.GameState:
    s = ts.GameState()
    ts.Engine.init_game(s, 7)
    settle(s, SettleMode.FORCED)
    while s.current_phase != ts.Phase.ACTION_ROUND:
        mask = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
        ts.Engine.step_flat(s, int(mask[0]))
        settle(s, SettleMode.FORCED)
    return s


def _late_round_state() -> ts.GameState:
    """A small round (a few thousand leaves) that fits under the cap without truncating."""
    s = ts.GameState()
    ts.Engine.init_game(s, 7)
    settle(s, SettleMode.FORCED)
    while not ts.Engine.is_terminal(s) and not (
            s.current_phase == ts.Phase.ACTION_ROUND and s.action_round >= 6
            and int(s.ctx().decision_type) == int(ts.DecisionType.SELECT_CARD)):
        mask = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
        ts.Engine.step_flat(s, int(mask[0]))
        settle(s, SettleMode.FORCED)
    return s


def _agent(k: int) -> RoundSearchAgent:
    cfg = RoundSearchConfig(reply_top_k=k, max_leaves=3000, eval_batch=512)
    return RoundSearchAgent(_StubNet(), device=torch.device("cpu"), config=cfg)


def test_returns_legal_action_and_does_not_touch_caller_state() -> None:
    s = _opening_round_state()
    before = bytes(s.raw_bytes())
    for k in (0, 1):
        a = _agent(k).select_action(s, None)
        assert bool(np.asarray(ActionEncoder.get_legal_mask(s))[a])
    assert bytes(s.raw_bytes()) == before


def test_reply_stage_scores_the_candidate() -> None:
    s = _late_round_state()
    cfg = RoundSearchConfig(reply_top_k=1, max_leaves=200_000, eval_batch=1024)
    ag = RoundSearchAgent(_StubNet(), device=torch.device("cpu"), config=cfg)
    ag.select_action(s, None)
    assert ag.last is not None and len(ag.last.v_reply) == 1
    assert ag.last.action in ag.last.v_reply


def test_truncation_is_counted() -> None:
    s = _opening_round_state()
    ag = _agent(0)  # a real round exceeds 3,000 leaves
    ag.select_action(s, None)
    assert ag.search.truncations >= 1


def test_follows_its_own_plan_within_a_round() -> None:
    s = _opening_round_state()
    ag = _agent(0)
    a = ag.select_action(s, None)
    assert ag.search.searches == 1
    ts.Engine.step_flat(s, a)
    settle(s, SettleMode.FORCED)
    # The first action of a round is a card; the next decision (play mode, targets) is still the
    # searcher's own and on the searched line, so it is answered from the plan without a search.
    legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
    if len(legal) > 1 and ag.search.planned(s) is not None:
        b = ag.select_action(s, None)
        assert ag.search.searches == 1 and ag.search.plan_hits == 1
        assert bool(np.asarray(ActionEncoder.get_legal_mask(s))[b])


def test_ranked_path_with_nothing_pruned_equals_the_exhaustive_search() -> None:
    """A beam wider than any layer forces the network-ranked code path without dropping a node."""
    s = _late_round_state()
    exact = RoundSearchAgent(_StubNet(), device=torch.device("cpu"), config=RoundSearchConfig(
        reply_top_k=0, max_leaves=200_000, eval_batch=1024)).search.search(s)
    ranked = RoundSearchAgent(_StubNet(), device=torch.device("cpu"), config=RoundSearchConfig(
        reply_top_k=0, max_leaves=200_000, eval_batch=1024, beam=10**9)).search.search(s)
    assert exact is not None and ranked is not None
    assert exact.v_round == ranked.v_round and exact.action == ranked.action


def test_pruning_keeps_the_top_policy_action_and_shrinks_the_search() -> None:
    s = _late_round_state()
    full = RoundSearchAgent(_StubNet(), device=torch.device("cpu"), config=RoundSearchConfig(
        reply_top_k=0, max_leaves=200_000, eval_batch=1024)).search.search(s)
    cut = RoundSearchAgent(_StubNet(), device=torch.device("cpu"), config=RoundSearchConfig(
        reply_top_k=0, max_leaves=200_000, eval_batch=1024, max_branch=2)).search.search(s)
    assert full is not None and cut is not None
    assert cut.leaves < full.leaves and len(cut.v_round) <= 2


def test_spec_options_set_any_config_field_and_name_the_entrant() -> None:
    from ai.search.round_search import round_search_config

    cfg, label = round_search_config(["2", "p_own=0.9", "beam=32", "max_leaves=20000"])
    assert (cfg.reply_top_k, cfg.p_own, cfg.beam, cfg.max_leaves) == (2, 0.9, 32, 20000)
    assert isinstance(cfg.beam, int) and not cfg.determinize
    assert label == "k2-p_own0.9-beam32-max_leaves20000"
    # The old positional form still reads as it did.
    assert round_search_config([])[1] == "k8"
    cfg, label = round_search_config(["4", "determinize"])
    assert (cfg.reply_top_k, cfg.determinize, label) == (4, True, "k4-det")
    cfg, label = round_search_config(["follow_plan=false"])
    assert cfg.follow_plan is False and label == "k8-follow_planfalse"


def test_spec_options_refuse_what_they_cannot_read() -> None:
    import pytest

    from ai.search.round_search import round_search_config

    for bad in (["p_owm=0.9"], ["beam=wide"], ["follow_plan=maybe"], ["2", "determinize", "3"]):
        with pytest.raises(ValueError):
            round_search_config(bad)


def test_a_pruned_tree_plays_a_tournament_game_from_an_agent_spec(tmp_path) -> None:
    """Smoke: `roundsearch:` with pruning options loads an ONNX export and finishes real games.

    The CI tournament enters round search the same way, so this is the local half of that check.
    """
    import json

    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from tools.export_onnx import export
    from tools.lib.batch_tournament import BatchMatchRunner
    from tools.lib.player_agent import load_agent

    run = tmp_path / "T-01-01_20260101_000000"
    run.mkdir()
    torch.manual_seed(7)
    ckpt = run / "snapshot_1000000steps.pt"
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), ckpt)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))
    onnx = run / "model.onnx"
    export(str(ckpt), str(onnx), positions=32)

    agent = load_agent(f"roundsearch:{onnx}:0:max_branch=3:beam=16:max_leaves=2000:eval_batch=512",
                       device="cpu")
    assert isinstance(agent, RoundSearchAgent)
    assert agent.name == "roundsearch-k0-max_branch3-beam16-max_leaves2000-eval_batch512"
    assert (agent.search.cfg.max_branch, agent.search.cfg.beam) == (3, 16)
    res = BatchMatchRunner.play_parallel_matchup(
        agent, load_agent("random", device="cpu"), games_per_side=1, device=torch.device("cpu"),
        temperature=0.0)
    assert res["total_games"] == 2
    assert agent.search.searches > 0
