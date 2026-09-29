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
