"""The root prior temperature and the value-based final pick of BatchedMCTS."""
import numpy as np
import pytest
import torch
import torch.nn as nn
import ts_engine as ts

from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig, _BNode
from tools.lib.player_agent import load_agent


class Sharp(nn.Module):
    """Puts almost all prior mass on the lowest legal action; value 0 everywhere."""

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        n = mask.shape[-1]
        logits = torch.zeros(mask.shape[0], n)
        first = mask.float().argmax(dim=-1)
        logits[torch.arange(mask.shape[0]), first] = 8.0
        logits[mask == 0] = -1e9
        z = torch.zeros(mask.shape[0], 1)
        return logits, z, z


def _root(mover: ts.Player, n, w, priors) -> _BNode:
    return _BNode(state=ts.GameState(), mover=int(mover), terminal=False, actions=[10, 20, 30],
                  priors=list(priors), n=list(map(float, n)), w=list(map(float, w)), expanded=True)


def _searcher(**kw) -> BatchedMCTS:
    return BatchedMCTS(Sharp(), device="cpu", config=BatchedMCTSConfig(**kw))


def test_value_select_overrules_visits_above_the_floor() -> None:
    # q (US view): 0.1, 0.4, 1.0 -- the third is best but has 4 visits, under the floor of 8.
    root = _root(ts.Player.US, n=[50, 10, 4], w=[5, 4, 4], priors=[0.9, 0.05, 0.05])
    assert _searcher(select="visits")._choose(root) == 0
    assert _searcher(select="value", value_min_visits=8)._choose(root) == 1
    assert _searcher(select="value", value_min_visits=4)._choose(root) == 2


def test_value_select_reads_the_value_from_the_mover_side() -> None:
    root = _root(ts.Player.USSR, n=[50, 10, 4], w=[5, 4, 4], priors=[0.9, 0.05, 0.05])
    assert _searcher(select="value", value_min_visits=8)._choose(root) == 0   # lowest US value


def test_value_select_falls_back_to_visits_when_nothing_clears_the_floor() -> None:
    root = _root(ts.Player.US, n=[5, 2, 1], w=[0, 2, 1], priors=[0.9, 0.05, 0.05])
    assert _searcher(select="value", value_min_visits=8)._choose(root) == 0


def test_root_temperature_flattens_the_root_priors_only() -> None:
    st = ts.GameState()
    ts.Engine.init_game(st, 3)
    assert int(np.asarray(ts.ActionMask.generate_flat_mask(st)).sum()) >= 2
    plain = _searcher(simulations=4)._search([st.clone()])[0]
    warm = _searcher(simulations=4, root_prior_temp=2.0)._search([st.clone()])[0]
    assert plain is not None and warm is not None
    assert max(warm.priors) < max(plain.priors)
    assert abs(sum(warm.priors) - 1.0) < 1e-9
    assert np.argmax(warm.priors) == np.argmax(plain.priors)        # the order is kept


def test_an_unknown_search_option_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown search option"):
        load_agent("search:model.onnx:64:determinize:bogus=1", device="cpu")
