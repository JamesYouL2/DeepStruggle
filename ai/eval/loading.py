"""The one way a probe loads a checkpoint: the network with the view it decides in.

Kept apart from `ai.eval.playout` because it imports `tools.lib.player_agent` (the model zoo, the
searchers, onnxruntime), which the probes the trainer imports must not drag in.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import torch

from bindings.ts_env import AgentView
from tools.lib.player_agent import NeuralAgent


class EvalModel(NamedTuple):
    """A loaded network in eval mode, its observation feature set and its action view."""

    model: Any
    features: int
    merged_influence: bool

    @property
    def view(self) -> AgentView:
        return AgentView(self.features, self.merged_influence)


def load_for_eval(path: str, device: Union[torch.device, str], *, merged_ok: bool) -> EvalModel:
    """Load `path` (architecture from its weights, action view from its run directory) for a probe.

    `merged_ok` is required: a probe that builds every mask in the E4 view must say so with False,
    and a merged-view (E4.1, P23) checkpoint is then refused here -- loudly, instead of being
    measured as a policy it is not.
    """
    agent = NeuralAgent.from_checkpoint(path, device=device)
    if agent.merged_influence and not merged_ok:
        raise ValueError(
            f"{path} decides in the merged-influence (E4.1) view and this probe builds its masks in "
            f"the E4 view only; it would measure a policy the checkpoint is not")
    return EvalModel(agent.model.eval(), int(agent.obs_features), bool(agent.merged_influence))
