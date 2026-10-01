"""The branch oracle on a real replay, played by a deterministic stub instead of a network."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import ts_engine as ts

from ai.eval.branch_oracle import apply_prefix, play_branches, position_before, report
from bindings.action_encoder import ActionEncoder
from tools.lib.self_play import generate_self_play_replay


class FirstLegal(nn.Module):
    """Plays the lowest legal flat action: deterministic, and enough to produce a real game."""

    TOTAL_OBS_SIZE = int(ts.OBS_SIZE)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        n = mask.shape[-1]
        logits = -torch.arange(n, dtype=torch.float32).expand(mask.shape[0], n).clone()
        logits[mask == 0] = -1e9
        z = torch.zeros(mask.shape[0], 1)
        return logits, z, z

    def sample_action(self, obs: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0,
                      deterministic: bool = False):
        logits, v, vp = self.forward(obs, mask)
        a = logits.argmax(dim=-1)
        zero = torch.zeros(a.shape[0])
        return a, zero, v.squeeze(-1), vp.squeeze(-1), zero


def first_legal(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return masks.argmax(axis=1).astype(np.int32)


@pytest.fixture(scope="module")
def replay(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("bo") / "g.tslog.json")
    generate_self_play_replay(FirstLegal(), seed=11, temperature=0.0, output_path=path,
                              device="cpu", verbose=False, trace=False)
    return path


def _a_choice_step(replay: str) -> int:
    import json
    steps = json.loads(Path(replay).read_text())["steps"]
    for s in steps[30:]:
        st, _, _ = position_before(replay, int(s["step_index"]))
        if int(np.asarray(ActionEncoder.get_legal_mask(st)).sum()) >= 3:
            return int(s["step_index"])
    raise AssertionError("no decision with three options")


def test_position_before_returns_the_recorded_decision(replay: str) -> None:
    step = _a_choice_step(replay)
    st, played, _ = position_before(replay, step)
    assert ActionEncoder.get_legal_mask(st)[played]


def test_an_alternative_identical_to_the_policy_scores_identically(replay: str) -> None:
    step = _a_choice_step(replay)
    st, played, _ = position_before(replay, step)
    greedy = int(np.asarray(ActionEncoder.get_legal_mask(st)).argmax())
    other = int(np.flatnonzero(ActionEncoder.get_legal_mask(st))[-1])
    rows = play_branches(st, {"policy": [], "same": [greedy], "other": [other]}, range(6),
                         first_legal, seed=3)
    _, js = report(rows)
    assert js["same"]["diff"] == 0.0 and js["same"]["se"] == 0.0     # pairing is exact
    assert all(r["finished"] for r in rows)


def test_an_illegal_prefix_is_refused(replay: str) -> None:
    step = _a_choice_step(replay)
    st, _, _ = position_before(replay, step)
    illegal = int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)) == 0)[0])
    with pytest.raises(ValueError):
        apply_prefix(st, [illegal])
