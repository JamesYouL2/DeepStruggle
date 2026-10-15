"""P33's critic diagnostics: ai/eval/leak_critic.py (the critic on labelled move pairs) and
ai/eval/critic_boundary.compare_report (the critic on greedy against sampled games)."""
from __future__ import annotations

import gzip
import json
import pathlib
from typing import Any, Dict, List

import numpy as np
import torch

import ts_engine as ts
from ai.eval import leak_critic
from ai.eval.critic_boundary import Decision, Game, compare_report
from ai.models.ladder_net import create_ladder_net
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import position_token

_LADDER: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=32,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=32, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _positions(n: int = 6) -> List[ts.GameState]:
    """Mid-game positions with at least two legal moves, from a fixed deal and fixed moves."""
    runner = ts.VectorizedBatchRunner(n, 777)
    for _ in range(50):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    out = []
    for i in range(n):
        st = runner.get_state(i)
        if not ts.Engine.is_terminal(st) and int(np.asarray(ActionEncoder.get_legal_mask(st)).sum()) >= 2:
            out.append(st)
    return out


def _bank(tmp: pathlib.Path, states: List[ts.GameState]) -> Dict[str, str]:
    """A bank, confirmation playouts and screening playouts over `states`: the player's move is the
    first legal one, the other the second; the labels are fixed numbers."""
    bank, confirm, screen = [], [], []
    for i, st in enumerate(states):
        legal = [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]
        played, other = legal[0], legal[1]
        rid = f"row{i}"
        bank.append({"id": rid, "pos": position_token(st), "played": played, "weight": 1.0 + i,
                     "decision_type": "SELECT_CARD" if i % 2 else "POINT_NODE", "side": "US" if i % 2 else "USSR"})
        confirm.append({"id": rid, "scores": {str(played): [0.0, 1.0] * 4, str(other): [1.0] * 8}})
        # a screened pair signed against the played move, and one with no signal
        screen.append({"id": rid, "scores": {str(played): [1.0] * 8, str(other): [0.0, 0.0, 1.0, 0.0] * 2,
                                             **({str(legal[2]): [1.0] * 7 + [0.0]} if len(legal) > 2 else {})}})
    paths = {}
    for name, rows in (("bank", bank), ("confirm", confirm), ("screen", screen)):
        p = tmp / f"{name}.jsonl.gz"
        with gzip.open(p, "wt") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        paths[name] = str(p)
    return paths


def test_pairs_carry_the_played_move_the_other_and_a_paired_label(tmp_path: pathlib.Path) -> None:
    states = _positions()
    paths = _bank(tmp_path, states)
    pairs = leak_critic.load_pairs(paths["bank"], paths["confirm"])
    assert len(pairs) == len(states)
    assert all(abs(p.label - 0.5) < 1e-9 for p in pairs)            # 1.0 against a mean of 0.5
    screened = leak_critic.load_screen_pairs(paths["bank"], paths["screen"], min_z=2.0)
    # only the clearly signed pair of each row survives (0.25 against 1.0); the coin-flip one does not
    assert len(screened) == len(states) and all(abs(p.label + 0.75) < 1e-9 for p in screened)


def test_critic_differences_read_both_moves_in_the_movers_frame(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **_LADDER).eval()
    states = _positions()
    pairs = leak_critic.load_pairs(**{"bank_path": _bank(tmp_path, states)["bank"],
                                      "playouts_path": _bank(tmp_path, states)["confirm"]})
    raw = leak_critic.critic_differences(net, "cpu", pairs, dice=2)
    settled = leak_critic.critic_differences(net, "cpu", pairs, dice=2, settled=True)
    assert raw.shape == settled.shape == (len(pairs),)
    assert np.all(np.abs(raw) <= 2.0) and np.all(np.abs(settled) <= 2.0)
    # the same move on both sides of a pair is no difference at all
    same = [leak_critic.LeakPair(p.id, p.state, p.mover, p.played, p.played, 0.0, 1.0, p.decision_type,
                                 p.defcon, p.turn, p.side) for p in pairs]
    assert np.allclose(leak_critic.critic_differences(net, "cpu", same, dice=2), 0.0, atol=1e-6)
    text = leak_critic.summarise("net", pairs, raw)
    assert "critic ranks wrong" in text and "SELECT_CARD" in text


def test_compare_report_reads_the_first_temperature_against_the_last() -> None:
    def games(bias: float) -> List[Game]:
        out = []
        for g in range(40):
            us_won = g % 2 == 0
            dec = [Decision(1, (0.6 if us_won else -0.6) + bias, 9, "ACTION_ROUND", 2),
                   Decision(-1, (-0.6 if us_won else 0.6) + bias, 9, "ACTION_ROUND", 2)]
            out.append(Game(decisions=dec, us_util=1.0 if us_won else -1.0))
        return out
    text = compare_report({0.0: games(0.2), 1.0: games(0.0)})
    row = next(line for line in text.splitlines() if line.startswith("DEFCON 2, turns 8-10"))
    assert "+0.1000" in row.split("  ")[1] or "+0.1000" in row       # greedy games biased by 0.1 in p
    assert "(+" in row.rsplit("  ", 2)[-2]                          # |bias| larger at T=0
