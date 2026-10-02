"""The workbench's paired playouts (web/ui/src/analysis/playouts.ts), on positions built here.

The page runs every candidate move from the same sampled worlds with the same dice and lets the
model play on; these tests hold its parts to what the Python probes do: the sampled world keeps
every count and every card the mover has seen (ai/search/dmcts.py `determinize`), an opponent's
chosen-but-unrevealed headline is forgotten, a battleground coup at DEFCON 2 counts as losing on
the spot, the engine is handed back byte-identical, the same move twice pairs to exactly zero, and
the same seed replays the same games. A stub network stands in for a model (its logits are a pure
function of the observation), so nothing here needs a checkpoint.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Tuple

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tests.web.js_runner import PUBLIC, run_ts
from tools.lib.game_step import drain_chance

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
NODE_OFFSET = 116


def _play_until(seed: int, stop: Callable[[ts.GameState], bool], limit: int = 3000) -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    drain_chance(st, context="test_page_playouts")
    rng = np.random.default_rng(seed)
    for _ in range(limit):
        if ts.Engine.is_terminal(st):
            break
        if stop(st):
            return st
        legal = [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]
        safe = [a for a in legal if not _loses(st, a)] or legal
        ts.Engine.step_flat(st, int(rng.choice(safe)))
        drain_chance(st, context="test_page_playouts")
    raise AssertionError("no such position reached")


def _loses(st: ts.GameState, a: int) -> bool:
    """Random play that blows the world up on turn 1 never reaches the positions these tests need."""
    probe = st.clone()
    mover = probe.ctx().decision_player
    ts.Engine.step_flat(probe, a)
    drain_chance(probe, context="test_page_playouts probe")
    if not ts.Engine.is_terminal(probe):
        return False
    u = float(ts.Engine.get_terminal_utility(probe))
    return (u < 0) if mover == ts.Player.US else (u > 0)


def _find(stop: Callable[[ts.GameState], bool], seeds: range = range(1, 200)) -> ts.GameState:
    for seed in seeds:
        try:
            return _play_until(seed, stop)
        except AssertionError:
            continue
    raise AssertionError("no seed reached such a position")


def _mid() -> ts.GameState:
    return _find(lambda s: s.turn >= 3 and s.current_phase == ts.Phase.ACTION_ROUND
                       and s.ctx().decision_type == ts.DecisionType.SELECT_CARD
                       and np.asarray(ActionEncoder.get_legal_mask(s)).sum() >= 3)


def _headline() -> ts.GameState:
    # The second side to choose a headline, with the first side's choice made but not revealed.
    return _find(lambda s: s.current_phase == ts.Phase.HEADLINE and s.headline_stage == 0
                       and s.ctx().decision_type == ts.DecisionType.SELECT_CARD
                       and (s.headline_us_card != 0 or s.headline_ussr_card != 0))


def _coup() -> Tuple[ts.GameState, int, int]:
    """A coup target choice at DEFCON 2 with a battleground and a non-battleground target legal."""
    bg = {i for i in range(84) if ts.MapData.get_country_info(i)["battleground"]}
    for seed in range(1, 300):
        try:
            st = _play_until(seed, lambda s: s.current_phase == ts.Phase.ACTION_ROUND and s.turn >= 4
                             and s.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
                             and bool(np.asarray(ActionEncoder.get_legal_mask(s))[MODE_BASE + 3]))
        except AssertionError:
            continue
        st.defcon = 2
        if not np.asarray(ActionEncoder.get_legal_mask(st))[MODE_BASE + 3]:
            continue
        ts.Engine.step_flat(st, MODE_BASE + 3)
        drain_chance(st, context="test_page_playouts")
        legal = [int(a) - NODE_OFFSET for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))
                 if NODE_OFFSET <= a < NODE_OFFSET + 84]
        b = [c for c in legal if c in bg]
        nb = [c for c in legal if c not in bg]
        if b and nb:
            return st, NODE_OFFSET + b[0], NODE_OFFSET + nb[0]
    raise AssertionError("no DEFCON-2 coup position found")


def test_page_playouts(tmp_path: Any) -> None:
    coup, coup_bg, coup_non_bg = _coup()
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"mid": _mid().to_save_json(), "headline": _headline().to_save_json(),
                                "coup": coup.to_save_json(), "coup_bg": coup_bg, "coup_non_bg": coup_non_bg}))
    out = run_ts("page_playouts.ts", str(tmp_path), PUBLIC, str(case))

    assert out["world"] == {"counts_kept": True, "seen_kept": True, "shuffled": True}
    assert out["headline"] == {"had_choice": True, "forgotten": True}
    assert out["suicide"] == {"bg_loses": True, "non_bg_safe": True, "engine_back": True}

    p = out["playouts"]
    assert p["engine_back"]
    assert p["pairs"] == 6 and p["games"] == 18 and p["capped"] == 0
    assert p["same_move_diff"] == 0 and p["same_move_se"] == 0     # the same move, the same worlds and dice
    assert p["scores_in_range"] and p["reproducible"]
    assert p["progress"] == [4, 6]                                  # one report per wave
    assert p["early_pairs"] < 12                                    # stopping ends the run early
