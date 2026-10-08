"""The search disagreement bank (tools/search_bank.py): a stored position reproduces its legal moves
and the raw network's logits when reloaded, every searcher's record is consistent with its own
choice, and the Gumbel root's diagnostic hooks leave its play unchanged.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from bindings.action_encoder import ActionEncoder
from tools.scripts.event_play_census import load_policy, state_from_token
from tools.search_bank import legal_actions, main, mover_of, parse_budgets, raw_rank, read_rows


def _checkpoint(tmp_path: Path) -> str:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    path = tmp_path / "m.pt"
    torch.save(m.state_dict(), str(path))
    return str(path)


def _states(n: int = 16, advance: int = 60) -> List[ts.GameState]:
    runner = ts.VectorizedBatchRunner(n, 4242)
    for _ in range(advance):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    return [runner.get_state(i) for i in range(n)
            if not ts.Engine.is_terminal(runner.get_state(i)) and len(legal_actions(runner.get_state(i))) >= 2]


def test_raw_rank_and_budgets() -> None:
    assert raw_rank([0.1, 2.0, -1.0], [5, 9, 12], 9) == 1
    assert raw_rank([0.1, 2.0, -1.0], [5, 9, 12], 12) == 3
    assert parse_budgets(["g16=16:4", "g256=256:8"]) == [("g16", 16, 4), ("g256", 256, 8)]


def test_annotated_positions_reload_to_the_same_moves_and_logits(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path)
    out = str(tmp_path / "bank.jsonl.gz")
    assert main(["annotate", "--model", ckpt, "--games", "1", "--sample", "16",
                 "--budgets", "g8=8:2", "g16=16:4", "--out", out]) == 0
    rows = read_rows([out])
    assert len(rows) >= 5
    logits_fn, features = load_policy(ckpt)
    meta = json.load(open(out + ".meta.json"))
    assert meta["schema"] == 1 and meta["budgets"]["g16"] == {"simulations": 16, "k": 4}
    for r in rows:
        st = state_from_token(r["pos"])
        assert legal_actions(st) == r["legal"]
        obs = np.asarray(ts.extract_observation_features(st, mover_of(st), features), dtype=np.float32)[None]
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)[None]
        lg = logits_fn(obs, mask)[0]
        assert np.allclose([lg[a] for a in r["legal"]], r["logits"], atol=1e-3)
        assert r["raw"] == r["legal"][int(np.argmax(r["logits"]))]
        for label, m in r["methods"].items():
            assert m["choice"] in r["legal"]
            assert m["choice"] in m["candidates"] and m["dropped"][str(m["choice"])] is None
            assert set(m["candidates"]) <= set(r["legal"])
            assert str(m["choice"]) in r["names"]


def test_resume_skips_finished_games(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path)
    out = str(tmp_path / "bank.jsonl.gz")
    args = ["annotate", "--model", ckpt, "--games", "2", "--sample", "32", "--budgets", "g8=8:2", "--out", out]
    main(args[:4] + ["1"] + args[5:])                       # the first game only
    n1 = sum(1 for _ in gzip.open(out, "rt"))
    main(args + ["--resume"])
    games = [json.loads(line)["game"] for line in open(out + ".games.jsonl")]
    assert games == [0, 1]
    assert sum(1 for _ in gzip.open(out, "rt")) > n1


def test_the_gumbel_hooks_leave_play_unchanged_and_forced_candidates_are_searched() -> None:
    states = _states()
    torch.manual_seed(0)
    model = create_coldwar_net_v2("cpu").eval()
    cfg = BatchedMCTSConfig(simulations=16, determinize=True, gumbel_k=4, gumbel_scale=0.0, seed=7)
    a = BatchedMCTS(model, device="cpu", config=cfg, featurise_capacity=256)
    b = BatchedMCTS(model, device="cpu", config=cfg, featurise_capacity=256)
    plain = a.best_actions(states)
    assert b._gumbel is not None
    same = b._gumbel.choose(states, candidates=[None] * len(states))
    assert plain == same
    stats = b._gumbel.last_stats
    assert all(len(s["candidates"]) == min(4, len(legal_actions(st))) for s, st in zip(stats, states))
    # Forcing each position's least probable legal move into the candidates puts it there.
    forced = []
    for s, st in zip(stats, states):
        worst = [x for x in legal_actions(st) if x not in s["candidates"]]
        forced.append(list(s["candidates"])[:3] + worst[:1] if worst else None)
    b.reseed(7)
    b._gumbel.choose(states, candidates=forced)
    for s, f in zip(b._gumbel.last_stats, forced):
        if f is not None:
            assert sorted(s["candidates"]) == sorted(f)


def _row(i: int, raw: int, choices: List[int], q_choice: float, dt: str = "POINT_NODE") -> dict:
    return {"id": f"{i:016x}", "raw": raw, "turn": 1 + i % 10, "ar": 1, "phase": "ACTION_ROUND",
            "decision_type": dt, "card": 0, "legal": [raw, 7, 8], "logits": [2.0, 1.0, 0.0],
            "methods": {f"g{j}": {"choice": c, "candidates": [raw, 7, 8], "q": {str(raw): 0.0, str(c): q_choice}}
                        for j, c in enumerate(choices)}}


def test_select_weights_reproduce_each_strata_population() -> None:
    from tools.search_bank import select, stratum
    rows = [_row(i, 5, [5, 5], 0.0) for i in range(300)] + [_row(1000 + i, 5, [7, 5], 0.1) for i in range(90)]
    bank = select(rows, size=60, control=20, seed=1)
    assert len(bank) == 60
    assert sum(r["stratum"].startswith("agree") for r in bank) == 20
    pop: dict = {}
    for r in rows:
        k = "/".join(stratum(r))
        pop[k] = pop.get(k, 0) + 1
    got: dict = {}
    for r in bank:
        got[r["stratum"]] = got.get(r["stratum"], 0.0) + r["weight"]
    for k, w in got.items():
        assert abs(w - pop[k]) < 1e-2          # weights are stored to 4 decimals


def test_regret_is_measured_on_worlds_the_best_move_was_not_chosen_on() -> None:
    from tools.search_bank import solve
    bank = {"raw": 5, "legal": [5, 7], "logits": [1.0, 0.0], "methods": {"g": {"choice": 7}}}
    # Move 7 is better on the even worlds only: chosen there, it shows no regret advantage on the odd ones.
    ref = {"values": {"5": [0.0, 0.1] * 8, "7": [0.2, 0.1] * 8}, "critic": {"5": [0.0] * 16, "7": [0.1] * 16},
           "reference": {"action": 7, "agreement": 1.0}}
    v = solve(bank, ref)
    assert v["pick"] == 7 and v["best"] == 7 and v["rank"] == 2
    assert abs(v["regret"]["raw"][0]) < 1e-9
    assert v["critic_best"] == 7


def _play_mode_row(i: int, card: int, side: str, legal: List[int], logits: List[float]) -> dict:
    return {"id": f"{i:016x}", "decision_type": "SELECT_PLAY_MODE", "phase": "ACTION_ROUND", "card": card,
            "side": side, "turn": 4, "legal": legal, "logits": logits,
            "raw": legal[max(range(len(legal)), key=lambda j: logits[j])]}


def test_event_moves_play_the_event_against_raws_alternative_and_space_on_an_opponent_card() -> None:
    from tools.search_bank import EVENT, SPACE, event_moves, is_event_decision
    infl, coup = EVENT + 2, EVENT + 3
    own = _play_mode_row(0, 5, "US", [EVENT, SPACE, infl, coup], [2.0, 0.5, 1.0, 0.0])
    assert event_moves(own, "own") == [EVENT, infl]          # raw events; the alternative is its best other play
    spaced = _play_mode_row(1, 5, "US", [EVENT, SPACE, infl], [0.0, 3.0, 1.0])
    assert event_moves(spaced, "own") == [EVENT, SPACE]      # Space is an alternative on one's own card
    opp = _play_mode_row(2, 5, "USSR", [EVENT, SPACE, infl, coup], [0.0, 3.0, 1.0, 2.0])
    assert event_moves(opp, "opponent") == [EVENT, coup, SPACE]
    for r, rel in ((own, "own"), (spaced, "own"), (opp, "opponent")):
        assert r["raw"] in event_moves(r, rel)
    assert not is_event_decision(dict(own, phase="HEADLINE"))
    assert not is_event_decision(_play_mode_row(3, 6, "US", [SPACE, infl], [0.0, 1.0]))   # no event (China Card)


def test_event_select_caps_each_card_and_side_and_weights_back_to_its_population() -> None:
    from tools.search_bank import EVENT, event_select
    cards = {5: {"side": "US"}, 9: {"side": "neutral"}}
    rows = ([_play_mode_row(i, 5, "US", [EVENT, EVENT + 2], [0.0, 1.0]) for i in range(50)]
            + [_play_mode_row(100 + i, 5, "USSR", [EVENT, EVENT + 2], [0.0, 1.0]) for i in range(7)]
            + [_play_mode_row(200 + i, 9, "US", [EVENT, EVENT + 2], [0.0, 1.0]) for i in range(12)])
    bank = event_select(iter(rows), cards, cap=10, seed=3)
    by: dict = {}
    for r in bank:
        by.setdefault(r["stratum"], []).append(r)
    assert {k: len(v) for k, v in by.items()} == {"5/US": 10, "5/USSR": 7, "9/US": 10}
    assert {k: round(sum(r["weight"] for r in v)) for k, v in by.items()} == {"5/US": 50, "5/USSR": 7, "9/US": 12}
    assert {r["relation"] for r in by["5/USSR"]} == {"opponent"} and by["9/US"][0]["relation"] == "neutral"


def test_event_regret_is_measured_on_pairs_the_best_move_was_not_picked_on() -> None:
    from tools.search_bank import EVENT, event_verdict
    infl = EVENT + 2
    b = {"raw": infl, "pmoves": [EVENT, infl], "legal": [EVENT, infl], "logits": [0.0, 1.0]}
    # The event wins the even pairs only: picked there, it shows no lead on the odd ones.
    v = event_verdict(b, {str(EVENT): [1.0, 0.0] * 8, str(infl): [0.0, 0.0] * 8})
    assert v["pick"] == EVENT and v["kind"] == "missed_event" and abs(v["regret"]) < 1e-9
    assert abs(v["gain"] - 50.0) < 1e-9
    # Better on every pair: a missed event worth the whole lead.
    v = event_verdict(b, {str(EVENT): [1.0] * 16, str(infl): [0.0] * 16})
    assert v["kind"] == "missed_event" and abs(v["regret"] - 100.0) < 1e-9
    w = event_verdict(dict(b, raw=EVENT), {str(EVENT): [0.0] * 16, str(infl): [1.0] * 16})
    assert w["kind"] == "wrong_event" and abs(w["regret"] - 100.0) < 1e-9 and w["raw_event"]


def test_subs_select_keeps_us_subs_decisions_with_raws_best_non_event_play() -> None:
    from tools.search_bank import EVENT, SPACE, is_subs_decision, subs_select
    infl, coup = EVENT + 2, EVENT + 3
    r = _play_mode_row(0, 41, "US", [EVENT, SPACE, infl, coup], [3.0, 0.0, 1.0, 2.0])
    assert is_subs_decision(r)
    assert subs_select([r])[0]["alt"] == coup                 # the best play other than the event
    assert not is_subs_decision(dict(r, side="USSR"))
    assert not is_subs_decision(_play_mode_row(1, 42, "US", [EVENT, infl], [0.0, 1.0]))


def _us_play_mode_state() -> ts.GameState:
    """A US action-round play-mode decision, reached by the first legal action at every step."""
    st = ts.GameState()
    ts.Engine.init_game(st, 7)
    for _ in range(5000):
        ctx = st.ctx()
        if (st.current_phase == ts.Phase.ACTION_ROUND and ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
                and ctx.decision_player == ts.Player.US
                and np.asarray(ActionEncoder.get_legal_mask(st))[ActionEncoder.PLAY_MODE_OFFSET + 3]):
            return st
        ts.Engine.step_flat(st, int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))[0]))
    raise AssertionError("no US play-mode decision with a coup reached")


def test_subs_watch_steers_to_a_battleground_coup_only_under_subs_and_counts_it() -> None:
    from ai.eval.subs_followup import OPS_COUP, SubsWatch, bg_coup_open, is_us_coup_target
    st = _us_play_mode_state()
    mask = np.asarray(ActionEncoder.get_legal_mask(st)).copy()
    w = SubsWatch([int(st.turn)], ["next"])
    assert (w.steer(0, st, mask) == mask).all()                # Subs not in effect: untouched
    st.persistent_effects |= ts.EffectBits.NUCLEAR_SUBS_ACTIVE
    assert bg_coup_open(st)
    forced = w.steer(0, st, mask)
    assert list(np.flatnonzero(forced)) == [OPS_COUP]
    ts.Engine.step_flat(st, OPS_COUP)
    assert is_us_coup_target(st)
    tmask = np.asarray(ActionEncoder.get_legal_mask(st)).copy()
    targets = np.flatnonzero(w.steer(0, st, tmask))
    assert len(targets) and all(ts.MapData.get_country_info(int(a) - ActionEncoder.NODE_OFFSET)["battleground"]
                                for a in targets)
    w.seen(0, st, int(targets[0]))
    assert w.counts[0]["coups"] == 1 and w.counts[0]["bg_coups"] == 1 and w.counts[0]["forced"] == 1
    assert w.left[0] == 0                                      # "next" forces one coup only
