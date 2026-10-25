"""The search disagreement bank (tools/search_bank.py): a stored position reproduces its legal moves
and the raw network's logits when reloaded, every searcher's record is consistent with its own
choice, and the Gumbel root's diagnostic hooks leave its play unchanged.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.eval.paired_playouts import ar_key, decider, play_safe
from ai.eval.subs_followup import OPS_COUP, SubsWatch, bg_coup_open, is_us_coup_target
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import load_policy, selfplay, selfplay_seeded, state_from_token
from tools.scripts.checkpoint_report import check_leak, leak_section
from tools.search_bank import (CHINA, EVENT, SPACE, _paired_scores, china_select, china_verdict, confirm_select,
                               event_moves, event_select, event_verdict, is_china_decision, is_event_decision,
                               is_subs_decision, kind_bootstrap, leak_analysis, leak_summary, legal_actions, main,
                               mover_of,
                               parse_budgets, playout_select, raw_rank, read_rows, screen_select, select, solve,
                               stratum, subs_select)


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
    rows = [_row(i, 5, [5, 5], 0.0) for i in range(300)] + [_row(1000 + i, 5, [7, 5], 0.1) for i in range(90)]
    bank = select(rows, size=60, control=20, seed=1)
    assert len(bank) == 60
    assert sum(r["stratum"].startswith("agree") for r in bank) == 20
    pop: dict = {}
    for r in rows:
        k = stratum(r).label
        pop[k] = pop.get(k, 0) + 1
    got: dict = {}
    for r in bank:
        got[r["stratum"]] = got.get(r["stratum"], 0.0) + r["weight"]
    for k, w in got.items():
        assert abs(w - pop[k]) < 1e-2          # weights are stored to 4 decimals


def test_regret_is_measured_on_worlds_the_best_move_was_not_chosen_on() -> None:
    bank = {"raw": 5, "legal": [5, 7], "logits": [1.0, 0.0], "methods": {"g": {"choice": 7}}}
    # Move 7 is better on the even worlds only: chosen there, it shows no regret advantage on the odd ones.
    ref = {"values": {"5": [0.0, 0.1] * 8, "7": [0.2, 0.1] * 8}, "critic": {"5": [0.0] * 16, "7": [0.1] * 16},
           "reference": {"action": 7, "agreement": 1.0}}
    v = solve(bank, ref)
    assert v["pick"] == 7 and v["best"] == 7 and v["rank"] == 2
    assert abs(v["regret"]["raw"][0]) < 1e-9
    assert v["critic_best"] == 7


def _verdict_pair(i: int, raw_value: float, alt_value: float, g256: int) -> tuple:
    """A bank row (raw 5; 7 then 8 by logit) and its reference: raw's move and move 7 valued flat
    over 16 worlds."""
    bank = {"id": f"{i:016x}", "raw": 5, "legal": [5, 7, 8], "logits": [2.0, 1.0, 0.0], "weight": 2.0,
            "methods": {"g256": {"choice": g256}}}
    ref = {"id": bank["id"], "values": {"5": [raw_value] * 16, "7": [alt_value] * 16},
           "critic": {"5": [0.0] * 16, "7": [0.0] * 16}, "reference": {"action": 7, "agreement": 1.0}}
    return bank, ref


def test_playout_select_takes_every_large_regret_samples_the_rest_and_weights_by_inclusion() -> None:
    big = [_verdict_pair(i, 0.0, 0.1, 7) for i in range(5)]               # raw loses 0.1: large
    ctrl = [_verdict_pair(100 + i, 0.1, 0.0, 5) for i in range(40)]       # raw best, k=8 @256 agrees
    rest = [_verdict_pair(200 + i, 0.0, 0.01, 8) for i in range(30)]      # raw loses 0.01: small
    pairs = big + ctrl + rest
    out = playout_select([b for b, _ in pairs], [r for _, r in pairs], rest=10, control=8, seed=3)
    by = {c: [r for r in out if r["pl_class"] == c] for c in ("big", "rest", "ctrl")}
    assert [len(by[c]) for c in ("big", "rest", "ctrl")] == [5, 10, 8]
    assert {r["pl_incl"] for r in by["big"]} == {1.0}
    assert {r["pl_incl"] for r in by["rest"]} == {10 / 30} and {r["pl_incl"] for r in by["ctrl"]} == {8 / 40}
    assert all(r["pmoves"] == [5, 7] and abs(r["regret"] - 0.1) < 1e-9 for r in by["big"])
    assert all(r["pmoves"] == [5, 7, 8] for r in by["rest"])           # raw, the best, k=8 @256's
    assert all(r["pmoves"] == [5, 7] and r["best"] == 5 for r in by["ctrl"])   # raw against raw's second


def test_leak_weights_each_rows_paired_gain_back_to_the_population() -> None:
    # Best beats raw by 0.2 in every pair; the row stands for 10 population decisions (weight 1,
    # sampled with probability 0.1) of a bank weighing 100, in games of 50 decisions:
    # 100 points x 50 / 100 x 10 x 0.2 = +100.0 a game.
    gain = _paired_scores({"5": [0.0, 0.5, 0.5, 1.0], "7": [0.2, 0.7, 0.7, 1.2]}, 7, 5)
    assert abs(gain[0] - 0.2) < 1e-9 and gain[1] < 1e-12
    assert _paired_scores({"5": [0.0]}, 5, 5) == (0.0, 0.0)
    row = {"id": "x", "w": 10.0, "regret": 0.1, "gain": gain, "g256": (0.1, 0.0), "kind": "AR card choice",
           "side": "US", "turn": 9, "defcon": 2, "sub": "opp 2 -> own 3", "raw_card": "A", "best_card": "B",
           "b": {"turn": 9, "ar": 3, "raw": 5, "best": 7, "names": {"5": "A", "7": "B"}}}
    md = leak_analysis([row], bank_weight=100.0, decisions_per_game=50.0)
    assert "| AR card choice | 1 | 10.00% | +100.0 ± 0.0 | +50.0 ± 0.0 |" in md


def _leak_row(i: int, kind: str, gain: float, escalated: bool) -> dict:
    return {"id": str(i), "w": 10.0, "regret": 0.0, "gain": (gain, 0.0001), "g256": (0.0, 0.0), "kind": kind,
            "side": "USSR", "turn": 5, "defcon": 3, "sub": "", "raw_card": "A", "best_card": "B",
            "b": {"turn": 5, "ar": 2, "raw": 5, "best": 7, "names": {"5": "A", "7": "B"}, "escalated": escalated}}


def test_leak_summary_carries_the_census_identity_and_checkpoint_report_checks_and_renders_it(
        tmp_path: Path) -> None:
    # Screened census (large 0): one escalated row per kind, one row not escalated (no loss); bank 100,
    # 50 decisions a game, so a row of weight 10 gaining 0.2 is +100.0 a game.
    rows = [_leak_row(0, "AR card choice", 0.2, True), _leak_row(1, "headline", 0.1, True),
            _leak_row(2, "headline", 0.0, False)]
    pt = tmp_path / "net.pt"
    pt.write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    meta = {"model": "net.pt", "model_sha256": sha, "player": "rollout:4:32:4:z2:earlier:own", "games": 1500}
    lk = leak_summary(rows, bank_weight=100.0, decisions_per_game=50.0, large=0, meta=meta)
    assert lk["census"] == "screened" and lk["player"] == meta["player"] and lk["games"] == 1500
    assert abs(lk["found_share"] - 0.2) < 1e-9 and abs(lk["total"]["leak"] - 150.0) < 1e-9
    assert [k["kind"] for k in lk["kinds"]] == ["AR card choice", "headline"]
    assert abs(lk["kinds"][0]["share"] - 2 / 3) < 1e-9 and not lk["bootstrap"]
    assert lk["top"][0]["played"] == "A" and abs(lk["top"][0]["a_game"] - 100.0) < 1e-9
    agent = f"rollout:{pt}:4:32:4:z2:earlier:own"
    assert check_leak(lk, str(pt), agent) is None
    assert "studied player" in str(check_leak(lk, str(pt), None))           # the greedy network's report
    assert "studied player" in str(check_leak(lk, str(pt), f"rollout:{pt}:4:64:4:z2:earlier:own"))
    other = tmp_path / "other.pt"
    other.write_bytes(b"other weights")
    assert "census ran on" in str(check_leak(lk, str(other), agent.replace(str(pt), str(other))))
    md = leak_section(json.loads(json.dumps(lk)))            # as the records dump stores it
    assert "loses **+150.0 ± " in md and "| AR card choice | +100.0 ± " in md
    assert "| T5 AR2 | USSR | 3 | AR card choice | A | B | +20.0 ± 1.0 | +100.00 |" in md


def _play_mode_row(i: int, card: int, side: str, legal: List[int], logits: List[float]) -> dict:
    return {"id": f"{i:016x}", "decision_type": "SELECT_PLAY_MODE", "phase": "ACTION_ROUND", "card": card,
            "side": side, "turn": 4, "legal": legal, "logits": logits,
            "raw": legal[max(range(len(legal)), key=lambda j: logits[j])]}


def test_event_moves_play_the_event_against_raws_alternative_and_space_on_an_opponent_card() -> None:
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


def test_a_state_selector_plays_the_continuation_and_reproduces_the_policy_it_wraps() -> None:
    def first(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([int(np.flatnonzero(m)[0]) for m in masks], dtype=np.int32)

    def last(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([int(np.flatnonzero(m)[-1]) for m in masks], dtype=np.int32)

    starts = _states(4)
    movers, keys = [decider(s) for s in starts], [ar_key(s) for s in starts]
    seen: List[int] = []

    def select_first(states: Sequence[ts.GameState]) -> List[int]:
        seen.append(len(states))
        return [legal_actions(s)[0] for s in states]

    plain = play_safe(starts, movers, keys, first, 5)
    # `act` is only the fallback for a choice outside a narrowed mask, so the selector decides
    assert play_safe(starts, movers, keys, last, 5, select=select_first) == plain
    assert seen


def test_china_bank_plays_the_china_card_against_raws_choice_or_its_best_other_card() -> None:
    def row(i: int, raw: int, legal: List[int], logits: List[float], ar: int = 2) -> dict:
        return {"id": f"r{i}", "decision_type": "SELECT_CARD", "phase": "ACTION_ROUND", "side": "US",
                "ar": ar, "turn": 4, "legal": legal, "logits": logits, "raw": raw}

    other = row(1, 40, [CHINA, 40, 60], [0.0, 2.0, 1.0])
    eager = row(2, CHINA, [CHINA, 40, 60], [3.0, 1.0, 2.0])
    assert is_china_decision(other) and not is_china_decision(row(3, CHINA, [CHINA, 208], [0.0, 0.0]))
    bank = {r["id"]: r for r in china_select([other, eager] + [dict(other, id=f"x{i}") for i in range(3)], 2, 0)}
    assert bank["r2"]["pmoves"] == [60, CHINA]                 # raw plays China: its best other card
    sample = [r for r in bank.values() if r["stratum"].endswith("raw other")]
    assert len(sample) == 2 and all(r["weight"] == 2.0 for r in sample)   # 4 in the stratum, 2 drawn
    v = china_verdict(bank["r2"], {"60": [1.0, 1.0, 1.0, 1.0], str(CHINA): [0.0, 0.0, 0.0, 0.0]})
    assert v["gain"] == -100.0 and v["kind"] == "too eager" and v["regret"] == 100.0


def test_kind_bootstrap_resamples_within_cells() -> None:
    """A kind's leak is scale x its weighted gains; identical rows in a cell resample to the same
    total (no spread), and a kind that leaks most in every resample is the largest in all of them."""
    def row(kind: str, cell: str, gain: float, g256: float, w: float = 1.0) -> dict:
        return {"kind": kind, "w": w, "gain": (gain, 0.0), "g256": (g256, 0.0),
                "b": {"stratum": cell, "pl_class": "big"}}

    rows = [row("placement", "a", 0.10, 0.02), row("placement", "a", 0.10, 0.02),
            row("card", "b", 0.03, 0.01), row("card", "b", 0.03, 0.01)]
    bs = kind_bootstrap(rows, scale=10.0, resamples=200)
    assert bs["placement"]["leak"] == pytest.approx(2.0) and bs["card"]["leak"] == pytest.approx(0.6)
    assert bs["all"]["leak"] == pytest.approx(2.6) and bs["all"]["g256"] == pytest.approx(0.6)
    assert bs["placement"]["leak_se"] == pytest.approx(0.0, abs=1e-9) and bs["placement"]["largest"] == 1.0
    # Two different rows in one cell: the resampled totals spread.
    noisy = kind_bootstrap([row("placement", "a", 0.0, 0.0), row("placement", "a", 0.2, 0.0)], 10.0, 200)
    assert noisy["placement"]["leak"] == pytest.approx(2.0) and noisy["placement"]["leak_se"] > 0.5


def test_a_kind_filtered_screen_leaves_the_other_rows_on_raws_move_unescalated() -> None:
    def row(i: str, dt: str, phase: str) -> dict:
        return {"id": i, "decision_type": dt, "phase": phase, "raw": 3, "legal": [1, 3, 5],
                "logits": [0.1, 0.5, 0.2], "methods": {"g256": {"choice": 1}}, "weight": 1.0}
    rows = screen_select([row("a", "SELECT_CARD", "ACTION_ROUND"), row("h", "SELECT_CARD", "HEADLINE"),
                          row("p", "POINT_NODE", "ACTION_ROUND")], decision="SELECT_CARD", phase="ACTION_ROUND")
    assert [r["pmoves"] for r in rows] == [[3, 5, 1], [3], [3]]
    run = {"a": {"scores": {"3": [0.0] * 8, "5": [1.0, 0.9] * 4, "1": [0.0] * 8}}}
    got = {r["id"]: r for r in confirm_select(rows, [run])}
    assert got["a"]["escalated"] and got["a"]["best"] == 5 and got["a"]["pmoves"] == [3, 5]
    assert not got["h"]["escalated"] and got["h"]["best"] == 3 and got["h"]["pmoves"] == [3]


class _Moves:
    def __init__(self) -> None:
        self.moves: List[int] = []

    def observe(self, state: "ts.GameState", action: int) -> None:
        self.moves.append(int(action))


def test_games_played_together_are_the_games_played_one_at_a_time(tmp_path: Path) -> None:
    """annotate --games-at-once: each game starts from the deal its one-game run starts from, so a
    deterministic policy plays the same moves together as alone."""
    logits_fn, features = load_policy(_checkpoint(tmp_path))
    seeds = [700_003, 700_009, 700_015]
    alone = [selfplay(logits_fn, features, 1, s, 1, 0.0, _Moves)[0].moves for s in seeds]
    together = [t.moves for t in selfplay_seeded(logits_fn, features, seeds, 0.0, [_Moves] * len(seeds))]
    assert together == alone
    assert len({tuple(m) for m in alone}) == len(seeds)
