"""tools/scripts/setup_deal_matrix.py: the deal it edits, the setups it forces, the worlds it plays
and the game it solves."""
import json
import os
from typing import List, Sequence

import numpy as np
import pytest
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.jsonl import read_jsonl, write_jsonl
from tools.scripts import setup_deal_matrix as sdm


def _names(cards: List[int]) -> List[str]:
    return sorted(ts.CardData.get_card_name(c) for c in cards)


def test_the_deal_holds_the_swapped_cards_and_nothing_else_moves() -> None:
    plain = ts.GameState()
    ts.Engine.init_game(plain, sdm.DEAL)
    st = sdm.build_deal(sdm.DEAL, sdm.SWAPS)
    ussr, us = _names(sdm.hand(st, ts.Player.USSR)), _names(sdm.hand(st, ts.Player.US))
    assert len(ussr) == len(us) == 8
    for c in ("Olympic Games", "Formosan Resolution", "Captured Nazi Scientist"):
        assert c in ussr
    assert "De-Stalinization" in us
    ids = sdm.card_ids()
    for c in ("Vietnam Revolts", "Middle East Scoring", "Socialist Governments"):
        assert st.get_card_location(ids[c]) == ts.CardLocation.DRAW_DECK
    moved = {c for c in range(1, 111) if st.get_card_location(c) != plain.get_card_location(c)}
    assert {ts.CardData.get_card_name(c) for c in moved} == {
        "Vietnam Revolts", "Middle East Scoring", "Socialist Governments", "Olympic Games",
        "Formosan Resolution", "Captured Nazi Scientist", "De-Stalinization"}  # USSR hand -> US hand
    assert st.rng_state == plain.rng_state
    with pytest.raises(ValueError):
        sdm.build_deal(sdm.DEAL, ["Olympic Games=Vietnam Revolts"])  # Olympic Games is in the deck


def test_every_setup_lands_as_named() -> None:
    base = sdm.build_deal(sdm.DEAL, sdm.SWAPS)
    ussr = [sdm.parse_setup(nm, base, ts.Player.USSR) for nm, _ in sdm.USSR_SETUPS]
    us = [sdm.parse_setup(nm, base, ts.Player.US) for nm, _ in sdm.US_SETUPS]
    for (nm, _), added in zip(sdm.USSR_SETUPS, ussr):
        assert sdm.setup_name(added, base, ts.Player.USSR) == nm
    for (nm, _), added in zip(sdm.US_SETUPS, us):
        assert sdm.setup_name(added, base, ts.Player.US) == nm
    assert len({tuple(sorted(a.items())) for a in ussr}) == len(ussr)
    assert len({tuple(sorted(a.items())) for a in us}) == len(us)
    poland = ts.MapData.get_country_by_name("Poland")
    hungary = ts.MapData.get_country_by_name("Hungary")
    assert sdm.setup_name({poland: 4, hungary: 2}, base, ts.Player.USSR) == "Pol4/Hun2"  # not a listed one
    for a in ussr:
        for b in us:
            st = sdm.cell_start(base, a, b)  # refuses a setup that lands elsewhere
            assert st.current_phase == ts.Phase.HEADLINE
    for name in set(sdm.HUMAN_USSR) | set(sdm.HUMAN_REPLY):
        assert name in dict(sdm.USSR_SETUPS)
    for name in set(sdm.HUMAN_US) | {u for r in sdm.HUMAN_REPLY.values() for u in r}:
        assert name in dict(sdm.US_SETUPS)
    with pytest.raises(ValueError):
        sdm.parse_setup("EG4/Pol4", base, ts.Player.USSR)  # five points
    with pytest.raises(ValueError):
        sdm.cell_start(base, ussr[0], sdm.parse_setup("WG4/It2/Ir3/Pan2", base, ts.Player.US))  # six in Western Europe


def _first_legal(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Logits that put the first legal action first."""
    return np.asarray(masks, dtype=np.float32)


def test_a_world_is_its_seed_whatever_part_or_batch_plays_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sdm, "USSR_SETUPS", sdm.USSR_SETUPS[3:5])
    monkeypatch.setattr(sdm, "US_SETUPS", sdm.US_SETUPS[:3])
    monkeypatch.setattr(sdm, "load_policy", lambda path: (_first_legal, 0))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a model")
    whole = str(tmp_path / "whole.jsonl.gz")
    sdm.run(str(model), 4, "1/1", 0, sdm.DEAL, sdm.SWAPS, 6, whole)
    parts = [str(tmp_path / f"matrix-{k}.jsonl.gz") for k in (1, 2)]
    for k, p in enumerate(parts, 1):
        sdm.run(str(model), 4, f"{k}/2", 0, sdm.DEAL, sdm.SWAPS, 1536, p)
    pooled = str(tmp_path / "pooled.jsonl.gz")
    sdm.pool(parts, pooled, expect=2)
    a = sorted(read_jsonl([whole]), key=lambda r: r["world"])
    b = sorted(read_jsonl([pooled]), key=lambda r: r["world"])
    assert [r["world"] for r in a] == [0, 1, 2, 3]
    assert a == b  # same worlds, other batches and parts: the same games
    assert all(len(r["s"]) == 6 for r in a)
    assert len({r["s"] for r in a}) > 1  # the worlds' dice differ
    meta = json.load(open(pooled + ".meta.json"))
    assert meta["ussr"] == ["Pol3/Yug3", "Pol3/Hun2/Aus1"] and meta["rows"] == 4
    assert json.load(open(pooled + ".parts.json"))["missing"] == []


def test_random_deals_give_each_world_its_own_deal_whatever_part_plays_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sdm, "load_policy", lambda path: (_first_legal, 0))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a model")
    ussr, us = ["Pol3/Yug3", "EG4/Pol5"], ["WG4/It4/Ir2", "WG4/It3/Fr2", "WG3/It3/Fr2/Ir2"]
    whole = str(tmp_path / "whole.jsonl.gz")
    sdm.run(str(model), 4, "1/1", 3, 0, [], 6, whole, ussr, us, random_deals=True)
    parts = [str(tmp_path / f"matrix-{k}.jsonl.gz") for k in (1, 2)]
    for k, p in enumerate(parts, 1):
        sdm.run(str(model), 4, f"{k}/2", 3, 0, [], 1536, p, ussr, us, random_deals=True)
    pooled = str(tmp_path / "pooled.jsonl.gz")
    sdm.pool(parts, pooled, expect=2)
    a = sorted(read_jsonl([whole]), key=lambda r: r["world"])
    b = sorted(read_jsonl([pooled]), key=lambda r: r["world"])
    assert a == b  # same worlds, other batches and parts: the same deals and games
    assert len({r["deal"] for r in a}) == 4
    for r in a:
        st = sdm.dealt(sdm.world_deal(3, r["world"]))
        assert r["deal"] == sdm.world_deal(3, r["world"])
        assert r["ussr_hand"] == sdm.hand(st, ts.Player.USSR) and r["us_hand"] == sdm.hand(st, ts.Player.US)
        assert len(r["ussr_hand"]) == len(r["us_hand"]) == 8 and len(r["s"]) == 6
    meta = json.load(open(pooled + ".meta.json"))
    assert meta["deal"] == sdm.RANDOM and meta["net"] is None and meta["ussr"] == ussr and meta["us"] == us


def test_the_solver_finds_a_mixed_equilibrium() -> None:
    a = np.array([[0.6, 0.4], [0.3, 0.7]])
    x, y, v, gap = sdm.solve_game(a, tol=1e-6)
    assert gap < 1e-5
    assert abs(v - 0.5) < 1e-5
    assert np.allclose(x, [2 / 3, 1 / 3], atol=1e-3) and np.allclose(y, [0.5, 0.5], atol=1e-3)
    saddle = np.array([[0.5, 0.2], [0.8, 0.6]])  # the row player takes row 0; the US then column 0
    x, y, v, _ = sdm.solve_game(saddle, tol=1e-6)
    assert abs(v - 0.5) < 1e-5 and x[0] > 0.999 and y[0] > 0.999


def test_the_report_reads_the_matrix_it_was_given(tmp_path) -> None:
    ussr, us = [nm for nm, _ in sdm.USSR_SETUPS], [nm for nm, _ in sdm.US_SETUPS]
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, size=(len(ussr), len(us)))
    worlds = 400
    rows = [{"world": w, "s": "".join(str(2 * int(rng.random() < q)) for q in p.ravel())} for w in range(worlds)]
    src = str(tmp_path / "matrix-1.jsonl.gz")
    write_jsonl(src, rows)
    net_us = {nm: "WG4/It3/Ir2/Pan2" for nm in ussr}
    with open(src + ".meta.json", "w") as f:
        json.dump({"model": "m.onnx", "model_sha256": "0" * 64, "deal": sdm.DEAL, "swaps": list(sdm.SWAPS),
                   "hands": sdm.deal_summary(sdm.build_deal(sdm.DEAL, sdm.SWAPS)), "ussr": ussr, "us": us,
                   "net": {"ussr": "Pol3/Yug3", "us": net_us}, "worlds": worlds, "part": "1/1", "seed": 0,
                   "rows": worlds}, f)
    out = str(tmp_path / "report.md")
    assert sdm.report([src], out, bootstrap=5) == 0
    data = json.load(open(os.path.splitext(out)[0] + ".json"))
    m = np.array(data["mean"])
    assert m.shape == p.shape and np.abs(m - p).max() < 0.15
    eq = data["equilibrium"]
    assert abs(sum(eq["ussr"]) - 1) < 1e-9 and abs(sum(eq["us"]) - 1) < 1e-9
    assert min(m @ np.array(eq["us"])) - 1e-3 <= eq["value"] <= max(np.array(eq["ussr"]) @ m) + 1e-3
    text = open(out).read()
    assert "## The equilibrium" in text and "Pol3/Yug3" in text


def test_the_report_splits_random_deals_by_where_a_scoring_card_went(tmp_path) -> None:
    ussr, us = ["EG4/Pol4/Aus1", "Pol3/Yug3"], ["WG4/It4/Ir2", "WG4/It3/Fr2"]
    ids = sdm.card_ids()
    me = ids["Middle East Scoring"]
    others = [c for c in range(1, 111) if c not in (me, ids["Asia Scoring"], ids["Europe Scoring"])][:30]
    rng = np.random.default_rng(1)
    rows = []
    for w in range(3000):
        hands = list(rng.choice(others, 16, replace=False).tolist())
        where = rng.integers(3)  # 0: the US hand, 1: the USSR hand, 2: neither
        if where < 2:
            hands[0 if where == 0 else 8] = me
        # WG4/It3/Fr2 gains 20 points when the US holds Middle East Scoring, none otherwise
        p = np.array([[0.5, 0.7 if where == 0 else 0.5]] * 2)
        rows.append({"world": w, "deal": w, "ussr_hand": hands[8:], "us_hand": hands[:8],
                     "s": "".join(str(2 * int(rng.random() < q)) for q in p.ravel())})
    src = str(tmp_path / "matrix-1.jsonl.gz")
    write_jsonl(src, rows)
    with open(src + ".meta.json", "w") as f:
        json.dump({"model": "m.onnx", "model_sha256": "0" * 64, "deal": sdm.RANDOM, "swaps": [], "hands": None,
                   "ussr": ussr, "us": us, "net": None, "worlds": 3000, "part": "1/1", "seed": 0, "rows": 3000}, f)
    out = str(tmp_path / "report.md")
    assert sdm.report([src], out, bootstrap=5) == 0
    split = json.load(open(os.path.splitext(out)[0] + ".json"))["split"]["Middle East Scoring"]
    assert sum(split[g]["n"] for g in ("us", "ussr", "deck")) == 3000
    gain = {g: np.array(split[g]["mean"])[:, 1].mean() - np.array(split[g]["mean"])[:, 0].mean() for g in split}
    assert abs(gain["us"] - 0.2) < 0.06 and abs(gain["ussr"]) < 0.06 and abs(gain["deck"]) < 0.06
    text = open(out).read()
    assert "over random deals" in text and "### Middle East Scoring" in text and "network" not in text


def test_headlines_keep_the_card_holders_worlds_and_name_both_headlines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sdm, "load_policy", lambda path: (_first_legal, 0))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a model")
    ussr, us = ["Pol3/Yug3", "EG4/Pol4/Yug1"], ["WG4/It4/Ir2", "WG3/Fr3/It2/Ir2"]
    card = "East European Unrest"
    whole = str(tmp_path / "whole.jsonl.gz")
    sdm.headlines(str(model), 40, "1/1", 0, 6, whole, ussr, us, card, "us")
    parts = [str(tmp_path / f"matrix-{k}.jsonl.gz") for k in (1, 2)]
    for k, p in enumerate(parts, 1):
        sdm.headlines(str(model), 40, f"{k}/2", 0, 1536, p, ussr, us, card, "us")
    pooled = str(tmp_path / "pooled.jsonl.gz")
    sdm.pool(parts, pooled, expect=2)
    a = sorted(read_jsonl([whole]), key=lambda r: r["world"])
    assert a == sorted(read_jsonl([pooled]), key=lambda r: r["world"])
    cid = sdm.card_ids()[card]
    held = [w for w in range(40) if cid in sdm.hand(sdm.dealt(sdm.world_deal(0, w)), ts.Player.US)]
    assert [r["world"] for r in a] == held and held
    for r in a:
        assert len(r["us_hl"]) == len(r["ussr_hl"]) == len(r["targets"]) == 4
        assert all(c in r["us_hand"] for c in r["us_hl"]) and all(c in r["ussr_hand"] for c in r["ussr_hl"])
        for c, tg in zip(r["us_hl"], r["targets"]):
            assert (len(tg) == 3) == (c == cid)  # Early War: one Influence from three countries
    assert json.load(open(pooled + ".meta.json"))["stage"] == "headlines"


class _Picker:
    """A stand-in for a searched player: the first legal move (the fake network's) or the last."""

    def __init__(self, last: bool) -> None:
        self.name, self.last = "picker", last

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        legal = [np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st))) for st in states]
        return [int(a[-1] if self.last else a[0]) for a in legal]


@pytest.mark.parametrize("last", [False, True])
def test_reply_forces_the_headline_and_pairs_the_two_arms(tmp_path, monkeypatch, last: bool) -> None:
    monkeypatch.setattr(sdm, "load_policy", lambda path: (_first_legal, 0))
    monkeypatch.setattr(sdm, "load_agent", lambda spec, device: _Picker(last))
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model")
    ussr, us = ["Pol3/Yug3", "EG4/Pol4/Yug1"], ["WG4/It4/Ir2"]
    card = "East European Unrest"
    out = str(tmp_path / "matrix-1.jsonl.gz")
    sdm.reply(str(model), "picker:{MODEL}", 16, "1/1", 0, 4, out, ussr, us, card, "us", 1)
    rows = list(read_jsonl([out]))
    cid = sdm.card_ids()[card]
    assert [r["world"] for r in rows] == [w for w in range(16) if cid in sdm.hand(sdm.dealt(sdm.world_deal(0, w)), ts.Player.US)]
    assert rows
    for r in rows:
        for arm in ("net", "searched"):
            a = r[arm]
            assert len(a["s"]) == len(a["hl"]) == len(a["moves"]) == len(a["end"]) == 2
            assert all(len(tg) == 3 for tg in a["targets"])  # the US headline was the card, every cell
            assert all(c in r["ussr_hand"] for c in a["hl"])
            assert all(m[0] == 1 for mv in a["moves"] for m in mv)
        # the network's own move as the searched player gives the network's games, cell for cell
        assert (r["net"] == r["searched"]) != last
    meta = json.load(open(out + ".meta.json"))
    assert meta["stage"] == "reply" and meta["player"] == "picker:{MODEL}" and meta["turns"] == 1


def test_ar1_forces_the_first_action_round_and_leaves_the_net_arm_to_the_network(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sdm, "load_policy", lambda path: (_first_legal, 0))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not a model")
    ussr, us = ["Pol3/Yug3", "EG4/Pol4/Yug1"], ["WG4/It4/Ir2"]
    card = "East European Unrest"
    out, ref = str(tmp_path / "ar1.jsonl.gz"), str(tmp_path / "reply.jsonl.gz")
    sdm.ar1(str(model), 16, "1/1", 0, 4, out, ussr, us, card, "us", sdm.AR1_ARMS)
    monkeypatch.setattr(sdm, "load_agent", lambda spec, device: _Picker(False))
    sdm.reply(str(model), "picker:{MODEL}", 16, "1/1", 0, 4, ref, ussr, us, card, "us", 1)
    rows, replies = list(read_jsonl([out])), list(read_jsonl([ref]))
    assert rows and [r["world"] for r in rows] == [r["world"] for r in replies]
    big = {c for c in range(1, 111) if c != sdm.CHINA and ts.CardData.get_card_info(c)["ops"] >= 3}
    italy, pol = sdm.country_ids()["Italy"], sdm.country_ids()["Poland"]
    coups = []
    for r, y in zip(rows, replies):
        assert r["net"] == y["net"]
        for arm in sdm.AR1_ARMS[1:]:
            for mv in r[arm]["moves"]:
                ar1 = [m for m in mv if m[1] == 1]
                if ar1[0][4] not in big:  # no 3+ Ops card in hand: the arm leaves the round alone
                    continue
                mode, node = ar1[1][4], ar1[2][4]
                if arm == "coup_italy":
                    assert mode == 3
                    coups.append(node == italy)  # not when a headline took DEFCON to 4
                if arm.startswith("repair"):
                    assert mode == 2
        # Pol3/Yug3: where the headline took Poland below control, repair puts a point back
        first = [m for m in r["repair"]["moves"][0] if m[1] == 1]
        if first[0][4] in big and pol in r["repair"]["targets"][0]:
            assert pol in [m[4] for m in first[2:]]
    assert any(coups)
    assert json.load(open(out + ".meta.json"))["arms"] == list(sdm.AR1_ARMS)
