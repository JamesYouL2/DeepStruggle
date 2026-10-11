"""tools/turn1_pricing.py: turn-1 decisions recorded and priced, a forced opening honoured, the
honest (split-half) regret, the bank and the report."""

from __future__ import annotations

import collections
import gzip
import json
from typing import List, Optional, Sequence

import numpy as np
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from tools.turn1_pricing import (KINDS, Row, position_token, price_game, priced, read_rows, report,
                                 select_bank, state_from_token)

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)


def _net() -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net(torch.device("cpu"), **CFG).eval()


def test_a_game_prices_its_turn1_decisions_and_honours_the_opening() -> None:
    net, dev = _net(), torch.device("cpu")
    rows = price_game(net, dev, 7, "us_wg3_fr3_it2+ussr_pol4_aus", worlds=4, max_options=12, point_cap=60)
    assert rows and all(r["turn"] == 1 for r in rows)
    kinds = collections.Counter(r["kind"] for r in rows)
    assert kinds["headline"] == 2 and kinds["card to play"] > 0
    assert set(kinds) <= set(KINDS)
    for r in rows:
        assert len(r["values"]) == len(r["options"]) == len(r["names"]) == len(r["probs"])
        assert 0 <= r["taken"] < len(r["options"])
        vals = [x for row in r["values"] for x in row if x is not None]
        assert all(abs(x) <= 1.0 + 1e-5 for x in vals)
    st = state_from_token(rows[0]["pos"])
    # The forced opening: West Germany 3, France 3 for the US; East Germany 4, Poland 4 for the USSR.
    wg, fr, eg, pl = (st.get_country(c) for c in (7, 8, 14, 15))
    assert (int(wg.us_influence), int(fr.us_influence)) == (3, 3)
    assert (int(eg.ussr_influence), int(pl.ussr_influence)) == (4, 4)
    assert position_token(st) == rows[0]["pos"]


def _row(values: Sequence[Sequence[Optional[float]]], taken: int, kind: str = "card to play", seed: int = 1) -> Row:
    return {"id": f"{seed}-{taken}-{kind}", "seed": seed, "choice": 0, "opening": "", "side": "US", "turn": 1,
            "ar": 1, "phase": "ACTION_ROUND", "kind": kind, "card": 0, "card_name": "",
            "names": [f"o{i}" for i in range(len(values))], "options": list(range(len(values))),
            "probs": [1 / len(values)] * len(values), "taken": taken, "values": [list(v) for v in values], "pos": ""}


def test_the_honest_regret_is_near_zero_between_equal_options_and_finds_a_real_lead() -> None:
    rng = np.random.default_rng(0)
    equal = [[float(x) for x in rng.normal(0, 0.3, 400)] for _ in range(5)]
    p = priced(_row(equal, 0))
    assert p is not None and p["diff"] > 0 and abs(p["honest"]) < 0.04       # the lead is the winner's curse
    real = [[float(x) for x in rng.normal(mu, 0.3, 400)] for mu in (0.0, 0.0, 0.2)]
    p = priced(_row(real, 0))
    assert p is not None and p["best"] == 2 and abs(p["honest"] - 0.2) < 0.05


def test_a_world_with_an_illegal_option_is_dropped() -> None:
    vals = [[0.1] * 10, [0.2] * 9 + [None]]
    p = priced(_row(vals, 0))
    assert p is not None and p["worlds"] == 9


def test_bank_and_report(tmp_path) -> None:
    rng = np.random.default_rng(1)
    rows = [_row([[float(x) for x in rng.normal(mu, 0.2, 64)] for mu in (0.0, 0.3)], 0, seed=s) for s in range(5)]
    rows += [_row([[float(x) for x in rng.normal(0, 0.2, 64)] for _ in range(3)], 1, kind="headline", seed=9)]
    path = tmp_path / "rows.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    back = read_rows([str(path)])
    assert len(back) == len(rows)
    bank = select_bank(back, top=10, z_min=3.0, per_kind_card=3)
    assert len(bank) == 3 and all(b["bank_best"] == 1 for b in bank)       # capped per (kind, card)
    conf = [{"id": bank[0]["id"], "taken": "o0", "best": "o1", "pairs": 64, "diff": 0.12, "se": 0.03}]
    text = report(back, conf, bank, "net", 3.0)
    assert "## By kind of decision" in text and "card to play" in text and "+12.0 ± 3.0" in text
