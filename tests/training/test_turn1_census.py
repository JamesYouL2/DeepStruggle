"""The turn-1 census (tools/turn1_census.py): per-game records rebuilt from turn1_pricing rows."""

from __future__ import annotations

import json
from typing import Dict

import torch

from ai.models.ladder_net import create_ladder_net
from tools.turn1_census import census, games_from_rows, option_label, priced_section, spread_label
from tools.turn1_pricing import price_game, state_from_token

from tests.training.test_turn1_pricing import CFG


def _by_id(spread: Dict[str, int]) -> Dict[int, int]:
    ids = {str(c["name"]): int(c["id"]) for c in json.load(open("rules/map.json"))["countries"]}
    return {ids[k]: v for k, v in spread.items()}


def _net() -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net(torch.device("cpu"), **CFG).eval()


def test_option_labels() -> None:
    assert option_label("SelectCard #25 (Containment)") == "Containment"
    assert option_label("PointNode #10 (Italy)") == "Italy"
    assert option_label("Resolution: OPS_COUP / OpMode: COUP") == "COUP"
    assert option_label("Resolution: EVENT") == "EVENT"
    assert spread_label({"Pakistan": 2, "France": 1, "Iran": 0}) == "Pakistan 2, France 1"


def test_games_from_rows_match_the_positions() -> None:
    net, dev = _net(), torch.device("cpu")
    rows = [r for seed in (3, 4) for r in price_game(net, dev, seed, "us_r34+ussr_r34", worlds=8, max_options=12, point_cap=60)]
    games = games_from_rows(rows)
    assert [g["seed"] for g in games] == [3, 4]
    for g in games:
        assert len(g["us_hand"]) == 8 and len(g["ussr_hand"]) >= 8      # the USSR also holds the China Card
        assert g["us_headline"] in g["us_hand"] and g["ussr_headline"] in g["ussr_hand"]
        a = g["us_ar1"]
        assert a is not None and a["card"]
        if a["ops"] == "INFLUENCE":
            assert sum(a["spread"].values()) > 0 and a["target"] in a["spread"]
    # the US's AR1 spread is the board change between its card choice and the USSR's AR2
    g = games[0]
    rs = sorted((r for r in rows if r["seed"] == g["seed"]), key=lambda r: r["choice"])
    us1 = next(r for r in rs if r["side"] == "US" and r["ar"] == 1 and r["kind"] == "card to play")
    ussr2 = next(r for r in rs if r["side"] == "USSR" and r["ar"] == 2 and r["kind"] == "card to play")
    before, after = state_from_token(us1["pos"]), state_from_token(ussr2["pos"])
    added = {c: int(after.get_country(c).us_influence) - int(before.get_country(c).us_influence) for c in range(84)}
    assert g["us_ar1"] is not None
    assert {c: v for c, v in added.items() if v} == _by_id(g["us_ar1"]["spread"])
    text = "\n".join(census(games, 5) + priced_section(rows, 5))
    assert "## The USSR's AR1" in text and "### USSR headline" in text
