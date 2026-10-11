"""The setup matrix (tools/setup_matrix.py): combined openings, the plan's coverage, the solver,
the report, and tools/tournament.py's --matchups."""

from __future__ import annotations

import collections
import itertools
import json
from typing import Dict, List, Mapping

import numpy as np
import pytest

import ts_engine as ts
from tools.lib.openings import (EAST_GERMANY, FRANCE, HUNGARY, IRAN, ITALY, POLAND, WEST_GERMANY,
                                opening_scripts, play_scripted_setup)
from tools.setup_matrix import DEFAULT_US, DEFAULT_USSR, Plan, describe, make_plan, report, solve_zero_sum
from tools.tournament import parse_matchups


def _fresh(seed: int = 11) -> "ts.GameState":
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    return st


def _inf(st: "ts.GameState", cid: int) -> tuple:
    c = st.get_country(cid)
    return int(c.us_influence), int(c.ussr_influence)


def test_a_combined_opening_scripts_both_sides_and_refuses_overlap() -> None:
    both = opening_scripts("us_e7+ussr_pol5")
    assert set(both) == {"US", "USSR"}
    with pytest.raises(ValueError, match="twice"):
        opening_scripts("us_e7+us_wg4_it4")
    with pytest.raises(ValueError, match="unknown"):
        opening_scripts("us_e7+nonsense")


@pytest.mark.parametrize("us,ussr", list(itertools.product(DEFAULT_US, DEFAULT_USSR)))
def test_every_default_cell_is_a_legal_setup(us: str, ussr: str) -> None:
    st = play_scripted_setup(_fresh(), f"{us}+{ussr}")
    assert st.current_phase != ts.Phase.SETUP and int(st.turn) == 1


def test_the_networks_setup_is_what_the_matrix_calls_it() -> None:
    # East Germany starts with 3 USSR, Iran with 1 US: the E7 setup adds France 6, Italy 2, Iran 1
    # and Poland 3, Hungary 2, Austria 1.
    st = play_scripted_setup(_fresh(), "us_e7+ussr_e7")
    assert _inf(st, FRANCE)[0] == 6 and _inf(st, ITALY)[0] == 2 and _inf(st, IRAN)[0] == 2
    assert _inf(st, WEST_GERMANY)[0] == 0
    assert _inf(st, POLAND)[1] == 3 and _inf(st, HUNGARY)[1] == 2 and _inf(st, EAST_GERMANY)[1] == 3
    st = play_scripted_setup(_fresh(), "us_wg4_it4+ussr_pol5")
    assert _inf(st, WEST_GERMANY)[0] == 4 and _inf(st, ITALY)[0] == 4 and _inf(st, IRAN)[0] == 2
    assert _inf(st, POLAND)[1] == 5 and _inf(st, EAST_GERMANY)[1] == 4


@pytest.mark.parametrize("n_us,n_ussr", [(4, 4), (3, 4), (3, 3), (2, 1)])
def test_the_plan_plays_every_cell_once_per_seat(n_us: int, n_ussr: int) -> None:
    us, ussr = list(DEFAULT_US[:n_us]), list(DEFAULT_USSR[:n_ussr])
    plan = make_plan(us, ussr, "models/base.pt")
    labels = {e["label"]: e for e in plan["entrants"]}
    seen = collections.Counter((c["us"], c["ussr"]) for c in plan["cells"])
    assert set(seen) == {(i, j) for i in range(n_us) for j in range(n_ussr)}
    assert len(plan["matchups"]) == (n_us * n_ussr + 1) // 2
    assert len({tuple(m) for m in plan["matchups"]}) == len(plan["matchups"])
    for c in plan["cells"]:
        # The US side of a cell plays its US setup, the USSR side its USSR setup.
        assert labels[c["us_label"]]["us"] == c["us"] and labels[c["ussr_label"]]["ussr"] == c["ussr"]
    # Each pairing's two seats are exactly the two cells listed for it.
    for (a, b) in plan["matchups"]:
        ea, eb = plan["entrants"][a]["label"], plan["entrants"][b]["label"]
        mine = [c for c in plan["cells"] if {c["us_label"], c["ussr_label"]} == {ea, eb}]
        assert len(mine) == 2
    for e in plan["entrants"]:
        assert e["spec"].startswith(f"name:{e['label']}:opening:{us[e['us']]}+{ussr[e['ussr']]}:")


def test_the_plan_refuses_a_two_sided_or_wrong_side_opening() -> None:
    with pytest.raises(ValueError, match="US-only"):
        make_plan(["human", "us_e7"], ["ussr_e7"], "x.pt")
    with pytest.raises(ValueError, match="USSR-only"):
        make_plan(["us_e7"], ["us_wg4_it4"], "x.pt")


def test_the_solver_on_known_games() -> None:
    x, y, v = solve_zero_sum(np.array([[1.0, 0.0], [0.0, 1.0]]))       # matching pennies
    assert np.allclose(x, [0.5, 0.5]) and np.allclose(y, [0.5, 0.5]) and v == pytest.approx(0.5)
    x, y, v = solve_zero_sum(np.array([[0.6, 0.7], [0.4, 0.5]]))       # a saddle point
    assert np.allclose(x, [1, 0]) and np.allclose(y, [1, 0]) and v == pytest.approx(0.6)
    rps = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
    x, y, v = solve_zero_sum(rps)
    assert np.allclose(x, 1 / 3) and np.allclose(y, 1 / 3) and v == pytest.approx(0.5)
    # A 3 x 2 game with a dominated row and a mixed solution.
    a = np.array([[0.7, 0.3], [0.2, 0.2], [0.4, 0.6]])
    x, y, v = solve_zero_sum(a)
    assert x[1] == pytest.approx(0) and v == pytest.approx(0.5)
    assert (x @ a >= v - 1e-9).all() and (a @ y <= v + 1e-9).all()


def _games(plan: Plan, truth: np.ndarray, deals: int, seed: int = 3) -> List[Mapping[str, object]]:
    rng = np.random.default_rng(seed)
    out: List[Mapping[str, object]] = []
    for s in range(deals):
        for c in plan["cells"]:
            u = rng.random()
            p = truth[c["us"], c["ussr"]]
            out.append({"seed": 10000 + s, "us_agent": c["us_label"], "ussr_agent": c["ussr_label"],
                        "winner": "US" if u < p else "USSR", "game_index": s})
    return out


def test_the_report_recovers_the_matrix_and_its_equilibrium() -> None:
    plan = make_plan(list(DEFAULT_US[:2]), list(DEFAULT_USSR[:2]), "models/base.pt")
    truth = np.array([[0.7, 0.3], [0.4, 0.6]])         # mixed equilibrium, value 0.5
    text, data = report(plan, _games(plan, truth, 3000), bootstrap=50)
    m = np.array(data["matrix"])
    assert np.abs(m - truth).max() < 0.03
    eq = data["equilibrium"]
    assert eq["value"] == pytest.approx(0.5, abs=0.03)
    assert np.allclose(eq["us"], [1 / 3, 2 / 3], atol=0.1)
    assert "## The solved game" in text and "West Germany 4, Italy 4, Iran 1" in text


def test_the_report_refuses_a_game_from_another_plan() -> None:
    plan = make_plan(list(DEFAULT_US[:2]), list(DEFAULT_USSR[:2]), "models/base.pt")
    bad: List[Mapping[str, object]] = [{"seed": 1, "us_agent": "someone", "ussr_agent": "U0S0", "winner": "US", "game_index": 0}]
    with pytest.raises(ValueError, match="not a cell"):
        report(plan, bad, bootstrap=1)


def test_describe_names_the_influence() -> None:
    assert describe("us_e7") == "France 6, Italy 2, Iran 1"
    assert describe("ussr_pol4_aus") == "Poland 4, East Germany 1, Austria 1"


def test_parse_matchups() -> None:
    assert parse_matchups(["1-2", "4-3"]) == [(0, 1), (2, 3)]
    for bad in ("1", "1-1", "0-2", "a-b"):
        with pytest.raises(ValueError):
            parse_matchups([bad])


def test_the_plan_round_trips_through_json(tmp_path) -> None:
    from tools.setup_matrix import main
    out = tmp_path / "plan.json"
    assert main(["plan", "--base", "models/base.pt", "--out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert len(plan["cells"]) == 16 and len(plan["matchups"]) == 8 and len(plan["entrants"]) == 16
