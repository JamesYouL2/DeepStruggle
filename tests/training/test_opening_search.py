"""The opening-search setup and report, without a model or a search."""
import json

import ts_engine as ts

from ai.eval.opening_search import BASELINE, CANDIDATES, report, set_up

EG, POL, AUT, HUN, YUG = 14, 15, 13, 17, 18
WANT = {
    "pol3_hun3": {EG: 3, POL: 3, HUN: 3},
    "pol3_yug3": {EG: 3, POL: 3, YUG: 3},
    "eg4_pol4_aut1": {EG: 4, POL: 4, AUT: 1},
    "eg4_pol4_yug1": {EG: 4, POL: 4, YUG: 1},
    "eg4_pol5": {EG: 4, POL: 5},
}


def test_every_candidate_sets_up_on_the_same_deal() -> None:
    hands = set()
    for name in CANDIDATES:
        st = set_up(7_000_000, CANDIDATES[name], "fixed")
        assert st.current_phase != ts.Phase.SETUP
        d = json.loads(st.to_save_json())
        for cid, n in WANT[name].items():
            assert d["ussr_influence"][cid] == n, (name, cid)
        assert [d["us_influence"][c] for c in (7, 8, 10, 25)] == [3, 3, 2, 2]
        hands.add(tuple(d["card_locations"]))
    assert len(hands) == 1, "the opening must not change the deal"


def test_report_pairs_by_deal() -> None:
    rows = []
    for seed, shift in ((1, 0.0), (2, 0.2)):         # deal 2 is better for the USSR everywhere
        rows.append({"seed": seed, "opening": BASELINE, "critic": shift, "search": {"64": [shift]}})
        rows.append({"seed": seed, "opening": "pol3_yug3", "critic": shift + 0.02,
                     "search": {"64": [shift + 0.04]}})
    _, js = report(rows)
    d = js["budgets"]["search 64 sims"]["pol3_yug3"]
    assert abs(d["diff"] - 2.0) < 1e-9 and d["se"] < 1e-9    # 50 x 0.04, identical on both deals
