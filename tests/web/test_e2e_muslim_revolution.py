"""Muslim Revolution with one eligible country: the page lets the USSR finish the Event.

"Remove all US Influence from 2 of: Sudan, Iran, Iraq, Egypt, Libya, Saudi Arabia, Syria,
Jordan." With US Influence in only one of them, the engine takes the one removal and then offers
only CONFIRM_DONE (the removal is mandatory, so the Event does not allow an early stop, and the
targets have run out). The page must show that as something to click, or the game cannot go on.
"""
from __future__ import annotations

from typing import Any, List

import ts_engine as ts

from tests.engine_logic.test_headline_card_leaves_hand import _at_headline, _set_hand
from tests.web.test_e2e_card_choices import _token
from tests.web.test_e2e_workbench import _open, browser, static_site  # noqa: F401 -- fixtures

MUSLIM_REVOLUTION = 56
NUCLEAR_TEST_BAN = 34
EAST_EUROPEAN_UNREST = 29
DUCK_AND_COVER = 4
FIDEL = 8
TARGETS = ["Sudan", "Iran", "Iraq", "Egypt", "Libya", "Saudi Arabia", "Syria", "Jordan"]


def _legal(s: ts.GameState) -> List[int]:
    return [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(s)) if v]


def _muslim_revolution_with(eligible: List[str]) -> ts.GameState:
    """The USSR's action round 1, Muslim Revolution played as its Event, US Influence only in
    `eligible` among the eight countries."""
    s = _at_headline()
    _set_hand(s, ts.Player.US, [DUCK_AND_COVER, NUCLEAR_TEST_BAN])
    _set_hand(s, ts.Player.USSR, [MUSLIM_REVOLUTION, EAST_EUROPEAN_UNREST, FIDEL])
    for _ in range(2):
        p = s.ctx().decision_player
        ts.Engine.step_flat(s, (NUCLEAR_TEST_BAN if p == ts.Player.US else EAST_EUROPEAN_UNREST) - 1)
    while s.current_phase == ts.Phase.HEADLINE:
        ts.Engine.step_flat(s, _legal(s)[0])
    for name in TARGETS:
        cid = ts.MapData.get_country_by_name(name)
        s.set_country(cid, 1 if name in eligible else 0, s.get_country(cid).ussr_influence)
    assert s.ctx().decision_player == ts.Player.USSR
    ts.Engine.step_flat(s, MUSLIM_REVOLUTION - 1)
    ts.Engine.step_flat(s, 110)   # the Event
    assert int(s.ctx().resolving_card) == MUSLIM_REVOLUTION
    return s


DONE = "#decision-body button[data-flags='128']"


def _remove(page: Any, name: str) -> None:
    page.click(f"#decision-body button[data-primary='{ts.MapData.get_country_by_name(name)}']")
    page.wait_for_function(f"window.__wb.state.countries[{name!r}].us_influence === 0")


def _finish(page: Any) -> None:
    page.click(DONE)
    page.wait_for_function("window.__wb.state.decision_context.resolving_card === 0")
    ctx = page.evaluate("window.__wb.state.decision_context")
    assert (ctx["decision_player"], ctx["decision_type_name"]) == ("US", "SELECT_CARD"), ctx


def test_one_eligible_country_then_the_event_can_be_finished(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&pos={_token(_muslim_revolution_with(['Iran']))}")
    _remove(page, "Iran")
    assert page.evaluate("window.__wb.state.legal_actions.valid_ids") == []
    assert "No eligible target is left" in page.inner_text("#decision-body")
    _finish(page)
    assert page.errors == []
    page.close()


def test_no_eligible_country_finishes_at_once(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&pos={_token(_muslim_revolution_with([]))}")
    _finish(page)
    assert page.errors == []
    page.close()


def test_two_eligible_countries_are_both_removed_before_it_ends(browser: Any, static_site: str) -> None:
    """The removal is mandatory: with a target left there is no way to stop early."""
    page = _open(browser, f"{static_site}/?model=off&pos={_token(_muslim_revolution_with(['Iran', 'Iraq']))}")
    _remove(page, "Iran")
    assert page.locator(DONE).count() == 0
    _remove(page, "Iraq")
    page.wait_for_function("window.__wb.state.decision_context.resolving_card === 0")
    assert page.errors == []
    page.close()
