"""Playing one side against the model: the page must not show the other side's private cards.

A player's view (`?view=us|ussr`) draws every panel from `redactState` (web/ui/src/game/view.ts),
never from the engine's full display state. These tests drive a game in the page -- both sides by
the auto-play path, `sendFlatAction(i, 0, true)`, so no model is needed -- and check what the
panels were handed (`__wb.shown`) and what the page actually says.

Needs web/ui/dist (tools/scripts/build_web.sh) and a Playwright Chromium.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from tests.web.test_e2e_workbench import _open, browser, static_site  # noqa: F401 -- fixtures

#: Play the first legal action for whoever is to move, until `stop(state)` holds.
_PLAY_UNTIL = """stop => { const wb = window.__wb, until = new Function('s', 'return ' + stop);
    for (let n = 0; n < 5000 && !wb.state.is_terminal && !until(wb.state); n++) {
        const m = wb.engine.mask(false); wb.sendFlatAction(m.findIndex(x => x), 0, true);
    }
    return wb.state.step_index; }"""


def _play_until(page: Any, stop: str) -> None:
    page.evaluate(_PLAY_UNTIL, stop)


def _names(page: Any, cards: List[int]) -> List[str]:
    return [page.evaluate(f"window.__wb.engine.cardName({c})") for c in cards]


def test_the_other_sides_hand_and_the_deck_are_one_unseen_pile(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off")
    _play_until(page, "s.current_phase_name === 'HEADLINE'")
    full: Dict[str, Any] = page.evaluate("window.__wb.state")
    page.evaluate("window.__wb.setView('US')")
    shown: Dict[str, Any] = page.evaluate("window.__wb.shown")

    hidden = [c for c in full["hands"]["USSR"] if full["card_locations"][str(c)] == "HAND_USSR_UNKNOWN"]
    assert hidden, "the USSR's opening hand is not known to the US"
    assert not set(hidden) & set(shown["hands"]["USSR"])
    assert shown["hidden_cards"] == {"US": 0, "USSR": len(hidden)}
    locations = set(shown["card_locations"].values())
    assert not locations & {"HAND_USSR_UNKNOWN", "DRAW_DECK"}
    assert shown["unseen_count"] == full["draw_deck_count"] + len(hidden)
    # Nothing the US cannot know reaches the page, on any tab of the cards panel.
    for tab in ("tab-ussr-hand", "tab-us-hand", "tab-board-cards"):
        page.click(f"button[data-tab='{tab}']")
        text = page.inner_text("body")
        assert [n for n in _names(page, hidden) if n in text] == [], f"a USSR card is named on {tab}"
        if tab == "tab-us-hand":
            assert all(n in text for n in _names(page, full["hands"]["US"]))
        if tab == "tab-ussr-hand":
            assert page.locator(".card-item.card-hidden:visible").count() == len(hidden)
    # The All tab lists every card, the USSR's included -- as the deck-or-hand pile, never the hand.
    page.click("button[data-tab='tab-all-cards']")
    page.select_option("#all-cards-filter", "DRAW_DECK")
    deck = page.inner_text("#tab-all-cards")
    assert all(n in deck for n in _names(page, hidden))
    assert "HAND_USSR" not in deck and "USSR hand" in deck
    assert page.evaluate("new URL(location.href).searchParams.get('view')") == "us"
    assert page.evaluate("[document.getElementById('btn-toggle-debug').disabled, "
                         "document.getElementById('btn-toggle-replay').disabled]") == [True, True]
    assert page.errors == []
    page.close()


def test_the_other_sides_deal_and_face_down_headline_are_not_logged(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off")
    _play_until(page, "s.turn >= 2 && s.current_phase_name === 'HEADLINE'")
    full_logs = json.dumps(page.evaluate("window.__wb.state.action_logs"))
    for view, opp in (("US", "USSR"), ("USSR", "US")):
        page.evaluate(f"window.__wb.setView('{view}')")
        logs = json.dumps(page.evaluate("window.__wb.shown.action_logs"))
        assert f"HAND_{opp}_UNKNOWN" in full_logs
        assert f"HAND_{opp}_UNKNOWN" not in logs, f"the {view} view logs the {opp} deal"
        assert f"HAND_{view}_UNKNOWN" in logs, f"the {view} view lost its own deal"

    # One side commits its headline; the side still choosing must not see it.
    page.evaluate("window.__wb.setView('DEV')")
    first = page.evaluate("window.__wb.state.decision_context.decision_player")
    _play_until(page, f"s.decision_context.decision_player !== '{first}'")
    second = "USSR" if first == "US" else "US"
    key = "headline_us_card" if first == "US" else "headline_ussr_card"
    card = page.evaluate(f"window.__wb.state.{key}")
    space = page.evaluate("window.__wb.state.space")
    perk = space[second] >= 4 and space[first] < 4   # Man in Space: the other side shows first
    assert card and not perk
    page.evaluate(f"window.__wb.setView('{second}')")
    assert page.evaluate(f"window.__wb.shown.{key}") == 0
    log = page.inner_text("#action-log-stream")
    assert _names(page, [card])[0] not in log
    assert "face down" in log
    page.close()


def test_the_other_sides_decision_is_hidden_and_refused(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&view=ussr")
    # Setup: the USSR places first, so play it out until the US is to move.
    _play_until(page, "s.decision_context.decision_player === 'US'")
    shown = page.evaluate("window.__wb.shown")
    assert shown["view_hidden_decision"] is True
    assert shown["legal_actions"]["valid_ids"] == []
    assert "to move" in page.inner_text("#decision-body")
    before = page.evaluate("window.__wb.state.step_index")
    page.evaluate("window.__wb.sendAction({decision_type: 1, primary_id: "
                  "window.__wb.state.legal_actions.valid_ids[0], secondary_id: 0, flags: 0})")
    page.evaluate("window.__wb.sendFlatAction(window.__wb.engine.mask(false).findIndex(x => x), 0)")
    assert page.evaluate("window.__wb.state.step_index") == before
    page.close()


def test_new_game_and_the_view_select_fit_a_laptop_screen(browser: Any, static_site: str) -> None:
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    page.goto(f"{static_site}/?model=off")
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    for el in ("view-select", "btn-new-game", "btn-toggle-debug", "btn-toggle-replay"):
        box = page.locator(f"#{el}").bounding_box()
        assert box and box["width"] > 0 and box["x"] + box["width"] <= 1366, f"#{el} is off screen"
    page.select_option("#view-select", "US")
    _play_until(page, "s.current_phase_name === 'HEADLINE'")
    page.click("#btn-new-game")
    page.wait_for_function("window.__wb.state.step_index === 0")
    assert page.evaluate("window.__wb.shown.view") == "US"   # a new game keeps the chosen view
    page.close()
