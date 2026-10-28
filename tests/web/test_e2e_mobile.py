"""The page on a phone, and the board aids borrowed from the Playdek app.

At 720px and narrower the page scrolls as one column: the bars wrap, the map comes first, then the
decision panel and the cards, and nothing is wider than the screen. The map pans with one finger,
pinches to zoom and jumps to a region from a row of buttons; a tap on a country that is not a target
shows its card (a target is still chosen by the tap). A region's badge opens its scoring summary --
the two sides' standing, battlegrounds, countries next to the enemy and totals, side by side -- and
each side's space race pill says what its next attempt needs.
"""
from __future__ import annotations

from typing import Any, Iterator

import pytest

from tests.web.test_e2e_board_display import mid_game_token  # noqa: F401 (fixture)
from tests.web.test_e2e_card_choices import browser, site  # noqa: F401 (fixtures)

MIDDLE_EAST = ["Lebanon", "Syria", "Israel", "Iraq", "Iran", "Libya", "Egypt", "Jordan", "Gulf States", "Saudi Arabia"]


@pytest.fixture
def phone(browser: Any, site: str, mid_game_token: str) -> Iterator[Any]:  # noqa: F811
    ctx = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    page = ctx.new_page()
    page.goto(f"{site}/?model=off&pos={mid_game_token}")
    page.wait_for_function("window.__wb && window.__wb.state && window.__wb.positionToken")
    yield page
    ctx.close()


def test_a_phone_scrolls_one_column_with_the_map_first(phone: Any) -> None:
    assert phone.evaluate("document.documentElement.scrollWidth") <= 390, "nothing wider than the screen"
    m, d, c = (phone.locator(sel).bounding_box() for sel in ("#map-container", "#decision-panel", "#cards-sidebar"))
    assert m["y"] + m["height"] <= d["y"] + 1 and d["y"] < c["y"], "map, then the decision, then the cards"
    assert m["width"] > 350 and m["height"] >= 380
    # The region buttons and the zoom buttons are on the map, not over each other.
    jump, zoom = phone.locator("#map-region-jump").bounding_box(), phone.locator(".map-viewport-controls").bounding_box()
    assert jump["y"] >= zoom["y"] + zoom["height"]
    assert m["y"] <= jump["y"] and jump["y"] + jump["height"] <= m["y"] + m["height"]


def test_a_region_button_frames_the_region(phone: Any) -> None:
    phone.tap("#map-region-jump [data-region='Middle East']")
    w = float(phone.get_attribute("#ts-map-svg", "viewBox").split()[2])
    assert w < 500, "zoomed in"
    box = phone.locator("#ts-map-svg").bounding_box()
    for name in MIDDLE_EAST:
        b = phone.locator(f".svg-country-node[data-name='{name}'] rect.country-card-bg").bounding_box()
        assert box["x"] <= b["x"] and b["x"] + b["width"] <= box["x"] + box["width"] + 1, name
        assert box["y"] <= b["y"] and b["y"] + b["height"] <= box["y"] + box["height"] + 1, name
    phone.tap("#btn-zoom-reset")
    assert phone.get_attribute("#ts-map-svg", "viewBox").split()[2] == "1000"


def test_a_tap_on_a_country_that_is_not_a_target_shows_its_card(phone: Any) -> None:
    phone.locator("#map-container").scroll_into_view_if_needed()
    phone.tap(".svg-country-node[data-name='East Germany'] rect.country-card-bg", force=True)
    tip = phone.locator("#card-tooltip")
    assert tip.is_visible()
    assert "East Germany" in tip.inner_text() and "USSR controls" in tip.inner_text()
    # A tap on open map closes it.
    empty = phone.evaluate("""() => { const svg = document.getElementById('ts-map-svg'), b = svg.getBoundingClientRect();
        for (let y = b.top + 10; y < b.bottom; y += 10) for (let x = b.left + 10; x < b.right; x += 10)
            if (document.elementFromPoint(x, y) === svg) return {x: x - b.left, y: y - b.top};
        return null; }""")
    assert empty, "some of the map is open water"
    phone.tap("#ts-map-svg", position=empty, force=True)
    assert not tip.is_visible()


def test_a_region_badge_opens_its_scoring_summary(phone: Any) -> None:
    """West Germany (US) and East Germany (USSR) are Europe's only controlled countries, both
    battlegrounds and neither next to the enemy superpower: Presence 3 + 1 each, an even 4 to 4."""
    phone.tap(".svg-region-badge[data-region='EUROPE'] .region-badge-bg", force=True)
    rs = phone.locator("#modal-body .rs")
    assert rs.is_visible()
    rows = {r.locator(".rs-label").inner_text(): (r.locator(".rs-us").inner_text(), r.locator(".rs-ussr").inner_text())
            for r in rs.locator(".rs-row").all()}
    assert rows["Standing"] == ("Presence · 3", "Presence · 3")
    assert rows["Battlegrounds of 5"] == ("1 · +1", "1 · +1")
    assert rows["Next to the enemy"] == ("0", "0")
    assert rows["Total"] == ("4", "4")
    assert rs.locator(".rs-net").inner_text() == "even"
    assert "West Germany" in rs.locator(".rs-list").first.inner_text()
    assert "East Germany" in rs.locator(".rs-list").nth(1).inner_text()
    # Europe's terms are quoted from the card itself, not restated by the page.
    assert "Control: Automatic Victory" in rs.inner_text()


def test_the_space_race_says_what_the_next_attempt_needs(phone: Any) -> None:
    """Each side's pill names the next box's Ops and roll (engine/src/space_race.cpp)."""
    need = {1: "2 Ops · 1–3", 2: "2 Ops · 1–4", 3: "2 Ops · 1–3", 4: "2 Ops · 1–4",
            5: "3 Ops · 1–3", 6: "3 Ops · 1–4", 7: "3 Ops · 1–3", 8: "4 Ops · 1–2"}
    space = phone.evaluate("window.__wb.state.space")
    ussr, us = phone.locator(".pill-next").all_inner_texts()
    assert ussr == f"next {need[space['USSR'] + 1]}" and us == f"next {need[space['US'] + 1]}"
