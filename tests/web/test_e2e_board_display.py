"""What the board says at a glance: the effects that last for this turn, and who controls what.

The turn's effects sit on top of the map, where they stay in view; the side panel still lists
everything. A country's controller is carried by the card's fill, because a country that can be
chosen has its border replaced by the pulsing highlight -- the moment control matters most.
"""
from __future__ import annotations

from typing import Any

import pytest

import ts_engine as ts
from tests.web.test_e2e_card_choices import _token, _us_action_round, browser, site  # noqa: F401 (fixtures)

US_FILL, USSR_FILL = "rgb(30, 64, 175)", "rgb(153, 27, 27)"


def _country(name: str) -> int:
    return next(i for i in range(84) if ts.MapData.get_country_name(i) == name)


def _open(browser: Any, site: str, query: str) -> Any:
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(f"{site}/?model=off&{query}")
    page.wait_for_function("window.__wb && window.__wb.state && window.__wb.positionToken")
    return page


@pytest.fixture(scope="module")
def mid_game_token() -> str:
    """A mid-game position with one effect of each kind and three countries of known control."""
    s = _us_action_round(7)
    s.set_country(_country("West Germany"), 4, 0)   # US control (stability 4)
    s.set_country(_country("East Germany"), 0, 3)   # USSR control (stability 3)
    s.set_country(_country("France"), 1, 1)         # influence on both sides, no control
    save = s.to_save_dict()
    bits = ts.EffectBits
    save["persistent_effects"] |= (int(bits.CONTAINMENT_ACTIVE) | int(bits.NATO_ACTIVE)
                                   | int(bits.QUAGMIRE_ACTIVE) | int(bits.SPACE_US_ATTEMPT_1))
    return _token(ts.state_from_save_dict(save))


def test_this_turns_effects_are_listed_on_top_of_the_map(browser: Any, site: str, mid_game_token: str) -> None:
    page = _open(browser, site, f"pos={mid_game_token}")
    flags = page.evaluate("window.__wb.state.flags")
    assert {"CONTAINMENT_ACTIVE", "NATO_ACTIVE", "QUAGMIRE_ACTIVE", "SPACE_US_ATTEMPT_1"} <= set(flags)
    strip = page.locator("#map-turn-effects")
    assert strip.is_visible()
    chips = strip.locator(".turn-effect")
    assert [chips.nth(i).get_attribute("data-flag") for i in range(chips.count())] == ["CONTAINMENT_ACTIVE"], (
        "only this turn's effects: not NATO (permanent), Quagmire (until it ends) or a space attempt")
    assert chips.first.inner_text() == "Containment"
    assert "+1 Op" in chips.first.get_attribute("title")

    # On the map, at its top, and clear of the zoom buttons.
    m, s, z = (page.locator(sel).bounding_box() for sel in ("#map-container", "#map-turn-effects", ".map-viewport-controls"))
    assert m["y"] <= s["y"] <= m["y"] + 20 and s["x"] >= z["x"] + z["width"]
    page.close()


def test_a_turn_without_effects_says_so(browser: Any, site: str) -> None:
    page = _open(browser, site, "")
    assert page.locator("#map-turn-effects .turn-effect").count() == 0
    assert "no effects" in page.inner_text("#map-turn-effects")
    page.close()


def test_control_is_the_countrys_fill(browser: Any, site: str, mid_game_token: str) -> None:
    page = _open(browser, site, f"pos={mid_game_token}")
    fill = lambda name: page.evaluate(
        f"getComputedStyle(document.querySelector(\".svg-country-node[data-name='{name}'] rect.country-card-bg\")).fill")
    control = lambda name: page.get_attribute(f".svg-country-node[data-name='{name}']", "data-control")
    assert control("West Germany") == "US" and fill("West Germany") == US_FILL
    assert control("East Germany") == "USSR" and fill("East Germany") == USSR_FILL
    assert control("France") == "NONE" and fill("France") not in (US_FILL, USSR_FILL)
    # The controller's influence box is outlined in white.
    assert page.get_attribute(".svg-country-node[data-name='West Germany'] rect.inf-us", "stroke") == "#FFFFFF"
    assert page.get_attribute(".svg-country-node[data-name='West Germany'] rect.inf-ussr", "stroke") != "#FFFFFF"
    page.close()


def test_a_highlighted_target_still_shows_its_controller(browser: Any, site: str) -> None:
    """The USSR setup: East Germany starts USSR-controlled and is one of the countries to choose."""
    page = _open(browser, site, "")
    g = ".svg-country-node[data-name='East Germany']"
    assert "legal-target" in page.get_attribute(g, "class")
    assert page.get_attribute(g, "data-control") == "USSR"
    assert page.evaluate(f"getComputedStyle(document.querySelector(\"{g} rect.country-card-bg\")).fill") == USSR_FILL
    page.close()


def test_the_iranian_hostage_crisis_is_named(browser: Any, site: str) -> None:
    """The engine calls the bit IRANIAN_HOSTAGE_CRISIS_PLAY; the side panel used to show it raw."""
    s = _us_action_round(7)
    save = s.to_save_dict()
    save["persistent_effects"] |= int(ts.EffectBits.IRANIAN_HOSTAGE_CRISIS_PLAY)
    page = _open(browser, site, f"pos={_token(ts.state_from_save_dict(save))}")
    assert "Iranian Hostage Crisis" in page.inner_text("#active-flags-container")
    page.close()
