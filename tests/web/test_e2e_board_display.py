"""What the board says at a glance: the effects that last for this turn, who controls what, what
each region would score, and the latest dice.

The turn's effects sit on top of the map, where they stay in view; the side panel still lists
everything. A country's controller is carried by the card's fill, because a country that can be
chosen has its border replaced by the pulsing highlight -- the moment control matters most. Its
influence markers follow the physical board: US on the left, USSR on the right, a controlling
marker flipped (white on the side's colour), an empty space blank.
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
    # Exactly these effects, not added to the walk's own: what the walk's game has in effect moves
    # with every change to the deal (since ENG-5 it holds a Brezhnev Doctrine, a second chip).
    bits = ts.EffectBits
    save["persistent_effects"] = (int(bits.CONTAINMENT_ACTIVE) | int(bits.NATO_ACTIVE)
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


def test_an_empty_space_is_blank_and_the_us_is_on_the_left(browser: Any, site: str, mid_game_token: str) -> None:
    page = _open(browser, site, f"pos={mid_game_token}")
    g = ".svg-country-node[data-name='East Germany']"
    us = page.locator(f"{g} rect.inf-us")
    ussr = page.locator(f"{g} rect.inf-ussr")
    assert "empty" in (us.get_attribute("class") or "")
    assert "empty" not in (ussr.get_attribute("class") or "")
    assert float(us.get_attribute("x") or 0) < float(ussr.get_attribute("x") or 0)
    # One number, the USSR's 3, white on red: the marker that controls is flipped.
    assert page.locator(f"{g} .inf-num").all_text_contents() == ["3"]
    assert ussr.get_attribute("fill") == "#DC2626"
    page.close()


def test_the_hover_card_says_what_control_costs(browser: Any, site: str, mid_game_token: str) -> None:
    """East Germany, USSR 3 against stability 3: the US breaks control with 1 point at 2 Ops."""
    page = _open(browser, site, f"pos={mid_game_token}")
    page.hover(".svg-country-node[data-name='East Germany'] rect.country-card-bg", force=True)
    text = page.inner_text("#card-tooltip")
    assert "USSR controls, with nothing to spare" in text
    assert "US breaks with +1 (2 Ops), controls with +6 (7 Ops)" in text
    assert "Adjacent:" in text and "Poland" in text
    # The neighbours light up and everything else steps back.
    assert "country-hovering" in (page.get_attribute("#ts-map-svg", "class") or "")
    assert "hover-neighbour" in (page.get_attribute(".svg-country-node[data-name='Poland']", "class") or "")
    page.close()


#: From a fixed deal, play random legal moves (a fixed sequence) until a coup has been rolled.
_PLAY_TO_COUP = """seed => { const wb = window.__wb; let x = seed;
    wb.session.newGame(seed); wb.refresh();
    const rnd = () => { x = (x * 1103515245 + 12345) % 2147483648; return x / 2147483648; };
    for (let n = 0; n < 5000 && !wb.state.is_terminal; n++) {
        const m = wb.engine.mask(false), legal = [];
        m.forEach((v, i) => v && legal.push(i));
        wb.sendFlatAction(legal[Math.floor(rnd() * legal.length)], 0, true);
        const coup = (wb.shown.recent_rolls || []).find(r => r.type === 'COUP');
        if (coup) return coup;
    }
    return null; }"""


def test_the_latest_dice_are_on_the_map(browser: Any, site: str) -> None:
    """The engine keeps only the last step's roll; the tray reads the rolls from the game's steps."""
    page = _open(browser, site, "")
    coup = page.evaluate(_PLAY_TO_COUP, 4242)
    assert coup, "random play from this deal rolls a coup"
    row = page.locator("#map-rolls .roll-row[data-type='COUP']").last
    assert row.is_visible()
    assert row.locator(".roll-where").inner_text() == coup["country_name"]
    body = row.locator(".roll-body").inner_text()
    assert f"= {coup['total1']}" in body.replace("\n", " ") and str(coup["mod2"]) in body
    assert page.locator(f"#map-rolls .die[aria-label='{coup['roller']} rolls {coup['roll1']}']").count() >= 1
    node = f".svg-country-node[data-name='{coup['country_name']}']"
    assert page.get_attribute(node, "data-roll") == "COUP"
    # Rolls are recent only for their half action round and the next: past the next turn's headline
    # (the half round after the turn's last action round), none is older than this turn.
    turn = page.evaluate("window.__wb.state.turn")
    page.evaluate("""turn => { const wb = window.__wb;
        for (let n = 0; n < 5000 && !wb.state.is_terminal
                 && (wb.state.turn === turn || wb.state.current_phase_name !== 'ACTION_ROUND'); n++) {
            const m = wb.engine.mask(false); wb.sendFlatAction(m.findIndex(x => x), 0, true);
        } }""", turn)
    now = page.evaluate("window.__wb.state.turn")
    whens = page.evaluate("(window.__wb.shown.recent_rolls || []).map(r => r.when)")
    assert all(w.startswith(f"T{now} ") for w in whens), whens
    if not whens:
        assert page.locator("#map-rolls").get_attribute("class").split().count("hidden") == 1
    page.close()


#: Draw the map with these rolls in the tray, as if they were the game's latest.
_RENDER_ROLLS = "rolls => { const wb = window.__wb; wb.mapView.render({ ...wb.shown, recent_rolls: rolls }); }"


def test_a_space_race_roll_marks_no_country(browser: Any, site: str) -> None:
    """A Space Race roll's country_id is the box it went for (engine/src/space_race.cpp), not a country."""
    page = _open(browser, site, "")
    page.evaluate(_RENDER_ROLLS, [{"type": "SPACE_RACE", "roller": "USSR", "country_id": 7, "roll1": 2,
                                   "mod1": 3, "success": True, "when": "T1 AR1"}])
    assert page.locator("#map-rolls .roll-row[data-type='SPACE_RACE'] .roll-where").inner_text() == "box 7"
    assert page.locator(".svg-country-node[data-roll]").count() == 0
    page.close()


def test_a_redraw_puts_a_tapped_card_away(browser: Any, site: str, mid_game_token: str) -> None:
    """The nodes a tap lit go with a redraw (the other side's move): the map must not stay dimmed.

    A tap, not a hover: a mouse still over the map gets a new mouseenter from the redrawn node.
    """
    page = _open(browser, site, f"pos={mid_game_token}")
    page.evaluate("""() => { window.__wb.mapView.lastPointer = 'touch';
        document.querySelector(".svg-country-node[data-name='East Germany']")
            .dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: 300, clientY: 200 })); }""")
    assert "country-hovering" in (page.get_attribute("#ts-map-svg", "class") or "")
    assert "hidden" not in (page.get_attribute("#card-tooltip", "class") or "")
    page.evaluate("window.__wb.mapView.render(window.__wb.shown)")
    assert "country-hovering" not in (page.get_attribute("#ts-map-svg", "class") or "")
    assert "hidden" in (page.get_attribute("#card-tooltip", "class") or "")
    page.close()


def test_a_drag_released_outside_the_window_ends(browser: Any, site: str) -> None:
    """The release sent no pointerup to the page: the mouse comes back with no button down."""
    page = _open(browser, site, "")
    view = page.evaluate("""() => { const svg = document.getElementById('ts-map-svg');
        const ev = (type, x, buttons) => new PointerEvent(type, { pointerId: 1, pointerType: 'mouse', bubbles: true,
                                                                  button: 0, buttons, clientX: x, clientY: 300 });
        svg.dispatchEvent(ev('pointerdown', 400, 1));
        window.dispatchEvent(ev('pointermove', 500, 1));
        const dragged = svg.getAttribute('viewBox');
        window.dispatchEvent(ev('pointermove', 600, 0));
        window.dispatchEvent(ev('pointermove', 700, 0));
        return { dragged, after: svg.getAttribute('viewBox'), pressed: window.__wb.mapView.pointers.size }; }""")
    assert view["after"] == view["dragged"], "a move with no button down pans nothing"
    assert view["pressed"] == 0
    page.close()


def test_the_space_perks_are_named_as_their_boxes(browser: Any, site: str) -> None:
    """Box 4 is Man in Earth Orbit and box 8 the Space Station (engine/src/space_race.cpp)."""
    save = _us_action_round(7).to_save_dict()
    save.update(us_space_track=8, ussr_space_track=3)
    page = _open(browser, site, f"pos={_token(ts.state_from_save_dict(save))}")
    titles = page.locator("#active-flags-container .effect-title").all_inner_texts()
    assert {"US: Man in Earth Orbit (#4)", "US: Eagle Has Landed (#6)", "US: Space Station (#8)"} <= set(titles)
    assert page.locator(".pill-next").all_inner_texts()[1] == "done"
    page.close()
