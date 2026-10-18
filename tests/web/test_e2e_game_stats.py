"""The page's side of the stats database: the game-over dialog, and a reload keeping the game.

A finished game opens the result dialog once (index.html #result-dialog, result_dialog.ts) with
how the game was played; its Escape closes it without taking a move back, and the Result button
opens it again. A build with no TS_STATS_ENDPOINT -- this one -- offers the download only. A
reload plays this browser's stored game back (game/saved_game.ts): same id, same steps, undo
still there, not a game from a loaded position. The record itself, and its replay in Python, are
tests/web/test_game_stats.py.

Needs web/ui/dist (tools/scripts/build_web.sh) and a Playwright Chromium.
"""
from __future__ import annotations

from typing import Any

from tests.web.test_e2e_workbench import _open, browser, static_site  # noqa: F401 -- fixtures

#: Play the US by clicks (the human) and the USSR by auto-play, first legal move, until `stop`.
_PLAY = """stop => { const wb = window.__wb, until = new Function('s', 'return ' + stop);
    for (let n = 0; n < 5000 && !wb.state.is_terminal && !until(wb.state); n++) {
        const m = wb.engine.mask(false), i = m.findIndex(x => x);
        if (wb.engine.decisionPlayer() < 0) wb.sendFlatAction(i, 0, true);
        else wb.sendAction(wb.engine.decodeFlat(i));
    }
    return wb.session.steps.length; }"""

#: The link names the position on the board (the token is written after each move, off the path).
_SYNCED = """() => { const wb = window.__wb;
    return !!wb.liveState.position && new URLSearchParams(location.search).get('pos') === wb.liveState.position; }"""


def test_a_finished_game_offers_its_result_once(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off&view=us")
    steps = page.evaluate(_PLAY, "false")
    assert page.evaluate("window.__wb.state.is_terminal")
    page.wait_for_selector("#result-dialog:not(.hidden)")
    outcome = page.inner_text("#result-dialog-outcome")
    assert "wins" in outcome or "Draw" in outcome
    assert "you won" in outcome or "you lost" in outcome or "Draw" in outcome
    assert "You played" in page.inner_text("#result-dialog-facts")
    # Auto-play moved with no model here, so the game is not clean -- and says why.
    assert "no model loaded" in page.inner_text("#result-dialog-notes")
    # No endpoint in this build: the record can be downloaded, not sent.
    assert page.is_disabled("#result-submit")
    assert "download" in page.inner_text("#result-status").lower()

    # Escape closes it and takes nothing back; it does not open again by itself.
    page.keyboard.press("Escape")
    page.wait_for_selector("#result-dialog", state="hidden")
    assert page.evaluate("window.__wb.session.steps.length") == steps
    page.evaluate("window.__wb.refresh()")
    assert page.locator("#result-dialog.hidden").count() == 1
    # The Result button opens it again.
    page.click("#btn-game-result")
    page.wait_for_selector("#result-dialog:not(.hidden)")
    assert page.errors == []


#: Click first legal moves for both sides, a target click carrying the HUD's die as the map's does,
#: until a step rolls with a forced die. Returns that step's tag and the selector after it, and the
#: selector after the last step that did not roll.
_UNTIL_FORCED_ROLL = """() => { const wb = window.__wb;
    wb.actionHud.selectedDieRoll = 6;
    let kept = null;
    for (let n = 0; n < 5000 && !wb.state.is_terminal; n++) {
        const i = wb.engine.mask(false).findIndex(x => x), a = wb.engine.decodeFlat(i);
        if (a.decision_type === 5) a.secondary_id = wb.actionHud.selectedDieRoll;
        wb.sendAction(a);
        const t = wb.session.tags[wb.session.tags.length - 1];
        if (t.forcedDie > 0 && t.rolled) return {tag: t, after: wb.actionHud.selectedDieRoll, kept};
        kept = wb.actionHud.selectedDieRoll;
    }
    return null; }"""


def test_a_forced_die_is_for_one_roll(browser: Any, static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off")
    out = page.evaluate(_UNTIL_FORCED_ROLL)
    assert out is not None, "no target click rolled a die"
    assert out["kept"] == 6, "a move that rolled nothing keeps the forced face"
    assert out["tag"]["forcedDie"] == 6 and out["after"] == 0
    assert page.errors == []


def test_a_reload_keeps_the_game_and_its_history(browser: Any, static_site: str) -> None:
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = _open(browser, f"{static_site}/?model=off&view=us", context=context)
    page.evaluate(_PLAY, "s.turn >= 2")
    page.wait_for_function(_SYNCED)
    before = page.evaluate("({id: window.__wb.session.id, steps: window.__wb.session.steps.length, "
                           "save: window.__wb.engine.saveJson()})")
    assert before["steps"] > 20

    page.reload()
    page.wait_for_function("window.__wb && window.__wb.liveState && window.__wb.positionToken", timeout=30000)
    after = page.evaluate("({id: window.__wb.session.id, steps: window.__wb.session.steps.length, "
                          "save: window.__wb.engine.saveJson(), start: window.__wb.session.startPosition, "
                          "undo: window.__wb.session.canUndo, "
                          "last: window.__wb.session.events[window.__wb.session.events.length - 1]?.kind})")
    assert after == {**before, "start": None, "undo": True, "last": "resume"}
    # The record goes on from there: the next move is step steps + 1 of the same game.
    page.evaluate(_PLAY, f"window.__wb.session.steps.length > {before['steps']}")
    assert page.evaluate("window.__wb.session.id") == before["id"]
    assert page.errors == []
    context.close()
