"""The page's action log says what each click did, the turn's last one included.

An action round that ends the turn stops on the turn's cleanup, a chance node the page resolves in
the same click: final VP, the military-ops reset, DEFCON, the next deal. Every engine step clears the
roll record, so read from the end state the turn's last realignment or coup lost its dice, and the
turn's end read as part of it. The page now describes the click from the engine's settled state (the
WebAssembly wrapper's `ts_settled_display_json`), logs the turn's end as its own entry, and prints
every die the click rolled (`ts_click_rolls_json`) -- a realignment with the opponent's war card rolls
twice. Games are seeded random play with realignments steered in late rounds, through the page.

Needs web/ui/dist (tools/scripts/build_web.sh) and a Playwright Chromium.
"""
from __future__ import annotations

from typing import Any, Dict, List

from tests.web.test_e2e_workbench import _open, browser, static_site  # noqa: F401 -- fixtures

#: Seeded random legal play (realignment, flat 114, whenever legal from round 6), until the first
#: realignment target choice for which `want(entries, before, after)` holds; returns that click's
#: log entries and the turns around it.
_FIND = """want => {
  const wb = window.__wb, ok = new Function('added', 'before', 'after', 'return ' + want);
  let x = 4242; const rnd = n => { x = (x * 1103515245 + 12345) % 2147483648; return x % n; };
  for (let g = 0; g < 40; g++) {
    wb.session.newGame(1000 + g); wb.refresh();
    for (let n = 0; n < 6000 && !wb.state.is_terminal; n++) {
      const before = wb.state, ctx = before.decision_context;
      const m = wb.engine.mask(false), legal = [];
      m.forEach((on, i) => { if (on) legal.push(i); });
      const a = (before.action_round >= 6 && legal.includes(114)) ? 114 : legal[rnd(legal.length)];
      const from = wb.session.logs.length;
      wb.sendFlatAction(a, 0, true);
      const after = wb.state;
      const realign = ctx.decision_type_name === 'POINT_NODE' && (ctx.resolving_card || 0) === 0 && ctx.op_mode === 2;
      const added = wb.session.logs.slice(from);
      if (realign && ok(added, before, after)) return { added, turn: before.turn, after_turn: after.turn };
    }
  }
  return null;
}"""


def _find(page: Any, want: str) -> Dict[str, Any]:
    found = page.evaluate(_FIND, want)
    assert found, f"no realignment matching {want!r} in 40 seeded games"
    return found


def test_the_turns_last_realignment_keeps_its_dice_and_the_turn_end_is_its_own_entry(browser: Any,
                                                                                     static_site: str) -> None:
    page = _open(browser, f"{static_site}/?model=off")
    found = _find(page, "after.turn !== before.turn")
    added: List[Dict[str, Any]] = found["added"]
    action, rest = added[0], added[1:]
    assert "Realignment" in action["text"]
    assert any("🎲 Realignment" in d for d in action.get("details") or []), "the realignment's dice"
    assert not any("DRAW_DECK -> HAND" in d for d in action.get("details") or []), "the deal is not the realignment's"
    turn_end = [e for e in rest if e["phase"] == "TURN_END"]
    assert turn_end and turn_end[0]["text"] == f"End of turn {found['turn']}"
    assert any("DRAW_DECK -> HAND" in d for d in turn_end[0].get("details") or []), "the next deal is the turn's end"
    assert page.errors == []
    page.close()


def test_a_click_that_rolls_twice_logs_both_dice(browser: Any, static_site: str) -> None:
    """A realignment with the opponent's war card: the realignment rolls, then the war's event does."""
    page = _open(browser, f"{static_site}/?model=off")
    found = _find(page, "(added[0].details || []).some(d => d.includes('War Roll'))")
    details = found["added"][0].get("details") or []
    dice = [d for d in details if "🎲" in d]
    assert len(dice) == 2 and "Realignment" in dice[0] and "War Roll" in dice[1], dice
    assert page.errors == []
    page.close()
