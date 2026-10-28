"""Why a finished game ended -- the one classifier every report names endings with."""

import ts_engine as ts


def classify_game_ending_reason(state: ts.GameState) -> str:
    """Why a finished game ended, from the canonical list:
    - 20 VP
    - Europe Control
    - DEFCON 1 (own decision)
    - DEFCON 1 (opponent decision)
    - final scoring
    - wargames

    Defined once, in C++ (`ts::state_json::ending_reason`, bindings/state_json.cpp), because the
    browser workbench names the ending too and a report and the page must never disagree about
    it. The rules it applies -- CMC suicide, DEFCON 1 before VP, Wargames through turn 10, Europe
    Control before an ordinary 20 VP -- are documented there.
    """
    return ts.game_ending_reason(state)
