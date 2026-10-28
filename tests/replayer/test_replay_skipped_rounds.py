"""Action rounds the log records by leaving them empty, and games it is not worth converting.

A player who has run out of cards skips their action round. The log gives it a header and
nothing beneath -- "Turn 5, USSR AR4" on a line of its own at the foot of the entry above --
and writes no entry of its own, so the round goes missing from the record and the next entry
belongs to the other player. There are 95 such rounds across 69 games.

Driving the pass matters for what follows it: at turn 10 AR7 of replay 105 the US passes the
last action round of the game, which ends turn 10 and runs the final scoring the log records
as 11 VP. Without it the game stopped 11 VP short of its own result.

Where the log skips a round and the reconstruction still holds cards, the two disagree about
the hand and that is reported rather than passed over -- passing with cards is not legal, and
a reconstruction that needs it to be is wrong somewhere earlier.
"""
import pytest

from tools.lib.ts_replayer_convert import distinct_replays, unsupported_handicap
from tools.lib.ts_replayer_parse import parse_entry
from tests.replay_corpus import ReplayCorpus



def test_a_bare_action_round_header_is_read_as_a_pass(corpus: ReplayCorpus) -> None:
    """Replay 114 turn 5: the USSR runs out and skips AR4 through AR7."""
    skipped = []
    for raw in corpus.turns(114):
        skipped.extend(parse_entry(raw).passed_rounds)
    assert (5, "USSR", 4) in skipped
    assert [ar for turn, side, ar in skipped if turn == 5 and side == "USSR"] == [4, 5, 6, 7]


def test_the_entrys_own_header_is_not_a_pass(corpus: ReplayCorpus) -> None:
    """"Turn 5, US AR3: Our Man in Tehran*: ..." carries on past the end of the line."""
    entry = parse_entry(corpus.turns(114)[59])
    assert entry.turn == 5 and entry.player == "US"
    assert (5, "US", 3) not in entry.passed_rounds
    assert entry.passed_rounds == [(5, "USSR", 4)]


def test_the_last_pass_of_a_game_lets_the_final_scoring_run(corpus: ReplayCorpus) -> None:
    """Replay 105: the US skips turn 10 AR7, and final scoring gives them 11 VP.

    The log's last line is "US gains 11 VP. Score is US 19." Stopping at the USSR's play left
    the game at 8 and the whole entry unconverted.
    """
    last = parse_entry(corpus.turns(105)[-1])
    assert last.passed_rounds == [(10, "US", 7)]
    conv = corpus.converted(105)
    assert conv.failure is None, f"replay 105 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total


@pytest.mark.parametrize("replay_id", [105, 116, 152])
def test_the_skipped_round_games_convert_end_to_end(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"


def test_a_game_played_with_a_handicap_the_engine_cannot_set_up_is_not_converted(corpus: ReplayCorpus) -> None:
    """Replay 196 opens "Handicap influence: US +5"; the engine's setup grants 2.

    Supporting it would mean making the handicap a setup parameter, and a handicap other than
    the tournament 2 says the players were mismatched -- the positions are not ones the engine
    will play from, so the game is left out rather than approximated.
    """
    assert unsupported_handicap(corpus.turns(196)) == "US +5"
    assert unsupported_handicap(corpus.turns(100)) is None
    conv = corpus.converted(196)
    assert conv.skipped is not None and "US +5" in conv.skipped
    assert conv.entries_total == 0
    assert conv.failure is None, "not converting is not the same as failing to convert"


def test_a_handicap_won_at_auction_counts_the_same(corpus: ReplayCorpus) -> None:
    """Some games bid for sides, and the winning bid is the same extra Influence.

    "lkslks bids 1 Influence for USSR ... Additional Influence from bidding: US +1" is a
    handicap of 1, and reading only the "Handicap influence:" form left those games looking
    standard: at turn 1 of replay 230 the US has 8 Influence to place where the engine offers
    9, and the handicap stage was left with a placement the log never made.
    """
    assert unsupported_handicap(corpus.turns(230)) == "US +1"
    assert unsupported_handicap(corpus.turns(250)) == "US +3"
    conv = corpus.converted(230)
    assert conv.skipped is not None and "US +1" in conv.skipped
    assert conv.failure is None


def test_a_bid_that_lands_on_two_is_the_standard_setup(corpus: ReplayCorpus) -> None:
    """18 of the corpus's games bid their way to the same 2 the tournament rules give."""
    assert unsupported_handicap(corpus.turns(136)) is None
    conv = corpus.converted(136)
    assert conv.skipped is None
    assert conv.failure is None, f"replay 136 stopped at {conv.failure}"


def test_duplicate_downloads_resolve_to_one_game(corpus: ReplayCorpus) -> None:
    """300 files, 266 games. One game arrived nine times over."""
    games = {rid: game["all_turns"] for rid, game in corpus.every_game()}
    mapping = distinct_replays(games)
    assert mapping[106] == 105, "replay 106 is replay 105 again"
    assert mapping[105] == 105, "the lowest id is the one kept"
    assert mapping[110] == 109
    nine = [rid for rid, orig in mapping.items() if orig == 90]
    assert sorted(nine) == [90, 214, 215, 216, 217, 218, 254, 298, 309]
    assert len(set(mapping.values())) == len(mapping) - 34, (
        "34 of the downloaded files repeat a game already in the set")
