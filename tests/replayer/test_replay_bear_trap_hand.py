"""A short hand list is not always a truncated record, and an exchanged card changes sides.

At turn 5 of replay 114 the USSR is caught by Bear Trap. They discard Five Year Plan at AR1 and
Che at AR2, each time rolling above 4 and staying caught; the US takes Nuclear Test Ban off
them with Missile Envy at the US's AR1; and by AR3 the only card they hold is Central America
Scoring, which cannot be discarded to a trap, so they play it. Then they skip AR4 through AR7
with nothing left.

Two things stopped that from being reconstructed:

* The turn's hand list holds five cards where a turn 5 hand holds nine, and padding read that
  as a truncated record and invented four -- all with 2 Ops or more, and a trap is escaped by
  discarding one. The engine offered those inventions and never offered the scoring card. A
  short list anywhere but the turn the record stops in has ordinary causes; this is one.
* Missile Envy's exchange was not tracked, so Nuclear Test Ban went back into the USSR's hand
  at the next entry and remained discardable two action rounds after it had changed hands.
"""
import pytest
import ts_engine as ts

from tools.lib.ts_replayer_parse import parse_entry
from tests.replay_corpus import ReplayCorpus

CENTRAL_AMERICA_SCORING = 37
NUCLEAR_TEST_BAN = 34


def test_the_log_records_the_trap_and_the_exchange(corpus: ReplayCorpus) -> None:
    raws = corpus.turns(114)
    assert "Trap Roll: 5 > 4 -- Trap Remains in Effect" in (raws[54].get("text") or "")
    assert ("USSR", "Nuclear Test Ban") in parse_entry(raws[55]).revealed
    assert parse_entry(raws[55]).card == "Missile Envy"


def test_the_ussr_holds_only_the_scoring_card_when_they_play_it(corpus: ReplayCorpus) -> None:
    """Turn 5 AR3: five cards were logged for the turn and one is left by now."""
    entry = parse_entry(corpus.turns(114)[58])
    assert entry.turn == 5 and entry.phase == "AR3" and entry.player == "USSR"
    assert entry.card == "Central America Scoring"


def test_replay_114_converts_end_to_end(corpus: ReplayCorpus) -> None:
    """It stopped at turn 5 AR3, unable to select a card the log says was played."""
    conv = corpus.converted(114)
    assert conv.failure is None, f"replay 114 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 124


def test_the_action_rounds_after_the_trap_convert_too(corpus: ReplayCorpus) -> None:
    """Turn 5 AR4 through AR7 are the US's alone, and they are all in the record."""
    raws = corpus.turns(114)
    tail = [(parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
            for r in raws[60:64]]
    assert tail == [(5, "AR4", "US"), (5, "AR5", "US"),
                    (5, "AR6", "US"), (5, "AR7", "US")]
    conv = corpus.converted(114)
    assert conv.entries_converted > 63, (
        "every one of them replays to the log's own board and score")


def test_padding_is_confined_to_the_turn_the_record_stops_in(corpus: ReplayCorpus) -> None:
    """Replay 119 stops mid turn 7 and is the case padding exists for."""
    raws = corpus.turns(119)
    last_turn = max(int(r["num"]) for r in raws if str(r.get("num", "")).isdigit())
    assert last_turn == 7
    conv = corpus.converted(119)
    assert conv.failure is None, f"replay 119 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [114, 115, 119])
def test_the_hand_tracking_games_convert(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
