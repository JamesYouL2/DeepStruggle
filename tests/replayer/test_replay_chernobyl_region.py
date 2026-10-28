"""Chernobyl names its region in the log, and nothing else can tell you which it was.

The card designates a region and bars the USSR from adding Influence there for the rest of the
turn. Which region is a choice, and the log records it outright -- "US chooses South America".

Left to the branch search, which judges a branch by the board and the score it reaches, the
choice was arbitrary: Chernobyl moves neither. At turn 8 AR2 of replay 161 the US chose South
America and the engine took Europe, and the consequence surfaced three action rounds later,
somewhere else entirely -- the USSR played North Sea Oil for 3 Ops into East Germany and
Spain/Portugal, both in Europe, and placed nothing at all. The entry was reported as two
countries disagreeing about Influence that had never been barred.
"""
import ts_engine as ts

from tests.replay_corpus import ReplayCorpus
from tools.lib.ts_replayer_parse import REGIONS, parse_entry

EAST_GERMANY, SPAIN_PORTUGAL = 14, 9


def test_the_log_names_the_region(corpus: ReplayCorpus) -> None:
    e = corpus.entry(161, 8, "AR2", "US")
    assert e.card == "Chernobyl*"
    assert "US chooses South America" in (e.text or "")
    assert e.region_choice == REGIONS["South America"] == 5


def test_the_region_numbers_are_the_engines() -> None:
    """The branch is answered with this number, so it has to be the engine's ordering."""
    for name, index in REGIONS.items():
        assert str(ts.Region(index)).split(".")[-1].replace("_", " ").title() == name.title()


def test_an_entry_without_a_region_line_names_none(corpus: ReplayCorpus) -> None:
    assert corpus.entry(161, 8, "AR4", "USSR").region_choice is None


def test_the_barred_region_lets_north_sea_oil_place_where_the_log_places_it(corpus: ReplayCorpus) -> None:
    """Turn 8 AR4: 3 Ops into East Germany and Spain/Portugal, both in Europe."""
    e = corpus.entry(161, 8, "AR4", "USSR")
    assert e.card == "North Sea Oil*"
    touched = {row.country for row in e.influence}
    assert {EAST_GERMANY, SPAIN_PORTUGAL} <= touched
    conv = corpus.converted(161)
    assert conv.failure is None, f"replay 161 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total


def test_every_region_name_the_corpus_uses_is_understood(corpus: ReplayCorpus) -> None:
    """55 region choices across the corpus, in five of the six regions."""
    seen = set()
    for _, game in corpus.every_game():
        for raw in game.get("all_turns") or []:
            choice = parse_entry(raw).region_choice
            if choice is not None:
                seen.add(choice)
    assert seen, "no region choice parsed at all"
    assert seen <= set(REGIONS.values())
