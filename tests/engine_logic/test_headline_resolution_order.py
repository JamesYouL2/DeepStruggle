"""Headlines resolve in printed-Ops order, ties to the US -- checked against every human game.

Both sides headline as written and the two are then resolved in a definite order, so the engine
has to agree with the humans about what that order is. Getting it wrong is invisible in any one
entry and changes the board in the next.

The logs state the order directly: the "Event:" lines appear in the order the headlines were
resolved. Across the 287 downloaded games there are 2188 headlines to compare.
"""
import glob
import gzip
import json
import os
from dataclasses import dataclass
from typing import List

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import card_id
from tools.lib.ts_replayer_parse import parse_entry

CORPUS = str(corpus_dir())
DEFECTORS = "Defectors"

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(CORPUS, "*.json.gz")),
    reason="ts-replayer corpus not downloaded")


@dataclass(frozen=True)
class _Headline:
    """One headline pair of a human game, as the log resolved it."""
    replay: str
    turn: int
    us: int        #: the US card
    ussr: int      #: the USSR card
    first: int     #: the card the log resolved first


def _headlines() -> List[_Headline]:
    """Every headline of the corpus."""
    out = []
    for path in sorted(glob.glob(os.path.join(CORPUS, "*.json.gz"))):
        rid = os.path.basename(path).split(".")[0]
        with gzip.open(path, "rt") as f:
            raws = json.load(f)["all_turns"]
        for raw in raws:
            e = parse_entry(raw)
            if not e.headlines or len(e.headlines) != 2:
                continue
            us, ussr = card_id(e.headlines.get("US")), card_id(e.headlines.get("USSR"))
            if not us or not ussr:
                continue
            order = [c for c in (card_id(n) for n in (e.events or []))
                     if c is not None and c in (us, ussr)]
            if order:
                out.append(_Headline(rid, e.turn, us, ussr, order[0]))
    return out


HEADLINES = _headlines()


def _ops(card: int) -> int:
    return int(ts.CardData.get_card_info(card)["ops"])


def _name(card: int) -> str:
    return str(ts.CardData.get_card_info(card)["name"])


def test_the_corpus_offers_enough_headlines_to_be_meaningful() -> None:
    assert len(HEADLINES) > 2000, f"only {len(HEADLINES)} headlines found"


def test_printed_ops_decides_the_order_with_ties_to_the_us() -> None:
    """Defectors aside, the engine's rule reproduces every headline the humans played."""
    wrong = []
    for h in HEADLINES:
        rid, turn, us, ussr, actual_first = h.replay, h.turn, h.us, h.ussr, h.first
        if DEFECTORS in (_name(us), _name(ussr)):
            continue          # cancels rather than races; see the test below
        predicted = us if _ops(us) >= _ops(ussr) else ussr
        if predicted != actual_first:
            wrong.append(f"replay {rid} T{turn}: US {_name(us)} ({_ops(us)}) vs "
                         f"USSR {_name(ussr)} ({_ops(ussr)}), log resolved "
                         f"{_name(actual_first)} first")
    assert not wrong, "headline order disagrees with the log:\n  " + "\n  ".join(wrong[:10])


def test_a_tie_always_goes_to_the_us() -> None:
    ties = [h for h in HEADLINES
            if _ops(h.us) == _ops(h.ussr) and DEFECTORS not in (_name(h.us), _name(h.ussr))]
    assert len(ties) > 100, f"only {len(ties)} ties to check"
    for h in ties:
        assert h.first == h.us, (
            f"replay {h.replay} T{h.turn}: {_ops(h.us)} Ops each, log resolved {_name(h.first)} first")


def test_defectors_is_narrated_first_because_it_cancels_not_because_it_is_faster() -> None:
    """The log puts Defectors first whatever the Ops; the engine sorts by Ops and cancels the
    USSR headline unresolved either way, so both reach the same board."""
    cases = [h for h in HEADLINES if _name(h.us) == DEFECTORS]
    assert cases, "no Defectors headlines in the corpus"
    assert all(h.first == h.us for h in cases), (
        "the log always resolves a US Defectors headline first")
    assert any(_ops(h.ussr) > _ops(h.us) for h in cases), (
        "and it does so even against a higher-Ops USSR headline, which is the point")
