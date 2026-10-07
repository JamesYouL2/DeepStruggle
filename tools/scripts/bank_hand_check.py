#!/usr/bin/env python3
"""Which disagreement-bank positions hold a hand the human did not have (BUGS.md CONV-1)?

The ts-replayer converter starts a turn with filler cards in a hand and adds the logged cards as
the turn needs them, without taking the fillers back out. A position is flagged when, for the
deciding side:

* **over** -- the hand (the China Card aside) holds more cards than the rules allow at that point:
  8 in turns 1-3 and 9 after, one fewer once the headline is spent;
* **missing** -- a card the log lists in that side's hand for the turn is still in the draw deck.

Since the CONV-1 fix, only **over** marks a bad hand. A card that arrives mid-turn (SALT
Negotiations, Missile Envy, Grain Sales, a reshuffle) is in the deck until it arrives, so
**missing** now flags correct positions too, and an exclusion list should be built from **over**
alone. The remaining **over** rows (58 of 27,753 after the fix) hold a card that arrived mid-turn.

Writes the flagged rows' ids with their reasons, and an exclusion list for
`disagreement_bank.py --pack --exclude` of the **over** rows -- and, with `--untrusted` (a
{replay id: first untrusted turn} map), every row from an untrusted turn. A **missing**-only row
is reported in `--flags` and kept.

    PYTHONPATH=.:build/release python tools/scripts/bank_hand_check.py --bank part*.jsonl.gz \
        --flags hand_flags.json --untrusted untrusted_turns.json --exclude exclude_ids.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from typing import Dict, List, Optional, Sequence, Set

import ts_engine as ts

from tools.lib.corpus_driver import state_from_token
from tools.lib.corpus_paths import corpus_path
from tools.lib.ts_replayer_convert import card_id
from tools.scripts.disagreement_bank import row_id

CHINA = 6


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--flags", required=True, help="JSON {row id: [reasons]} of the flagged rows")
    ap.add_argument("--untrusted", default=None, help="JSON {replay id: first untrusted turn}")
    ap.add_argument("--exclude", default=None, help="JSON list: flagged rows and untrusted-turn rows")
    a = ap.parse_args(argv)
    hands: Dict[int, Dict[str, Dict[str, List[str]]]] = {}
    untrusted: Dict[str, Optional[int]] = json.load(open(a.untrusted)) if a.untrusted else {}
    flags: Dict[str, List[str]] = {}
    over: Set[str] = set()
    late: Set[str] = set()
    why: collections.Counter = collections.Counter()
    total = 0
    for path in a.bank:
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                total += 1
                rid = row_id(r)
                u = untrusted.get(str(r["game"]))
                if u is not None and r["turn"] >= u:
                    late.add(rid)
                if r["game"] not in hands:
                    with gzip.open(corpus_path(r["game"]), "rt") as g:
                        hands[r["game"]] = json.load(g)["hands"]
                logged = {c for c in (card_id(n) for n in (hands[r["game"]].get(str(r["turn"])) or {})
                                      .get(r["side"].lower(), [])) if c}
                st = state_from_token(r["pos"])
                side = ts.Player.US if r["side"] == "US" else ts.Player.USSR
                held = [c for c in range(1, 111) if c != CHINA and ts.in_hand_of(st.get_card_location(c), side)]
                limit = (8 if r["turn"] <= 3 else 9) - (0 if r["kind"] == "headline" else 1)
                reasons = []
                if len(held) > limit:
                    reasons.append("over")
                    over.add(rid)
                if any(st.get_card_location(c) == ts.CardLocation.DRAW_DECK for c in logged):
                    reasons.append("missing")
                if reasons:
                    flags[rid] = reasons
                    why["+".join(reasons)] += 1
    with open(a.flags, "w") as f:
        json.dump(flags, f)
    print(f"{len(flags)} of {total} rows flagged: {dict(why)}; {len(late)} rows in untrusted turns")
    if a.exclude:
        excluded = sorted(over | late)
        with open(a.exclude, "w") as f:
            json.dump(excluded, f)
        print(f"{len(excluded)} rows excluded (over the limit, or in an untrusted turn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
