#!/usr/bin/env python3
"""A blind human judge for P36c's departures: rows for tools/scripts/distill_label.html.

The rollout root's held-out picks that leave the network's move on both seeds (`pick == pick2 !=
net_argmax`, the "stable departures") are either a strength the network lacks or a bias the playouts
share. A human reads each one as two cards, A and B, in a random order and with no evidence, and says
which is better. Departures are drawn stratified by the pick's paired lead (thirds), the period (turns
1-3, 4-7, 8-10) and whether the critic alone (`h0`) agrees with the pick; controls -- a candidate the
root rates clearly below the network's move on both seeds -- are mixed in blind, to read the judge's own
agreement with the root where the root endorses the network.

Two outputs: `--rows` (JSON for the page: id, position token, what the board shows, the two cards) and
`--key` (which card is the root's, the row's stratum and its leads), kept apart so the page carries
nothing that unblinds it. The verdicts are joined back to the key by id.

    PYTHONPATH=.:build/release .venv/bin/python tools/scripts/distill_label_bank.py \\
        --targets data/reports/distill_ladder/40244320871/targets.npz --teacher h4w128 \\
        --rows label_rows.json --key label_key.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import position_token
from tools.lib.position_bank import CARD_NAMES, move_name


def lead_of(cand: np.ndarray, lead: np.ndarray, se: np.ndarray, a: int) -> Tuple[float, float]:
    j = np.flatnonzero(cand == a)
    return (float(lead[j[0]]), float(se[j[0]])) if j.size else (float("nan"), float("nan"))


def period(turn: int) -> str:
    return "early" if turn <= 3 else "mid" if turn <= 7 else "late"


def describe(st: ts.GameState) -> Dict[str, Any]:
    """What the page prints beside the link: whose decision, when, and the mover's legal cards."""
    side = "US" if st.ctx().decision_player == ts.Player.US else "USSR"
    headline = st.current_phase == ts.Phase.HEADLINE
    mask = ts.ActionMask.generate_flat_mask(st)
    off = int(ActionEncoder.PLAY_MODE_OFFSET)
    hand = sorted({CARD_NAMES.get(int(ts.decode_flat_action(st, int(a)).primary_id), "?")
                   for a in np.flatnonzero(mask) if a < off})
    vp = int(st.victory_points)
    return {"side": side, "turn": int(st.turn), "ar": 0 if headline else int(st.action_round),
            "defcon": int(st.defcon), "vp": vp, "hand": hand}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", required=True)
    p.add_argument("--teacher", default="h4w128")
    p.add_argument("--departures", type=int, default=100)
    p.add_argument("--controls", type=int, default=15)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--rows", required=True)
    p.add_argument("--key", required=True)
    a = p.parse_args()

    z = np.load(a.targets)
    t = a.teacher
    held = z["heldout"] & (z[f"worlds_{t}"] > 0) & (z[f"worlds2_{t}"] > 0)
    net, pick, pick2 = z["net_argmax"], z[f"pick_{t}"], z[f"pick2_{t}"]
    rng = np.random.default_rng(a.seed)
    states = z["states"]

    def leads(i: int, move: int) -> Tuple[float, float, float, float]:
        l1, s1 = lead_of(z[f"cand_{t}"][i], z[f"lead_{t}"][i], z[f"lead_se_{t}"][i], move)
        l2, s2 = lead_of(z[f"cand2_{t}"][i], z[f"lead2_{t}"][i], z[f"lead_se2_{t}"][i], move)
        return l1, s1, l2, s2

    # Departures, stratified: lead third x period x h0 agreement, one row per game.
    dep = np.flatnonzero(held & (pick == pick2) & (pick != net))
    margin = np.array([np.nanmean(leads(int(i), int(pick[i]))[::2]) for i in dep])
    cuts = np.nanquantile(margin, [1 / 3, 2 / 3])
    cells: Dict[Tuple[int, str, bool], List[int]] = {}
    for i, m in zip(dep, margin):
        st = ts.state_from_raw_bytes(states[i].tobytes())
        cell = (int(np.searchsorted(cuts, m)), period(int(st.turn)), bool(z["pick_h0"][i] == pick[i]))
        cells.setdefault(cell, []).append(int(i))
    for v in cells.values():
        rng.shuffle(v)
    chosen: List[Tuple[int, int, int, str]] = []    # row, root's move, other move, kind
    games: set = set()
    keys = sorted(cells)
    while len(chosen) < a.departures and any(cells.values()):
        for c in keys:
            while cells[c] and int(z["game"][cells[c][-1]]) in games:
                cells[c].pop()
            if cells[c] and len(chosen) < a.departures:
                i = cells[c].pop()
                games.add(int(z["game"][i]))
                chosen.append((i, int(pick[i]), int(net[i]), "departure"))

    # Controls: the root keeps the network's move and rates another candidate clearly below it.
    ctrl: List[Tuple[int, int]] = []
    for i in rng.permutation(np.flatnonzero(held & (pick == net) & (pick2 == net))):
        if int(z["game"][i]) in games:
            continue
        for c in z[f"cand_{t}"][i][1:]:
            if c < 0:
                break
            l1, s1, l2, s2 = leads(int(i), int(c))
            if l1 < -3 * s1 and l2 < -3 * s2 and (l1 + l2) / 2 < -0.05:
                ctrl.append((int(i), int(c)))
                games.add(int(z["game"][i]))
                break
        if len(ctrl) >= a.controls:
            break
    chosen += [(i, int(net[i]), c, "control") for i, c in ctrl]

    rows: List[Dict[str, Any]] = []
    key: Dict[str, Dict[str, Any]] = {}
    for n in rng.permutation(len(chosen)):
        i, root_move, other, kind = chosen[n]
        st = ts.state_from_raw_bytes(states[i].tobytes())
        pos = position_token(st)
        rid = hashlib.sha1(f"{t}:{i}:{pos}".encode()).hexdigest()[:12]
        flip = bool(rng.integers(2))
        a_move, b_move = (other, root_move) if flip else (root_move, other)
        rows.append({"id": rid, "pos": pos, **describe(st),
                     "a": move_name(st, a_move), "b": move_name(st, b_move)})
        l1, s1, l2, s2 = leads(i, root_move if kind == "departure" else other)
        key[rid] = {"row": int(i), "game": int(z["game"][i]), "kind": kind, "root": "B" if flip else "A",
                    "root_move": move_name(st, root_move), "other_move": move_name(st, other),
                    "net_move": move_name(st, int(net[i])), "h0_pick": move_name(st, int(z["pick_h0"][i])),
                    "lead": [l1, l2], "lead_se": [s1, s2], "period": period(int(st.turn))}
    with open(a.rows, "w") as f:
        json.dump({"teacher": t, "targets": a.targets, "rows": rows}, f)
    with open(a.key, "w") as f:
        json.dump({"teacher": t, "targets": a.targets, "cuts": [float(c) for c in cuts], "key": key}, f, indent=1)
    n_dep = sum(k == "departure" for *_, k in chosen)
    print(f"{n_dep} departures from {len(dep):,} stable ({len(cells)} cells; lead thirds at "
          f"{cuts[0]:+.3f}, {cuts[1]:+.3f}), {len(ctrl)} controls -> {a.rows}, {a.key}")


if __name__ == "__main__":
    main()
