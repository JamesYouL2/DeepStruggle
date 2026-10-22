"""Turn 1 setups: every USSR setup against every US setup, by paired playouts, on one fixed deal
or on a fresh deal in every world.

How good a setup is depends on the hands, and an average over deals mixes hands that no game
puts together. By default this fixes one deal -- both hands as a seed deals them, with cards
swapped in by name -- forces each USSR setup and then each US setup on it, and plays every
pairing to the end with one network on both sides. Only what follows the setups varies: a world
fixes the dice and every later draw (`rng_state`, set after the setups), and every cell of a
world shares it, so all comparisons are paired.

With `run --random-deals` every world deals afresh instead (`init_game` from a seed of the world
alone), still shared by every cell of the world, and records both hands, so the report can split
the matrix by where each scoring card was dealt.

* `deal` -- the edited deal's hands and its workbench link, before the setups.
* `run` -- this part's share of the worlds (`--part k/N`, for CI runners); `--ussr` / `--us`
  choose the setups (default: all listed below). On the fixed deal, the network's own greedy
  setups (as USSR, and as US against each USSR setup) go in the metadata.
* `headlines` -- over random deals, each cell's greedy turn-1 headlines (no playouts) on the
  worlds where a card (`--card`, default East European Unrest) was dealt to one side (`--holder`),
  and the countries the card's event targeted; rows carry the world, so they join `run` rows of
  the same seed.
* `reply` -- on the same worlds, the holder's headline forced to that card and the other side's
  first turn(s) played twice from one start: by the network, and by a searched player
  (`--player`, a load_agent spec, {MODEL} for the checkpoint); the holder plays the network, and
  both sides the network to the end. Each arm's every decision of the replying side, the board
  at the end of the searched turns and the result, paired cell by cell.
* `ar1` -- on the same worlds and the same forced headline, the other side's first action round
  of turn 1 forced, once per arm (`--arms`; `ar1_restrict`): a 3+ Ops card, as a coup on Italy,
  or for influence back into the countries the headline broke; `net` leaves it to the network.
  Everything else is the network's, greedy, to the end.
* `draws` -- what each card is worth to draw, on every turn: games from random deals played by
  the network, each snapshotted just after every turn's deal, and each snapshot's newly drawn
  cards redealt `--redeals` ways from the same pool (the deck plus those cards; the cards each
  side kept stay put), every redeal played to the end. A row is one snapshot; `redeal` rebuilds
  its hands, so rows carry the pool rather than every redeal's cards.
* `pool` -- merge the parts.
* `report` -- the US score of every cell, each side's best reply to each setup, the score against
  the corpus's human setups, the equilibrium of the matrix game and paired contrasts; over random
  deals, the same split by where each scoring card was dealt.

A cell's score is what the two setups are worth to *this network's* play afterwards.

    PYTHONPATH=.:build/release python tools/scripts/setup_deal_matrix.py deal
    PYTHONPATH=.:build/release python tools/scripts/setup_deal_matrix.py run --model swa.onnx \\
        --worlds 16000 --part 1/40 --out matrix-1.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/setup_deal_matrix.py pool \\
        --parts matrix-*.jsonl.gz --expect 40 --out matrix.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/setup_deal_matrix.py report \\
        --input matrix.jsonl.gz --out report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import ts_engine as ts

from ai.eval.paired_playouts import apply_move, decider, play_safe
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import LogitsFn, load_policy, position_token
from tools.lib.jsonl import read_jsonl, sha256_file, write_jsonl
from tools.lib.player_agent import BatchSelector, load_agent

FloatArray = npt.NDArray[np.float64]
#: the legal mask of game k's state, narrowed: (k, state, legal mask) -> mask
Restrict = Callable[[int, ts.GameState, np.ndarray], np.ndarray]

NODE = ActionEncoder.NODE_OFFSET
N_COUNTRIES = 84
WORKBENCH = "https://mihaild.github.io/DeepStruggle"
#: no action round has this key, so play_safe's one-round safety guard never engages
NO_KEY = (-1, -1, -1, -1)

#: The deal: init_game(500310), the deal of 1,000 on which the networks' and the humans' setups
#: move least (data/reports/t1_openings), with the owner's swaps (2026-10-19): out of the USSR
#: hand Vietnam Revolts, De-Stalinization and Middle East Scoring for Olympic Games, Formosan
#: Resolution and Captured Nazi Scientist; out of the US hand Socialist Governments for
#: De-Stalinization.
DEAL = 500310
SWAPS = ("Vietnam Revolts=Olympic Games", "De-Stalinization=Formosan Resolution",
         "Middle East Scoring=Captured Nazi Scientist", "Socialist Governments=De-Stalinization")

#: Setup names give final influence: "EG4/Pol4/Aus1" is East Germany 4 (3 + 1), Poland 4,
#: Austria 1. A country not named keeps its starting influence. The order here is the order of
#: equal values in a name.
ABBR = (("EG", "East Germany"), ("Pol", "Poland"), ("Yug", "Yugoslavia"), ("Hun", "Hungary"),
        ("Aus", "Austria"), ("Cze", "Czechoslovakia"), ("Rom", "Romania"), ("Bul", "Bulgaria"),
        ("Fin", "Finland"), ("WG", "West Germany"), ("Fr", "France"), ("It", "Italy"),
        ("UK", "United Kingdom"), ("Ben", "Benelux"), ("Gr", "Greece"), ("Tur", "Turkey"),
        ("Ir", "Iran"), ("Pan", "Panama"), ("Isr", "Israel"))

#: (setup, who plays it). Human shares: 254 corpus games with a full turn-1 board; networks: their
#: greedy setup over 1,000 deals (A13+R48 SWA, R34 soup, R39 SWA; R32 SWA and the heads soup 300), and
#: their mean probability where it tops 5%; "playouts": a best reply or equilibrium setup of the
#: fixed-deal (500310) matrices under A13 or R39.
USSR_SETUPS: Tuple[Tuple[str, str], ...] = (
    ("EG4/Pol4/Aus1", "human 45%"),
    ("EG4/Pol4/Yug1", "human 30%"),
    ("EG4/Pol5", "human 24%"),
    ("Pol3/Yug3", "A13, R34, R39: every deal"),
    ("Pol3/Hun2/Aus1", "R32, heads soup"),
    ("Pol3/Yug2/Hun1", "network tail"),
    ("EG4/Pol3/Yug2", "playouts: A13 fixed-deal equilibrium"),
    ("Pol4/Yug2", "variant"),
    ("Pol3/Yug2/Aus1", "variant"),
    ("EG4/Pol4/Hun1", "variant"),
    ("Pol3/Hun3", "probe: Yugoslavia's place"),
    ("Pol3/Rom3", "probe: Yugoslavia's place"),
    ("Pol3/Cze3", "probe: Yugoslavia's place"),
    ("Pol3/Bul3", "playouts: R39 fixed-deal equilibrium"),
)
US_SETUPS: Tuple[Tuple[str, str], ...] = (
    ("WG4/It4/Ir2", "human 48%"),
    ("WG3/Fr3/It2/Ir2", "human 31%, R39 52%, R34 53%"),
    ("WG3/It3/Fr2/Ir2", "human 7%"),
    ("WG4/It3/Ir3", "human 5%"),
    ("WG4/It3/Fr1/Ir2", "human 3%"),
    ("WG4/Fr3/It2", "human 3%, R34 10%, R39 8%"),
    ("WG4/It3/Ir2/Pan2", "A13 90%"),
    ("WG4/Fr2/It2/Ir2", "R34 26%, R39 14%"),
    ("Fr6/It2/Ir2", "R32, heads soup"),
    ("WG4/It2/Fr1/Ir2/Pan2", "A13 5%"),
    ("WG4/It4/Pan2", "variant"),
    ("WG3/Fr3/It2/Pan2", "variant"),
    ("WG4/It3/Fr2", "playouts: fixed-deal best"),
    ("Fr4/WG3/It2", "variant"),
    ("WG3/Fr3/It3", "playouts: A13 fixed-deal equilibrium"),
    ("WG4/Fr3/It1/Ir2", "variant"),
    ("WG4/Fr3/Ir3", "probe: Italy left empty"),
    ("WG4/It5", "probe: Italy overprotected"),
)
#: The corpus's human setups (data/reports/t1_openings/human_setups.json): counts of each side's
#: setup, and of the US's reply to each of the three human USSR setups.
HUMAN_USSR: Dict[str, int] = {"EG4/Pol4/Aus1": 115, "EG4/Pol4/Yug1": 76, "EG4/Pol5": 60}
HUMAN_US: Dict[str, int] = {"WG4/It4/Ir2": 122, "WG3/Fr3/It2/Ir2": 78, "WG3/It3/Fr2/Ir2": 18,
                            "WG4/It3/Ir3": 12, "WG4/It3/Fr1/Ir2": 8, "WG4/Fr3/It2": 8}
_REPLY_ORDER = ("WG4/It4/Ir2", "WG3/Fr3/It2/Ir2", "WG3/It3/Fr2/Ir2", "WG4/It3/Ir3",
                "WG4/It3/Fr1/Ir2", "WG4/Fr3/It2")
HUMAN_REPLY: Dict[str, Dict[str, int]] = {
    u: dict(zip(_REPLY_ORDER, counts)) for u, counts in (
        ("EG4/Pol4/Aus1", (52, 40, 7, 3, 4, 4)),
        ("EG4/Pol4/Yug1", (52, 12, 6, 2, 1, 2)),
        ("EG4/Pol5", (18, 25, 5, 7, 3, 0)))}
#: the reference row of the contrasts: the networks' USSR setup
REFERENCE_USSR = "Pol3/Yug3"
#: over random deals, the report splits the matrix by where each of these was dealt
SPLIT_CARDS = ("Middle East Scoring", "Asia Scoring", "Europe Scoring")
#: the `deal` of a run whose every world deals afresh
RANDOM = "random"


def card_ids() -> Dict[str, int]:
    return {ts.CardData.get_card_name(c): c for c in range(1, 111)}


def country_ids() -> Dict[str, int]:
    return {ts.MapData.get_country_name(c): c for c in range(N_COUNTRIES)}


# --------------------------------------------------------------------------- the deal and setups


def build_deal(deal: int, swaps: Sequence[str]) -> ts.GameState:
    """init_game(deal), then each "out=in" swap: the out card, in either hand, goes to the draw
    deck and the in card, from the draw deck, takes its place."""
    ids = card_ids()
    st = ts.GameState()
    ts.Engine.init_game(st, deal)
    for pair in swaps:
        out_name, in_name = (x.strip() for x in pair.split("="))
        if out_name not in ids or in_name not in ids:
            raise ValueError(f"unknown card in {pair!r}")
        out, inn = ids[out_name], ids[in_name]
        loc = st.get_card_location(out)
        if ts.hand_holder(loc) == ts.Player.NONE:
            raise ValueError(f"{out_name} is in no hand ({loc})")
        if st.get_card_location(inn) != ts.CardLocation.DRAW_DECK:
            raise ValueError(f"{in_name} is not in the draw deck ({st.get_card_location(inn)})")
        st.set_card_location(out, ts.CardLocation.DRAW_DECK)
        st.set_card_location(inn, loc)
    return st


def hand(st: ts.GameState, side: ts.Player) -> List[int]:
    return [c for c in range(1, 111) if ts.hand_holder(st.get_card_location(c)) == side]


def influence(st: ts.GameState, side: ts.Player) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for c in range(N_COUNTRIES):
        cs = st.get_country(c)
        out[c] = cs.us_influence if side == ts.Player.US else cs.ussr_influence
    return out


def parse_setup(name: str, base: ts.GameState, side: ts.Player) -> Dict[int, int]:
    """A setup's name -> {country: points added}, checked to add 6 (USSR) or 9 (US) points."""
    by_abbr = {a: ts.MapData.get_country_by_name(full) for a, full in ABBR}
    start = influence(base, side)
    added: Dict[int, int] = {}
    tokens = name.split("/")
    for tok in tokens:
        m = re.fullmatch(r"([A-Za-z]+)(\d+)", tok)
        if not m or m.group(1) not in by_abbr:
            raise ValueError(f"cannot read {tok!r} in {name!r}")
        c = by_abbr[m.group(1)]
        k = int(m.group(2)) - start[c]
        if k <= 0:
            raise ValueError(f"{name!r} adds nothing to {tok} (it starts at {start[c]})")
        added[c] = k
    want = 6 if side == ts.Player.USSR else 9
    if sum(added.values()) != want:
        raise ValueError(f"{name!r} adds {sum(added.values())} points, not {want}")
    return added


def setup_name(added: Mapping[int, int], base: ts.GameState, side: ts.Player) -> str:
    """The name of a setup given as {country: points added}: its name in the setup lists, else
    one with the countries by final influence."""
    pts = {c: k for c, k in added.items() if k}
    for nm, _ in USSR_SETUPS if side == ts.Player.USSR else US_SETUPS:
        if parse_setup(nm, base, side) == pts:
            return nm
    start = influence(base, side)
    order = {ts.MapData.get_country_by_name(full): i for i, (_, full) in enumerate(ABBR)}
    abbr = {ts.MapData.get_country_by_name(full): a for a, full in ABBR}
    cs = sorted((c for c, k in added.items() if k), key=lambda c: (-(start[c] + added[c]), order.get(c, 99), c))
    return "/".join(f"{abbr.get(c, ts.MapData.get_country_name(c))}{start[c] + added[c]}" for c in cs)


def place(st: ts.GameState, added: Mapping[int, int]) -> ts.GameState:
    """A copy of `st` with one side's setup placed, a point at a time round robin over the
    countries; a point the engine refuses now (a US bonus point before the seven in Western
    Europe, or in a country still without US influence) waits for a later round."""
    t = st.clone()
    left = {c: k for c, k in added.items() if k}
    while left:
        moved = False
        for c in list(left):
            if ts.Engine.try_step_flat(t, NODE + c, auto_advance=True):
                moved = True
                left[c] -= 1
                if not left[c]:
                    del left[c]
        if not moved:
            raise ValueError(f"the engine refuses every remaining point: {left}")
    return t


def cell_start(base: ts.GameState, ussr: Mapping[int, int], us: Mapping[int, int]) -> ts.GameState:
    """The deal with both setups placed: the first decision of the turn-1 headline."""
    st = place(place(base, ussr), us)
    if st.current_phase == ts.Phase.SETUP:
        raise ValueError("the setups did not finish the setup phase")
    for side, added in ((ts.Player.USSR, ussr), (ts.Player.US, us)):
        before, after = influence(base, side), influence(st, side)
        if any(after[c] - before[c] != added.get(c, 0) for c in range(N_COUNTRIES)):
            raise ValueError(f"the {side} setup landed elsewhere than asked")
    return st


def greedy_setup(st: ts.GameState, logits_fn: LogitsFn, feats: int) -> Tuple[ts.GameState, Dict[int, int]]:
    """The network's greedy setup for the side placing now: the state after it, and the points."""
    side = st.ctx().decision_player
    before = influence(st, side)
    t = st.clone()
    while t.current_phase == ts.Phase.SETUP and t.ctx().decision_player == side:
        obs = ts.extract_observation_features(t, side, feats)[None]
        mask = np.asarray(ActionEncoder.get_legal_mask(t))[None]
        t = apply_move(t, int(np.asarray(logits_fn(obs, mask)).argmax(axis=1)[0]))
    after = influence(t, side)
    return t, {c: after[c] - before[c] for c in range(N_COUNTRIES) if after[c] != before[c]}


def world_rng(deal: int, seed: int, world: int) -> int:
    """The dice and draws of one world: an rng_state from (deal, seed, world) alone, so a world
    is the same whatever part or batch plays it."""
    h = hashlib.sha256(f"setup-deal:{deal}:{seed}:{world}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def world_deal(seed: int, world: int) -> int:
    """A world's deal when every world deals afresh: an init_game seed from (seed, world) alone."""
    h = hashlib.sha256(f"setup-random-deal:{seed}:{world}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def dealt(deal_seed: int) -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, deal_seed)
    return st


def deal_summary(st: ts.GameState) -> Dict[str, Any]:
    info = {c: ts.CardData.get_card_info(c) for c in range(1, 111)}
    out: Dict[str, Any] = {"pos": position_token(st)}
    for side, key in ((ts.Player.USSR, "ussr"), (ts.Player.US, "us")):
        cards = sorted(hand(st, side), key=lambda c: (-int(info[c]["ops"]), str(info[c]["name"])))
        out[key] = [str(info[c]["name"]) for c in cards]
        out[key + "_ops"] = sum(int(info[c]["ops"]) for c in cards)
    return out


def deal(deal_seed: int, swaps: Sequence[str]) -> int:
    d = deal_summary(build_deal(deal_seed, swaps))
    print(f"deal {deal_seed}, swaps: {'; '.join(swaps)}")
    print(f"USSR ({d['ussr_ops']} ops): {', '.join(d['ussr'])}")
    print(f"US ({d['us_ops']} ops): {', '.join(d['us'])}")
    print(f"{WORKBENCH}/?pos={d['pos']}")
    return 0


# --------------------------------------------------------------------------- run


def net_setups(base: ts.GameState, ussr: Sequence[Tuple[str, Dict[int, int]]], logits_fn: LogitsFn,
               feats: int) -> Dict[str, Any]:
    """The network's greedy USSR setup on the deal, and its greedy US reply to each USSR setup
    (its own included)."""
    after, own = greedy_setup(base, logits_fn, feats)
    own_name = setup_name(own, base, ts.Player.USSR)
    reply: Dict[str, str] = {}
    for name, added in list(ussr) + [(own_name, own)]:
        _, us = greedy_setup(place(base, added), logits_fn, feats)
        reply[name] = setup_name(us, base, ts.Player.US)
    return {"ussr": own_name, "us": reply}


def run(model: str, worlds: int, part: str, seed: int, deal_seed: int, swaps: Sequence[str],
        chunk: int, out: str, ussr_names: Optional[Sequence[str]] = None,
        us_names: Optional[Sequence[str]] = None, random_deals: bool = False) -> int:
    """Play this part's worlds. With `random_deals` each world deals afresh (`world_deal`) and its
    row carries both hands; `deal_seed` and `swaps` are then unused."""
    k, n = (int(x) for x in part.split("/"))
    base = dealt(world_deal(seed, 0)) if random_deals else build_deal(deal_seed, swaps)
    ussr_names = list(ussr_names or [nm for nm, _ in USSR_SETUPS])
    us_names = list(us_names or [nm for nm, _ in US_SETUPS])
    # the setups' starting influence is the rules', the same on every deal
    ussr = [(nm, parse_setup(nm, base, ts.Player.USSR)) for nm in ussr_names]
    us = [(nm, parse_setup(nm, base, ts.Player.US)) for nm in us_names]

    def cells_of(b: ts.GameState) -> List[ts.GameState]:
        return [cell_start(b, a, c) for _, a in ussr for _, c in us]

    fixed = None if random_deals else cells_of(base)
    n_cells = len(ussr) * len(us)
    logits_fn, feats = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    t0 = time.time()
    net = None if random_deals else net_setups(base, ussr, logits_fn, feats)
    mine = [w for w in range(worlds) if w % n == k - 1]
    step = max(1, chunk // n_cells)
    res: List[Dict[str, Any]] = []
    for lo in range(0, len(mine), step):
        batch = mine[lo:lo + step]
        starts: List[ts.GameState] = []
        rows: List[Dict[str, Any]] = []
        for w in batch:
            row: Dict[str, Any] = {"world": w}
            if fixed is None:
                d = world_deal(seed, w)
                b0 = dealt(d)
                cells, r = cells_of(b0), world_rng(d, seed, w)
                row.update({"deal": d, "ussr_hand": hand(b0, ts.Player.USSR), "us_hand": hand(b0, ts.Player.US)})
            else:
                cells, r = fixed, world_rng(deal_seed, seed, w)
            for c in cells:
                s = c.clone()
                s.rng_state = r
                starts.append(s)
            rows.append(row)
        scores = play_safe(starts, [ts.Player.US] * len(starts), [NO_KEY] * len(starts), act, seed)
        for b, row in enumerate(rows):
            got = scores[b * n_cells:(b + 1) * n_cells]
            row["s"] = "".join(str(round(2 * x)) for x in got)
            res.append(row)
        print(f"[{lo + len(batch)}/{len(mine)} worlds] {len(starts)} games in this batch, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    write_jsonl(out, res)
    with open(out + ".meta.json", "w") as f:
        json.dump({"schema": 1, "stage": "run", "model": os.path.basename(model),
                   "model_sha256": sha256_file(model), "deal": RANDOM if random_deals else deal_seed,
                   "swaps": [] if random_deals else list(swaps),
                   "hands": None if random_deals else deal_summary(base), "ussr": ussr_names,
                   "us": us_names, "net": net, "worlds": worlds, "part": part,
                   "seed": seed, "rows": len(res), "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"{len(res)} worlds, {len(res) * len(cells)} games in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def greedy_headlines(states: List[ts.GameState], logits_fn: LogitsFn, feats: int,
                     card: int) -> Tuple[List[int], List[int], List[List[int]]]:
    """Play each state's turn-1 headline phase greedily: each state's US and USSR headline, read
    from the choices themselves (the engine clears the pair once both resolve, which a USSR card
    chosen second and resolving without a decision does inside one step), and the countries the
    `card` event targeted when either side headlined it."""
    us, ussr = [0] * len(states), [0] * len(states)
    targets: List[List[int]] = [[] for _ in states]
    live = list(range(len(states)))
    for _ in range(200):
        live = [k for k in live if states[k].current_phase == ts.Phase.HEADLINE]
        if not live:
            return us, ussr, targets
        obs = np.stack([ts.extract_observation_features(states[k], states[k].ctx().decision_player, feats)
                        for k in live])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(states[k])) for k in live])
        for k, a in zip(live, np.asarray(logits_fn(obs, masks)).argmax(axis=1)):
            st, a = states[k], int(a)
            c = st.ctx()
            if (c.decision_type == ts.DecisionType.SELECT_CARD and st.headline_stage == 0
                    and c.resolving_card == 0):
                (us if c.decision_player == ts.Player.US else ussr)[k] = ActionEncoder.decode(st, a).primary_id
            elif c.resolving_card == card and NODE <= a < NODE + N_COUNTRIES:
                targets[k].append(a - NODE)
            states[k] = apply_move(st, a)
    raise RuntimeError(f"{len(live)} headline phases still unfinished after 200 decisions")


def headlines(model: str, worlds: int, part: str, seed: int, chunk: int, out: str,
              ussr_names: Sequence[str], us_names: Sequence[str], card: str, holder: str) -> int:
    """Over random deals, every cell's greedy turn-1 headlines on the worlds where `card` was dealt
    to `holder`, and the countries the card's event targeted when it was headlined. Rows carry
    the world, so they join the `run --random-deals` rows of the same seed."""
    k, n = (int(x) for x in part.split("/"))
    ids = card_ids()
    if card not in ids:
        raise ValueError(f"unknown card {card!r}")
    cid, side = ids[card], ts.Player.US if holder == "us" else ts.Player.USSR
    base = dealt(world_deal(seed, 0))
    ussr = [parse_setup(nm, base, ts.Player.USSR) for nm in ussr_names]
    us = [parse_setup(nm, base, ts.Player.US) for nm in us_names]
    n_cells = len(ussr) * len(us)
    logits_fn, feats = load_policy(model)
    t0 = time.time()
    mine = [w for w in range(worlds) if w % n == k - 1]
    step = max(1, chunk // n_cells)
    res: List[Dict[str, Any]] = []
    for lo in range(0, len(mine), step):
        rows, starts = card_worlds(mine[lo:lo + step], seed, cid, side, ussr, us)
        if not rows:
            continue
        us_hl, ussr_hl, tg = greedy_headlines(starts, logits_fn, feats, cid)
        for b, row in enumerate(rows):
            cells = slice(b * n_cells, (b + 1) * n_cells)
            row.update({"us_hl": us_hl[cells], "ussr_hl": ussr_hl[cells], "targets": tg[cells]})
            res.append(row)
        print(f"[{min(lo + step, len(mine))}/{len(mine)} worlds] {len(res)} with {card} in the {holder.upper()} hand, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    write_jsonl(out, res)
    with open(out + ".meta.json", "w") as f:
        json.dump({"schema": 1, "stage": "headlines", "model": os.path.basename(model),
                   "model_sha256": sha256_file(model), "deal": RANDOM, "swaps": [], "hands": None,
                   "ussr": list(ussr_names), "us": list(us_names), "net": None, "card": card,
                   "holder": holder, "worlds": worlds, "part": part, "seed": seed, "rows": len(res),
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"{len(res)} worlds, {len(res) * n_cells} headline phases in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def forced_card(st: ts.GameState, card: int) -> int:
    """The legal action of a SELECT_CARD decision that picks `card`."""
    for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st))):
        if ActionEncoder.decode(st, int(a)).primary_id == card:
            return int(a)
    raise ValueError(f"card {card} is not a legal choice here")


def play_turns(states: List[ts.GameState], logits_fn: LogitsFn, feats: int, turns: int, holder: ts.Player,
               card: int, player: Optional[BatchSelector], restrict: Optional[Restrict] = None
               ) -> Tuple[List[int], List[List[List[int]]], List[List[int]]]:
    """Play each state through turn `turns`: the holder's turn-1 headline forced to `card`, the
    other side by `player` (the network greedily when None), everything else the network greedily,
    within `restrict(k, state, legal mask)` when given. Per state the other side's turn-1
    headline, its every decision as [turn, action round, decision type, resolving card, decoded
    id], and the countries the card's event targeted."""
    other = ts.Player.USSR if holder == ts.Player.US else ts.Player.US
    hl = [0] * len(states)
    moves: List[List[List[int]]] = [[] for _ in states]
    targets: List[List[int]] = [[] for _ in states]
    live = list(range(len(states)))
    for _ in range(5000):
        live = [k for k in live if states[k].turn <= turns and not ts.Engine.is_terminal(states[k])]
        if not live:
            return hl, moves, targets
        acts: Dict[int, int] = {}
        searched, greedy = [], []
        for k in live:
            st = states[k]
            c, p = st.ctx(), decider(st)
            first = (st.turn == 1 and st.current_phase == ts.Phase.HEADLINE and st.headline_stage == 0
                     and c.decision_type == ts.DecisionType.SELECT_CARD and c.resolving_card == 0)
            if first and p == holder:
                acts[k] = forced_card(st, card)
            elif player is not None and p == other:
                searched.append(k)
            else:
                greedy.append(k)
        if searched and player is not None:
            acts.update(zip(searched, player.select_actions_batch([states[k] for k in searched])))
        if greedy:
            obs = np.stack([ts.extract_observation_features(states[k], decider(states[k]), feats) for k in greedy])
            masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(states[k])) for k in greedy])
            if restrict is not None:
                masks = np.stack([restrict(k, states[k], m) for k, m in zip(greedy, masks)])
            acts.update(zip(greedy, (int(a) for a in np.asarray(logits_fn(obs, masks)).argmax(axis=1))))
        for k in live:
            st, a = states[k], int(acts[k])
            c = st.ctx()
            if decider(st) == other:
                pid = int(ActionEncoder.decode(st, a).primary_id)
                if (st.turn == 1 and st.current_phase == ts.Phase.HEADLINE and st.headline_stage == 0
                        and c.decision_type == ts.DecisionType.SELECT_CARD and c.resolving_card == 0):
                    hl[k] = pid
                moves[k].append([int(st.turn), int(st.action_round), int(c.decision_type), int(c.resolving_card), pid])
            if c.resolving_card == card and NODE <= a < NODE + N_COUNTRIES:
                targets[k].append(a - NODE)
            states[k] = apply_move(st, a)
    raise RuntimeError(f"{len(live)} games still inside turn {turns} after 5000 decisions")


def board(st: ts.GameState) -> Dict[str, Any]:
    """VP, DEFCON and [country, US, USSR] influence wherever either side has some."""
    us, ussr = influence(st, ts.Player.US), influence(st, ts.Player.USSR)
    return {"vp": int(st.victory_points), "defcon": int(st.defcon),
            "inf": [[c, us[c], ussr[c]] for c in range(N_COUNTRIES) if us[c] or ussr[c]]}


def card_worlds(worlds: Sequence[int], seed: int, card: int, side: ts.Player, ussr: Sequence[Mapping[int, int]],
                us: Sequence[Mapping[int, int]]) -> Tuple[List[Dict[str, Any]], List[ts.GameState]]:
    """Of `worlds`, those where `card` was dealt to `side`: a row each (world, deal, both hands)
    and every cell's start, USSR setups outer, with the world's dice."""
    rows: List[Dict[str, Any]] = []
    starts: List[ts.GameState] = []
    for w in worlds:
        d = world_deal(seed, w)
        b0 = dealt(d)
        if card not in hand(b0, side):
            continue
        r = world_rng(d, seed, w)
        for a in ussr:
            for c in us:
                s = cell_start(b0, a, c)
                s.rng_state = r
                starts.append(s)
        rows.append({"world": w, "deal": d, "ussr_hand": hand(b0, ts.Player.USSR), "us_hand": hand(b0, ts.Player.US)})
    return rows, starts


def reply(model: str, player_spec: str, worlds: int, part: str, seed: int, chunk: int, out: str,
          ussr_names: Sequence[str], us_names: Sequence[str], card: str, holder: str, turns: int) -> int:
    """On the worlds where `card` was dealt to `holder`, every cell twice from one start (the
    world's dice): the holder's headline forced to `card`, the other side through turn `turns` by
    the network (`net`) and by `player_spec` (`searched`), then the network on both sides to the
    end. Rows carry the world, so they join the `headlines` and `run --random-deals` rows."""
    k, n = (int(x) for x in part.split("/"))
    ids = card_ids()
    if card not in ids:
        raise ValueError(f"unknown card {card!r}")
    cid, side = ids[card], ts.Player.US if holder == "us" else ts.Player.USSR
    base = dealt(world_deal(seed, 0))
    ussr = [parse_setup(nm, base, ts.Player.USSR) for nm in ussr_names]
    us = [parse_setup(nm, base, ts.Player.US) for nm in us_names]
    n_cells = len(ussr) * len(us)
    logits_fn, feats = load_policy(model)
    agent = load_agent(player_spec.replace("{MODEL}", model), device="cpu")
    if not isinstance(agent, BatchSelector):
        raise ValueError(f"{player_spec!r} does not decide a batch of positions")
    reseed = getattr(agent, "reseed", None)
    if reseed is not None:
        reseed(seed * 1000 + k)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    t0 = time.time()
    mine = [w for w in range(worlds) if w % n == k - 1]
    step = max(1, chunk // n_cells)
    res: List[Dict[str, Any]] = []
    for lo in range(0, len(mine), step):
        rows, starts = card_worlds(mine[lo:lo + step], seed, cid, side, ussr, us)
        if not rows:
            continue
        for arm, who in (("net", None), ("searched", agent)):
            states = [s.clone() for s in starts]
            hl, moves, tg = play_turns(states, logits_fn, feats, turns, side, cid, who)
            ends = [board(s) for s in states]
            scores = play_safe(states, [ts.Player.US] * len(states), [NO_KEY] * len(states), act, seed)
            for b, row in enumerate(rows):
                cells = slice(b * n_cells, (b + 1) * n_cells)
                row[arm] = {"s": "".join(str(round(2 * x)) for x in scores[cells]), "hl": hl[cells],
                            "moves": moves[cells], "targets": tg[cells], "end": ends[cells]}
        res.extend(rows)
        print(f"[{min(lo + step, len(mine))}/{len(mine)} worlds] {len(res)} with {card} in the {holder.upper()} hand, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    write_jsonl(out, res)
    with open(out + ".meta.json", "w") as f:
        json.dump({"schema": 1, "stage": "reply", "model": os.path.basename(model),
                   "model_sha256": sha256_file(model), "player": player_spec, "player_name": agent.name,
                   "deal": RANDOM, "swaps": [], "hands": None, "ussr": list(ussr_names), "us": list(us_names),
                   "net": None, "card": card, "holder": holder, "turns": turns, "worlds": worlds, "part": part,
                   "seed": seed, "rows": len(res), "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"{len(res)} worlds, {2 * len(res) * n_cells} games in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


#: the replying side's first action round of turn 1, forced (stage `ar1`; `ar1_restrict`)
AR1_ARMS = ("net", "card3", "coup_italy", "repair", "repair_bg")
#: the Ops resolutions' slots (OP_MODE_OFFSET + OpMode lands on the same ones)
INFLUENCE, COUP = ActionEncoder.PLAY_MODE_OFFSET + 2, ActionEncoder.PLAY_MODE_OFFSET + 3
CHINA = 6


def ar1_restrict(arm: str, side: ts.Player, ctrl0: Sequence[Sequence[int]]) -> Optional[Restrict]:
    """`side`'s legal moves in its first action round of turn 1 under `arm`, game k having
    started with `side` controlling `ctrl0[k]`; outside that round, and wherever the arm leaves no
    legal move, the legal mask itself. None for `net`.

    * card3 -- a card of 3+ Ops (not the China Card; scoring cards have none), the network's pick,
      played as the network likes.
    * coup_italy -- that card, as a coup on Italy.
    * repair / repair_bg -- that card for influence, every point into a country `side` controlled
      at the start and no longer does (repair_bg: battlegrounds only) while one is left, the rest
      the network's."""
    if arm not in AR1_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "net":
        return None
    big = np.zeros(ActionEncoder.FLAT_ACTION_SIZE, dtype=np.uint8)
    for c in range(1, 111):
        if c != CHINA and ts.CardData.get_card_info(c)["ops"] >= 3:
            big[c - 1] = 1
    italy = country_ids()["Italy"]
    bg = {c for c in range(N_COUNTRIES) if ts.MapData.get_country_info(c)["battleground"]}
    choose_mode = (ts.DecisionType.SELECT_PLAY_MODE, ts.DecisionType.SELECT_OP_MODE)

    def restrict(k: int, st: ts.GameState, legal: np.ndarray) -> np.ndarray:
        if (decider(st) != side or st.turn != 1 or st.current_phase != ts.Phase.ACTION_ROUND
                or st.action_round != 1):
            return legal
        c = st.ctx()
        want = np.zeros_like(big)
        if c.decision_type == ts.DecisionType.SELECT_CARD and c.resolving_card == 0:
            want = big
        elif arm == "card3":
            return legal
        elif c.decision_type in choose_mode:
            want[COUP if arm == "coup_italy" else INFLUENCE] = 1
        elif c.decision_type == ts.DecisionType.POINT_NODE and c.resolving_card == 0:
            if arm == "coup_italy":
                want[NODE + italy] = 1
            else:
                for x in ctrl0[k]:
                    if (arm == "repair" or x in bg) and not ts.Scoring.is_controlled_by(st, x, side):
                        want[NODE + x] = 1
        out = legal * want
        return out if out.any() else legal
    return restrict


def ar1(model: str, worlds: int, part: str, seed: int, chunk: int, out: str, ussr_names: Sequence[str],
        us_names: Sequence[str], card: str, holder: str, arms: Sequence[str]) -> int:
    """On the worlds where `card` was dealt to `holder`, every cell once per arm from one start
    (the world's dice): the holder's headline forced to `card`, the other side's first action
    round of turn 1 forced by the arm (`ar1_restrict`), the network greedy everywhere else and
    to the end. `net` is the `reply` stage's `net` arm, game for game."""
    k, n = (int(x) for x in part.split("/"))
    ids = card_ids()
    if card not in ids:
        raise ValueError(f"unknown card {card!r}")
    cid, side = ids[card], ts.Player.US if holder == "us" else ts.Player.USSR
    other = ts.Player.USSR if side == ts.Player.US else ts.Player.US
    base = dealt(world_deal(seed, 0))
    ussr = [parse_setup(nm, base, ts.Player.USSR) for nm in ussr_names]
    us = [parse_setup(nm, base, ts.Player.US) for nm in us_names]
    n_cells = len(ussr) * len(us)
    logits_fn, feats = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    t0 = time.time()
    mine = [w for w in range(worlds) if w % n == k - 1]
    step = max(1, chunk // n_cells)
    res: List[Dict[str, Any]] = []
    for lo in range(0, len(mine), step):
        rows, starts = card_worlds(mine[lo:lo + step], seed, cid, side, ussr, us)
        if not rows:
            continue
        ctrl0 = [[x for x in range(N_COUNTRIES) if ts.Scoring.is_controlled_by(s, x, other)] for s in starts]
        for arm in arms:
            states = [s.clone() for s in starts]
            hl, moves, tg = play_turns(states, logits_fn, feats, 1, side, cid, None, ar1_restrict(arm, other, ctrl0))
            ends = [board(s) for s in states]
            scores = play_safe(states, [ts.Player.US] * len(states), [NO_KEY] * len(states), act, seed)
            for b, row in enumerate(rows):
                cells = slice(b * n_cells, (b + 1) * n_cells)
                row[arm] = {"s": "".join(str(round(2 * x)) for x in scores[cells]), "hl": hl[cells],
                            "moves": moves[cells], "targets": tg[cells], "end": ends[cells]}
        res.extend(rows)
        print(f"[{min(lo + step, len(mine))}/{len(mine)} worlds] {len(res)} with {card} in the {holder.upper()} hand, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    write_jsonl(out, res)
    with open(out + ".meta.json", "w") as f:
        json.dump({"schema": 1, "stage": "ar1", "model": os.path.basename(model),
                   "model_sha256": sha256_file(model), "arms": list(arms),
                   "deal": RANDOM, "swaps": [], "hands": None, "ussr": list(ussr_names), "us": list(us_names),
                   "net": None, "card": card, "holder": holder, "turns": 1, "worlds": worlds, "part": part,
                   "seed": seed, "rows": len(res), "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"{len(res)} worlds, {len(arms) * len(res) * n_cells} games in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def redeal(pool: Sequence[int], n_ussr: int, n_us: int, seed: int, game: int, turn: int, k: int
           ) -> Tuple[List[int], List[int]]:
    """Redeal k of a `draws` snapshot: the cards each side draws, from (seed, game, turn, k) alone."""
    h = hashlib.sha256(f"draws:{seed}:{game}:{turn}:{k}".encode()).digest()
    got = random.Random(int.from_bytes(h[:8], "little")).sample(sorted(pool), n_ussr + n_us)
    return got[:n_ussr], got[n_ussr:]


def redeal_start(row: Mapping[str, Any], st: ts.GameState, seed: int, k: int) -> ts.GameState:
    """Snapshot `st` with redeal k's cards in the hands and the rest of the pool in the deck; the
    dice are the snapshot's, the same for every redeal."""
    u, a = redeal(row["pool"], row["n_ussr"], row["n_us"], seed, row["game"], row["turn"], k)
    s = st.clone()
    for c in row["pool"]:
        s.set_card_location(c, ts.CardLocation.DRAW_DECK)
    for cards, loc in ((u, row["loc_ussr"]), (a, row["loc_us"])):
        for c in cards:
            s.set_card_location(c, ts.CardLocation(loc))
    h = hashlib.sha256(f"draws-dice:{seed}:{row['game']}:{row['turn']}".encode()).digest()
    s.rng_state = int.from_bytes(h[:8], "little")
    return s


def draw_snapshots(games: Sequence[int], seed: int, logits_fn: LogitsFn, feats: int
                   ) -> List[Tuple[Dict[str, Any], ts.GameState]]:
    """Each game from its random deal, the network greedy on both sides, snapshotted just after
    every deal (turn 1: before the setups): a row (game, turn, board, the cards each side kept,
    the pool, how many each side drew and where they lie) and the state."""
    out: List[Tuple[Dict[str, Any], ts.GameState]] = []

    def snap(g: int, before: Optional[ts.GameState], st: ts.GameState) -> None:
        row: Dict[str, Any] = {"game": g, "turn": int(st.turn), "vp": int(st.victory_points), "defcon": int(st.defcon)}
        pool = [c for c in range(1, 111) if st.get_card_location(c) == ts.CardLocation.DRAW_DECK]
        for side, key in ((ts.Player.USSR, "ussr"), (ts.Player.US, "us")):
            now = hand(st, side)
            kept = set(hand(before, side)) if before is not None else set()
            drawn = [c for c in now if c not in kept]
            pool += drawn
            row.update({f"kept_{key}": [c for c in now if c in kept], f"n_{key}": len(drawn),
                        f"loc_{key}": int(st.get_card_location(drawn[0])) if drawn else 0})
        row["pool"] = sorted(pool)
        out.append((row, st.clone()))

    states = []
    for g in games:
        st = dealt(world_deal(seed, g))
        snap(g, None, st)
        states.append(st)
    live = list(range(len(states)))
    for _ in range(10000):
        live = [k for k in live if not ts.Engine.is_terminal(states[k])]
        if not live:
            return out
        obs = np.stack([ts.extract_observation_features(states[k], decider(states[k]), feats) for k in live])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(states[k])) for k in live])
        acts = np.asarray(logits_fn(obs, masks)).argmax(axis=1)
        for k, a in zip(live, acts):
            before = states[k]
            states[k] = apply_move(before, int(a))
            st = states[k]
            if st.turn > before.turn and not ts.Engine.is_terminal(st):
                snap(games[k], before, st)
    raise RuntimeError(f"{len(live)} games unfinished after 10000 decisions")


def draws(model: str, worlds: int, part: str, seed: int, redeals: int, chunk: int, out: str) -> int:
    """This part's games (`worlds` of them in all), every snapshot redealt `redeals` ways and
    played to the end; per snapshot the US score of each redeal (`s`, 0/1/2 for loss/draw/win)."""
    k, n = (int(x) for x in part.split("/"))
    logits_fn, feats = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    t0 = time.time()
    mine = [w for w in range(worlds) if w % n == k - 1]
    step = max(1, chunk // redeals)
    res: List[Dict[str, Any]] = []
    for lo in range(0, len(mine), step):
        snaps = draw_snapshots(mine[lo:lo + step], seed, logits_fn, feats)
        starts: List[ts.GameState] = []
        for row, st in snaps:
            starts += [redeal_start(row, st, seed, j) for j in range(redeals)]
        scores: List[float] = []
        for b in range(0, len(starts), chunk):
            part_starts = starts[b:b + chunk]
            scores += play_safe(part_starts, [ts.Player.US] * len(part_starts), [NO_KEY] * len(part_starts), act, seed)
        for i, (row, _) in enumerate(snaps):
            row["s"] = "".join(str(round(2 * x)) for x in scores[i * redeals:(i + 1) * redeals])
            res.append(row)
        print(f"[{min(lo + step, len(mine))}/{len(mine)} games] {len(res)} snapshots, {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
    write_jsonl(out, res)
    with open(out + ".meta.json", "w") as f:
        json.dump({"schema": 1, "stage": "draws", "model": os.path.basename(model),
                   "model_sha256": sha256_file(model), "redeals": redeals, "deal": RANDOM, "swaps": [],
                   "hands": None, "ussr": [], "us": [], "net": None, "worlds": worlds, "part": part,
                   "seed": seed, "rows": len(res), "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    print(f"{len(res)} snapshots, {len(res) * redeals} games in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def ever_games(games: Sequence[int], seed: int, logits_fn: LogitsFn, feats: int) -> List[Dict[str, Any]]:
    """Each game from its random deal (the `draws` stage's base games), the network greedy on both
    sides to the end: every card each side ever held, with the turn it first did, and the result."""
    states = [dealt(world_deal(seed, g)) for g in games]
    held: List[Dict[str, Dict[int, int]]] = [{"ussr": {}, "us": {}} for _ in games]

    def look(k: int) -> None:
        st = states[k]
        for c in range(1, 111):
            side = ts.hand_holder(st.get_card_location(c))
            key = "ussr" if side == ts.Player.USSR else "us" if side == ts.Player.US else None
            if key is not None and c not in held[k][key]:
                held[k][key][c] = int(st.turn)

    for k in range(len(states)):
        look(k)
    live = list(range(len(states)))
    for _ in range(10000):
        live = [k for k in live if not ts.Engine.is_terminal(states[k])]
        if not live:
            break
        obs = np.stack([ts.extract_observation_features(states[k], decider(states[k]), feats) for k in live])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(states[k])) for k in live])
        acts = np.asarray(logits_fn(obs, masks)).argmax(axis=1)
        for k, a in zip(live, acts):
            states[k] = apply_move(states[k], int(a))
            look(k)
    else:
        raise RuntimeError(f"{len(live)} games unfinished after 10000 decisions")
    out: List[Dict[str, Any]] = []
    for g, st, h in zip(games, states, held):
        u = float(ts.Engine.get_terminal_utility(st))
        out.append({"game": g, "turn": int(st.turn), "vp": int(st.victory_points), "defcon": int(st.defcon),
                    "s": 2 if u > 0 else (1 if u == 0 else 0),
                    "ussr": sorted([c, t] for c, t in h["ussr"].items()),
                    "us": sorted([c, t] for c, t in h["us"].items())})
    return out


def ever(model: str, worlds: int, part: str, seed: int, chunk: int, out: str) -> int:
    """This part's games (`worlds` of them in all), greedy from random deals, one row per game:
    the cards each side ever held ([card, first turn]) and the US score (`s`, 0/1/2)."""
    k, n = (int(x) for x in part.split("/"))
    logits_fn, feats = load_policy(model)
    t0 = time.time()
    mine = [w for w in range(worlds) if w % n == k - 1]
    res: List[Dict[str, Any]] = []
    for lo in range(0, len(mine), chunk):
        res += ever_games(mine[lo:lo + chunk], seed, logits_fn, feats)
        print(f"[{len(res)}/{len(mine)} games] {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    write_jsonl(out, res)
    with open(out + ".meta.json", "w") as f:
        json.dump({"schema": 1, "stage": "ever", "model": os.path.basename(model),
                   "model_sha256": sha256_file(model), "deal": RANDOM, "swaps": [],
                   "hands": None, "ussr": [], "us": [], "net": None, "worlds": worlds, "part": part,
                   "seed": seed, "rows": len(res), "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    return 0

def pool(parts: Sequence[str], out: str, expect: int = 0) -> int:
    """Merge the parts (`matrix-<k>.jsonl.gz`, each with its `.meta.json`); with `expect`, name
    the parts that are missing in `<out>.parts.json`."""
    write_jsonl(out, list(read_jsonl(parts)))
    metas = [json.load(open(p + ".meta.json")) for p in parts if os.path.exists(p + ".meta.json")]
    if metas:
        m = dict(metas[0])
        same = ("model_sha256", "deal", "swaps", "ussr", "us", "seed", "worlds", "net")
        if any(x[key] != m[key] for x in metas for key in same):
            raise ValueError("the parts disagree on the model, the deal, the setups or the seed")
        m["parts"] = sorted(x["part"] for x in metas)
        m["rows"] = sum(int(x["rows"]) for x in metas)
        m.pop("wall_s", None)
        with open(out + ".meta.json", "w") as f:
            json.dump(m, f, indent=1)
    found = sorted(int(g.group(1)) for g in (re.search(r"matrix-(\d+)\.jsonl\.gz$", os.path.basename(p))
                                             for p in parts) if g)
    missing = [k for k in range(1, expect + 1) if k not in found]
    with open(out + ".parts.json", "w") as f:
        json.dump({"expected": expect, "found": found, "missing": missing}, f)
    if missing:
        print(f"parts missing: {missing} -- their worlds are not in {out}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- report


def solve_game(a: FloatArray, tol: float = 1e-6, iters: int = 200_000) -> Tuple[FloatArray, FloatArray, float, float]:
    """An equilibrium of the zero-sum game in which the column player gets `a[i, j]` and maximises
    and the row player minimises: (row mix, column mix, value, gap). Alternating regret matching+
    with linearly weighted averages; the gap is the column player's best reply to the row mix less
    the row player's best reply to the column mix, so the value is known to within it."""
    m, n = a.shape
    rx, ry = np.zeros(m), np.zeros(n)
    x, y = np.full(m, 1.0 / m), np.full(n, 1.0 / n)
    sx, sy = np.zeros(m), np.zeros(n)
    gap = math.inf
    for t in range(1, iters + 1):
        u = x @ a
        ry = np.maximum(ry + u - y @ u, 0.0)
        y = ry / ry.sum() if ry.sum() > 0 else np.full(n, 1.0 / n)
        v = -(a @ y)
        rx = np.maximum(rx + v - x @ v, 0.0)
        x = rx / rx.sum() if rx.sum() > 0 else np.full(m, 1.0 / m)
        sx += t * x
        sy += t * y
        if t % 100 == 0:
            xb, yb = sx / sx.sum(), sy / sy.sum()
            gap = float((xb @ a).max() - (a @ yb).min())
            if gap < tol:
                break
    xb, yb = sx / sx.sum(), sy / sy.sum()
    gap = float((xb @ a).max() - (a @ yb).min())
    return xb, yb, float(xb @ a @ yb), gap


def load(inputs: Sequence[str]) -> Tuple[Dict[str, Any], FloatArray, List[Dict[str, Any]]]:
    """The metadata, the US scores (worlds, USSR setups, US setups) and the rows, by world."""
    meta = json.load(open(inputs[0] + ".meta.json"))
    rows = sorted(read_jsonl(inputs), key=lambda r: int(r["world"]))
    nr, nc = len(meta["ussr"]), len(meta["us"])
    if len({int(r["world"]) for r in rows}) != len(rows):
        raise ValueError("a world appears twice")
    y = np.array([[int(ch) / 2 for ch in r["s"]] for r in rows], dtype=np.float64)
    if y.shape[1] != nr * nc:
        raise ValueError(f"{y.shape[1]} cells per world, the metadata names {nr} x {nc}")
    return meta, y.reshape(len(rows), nr, nc), rows


def mix(names: Sequence[str], counts: Mapping[str, int]) -> Optional[FloatArray]:
    """The counts' shares over the setups in `names`, renormalised; None if none of them is there."""
    w = np.array([float(counts.get(nm, 0)) for nm in names])
    return w / w.sum() if w.sum() > 0 else None


def coverage(names: Sequence[str], counts: Mapping[str, int]) -> float:
    """The share of the counted games whose setup is one of `names`."""
    return sum(counts.get(nm, 0) for nm in names) / sum(counts.values())


def se_of(d: FloatArray) -> float:
    return float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan


def pp(v: float, se: Optional[float] = None, signed: bool = False) -> str:
    s = f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"
    return s if se is None else f"{s} ± {100 * se:.1f}"


def where_dealt(rows: Sequence[Mapping[str, Any]], card: int) -> npt.NDArray[np.str_]:
    """Per world: "ussr" or "us" for the hand the card was dealt to, else "deck"."""
    return np.array(["ussr" if card in r["ussr_hand"] else "us" if card in r["us_hand"] else "deck"
                     for r in rows])


def split_tables(y: FloatArray, rows: Sequence[Mapping[str, Any]], ussr: Sequence[str], us: Sequence[str],
                 w_r: FloatArray, w_c: FloatArray, lab_r: str, lab_c: str, ref_r: int, ref_c: int
                 ) -> Tuple[List[str], Dict[str, Any]]:
    """The matrix split by where each of SPLIT_CARDS was dealt: each US setup against the USSR mix
    `w_r`, each USSR setup against the US mix `w_c`, with the paired difference from the reference
    setup in the same worlds."""
    ids = card_ids()
    w = y.shape[0]
    yc = np.einsum("wij,i->wj", y, w_r)   # each US setup against the USSR mix
    yr = y @ w_c                           # each USSR setup against the US mix

    def cell(a: FloatArray, b: FloatArray, is_ref: bool) -> str:
        if len(a) < 2:
            return "--"
        return f"{100 * a.mean():.1f}" if is_ref else f"{100 * a.mean():.1f} ({pp(float((a - b).mean()), se_of(a - b), True)})"

    lines = ["", "## By where the scoring cards were dealt", "",
             "Each world dealt its own hands, and every cell of it forced the setups whatever they were, so "
             "a setup here is played blind to its own hand. Splitting the worlds by where a scoring card went "
             "shows what the hand would change. In brackets: the paired difference from the reference setup "
             "in the same worlds, with its SE."]
    out: Dict[str, Any] = {}
    for name in SPLIT_CARDS:
        where = where_dealt(rows, ids[name])
        n = {g: int((where == g).sum()) for g in ("us", "ussr", "deck")}
        lines += ["", f"### {name}", "",
                  f"Dealt to the US in {n['us']:,} worlds, to the USSR in {n['ussr']:,}, to neither in {n['deck']:,}.", "",
                  f"US setups against {lab_r} (reference {us[ref_c]}):", "",
                  "| US setup | in the US hand | in the USSR hand | in neither | all |", "|:---|---:|---:|---:|---:|"]
        groups = [where == "us", where == "ussr", where == "deck", np.ones(w, dtype=bool)]
        for j in range(len(us)):
            lines.append(f"| {us[j]} | " + " | ".join(cell(yc[g, j], yc[g, ref_c], j == ref_c) for g in groups) + " |")
        lines += ["", f"USSR setups against {lab_c} (reference {ussr[ref_r]}; lower is better for the USSR):", "",
                  "| USSR setup | in the USSR hand | in the US hand | in neither | all |", "|:---|---:|---:|---:|---:|"]
        groups_r = [where == "ussr", where == "us", where == "deck", np.ones(w, dtype=bool)]
        for i in range(len(ussr)):
            lines.append(f"| {ussr[i]} | " + " | ".join(cell(yr[g, i], yr[g, ref_r], i == ref_r) for g in groups_r) + " |")
        out[name] = {g: {"n": n[g], "mean": y[where == g].mean(axis=0).tolist() if n[g] else None}
                     for g in ("us", "ussr", "deck")}
    return lines, out


def report(inputs: Sequence[str], out: str, bootstrap: int = 200, seed: int = 0) -> int:
    meta, y, rows = load(inputs)
    ussr, us = list(meta["ussr"]), list(meta["us"])
    who_r = dict(USSR_SETUPS)
    who_c = dict(US_SETUPS)
    w, nr, nc = y.shape
    m = y.mean(axis=0)
    rng = np.random.default_rng(seed)
    boots = [np.tensordot(rng.multinomial(w, np.full(w, 1.0 / w)) / w, y, axes=1) for _ in range(bootstrap)]
    random_deals = meta["deal"] == RANDOM
    net: Optional[Dict[str, Any]] = meta.get("net")
    net_us: Dict[str, str] = dict(net["us"]) if net else {}
    hu_r, hu_c = mix(ussr, HUMAN_USSR), mix(us, HUMAN_US)
    cov_r, cov_c = coverage(ussr, HUMAN_USSR), coverage(us, HUMAN_US)
    ref = ussr.index(REFERENCE_USSR) if REFERENCE_USSR in ussr else 0
    lines: List[str] = [
        "# Turn 1 setups " + ("over random deals" if random_deals else "on one deal")
        + ": every USSR setup against every US setup", "",
        f"Model `{meta['model']}` (sha256 `{meta['model_sha256'][:12]}`), greedy on both sides from the "
        f"turn-1 headline on. {w:,} worlds; each plays all {nr} x {nc} cells ({w * nr * nc:,} games), "
        f"the cells of a world on the same {'deal, ' if random_deals else ''}dice and later draws, so every "
        "comparison is paired. Score: the US's, win 1, draw 0.5; in percent.", ""]
    if random_deals:
        lines += ["## The deals", "",
                  f"Every world deals afresh: `init_game` from a seed of the world alone (`world_deal`), "
                  f"{len({r['deal'] for r in rows}):,} distinct deals, the hands in each row of the data. "
                  "Each cell forces both setups whatever the hands."]
    else:
        h = meta["hands"]
        lines += ["## The deal", "",
                  f"`init_game({meta['deal']})` with {'; '.join(meta['swaps'])}.", "",
                  f"* USSR ({h['ussr_ops']} ops): {', '.join(h['ussr'])}",
                  f"* US ({h['us_ops']} ops): {', '.join(h['us'])}",
                  f"* [the deal before setup in the workbench]({WORKBENCH}/?pos={h['pos']})"]
    if net:
        lines += ["", "## The network on this deal", "",
                  f"Its greedy USSR setup: **{net['ussr']}**{'' if net['ussr'] in ussr else ' (not a row here)'}. "
                  "Its greedy US reply to each USSR setup:", "",
                  "| USSR setup | network's US reply |", "|:---|:---|"]
        for nm, r in net_us.items():
            lines.append(f"| {nm} | {r}{'' if r in us else ' (not a column here)'} |")
    # the matrix
    cell_se = float(np.sqrt((y.var(axis=0, ddof=1)).mean() / w))
    diff_se = float(np.mean([se_of(y[:, i, j] - y[:, i, j2]) for i in range(nr) for j, j2 in
                             zip(range(nc), range(1, nc))])) if nc > 1 else math.nan
    dagger = "; a dagger: the network's own US reply" if net else ""
    lines += ["", "## The matrix", "",
              f"US score per cell. Columns: USSR setups; **bold**: the US's best reply in that column"
              f"{dagger}. One cell's SE is about "
              f"{100 * cell_se:.2f}; the difference of two cells in one column, about {100 * diff_se:.2f} (paired).", "",
              "| US setup | who | " + " | ".join(ussr) + " |", "|:---|:---|" + "---:|" * nr]
    best_c = m.argmax(axis=1)
    for j in range(nc):
        cells = []
        for i in range(nr):
            s = f"{100 * m[i, j]:.1f}"
            s = f"**{s}**" if best_c[i] == j else s
            cells.append(s + ("†" if net_us.get(ussr[i]) == us[j] else ""))
        lines.append(f"| {us[j]} | {who_c.get(us[j], '')} | " + " | ".join(cells) + " |")
    lines.append("| *US best reply* | | " + " | ".join(f"{100 * m[i].max():.1f}" for i in range(nr)) + " |")
    if hu_c is not None:
        lines.append("| *human US mix* | | " + " | ".join(f"{100 * m[i] @ hu_c:.1f}" for i in range(nr)) + " |")
    lines.append("| *mean of the columns* | | " + " | ".join(f"{100 * m[i].mean():.1f}" for i in range(nr)) + " |")
    # USSR rows
    lines += ["", "## Each USSR setup", "",
              "Sorted by the US's best reply (best for the USSR first). Runner-up: the second-best US "
              "reply and how far behind (paired SE). Human mix: the corpus's human US setups among the "
              f"columns at their pooled shares ({100 * cov_c:.0f}% of the corpus's US setups); humans' reply: "
              "their shares against that USSR setup. Δ ref: the best-reply score against "
              f"{ussr[ref]}'s (bootstrap SE over worlds).", "",
              "| USSR setup | who | US best reply | US score | runner-up | vs human US mix | "
              "vs humans' reply |" + (" vs network's reply |" if net else "") + " Δ ref at best reply |",
              "|:---|:---|:---|---:|:---|---:|---:|" + ("---:|" if net else "") + "---:|"]
    for i in sorted(range(nr), key=lambda i: m[i].max()):
        order = np.argsort(-m[i])
        b = int(order[0])
        r2 = int(order[1]) if nc > 1 else b
        hm = y[:, i, :] @ hu_c if hu_c is not None else None
        hr = mix(us, HUMAN_REPLY[ussr[i]]) if ussr[i] in HUMAN_REPLY else None
        reply = pp(float((y[:, i, :] @ hr).mean())) if hr is not None else "--"
        nj = us.index(net_us[ussr[i]]) if net_us.get(ussr[i]) in us else None
        nrep = (f" {pp(float(m[i, nj])) if nj is not None else '--'} |") if net else ""
        dref = [float(bm[i].max() - bm[ref].max()) for bm in boots]
        lines.append(
            f"| {ussr[i]} | {who_r.get(ussr[i], '')} | {us[b]} | {pp(float(m[i, b]), se_of(y[:, i, b]))} | "
            f"{us[r2]} ({pp(float(m[i, r2] - m[i, b]), se_of(y[:, i, r2] - y[:, i, b]), True)}) | "
            f"{pp(float(hm.mean()), se_of(hm)) if hm is not None else '--'} | {reply} |{nrep} "
            f"{pp(float(m[i].max() - m[ref].max()), float(np.std(dref, ddof=1)), True) if i != ref else '--'} |")
    # US columns
    lines += ["", "## Each US setup", "",
              "Sorted by the USSR's best reply (best for the US first): the US score when the USSR "
              f"picks the row that hurts this setup most; against the human USSR mix ({100 * cov_r:.0f}% of "
              f"the corpus's USSR setups); against {ussr[ref]}.", "",
              "| US setup | who | USSR best reply | US score | vs human USSR mix | "
              f"vs {ussr[ref]} | mean of the rows |", "|:---|:---|:---|---:|---:|---:|---:|"]
    for j in sorted(range(nc), key=lambda j: -m[:, j].min()):
        b = int(m[:, j].argmin())
        hm = y[:, :, j] @ hu_r if hu_r is not None else None
        lines.append(f"| {us[j]} | {who_c.get(us[j], '')} | {ussr[b]} | {pp(float(m[b, j]), se_of(y[:, b, j]))} | "
                     f"{pp(float(hm.mean()), se_of(hm)) if hm is not None else '--'} | "
                     f"{pp(float(m[ref, j]), se_of(y[:, ref, j]))} | {pp(float(m[:, j].mean()))} |")
    # the equilibrium
    xr, yc, val, gap = solve_game(m)
    bvals, bsup_r, bsup_c = [], np.zeros(nr), np.zeros(nc)
    for bm in boots:
        bx, by, bv, _ = solve_game(bm, tol=1e-5)
        bvals.append(bv)
        bsup_r += bx > 0.01
        bsup_c += by > 0.01
    lines += ["", "## The equilibrium", "",
              f"The matrix as a game (the USSR picks a row, the US a column without seeing it): value "
              f"**{pp(val, float(np.std(bvals, ddof=1)))}** US (bootstrap SE; solver gap {100 * gap:.4f}). "
              f"Support share: how often a setup has weight over 1% in {bootstrap} bootstrap resamples "
              "of the worlds.", "",
              "| side | setup | weight | support share |", "|:---|:---|---:|---:|"]
    for side, names, wts, sup in (("USSR", ussr, xr, bsup_r), ("US", us, yc, bsup_c)):
        for i in np.argsort(-wts):
            if wts[i] > 0.005 or sup[i] / bootstrap > 0.1:
                lines.append(f"| {side} | {names[i]} | {100 * wts[i]:.1f}% | {100 * sup[i] / bootstrap:.0f}% |")
    # exploitability of the populations
    lines += ["", "### What a best reply takes from each population", "",
              "| strategy | the opponent's best reply | US score | Δ from the value |", "|:---|:---|---:|---:|"]
    if hu_r is not None:
        hr_m = hu_r @ m
        lines.append(f"| USSR: the human mix | {us[int(hr_m.argmax())]} | {pp(float(hr_m.max()))} | "
                     f"{pp(float(hr_m.max() - val), signed=True)} |")
    if hu_c is not None:
        hc_m = m @ hu_c
        lines.append(f"| US: the human mix | {ussr[int(hc_m.argmin())]} | {pp(float(hc_m.min()))} | "
                     f"{pp(float(hc_m.min() - val), signed=True)} |")
    if net and net["ussr"] in ussr:
        i = ussr.index(net["ussr"])
        lines.append(f"| USSR: the network's {net['ussr']} | {us[int(m[i].argmax())]} | {pp(float(m[i].max()))} | "
                     f"{pp(float(m[i].max() - val), signed=True)} |")
    own_us = net_us.get(net["ussr"]) if net else None
    if own_us in us:
        j = us.index(own_us)
        lines.append(f"| US: the network's {own_us} | {ussr[int(m[:, j].argmin())]} | {pp(float(m[:, j].min()))} | "
                     f"{pp(float(m[:, j].min() - val), signed=True)} |")
    lines.append(f"| USSR: the equilibrium mix | | {pp(float((xr @ m).max()))} | {pp(float((xr @ m).max() - val), signed=True)} |")
    # paired contrasts against the reference row, column by column
    if nr > 1:
        lines += ["", f"## Every USSR setup against {ussr[ref]}, column by column", "",
                  f"US score of the row less {ussr[ref]}'s, in each column (paired); positive: worse for the USSR "
                  f"than {ussr[ref]}. A typical SE is {100 * float(np.mean([se_of(y[:, i, j] - y[:, ref, j]) for i in range(nr) if i != ref for j in range(nc)])):.2f}.", "",
                  "| US setup | " + " | ".join(nm for i, nm in enumerate(ussr) if i != ref) + " |",
                  "|:---|" + "---:|" * (nr - 1)]
        for j in range(nc):
            lines.append(f"| {us[j]} | " + " | ".join(f"{100 * (m[i, j] - m[ref, j]):+.1f}" for i in range(nr) if i != ref) + " |")
    split: Dict[str, Any] = {}
    if rows and "ussr_hand" in rows[0]:
        top_c = max((nm for nm in us if nm in HUMAN_US), key=lambda nm: HUMAN_US[nm], default=us[0])
        more, split = split_tables(
            y, rows, ussr, us,
            hu_r if hu_r is not None else np.full(nr, 1.0 / nr), hu_c if hu_c is not None else np.full(nc, 1.0 / nc),
            "the human USSR mix" if hu_r is not None else "the mean of the USSR setups",
            "the human US mix" if hu_c is not None else "the mean of the US setups", ref, us.index(top_c))
        lines += more
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"ussr": ussr, "us": us, "mean": m.tolist(), "worlds": w, "random_deals": random_deals,
                   "equilibrium": {"ussr": xr.tolist(), "us": yc.tolist(), "value": val, "gap": gap},
                   "split": split}, f, indent=1)
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deal")
    d.add_argument("--deal", type=int, default=DEAL)
    d.add_argument("--swap", nargs="*", default=list(SWAPS), help='"out card=in card", by name')
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help="a .pt checkpoint or its .onnx export")
    r.add_argument("--worlds", type=int, default=16000)
    r.add_argument("--part", default="1/1")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--deal", type=int, default=DEAL)
    r.add_argument("--swap", nargs="*", default=list(SWAPS), help='"out card=in card", by name')
    r.add_argument("--chunk", type=int, default=1536, help="games per playout batch")
    r.add_argument("--ussr", nargs="+", default=None, help="USSR setups by name (default: every listed one)")
    r.add_argument("--us", nargs="+", default=None, help="US setups by name (default: every listed one)")
    r.add_argument("--random-deals", action="store_true",
                   help="every world deals afresh (--deal and --swap unused); rows record both hands")
    r.add_argument("--out", required=True)
    h = sub.add_parser("headlines")
    h.add_argument("--model", required=True, help="a .pt checkpoint or its .onnx export")
    h.add_argument("--worlds", type=int, default=200000)
    h.add_argument("--part", default="1/1")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--chunk", type=int, default=1536, help="headline phases per batch")
    h.add_argument("--ussr", nargs="+", required=True, help="USSR setups by name")
    h.add_argument("--us", nargs="+", required=True, help="US setups by name")
    h.add_argument("--card", default="East European Unrest", help="keep the worlds where this card was dealt ...")
    h.add_argument("--holder", choices=("us", "ussr"), default="us", help="... to this side")
    h.add_argument("--out", required=True)
    y = sub.add_parser("reply")
    y.add_argument("--model", required=True, help="a .pt checkpoint (the searched player needs torch)")
    y.add_argument("--player", required=True, help="a load_agent spec, {MODEL} for --model, e.g. rollout:{MODEL}:4:32:4:z2:earlier:own")
    y.add_argument("--worlds", type=int, default=2000)
    y.add_argument("--part", default="1/1")
    y.add_argument("--seed", type=int, default=0)
    y.add_argument("--chunk", type=int, default=64, help="cells per batch")
    y.add_argument("--ussr", nargs="+", required=True, help="USSR setups by name")
    y.add_argument("--us", nargs="+", required=True, help="US setups by name")
    y.add_argument("--card", default="East European Unrest", help="the holder's forced turn-1 headline ...")
    y.add_argument("--holder", choices=("us", "ussr"), default="us", help="... and the side it was dealt to")
    y.add_argument("--turns", type=int, default=1, help="the replying side is searched through this turn")
    y.add_argument("--out", required=True)
    f1 = sub.add_parser("ar1")
    f1.add_argument("--model", required=True, help="a .pt checkpoint or its .onnx export")
    f1.add_argument("--worlds", type=int, default=200000)
    f1.add_argument("--part", default="1/1")
    f1.add_argument("--seed", type=int, default=0)
    f1.add_argument("--chunk", type=int, default=1536, help="cells per batch")
    f1.add_argument("--ussr", nargs="+", required=True, help="USSR setups by name")
    f1.add_argument("--us", nargs="+", required=True, help="US setups by name")
    f1.add_argument("--card", default="East European Unrest", help="the holder's forced turn-1 headline ...")
    f1.add_argument("--holder", choices=("us", "ussr"), default="us", help="... and the side it was dealt to")
    f1.add_argument("--arms", nargs="+", choices=AR1_ARMS, default=list(AR1_ARMS))
    f1.add_argument("--out", required=True)
    w = sub.add_parser("draws")
    w.add_argument("--model", required=True, help="a .pt checkpoint or its .onnx export")
    w.add_argument("--worlds", type=int, default=24000, help="games, each snapshotted after every deal")
    w.add_argument("--part", default="1/1")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--redeals", type=int, default=16, help="redeals of each snapshot's drawn cards")
    w.add_argument("--chunk", type=int, default=1536, help="games per playout batch")
    w.add_argument("--out", required=True)
    e = sub.add_parser("ever")
    e.add_argument("--model", required=True, help="a .pt checkpoint or its .onnx export")
    e.add_argument("--worlds", type=int, default=100000, help="games")
    e.add_argument("--part", default="1/1")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--chunk", type=int, default=1536, help="games per batch")
    e.add_argument("--out", required=True)
    q = sub.add_parser("pool")
    q.add_argument("--parts", nargs="+", required=True)
    q.add_argument("--expect", type=int, default=0)
    q.add_argument("--out", required=True)
    s = sub.add_parser("report")
    s.add_argument("--input", nargs="+", required=True)
    s.add_argument("--bootstrap", type=int, default=200)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "deal":
        return deal(a.deal, a.swap)
    if a.cmd == "run":
        return run(a.model, a.worlds, a.part, a.seed, a.deal, a.swap, a.chunk, a.out, a.ussr, a.us, a.random_deals)
    if a.cmd == "headlines":
        return headlines(a.model, a.worlds, a.part, a.seed, a.chunk, a.out, a.ussr, a.us, a.card, a.holder)
    if a.cmd == "reply":
        return reply(a.model, a.player, a.worlds, a.part, a.seed, a.chunk, a.out, a.ussr, a.us, a.card,
                     a.holder, a.turns)
    if a.cmd == "ar1":
        return ar1(a.model, a.worlds, a.part, a.seed, a.chunk, a.out, a.ussr, a.us, a.card, a.holder, a.arms)
    if a.cmd == "draws":
        return draws(a.model, a.worlds, a.part, a.seed, a.redeals, a.chunk, a.out)
    if a.cmd == "ever":
        return ever(a.model, a.worlds, a.part, a.seed, a.chunk, a.out)
    if a.cmd == "pool":
        return pool(a.parts, a.out, a.expect)
    return report(a.input, a.out, a.bootstrap)


if __name__ == "__main__":
    sys.exit(main())
