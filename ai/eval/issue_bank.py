"""P36a: candidate positions for the issues bank, for the owner's review before anything is labelled.

The owner's issue list (research/plans/P36_issue_bank_and_distillation_test.md), as five groups of
patterns. Each pattern is a predicate on a decision of A13+R48's greedy self-play; two that self-play
almost never reaches are constructed from real positions it does reach, and say so:

* group 1, options never tried -- `subs_coup` (a US Ops decision at DEFCON <= 3 with Nuclear Subs in
  effect; constructed by setting the flag where the event is absent), `wargames_lead` (Wargames held at
  DEFCON 2 with a lead over 6), the expert review's event spots (`oss_behind`, `kal_sk`,
  `glasnost_reformer`, `special_relationship_nato`, `star_wars_ahead`, each at the card's play-mode
  decision), `ames_discard` (the USSR's choice inside Aldrich Ames);
* group 2, delayed value -- `europe_control` (one side controls Europe, or is one battleground or one
  country short, Europe Scoring still to come), `opp_scoring` (the opponent holds a scoring card for a
  region where the mover is behind), `blockade_hold` (the US holds one card of 3+ Ops while Blockade is
  unseen and it has influence in West Germany);
* group 3, turn planning -- `fyp_early` (Five Year Plan with other cards, before the last round),
  `fyp_formosan_nato` (constructed: NATO and Formosan Resolution swapped into such a hand -- the owner's
  example), `scoring_held` (a scoring card held at AR1-2 while the network plays another card);
* group 4, decided positions -- `drift_win` (the critic past 0.9 and an exact instant win the network
  does not take), `drift_lost` (the critic below -0.9: a sample for the owner);
* group 5, the opening -- `setup` (the USSR's first setup decision of each deal, with the setup the
  network ends up placing) and `yugo_reds` (the US holding Independent Reds with the USSR at 3+ in
  Yugoslavia, turns 1-3).

Every row carries the position (`pos`, the workbench's link token), the network's greedy move, the
critic's reading, what the pattern asks about, and the proposed label source: `exact` (the game ends
within the line: `ai.eval.safety`), `oracle_root` (the true hand decides), `keyed_playouts` (long keyed
paired playouts, the follow-up's bias to be reported against an exact slice), or `owner`. Never the
network's own greedy continuation on a multi-step plan (P33c).

    PYTHONPATH=.:build/release python -m ai.eval.issue_bank --checkpoint swa.pt --games 2000 \\
        --out ai/eval/banks/issues_P36a_candidates.jsonl --report research/log/P36a_issue_bank_candidates.md
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.safety import find_instant_win
from bindings.ts_env import model_obs_features
from tools.lib.corpus_driver import position_token
from tools.lib.game_step import drain_chance
from tools.lib.jsonl import write_jsonl
from tools.lib.player_agent import NeuralAgent
from tools.lib.position_bank import CARD_NAMES, move_name

WORKBENCH = "https://mihaild.github.io/DeepStruggle"
C = {name: cid for cid, name in CARD_NAMES.items()}
NUCLEAR_SUBS, WARGAMES, OSS, KAL, GLASNOST = C["Nuclear Subs"], C["Wargames"], C["“One Small Step…”"], \
    C["Soviets Shoot Down KAL-007"], C["Glasnost"]
SPECIAL_REL, STAR_WARS, AMES, FYP, NATO = C["Special Relationship"], C["Star Wars"], C["Aldrich Ames Remix"], \
    C["Five Year Plan"], C["NATO"]
FORMOSAN, BLOCKADE, IND_REDS = C["Formosan Resolution"], C["Blockade"], C["Independent Reds"]
SCORING = {C["Europe Scoring"]: ts.Region.EUROPE, C["Asia Scoring"]: ts.Region.ASIA,
           C["Middle East Scoring"]: ts.Region.MIDDLE_EAST, C["Central America Scoring"]: ts.Region.CENTRAL_AMERICA,
           C["South America Scoring"]: ts.Region.SOUTH_AMERICA, C["Africa Scoring"]: ts.Region.AFRICA}
# engine/include/ts/constants.hpp effect_bits
BIT_NATO, BIT_NUCLEAR_SUBS, BIT_REFORMER = 1 << 0, 1 << 13, 1 << 28
SELECT_CARD, SELECT_PLAY_MODE, SELECT_OP_MODE = (int(ts.DecisionType.SELECT_CARD), int(ts.DecisionType.SELECT_PLAY_MODE),
                                                 int(ts.DecisionType.SELECT_OP_MODE))
COUNTRIES = [ts.MapData.get_country_info(i) for i in range(84)]
WEST_GERMANY, SOUTH_KOREA, YUGOSLAVIA = (ts.MapData.get_country_by_name(n)
                                         for n in ("West Germany", "South Korea", "Yugoslavia"))
LABEL = {"exact": "the exact solver (ai.eval.safety): the game ends within the line",
         "oracle_root": "the oracle root: the true hand decides",
         "keyed_playouts": "long keyed paired playouts, the follow-up's bias reported against an exact slice",
         "owner": "the owner's judgment"}


def side(p: ts.Player) -> int:
    return 1 if p == ts.Player.US else 2


def hand(st: ts.GameState, s: int) -> List[int]:
    locs = ((ts.CardLocation.HAND_US_UNKNOWN, ts.CardLocation.HAND_US_KNOWN) if s == 1
            else (ts.CardLocation.HAND_USSR_UNKNOWN, ts.CardLocation.HAND_USSR_KNOWN))
    return [c for c in range(1, 111) if st.get_card_location(c) in locs]


def effect(st: ts.GameState, bit: int) -> bool:
    return bool(int(st.persistent_effects) & bit)


def controls(st: ts.GameState, cid: int, s: int) -> bool:
    c = st.get_country(cid)
    mine, theirs = (c.us_influence, c.ussr_influence) if s == 1 else (c.ussr_influence, c.us_influence)
    return mine > 0 and mine - theirs >= COUNTRIES[cid]["stability"]


def region_level(st: ts.GameState, region: Any, s: int) -> int:
    """0 none, 1 presence, 2 domination, 3 control -- the scoring rules' levels for side s."""
    ids = [i for i in range(84) if COUNTRIES[i]["region"] == region]
    bgs = [i for i in ids if COUNTRIES[i]["battleground"]]
    mine = [i for i in ids if controls(st, i, s)]
    theirs = [i for i in ids if controls(st, i, 3 - s)]
    mb, tb = sum(i in bgs for i in mine), sum(i in bgs for i in theirs)
    if not mine:
        return 0
    if mb == len(bgs) and len(mine) > len(theirs):
        return 3
    if len(mine) > len(theirs) and mb > tb and len(mine) > mb:
        return 2
    return 1


def europe_shortfall(st: ts.GameState, s: int) -> Optional[str]:
    """'control', or how side s is one step short of controlling Europe, else None."""
    ids = [i for i in range(84) if COUNTRIES[i]["region"] == ts.Region.EUROPE]
    bgs = [i for i in ids if COUNTRIES[i]["battleground"]]
    mine = [i for i in ids if controls(st, i, s)]
    theirs = [i for i in ids if controls(st, i, 3 - s)]
    missing = sum(i not in mine for i in bgs)
    if missing == 0 and len(mine) > len(theirs):
        return "control"
    if missing == 1 and len(mine) + 1 > len(theirs):
        return "one battleground short"
    if missing == 0 and len(mine) == len(theirs):
        return "one country short"
    return None


def last_ar(st: ts.GameState) -> int:
    return 6 if int(st.turn) <= 3 else 7


Row = Dict[str, Any]


def base_row(st: ts.GameState, group: int, pattern: str, label: str, asks: str, net: int, critic: float,
             constructed: str = "") -> Row:
    tok = position_token(st)
    me = st.ctx().decision_player
    return {"schema": 1, "id": hashlib.sha256((pattern + tok).encode()).hexdigest()[:16], "group": group,
            "pattern": pattern, "pos": tok, "link": f"{WORKBENCH}/?pos={tok}&model=off",
            "side": "US" if me == ts.Player.US else "USSR", "turn": int(st.turn), "ar": int(st.action_round),
            "decision_type": ts.DecisionType(int(st.ctx().decision_type)).name, "defcon": int(st.defcon),
            "vp": int(st.victory_points), "network": {"action": int(net), "name": move_name(st, int(net))},
            "critic": round(float(critic), 4), "asks": asks, "label_source": label, "constructed": constructed}


def mined(st: ts.GameState, net: int, critic: float) -> List[Row]:
    """The mined patterns that hold at decision `st` (the network's greedy move `net`)."""
    out: List[Row] = []
    me = side(st.ctx().decision_player)
    dt = int(st.ctx().decision_type)
    pend = int(st.ctx().pending_op_card)
    mine = hand(st, me)
    lead = int(st.victory_points) * (1 if me == 1 else -1)
    if me == 1 and effect(st, BIT_NUCLEAR_SUBS) and dt in (SELECT_PLAY_MODE, SELECT_OP_MODE) and int(st.defcon) <= 3:
        out.append(base_row(st, 1, "subs_coup", "keyed_playouts", "coup a battleground: Nuclear Subs keeps DEFCON",
                            net, critic))
    if WARGAMES in mine and int(st.defcon) == 2 and lead > 6 and dt == SELECT_CARD:
        win = find_instant_win(st)
        out.append(base_row(st, 1, "wargames_lead", "exact",
                            f"play Wargames for the win (exact instant win: {None if win is None else move_name(st, win)})",
                            net, critic))
    if dt == SELECT_PLAY_MODE:
        my_space = int(st.us_space_track if me == 1 else st.ussr_space_track)
        op_space = int(st.ussr_space_track if me == 1 else st.us_space_track)
        spots = [(OSS, op_space - my_space == 1, "oss_behind", "event One Small Step one box behind"),
                 (KAL, me == 1 and controls(st, SOUTH_KOREA, 1) and int(st.defcon) >= 3, "kal_sk",
                  "event KAL-007 with South Korea US-controlled"),
                 (GLASNOST, me == 2 and effect(st, BIT_REFORMER), "glasnost_reformer", "event Glasnost with The Reformer in play"),
                 (SPECIAL_REL, me == 1 and effect(st, BIT_NATO), "special_relationship_nato",
                  "event Special Relationship with NATO in effect"),
                 (STAR_WARS, me == 1 and my_space > op_space, "star_wars_ahead", "event Star Wars ahead in space")]
        for cid, cond, name, asks in spots:
            if pend == cid and cond:
                out.append(base_row(st, 1, name, "keyed_playouts", asks, net, critic))
    if me == 2 and int(st.ctx().resolving_card) == AMES and dt == SELECT_CARD:
        out.append(base_row(st, 1, "ames_discard", "keyed_playouts", "which US card Aldrich Ames discards", net, critic))
    if dt == SELECT_CARD and st.current_phase == ts.Phase.ACTION_ROUND:
        eu_loc = st.get_card_location(C["Europe Scoring"])
        if eu_loc not in (ts.CardLocation.DISCARD_PILE, ts.CardLocation.REMOVED_FROM_GAME):
            for s, who in ((me, "the mover"), (3 - me, "the opponent")):
                short = europe_shortfall(st, s)
                if short:
                    out.append(base_row(st, 2, "europe_control", "owner",
                                        f"{who} has Europe {short}, Europe Scoring still to come", net, critic))
                    break
        theirs = hand(st, 3 - me)
        for cid, region in SCORING.items():
            if cid in theirs and region_level(st, region, me) < region_level(st, region, 3 - me):
                out.append(base_row(st, 2, "opp_scoring", "oracle_root",
                                    f"the opponent holds {CARD_NAMES[cid]} and the mover is behind there", net, critic))
                break
        if me == 1 and int(st.turn) <= 3 and st.get_card_location(BLOCKADE) in (
                ts.CardLocation.DRAW_DECK, ts.CardLocation.HAND_USSR_UNKNOWN) \
                and st.get_country(WEST_GERMANY).us_influence > 0:
            big = [c for c in mine if int(ts.CardData.get_card_info(c)["ops"]) >= 3]
            if len(big) == 1:
                out.append(base_row(st, 2, "blockade_hold", "oracle_root",
                                    f"hold {CARD_NAMES[big[0]]}, the only 3+ Ops card, while Blockade is unseen", net, critic))
        if me == 2 and FYP in mine and len(mine) >= 3 and int(st.action_round) < last_ar(st):
            out.append(base_row(st, 3, "fyp_early", "owner", "Five Year Plan now, with other cards and rounds left",
                                net, critic))
        held = [c for c in mine if c in SCORING]
        if held and int(st.action_round) <= 2 and int(net) - 1 not in [c - 1 for c in held]:
            out.append(base_row(st, 3, "scoring_held", "owner",
                                f"{CARD_NAMES[held[0]]} held at AR{int(st.action_round)} while the network plays another card",
                                net, critic))
        if me == 1 and IND_REDS in mine and int(st.turn) <= 3 and st.get_country(YUGOSLAVIA).ussr_influence >= 3:
            out.append(base_row(st, 5, "yugo_reds", "owner",
                                "Independent Reds against 3+ USSR influence in Yugoslavia", net, critic))
    if abs(critic) > 0.9:
        if critic > 0.9:
            win = find_instant_win(st)
            if win is not None and win != net:
                out.append(base_row(st, 4, "drift_win", "exact",
                                    f"take the instant win ({move_name(st, win)}) in a decided position", net, critic))
        else:
            out.append(base_row(st, 4, "drift_lost", "owner", "the critic says lost: does the move still make sense",
                                net, critic))
    return out


def constructed(st: ts.GameState, net_of: Callable[[ts.GameState], Tuple[int, float]]) -> List[Row]:
    """The two patterns self-play rarely reaches, built from the real position `st`."""
    out: List[Row] = []
    me = side(st.ctx().decision_player)
    dt = int(st.ctx().decision_type)
    if me == 1 and dt == SELECT_OP_MODE and int(st.defcon) == 2 and not effect(st, BIT_NUCLEAR_SUBS) \
            and any(COUNTRIES[i]["battleground"] and st.get_country(i).ussr_influence > 0 for i in range(84)):
        w = st.clone()
        w.persistent_effects = int(w.persistent_effects) | BIT_NUCLEAR_SUBS
        net, crit = net_of(w)
        out.append(base_row(w, 1, "subs_coup", "keyed_playouts", "coup a battleground: Nuclear Subs keeps DEFCON",
                            net, crit, constructed="Nuclear Subs set in effect on a real position"))
    if me == 2 and dt == SELECT_CARD and int(st.turn) <= 3 and int(st.action_round) < last_ar(st) - 1:
        mine = hand(st, 2)
        if FYP in mine and NATO not in mine and FORMOSAN not in mine and len(mine) >= 4:
            free = (ts.CardLocation.DRAW_DECK, ts.CardLocation.HAND_US_UNKNOWN)
            if st.get_card_location(NATO) in free and st.get_card_location(FORMOSAN) in free:
                others = [c for c in mine if c not in (FYP,) and c not in SCORING][:2]
                if len(others) == 2:
                    w = st.clone()
                    for new, old in zip((NATO, FORMOSAN), others):
                        where = w.get_card_location(new)
                        w.set_card_location(new, w.get_card_location(old))
                        w.set_card_location(old, where)
                    net, crit = net_of(w)
                    out.append(base_row(w, 3, "fyp_formosan_nato", "owner",
                                        "Five Year Plan held with NATO and Formosan Resolution: NATO first (owner's example)",
                                        net, crit, constructed="NATO and Formosan Resolution swapped into a real USSR hand"))
    return out


def setup_rows(model: Any, dev: torch.device, deals: int, seed: int) -> List[Row]:
    """The USSR's first setup decision of `deals` deals, with the setup the network places from it."""
    feats = model_obs_features(model)
    out: List[Row] = []
    for d in range(deals):
        st = ts.GameState()
        ts.Engine.init_game(st, seed + d)
        drain_chance(st)
        first = st.clone()
        placed: Counter = Counter()
        while st.current_phase == ts.Phase.SETUP and st.ctx().decision_player == ts.Player.USSR:
            net, _ = _greedy(model, dev, feats, st)
            ma = ts.decode_flat_action(st, net)
            placed[ts.MapData.get_country_name(int(ma.primary_id))] += 1
            ts.Engine.step(st, ma)
            drain_chance(st)
        net, crit = _greedy(model, dev, feats, first)
        row = base_row(first, 5, "setup", "owner",
                       "the USSR setup: the network places " + ", ".join(f"{k} {v}" for k, v in placed.most_common()),
                       net, crit)
        out.append(row)
    return out


def _greedy(model: Any, dev: torch.device, feats: int, st: ts.GameState) -> Tuple[int, float]:
    p = st.ctx().decision_player
    obs = torch.from_numpy(np.asarray(ts.extract_observation_features(st, p, feats), dtype=np.float32))[None].to(dev)
    m = torch.from_numpy(np.asarray(ts.ActionMask.generate_flat_mask(st), dtype=np.uint8))[None].to(dev)
    with torch.no_grad():
        lg, v, _ = model(obs, m)
    return int(lg[0].float().masked_fill(m[0] == 0, float("-inf")).argmax()), float(v.reshape(-1)[0])


def mine_games(model: Any, dev: torch.device, games: int, seed: int, cap: int) -> List[Row]:
    feats = model_obs_features(model)
    runner = ts.VectorizedBatchRunner(games, seed)
    if feats:
        runner.set_obs_features([feats] * games, [feats] * games)
    done = np.zeros(games, dtype=bool)
    rows: Dict[str, List[Row]] = defaultdict(list)
    per_game: Counter = Counter()
    for _ in range(6000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32)).to(dev)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            lg, v, _ = model(obs, torch.from_numpy(masks).to(dev))
        lg_np = lg.float().cpu().numpy()
        v_np = v.float().reshape(-1).cpu().numpy()
        acts = np.where(masks > 0, lg_np, -np.inf).argmax(axis=1)
        acts[done] = 0
        players = np.asarray(runner.get_decision_players())
        for i in np.flatnonzero(~done & (players != 0) & (masks.sum(axis=1) >= 2)):
            st = runner.get_state(int(i))
            found = mined(st, int(acts[i]), float(v_np[i]))
            found += constructed(st, lambda w: _greedy(model, dev, feats, w))
            for r in found:
                key = (int(i), r["pattern"])
                if len(rows[r["pattern"]]) < cap and per_game[key] < 2:
                    rows[r["pattern"]].append(r)
                    per_game[key] += 1
        runner.step_flat_all(acts.tolist(), True)
    return [r for rs in rows.values() for r in rs]


def report(rows: List[Row], games: int, checkpoint: str) -> str:
    by: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by[r["pattern"]].append(r)
    groups = {1: "options never tried", 2: "delayed, conditional value", 3: "no plan for the turn's cards",
              4: "drift in decided positions", 5: "opening quirks"}
    out = ["# P36a -- the issues bank: candidate positions for the owner's review", "",
           f"Mined from {games} greedy self-play games of `{checkpoint}` (`ai/eval/issue_bank.py`); two patterns "
           "constructed from real positions, marked. **Nothing is labelled yet**: each row names the proposed label "
           "source, and the owner chooses which situations go in. Positions: "
           "`ai/eval/banks/issues_P36a_candidates.jsonl` (each with a workbench link).", "",
           "Label sources: " + "; ".join(f"`{k}` -- {v}" for k, v in LABEL.items()) + ".", "",
           "| group | pattern | rows | proposed label | network's move matches the question* | example |",
           "|:---|:---|---:|:---|:---|:---|"]
    for g in sorted(groups):
        for pat in sorted({r["pattern"] for r in rows if r["group"] == g}):
            rs = by[pat]
            ex = rs[0]
            out.append(f"| {g}. {groups[g]} | `{pat}` | {len(rs)} | {ex['label_source']} | "
                       f"{Counter(r['network']['name'] for r in rs).most_common(2)} | "
                       f"[{ex['side']} T{ex['turn']} AR{ex['ar']}]({ex['link']}): {ex['asks']} |")
    out += ["", "\\* the network's two commonest moves at the pattern's rows, for orientation only.", ""]
    for g in sorted(groups):
        for pat in sorted({r["pattern"] for r in rows if r["group"] == g}):
            out += [f"## `{pat}` (group {g})", ""]
            for r in by[pat][:6]:
                out.append(f"* [{r['side']} T{r['turn']} AR{r['ar']} {r['decision_type']}, DEFCON {r['defcon']}, VP {r['vp']}, "
                           f"critic {r['critic']:+.2f}]({r['link']}) -- {r['asks']}; network: {r['network']['name']}"
                           + (f" ({r['constructed']})" if r["constructed"] else ""))
            out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--cap", type=int, default=60, help="rows per pattern")
    ap.add_argument("--deals", type=int, default=8, help="setup positions")
    ap.add_argument("--seed", type=int, default=3636)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    rows = mine_games(model, dev, a.games, a.seed, a.cap) + setup_rows(model, dev, a.deals, a.seed)
    write_jsonl(a.out, rows)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(report(rows, a.games, a.checkpoint) + "\n")
    print(f"{len(rows)} candidate rows: " + ", ".join(f"{k} {v}" for k, v in Counter(r['pattern'] for r in rows).most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
