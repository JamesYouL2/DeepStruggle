"""The doctrine census: how often the model follows a strong player's card rules, from its self-play.

The rules are the fork owner's (a top Twilight Struggle player; struggler's
`docs/EXPERT_STRATEGY.md` and `docs/notes/claude/principles-from-strong-play.md`, revised
2026-10-01). Greedy self-play from many deals; every card play where the model had a real choice
(two or more legal play modes) is logged with its card, side, mode and context, and every
headline with the hand it was chosen from. The report gives:

* each rule: how often it applied and how often the model complied;
* per card, how often the model events it against playing it for Ops or spacing it;
* turn-1 headlines: P(headlined | held) for each card, against the owner's lists;
* coup targets (2-stability non-battlegrounds, the USSR's turn-1 first coup) and the Nordics.

A rule broken often is a candidate for paired playouts (`branch_oracle`), which say whether
breaking it costs this model anything; the census alone does not.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.ops_block import NODE_OFFSET, country_table
from bindings.action_encoder import ActionEncoder

PolicyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
MODES = ("event", "space", "influence", "coup", "realign")
N_CARDS = 110
LATE_WAR_TURN = 8
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Card ids (rules/cards.json).
GRAIN_SALES, VOICE_OF_AMERICA, COLONIAL_REAR_GUARDS = 67, 74, 63
ALDRICH_AMES, DESTALINIZATION, DECOLONIZATION = 98, 33, 30
NAZI_SCIENTIST, JUNTA, TERRORISM = 18, 47, 92
FORMOSAN, COMECON, NORAD = 35, 14, 106
STAR_WARS, OPEC, ALLIANCE_FOR_PROGRESS, CHE = 85, 61, 78, 107
FIVE_YEAR_PLAN = 5
NORDICS = {"Norway": 2, "Sweden": 3, "Finland": 5}
OPEC_COUNTRIES = ("Egypt", "Iran", "Libya", "Saudi Arabia", "Iraq", "Gulf States", "Venezuela")
IHC_FLAG = 1 << 36            # effect_bits::IRANIAN_HOSTAGE_CRISIS_PLAY

# The owner's turn-1 headline candidates (2026-10-01).
US_T1_HEADLINES = ("Middle East Scoring", "Defectors", "Containment", "Marshall Plan", "Captured Nazi Scientist")
USSR_T1_HEADLINES = ("Vietnam Revolts", "Nasser", "Socialist Governments", "Suez Crisis", "Red Scare/Purge",
                     "Arab-Israeli War", "Captured Nazi Scientist")


def cards() -> Dict[int, Dict[str, Any]]:
    with open(os.path.join(_ROOT, "rules", "cards.json"), encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw["cards"] if isinstance(raw, dict) else raw
    return {int(c["id"]): c for c in rows}


def _side_name(p: ts.Player) -> str:
    return "US" if p == ts.Player.US else "USSR"


def _owner(card_side: str, player: str) -> str:
    """'own', 'opp' or 'neutral' from the player's seat."""
    if card_side == "neutral":
        return "neutral"
    return "own" if card_side.upper() == player else "opp"


def _map() -> List[Dict[str, Any]]:
    with open(os.path.join(_ROOT, "rules", "map.json"), encoding="utf-8") as f:
        return sorted(json.load(f)["countries"], key=lambda c: int(c["id"]))


def card_context(st: ts.GameState, cid: int) -> Dict[str, Any]:
    """The board facts a conditional rule reads, for the cards that have one: the space race for
    Star Wars, the VP OPEC and Alliance for Progress would score (`mid_war.cpp`)."""
    if cid == STAR_WARS:
        return {"us_space": int(st.us_space_track), "ussr_space": int(st.ussr_space_track)}
    if cid not in (OPEC, ALLIANCE_FOR_PROGRESS):
        return {}
    stab, _, _ = country_table()
    rows = _map()
    def ctrl(i: int, us: bool) -> bool:
        c = st.get_country(i)
        mine, theirs = (c.us_influence, c.ussr_influence) if us else (c.ussr_influence, c.us_influence)
        return int(mine) - int(theirs) >= int(stab[i])
    if cid == OPEC:
        return {"event_vp": sum(ctrl(int(r["id"]), False) for r in rows if r["name"] in OPEC_COUNTRIES)}
    return {"event_vp": sum(ctrl(int(r["id"]), True) for r in rows if r["battleground"]
                            and str(r["region"]).lower().replace(" ", "_") in ("central_america", "south_america"))}


def _hand(st: ts.GameState, p: ts.Player) -> List[int]:
    locs = ((ts.CardLocation.HAND_US_KNOWN, ts.CardLocation.HAND_US_UNKNOWN) if p == ts.Player.US
            else (ts.CardLocation.HAND_USSR_KNOWN, ts.CardLocation.HAND_USSR_UNKNOWN))
    return [c for c in range(1, N_CARDS + 1) if st.get_card_location(c) in locs]


def _ihc(st: ts.GameState) -> bool:
    try:
        return bool(st.has_flag(IHC_FLAG))
    except TypeError:
        return st.get_card_location(82) == ts.CardLocation.REMOVED_FROM_GAME


def collect(act: PolicyFn, n_games: int, seed: int, envs: int = 32, max_steps: int = 2_000_000) -> Dict[str, Any]:
    """Greedy self-play of `n_games` games. Returns the logged plays, headlines, coups and boards."""
    info = cards()
    stab, bg, names = country_table()
    runner = ts.VectorizedBatchRunner(envs, seed)
    runner.refresh_all()
    plays: List[Dict[str, Any]] = []
    headlines: List[Dict[str, Any]] = []
    coups: List[Dict[str, Any]] = []
    boards: List[Dict[str, Any]] = []
    game_id = list(range(envs))
    next_game = envs
    started = envs
    seen_turn: List[set] = [set() for _ in range(envs)]
    first_coup: List[set] = [set() for _ in range(envs)]
    done_games = 0
    finished: set = set()
    for _ in range(max_steps):
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        acts = act(obs, masks)
        for i in range(envs):
            m = masks[i]
            modes_legal = m[MODE_BASE:MODE_BASE + 5].astype(bool)
            card_select = bool(m[:N_CARDS].any())
            node = bool(m[NODE_OFFSET:NODE_OFFSET + 84].any())
            if not (modes_legal.any() or card_select or node):
                continue
            st = runner.get_state(i)
            p = st.ctx().decision_player
            if p == ts.Player.NONE:
                continue
            side = _side_name(p)
            turn = int(st.turn)
            if turn not in seen_turn[i] and st.current_phase == ts.Phase.HEADLINE:
                seen_turn[i].add(turn)
                if turn in (2, 4, 8):
                    us = [int(st.get_country(c).us_influence) for c in NORDICS.values()]
                    ussr = [int(st.get_country(c).ussr_influence) for c in NORDICS.values()]
                    boards.append({"game": game_id[i], "turn": turn, "nordics_us": us, "nordics_ussr": ussr,
                                   "space": [int(st.us_space_track), int(st.ussr_space_track)]})
            a = int(acts[i])
            vp_mine = int(st.victory_points) * (1 if p == ts.Player.US else -1)
            if card_select and st.current_phase == ts.Phase.HEADLINE and st.headline_stage == 0 and a < N_CARDS:
                headlines.append({"game": game_id[i], "turn": turn, "side": side, "card": a + 1,
                                  "hand": _hand(st, p)})
            elif modes_legal.sum() >= 2 and MODE_BASE <= a < MODE_BASE + 5:
                cid = int(st.ctx().pending_op_card)
                c = info.get(cid)
                if c is None:
                    continue
                mode = MODES[a - MODE_BASE]
                own = _owner(str(c["side"]), side)
                fires = mode == "event" or (own == "opp" and mode != "space")
                plays.append({"game": game_id[i], "turn": turn, "ar": int(st.action_round), "side": side,
                              "card": cid, "owner": own, "mode": mode, "fires": fires,
                              "legal": [MODES[k] for k in np.flatnonzero(modes_legal)],
                              "vp": vp_mine, "defcon": int(st.defcon), "ihc": _ihc(st),
                              **card_context(st, cid)})
            elif node and st.ctx().op_mode == ts.OpMode.COUP and NODE_OFFSET <= a < NODE_OFFSET + 84:
                ctry = a - NODE_OFFSET
                first = (turn == 1 and side == "USSR" and side not in first_coup[i]
                         and st.current_phase == ts.Phase.ACTION_ROUND)
                if first:
                    first_coup[i].add(side)
                coups.append({"game": game_id[i], "turn": turn, "ar": int(st.action_round), "side": side,
                              "country": names[ctry], "stability": int(stab[ctry]), "bg": bool(bg[ctry]),
                              "card": int(st.ctx().pending_op_card or st.ctx().resolving_card), "ussr_t1_first": first})
        runner.step_flat_all([int(x) for x in acts], auto_advance=True)
        ends = np.flatnonzero(np.array(runner.get_terminals()))
        for i in ends:
            if game_id[i] < 0:
                continue                      # already over and not reset: not a new ending
            done_games += 1
            finished.add(game_id[i])
            if started < n_games:
                runner.reset_game(int(i), seed * 1_000_003 + next_game)
                game_id[i] = next_game
                next_game += 1
                started += 1
                seen_turn[i] = set()
                first_coup[i] = set()
            else:
                game_id[i] = -1
        if len(ends):
            runner.refresh_all()
        if done_games >= n_games:
            break
    # Games still running when the quota ends are dropped from the logs.
    keep = lambda rows: [r for r in rows if r["game"] in finished]  # noqa: E731
    return {"plays": keep(plays), "headlines": keep(headlines), "coups": keep(coups),
            "boards": keep(boards), "games": done_games}


# --- the rules ----------------------------------------------------------------------------------

Rule = Tuple[str, Callable[[Dict[str, Any]], bool], Callable[[Dict[str, Any]], bool]]


def is_last_round(turn: int, ar: int) -> bool:
    """The phasing player's last action round of the turn (an eighth, when earned, counts)."""
    return ar >= (6 if turn <= 3 else 7)


def timing_rules() -> List[Tuple[str, int, str]]:
    """(name, card, side): cards that side holds for its last action round."""
    return [("USSR plays Five Year Plan in its last action round", FIVE_YEAR_PLAN, "USSR"),
            ("US plays Aldrich Ames Remix in its last action round", ALDRICH_AMES, "US")]


def rules() -> List[Rule]:
    """(name, applies(play), complies(play)) over the logged card plays (all with a real choice)."""
    def own_event(cid: int, side: Optional[str] = None, before: Optional[int] = None) -> Tuple[Callable, Callable]:
        def applies(r: Dict[str, Any]) -> bool:
            return (r["card"] == cid and "event" in r["legal"] and (side is None or r["side"] == side)
                    and (before is None or r["turn"] < before))
        return applies, lambda r: r["mode"] == "event"

    def never_fires(cid: int, side: str, before: Optional[int] = None) -> Tuple[Callable, Callable]:
        def applies(r: Dict[str, Any]) -> bool:
            return r["card"] == cid and r["side"] == side and (before is None or r["turn"] < before)
        return applies, lambda r: not r["fires"]

    def never_event(cid: int, side: str) -> Tuple[Callable, Callable]:
        return (lambda r: r["card"] == cid and r["side"] == side and "event" in r["legal"],
                lambda r: r["mode"] != "event")

    out: List[Rule] = [
        ("US always events Grain Sales to Soviets", *own_event(GRAIN_SALES, "US")),
        ("US always events The Voice of America", *own_event(VOICE_OF_AMERICA, "US")),
        ("US (basically) always events Colonial Rear Guards", *own_event(COLONIAL_REAR_GUARDS, "US")),
        ("USSR always events Aldrich Ames Remix", *own_event(ALDRICH_AMES, "USSR")),
        ("USSR always events De-Stalinization before turn 5", *own_event(DESTALINIZATION, "USSR", before=5)),
        ("Either side always events Captured Nazi Scientist", *own_event(NAZI_SCIENTIST)),
        ("Either side always events Junta", *own_event(JUNTA)),
        ("USSR never lets Grain Sales to Soviets fire", *never_fires(GRAIN_SALES, "USSR")),
        ("USSR never lets The Voice of America fire", *never_fires(VOICE_OF_AMERICA, "USSR")),
        ("US never lets Decolonization fire before the Late War", *never_fires(DECOLONIZATION, "US", before=LATE_WAR_TURN)),
        ("US never events Formosan Resolution", *never_event(FORMOSAN, "US")),
        ("US never events NORAD", *never_event(NORAD, "US")),
        ("USSR never events Comecon", *never_event(COMECON, "USSR")),
        ("US always events Star Wars when ahead in space",
         lambda r: r["card"] == STAR_WARS and r["side"] == "US" and "event" in r["legal"]
         and r.get("us_space", 0) > r.get("ussr_space", 0),
         lambda r: r["mode"] == "event"),
        ("USSR always events OPEC when it scores 5+ VP",
         lambda r: r["card"] == OPEC and r["side"] == "USSR" and "event" in r["legal"] and r.get("event_vp", 0) >= 5,
         lambda r: r["mode"] == "event"),
        ("US always events Alliance for Progress when it scores 5+ VP",
         lambda r: r["card"] == ALLIANCE_FOR_PROGRESS and r["side"] == "US" and "event" in r["legal"]
         and r.get("event_vp", 0) >= 5,
         lambda r: r["mode"] == "event"),
        ("USSR events Che well over half the time (target: >50%)",
         lambda r: r["card"] == CHE and r["side"] == "USSR" and "event" in r["legal"],
         lambda r: r["mode"] == "event"),
        ("Terrorism is evented when behind or after Iranian Hostage Crisis",
         lambda r: r["card"] == TERRORISM and r["owner"] == "neutral" and "event" in r["legal"]
         and (r["vp"] < 0 or (r["ihc"] and r["side"] == "USSR")),
         lambda r: r["mode"] == "event"),
    ]
    return out


def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%" if n else "—"


def report(data: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    info = cards()
    plays, heads, coups, boards = data["plays"], data["headlines"], data["coups"], data["boards"]
    games = data["games"]
    out = [f"# Doctrine census — {meta.get('model', '?')}", "",
           f"{games} greedy self-play games; {len(plays)} card plays with a real choice of mode, "
           f"{len(heads)} headlines, {len(coups)} coup targets.", "",
           "## The rules", "", "| rule | applied | complied |", "|:---|---:|---:|"]
    summary: Dict[str, Any] = {"meta": meta, "games": games, "rules": {}}
    for name, applies, complies in rules():
        sel = [r for r in plays if applies(r)]
        k = sum(complies(r) for r in sel)
        out.append(f"| {name} | {len(sel)} | {_pct(k, len(sel))} |")
        summary["rules"][name] = {"n": len(sel), "complied": k}

    # Timing (the owner, 2026-10-01): an opponent card whose event makes you discard is held for
    # your last action round, when the discard finds nothing -- or only a bad scoring card.
    out += ["", "## Timing: discard events held for the last action round", "",
            "Every play of the card by that side, in any mode, headlines included (a headline is never "
            "the last round). Last round = round 6 in turns 1–3, round 7 after (an eighth counts too).", "",
            "| rule | plays | in the last round | headlined | rounds played in |", "|:---|---:|---:|---:|:---|"]
    for name, cid, side in timing_rules():
        sel = [r for r in plays if r["card"] == cid and r["side"] == side]
        nh = sum(1 for h in heads if h["card"] == cid and h["side"] == side)
        k = sum(1 for r in sel if is_last_round(r["turn"], r["ar"]))
        n = len(sel) + nh
        ars = Counter(r["ar"] for r in sel)
        spread = ", ".join(f"AR{a} {ars[a]}" for a in sorted(ars))
        out.append(f"| {name} | {n} | {_pct(k, n)} | {_pct(nh, n)} | {spread} |")
        summary["rules"][name] = {"n": n, "complied": k}

    # Per card: own and neutral cards, the mode mix.
    per: Dict[Tuple[int, str], Counter] = defaultdict(Counter)
    for r in plays:
        if r["owner"] != "opp" and "event" in r["legal"]:
            per[(r["card"], r["side"])][r["mode"]] += 1
    for h in heads:                     # a headline is an event play
        c = info.get(h["card"])
        if c is not None and _owner(str(c["side"]), h["side"]) != "opp":
            per[(h["card"], h["side"])]["event"] += 1
    rows = []
    for (cid, side), cnt in per.items():
        n = sum(cnt.values())
        if n < 30:
            continue
        ops = cnt["influence"] + cnt["coup"] + cnt["realign"]
        rows.append((cnt["event"] / n, info[cid]["name"], side, n, cnt["event"], ops, cnt["space"]))
    rows.sort(key=lambda x: -x[0])
    head = ["| card | side | plays | event | ops | space |", "|:---|:---|---:|---:|---:|---:|"]
    fmt = lambda x: f"| {x[1]} | {x[2]} | {x[3]} | {_pct(x[4], x[3])} | {_pct(x[5], x[3])} | {_pct(x[6], x[3])} |"  # noqa: E731
    out += ["", "## Own and neutral cards: event against Ops", "",
            f"Cards played at least 30 times with the event legal ({len(rows)} card-side pairs). "
            "Headlines count as event plays.", "",
            "**Evented most:**", ""] + head + [fmt(x) for x in rows[:20]]
    out += ["", "**Played for Ops (or spaced) most:**", ""] + head + [fmt(x) for x in rows[::-1][:20]]
    summary["per_card"] = [{"card": x[1], "side": x[2], "n": x[3], "event": x[4], "ops": x[5], "space": x[6]} for x in rows]

    # Opponent cards: fire or space.
    opp: Dict[Tuple[int, str], Counter] = defaultdict(Counter)
    for r in plays:
        if r["owner"] == "opp":
            opp[(r["card"], r["side"])]["space" if r["mode"] == "space" else "fires"] += 1
    orows = sorted(((c["space"] / max(sum(c.values()), 1), info[cid]["name"], side, sum(c.values()), c["space"])
                    for (cid, side), c in opp.items() if sum(c.values()) >= 30), key=lambda x: -x[0])
    out += ["", "## Opponent cards: spaced (event avoided)", "",
            "| card | side holding it | plays | spaced |", "|:---|:---|---:|---:|"]
    out += [f"| {x[1]} | {x[2]} | {x[3]} | {_pct(x[4], x[3])} |" for x in orows[:20]]

    # Turn-1 headlines.
    for side, wanted in (("US", US_T1_HEADLINES), ("USSR", USSR_T1_HEADLINES)):
        t1 = [h for h in heads if h["turn"] == 1 and h["side"] == side]
        held: Counter = Counter()
        chosen: Counter = Counter()
        for h in t1:
            held.update(h["hand"])
            chosen[h["card"]] += 1
        by_name = {info[c]["name"]: c for c in info}
        out += ["", f"## Turn-1 {side} headlines ({len(t1)} games)", "",
                "The owner's candidates first, then the model's most-headlined cards.", "",
                "| card | held | headlined when held | share of all headlines |", "|:---|---:|---:|---:|"]
        listed = [by_name[n] for n in wanted if n in by_name]
        top = [c for c, _ in chosen.most_common(10) if c not in listed]
        for c in listed + top:
            mark = "" if c in listed else " (model)"
            out.append(f"| {info[c]['name']}{mark} | {held[c]} | {_pct(chosen[c], held[c])} | {_pct(chosen[c], len(t1))} |")
        summary[f"t1_headlines_{side}"] = {info[c]["name"]: {"held": held[c], "chosen": chosen[c]} for c in set(listed + top)}

    # Coups.
    ops_coups = [c for c in coups]
    nonbg2 = [c for c in ops_coups if not c["bg"] and c["stability"] <= 2]
    first = Counter(c["country"] for c in coups if c["ussr_t1_first"])
    out += ["", "## Coups", "",
            f"* {_pct(len(nonbg2), len(ops_coups))} of {len(ops_coups)} coup targets are non-battlegrounds of "
            "stability 1–2 (an inefficient use of Ops by the owner's rule; free coups from events included).",
            f"* The USSR's first coup of turn 1 ({sum(first.values())} games): "
            + ", ".join(f"{k} {_pct(v, sum(first.values()))}" for k, v in first.most_common(6)) + "."]

    # Nordics and the space race.
    for t in (2, 4, 8):
        bt = [b for b in boards if b["turn"] == t]
        if not bt:
            continue
        us_any = sum(any(x > 0 for x in b["nordics_us"]) for b in bt)
        ussr_any = sum(any(x > 0 for x in b["nordics_ussr"]) for b in bt)
        out.append(f"* Start of turn {t}: US influence in Norway/Sweden/Finland in {_pct(us_any, len(bt))} of games, "
                   f"USSR in {_pct(ussr_any, len(bt))}.")
    t4 = [b for b in boards if b["turn"] == 4]
    if t4:
        lead21 = sum(tuple(b["space"]) in ((2, 1), (1, 2)) for b in t4)
        out.append(f"* Entering the Mid War (start of turn 4) with a 2–1 space lead (recommended against, One Small "
                   f"Step): {_pct(lead21, len(t4))} of games.")
    return "\n".join(out) + "\n", summary
