#!/usr/bin/env python3
"""The human plays behind an event-census gap, each with a workbench link and the bot's view.

`event_census_compare.py` says *that* humans and bots use a card differently; this lists *where*.
It reads a human holdings dump (`event_play_census.py --human-corpus --dump`, which keeps the
position of the decision that spent each holding) and, for one card, lists every human play of it:
when, by whom, how it was used, and -- given a model -- the probability the model puts on the
human's move at that very position, with its own favourite. Sorted by that probability, lowest
first, so the plays the model would least make come first. Kept holdings have no decision and are
not listed.

The decision is the one that settled the holding's use: the headline choice for a headline, the
card choice for a scoring card, the play-mode choice (event / Ops / space) for everything else.

    PYTHONPATH=.:build/release python tools/scripts/census_spots.py --human human.json \\
        --card "John Paul II" --checkpoint E7line_swa_4720-4800M.onnx --output-md jp2.md
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.scripts.event_play_census import Holding, load_holding, load_policy, state_from_token

WORKBENCH = "https://jamesyoul2.github.io/DeepStruggle/"
HOLDER = {"us": 1, "ussr": -1}


def find_card(query: str) -> int:
    cards = json.load(open("rules/cards.json"))
    hits = [c for c in cards if query.lower() in str(c["name"]).lower()]
    exact = [c for c in hits if str(c["name"]).lower() == query.lower()]
    if len(exact) == 1 or len(hits) == 1:
        return int((exact or hits)[0]["id"])
    raise SystemExit(f"--card {query!r} matches {[c['name'] for c in hits] or 'nothing'}")


def probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    z = np.where(m, logits, -np.inf)
    z = z - z[m].max()
    p = np.exp(z)
    return p / p.sum()


def spots(holdings: Sequence[Holding], card: int, holder: Optional[int], use: Optional[str],
          checkpoint: Optional[str], workbench: str) -> List[Dict]:
    sel = [h for h in holdings if h.card == card and h.pos and (holder is None or h.side == holder)
           and (use is None or h.outcome == use)]
    rows: List[Dict] = []
    states = [state_from_token(h.pos) for h in sel]
    p_h: List[Optional[float]] = [None] * len(sel)
    best: List[str] = [""] * len(sel)
    if checkpoint and sel:
        fn, features = load_policy(checkpoint)
        movers = [s.ctx().decision_player if s.ctx().decision_player != ts.Player.NONE else s.phasing_player
                  for s in states]
        obs = np.stack([np.asarray(ts.extract_observation_features(s, m, features), dtype=np.float32)
                        for s, m in zip(states, movers)])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        lg = fn(obs, masks)
        for i, (h, s) in enumerate(zip(sel, states)):
            p = probs(lg[i], masks[i])
            p_h[i] = float(p[h.action])
            b = int(np.argmax(p))
            best[i] = f"{ActionEncoder.get_action_name(s, b)} ({100 * p[b]:.0f}%)"
    for i, (h, s) in enumerate(zip(sel, states)):
        rows.append({"game": h.game, "turn": h.turn, "ar": h.ar, "side": "US" if h.side == 1 else "USSR",
                     "use": h.outcome, "human": ActionEncoder.get_action_name(s, h.action),
                     "p_human": p_h[i], "model": best[i], "defcon": int(s.defcon),
                     "vp": int(s.victory_points), "link": f"{workbench}?pos={h.pos}"})
    rows.sort(key=lambda r: (r["p_human"] if r["p_human"] is not None else 0.0, r["game"], r["turn"], r["ar"]))
    return rows


def render(rows: Sequence[Dict], name: str, model: Optional[str]) -> str:
    out = [f"# {name}: the human plays", "",
           f"{len(rows)} plays from the ts-replayer corpus"
           + (f"; the model is `{model}`, asked at each position (teacher-forced)." if model else "."), ""]
    if model and rows:
        by: Dict[str, List[float]] = {}
        for r in rows:
            by.setdefault(r["use"], []).append(float(r["p_human"]))
        out += ["| human's use | plays | model's mean probability of it | model agrees (p > 0.5) |",
                "|:---|---:|---:|---:|"]
        for use, ps in sorted(by.items()):
            out.append(f"| {use} | {len(ps)} | {100 * float(np.mean(ps)):.0f}% | {100 * float(np.mean([p > 0.5 for p in ps])):.0f}% |")
        out.append("")
    out += ["| replay | turn.AR | side | DEFCON | VP | human | model's p of it | model's favourite | |",
            "|---:|:---|:---|---:|---:|:---|---:|:---|:---|"]
    for r in rows:
        ar = "H" if r["ar"] == 0 else str(r["ar"])
        ph = "—" if r["p_human"] is None else f"{100 * r['p_human']:.0f}%"
        out.append(f"| {r['game']} | {r['turn']}.{ar} | {r['side']} | {r['defcon']} | {r['vp']:+d} | "
                   f"{r['human']} | {ph} | {r['model'] or '—'} | [position]({r['link']}) |")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", required=True, help="dump from event_play_census.py --human-corpus")
    ap.add_argument("--card", required=True, help="card name, or a unique part of it")
    ap.add_argument("--holder", choices=("us", "ussr"), default=None)
    ap.add_argument("--use", choices=("headline", "event", "ops", "space"), default=None)
    ap.add_argument("--checkpoint", default=None, help="a .pt or .onnx model to ask at each position")
    ap.add_argument("--workbench", default=WORKBENCH, help="the workbench URL the links open")
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    card = find_card(a.card)
    name = next(str(c["name"]) for c in json.load(open("rules/cards.json")) if int(c["id"]) == card)
    holdings = [load_holding(r) for r in json.load(open(a.human))["holdings"]]
    if not any(h.pos for h in holdings):
        raise SystemExit(f"{a.human} carries no positions: re-run event_play_census.py --human-corpus --dump")
    rows = spots(holdings, card, HOLDER.get(a.holder or ""), a.use, a.checkpoint, a.workbench)
    md = render(rows, name, a.checkpoint)
    print(md)
    if a.output_md:
        open(a.output_md, "w").write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
