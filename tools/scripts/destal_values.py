"""Turn 1 De-Stalinization: what each destination is worth, by paired playouts at one position.

Humans and the networks spend a turn-1 De-Stalinization differently -- humans put a point in Chile
88% of the time, the networks never (P ~ 0.001, below any search's candidate list) -- and too many
destinations are close for a searcher to rank. This prices them all at once: in each sampled
world, every package of the freed points over a set of countries (at most two per country, as the
event allows) is played to the end by one network on both sides, so packages are compared on the
same hands and the same dice.

* `position` -- the T1 De-Stalinization of a corpus game: the position at its first decision, the
  human's removals and placements. The converter parks every engine-dealt card that the log does
  not name in DISCARD_PILE (`_set_hand` / `_apply_hands`), so at turn 1 the pile holds cards
  nobody played; they are put back in the draw deck before anything is redealt, or a redeal would
  never give anyone those cards and the network would read them as gone.
* `run` -- this part's share of the worlds (`--part k/N`, for CI runners). A world redeals both
  hands (`--redeal both`; `mover` keeps the USSR's own) and fixes the dice; the human's removals
  are applied, then each package; the rest is `ai.eval.paired_playouts.play_safe` with greedy play.
  The network's own placement after the same removals is recorded and scored too.
* `pool` -- merge the parts.
* `report` -- per country, the value of the first and of the second point (an additive model with
  a world effect, standard errors clustered by world); the best packages, the human's, the
  network's own choice; and the same values split by who holds the scoring cards.

A value is what a point is worth to *this network's* play afterwards: a point whose payoff lies in
a follow-up the network does not find reads lower than it is.

    PYTHONPATH=.:build/release python tools/scripts/destal_values.py position \\
        --games data/datasets/ts_replayer/51.json.gz --out positions.jsonl
    PYTHONPATH=.:build/release python tools/scripts/destal_values.py run --input positions.jsonl \\
        --model swa.onnx --worlds 2000 --part 1/20 --out values-1.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/destal_values.py pool --parts values-*.jsonl.gz \\
        --expect 20 --out values.jsonl.gz
    PYTHONPATH=.:build/release python tools/scripts/destal_values.py report --input values.jsonl.gz \\
        --out report.md
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.paired_playouts import apply_move, ar_key, play_safe
from ai.search.dmcts import determinize
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import (feed_corpus_game, load_policy, position_token,
                                     read_corpus_game, state_from_token)
from tools.lib.jsonl import read_jsonl, sha256_file, write_jsonl
from tools.lib.ts_replayer_convert import card_id

DESTAL = 33
N_COUNTRIES = 84
STAGE_REMOVE = 0  # de_stalinization::STAGE_REMOVE
NODE = int(ActionEncoder.NODE_OFFSET)
#: the destinations priced by default: every country a human or a network commonly picks
COUNTRIES = ("Chile", "Venezuela", "Angola", "South Africa", "Algeria", "Indonesia", "Thailand",
             "Laos/Cambodia", "Libya", "Lebanon")
#: hand cards whose holder the report splits the values by
SPLIT_CARDS = (2, 3)  # Europe Scoring, Middle East Scoring
HOLDERS = ("ussr", "us", "deck")
_UINT64 = 1 << 64


def country_ids() -> Dict[str, int]:
    return {ts.MapData.get_country_name(c): c for c in range(N_COUNTRIES)}


def country_name(c: int) -> str:
    return ts.MapData.get_country_name(c)


def card_name(c: int) -> str:
    return ts.CardData.get_card_name(c)


# --------------------------------------------------------------------------- position


class DestalTrace:
    """The first De-Stalinization of a game: the state at its first decision, and every action
    taken while it resolves, with the stage it was taken in."""

    def __init__(self) -> None:
        self.state: Optional[ts.GameState] = None
        self.actions: List[Tuple[int, int]] = []  # (event_stage, action)
        self.done = False

    def observe(self, st: ts.GameState, a: int) -> None:
        if self.done:
            return
        if int(st.ctx().resolving_card) == DESTAL:
            if self.state is None:
                self.state = st.clone()
            self.actions.append((int(st.ctx().event_stage), int(a)))
        elif self.state is not None:
            self.done = True


def turn1_played(log: Dict[str, Any]) -> List[int]:
    """The cards the log says were played on turn 1 up to and including De-Stalinization."""
    out: List[int] = []
    for e in log["all_turns"]:
        if str(e.get("num")) != "1":
            break
        names = str(e.get("card") or "")
        out += [c for c in (card_id(x) for x in names.split(" & ")) if c]
        if "De-Stalinization" in names:
            break
    return out


def repair_turn1_deck(st: ts.GameState, log: Dict[str, Any]) -> List[int]:
    """Put back in the draw deck every discarded card that turn 1 cannot have discarded: one
    neither played (the log's card of each entry) nor in either logged turn-1 hand (an event can
    discard a held card). Returns the cards moved."""
    if int(st.turn) != 1:
        raise ValueError(f"the repair is for turn 1 only; this position is turn {int(st.turn)}")
    hands = log["hands"]["1"]
    allowed = set(turn1_played(log))
    for side in ("us", "ussr"):
        for nm in hands[side]:
            c = card_id(nm)
            if c is None:
                raise ValueError(f"unknown card in the logged hand: {nm!r}")
            allowed.add(c)
    moved = [c for c in range(1, 111)
             if st.get_card_location(c) == ts.CardLocation.DISCARD_PILE and c not in allowed]
    for c in moved:
        st.set_card_location(c, ts.CardLocation.DRAW_DECK)
    return moved


def position(games: Sequence[str], out: str) -> int:
    rows: List[Dict[str, Any]] = []
    for path in games:
        t = DestalTrace()
        status = feed_corpus_game(path, t)
        if t.state is None:
            print(f"{path}: no De-Stalinization ({status})", file=sys.stderr)
            continue
        st = t.state
        if int(st.turn) != 1:
            print(f"{path}: De-Stalinization on turn {int(st.turn)}, not 1 -- skipped", file=sys.stderr)
            continue
        moved = repair_turn1_deck(st, read_corpus_game(path))
        removal = [a for stage, a in t.actions if stage == STAGE_REMOVE]
        placement = [a for stage, a in t.actions if stage != STAGE_REMOVE]
        gid = re.sub(r"\.json(\.gz)?$", "", os.path.basename(path))
        rows.append({
            "schema": 1, "id": f"r{gid}", "game": os.path.basename(path), "status": status,
            "turn": int(st.turn), "ar": int(st.action_round), "phase": str(st.current_phase),
            "defcon": int(st.defcon), "vp": int(st.victory_points),
            "pos": position_token(st), "removal": removal, "human": placement,
            "removal_names": [country_name(a - NODE) for a in removal if a >= NODE],
            "human_names": [country_name(a - NODE) for a in placement],
            "repaired": moved, "repaired_names": [card_name(c) for c in moved],
        })
        print(f"{path}: T1 AR{int(st.action_round)}, removed {rows[-1]['removal_names']}, placed "
              f"{rows[-1]['human_names']}; {len(moved)} discards put back in the deck", file=sys.stderr)
    write_jsonl(out, rows)
    return 0


# --------------------------------------------------------------------------- run


def packages(countries: Sequence[int], points: int) -> List[Tuple[int, ...]]:
    """Every way to place `points` over `countries`, at most two in one country, as a sorted
    tuple of country ids (a country twice for two points)."""
    out = [p for p in itertools.combinations_with_replacement(sorted(countries), points)
           if all(p.count(c) <= 2 for c in set(p))]
    return sorted(out)


def package_name(p: Sequence[int]) -> str:
    return " ".join(f"{country_name(c)}{'x2' if p.count(c) == 2 else ''}"
                    for c in sorted(set(p), key=lambda c: (-p.count(c), country_name(c))))


def world_seed(pid: str, seed: int, world: int) -> int:
    h = hashlib.sha256(f"{pid}:{seed}:{world}".encode()).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def world_start(st: ts.GameState, wseed: int, redeal: str) -> ts.GameState:
    """A world: the US hand and the deck redealt from the USSR's view, then (`both`) the USSR hand
    and the deck from the US's -- together a uniform deal of every card nobody has seen -- and its
    own dice."""
    rng = random.Random(wseed)
    w = determinize(st, ts.Player.USSR, rng)
    if redeal == "both":
        w = determinize(w, ts.Player.US, rng)
    w.rng_state = rng.getrandbits(64) % _UINT64
    return w


def hands_of(st: ts.GameState) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {"ussr": [], "us": []}
    for c in range(1, 111):
        h = ts.hand_holder(st.get_card_location(c))
        if h == ts.Player.USSR:
            out["ussr"].append(c)
        elif h == ts.Player.US:
            out["us"].append(c)
    return out


def apply_all(st: ts.GameState, actions: Sequence[int]) -> ts.GameState:
    for a in actions:
        st = apply_move(st, int(a))
    return st


def placements(pkg: Sequence[int]) -> List[int]:
    return [NODE + c for c in pkg]


def net_placement(starts: Sequence[ts.GameState], logits_fn: Any, feats: int) -> List[Tuple[int, ...]]:
    """The network's own greedy placement in each post-removal state."""
    sts = [s.clone() for s in starts]
    picked: List[List[int]] = [[] for _ in sts]
    for _ in range(8):
        live = [i for i, s in enumerate(sts)
                if not ts.Engine.is_terminal(s) and int(s.ctx().resolving_card) == DESTAL]
        if not live:
            break
        obs = np.stack([ts.extract_observation_features(sts[i], ts.Player.USSR, feats) for i in live])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(sts[i])) for i in live])
        acts = np.asarray(logits_fn(obs, masks)).argmax(axis=1)
        for i, a in zip(live, acts):
            picked[i].append(int(a) - NODE)
            sts[i] = apply_move(sts[i], int(a))
    return [tuple(sorted(p)) for p in picked]


def run(inputs: Sequence[str], model: str, worlds: int, part: str, seed: int, redeal: str,
        names: Sequence[str], chunk: int, out: str) -> int:
    k, n = (int(x) for x in part.split("/"))
    ids = country_ids()
    unknown = [nm for nm in names if nm not in ids]
    if unknown:
        raise ValueError(f"unknown countries: {unknown}")
    countries = [ids[nm] for nm in names]
    logits_fn, feats = load_policy(model)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    rows = list(read_jsonl(inputs))
    jobs = [(r, w) for r in rows for w in range(worlds)]
    mine = [j for i, j in enumerate(jobs) if i % n == k - 1]
    t0 = time.time()
    res: List[Dict[str, Any]] = []
    meta_pos: Dict[str, Any] = {}
    plans: Dict[str, Tuple[ts.GameState, List[Tuple[int, ...]]]] = {}
    for r in rows:
        st = state_from_token(r["pos"])
        pk = packages(countries, len(r["human"]))
        human = tuple(sorted(a - NODE for a in r["human"]))
        if human not in pk:
            pk.append(human)
        plans[r["id"]] = (st, pk)
        meta_pos[r["id"]] = {"packages": [list(p) for p in pk], "human": list(human),
                             "removal": r["removal"]}
    done = 0
    per_world = max(len(pk) for _, pk in plans.values()) + 1
    step = max(1, chunk // per_world)
    for lo in range(0, len(mine), step):
        batch = mine[lo:lo + step]
        starts: List[ts.GameState] = []
        index: List[Tuple[int, int]] = []  # (row in batch, package index; -1 = the network's own)
        info: List[Dict[str, Any]] = []
        removed: List[ts.GameState] = []
        for b, (r, w) in enumerate(batch):
            st, pk = plans[r["id"]]
            ws = world_seed(r["id"], seed, w)
            base = world_start(st, ws, redeal)
            removed.append(apply_all(base, r["removal"]))
            info.append({"pos": r["id"], "world": w, "seed": ws, "hands": hands_of(base)})
        own = net_placement(removed, logits_fn, feats)
        for b, (r, w) in enumerate(batch):
            _, pk = plans[r["id"]]
            for j, p in enumerate(pk):
                starts.append(apply_all(removed[b], placements(p)))
                index.append((b, j))
            info[b]["net"] = list(own[b])
            if own[b] in pk:
                info[b]["net_index"] = pk.index(own[b])
            else:
                starts.append(apply_all(removed[b], placements(own[b])))
                index.append((b, -1))
        movers = [ts.Player.USSR] * len(starts)
        keys = [ar_key(plans[batch[b][0]["id"]][0]) for b, _ in index]
        scores = play_safe(starts, movers, keys, act, info[0]["seed"] % (1 << 31))
        half = [[0] * len(plans[r["id"]][1]) for r, _ in batch]
        for (b, j), s in zip(index, scores):
            if j < 0:
                info[b]["net_s"] = int(round(2 * s))
            else:
                half[b][j] = int(round(2 * s))
        for b in range(len(batch)):
            info[b]["s"] = "".join(str(x) for x in half[b])
            if "net_index" in info[b]:
                info[b]["net_s"] = half[b][info[b]["net_index"]]
            res.append(info[b])
        done += len(batch)
        print(f"[{done}/{len(mine)} worlds] {len(starts)} games in this batch, "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    write_jsonl(out, res)
    json.dump({"schema": 1, "stage": "run", "model": os.path.basename(model),
               "model_sha256": sha256_file(model), "worlds": worlds, "part": part, "seed": seed,
               "redeal": redeal, "countries": list(names), "positions": meta_pos,
               "rows": len(res), "wall_s": round(time.time() - t0, 1)},
              open(out + ".meta.json", "w"), indent=1)
    print(f"{len(res)} worlds in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def pool(parts: Sequence[str], out: str, expect: int = 0) -> int:
    """Merge the parts (`values-<k>.jsonl.gz`, each with its `.meta.json`); with `expect`, name
    the parts that are missing in `<out>.parts.json`."""
    write_jsonl(out, list(read_jsonl(parts)))
    metas = [json.load(open(p + ".meta.json")) for p in parts if os.path.exists(p + ".meta.json")]
    if metas:
        m = dict(metas[0])
        m["parts"] = sorted(x["part"] for x in metas)
        m["rows"] = sum(int(x["rows"]) for x in metas)
        m.pop("wall_s", None)
        if any(x["positions"] != m["positions"] or x["model_sha256"] != m["model_sha256"] for x in metas):
            raise ValueError("the parts disagree on the model or the packages")
        json.dump(m, open(out + ".meta.json", "w"), indent=1)
    found = sorted(int(g.group(1)) for g in (re.search(r"values-(\d+)\.jsonl\.gz$", os.path.basename(p))
                                             for p in parts) if g)
    missing = [k for k in range(1, expect + 1) if k not in found]
    with open(out + ".parts.json", "w") as f:
        json.dump({"expected": expect, "found": found, "missing": missing}, f)
    if missing:
        print(f"parts missing: {missing} -- their worlds are not in {out}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- report


def design(pk: Sequence[Sequence[int]], countries: Sequence[int]) -> np.ndarray:
    """One row per package: [first point in country c] for each c, then [second point in c]."""
    x = np.zeros((len(pk), 2 * len(countries)))
    for i, p in enumerate(pk):
        for j, c in enumerate(countries):
            m = list(p).count(c)
            x[i, j] = 1.0 if m >= 1 else 0.0
            x[i, len(countries) + j] = 1.0 if m >= 2 else 0.0
    return x


def fit(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least squares of y (worlds x packages) on the package design with a world effect, the
    standard errors clustered by world. Every package places the same number of points, so the
    coefficients are known only up to one constant; they are shifted so the first points average
    0 -- a value reads as "against the average first point"."""
    xc = x - x.mean(axis=0)
    pinv = np.linalg.pinv(xc, rcond=1e-10)
    yc = y - y.mean(axis=1, keepdims=True)
    per_world = yc @ pinv.T  # (worlds, coefficients): each world's own fit, the same operator
    beta = per_world.mean(axis=0)
    se = per_world.std(axis=0, ddof=1) / math.sqrt(len(y))
    c = len(beta) // 2
    shift = beta[:c].mean()
    return beta - shift, se


def pct(v: float) -> str:
    return f"{100 * v:+.1f}"


def report(inputs: Sequence[str], out: str, top: int = 25) -> int:
    rows = list(read_jsonl(inputs))
    meta = json.load(open(inputs[0] + ".meta.json"))
    names = meta["countries"]
    ids = country_ids()
    countries = [ids[nm] for nm in names]
    lines: List[str] = [
        "# Turn 1 De-Stalinization: what each destination is worth",
        "",
        f"Model `{meta['model']}` (sha256 `{meta['model_sha256'][:12]}`), greedy on both sides after the "
        f"event; redeal `{meta['redeal']}`, seed {meta['seed']}. USSR score: win 1, draw 0.5. Every "
        "package is played in every world, so all comparisons are paired.",
        "",
    ]
    for pid, pm in meta["positions"].items():
        rs = [r for r in rows if r["pos"] == pid]
        if not rs:
            continue
        pk = [tuple(p) for p in pm["packages"]]
        y = np.array([[int(ch) / 2 for ch in r["s"]] for r in rs])
        w = len(rs)
        x = design(pk, countries)
        beta, se = fit(y, x)
        c = len(countries)
        mean = y.mean(axis=0)
        best = int(np.argmax(mean))
        human = pk.index(tuple(pm["human"]))
        lines += [f"## {pid}: {w} worlds, {len(pk)} packages, {w * len(pk):,} games", "",
                  f"Removed: {', '.join(country_name(a - NODE) for a in pm['removal'] if a >= NODE)}. "
                  f"Human placed: {package_name(pm['human'])}. Mean USSR score over all packages "
                  f"{100 * y.mean():.1f}%.", ""]
        # the additive model on the package means, against their sampling noise around the average
        # package (the coefficients' common constant cancels: every package has the same points)
        pred = mean.mean() + (x - x.mean(axis=0)) @ beta
        resid = mean - pred
        noise = float(np.sqrt(np.mean(((y - y.mean(axis=1, keepdims=True)).std(axis=0, ddof=1)) ** 2) / w))
        se_pk = (y - y[:, [best]]).std(axis=0, ddof=1) / math.sqrt(w)
        lines += ["### Value of a point, by country", "",
                  "Percentage points of USSR score against the average first point (the additive model: "
                  "a package's value is the sum of its points' values). ± one standard error, "
                  "clustered by world.", "",
                  "| country | first point | second point | both points |", "|:---|---:|---:|---:|"]
        order = sorted(range(c), key=lambda j: -beta[j])
        for j in order:
            both = beta[j] + beta[c + j]
            lines.append(f"| {names[j]} | {pct(beta[j])} ± {100 * se[j]:.2f} | "
                         f"{pct(beta[c + j])} ± {100 * se[c + j]:.2f} | {pct(both)} |")
        lines += ["", f"Additive model on the {len(pk)} package means: residual SD "
                  f"{100 * float(resid.std()):.2f} pts against a sampling SE of about {100 * noise:.2f}; "
                  f"largest residuals {', '.join(f'{package_name(pk[i])} {pct(resid[i])}' for i in np.argsort(-np.abs(resid))[:4])}.", ""]
        # packages
        lines += ["### Packages", "",
                  f"Best {top} by mean, then the human's and the network's. Δ best and Δ human: paired "
                  "difference ± SE.", "",
                  "| rank | package | USSR score | Δ best | Δ human | additive model |",
                  "|---:|:---|---:|---:|---:|---:|"]
        rank = np.argsort(-mean)
        pos_of = {int(i): r + 1 for r, i in enumerate(rank)}
        d_h = y - y[:, [human]]
        se_h = d_h.std(axis=0, ddof=1) / math.sqrt(w)
        shown = [int(i) for i in rank[:top]]
        net_counts: Dict[Tuple[int, ...], int] = {}
        for r in rs:
            net_counts[tuple(r["net"])] = net_counts.get(tuple(r["net"]), 0) + 1
        extra = [human] + [pk.index(p) for p, _ in sorted(net_counts.items(), key=lambda kv: -kv[1])[:3]
                           if p in pk]
        for i in shown + [e for e in extra if e not in shown]:
            tag = " (human)" if i == human else ""
            if pk[i] in net_counts:
                tag += f" (network {100 * net_counts[pk[i]] / w:.0f}%)"
            lines.append(f"| {pos_of[i]} | {package_name(pk[i])}{tag} | {100 * mean[i]:.2f} | "
                         f"{pct(mean[i] - mean[best])} ± {100 * se_pk[i]:.2f} | "
                         f"{pct(mean[i] - mean[human])} ± {100 * se_h[i]:.2f} | {100 * pred[i]:.2f} |")
        # the network's own choice, world by world
        net_s = np.array([r["net_s"] / 2 for r in rs])
        d_nb, d_nh = net_s - y[:, best], net_s - y[:, human]
        lines += ["", "### The network's own placement", "",
                  f"After the same removals, the network's greedy placement in each world (its hand is "
                  f"the world's): mean {100 * net_s.mean():.2f}%, against the best package "
                  f"{pct(d_nb.mean())} ± {100 * d_nb.std(ddof=1) / math.sqrt(w):.2f}, against the human's "
                  f"{pct(d_nh.mean())} ± {100 * d_nh.std(ddof=1) / math.sqrt(w):.2f}.", "",
                  "| package | share of worlds | rank of that package |", "|:---|---:|---:|"]
        for p, cnt in sorted(net_counts.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"| {package_name(p)} | {100 * cnt / w:.1f}% | "
                         f"{pos_of[pk.index(p)] if p in pk else 'outside the set'} |")
        # split by holder
        lines += ["", "### First-point values by who holds a scoring card", "",
                  "The same model fitted on the worlds where the card is in that hand (or in the deck).", ""]
        for card in SPLIT_CARDS:
            groups: Dict[str, List[int]] = {h: [] for h in HOLDERS}
            for i, r in enumerate(rs):
                h = "ussr" if card in r["hands"]["ussr"] else ("us" if card in r["hands"]["us"] else "deck")
                groups[h].append(i)
            head = [f"{h} ({len(groups[h])})" for h in HOLDERS]
            lines += [f"**{card_name(card)}**", "", "| country | " + " | ".join(head) + " |",
                      "|:---|" + "---:|" * len(HOLDERS)]
            fits = {h: fit(y[g], x) for h, g in groups.items() if len(g) >= 30}
            for j in order:
                cells = [f"{pct(fits[h][0][j])} ± {100 * fits[h][1][j]:.2f}" if h in fits else "--"
                         for h in HOLDERS]
                lines.append(f"| {names[j]} | " + " | ".join(cells) + " |")
            lines.append("")
        # card x country interactions on the USSR's own hand
        cards = sorted({cc for r in rs for cc in r["hands"]["ussr"] + r["hands"]["us"]})
        tests: List[Tuple[float, str, float, float]] = []
        for card in cards:
            held = [i for i, r in enumerate(rs) if card in r["hands"]["ussr"]]
            rest = [i for i, r in enumerate(rs) if card not in r["hands"]["ussr"]]
            if len(held) < 20 or len(rest) < 20:
                continue
            bh, sh = fit(y[held], x)
            br, sr = fit(y[rest], x)
            for j in range(c):
                d = bh[j] - br[j]
                s = math.hypot(sh[j], sr[j])
                if s > 0:
                    tests.append((abs(d) / s, f"{card_name(card)} in the USSR hand -> {names[j]}", d, s))
        tests.sort(key=lambda t: -t[0])
        z_bonf = _z_two_sided(0.05 / max(1, len(tests)))
        lines += ["### Which USSR hand cards move a country's first point", "",
                  f"Difference in a first point's value between worlds where the USSR holds the card and "
                  f"where it does not; {len(tests)} tests, Bonferroni at 5% needs |z| > {z_bonf:.2f}.", "",
                  "| card -> country | Δ value | z |", "|:---|---:|---:|"]
        for z, label, d, s in tests[:15]:
            lines.append(f"| {label} | {pct(d)} ± {100 * s:.2f} | {z:.1f}{' *' if z > z_bonf else ''} |")
        lines.append("")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _z_two_sided(alpha: float) -> float:
    """The two-sided normal quantile for `alpha`, by bisection on erfc."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("position")
    p.add_argument("--games", nargs="+", required=True, help="corpus games (.json.gz)")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", nargs="+", required=True)
    r.add_argument("--model", required=True, help="a .pt checkpoint or its .onnx export")
    r.add_argument("--worlds", type=int, default=2000, help="worlds per position")
    r.add_argument("--part", default="1/1")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--redeal", choices=("both", "mover"), default="both")
    r.add_argument("--countries", nargs="+", default=list(COUNTRIES))
    r.add_argument("--chunk", type=int, default=1536, help="games per playout batch")
    r.add_argument("--out", required=True)
    q = sub.add_parser("pool")
    q.add_argument("--parts", nargs="+", required=True)
    q.add_argument("--expect", type=int, default=0)
    q.add_argument("--out", required=True)
    s = sub.add_parser("report")
    s.add_argument("--input", nargs="+", required=True)
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "position":
        return position(a.games, a.out)
    if a.cmd == "run":
        return run(a.input, a.model, a.worlds, a.part, a.seed, a.redeal, a.countries, a.chunk, a.out)
    if a.cmd == "pool":
        return pool(a.parts, a.out, a.expect)
    return report(a.input, a.out, a.top)


if __name__ == "__main__":
    sys.exit(main())
