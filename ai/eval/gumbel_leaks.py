"""Where does the raw network leak? Its move against a Gumbel root's, priced by paired playouts.

A noise-free Gumbel root at 32 simulations over the network's 4 most probable moves (first-play
urgency 0.2) scores about +60 Elo over the same network playing its argmax
(research/log/E7_gumbel_headroom.md, CI tournaments 37297751537 and 37318415364). That gain is
made of individual decisions where the root plays something else. This probe finds them and asks
how much each is worth:

* **Positions** come from the network's own games at the tournament temperature, one in eight of
  the decisions with a real choice (outside setup), drawn from whole games so every phase is
  represented.
* At each, the network's argmax and the Gumbel root's choice. Where they differ (a
  **departure**), both moves are played to the end of the game by the network from paired copies --
  the same redeal of the cards the mover cannot see and the same dice in both branches
  (`target_forms.paired_advantage`) -- and the paired difference is the departure's **advantage**,
  from the mover's side, in win probability (+1 = always turns a loss into a win).
* **Leak per decision** = departure rate x mean advantage: the win probability the network gives
  away at an average decision with a real choice by not playing the root's move there (the rest
  of the game played as before). Positions are a uniform sample of those decisions, so a group's
  share of the summed advantage is its share of the leak. Grouped by decision segment, side, turn,
  card and play-mode change, it says where the network leaks most; the largest single departures
  are listed with a workbench link to the position (`?pos=`), where the page's Gumbel search can
  be run on it. Per decision, not per game: the network makes several hundred decisions with a
  choice a game (every Influence point is one), and single-decision gains do not add up over a
  game.

What it cannot say: the playouts value a move as THIS network continues from it, so a move whose
worth lies in a follow-up the network does not find reads as worse than it is. A departure the
playouts refute (advantage below -2 SE) is the root being wrong, not the network.
"""
from __future__ import annotations

import base64
import json
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.target_forms import Position, paired_advantage
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig, decision_segment
from ai.search.gumbel_root import GumbelRoot
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder

#: One in eight decisions with a real choice, as the training searcher and target_forms sample.
SAMPLE_PROB = 0.125
#: The tournament temperature: positions are the ones the network reaches when it plays.
DEFAULT_TEMPERATURE = 0.1
#: ts::Resolution (engine/include/ts/types.hpp), the play modes of SELECT_PLAY_MODE.
MODE_NAMES = {0: "Event", 1: "Space", 2: "Influence", 3: "Coup", 4: "Realign"}
#: Where a leak's position opens: the local workbench server (`python -m web.server.main`).
DEFAULT_WORKBENCH = "http://localhost:8000/"


@dataclass(frozen=True)
class RootSpec:
    """The Gumbel root the network is compared against: `sims` simulations, `k` candidates,
    first-play urgency `fpu`, no Gumbel noise. Defaults: the cheapest root measured strong."""
    sims: int = 32
    k: int = 4
    fpu: float = 0.2

    @property
    def name(self) -> str:
        return f"gumbel{self.sims}-k{self.k}-fpu{self.fpu:g}"


# -- positions ---------------------------------------------------------------------------------

def collect_positions(model: torch.nn.Module, n: int, seed: int, games: int = 64,
                      temperature: float = DEFAULT_TEMPERATURE, sample_prob: float = SAMPLE_PROB,
                      max_steps: int = 3000, device: str = "cpu") -> Tuple[List[Position], float, int]:
    """`n` positions from the model's own games, the mean number of decisions with a real choice
    per finished game, and how many finished games that mean is over. Played in the model's own observation view (its feature set), so a
    model with appended feature blocks reads what it was trained on."""
    from bindings.ts_env import model_obs_features

    feats = int(model_obs_features(model))
    rng = np.random.default_rng(seed)
    pool: List[Position] = []
    choice_counts: List[int] = []
    base = seed * 1000
    while len(pool) < 2 * n:
        runner = ts.VectorizedBatchRunner(games, base)
        base += 1
        if feats:
            runner.set_obs_features([feats] * games, [feats] * games)
            runner.refresh_all()
        choices = np.zeros(games, dtype=np.int64)
        for _ in range(max_steps):
            terms = np.asarray(runner.get_terminals())
            if terms.all():
                break
            obs = torch.from_numpy(np.array(runner.get_observations(), copy=True)).to(device)
            masks = np.array(runner.get_action_masks(), copy=True)
            with torch.no_grad():
                lg, _v, _ = model(obs, torch.from_numpy(masks).bool().to(device))
            probs = torch.softmax(lg.float() / max(temperature, 1e-6), dim=-1).cpu().numpy()
            acts = []
            for i in range(games):
                legal = np.flatnonzero(masks[i])
                if terms[i] or len(legal) == 0:
                    acts.append(0)
                    continue
                if len(legal) > 1:
                    st = runner.get_state(i)
                    if st.current_phase != ts.Phase.SETUP:
                        choices[i] += 1
                        if rng.random() < sample_prob:
                            pool.append(Position(state=st.clone(), mover=int(acting_player(st)),
                                                 segment=decision_segment(st), turn=int(st.turn),
                                                 legal=[], prior={}, logits={}, value_mover=0.0))
                p = probs[i].astype(np.float64) * (masks[i] > 0)
                acts.append(int(rng.choice(len(p), p=p / p.sum())) if p.sum() > 0 else int(legal[0]))
            runner.step_flat_all(acts, True)
        done = np.asarray(runner.get_terminals())
        choice_counts += [int(c) for c, d in zip(choices, done) if d]
    out = [pool[int(i)] for i in sorted(rng.choice(len(pool), size=n, replace=False))]
    _fill_network(model, out, feats, device)
    per_game = float(np.mean(choice_counts)) if choice_counts else 0.0
    return out, per_game, len(choice_counts)


def _fill_network(model: torch.nn.Module, positions: Sequence[Position], feats: int,
                  device: str, chunk: int = 512) -> None:
    """The network's legal moves, prior, logits and value (mover's side) at each position."""
    for lo in range(0, len(positions), chunk):
        part = positions[lo:lo + chunk]
        obs = np.stack([np.asarray(ts.extract_observation_features(p.state, ts.Player(p.mover), feats),
                                   dtype=np.float32) for p in part])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(p.state), dtype=np.uint8) for p in part])
        with torch.no_grad():
            lg, v, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
        lg_np = lg.float().cpu().numpy()
        v_np = v.float().reshape(-1).cpu().numpy()
        for i, p in enumerate(part):
            legal = np.flatnonzero(masks[i])
            z = lg_np[i, legal] - lg_np[i, legal].max()
            pr = np.exp(z) / np.exp(z).sum()
            p.legal = [int(a) for a in legal]
            p.prior = {int(a): float(x) for a, x in zip(legal, pr)}
            p.logits = {int(a): float(x) for a, x in zip(legal, lg_np[i, legal])}
            p.value_mover = float(v_np[i])


# -- the root ----------------------------------------------------------------------------------

def gumbel_choices(model: torch.nn.Module, positions: Sequence[Position], spec: RootSpec,
                   seed: int, device: str = "cpu", chunk: int = 64
                   ) -> Tuple[List[int], List[Dict[str, Any]]]:
    """The Gumbel root's move at each position and its evidence (`GumbelRoot.choose` stats).
    Honest: each halving phase samples one world from the mover's side."""
    cfg = BatchedMCTSConfig(simulations=spec.sims, temperature=0.0, auto_advance=True,
                            advance_root=False, determinize=True, node_filter="all",
                            subsample=1.0, seed=seed, fpu_reduction=spec.fpu,
                            gumbel_k=spec.k, gumbel_scale=0.0)
    root = GumbelRoot(BatchedMCTS(model, device=torch.device(device), config=cfg))
    picks: List[int] = []
    stats: List[Dict[str, Any]] = []
    for lo in range(0, len(positions), chunk):
        picks += root.choose([p.state for p in positions[lo:lo + chunk]], stats)
    return [int(a) for a in picks], stats


# -- naming ------------------------------------------------------------------------------------

def card_name(cid: int) -> str:
    return str(ts.CardData.get_card_info(int(cid))["name"]) if 1 <= cid <= 110 else f"#{cid}"


def context_card(state: ts.GameState, action: int) -> int:
    """The card a decision is about: the one being resolved or played for Ops, else (choosing a
    card to play) the card the action plays; 0 if none."""
    c = state.ctx()
    for cid in (int(c.resolving_card), int(c.pending_op_card)):
        if cid:
            return cid
    ma = ts.decode_flat_action(state, int(action))
    if ma.decision_type == ts.DecisionType.SELECT_CARD:
        return int(ma.primary_id)
    return 0


def play_mode(state: ts.GameState, action: int) -> Optional[str]:
    """Event / Space / Influence / Coup / Realign for a play-mode action, else None."""
    ma = ts.decode_flat_action(state, int(action))
    if ma.decision_type not in (ts.DecisionType.SELECT_PLAY_MODE, ts.DecisionType.SELECT_OP_MODE):
        return None
    mode = int(ma.secondary_id) or int(ma.primary_id)
    return MODE_NAMES.get(mode, str(mode))


def position_token(state: ts.GameState) -> str:
    """The workbench's `pos=` token (web/ui/src/game/position.ts): the save JSON, zlib-deflated,
    base64url without padding."""
    raw = zlib.compress(state.to_save_json().encode("utf-8"), 9)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# -- rows --------------------------------------------------------------------------------------

def rows(positions: Sequence[Position], picks: Sequence[int], stats: Sequence[Dict[str, Any]],
         verdicts: Dict[int, Tuple[float, float]]) -> List[Dict[str, Any]]:
    """One JSON row per position. Departures carry both moves, the playout verdict, the root's
    own estimate of the gap and the position token."""
    out: List[Dict[str, Any]] = []
    for idx, (p, g, st) in enumerate(zip(positions, picks, stats)):
        net = max(p.prior, key=lambda a: p.prior[a])
        row: Dict[str, Any] = {"segment": p.segment, "turn": p.turn, "mover": p.mover,
                               "n_legal": len(p.legal), "net": net, "net_p": p.prior[net],
                               "depart": g != net}
        if g != net:
            cand = {int(c["action"]): c for c in st["candidates"]}
            q_g, q_n = cand.get(g, {}).get("q"), cand.get(net, {}).get("q")
            card = context_card(p.state, g)
            adv, se = verdicts[idx]
            row.update({
                "gumbel": g, "gumbel_p": p.prior.get(g, 0.0),
                "net_name": ActionEncoder.get_action_name(p.state, net),
                "gumbel_name": ActionEncoder.get_action_name(p.state, g),
                "net_mode": play_mode(p.state, net), "gumbel_mode": play_mode(p.state, g),
                "net_card": context_card(p.state, net), "card": card,
                "value_mover": p.value_mover,
                "search_gap": (q_g - q_n) if q_g is not None and q_n is not None else None,
                "adv": adv, "se": se, "token": position_token(p.state),
            })
        out.append(row)
    return out


def dump(rows_: Sequence[Dict[str, Any]], meta: Dict[str, Any], path: str) -> None:
    """A part file: a meta line, then one row per line."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for r in rows_:
            fh.write(json.dumps(r) + "\n")


def load(paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(rows, metas) of part files."""
    rows_: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if "meta" in obj:
                    metas.append(obj["meta"])
                else:
                    rows_.append(obj)
    return rows_, metas


# -- the report --------------------------------------------------------------------------------

def _group(rs: Sequence[Dict[str, Any]], n_all: int) -> Dict[str, float]:
    """Totals for a group of rows; its leak is its summed advantage over all `n_all` positions."""
    dep = [r for r in rs if r["depart"]]
    adv = np.array([r["adv"] for r in dep]) if dep else np.zeros(0)
    se = math.sqrt(sum(r["se"] ** 2 for r in dep)) if dep else 0.0
    return {
        "positions": float(len(rs)), "departs": float(len(dep)),
        "mean_adv": float(adv.mean()) if dep else 0.0,
        "mean_adv_se": se / len(dep) if dep else 0.0,
        "confirmed": float(sum(r["adv"] > 2 * r["se"] for r in dep)),
        "refuted": float(sum(r["adv"] < -2 * r["se"] for r in dep)),
        "leak": float(adv.sum()) / n_all if n_all else 0.0,
        "leak_se": se / n_all if n_all else 0.0,
    }


def _pct(x: float) -> str:
    return f"{100 * x:+.2f}"


def _per_mille(x: float) -> str:
    return f"{1000 * x:+.2f}"


def _leak_table(title: str, key: Callable[[Dict[str, Any]], Optional[str]],
                rows_: Sequence[Dict[str, Any]], n: int, total: float,
                top: int = 0, departures_only: bool = False) -> List[str]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows_:
        if departures_only and not r["depart"]:
            continue
        k = key(r)
        if k is not None:
            groups[k].append(r)
    stats = sorted(((k, _group(v, n)) for k, v in groups.items()), key=lambda kv: -kv[1]["leak"])
    if top:
        stats = stats[:top]
    out = ["", f"## {title}", "",
           "| | positions | departs | mean advantage (pp) | confirmed / refuted | **leak per decision (‰)** | share |",
           "|:---|---:|---:|---:|---:|---:|---:|"]
    for k, s in stats:
        share = s["leak"] / total if total > 0 else 0.0
        out.append(f"| {k} | {s['positions']:.0f} | {s['departs']:.0f} | "
                   f"{_pct(s['mean_adv'])} ± {100 * s['mean_adv_se']:.2f} | "
                   f"{s['confirmed']:.0f} / {s['refuted']:.0f} | "
                   f"**{_per_mille(s['leak'])}** ± {1000 * s['leak_se']:.2f} | {100 * share:.0f}% |")
    return out


def _decision(r: Dict[str, Any]) -> Optional[str]:
    """What a departure changes, as a group key: a card switched for another, a play mode
    changed, or a choice inside an event or an Ops play of a card."""
    if not r["depart"]:
        return None
    seg = r["segment"]
    if seg in ("headline", "ar_card"):
        return f"{seg}: {card_name(r['net_card'])} → {card_name(r['card'])}"
    if r.get("net_mode") or r.get("gumbel_mode"):
        return f"mode: {card_name(r['card'])} {r.get('net_mode')} → {r.get('gumbel_mode')}"
    if r["card"]:
        return f"{seg}: {card_name(r['card'])}"
    return seg


def _is_event(r: Dict[str, Any]) -> bool:
    """A departure about an event: playing (or not playing) a card for its event, or a choice
    while one resolves."""
    return r["depart"] and (r["segment"] == "event" or "Event" in (r.get("net_mode"), r.get("gumbel_mode")))


def _event_key(r: Dict[str, Any]) -> Optional[str]:
    if not _is_event(r):
        return None
    if r["segment"] == "event":
        return f"{card_name(r['card'])}: inside the event"
    if r.get("gumbel_mode") == "Event":
        return f"{card_name(r['card'])}: search plays the event (net: {r.get('net_mode')})"
    return f"{card_name(r['card'])}: search avoids the event (plays {r.get('gumbel_mode')})"


def _link(base: str, token: str, model_param: str) -> str:
    q = f"pos={token}" + (f"&model={model_param}" if model_param else "")
    return f"{base}?{q}"


def report(rows_: Sequence[Dict[str, Any]], metas: Sequence[Dict[str, Any]],
           workbench: str = DEFAULT_WORKBENCH, workbench_model: str = "", top: int = 40) -> str:
    n = len(rows_)
    games = sum(float(m.get("games_counted", 0)) for m in metas)
    per_game = (sum(float(m.get("choices_per_game", 0)) * float(m.get("games_counted", 0)) for m in metas)
                / games) if games else 0.0
    m0 = metas[0] if metas else {}
    s = _group(rows_, n)
    lines = [
        "# Where the network leaks: its move against a Gumbel root's, by paired playouts", "",
        f"Model `{m0.get('model', '?')}`; root **{m0.get('root', '?')}** (no Gumbel noise, honest: one "
        f"world sampled from the mover's side per halving phase); positions from the model's own games at "
        f"temperature {m0.get('temperature', '?')}, one in {round(1 / float(m0.get('sample_prob', SAMPLE_PROB)))} "
        f"decisions with a real choice (it makes {per_game:.0f} such decisions a game, every Influence point one); "
        f"{m0.get('pairs', '?')} paired playouts per departure, played greedily by the model.", "",
        "**Advantage** = the root's move minus the network's, from the mover's side, in percentage points of win "
        "probability (+100 = always turns a loss into a win). **Leak per decision** = departure rate x mean "
        "advantage, in ‰ of win probability: what the network gives away at an average decision by not playing "
        "the root's move there. A group's leak is its summed advantage over all positions, so the shares add up "
        "to 100%. ± is one standard error from the playouts.", "",
        "| positions | departs | mean advantage on departures | confirmed (> 2 SE) | refuted (< -2 SE) | **leak per decision** |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {n} | {s['departs']:.0f} ({100 * s['departs'] / max(1, n):.1f}%) | {_pct(s['mean_adv'])} ± "
        f"{100 * s['mean_adv_se']:.2f} pp | {s['confirmed']:.0f} | {s['refuted']:.0f} | "
        f"**{_per_mille(s['leak'])} ± {1000 * s['leak_se']:.2f}‰** |",
    ]
    total = s["leak"]
    lines += _leak_table("By decision segment", lambda r: r["segment"], rows_, n, total)
    lines += _leak_table("By turn", lambda r: f"turn {r['turn']:2d}", rows_, n, total)
    lines += _leak_table("By side", lambda r: "US" if r["mover"] == int(ts.Player.US) else "USSR",
                         rows_, n, total)
    lines += _leak_table("Event leaks: by card and direction", _event_key, rows_, n, total,
                         top=top, departures_only=True)
    lines += _leak_table("All leaks: by what the departure changes", _decision, rows_, n, total,
                         top=top, departures_only=True)
    lines += _leak_table("By card the decision is about",
                         lambda r: card_name(r["card"]) if r["depart"] and r["card"] else None,
                         rows_, n, total, top=top, departures_only=True)

    dep = sorted((r for r in rows_ if r["depart"]), key=lambda r: -r["adv"])
    lines += ["", f"## The {min(top, len(dep))} largest single leaks", "",
              "Each opens in the workbench at the position (Search there runs the page's Gumbel root on it).", "",
              "| # | turn | side | segment | network plays (p) | root plays (p) | advantage (pp) | root's own estimate (pp) | position |",
              "|---:|---:|:---|:---|:---|:---|---:|---:|:---|"]
    for i, r in enumerate(dep[:top], 1):
        side = "US" if r["mover"] == int(ts.Player.US) else "USSR"
        # The root's values are on the [-1, 1] utility scale: half a unit is 100 pp of win probability.
        gap = f"{50 * r['search_gap']:+.1f}" if r.get("search_gap") is not None else "—"
        lines.append(f"| {i} | {r['turn']} | {side} | {r['segment']} | {r['net_name']} ({r['net_p']:.2f}) | "
                     f"{r['gumbel_name']} ({r['gumbel_p']:.2f}) | {_pct(r['adv'])} ± {100 * r['se']:.1f} | {gap} | "
                     f"[open]({_link(workbench, r['token'], workbench_model)}) |")
    lines += ["", "## Where the root is wrong (refuted departures)", "",
              "Departures the playouts refute: the network's move was better. Many here mean the root's "
              "own estimates are noisy at this budget for that kind of decision.", "",
              "| segment | refuted | of departures |", "|:---|---:|---:|"]
    for seg in sorted({r["segment"] for r in rows_}):
        ds = [r for r in rows_ if r["depart"] and r["segment"] == seg]
        ref = sum(r["adv"] < -2 * r["se"] for r in ds)
        if ds:
            lines.append(f"| {seg} | {ref} | {len(ds)} |")
    return "\n".join(lines) + "\n"


# -- one part ----------------------------------------------------------------------------------

def run_part(model: torch.nn.Module, positions: int, seed: int, pairs: int, spec: RootSpec,
             temperature: float = DEFAULT_TEMPERATURE, games: int = 64,
             log: Callable[[str], None] = print) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample positions, find the root's departures, price them; (rows, meta)."""
    import time

    t0 = time.time()
    pos, per_game, n_games = collect_positions(model, positions, seed, games=games, temperature=temperature)
    log(f"[leaks] {len(pos)} positions, {per_game:.1f} decisions with a choice per game, {time.time() - t0:.0f}s")
    t1 = time.time()
    picks, stats = gumbel_choices(model, pos, spec, seed)
    deps = [i for i, (p, g) in enumerate(zip(pos, picks)) if g != max(p.prior, key=lambda a: p.prior[a])]
    log(f"[leaks] {spec.name}: {len(deps)} departures in {time.time() - t1:.0f}s")
    t2 = time.time()
    items = [(pos[i], picks[i], max(pos[i].prior, key=lambda a: pos[i].prior[a])) for i in deps]
    adv = paired_advantage(model, items, pairs, seed) if items else []
    log(f"[leaks] {len(items)} departures x {pairs} pairs in {time.time() - t2:.0f}s")
    verdicts = dict(zip(deps, adv))
    meta = {"root": spec.name, "positions": len(pos), "pairs": pairs, "seed": seed,
            "temperature": temperature, "sample_prob": SAMPLE_PROB,
            "choices_per_game": per_game, "games_counted": n_games}
    return rows(pos, picks, stats, verdicts), meta
