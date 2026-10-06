"""Step 0 of the macro-search experiment: would a beam of the network's complete decisions contain
the move a strong micro-search plays, and where does each searcher's choice sit in the network's
own ranking?

At macro starts sampled from the network's greedy self-play, for each position:

* the beam's candidates (`MacroGenerator`, k = 8 and k = 16), and the network's greedy macro;
* the macro a micro-action searcher plays from there (by default the strongest measured: Gumbel
  k = 8 at 256 simulations, honest), stepped decision by decision on the real state to the end of
  the macro -- its result is matched against the candidates, and its policy probability computed;
* the macro each macro-search setting chooses.

The cost of each method is counted in network rows evaluated and seconds, so strength can later be
read against compute. If the micro-searcher's macro is nearly always among the top few candidates,
a macro search can only reorder what the beam already holds; if it is often missing, the beam's
recall is the first thing to fix.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.batched_mcts import settle
from ai.search.macro_search import (MAX_MACRO_STEPS, MacroSearch, MacroSearchConfig, _Line, _Net,
                                    _result_key, _stage, greedy_macro, macro_boundary)
from ai.search.pimcts import acting_player


def count_rows(model: torch.nn.Module) -> Callable[[], int]:
    """Count every row the model evaluates from now on (wraps the instance's forward)."""
    box = [0]
    orig = model.forward

    def forward(obs: torch.Tensor, *a: Any, **kw: Any) -> Any:
        box[0] += int(obs.shape[0])
        return orig(obs, *a, **kw)

    model.forward = forward  # type: ignore[method-assign]
    return lambda: box[0]


def sample_positions(net: _Net, games: int, seed: int, rate: float,
                     auto_advance: bool = True) -> List[ts.GameState]:
    """Macro starts (outside setup, with a real choice) from the network's greedy self-play, each
    kept with probability `rate`. The games are played in lockstep, one batch per decision."""
    rng = random.Random(seed)
    states: List[ts.GameState] = []
    for g in range(games):
        s = ts.GameState()
        ts.Engine.init_game(s, seed * 1_000_003 + g)
        settle(s, auto_advance)
        states.append(s)
    start = [True] * games
    out: List[ts.GameState] = []
    for _ in range(100_000):
        live = [i for i, s in enumerate(states) if not ts.Engine.is_terminal(s)]
        if not live:
            break
        obs, masks = net.featurise([states[i] for i in live])
        logp, _ = net.forward(obs, masks)
        for row, i in enumerate(live):
            s = states[i]
            if (start[i] and s.current_phase != ts.Phase.SETUP and int(masks[row].sum()) > 1
                    and rng.random() < rate):
                out.append(s.clone())
            mover, stage, r0 = int(acting_player(s)), _stage(s), int(s.rng_state)
            ts.Engine.step_flat(s, int(np.argmax(logp[row])))
            settle(s, auto_advance)
            start[i] = macro_boundary(stage, r0, mover, s)[0]
    return out


def searcher_macros(agent: Any, net: _Net, positions: Sequence[ts.GameState],
                    auto_advance: bool = True) -> List[Tuple[List[int], bytes, float]]:
    """Per position: the macro `agent` plays from it (actions, result key, the network's log
    probability of that line), every position stepped in lockstep on a clone of the real state."""
    cur = [p.clone() for p in positions]
    lines = [_Line(root=i, state=c, mover=int(acting_player(c))) for i, c in enumerate(cur)]
    for _ in range(MAX_MACRO_STEPS):
        live = [ln for ln in lines if not ln.done]
        if not live:
            break
        obs, masks = net.featurise([ln.state for ln in live])
        logp, _ = net.forward(obs, masks)
        picks = agent.select_actions_batch([ln.state for ln in live])
        for row, (ln, a) in enumerate(zip(live, picks)):
            a = int(a)
            ln.logp += float(logp[row][a])
            stage, r0 = _stage(ln.state), int(ln.state.rng_state)
            ts.Engine.step_flat(ln.state, a)
            settle(ln.state, auto_advance)
            ln.actions.append(a)
            ln.done, ln.drew = macro_boundary(stage, r0, ln.mover, ln.state)
    out: List[Tuple[List[int], bytes, float]] = []
    for ln in lines:
        after = np.asarray(ts.extract_observation_features(ln.state, ts.Player(ln.mover), net.features),
                           dtype=np.float32)
        out.append((ln.actions, _result_key(ln, after), ln.logp))
    return out


def probe(model: torch.nn.Module, positions: Sequence[ts.GameState], micro_specs: Dict[str, Any],
          macro_settings: Dict[str, MacroSearchConfig], ks: Sequence[int] = (8, 16)) -> Dict[str, Any]:
    """The per-position records and per-method costs (see the module docstring)."""
    rows = count_rows(model)
    dev = torch.device("cpu")
    net = _Net(model, dev)
    base = MacroSearch(model, dev, MacroSearchConfig())
    recs: List[Dict[str, Any]] = [{"turn": int(p.turn), "ar": int(p.action_round),
                                   "phase": int(p.current_phase),
                                   "side": "US" if int(acting_player(p)) == int(ts.Player.US) else "USSR"}
                                  for p in positions]
    cost: Dict[str, Dict[str, float]] = {}

    # The beam at each k, and the greedy macro.
    cands: Dict[int, List[Any]] = {}
    for k in ks:
        r0, t0 = rows(), time.time()
        cands[k] = base.gen.candidates(positions, k)
        cost[f"beam{k}"] = {"rows": rows() - r0, "seconds": time.time() - t0}
    kmax = max(ks)
    for i, p in enumerate(positions):
        g_actions, g_key = greedy_macro(net, p)
        cs = cands[kmax][i]
        recs[i]["n_candidates"] = {str(k): len(cands[k][i]) for k in ks}
        recs[i]["greedy_len"] = len(g_actions)
        recs[i]["greedy_rank"] = next((m.rank for m in cs if m.result == g_key), -1)
        recs[i]["top_p"] = math.exp(cs[0].logp) if cs else 0.0
        recs[i]["greedy_key"] = g_key.hex()

    def locate(key: bytes, i: int) -> Dict[str, Any]:
        hit: Dict[str, Any] = {}
        for k in ks:
            m = next((m for m in cands[k][i] if m.result == key), None)
            hit[str(k)] = m.rank if m is not None else -1
        return hit

    # Micro-action searchers, stepped to the end of the macro.
    for name, agent in micro_specs.items():
        r0, t0 = rows(), time.time()
        got = searcher_macros(agent, net, positions)
        cost[name] = {"rows": rows() - r0, "seconds": time.time() - t0}
        for i, (acts, key, lp) in enumerate(got):
            recs[i][name] = {"len": len(acts), "logp": lp, "greedy": key.hex() == recs[i]["greedy_key"],
                             "in_beam": locate(key, i), "key": key.hex()}

    # Macro searches.
    for name, cfg in macro_settings.items():
        ms = MacroSearch(model, dev, cfg)
        r0, t0 = rows(), time.time()
        picks = ms.choose(positions)
        cost[name] = {"rows": rows() - r0, "seconds": time.time() - t0}
        for i, m in enumerate(picks):
            if m is None:
                recs[i][name] = None
                continue
            recs[i][name] = {"rank": m.rank, "logp": m.logp, "greedy": m.greedy,
                             "key": m.result.hex()}
        recs_misses = ms.world_misses
        cost[name]["world_misses"] = recs_misses
    for r in recs:
        r.pop("greedy_key", None)
    return {"positions": recs, "cost": cost, "n": len(positions)}


def _pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:.0f}%" if b else "—"


def report(parts: Sequence[Dict[str, Any]], micro: Sequence[str], macro: Sequence[str],
           ks: Sequence[int] = (8, 16)) -> str:
    recs = [r for p in parts for r in p["positions"]]
    n = len(recs)
    cost: Dict[str, Dict[str, float]] = {}
    for p in parts:
        for name, c in p["cost"].items():
            d = cost.setdefault(name, {})
            for key, v in c.items():
                d[key] = d.get(key, 0.0) + float(v)
    out = [f"# Macro candidates: recall and rank ({n} positions)", "",
           "Macro starts (outside setup, with a real choice) sampled from the network's greedy "
           "self-play. A candidate's rank is its place in the network's own ranking of complete "
           "decisions (0 = most probable).", ""]
    kmax = max(ks)
    gr = [r["greedy_rank"] for r in recs]
    out += ["## The beam", "",
            "| | value |", "|:---|---:|",
            *[f"| candidates per position, k = {k} (mean) | {np.mean([r['n_candidates'][str(k)] for r in recs]):.1f} |" for k in ks],
            f"| top candidate's probability (median) | {np.median([r['top_p'] for r in recs]):.3f} |",
            f"| network's greedy macro is the beam's rank 0 | {_pct(sum(g == 0 for g in gr), n)} |",
            f"| greedy macro missing from the k = {kmax} beam | {_pct(sum(g < 0 for g in gr), n)} |",
            f"| greedy macro length (mean decisions) | {np.mean([r['greedy_len'] for r in recs]):.2f} |", ""]
    out += ["## The micro-searchers' macros", "",
            "| searcher | = greedy macro | departs | departures in top-8 beam | in top-16 beam | missing from top-16 | departure's network prob. (median) |",
            "|:---|---:|---:|---:|---:|---:|---:|"]
    for name in micro:
        rs = [r[name] for r in recs if r.get(name)]
        dep = [x for x in rs if not x["greedy"]]
        in8 = sum(x["in_beam"].get("8", -1) >= 0 for x in dep)
        in16 = sum(x["in_beam"].get(str(kmax), -1) >= 0 for x in dep)
        med = np.median([math.exp(x["logp"]) for x in dep]) if dep else float("nan")
        out.append(f"| {name} | {_pct(len(rs) - len(dep), len(rs))} | {len(dep)} | {_pct(in8, len(dep))} | "
                   f"{_pct(in16, len(dep))} | {_pct(len(dep) - in16, len(dep))} | {med:.3f} |")
    out += ["", "Rank of a departure in the network's ranking (where the beam holds it):", ""]
    for name in micro:
        dep = [r[name] for r in recs if r.get(name) and not r[name]["greedy"]]
        ranks = [x["in_beam"].get(str(kmax), -1) for x in dep if x["in_beam"].get(str(kmax), -1) >= 0]
        if ranks:
            hist = np.bincount(np.minimum(ranks, 16), minlength=17)
            out.append(f"* {name}: " + ", ".join(f"rank {j}{'+' if j == 16 else ''}: {int(c)}"
                                                 for j, c in enumerate(hist) if c))
    out += ["", "## The macro searches", "",
            "| setting | = greedy macro | chosen rank (mean) | rank ≥ 4 | agrees with " + " / ".join(micro) + " |",
            "|:---|---:|---:|---:|---:|"]
    for name in macro:
        rs = [(r, r[name]) for r in recs if r.get(name)]
        if not rs:
            continue
        agree = []
        for mname in micro:
            both = [(r, x) for r, x in rs if r.get(mname)]
            agree.append(_pct(sum(x["key"] == r[mname]["key"] for r, x in both), len(both)))
        out.append(f"| {name} | {_pct(sum(x['greedy'] for _, x in rs), len(rs))} | "
                   f"{np.mean([x['rank'] for _, x in rs]):.2f} | {_pct(sum(x['rank'] >= 4 for _, x in rs), len(rs))} | "
                   f"{' / '.join(agree)} |")
    out += ["", "## Cost per position", "", "| method | network rows | seconds |", "|:---|---:|---:|"]
    for name, c in cost.items():
        out.append(f"| {name} | {c['rows'] / max(1, n):.0f} | {c['seconds'] / max(1, n):.3f} |")
    return "\n".join(out) + "\n"
