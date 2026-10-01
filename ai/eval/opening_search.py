"""Which USSR opening does deep search prefer, on the same deals?

Each candidate opening is treated as one macro-action. On a deal, the USSR's six setup points are
placed by the candidate's script, the US then sets up (a fixed script, or the network's own
choice), and a determinized search runs from the first decision after setup -- the turn-1
headline -- at several budgets. The quantity compared is the searched value of that position for
the USSR, paired by deal across openings, so the deal's own luck cancels.

Two readings per budget:

* `critic` -- the network's value at the root, no search (the same at every budget);
* `search` -- the root's mean backed-up value over all simulations, from `worlds` independent
  determinizations of the hidden cards.

Search backs up the critic, so a deep search's verdict inherits the critic's errors with more
confidence; the paired-deal playouts in the tournament runs are the check on it, and the two
should be read together.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from bindings.settle import SettleMode, settle
from tools.lib.openings import (NODE_OFFSET, US_WG3_FR3_IT2, USSR_EG4_POL4_AUT1, USSR_EG4_POL5,
                                USSR_OPENING, USSR_POLAND_HUNGARY, USSR_POLAND_YUGOSLAVIA, expand)

#: The candidates, as six USSR placements each (East Germany starts at 3, so "EG 4" is one point).
CANDIDATES: Dict[str, List[int]] = {
    "pol3_hun3": expand(USSR_POLAND_HUNGARY),
    "pol3_yug3": expand(USSR_POLAND_YUGOSLAVIA),
    "eg4_pol4_aut1": expand(USSR_EG4_POL4_AUT1),
    "eg4_pol4_yug1": expand(USSR_OPENING),
    "eg4_pol5": expand(USSR_EG4_POL5),
}
BASELINE = "pol3_hun3"
#: Tree nodes searched at once; see search_deals.
MAX_NODES = 160_000
US_FIXED = expand(US_WG3_FR3_IT2)


def _decider(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def set_up(seed: int, ussr: Sequence[int], us_setup: str, net: Any = None) -> ts.GameState:
    """A fresh game on deal `seed`, set up with the USSR script `ussr` and the US either fixed
    (`us_setup == "fixed"`, WG 3 / France 3 / Italy 2 / Iran 2) or the network's argmax ("net").
    Returns the state at the first decision after setup. Raises on an illegal placement rather
    than fall back to another choice."""
    from bindings.action_encoder import ActionEncoder

    runner = ts.VectorizedBatchRunner(1, seed)
    runner.reset_game(0, seed)
    st = runner.get_state(0)
    k = {"US": 0, "USSR": 0}
    for _ in range(64):
        settle(st, SettleMode.CHANCE)
        if st.current_phase != ts.Phase.SETUP:
            return st
        side = "US" if st.ctx().decision_player == ts.Player.US else "USSR"
        if side == "USSR" or us_setup == "fixed":
            script = ussr if side == "USSR" else US_FIXED
            idx = NODE_OFFSET + script[k[side]]
            k[side] += 1
        else:
            idx = int(net.select_action(st, ts.Player.US, temperature=0.0))
        if not ActionEncoder.get_legal_mask(st, False)[idx]:
            raise RuntimeError(f"seed {seed}: {side} placement {idx} is illegal in setup")
        ts.Engine.step_flat(st, idx, False)
    raise RuntimeError("setup did not end")


def search_deals(model_path: str, seeds: Sequence[int], budgets: Sequence[int], worlds: int,
                 us_setup: str = "fixed", search_seed: int = 0,
                 deals_per_batch: int = 4) -> List[Dict[str, Any]]:
    """One row per (deal, opening): the critic's value and the searched value at each budget,
    both from the USSR's side, plus the US setup actually played."""
    from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
    from tools.lib.player_agent import OnnxAgent

    net = OnnxAgent(model_path)
    module = net.as_module()
    searchers = {b: BatchedMCTS(module, device="cpu", config=BatchedMCTSConfig(
        simulations=b, temperature=0.0, auto_advance=True, advance_root=False,
        determinize=True, seed=search_seed + b)) for b in budgets}
    names = list(CANDIDATES)
    rows: List[Dict[str, Any]] = []
    for c0 in range(0, len(seeds), deals_per_batch):
        batch = seeds[c0:c0 + deals_per_batch]
        states = {(s, o): set_up(s, CANDIDATES[o], us_setup, net) for s in batch for o in names}
        keys = list(states)
        out: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for key in keys:
            st = states[key]
            out[key] = {"seed": key[0], "opening": key[1],
                        "first_mover": "US" if _decider(st) == ts.Player.US else "USSR",
                        "us_setup_influence": _us_influence(st), "search": {}}
        for b, sr in searchers.items():
            rep = [states[key].clone() for key in keys for _ in range(worlds)]
            # Every tree node holds a 4 KB GameState, so a batch is capped at MAX_NODES nodes
            # (~1 GB with Python overhead): at 8,192 simulations, 19 roots at a time.
            step = max(1, MAX_NODES // b)
            roots = []
            for r0 in range(0, len(rep), step):
                roots += sr._search(rep[r0:r0 + step])
            for i, key in enumerate(keys):
                vals = []
                for w in range(worlds):
                    r = roots[i * worlds + w]
                    if r is None or not r.n or sum(r.n) == 0:
                        continue
                    vals.append(-sum(r.w) / sum(r.n))          # US view -> USSR view
                    if b == budgets[0] and w == 0:
                        out[key]["critic"] = -float(r.value_us)
                out[key]["search"][str(b)] = vals
        rows += [out[key] for key in keys]
    return rows


def _us_influence(st: ts.GameState) -> List[int]:
    """US influence by country after setup, so a "net" US setup can be read back."""
    import json
    return [int(x) for x in json.loads(st.to_save_json())["us_influence"]]


# ---------------------------------------------------------------------------------- report

def report(rows: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Per budget: each opening's mean USSR value (as win-probability points, 50 x value), and
    its paired difference from the baseline over deals, with a standard error over deals."""
    by: Dict[Tuple[int, str], Dict[str, Any]] = {(r["seed"], r["opening"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    openings = [o for o in CANDIDATES if any((s, o) in by for s in seeds)]
    budgets = sorted({int(b) for r in rows for b in r["search"]})

    def reading(r: Dict[str, Any], b: Optional[int]) -> Optional[float]:
        if b is None:
            return r.get("critic")
        xs = r["search"].get(str(b), [])
        return float(np.mean(xs)) if xs else None

    js: Dict[str, Any] = {"deals": len(seeds), "budgets": {}}
    head = "| reading | " + " | ".join(openings) + " |"
    lines = ["# Deep search on the USSR opening", "",
             f"{len(seeds)} deals. Values are USSR win-probability points (50 x value, so 0 = even); "
             f"differences are paired by deal against `{BASELINE}`, ± one standard error over deals.", "",
             head, "|" + "---|" * (len(openings) + 1)]
    for b in [None] + budgets:
        label = "critic (0 sims)" if b is None else f"search {b} sims"
        cells, entry = [], {}
        base = {s: reading(by[(s, BASELINE)], b) for s in seeds if (s, BASELINE) in by}
        for o in openings:
            pairs = [(reading(by[(s, o)], b), base.get(s)) for s in seeds if (s, o) in by]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                cells.append("—")
                continue
            mean = 50 * float(np.mean([x for x, _ in pairs]))
            if o == BASELINE:
                cells.append(f"{mean:+.1f}")
                entry[o] = {"mean": mean}
                continue
            d = [50 * (x - y) for x, y in pairs]
            m = float(np.mean(d))
            se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
            cells.append(f"{mean:+.1f} (Δ {m:+.1f} ± {se:.1f})")
            entry[o] = {"mean": mean, "diff": m, "se": se, "n": len(d)}
        js["budgets"][label] = entry
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", js
