"""Ops-efficient realignment spots at DEFCON 2: is realigning there better than what the net plays?

The fork owner's rule (a strong player's): at DEFCON 2, realign whenever it is ops-efficient --
**several legal targets at a net modifier of +1 or better, or any battleground at +2 or better**,
each holding opponent influence -- and often at lower modifiers than the net waits for.
`research/log/E5_11_selfplay_review.md` found forced realignments lose 3-5 points, but its forced
branch let the policy choose its own targets, so it measured "realign the way you would", not the
rule. This separates the two.

A **spot** is a play-mode decision (event / space / influence / coup / realign) at DEFCON 2 where
realignment is legal and the rule's condition holds, found in the model's own self-play. From each,
three branches, played to the end by the model on both sides:

* `policy` -- the model's own greedy choice (which may itself be realignment);
* `realign_own` -- realignment forced, the model choosing every target;
* `realign_rule` -- realignment forced, every roll by the rule (the owner's, 2026-10-01): roll only
  where the expected influence swing is positive (`expected_swing`; always so where the mover holds
  no influence and the opponent some), battlegrounds first, then the largest swing, re-read after
  each roll; stop when no target is worth a roll, forfeiting the remaining ops.

Pair k of a spot uses the same redeal of the cards the mover cannot see and the same dice in every
branch (`branch_oracle._pair_start`), so the branches differ only in the decision. The net modifier
is the engine's own (`observation.cpp` `compute_net_realign_mod`, board slot 2 x 5, mover's view):
adjacent control, influence superiority and superpower adjacency, mine minus the opponent's.

A gain for `realign_rule` is a lower bound on what realignment is worth here: the model plays the
continuation, and a realignment whose value lies in the follow-up is worth what the model makes of it.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _pair_start, play_out
from ai.eval.ops_block import CONFIRM_DONE, NODE_OFFSET, N_COUNTRIES, country_table, influence
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

MODE_BASE = 110
MODES = ("event", "space", "influence", "coup", "realign")
REALIGN = MODE_BASE + 4
_BOARD_W = 26
_NET_SLOT = 2
BRANCHES = ("policy", "realign_own", "realign_rule")


def net_modifiers(st: ts.GameState, mover: ts.Player) -> np.ndarray:
    """[84] net realignment modifier for `mover` (engine feature, un-normalised)."""
    board = np.asarray(ts.extract_observation(st, mover))[:N_COUNTRIES * _BOARD_W].reshape(N_COUNTRIES, _BOARD_W)
    return np.rint(board[:, _NET_SLOT] * 5.0).astype(np.int32)


def _realign_targets(st: ts.GameState) -> np.ndarray:
    """[84] bool: countries a realignment roll may target at this node."""
    mask = np.asarray(ActionEncoder.get_legal_mask(st))
    return mask[NODE_OFFSET:NODE_OFFSET + N_COUNTRIES].astype(bool)


def _in_realign(st: ts.GameState, mover: ts.Player, turn_ar: Tuple[int, int]) -> bool:
    ctx = st.ctx()
    return (not ts.Engine.is_terminal(st) and (st.turn, st.action_round) == turn_ar
            and ctx.decision_player == mover and ctx.op_mode == ts.OpMode.REALIGN
            and ctx.decision_type == ts.DecisionType.POINT_NODE and bool(_realign_targets(st).any()))


def spot(st: ts.GameState) -> Optional[Dict[str, Any]]:
    """The rule's condition at a play-mode node, or None. Reads the targets legal once realignment
    is chosen (DEFCON region limits, The Reformer and the like are the engine's)."""
    ctx = st.ctx()
    if int(st.defcon) != 2 or ctx.decision_type != ts.DecisionType.SELECT_PLAY_MODE:
        return None
    mask = np.asarray(ActionEncoder.get_legal_mask(st))
    if not mask[REALIGN] or mask[MODE_BASE:MODE_BASE + len(MODES)].sum() < 2:
        return None
    mover = ctx.decision_player
    probe = st.clone()
    ts.Engine.step_flat(probe, REALIGN)
    drain_chance(probe, context="realign_spots probe")
    if probe.ctx().op_mode != ts.OpMode.REALIGN:
        return None
    legal = _realign_targets(probe)
    net = net_modifiers(probe, mover)
    _, bg, _ = country_table()
    side = 0 if mover == ts.Player.US else 1
    opp = influence(probe)[1 - side] > 0
    plus1 = legal & opp & (net >= 1)
    bg2 = legal & opp & bg & (net >= 2)
    if plus1.sum() < 2 and not bg2.any():
        return None
    return {"side": "US" if side == 0 else "USSR", "turn": int(st.turn), "ar": int(st.action_round),
            "ops": int(probe.ctx().remaining_steps), "n_plus1": int(plus1.sum()), "n_bg2": int(bg2.sum()),
            "kind": "bg+2" if bg2.any() else "multi+1", "best_net": int(net[legal & opp].max())}


def expected_swing(net: int, mine: int, theirs: int) -> float:
    """Expected (opponent influence removed - own influence lost) of one realignment roll at net
    modifier `net`, exactly over the 36 dice pairs: the higher total removes the difference from
    the loser, capped at what the loser holds (`Operations::execute_realign`)."""
    total = 0
    for a in range(1, 7):
        for b in range(1, 7):
            d = a - b + net
            total += min(d, theirs) if d > 0 else -min(-d, mine)
    return total / 36.0


def rule_target(st: ts.GameState, mover: ts.Player) -> Optional[int]:
    """The rule's roll, or None to stop. A target is worth a roll only if its expected swing is
    positive -- so any target holding opponent influence where the mover holds none, since a lost
    roll there costs nothing. Of those, battlegrounds first, then the largest expected swing."""
    legal = np.flatnonzero(_realign_targets(st))
    net = net_modifiers(st, mover)
    _, bg, _ = country_table()
    side = 0 if mover == ts.Player.US else 1
    inf = influence(st)
    swing = {int(c): expected_swing(int(net[c]), int(inf[side, c]), int(inf[1 - side, c])) for c in legal}
    worth = [c for c, v in swing.items() if v > 0]
    if not worth:
        return None
    return max(worth, key=lambda c: (bool(bg[c]), swing[c]))


def drive_rule(start: ts.GameState) -> Tuple[ts.GameState, List[int]]:
    """Realignment at `start` with every roll by the rule; when no target is worth a roll, the
    mover stops (the remaining ops are forfeited). Returns the state and the net modifier at each
    roll."""
    st = start.clone()
    mover = st.ctx().decision_player
    turn_ar = (st.turn, st.action_round)
    ts.Engine.step_flat(st, REALIGN)
    drain_chance(st, context="realign_spots rule")
    mods: List[int] = []
    for _ in range(64):
        if not _in_realign(st, mover, turn_ar):
            return st, mods
        c = rule_target(st, mover)
        if c is None:
            if not ActionEncoder.get_legal_mask(st)[CONFIRM_DONE]:
                raise RuntimeError("the rule wants to stop, but the engine offers no way to")
            ts.Engine.step_flat(st, CONFIRM_DONE)
        else:
            mods.append(int(net_modifiers(st, mover)[c]))
            ts.Engine.step_flat(st, NODE_OFFSET + c)
        drain_chance(st, context="realign_spots rule")
    raise RuntimeError("the rule's realignment did not end")


def own_first_target(st: ts.GameState, act: PolicyFn) -> Tuple[int, int]:
    """The model's first realignment target at this spot and its net modifier, on the real state."""
    probe = st.clone()
    mover = probe.ctx().decision_player
    ts.Engine.step_flat(probe, REALIGN)
    drain_chance(probe, context="realign_spots own")
    obs = np.asarray(ts.extract_observation(probe, mover), dtype=np.float32)[None]
    mask = np.asarray(ActionEncoder.get_legal_mask(probe), dtype=np.uint8)[None]
    a = int(act(obs, mask)[0])
    c = a - NODE_OFFSET
    if not 0 <= c < N_COUNTRIES:
        return -1, 0
    return c, int(net_modifiers(probe, mover)[c])


def collect(act: PolicyFn, probs_fn: Callable[[ts.GameState], np.ndarray], n: int, seed: int,
            envs: int = 16, accept: float = 0.03, sample: Optional[PolicyFn] = None,
            max_steps: int = 200_000) -> Dict[str, Any]:
    """Spots from the model's self-play (moves sampled by `sample`, typically temperature 0.1, so
    games spread), until `n` are kept. Every spot is counted; each is kept with probability
    `accept`, and at most one per game, so the sample spreads over the game rather than taking the
    first spot of each (a game holds ~40)."""
    sample = sample or act
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(envs, seed)
    runner.refresh_all()
    kept: List[Dict[str, Any]] = []
    seen = greedy_realign = games = 0
    taken = np.zeros(envs, dtype=bool)
    for _ in range(max_steps):
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        greedy = act(obs, masks)
        for i in np.flatnonzero(masks[:, REALIGN].astype(bool)):
            st = runner.get_state(int(i))
            info = spot(st)
            if info is None:
                continue
            seen += 1
            info["policy_mode"] = MODES[int(greedy[i]) - MODE_BASE] if MODE_BASE <= greedy[i] < MODE_BASE + 5 else "?"
            greedy_realign += info["policy_mode"] == "realign"
            if taken[i] or rng.random() >= accept:
                continue
            taken[i] = True
            info["p_realign"] = float(probs_fn(st)[REALIGN])
            info["own_target"], info["own_net"] = own_first_target(st, act)
            kept.append({"state": st.clone(), **info})
        if len(kept) >= n:
            break
        acts = sample(obs, masks)
        runner.step_flat_all(acts.tolist(), auto_advance=True)
        done = np.flatnonzero(np.array(runner.get_terminals()))
        for i in done:
            games += 1
            taken[i] = False
            runner.reset_game(int(i), seed * 1_000_003 + envs + games)
        if len(done):
            runner.refresh_all()
    return {"spots": kept[:n], "seen": seen, "greedy_realign": greedy_realign, "games": games}


def play_spots(spots: Sequence[Dict[str, Any]], pairs: int, act: PolicyFn, seed: int,
               chunk: int = 768) -> List[Dict[str, Any]]:
    """Every spot's three branches over `pairs` paired playouts. Returns per-spot rows (no state)."""
    rows: List[Dict[str, Any]] = []
    per_spot = 3 * pairs
    step = max(1, chunk // per_spot)
    for lo in range(0, len(spots), step):
        group = spots[lo:lo + step]
        starts: List[ts.GameState] = []
        movers: List[ts.Player] = []
        rule_mods: List[List[int]] = []
        for j, sp in enumerate(group):
            st = sp["state"]
            mover = st.ctx().decision_player
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo + j, "resample")
                own = base.clone()
                ts.Engine.step_flat(own, REALIGN)
                drain_chance(own, context="realign_spots own")
                rule, mods = drive_rule(base)
                if k == 0:
                    rule_mods.append(mods)
                starts += [base, own, rule]
                movers += [mover] * 3
        ends = play_out(starts, movers, act, seed)
        for j, sp in enumerate(group):
            sc = np.array([e["score"] for e in ends[j * per_spot:(j + 1) * per_spot]]).reshape(pairs, 3)
            row = {k: v for k, v in sp.items() if k != "state"}
            row.update({b: float(sc[:, i].mean()) for i, b in enumerate(BRANCHES)})
            row["pairs"] = pairs
            row["rule_mods"] = rule_mods[j]
            rows.append(row)
    return rows


def _diff(rows: Sequence[Dict[str, Any]], a: str, b: str) -> Tuple[float, float]:
    d = np.array([r[a] - r[b] for r in rows])
    if len(d) < 2:
        return float(d.mean()) if len(d) else float("nan"), float("nan")
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def report(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Markdown and JSON summary, by spot kind, side and what the model chose."""
    def line(label: str, sel: Sequence[Dict[str, Any]]) -> str:
        if not sel:
            return f"| {label} | 0 | | | | |"
        cells = []
        for a, b in (("realign_rule", "policy"), ("realign_own", "policy"), ("realign_rule", "realign_own")):
            m, se = _diff(sel, a, b)
            cells.append(f"{100 * m:+.1f} ± {100 * se:.1f}")
        pol = 100 * float(np.mean([r["policy"] for r in sel]))
        return f"| {label} | {len(sel)} | {pol:.1f}% | " + " | ".join(cells) + " |"

    groups: List[Tuple[str, List[Dict[str, Any]]]] = [("all", list(rows))]
    for key, vals in (("kind", ("bg+2", "multi+1")), ("side", ("US", "USSR"))):
        groups += [(f"{key} {v}", [r for r in rows if r[key] == v]) for v in vals]
    groups += [("model realigns", [r for r in rows if r["policy_mode"] == "realign"]),
               ("model declines", [r for r in rows if r["policy_mode"] != "realign"])]
    for mode in MODES[:4]:
        groups.append((f"declines for {mode}", [r for r in rows if r["policy_mode"] == mode]))
    for lo, hi in ((1, 2), (3, 3), (4, 9)):
        groups.append((f"ops {lo}-{hi}" if hi > lo else f"ops {lo}", [r for r in rows if lo <= r["ops"] <= hi]))

    seen = meta.get("seen", 0)
    out = [f"# Ops-efficient realignment spots at DEFCON 2 — {meta.get('model', '?')}", "",
           f"{len(rows)} spots, {rows[0]['pairs'] if rows else 0} paired playouts per branch; "
           f"{seen} spots seen in {meta.get('games', 0)} self-play games, where the greedy model "
           f"realigns in {100 * meta.get('greedy_realign', 0) / max(seen, 1):.1f}%. "
           "Differences in points (win % from the mover's side), ± one standard error over spots.", "",
           "| group | spots | policy | rule − policy | own targets − policy | rule − own targets |",
           "|:---|---:|---:|---:|---:|---:|"]
    out += [line(lbl, sel) for lbl, sel in groups]
    own_net = [r["own_net"] for r in rows if r["own_target"] >= 0]
    rule_first = [r["rule_mods"][0] for r in rows if r["rule_mods"]]
    all_mods = [m for r in rows for m in r["rule_mods"]]
    out += ["", "**Targeting.** Net modifier of the first roll: the model's own "
            f"{np.mean(own_net):+.2f} (≤ 0 in {100 * np.mean(np.array(own_net) <= 0):.0f}%), "
            f"the rule's {np.mean(rule_first):+.2f}; the rule's rolls over the whole realignment "
            f"{np.mean(all_mods):+.2f} (≤ 0 in {100 * np.mean(np.array(all_mods) <= 0):.0f}%). The rule rolls "
            f"{np.mean([len(r['rule_mods']) for r in rows]):.2f} times per realignment against "
            f"{np.mean([r['ops'] for r in rows]):.2f} ops, and stops early in "
            f"{100 * np.mean([len(r['rule_mods']) < r['ops'] for r in rows]):.0f}% of spots."
            if own_net and rule_first else ""]
    summary = {"meta": meta, "groups": {}}
    for lbl, sel in groups:
        summary["groups"][lbl] = {"n": len(sel), **{f"{a}-{b}": _diff(sel, a, b) for a, b in
                                  (("realign_rule", "policy"), ("realign_own", "policy"), ("realign_rule", "realign_own"))}}
    return "\n".join(out) + "\n", summary
