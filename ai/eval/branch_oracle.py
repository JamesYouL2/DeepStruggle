"""Was a decision in a recorded game a mistake? Paired playouts of the alternatives.

The method of `research/log/E5_11_selfplay_review.md`, which ran from a scratch script and was
never committed. A reviewer names a step of a .tslog.json replay and the alternatives they would
have played; this re-drives the game to that decision (checking every step against the replay's
own snapshots), forces each alternative, and plays every branch to the end with the model on both
sides. Pair k uses the same dice and, under `hidden="resample"`, the same redeal of the hidden
cards in every branch, so the branches differ only in the decision.

Branches:

* `policy` -- the model's own greedy choice from the decision on (always present, the baseline);
* `played` -- the action the replay recorded, when it differs from the greedy choice;
* each alternative: a list of flat actions forced in order from the decision (a card, then its
  mode, then targets...), after which the model plays on. Each must be legal where it is applied;
  `options()` lists what is legal at any point of a prefix.

`hidden` decides what the playouts know:

* `"resample"` (default) -- before each pair, the cards the deciding player could not see (the
  opponent's unseen hand and the deck) are redealt, as a determinized search would. The verdict is
  then about the decision as the player faced it, not with hindsight of the opponent's hand.
* `"true"` -- the recorded game's real hidden cards, as the E5-11 review did.

The verdict is only as strong as the model playing the continuations: a move whose value lies in
a follow-up the model does not find scores as worse than it is.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.replay_critic import load_actions, verify
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

#: (observations, masks) -> one greedy action per row.
PolicyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

_UINT64 = 1 << 64


def _decider(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def position_before(replay_path: str, step: int) -> Tuple[ts.GameState, int, Dict[str, Any]]:
    """The state at which the replay's step `step` was decided, the action it recorded there, and
    that step's record. Every earlier step is checked against the replay's snapshot; a divergence
    raises, because a drifted reconstruction still returns a verdict."""
    seed, steps = load_actions(replay_path)
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state, context="branch_oracle chance node")
    for rec in steps:
        if rec["step_index"] == step:
            if not ActionEncoder.get_legal_mask(state)[rec["flat"]]:
                raise RuntimeError(f"step {step}: the recorded action is illegal in the re-driven game")
            return state, int(rec["flat"]), rec
        ts.Engine.step_flat(state, rec["flat"])
        drain_chance(state, context="branch_oracle chance node")
        problem = verify(state, rec)
        if problem is not None:
            raise RuntimeError(f"step {rec['step_index']}: the re-driven game no longer matches the "
                               f"replay ({problem}); was the engine rebuilt since it was recorded?")
    raise ValueError(f"{replay_path} has no step {step}")


def apply_prefix(state: ts.GameState, prefix: Sequence[int]) -> ts.GameState:
    """A copy of `state` with the flat actions in `prefix` applied in order, each checked legal,
    with chance resolved after each. Forced single-choice nodes are NOT skipped, so a prefix names
    every decision it passes through, as `options()` lists them."""
    st = state.clone()
    for k, a in enumerate(prefix):
        if ts.Engine.is_terminal(st):
            raise ValueError(f"the game ended before prefix action {k} ({a})")
        if not ActionEncoder.get_legal_mask(st)[a]:
            legal = np.flatnonzero(ActionEncoder.get_legal_mask(st)).tolist()
            raise ValueError(f"prefix action {k} ({a}) is illegal here; legal: {legal}")
        ts.Engine.step_flat(st, int(a))
        drain_chance(st, context="branch_oracle prefix")
    return st


def options(state: ts.GameState, policy_probs: Optional[Callable[[ts.GameState], np.ndarray]] = None
            ) -> List[Dict[str, Any]]:
    """Every legal action at `state`: flat index, name, and the model's probability if given."""
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    probs = policy_probs(state) if policy_probs is not None else None
    out = []
    for a in np.flatnonzero(mask):
        row: Dict[str, Any] = {"idx": int(a), "name": ActionEncoder.get_action_name(state, int(a))}
        if probs is not None:
            row["p"] = float(probs[a])
        out.append(row)
    out.sort(key=lambda r: -r.get("p", 0.0))
    return out


def onnx_policy(model_path: str) -> Tuple[PolicyFn, Callable[[ts.GameState], np.ndarray]]:
    """A greedy batched policy and a per-state probability readout from a published export."""
    from tools.lib.player_agent import OnnxAgent

    agent = OnnxAgent(model_path)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return agent.act_batch(obs, masks, 0.0, True)

    def probs(st: ts.GameState) -> np.ndarray:
        obs = np.asarray(ts.extract_observation(st, _decider(st)), dtype=np.float32)[None, :]
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)[None, :]
        lg = agent.logits(obs, mask)[0]
        p = np.exp(lg - lg[mask[0].astype(bool)].max())
        p[~mask[0].astype(bool)] = 0.0
        return p / p.sum()

    return act, probs


def _pair_start(state: ts.GameState, k: int, seed: int, hidden: str) -> ts.GameState:
    st = state.clone()
    rng = random.Random(seed * 1_000_003 + k)
    if hidden == "resample":
        from ai.search.dmcts import determinize
        st = determinize(st, _decider(st), rng)
    elif hidden != "true":
        raise ValueError(f"hidden must be 'resample' or 'true', not {hidden!r}")
    st.rng_state = rng.getrandbits(64) % _UINT64
    return st


def play_branches(state: ts.GameState, branches: Mapping[str, Sequence[int]], pairs: Sequence[int],
                  act: PolicyFn, seed: int = 0, hidden: str = "resample",
                  max_steps: int = 4000) -> List[Dict[str, Any]]:
    """Play every (pair, branch) to the end. Returns one row per playout with the outcome from
    the deciding player's side: 1 win, 0.5 draw, 0 loss."""
    mover = _decider(state)
    starts: List[Tuple[int, str, ts.GameState]] = []
    for name, prefix in branches.items():
        if not prefix and name != "policy":
            raise ValueError(f"branch {name!r} forces nothing, so it is the policy branch again")
    for k in pairs:
        base = _pair_start(state, k, seed, hidden)
        for name, prefix in branches.items():
            # An empty prefix is the policy's own branch: its greedy move from the decision on.
            starts.append((k, name, apply_prefix(base, prefix)))
    n = len(starts)
    runner = ts.VectorizedBatchRunner(n, seed)
    for i, (_, _, st) in enumerate(starts):
        runner.set_state(i, st)
    runner.refresh_all()
    active = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        active &= ~np.array(runner.get_terminals())
        if not active.any():
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        acts = np.zeros(n, dtype=np.int32)
        rows = np.where(active)[0]
        acts[rows] = act(obs[rows], masks[rows])
        res = runner.step_flat_all(acts.tolist(), auto_advance=True)
        refused = [i for i in rows if res[i] == 0]
        if refused:
            raise RuntimeError(f"the engine refused a playout action in game {refused[0]}")
    out = []
    for i, (k, name, _) in enumerate(starts):
        st = runner.get_state(i)
        u = float(ts.Engine.get_terminal_utility(st))       # + = US
        u_mover = u if mover == ts.Player.US else -u
        out.append({"pair": int(k), "branch": name, "score": 1.0 if u_mover > 0 else (0.5 if u_mover == 0 else 0.0),
                    "vp": int(st.victory_points), "turn": int(st.turn),
                    "finished": bool(ts.Engine.is_terminal(st))})
    return out


def report(rows: Sequence[Dict[str, Any]], baseline: str = "policy") -> Tuple[str, Dict[str, Any]]:
    """Each branch's score for the deciding player, and its paired difference from `baseline`."""
    by: Dict[str, Dict[int, float]] = {}
    for r in rows:
        by.setdefault(r["branch"], {})[int(r["pair"])] = float(r["score"])
    base = by.get(baseline, {})
    lines = ["| branch | pairs | mover's score | − policy, paired |", "|---|---:|---:|---:|"]
    js: Dict[str, Any] = {}
    for name, sc in by.items():
        xs = list(sc.values())
        m = float(np.mean(xs))
        entry: Dict[str, Any] = {"pairs": len(xs), "score": m}
        cell = "—"
        if name != baseline and base:
            d = [sc[k] - base[k] for k in sc if k in base]
            dm = float(np.mean(d))
            se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
            entry.update(diff=dm, se=se)
            cell = f"{100 * dm:+.1f} ± {100 * se:.1f}"
        js[name] = entry
        lines.append(f"| {name} | {len(xs)} | {100 * m:.1f}% | {cell} |")
    return "\n".join(lines) + "\n", js
