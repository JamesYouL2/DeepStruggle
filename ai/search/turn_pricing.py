"""Pricing a decision's options to the end of the turn over paired worlds (P32 B4').

Every legal option of a small decision is played out over `worlds` worlds: a world redeals the cards
the decider cannot see once (`ai.search.dmcts.determinize`) for every option, while each option in
each world has its own dice. The network plays both sides greedily to the end of the current turn,
and the option's value in that world is the critic's, from the decider's side, at the first position
of the next turn -- or the result, if the game ends first. Shared by the probe
(`ai/eval/option_pricing.py`) and the trainer's turn credit (`ai/training/turn_credit.py`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.paired_playouts import apply_move, decider
from ai.search.dmcts import determinize
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features

_UINT64 = 1 << 64


@dataclass
class Candidate:
    game: int                 # index into the games
    choice: int               # how many of the game's decisions came before this one
    state: Any                # the position, before the move
    options: List[int]        # the legal flat actions
    probs: List[float]        # the policy's probability for each, at temperature 1
    taken: int                # the greedy move the game played
    values: Optional[np.ndarray] = None    # (options, worlds) turn-end values from the decider's side
    target: int = 0           # the target card this decision was recorded for (0: sampled)
    focus: int = -1           # the option of interest for a target decision: its event / its headline


def forward(model: Any, obs: np.ndarray, masks: np.ndarray, device: torch.device
             ) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        lg, v, _ = model(torch.from_numpy(obs).to(device, torch.float32), torch.from_numpy(masks).to(device))
    return lg.float().cpu().numpy(), v.float().reshape(-1).cpu().numpy()


def priceable(st: ts.GameState, cap: int) -> bool:
    """A decision small enough to price: 2..cap options, past setup, and not one point of a
    multi-point placement -- a coup's target (a POINT_NODE in the COUP op mode) is one choice and is
    priced."""
    if st.current_phase == ts.Phase.SETUP:
        return False
    ctx = st.ctx()
    if int(ctx.decision_type) == int(ts.DecisionType.POINT_NODE) and ctx.op_mode != ts.OpMode.COUP:
        return False
    n = int(np.asarray(ActionEncoder.get_legal_mask(st)).sum())
    return 2 <= n <= cap


@dataclass
class Nested:
    parent: int               # index of the candidate whose branch it was met in
    option: int               # that branch's option index
    cand: Candidate


def price(model: Any, device: torch.device, cands: Sequence[Candidate], worlds: int, seed: int,
          chunk: int = 2048, max_plies: int = 1500, lookahead: int = 0, look_worlds: int = 4,
          look_cap: int = 12, nested: Optional[List[Nested]] = None, nested_p: float = 0.0) -> None:
    """Fill each candidate's `values`: every option over `worlds` worlds, to the end of the turn.

    `lookahead` > 0 (depth 2): in each branch, the decider's next `lookahead` small decisions of the
    same turn are chosen by pricing them in turn (depth 1, `look_worlds` worlds) instead of greedily;
    the opponent stays greedy. `nested` collects, with probability `nested_p`, the decider's small
    decisions met in the branches of options the policy did not take -- positions its own play does
    not reach -- for pricing on their own."""
    feats = model_obs_features(model)
    jobs: List[Tuple[int, int, int, ts.GameState]] = []        # (candidate, option, world, start)
    rng = random.Random(seed)
    for ci, c in enumerate(cands):
        c.values = np.full((len(c.options), worlds), np.nan)
        me = decider(c.state)
        for w in range(worlds):
            base = determinize(c.state.clone(), me, random.Random(rng.getrandbits(62)))
            for oi, a in enumerate(c.options):
                st = base.clone()
                st.rng_state = rng.getrandbits(64) % _UINT64        # this option's own dice
                try:
                    jobs.append((ci, oi, w, apply_move(st, a)))
                except ValueError:
                    pass                    # illegal in this deal: the world is dropped below
    for lo in range(0, len(jobs), chunk):
        part = jobs[lo:lo + chunk]
        n = len(part)
        runner = ts.VectorizedBatchRunner(n, seed + lo)
        if feats:
            runner.set_obs_features([feats] * n, [feats] * n)
        start_turn = np.zeros(n, dtype=np.int64)
        sides = []
        for i, (ci, _oi, _w, st) in enumerate(part):
            runner.set_state(i, st)
            start_turn[i] = int(cands[ci].state.turn)
            sides.append(decider(cands[ci].state))
        runner.refresh_all()
        done = np.zeros(n, dtype=bool)
        value = np.zeros(n)
        look_left = np.full(n, lookahead, dtype=np.int64)
        branch_rng = random.Random(seed + 7 * lo + 1)
        for _ply in range(max_plies + 1):
            term = np.asarray(runner.get_terminals(), dtype=bool)
            turns = np.asarray(runner.get_turns(), dtype=np.int64)
            new = ~done & (term | (turns > start_turn))
            if new.any():
                util = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
                idx = np.flatnonzero(new)
                rows = [int(i) for i in idx if not term[i]]
                for i in idx:
                    if term[i]:
                        value[i] = util[i] if sides[i] == ts.Player.US else -util[i]
                if rows:
                    sts = [runner.get_state(i) for i in rows]
                    obs = np.stack([np.asarray(ts.extract_observation_features(s, sides[i], feats),
                                               dtype=np.float32) for s, i in zip(sts, rows)])
                    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8)
                                      for s in sts])
                    _lg, v = forward(model, obs, masks, device)
                    value[rows] = v
                done |= new
            if done.all():
                break
            obs = np.asarray(runner.get_observations(), dtype=np.float32)
            masks = np.asarray(runner.get_action_masks())
            lg, _ = forward(model, obs, masks, device)
            lg = np.where(masks > 0, lg, -np.inf)
            acts = lg.argmax(axis=1)
            acts[done] = masks[done].argmax(axis=1)     # finished rows: any legal move, ignored
            if lookahead > 0 or nested is not None:
                dps = np.asarray(runner.get_decision_players(), dtype=np.int64)
                subs: List[Candidate] = []
                owners: List[int] = []
                for i in range(n):
                    if done[i] or turns[i] != start_turn[i] or dps[i] != int(sides[i]):
                        continue
                    ci, oi = part[i][0], part[i][1]
                    off_policy = cands[ci].options[oi] != cands[ci].taken
                    want_nest = nested is not None and off_policy and branch_rng.random() < nested_p
                    if look_left[i] <= 0 and not want_nest:
                        continue
                    st = runner.get_state(i)
                    if not priceable(st, look_cap):
                        continue
                    legal = [int(a) for a in np.flatnonzero(masks[i])]
                    if want_nest and nested is not None:
                        z = lg[i][legal] - lg[i][legal].max()
                        pz = np.exp(z) / np.exp(z).sum()
                        nested.append(Nested(ci, oi, Candidate(game=-1, choice=-1, state=st.clone(), options=legal,
                                                               probs=[float(x) for x in pz], taken=int(acts[i]))))
                    if look_left[i] > 0:
                        subs.append(Candidate(game=-1, choice=-1, state=st, options=legal, probs=[],
                                              taken=int(acts[i])))
                        owners.append(i)
                if subs:
                    price(model, device, subs, look_worlds, branch_rng.getrandbits(30), chunk, max_plies)
                    for i, c in zip(owners, subs):
                        if c.values is not None and c.values.shape[1] > 0:
                            acts[i] = c.options[int(np.argmax(c.values.mean(axis=1)))]
                        look_left[i] -= 1
            res = np.asarray(runner.step_flat_all(acts.tolist(), True))
            if int(((res == 0) & ~done).sum()):
                raise RuntimeError("the engine refused a greedy playout action")
        else:
            raise RuntimeError(f"{int((~done).sum())} playouts still running after {max_plies} plies")
        for (ci, oi, w, _st), v in zip(part, value):
            vals = cands[ci].values
            assert vals is not None
            vals[oi, w] = v
    for c in cands:                          # a world with any option missing is dropped whole
        assert c.values is not None
        c.values = c.values[:, ~np.isnan(c.values).any(axis=0)]


