"""Pricing a decision's options to the end of the turn over paired worlds (P32 B4').

Every legal option of a small decision is played out over `worlds` worlds: a world redeals the cards
the decider cannot see once (`ai.search.dmcts.determinize`) for every option, while each option in
each world has its own dice. The network plays both sides greedily to the end of the current turn,
and the option's value in that world is the critic's, from the decider's side, at the first position
of the next turn -- or the result, if the game ends first. Shared by the probe
(`ai/eval/option_pricing.py`) and the trainer's turn credit (`ai/training/turn_credit.py`).

The rollout root (`ai/search/rollout_root.py`) prices through the same function with three switches:
`horizon`, a stop after that many action-round boundaries instead of the turn's end; `shared_dice`,
one dice stream per world for every option, so the options are paired on the dice as well as the
deal; and `guard`, the rollout root's suicide guard at DEFCON 2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.paired_playouts import apply_move, ar_key, decider, loses_now
from ai.search.dmcts import determinize
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features

_UINT64 = 1 << 64
#: The base observation's 100 global floats start here (84 x 26 board, 110 x 14 card; v2.3).
_GLOBALS = int(ts.OBS_SIZE) - 100
#: Global slots read back from a batch runner's observations, so a playout never copies a state out
#: per row per ply (engine/src/observation.cpp, ctx_slots in engine/include/ts/game_state.hpp).
_G_DEFCON, _G_ACTION_ROUND, _G_PHASING_IS_ME, _G_PHASE = 1, 7, 8, 9
_G_DECISION_TYPE = 72            # one-hot over DecisionType, 8 slots
_G_OP_MODE = 80                  # one-hot INFLUENCE, COUP, REALIGN
#: Decisions at which one move can end the game against its own player at DEFCON 2: a card's play
#: mode, an event branch, and a coup's target. An Influence or realignment point cannot, so a
#: POINT_NODE counts only in the COUP op mode -- the same picks as guarding every POINT_NODE, since
#: `loses_now` is false at the others, at a fraction of the probes.
GUARDED = frozenset({int(ts.DecisionType.SELECT_PLAY_MODE), int(ts.DecisionType.CHOOSE_BRANCH),
                     int(ts.DecisionType.POINT_NODE)})


def ar_key_from_obs(row: np.ndarray, turn: int, decision_player: int) -> Tuple[int, int, int, int]:
    """`ai.eval.paired_playouts.ar_key` of a batch-runner position read from its observation row
    (extracted from the decision player's side): the action round is global slot 7 (/8), whether the
    phasing player is the viewer slot 8 (+1 / -1), the phase slot 9 (/6); the turn is the runner's.
    `tests/training/test_turn_pricing.py` holds it to `ar_key` of the state."""
    g = row[_GLOBALS:]
    other = int(ts.Player.USSR) if decision_player == int(ts.Player.US) else int(ts.Player.US)
    return (int(turn), int(round(float(g[_G_ACTION_ROUND]) * 8.0)),
            decision_player if g[_G_PHASING_IS_ME] > 0 else other, int(round(float(g[_G_PHASE]) * 6.0)))


def ar_keys_from_obs(obs: np.ndarray, turns: np.ndarray, dps: np.ndarray) -> np.ndarray:
    """`ar_key_from_obs` for every row at once: an (n, 4) int array."""
    g = obs[:, _GLOBALS:]
    us, ussr = int(ts.Player.US), int(ts.Player.USSR)
    other = np.where(dps == us, ussr, us)
    return np.stack([turns.astype(np.int64), np.rint(g[:, _G_ACTION_ROUND] * 8.0).astype(np.int64),
                     np.where(g[:, _G_PHASING_IS_ME] > 0, dps, other).astype(np.int64),
                     np.rint(g[:, _G_PHASE] * 6.0).astype(np.int64)], axis=1)


def guard_rows(obs: np.ndarray) -> np.ndarray:
    """The rows whose greedy move the DEFCON-2 suicide guard must probe, from their observations:
    DEFCON at 2 or below (slot 1, /5) and a guarded decision -- a play mode, an event branch, or a
    coup's target (POINT_NODE in the COUP op mode). Held to the states by a test."""
    g = obs[:, _GLOBALS:]
    low = np.rint(g[:, _G_DEFCON] * 5.0) <= 2
    play_mode = g[:, _G_DECISION_TYPE + int(ts.DecisionType.SELECT_PLAY_MODE)] > 0.5
    branch = g[:, _G_DECISION_TYPE + int(ts.DecisionType.CHOOSE_BRANCH)] > 0.5
    coup_target = (g[:, _G_DECISION_TYPE + int(ts.DecisionType.POINT_NODE)] > 0.5) & \
                  (g[:, _G_OP_MODE + int(ts.OpMode.COUP)] > 0.5)
    return low & (play_mode | branch | coup_target)


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
          chunk: int = 16384, max_plies: int = 1500, lookahead: int = 0, look_worlds: int = 4,
          look_cap: int = 12, nested: Optional[List[Nested]] = None, nested_p: float = 0.0, *,
          horizon: Optional[int] = None, shared_dice: bool = False, guard: bool = False) -> int:
    """Fill each candidate's `values`: every option over `worlds` worlds, to the end of the turn.
    Returns the network rows evaluated.

    `lookahead` > 0 (depth 2): in each branch, the decider's next `lookahead` small decisions of the
    same turn are chosen by pricing them in turn (depth 1, `look_worlds` worlds) instead of greedily;
    the opponent stays greedy. `nested` collects, with probability `nested_p`, the decider's small
    decisions met in the branches of options the policy did not take -- positions its own play does
    not reach -- for pricing on their own.

    The rollout root's switches: `horizon` (an int) stops a playout after that many action-round
    boundaries from the position after the move instead of at the turn's end -- 0 reads the critic
    there at once; `shared_dice` gives every option of a world the same dice stream rather than its
    own; `guard` replaces, at DEFCON 2 at the decisions where one move can end the game against its
    own player (`GUARDED`), a greedy move that does so with the next-best of the top four."""
    feats = model_obs_features(model)
    jobs: List[Tuple[int, int, int, ts.GameState]] = []        # (candidate, option, world, start)
    rng = random.Random(seed)
    for ci, c in enumerate(cands):
        c.values = np.full((len(c.options), worlds), np.nan)
        me = decider(c.state)
        for w in range(worlds):
            base = determinize(c.state.clone(), me, random.Random(rng.getrandbits(62)))
            if shared_dice:
                base.rng_state = rng.getrandbits(64) % _UINT64    # this world's dice, every option
            for oi, a in enumerate(c.options):
                st = base.clone()
                if not shared_dice:
                    st.rng_state = rng.getrandbits(64) % _UINT64    # this option's own dice
                try:
                    jobs.append((ci, oi, w, apply_move(st, a)))
                except ValueError:
                    pass                    # illegal in this deal: the world is dropped below
    rows_total = 0
    for lo in range(0, len(jobs), chunk):
        part = jobs[lo:lo + chunk]
        n = len(part)
        runner = ts.VectorizedBatchRunner(n, seed + lo)
        if feats:
            runner.set_obs_features([feats] * n, [feats] * n)
        start_turn = np.zeros(n, dtype=np.int64)
        sides = np.zeros(n, dtype=np.int64)
        off_policy = np.zeros(n, dtype=bool)
        keys = np.zeros((n, 4), dtype=np.int64)
        for i, (ci, oi, _w, st) in enumerate(part):
            runner.set_state(i, st)
            start_turn[i] = int(cands[ci].state.turn)
            sides[i] = int(decider(cands[ci].state))
            off_policy[i] = cands[ci].options[oi] != cands[ci].taken
            keys[i] = ar_key(st)
        crossed = np.zeros(n, dtype=np.int64)
        runner.refresh_all()
        done = np.zeros(n, dtype=bool)
        value = np.zeros(n)
        look_left = np.full(n, lookahead, dtype=np.int64)
        branch_rng = random.Random(seed + 7 * lo + 1)
        np_rng = np.random.default_rng(seed + 7 * lo + 2)
        pos = np.full(n, -1, dtype=np.int64)
        for _ply in range(max_plies + 1):
            term = np.asarray(runner.get_terminals(), dtype=bool)
            turns = np.asarray(runner.get_turns(), dtype=np.int64)
            if horizon is None:
                reached = turns > start_turn
            else:
                # Boundaries crossed so far, read from each live row's observation after every
                # step (below); a row stops once it has crossed `horizon` of them.
                reached = crossed >= horizon
            new = ~done & (term | reached)
            new_term = new & term
            new_turn = new & ~term
            if new_term.any():
                util = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
                value[new_term] = np.where(sides[new_term] == int(ts.Player.US), util[new_term], -util[new_term])
            active = ~done & ~new
            # Only rows still playing, and rows whose turn just ended (for their critic read), go
            # through the network: a finished playout's observation is never copied to the device.
            rows = np.flatnonzero(active | new_turn)
            if rows.size == 0:
                done |= new
                break
            dps = np.asarray(runner.get_decision_players(), dtype=np.int64)
            obs_all = np.asarray(runner.get_observations(), dtype=np.float32)
            masks_all = np.asarray(runner.get_action_masks())
            # While most rows still play, the whole batch goes to the device and is indexed there (a
            # host-side gather copied tens of MB per ply); once fewer than half do, only they are
            # copied. The masked argmax runs on the device.
            with torch.no_grad():
                if rows.size * 2 > n:
                    ridx = torch.from_numpy(rows).to(device)
                    obs_t = torch.from_numpy(obs_all).to(device).index_select(0, ridx)
                    mask_t = torch.from_numpy(masks_all).to(device).index_select(0, ridx)
                else:                                 # fewer than half still playing: gather here
                    obs_t = torch.from_numpy(obs_all[rows]).to(device)
                    mask_t = torch.from_numpy(masks_all[rows]).to(device)
                lg_t, v_t, _ = model(obs_t, mask_t)
                arg_rows = lg_t.argmax(dim=-1).cpu().numpy()
                v = v_t.float().reshape(-1).cpu().numpy()
            rows_total += int(rows.size)
            pos[rows] = np.arange(rows.size)
            # The critic's value at the turn's end, from the decider's side: the forward's own value
            # where the decider is the side to act there, its own view extracted where it is not.
            nt = np.flatnonzero(new_turn)
            if nt.size:
                same = dps[nt] == sides[nt]
                value[nt[same]] = v[pos[nt[same]]]
                other = nt[~same]
                if other.size:
                    sts = [runner.get_state(int(i)) for i in other]
                    o = np.stack([np.asarray(ts.extract_observation_features(st, ts.Player(int(sides[i])), feats),
                                             dtype=np.float32) for st, i in zip(sts, other)])
                    m = np.stack([np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8) for st in sts])
                    _lg2, v2 = forward(model, o, m, device)
                    value[other] = v2
                    rows_total += int(other.size)
            done |= new
            if done.all():
                break
            acts = np.zeros(n, dtype=np.int64)       # finished rows: refused or not, ignored
            act_rows = np.flatnonzero(active)
            acts[act_rows] = arg_rows[pos[act_rows]]
            if guard:
                # Only a row at DEFCON 2 at a guarded decision, read from its observation, copies
                # its state out for the probe; more than half a game is played at DEFCON 2.
                probe = act_rows[guard_rows(obs_all[act_rows])]
                top = None
                for i in probe:
                    i = int(i)
                    if top is None:
                        top = lg_t.topk(min(4, int(lg_t.shape[-1])), dim=-1).indices.cpu().numpy()
                    st = runner.get_state(i)
                    who = ts.Player(int(dps[i]))
                    for alt in top[int(pos[i])]:
                        alt = int(alt)
                        if masks_all[i, alt] and not loses_now(st, alt, who):
                            acts[i] = alt
                            break
            if lookahead > 0 or nested is not None:
                mine = active & (turns == start_turn) & (dps == sides)
                look_rows = mine & (look_left > 0)
                nest_rows = (mine & off_policy & (np_rng.random(n) < nested_p)) if nested is not None \
                    else np.zeros(n, dtype=bool)
                subs: List[Candidate] = []
                owners: List[int] = []
                for i in np.flatnonzero(look_rows | nest_rows):
                    i = int(i)
                    st = runner.get_state(i)
                    if not priceable(st, look_cap):
                        continue
                    legal = [int(a) for a in np.flatnonzero(masks_all[i])]
                    if nest_rows[i] and nested is not None:
                        lg_i = lg_t[int(pos[i])].float().cpu().numpy()
                        z = lg_i[legal] - lg_i[legal].max()
                        pz = np.exp(z) / np.exp(z).sum()
                        nested.append(Nested(part[i][0], part[i][1],
                                             Candidate(game=-1, choice=-1, state=st.clone(), options=legal,
                                                       probs=[float(x) for x in pz], taken=int(acts[i]))))
                    if look_rows[i]:
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
            if horizon is not None:
                # The boundary count after the step, from the new observations of the live rows.
                live = np.flatnonzero(~done & ~np.asarray(runner.get_terminals(), dtype=bool))
                if live.size:
                    obs_live = np.asarray(runner.get_observations(), dtype=np.float32)
                    turns_live = np.asarray(runner.get_turns(), dtype=np.int64)
                    dps_live = np.asarray(runner.get_decision_players(), dtype=np.int64)
                    now = ar_keys_from_obs(obs_live[live], turns_live[live], dps_live[live])
                    moved = (now != keys[live]).any(axis=1)
                    keys[live[moved]] = now[moved]
                    crossed[live[moved]] += 1
        else:
            raise RuntimeError(f"{int((~done).sum())} playouts still running after {max_plies} plies")
        for (ci, oi, w, _st), v_ in zip(part, value):
            vals = cands[ci].values
            assert vals is not None
            vals[oi, w] = v_
    for c in cands:                          # a world with any option missing is dropped whole
        assert c.values is not None
        c.values = c.values[:, ~np.isnan(c.values).any(axis=0)]
    return rows_total


