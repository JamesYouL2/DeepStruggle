"""What the engine does, as a digest: the leaderboard's definition of an engine epoch.

An Elo scale holds only while every game in it is played under the same rules, so an epoch must
say which engine builds it accepts. A hash of the sources (tools/lib/engine_fingerprint.py) is the
wrong test for that -- a comment, a refactor or a change to the WebAssembly page moves it without
changing one game -- so each new build used to be added to epochs.json by hand, once a recorded
pairing was seen to replay identically. This is that check, made exact and automatic: seeded random
games, hashing everything a player is shown and everything it does, through every path a
leaderboard player takes.

The games are random, except that a move which ends the game at once, or takes DEFCON down to 2
or below, is passed over while another is legal (`_choose`) -- uniformly random games mostly end in
turn 1 to 3 by DEFCON, and would leave the Mid and Late War untested. Of the runner's games about
one in five reaches turn 8 and some reach final scoring.

Four sections, each a sha256, so a mismatch says where the engines part:

* `runner` -- the batch runner, which plays leaderboard games (tools/lib/batch_tournament.py):
  each decision's observation row and legal mask, under every combination of the two action views
  and the two observation feature sets for each side, the decider, the random legal action taken,
  and at the end each game's result, VP, turn and why it ended.
* `state` -- the same through the single-position API the searchers and the page use
  (`get_flat_action_mask` in both views, `extract_observation_features` from both sides).
* `rollout` -- the rollout root's batch (`RolloutBatch`) on positions from the runner's games,
  under each stop and leaf setting and keyed chance (since version 2), the keyed batch also on
  positions at a Space Race box 4 headline (since version 3): playout observations and masks, leaf
  values, each position's decision.
* `mcts` -- the C++ search tree (`BatchedSearch`) on the same positions: its leaves, the roots'
  statistics and the generator state it hands back.

The searches are driven by a stand-in network, a function of the bytes it is shown: each row's
logits and value come from a generator seeded with a hash of the row, as integers and exact
fractions, so the digest depends on what the engine computed and on no floating-point arithmetic
that could differ from one CPU or BLAS to another.

    PYTHONPATH=.:build/release python tools/lib/engine_behaviour.py   # the "behaviour" of epochs.json
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import sys
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

import ts_engine as ts

from tools.lib.leaderboard import BEHAVIOUR_VERSION, Behaviour

_BASE_SEED = 880_000
#: The observation feature set the digest's "with features" paths use: the bits that existed when the
#: digest's version was defined (OPS_BUDGET), not ts.OBS_FEATURES_ALL. A bit added later changes no game
#: a player without it plays, but under ALL it would move every section and read as a new engine epoch
#: (HEADLINE_REVEAL, 2026-10-20, did exactly that); a new bit is covered by its own tests instead.
_FEATURES = int(ts.OBS_FEATURE_OPS_BUDGET)
#: 16 combinations of the two views and two feature sets for each side, 32 games of each.
_RUNNER_ENVS = 512
_STATE_GAMES = 64
_MAX_STEPS = 10_000
#: Every this many runner steps, one live game's position is kept for the searches.
_POSITION_EVERY = 11
_POSITIONS = 48
#: Runner positions at a Space Race box 4 headline (the decider holds box 4 alone and has seen the
#: opponent's headline), kept for the keyed batch: its box4 acts only there, and the positions taken
#: every _POSITION_EVERY steps never land on one, so without them box4 moved no section.
_BOX4_POSITIONS = 4
#: (stop, leaf, features, rule, keyed) of each rollout batch. The keyed one is the deployed root's
#: (earlier:own:keyed:aligned:box4), whose draws come from bindings/keyed_chance.hpp, not the stream;
#: it also plays the _BOX4_POSITIONS.
_ROLLOUTS: Tuple[Tuple[str, str, int, str, bool], ...] = (
    ("boundaries", "mover", 0, "argmax", False),
    ("turn", "own", _FEATURES, "z2", False),
    ("later", "decider", _FEATURES, "kl0.5", False),
    ("earlier", "own", _FEATURES, "z2", True),
)
_ROLLOUT_K, _ROLLOUT_WORLDS, _ROLLOUT_HORIZON = 4, 4, 2
#: (features, merged influence, fpu reduction, groups) of each search.
_SEARCHES: Tuple[Tuple[int, bool, float, int], ...] = ((0, False, 0.0, 1), (_FEATURES, True, 0.25, 2))
_SIMULATIONS = 24


class BehaviourError(RuntimeError):
    """The engine did something a digest cannot be taken of: refused a legal action, offered no
    legal action, or did not finish a game."""


class _Stream:
    """A sha256 of typed values, each with its type and shape, so no two sequences collide."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()

    def ints(self, *xs: int) -> None:
        self._h.update(b"i" + struct.pack(f"<I{len(xs)}q", len(xs), *(int(x) for x in xs)))

    def floats(self, *xs: float) -> None:
        self._h.update(b"f" + struct.pack(f"<I{len(xs)}d", len(xs), *(float(x) for x in xs)))

    def text(self, s: str) -> None:
        b = s.encode()
        self._h.update(b"s" + struct.pack("<I", len(b)) + b)

    def array(self, a: npt.ArrayLike) -> None:
        x = np.ascontiguousarray(a)
        self._h.update(b"a" + x.dtype.str.encode() + struct.pack(f"<I{x.ndim}q", x.ndim, *x.shape))
        self._h.update(x.tobytes())

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def stand_in(obs: npt.NDArray[np.float32], masks: npt.NDArray[np.uint8]
             ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """(logits, probabilities, values) for these rows: integer logits below 2^24, so exact in
    float32; the probabilities, those logits over the legal moves divided by their sum; a value in
    thousandths of [-1, 1]. Each row's from a generator seeded with a hash of that row alone."""
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    masks = np.ascontiguousarray(masks, dtype=np.uint8)
    n, width = masks.shape
    logits = np.zeros((n, width), dtype=np.float32)
    probs = np.zeros((n, width), dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    for r in range(n):
        key = hashlib.blake2b(obs[r].tobytes() + masks[r].tobytes(), digest_size=8).digest()
        g = np.random.default_rng(int.from_bytes(key, "little"))
        raw = g.integers(1, 1 << 24, size=width, dtype=np.int64)
        logits[r] = raw
        legal = np.where(masks[r] != 0, raw, 0)
        total = int(legal.sum())
        if total:
            probs[r] = legal / total
        values[r] = int(g.integers(-1000, 1001)) / 1000
    return logits, probs, values


def _choose(st: ts.GameState, legal: List[int], merged: bool) -> int:
    """The first of `legal` (already shuffled) that neither ends the game at once nor takes DEFCON
    down to 2 or below, tried on a copy stepped as the runner steps (auto-advance: the dice and any
    decision with one legal move); the first of them when every move does. A merged-view influence
    placement cannot do either, and is not decoded in the plain view."""
    for a in legal:
        if merged and ts.ActionMask.is_merged_influence_action(st, a):
            return a
        after = st.clone()
        if not ts.Engine.try_step(after, ts.decode_flat_action(after, a), True):
            continue
        if not ts.Engine.is_terminal(after) and not (after.defcon < st.defcon and after.defcon <= 2):
            return a
    return legal[0]


def _runner(out: _Stream, positions: List[ts.GameState], box4: List[ts.GameState]) -> None:
    n = _RUNNER_ENVS
    runner = ts.VectorizedBatchRunner(n, _BASE_SEED)
    for i in range(n):
        runner.reset_game(i, _BASE_SEED + i)
    runner.refresh_all()
    merged_us, merged_ussr = [bool(i & 1) for i in range(n)], [bool(i & 2) for i in range(n)]
    runner.set_merged_influence(merged_us, merged_ussr)
    runner.set_obs_features([_FEATURES if i & 4 else 0 for i in range(n)],
                            [_FEATURES if i & 8 else 0 for i in range(n)])
    pick = [random.Random(_BASE_SEED + i) for i in range(n)]
    active = np.ones(n, dtype=bool)
    for step in range(_MAX_STEPS):
        terminal = np.array(runner.get_terminals(), dtype=bool)
        for i in np.flatnonzero(active & terminal):
            st = runner.get_state(int(i))
            out.ints(step, int(i), int(st.victory_points), int(st.turn), int(st.action_round))
            out.floats(float(ts.Engine.get_terminal_utility(st)))
            out.text(ts.game_ending_reason(st))
        active &= ~terminal
        if not active.any():
            return
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        players = runner.get_decision_players()
        actions = np.zeros(n, dtype=np.int32)
        for i in np.flatnonzero(active):
            legal = [int(x) for x in np.flatnonzero(masks[i])]
            if not legal:
                raise BehaviourError(f"runner game {i}: no legal action at step {step}")
            pick[i].shuffle(legal)
            merged = merged_us[i] if players[i] == int(ts.Player.US) else merged_ussr[i]
            st = runner.get_state(int(i))
            if len(box4) < _BOX4_POSITIONS and ts.revealed_headline(st, st.ctx().decision_player):
                box4.append(st.clone())
            actions[i] = _choose(st, legal, merged)
            out.ints(step, int(i), int(players[i]), int(actions[i]))
            out.array(obs[i])
            out.array(masks[i])
        if step % _POSITION_EVERY == 0 and len(positions) < _POSITIONS:
            live = np.flatnonzero(active)
            positions.append(runner.get_state(int(live[(step // _POSITION_EVERY) % len(live)])))
        took = runner.step_flat_all(actions.tolist(), auto_advance=True)
        refused = [i for i in np.flatnonzero(active) if not took[i]]
        if refused:
            raise BehaviourError(f"runner games {refused}: the engine refused a legal action at step {step}")
    raise BehaviourError(f"runner games {np.flatnonzero(active).tolist()} unfinished after {_MAX_STEPS} steps")


def _state_api(out: _Stream) -> None:
    for g in range(_STATE_GAMES):
        seed = _BASE_SEED + 10_000 + g
        pick = random.Random(seed)
        st = ts.GameState()
        ts.Engine.init_game(st, seed)
        for ply in range(_MAX_STEPS):
            if ts.Engine.is_terminal(st):
                break
            ctx = st.ctx()
            if ctx.decision_player == ts.Player.NONE:
                # A chance node, drained. try_step returns whether it applied; step raises.
                if not ts.Engine.try_step(st, ts.MicroAction(ctx.decision_type, 0, 0, 0)):
                    raise BehaviourError(f"state game {g}: a chance node refused at ply {ply}")
                continue
            plain = ts.get_flat_action_mask(st, False)
            out.ints(g, ply, int(ctx.decision_type), int(ctx.decision_player))
            out.array(plain)
            out.array(ts.get_flat_action_mask(st, True))
            for side in (ts.Player.US, ts.Player.USSR):
                out.array(ts.extract_observation_features(st, side, _FEATURES))
            legal = [int(x) for x in np.flatnonzero(plain)]
            if not legal:
                raise BehaviourError(f"state game {g}: no legal action at ply {ply}")
            pick.shuffle(legal)
            action = _choose(st, legal, False)
            out.ints(action)
            if not ts.Engine.try_step(st, ts.decode_flat_action(st, action)):
                raise BehaviourError(f"state game {g}: the engine refused legal action {action} at ply {ply}")
        else:
            raise BehaviourError(f"state game {g} unfinished after {_MAX_STEPS} plies")
        out.ints(g, int(st.victory_points), int(st.turn), int(st.action_round))
        out.text(ts.game_ending_reason(st))


def _rollout(out: _Stream, positions: List[ts.GameState], box4: List[ts.GameState]) -> None:
    for c, (stop, leaf, features, rule, keyed) in enumerate(_ROLLOUTS):
        batch = ts.RolloutBatch(horizon=_ROLLOUT_HORIZON, shared_dice=True, guard=True, features=features,
                                seed=_BASE_SEED + c, stop=stop, leaf=leaf, keyed=keyed,
                                aligned=keyed, box4=keyed)
        added: List[Tuple[int, List[int], npt.NDArray[np.float32], npt.NDArray[np.uint8]]] = []
        for st in positions + box4 if keyed else positions:
            obs = ts.extract_observation_features(st, st.ctx().decision_player, features)
            mask = np.asarray(ts.get_flat_action_mask(st, False), dtype=np.uint8)
            logits = stand_in(obs[None, :], mask[None, :])[0][0]
            cands = ts.rollout_candidates(logits, mask, _ROLLOUT_K)
            out.ints(c, *cands)
            if len(cands) >= 2:
                added.append((batch.positions(), cands, logits, mask))
                batch.add(st, cands, _ROLLOUT_WORLDS)
        for _ in range(_MAX_STEPS):
            if batch.all_done():
                break
            obs_l, masks_l = batch.fill_live()
            out.array(obs_l)
            out.array(masks_l)
            batch.step(stand_in(obs_l, masks_l)[0])
        else:
            raise BehaviourError(f"rollout batch {c}: {batch.live()} playouts unfinished after {_MAX_STEPS} plies")
        if batch.refused():
            raise BehaviourError(f"rollout batch {c}: the engine refused a playout action in {batch.refused()} rows")
        obs_v, masks_v = batch.fill_leaves()
        out.array(obs_v)
        out.array(masks_v)
        if obs_v.shape[0]:
            batch.set_leaf_values(stand_in(obs_v, masks_v)[2])
        out.array(batch.values())
        for p, cands, logits, mask in added:
            d = ts.rollout_decide(batch, p, cands, logits, mask, rule)
            out.ints(c, p, int(d["pick"]), int(d["worlds"]))
            out.floats(*d["q"], *d["lead"], *d["lead_se"])


def _mcts(out: _Stream, positions: List[ts.GameState]) -> None:
    """Driven as ai/search/batched_mcts.py drives it: every group's roots selected before any is
    expanded, then the rounds taken group 0, 1, 0, 1, ..., each group's next leaves selected as soon
    as its last are expanded."""
    for c, (features, merged, fpu, groups) in enumerate(_SEARCHES):
        cs = ts.BatchedSearch(len(positions), 1.5, True, features, merged, fpu)
        cs.reset(positions, _SIMULATIONS, list(random.Random(_BASE_SEED + 20_000 + c).getstate()[1]), groups)

        def launch(g: int) -> Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
            k = cs.select_leaves(g)
            if k == 0:
                return None
            off = cs.group_offset(g)
            obs = np.array(cs.observations()[off:off + k])
            masks = np.array(cs.masks()[off:off + k])
            out.ints(c, g, k)
            out.array(obs)
            out.array(masks)
            _logits, probs, values = stand_in(obs, masks)
            return probs, values

        def finish(g: int, job: Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]) -> None:
            if job is not None:
                cs.expand_and_backup(g, job[0], job[1])

        jobs = [launch(g) for g in range(groups)]
        for g, job in enumerate(jobs):
            finish(g, job)
        queue: Deque[Tuple[int, Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]]] = deque(
            (g, launch(g)) for g in range(groups) if cs.group_has_budget(g))
        while queue:
            g, job = queue.popleft()
            finish(g, job)
            if cs.group_has_budget(g):
                queue.append((g, launch(g)))
        out.ints(*cs.mt_state())
        for i in range(len(positions)):
            terminal, mover, value_us, actions, priors, visits, sums = cs.root(i)
            out.ints(c, i, int(terminal), int(mover), *actions, *visits)
            out.floats(float(value_us), *priors, *sums)


def measure() -> Behaviour:
    """The digest of the engine this process imported."""
    positions: List[ts.GameState] = []
    box4: List[ts.GameState] = []
    runner, state, rollout, mcts = _Stream(), _Stream(), _Stream(), _Stream()
    _runner(runner, positions, box4)
    if len(box4) < _BOX4_POSITIONS:
        raise BehaviourError(f"the runner's games reached {len(box4)} Space Race box 4 headlines, "
                             f"not {_BOX4_POSITIONS}")
    _state_api(state)
    _rollout(rollout, positions, box4)
    _mcts(mcts, positions)
    return {"version": BEHAVIOUR_VERSION, "runner": runner.hexdigest(), "state": state.hexdigest(),
            "rollout": rollout.hexdigest(), "mcts": mcts.hexdigest()}


def main(argv: Optional[List[str]] = None) -> int:
    if argv if argv is not None else sys.argv[1:]:
        print("usage: engine_behaviour.py   (prints the imported engine's behaviour digest)", file=sys.stderr)
        return 2
    print(json.dumps(measure(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
