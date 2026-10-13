"""Forced openings as environment: the network learns to play ON from openings it is not taught to choose.

`--forced-openings-frac f` (owner, 2026-10-13): at each game start, each side independently has, with
probability f, its setup played by one of its listed openings (`tools/lib/openings.py`
FORCED_OPENINGS_US / _USSR), drawn uniformly -- so some games force one side, some both, most neither.
The forced placements are the environment's: stored with learner = 0, no policy gradient, so the
policy's own setup is untouched and only what follows the opening is learned. The US's `us_marshall`
opening is conditional on its hand at setup: with Marshall Plan held it is played and Marshall Plan is
forced as the US headline of turn 1 (also as environment); without, one of the other US openings is
played. Whoever controls the side -- the learner or a frozen pool opponent -- the forced move is the
environment's.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

import ts_engine as ts
from tools.lib.openings import MARSHALL_PLAN, NODE_OFFSET, OPENINGS

_SIDES = (int(ts.Player.US), int(ts.Player.USSR))


class OpeningSeeder:
    def __init__(self, frac: float, us: Sequence[str], ussr: Sequence[str], num_envs: int,
                 seed: int = 31415) -> None:
        if not 0.0 < frac <= 1.0:
            raise ValueError("--forced-openings-frac must be in (0, 1]")
        for name, side in [(n, "US") for n in us] + [(n, "USSR") for n in ussr]:
            if name not in OPENINGS or side not in OPENINGS[name]:
                raise ValueError(f"opening {name!r} does not script the {side}")
        if not us and not ussr:
            raise ValueError("--forced-openings needs at least one opening")
        self.frac = float(frac)
        self.lists: Dict[int, List[str]] = {int(ts.Player.US): list(us), int(ts.Player.USSR): list(ussr)}
        self.rng = np.random.default_rng(seed)
        #: per env and side: the opening's name ('' = its own setup), its flat actions, the cursor
        self.name: Dict[int, List[str]] = {s: [""] * num_envs for s in _SIDES}
        self.script: Dict[int, List[List[int]]] = {s: [[] for _ in range(num_envs)] for s in _SIDES}
        self.cursor: Dict[int, np.ndarray] = {s: np.zeros(num_envs, dtype=np.int32) for s in _SIDES}
        self.resolved = {s: np.zeros(num_envs, dtype=bool) for s in _SIDES}
        self.headline_mp = np.zeros(num_envs, dtype=bool)
        self.games: Dict[str, int] = {}
        self.forced_moves = 0

    def draw(self, envs: np.ndarray) -> None:
        """At a game start: decide afresh, per side, whether each of `envs` plays a forced opening."""
        for e in np.asarray(envs, dtype=np.int64):
            e = int(e)
            self.headline_mp[e] = False
            for s in _SIDES:
                names = self.lists[s]
                pick = names[int(self.rng.integers(len(names)))] if names and self.rng.random() < self.frac else ""
                self.name[s][e] = pick
                self.script[s][e] = []
                self.cursor[s][e] = 0
                self.resolved[s][e] = False

    def _resolve(self, e: int, s: int, st: ts.GameState) -> None:
        """Fix the env's script for side s at its first setup decision (the hand is dealt by then)."""
        name = self.name[s][e]
        if name == "us_marshall":
            if ts.in_hand_of(st.get_card_location(MARSHALL_PLAN), ts.Player.US):
                self.headline_mp[e] = True
            else:
                others = [n for n in self.lists[s] if n != "us_marshall"]
                name = others[int(self.rng.integers(len(others)))] if others else ""
                self.name[s][e] = name
        side = "US" if s == int(ts.Player.US) else "USSR"
        self.script[s][e] = [NODE_OFFSET + c for c in OPENINGS[name][side]] if name else []
        self.resolved[s][e] = True
        key = f"{side}:{name or 'own'}" + (":headline_mp" if self.headline_mp[e] and s == int(ts.Player.US) else "")
        self.games[key] = self.games.get(key, 0) + 1

    def apply(self, actions: torch.Tensor, masks_np: np.ndarray, dp: np.ndarray, runner: object) -> np.ndarray:
        """Overwrite the forced rows' actions in place; return the bool mask of the rows forced."""
        forced = np.zeros(actions.shape[0], dtype=bool)
        for e in range(actions.shape[0]):
            s = int(dp[e])
            if s not in _SIDES:
                continue
            setup_pending = self.name[s][e] and (not self.resolved[s][e] or self.cursor[s][e] < len(self.script[s][e]))
            headline_pending = s == int(ts.Player.US) and self.headline_mp[e]
            if not setup_pending and not headline_pending:
                continue
            st = runner.get_state(e)                                       # type: ignore[attr-defined]
            if st.current_phase == ts.Phase.SETUP:
                if not setup_pending:
                    continue
                if not self.resolved[s][e]:
                    self._resolve(e, s, st)
                k = int(self.cursor[s][e])
                if k >= len(self.script[s][e]):
                    continue
                a = self.script[s][e][k]
                if not masks_np[e][a]:
                    raise RuntimeError(f"forced opening {self.name[s][e]!r}: placement {k} (flat {a}) is illegal here")
                actions[e] = a
                self.cursor[s][e] = k + 1
                forced[e] = True
                self.forced_moves += 1
            elif headline_pending and st.current_phase == ts.Phase.HEADLINE and int(st.turn) == 1 \
                    and int(st.ctx().decision_type) == int(ts.DecisionType.SELECT_CARD):
                a = MARSHALL_PLAN - 1                                      # card c is flat slot c - 1
                if masks_np[e][a]:
                    actions[e] = a
                    forced[e] = True
                    self.forced_moves += 1
                self.headline_mp[e] = False
            elif headline_pending and (int(st.turn) > 1 or st.current_phase == ts.Phase.ACTION_ROUND):
                self.headline_mp[e] = False                                # the headline has passed
        return forced
