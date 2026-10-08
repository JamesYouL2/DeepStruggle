"""Nuclear Subs and its follow-up: does the model coup battlegrounds once Subs is in effect, and what
is the event worth when it does?

Nuclear Subs (US) stops the US's battleground coups from lowering DEFCON for the rest of the turn.
Its value is entirely in that follow-up -- above all at DEFCON 2, where a battleground coup would
otherwise end the game -- so a paired playout whose continuation never makes those coups prices the
card as a turn's Ops thrown away. This plays, at a US Subs play-mode position, four branches of each
pair (`ai.eval.paired_playouts`: same redeal, same dice):

* ``alt``     -- the raw network's own choice there (its most probable non-event play);
* ``event``   -- Subs for its event, the model playing on unaided;
* ``next``    -- the event, then at the US's first later play-mode decision of the turn where a
  battleground coup is possible, a coup there: the card stays the model's choice, the target is
  its most probable battleground;
* ``all``     -- the same at every such decision for the rest of the turn.

and counts, per branch, the US's coups for the rest of the turn: how many, how many in
battlegrounds, and how many of those at DEFCON 2 under Subs (the coups only Subs makes possible).

``next`` and ``all`` are scripted, not searched: they bound the follow-up from one side as ``event``
does from the other, and a better follow-up (which card, which battleground, when) is worth more.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import ts_engine as ts

from ai.eval.paired_playouts import PolicyFn, apply_move, ar_key, decider, pair_start, play_safe
from bindings.action_encoder import ActionEncoder

NUCLEAR_SUBS = 41
EVENT = ActionEncoder.PLAY_MODE_OFFSET
OPS_COUP = ActionEncoder.PLAY_MODE_OFFSET + 3
NODE = ActionEncoder.NODE_OFFSET
BRANCHES = ("alt", "event", "next", "all")
COUNTS = ("coups", "bg_coups", "bg_coups_d2_subs", "forced")
_BG = np.zeros(ActionEncoder.FLAT_ACTION_SIZE, dtype=bool)
for _c in range(84):
    if ts.MapData.get_country_info(_c)["battleground"]:
        _BG[NODE + _c] = True


def is_us_coup_target(st: ts.GameState) -> bool:
    ctx = st.ctx()
    return (decider(st) == ts.Player.US and ctx.decision_type == ts.DecisionType.POINT_NODE
            and ctx.op_mode == ts.OpMode.COUP)


def bg_coup_open(st: ts.GameState) -> bool:
    """Whether choosing OPS_COUP here leads to a coup target choice with a battleground in it."""
    probe = st.clone()
    ts.Engine.step_flat(probe, OPS_COUP)
    if not is_us_coup_target(probe):
        return False
    return bool((np.asarray(ActionEncoder.get_legal_mask(probe)).astype(bool) & _BG).any())


class SubsWatch:
    """Watches the US for the rest of the starting turn; in the ``next`` / ``all`` branches, steers
    its play-mode decisions to a battleground coup while Subs is in effect."""

    def __init__(self, turns: Sequence[int], modes: Sequence[str]) -> None:
        self.turn = list(turns)
        self.mode = list(modes)
        self.live = [True] * len(turns)
        self.forcing = [False] * len(turns)      # a coup forced at play mode, its target next
        self.left = [0 if m in ("alt", "event") else (1 if m == "next" else 99) for m in modes]
        self.counts: List[Dict[str, int]] = [{k: 0 for k in COUNTS} for _ in turns]

    def wants(self, i: int) -> bool:
        return self.live[i]

    def steer(self, i: int, st: ts.GameState, mask: np.ndarray) -> np.ndarray:
        if int(st.turn) != self.turn[i] or ts.Engine.is_terminal(st):
            self.live[i] = False
            return mask
        if decider(st) != ts.Player.US or not st.has_flag(ts.EffectBits.NUCLEAR_SUBS_ACTIVE):
            return mask
        ctx = st.ctx()
        if (self.left[i] > 0 and ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE
                and mask[OPS_COUP] and bg_coup_open(st)):
            out = np.zeros_like(mask)
            out[OPS_COUP] = 1
            self.forcing[i] = True
            return out
        if self.forcing[i] and is_us_coup_target(st):
            bg = mask.astype(bool) & _BG
            if bg.any():
                return bg.astype(mask.dtype)
        return mask

    def seen(self, i: int, st: ts.GameState, action: int) -> None:
        if not self.live[i] or not is_us_coup_target(st) or not NODE <= action < NODE + 84:
            return
        c = self.counts[i]
        c["coups"] += 1
        if _BG[action]:
            c["bg_coups"] += 1
            if int(st.defcon) <= 2 and st.has_flag(ts.EffectBits.NUCLEAR_SUBS_ACTIVE):
                c["bg_coups_d2_subs"] += 1
        if self.forcing[i]:
            c["forced"] += 1
            self.forcing[i] = False
            self.left[i] -= 1


def subs_branches(positions: Sequence[ts.GameState], alts: Sequence[int], act: PolicyFn, pairs: int,
                  seed: int, chunk: int = 1536) -> List[Dict[str, Dict[str, List[float]]]]:
    """Per position, per branch: the mover's score in each pair ("score") and each count's value in
    each pair. Every position must be a US play-mode decision for Nuclear Subs with the event legal."""
    out: List[Dict[str, Dict[str, List[float]]]] = []
    per = len(BRANCHES) * pairs
    step = max(1, chunk // per)
    for lo in range(0, len(positions), step):
        hi = min(len(positions), lo + step)
        starts, movers, keys, turns, modes, index = [], [], [], [], [], []
        for p in range(lo, hi):
            st = positions[p]
            if decider(st) != ts.Player.US or st.ctx().decision_type != ts.DecisionType.SELECT_PLAY_MODE:
                raise ValueError(f"position {p} is not a US play-mode decision")
            for k in range(pairs):
                base = pair_start(st, k, seed + p)
                for b in BRANCHES:
                    starts.append(apply_move(base, alts[p] if b == "alt" else EVENT))
                    movers.append(ts.Player.US)
                    keys.append(ar_key(st))
                    turns.append(int(st.turn))
                    modes.append(b)
                    index.append((p, b))
        watch = SubsWatch(turns, modes)
        scores = play_safe(starts, movers, keys, act, seed + lo, watch=watch)
        res = [{b: {k: [] for k in ("score",) + COUNTS} for b in BRANCHES} for _ in range(lo, hi)]
        for j, ((p, b), s) in enumerate(zip(index, scores)):
            res[p - lo][b]["score"].append(s)
            for k in COUNTS:
                res[p - lo][b][k].append(float(watch.counts[j][k]))
        out += res
    return out
