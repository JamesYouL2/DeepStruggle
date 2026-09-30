"""Engine-labelled card x board targets (P30), shared by the probe tools and the auxiliary target.

Every card in the mover's hand is labelled with quantities that depend on that card *and* the board:

* **T2, ops arithmetic** (4 numbers, from the observation's own slots -- the engine's definitions):
  with the card's printed Ops, the countries the mover could bring under control (it can place
  there, does not control it, needs no more influence points than the card has Ops; `board[24]`),
  the same for battlegrounds, and the best coup success chance over countries it can coup, overall
  and among battlegrounds (a die d succeeds when d + Ops > 2 x stability). Ops modifiers are not
  applied. None for 0-Ops cards.
* **T3, the event can fire**: `CardHandlers.can_trigger_event` for the player it fires for -- the
  card's owner for a side card (an opponent's card played for Ops fires for the opponent), the
  mover for a neutral one.
* **T4, what the event does** (12 numbers): the event fired on a clone for that player, and the
  change read from the mover's observation before and after -- VP, DEFCON, the six regional
  scoring margins, battlegrounds controlled by each side, total influence of each side. Averaged
  over the given engine seeds. None for an event that stops for a further decision.

`research/log/P30_card_board_targets.md` has what these showed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts

N_COUNTRIES, BOARD_W, N_CARDS, CARD_W = 84, 26, 110, 14
CARD_OFFSET = N_COUNTRIES * BOARD_W
GLOBAL_OFFSET = CARD_OFFSET + N_CARDS * CARD_W
MY_HAND = 1
DECISION_TYPE = GLOBAL_OFFSET + 72
CHINA = 6
MAX_HAND = 12
T2_NAMES = ("countries to control", "battlegrounds to control", "best coup chance", "best battleground coup chance")
T4_NAMES = ("VP", "DEFCON", "margin Europe", "margin Asia", "margin Middle East", "margin Africa",
            "margin Central America", "margin South America", "my battlegrounds", "their battlegrounds",
            "my influence", "their influence")

_INFO = {c: ts.CardData.get_card_info(c) for c in range(1, N_CARDS + 1)}
OPS = np.array([0] + [int(_INFO[c]["ops"]) for c in range(1, N_CARDS + 1)], dtype=np.int64)
SIDE = {c: str(_INFO[c]["side"]) for c in range(1, N_CARDS + 1)}
SCORING = sorted(c for c in range(1, N_CARDS + 1) if _INFO[c]["is_scoring"])

#: The auxiliary target's 16 outputs per card, T2 then T4, and their scales: the mean and standard
#: deviation over held cards of self-play at E6-12-44@560M / E6-03-44@560M
#: (`data/datasets/card_board/e6_12_03_560M.npz`). Fixed rather than running, so a checkpoint's
#: head means the same thing whenever it was trained.
AUX_DIM = 16
AUX_MEAN = np.array([19.689, 3.978, 0.857, 0.817,
                     -0.625, -0.015, -0.031, -0.029, -0.029, -0.006, -0.024, -0.014, -0.029, 0.022, 0.152, 0.371],
                    dtype=np.float32)
AUX_SCALE = np.array([10.403, 3.245, 0.271, 0.290,
                      3.966, 0.337, 0.800, 0.716, 0.749, 0.348, 0.602, 0.322, 0.317, 0.302, 0.987, 1.086],
                     dtype=np.float32)


def _summary(obs: np.ndarray) -> np.ndarray:
    """The quantities T4 differences, from one observation (mover's side)."""
    b = obs[:CARD_OFFSET].reshape(N_COUNTRIES, BOARD_W)
    bg = b[:, 4] > 0.5
    return np.concatenate([
        (obs[GLOBAL_OFFSET + 64:GLOBAL_OFFSET + 70] * 20.0),
        [float(((b[:, 5] > 0.5) & bg).sum()), float(((b[:, 6] > 0.5) & bg).sum()),
         float((b[:, 0] * 10.0).sum()), float((b[:, 1] * 10.0).sum())]])


def t2(obs: np.ndarray, card: int) -> Optional[np.ndarray]:
    ops = int(OPS[card])
    if ops <= 0:
        return None
    b = obs[:CARD_OFFSET].reshape(N_COUNTRIES, BOARD_W)
    can_place = b[:, 19] > 0.5
    mine = b[:, 5] > 0.5
    bg = b[:, 4] > 0.5
    deficit = np.rint(b[:, 24] * 5.0)
    stab = np.rint(b[:, 3] * 5.0)
    can_coup = b[:, 21] > 0.5
    reach = can_place & ~mine & (deficit > 0) & (deficit <= ops)
    p = np.clip(6.0 - 2.0 * stab + ops, 0.0, 6.0) / 6.0
    best = float(p[can_coup].max()) if can_coup.any() else 0.0
    best_bg = float(p[can_coup & bg].max()) if (can_coup & bg).any() else 0.0
    return np.array([float(reach.sum()), float((reach & bg).sum()), best, best_bg], dtype=np.float32)


def _ctx(st: "ts.GameState") -> Tuple[int, int, int, int]:
    c = st.ctx()
    return int(c.decision_type), int(c.decision_player), int(c.pending_op_card), int(c.remaining_steps)


def t3_t4(st: "ts.GameState", mover: "ts.Player", card: int, base_obs: np.ndarray,
          seeds: Sequence[int]) -> Tuple[Optional[bool], Optional[np.ndarray]]:
    if card == CHINA:
        return None, None
    side = SIDE[card]
    fire = ts.Player.US if side == "US" else ts.Player.USSR if side == "USSR" else mover
    if not bool(ts.CardHandlers.can_trigger_event(st, card, fire)):
        return False, None
    sign = 1.0 if mover == ts.Player.US else -1.0
    base = _summary(base_obs)
    c0 = _ctx(st)
    deltas: List[np.ndarray] = []
    for s in seeds:
        x = st.clone()
        x.rng_state = int(s)
        vp0, dc0 = int(x.victory_points), int(x.defcon)
        ts.CardHandlers.trigger_event(x, card, fire, 0)
        c1 = _ctx(x)
        if c1 != c0 and c1[0] != 0:
            return True, None                         # stops for a choice: not labelled
        after = _summary(np.asarray(ts.extract_observation(x, mover)))
        deltas.append(np.concatenate([[sign * (int(x.victory_points) - vp0), float(int(x.defcon) - dc0)],
                                      after - base]).astype(np.float32))
    return True, np.mean(deltas, 0)


def held_cards(obs: np.ndarray) -> List[int]:
    rows = obs[CARD_OFFSET:GLOBAL_OFFSET].reshape(N_CARDS, CARD_W)
    return [c for c in range(1, N_CARDS + 1) if rows[c - 1, MY_HAND] > 0.5][:MAX_HAND]


def label(st: "ts.GameState", mover: "ts.Player", obs: np.ndarray, seeds: Sequence[int]) -> Dict[str, np.ndarray]:
    """All targets for the mover's held cards, padded to MAX_HAND (card id 0 = none)."""
    cards = np.zeros(MAX_HAND, dtype=np.int16)
    y2 = np.zeros((MAX_HAND, 4), dtype=np.float32)
    y3 = np.zeros(MAX_HAND, dtype=np.float32)
    y4 = np.zeros((MAX_HAND, 12), dtype=np.float32)
    m2 = np.zeros(MAX_HAND, dtype=bool)
    m3 = np.zeros(MAX_HAND, dtype=bool)
    m4 = np.zeros(MAX_HAND, dtype=bool)
    for j, c in enumerate(held_cards(obs)):
        cards[j] = c
        v2 = t2(obs, c)
        if v2 is not None:
            y2[j], m2[j] = v2, True
        can, v4 = t3_t4(st, mover, c, obs, seeds)
        if can is not None:
            y3[j], m3[j] = float(can), True
        if v4 is not None:
            y4[j], m4[j] = v4, True
    return {"cards": cards, "y2": y2, "y3": y3, "y4": y4, "m2": m2, "m3": m3, "m4": m4}


def aux_label(st: "ts.GameState", mover: "ts.Player", obs: np.ndarray, seed: int
              ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The auxiliary target for one position: (cards (12,), standardised targets (12, 16), mask
    (12, 16)), or None when no held card has any label. One engine seed, for speed."""
    lab = label(st, mover, obs, (seed,))
    mask = np.concatenate([np.repeat(lab["m2"][:, None], 4, 1), np.repeat(lab["m4"][:, None], 12, 1)], 1)
    if not mask.any():
        return None
    y = (np.concatenate([lab["y2"], lab["y4"]], 1) - AUX_MEAN) / AUX_SCALE
    return lab["cards"], np.where(mask, y, 0.0).astype(np.float32), mask
