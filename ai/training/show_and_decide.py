"""P31, "show, then let training decide": the behaviour floor (1a) and scenario seeding (1b).

Both put rare situations in front of the learner without telling it what they are worth.

* **The floor (1a)** mixes the learner's behaviour policy with a uniform distribution over the legal
  options at play-mode decisions and at the non-country choices inside events:
  mu = (1 - eps) * pi + eps * uniform(legal). The stored log-prob is log mu, so PPO's ratio
  pi_theta / mu corrects for the extra exploration; the target policy is untouched.
* **Scenario seeding (1b)** forces, in a drawn fraction of games, the precursor of a skill chain as
  *environment*: when the US first holds a listed card at an action round's card play and its event
  can trigger, the card is played for its event (and, for a card that asks for a region, the region
  is drawn uniformly). The forced rows are stored with learner = 0, so they keep the GAE recursion
  and train the critic but receive no policy gradient -- E7-11-44 showed a scripted action trained
  as the policy's own gets adopted whatever it is worth. Everything after the forced play is the
  policy's own.

No card list enters the trainer's logic: a scenario is a card and whether a region choice follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder as _A

PLAY_MODE_LO = _A.PLAY_MODE_OFFSET
PLAY_MODE_HI = _A.PLAY_MODE_OFFSET + 5
EVENT_SLOT = _A.PLAY_MODE_OFFSET
#: The branch block: event branches, CONFIRM_DONE, DEFCON values and regions (200..219).
BRANCH_LO = _A.BRANCH_OFFSET
REGION_LO = _A.REGION_OFFSET
ACTIONS = _A.FLAT_ACTION_SIZE


def play_mode_rows(masks: torch.Tensor) -> torch.Tensor:
    """Rows whose decision is a play-mode choice for a chosen card: every legal action is a
    play-mode slot, and event or space is among them (the deferred Ops-mode choice, which offers
    only the three Ops slots, is not one)."""
    masks = masks.bool()
    legal = masks[:, PLAY_MODE_LO:PLAY_MODE_HI]
    return legal[:, :2].any(dim=1) & (masks.sum(dim=1) == legal.sum(dim=1))


def event_choice_rows(masks: torch.Tensor) -> torch.Tensor:
    """Rows whose whole legal set lies in the branch block -- an event's branch, a DEFCON value or
    a region -- with at least two options. Country targets (POINT_NODE) are not among them."""
    m = masks.to(torch.int32)
    n = m.sum(dim=1)
    return (m[:, BRANCH_LO:].sum(dim=1) == n) & (n >= 2)


def floor_rows(masks: torch.Tensor) -> torch.Tensor:
    """The decisions the floor applies to (P31 1a, owner 2026-10-06: play mode plus the
    non-country choices inside events)."""
    return play_mode_rows(masks) | event_choice_rows(masks)


def apply_floor(actions: torch.Tensor, log_probs_pi: torch.Tensor, masks: torch.Tensor,
                rows: torch.Tensor, eps: float, generator: Optional[torch.Generator] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample from mu = (1 - eps) * pi + eps * uniform(legal) on `rows`, given actions already
    sampled from pi.

    With probability eps a row's action is replaced by a uniform legal one; otherwise it stays the
    pi sample. That is exactly a draw from the mixture. Returns (actions, log mu(action) on `rows`
    and log pi elsewhere, the rows that took the uniform draw).

    `log_probs_pi` is the full [N, A] canonical log-softmax, because which action a row ends up
    with is only known after the draw.
    """
    n = actions.shape[0]
    legal_f = masks.to(torch.float32)
    coin = torch.rand(n, device=actions.device, generator=generator) < eps
    uniform = torch.multinomial(legal_f, 1, generator=generator).squeeze(1)
    took = rows.bool() & coin
    actions = torch.where(took, uniform, actions)
    lp_pi = log_probs_pi.gather(1, actions.unsqueeze(1)).squeeze(1)
    n_legal = legal_f.sum(dim=1).clamp(min=1.0)
    mix = torch.log((1.0 - eps) * lp_pi.exp() + eps / n_legal)
    return actions, torch.where(rows.bool(), mix, lp_pi), took


def floor_eps(base: float, steps: int, start: int, anneal_from: Optional[int],
              anneal_steps: int) -> float:
    """The floor's eps at `steps`: 0 before `start`, `base` after, and linear to 0 over
    `anneal_steps` from `anneal_from` when that is set."""
    if base <= 0.0 or steps < start:
        return 0.0
    if anneal_from is None or steps < anneal_from:
        return base
    if anneal_steps <= 0:
        return 0.0
    return base * max(0.0, 1.0 - (steps - anneal_from) / float(anneal_steps))


# ------------------------------------------------------------------ scenario seeding (1b)

@dataclass(frozen=True)
class Scenario:
    card: int            # card id (1..110)
    region: bool         # the event then asks for a region, drawn uniformly


#: A scenario is data: the card whose event is forced, and whether a region choice follows.
SCENARIOS: Dict[str, Scenario] = {
    "subs": Scenario(card=41, region=False),       # Nuclear Subs
    "chernobyl": Scenario(card=94, region=True),   # Chernobyl
}

#: Per-env stage of a forced play: 0 nothing pending, 1 card chosen (play mode next), 2 event
#: chosen (region next).
_IDLE, _MODE, _REGION = 0, 1, 2


class ScenarioSeeder:
    """Forces listed precursors as environment in a drawn fraction of games (P31 1b).

    In a seeded game every listed card is forced once, the first time the US holds it at an action
    round's own card play (not a headline, not a card chosen inside another event) with its event
    able to trigger. Whoever controls the US -- the learner or a frozen pool opponent -- the play is
    the environment's.
    """

    def __init__(self, names: Sequence[str], frac: float, num_envs: int, start_step: int = 0,
                 seed: int = 24680) -> None:
        unknown = [n for n in names if n not in SCENARIOS]
        if unknown:
            raise ValueError(f"unknown scenario(s) {unknown}; known: {sorted(SCENARIOS)}")
        if not names:
            raise ValueError("scenario seeding needs at least one scenario")
        if not 0.0 < frac <= 1.0:
            raise ValueError("--seed-frac must be in (0, 1]")
        for n in names:
            name = ts.CardData.get_card_info(SCENARIOS[n].card)["name"]
            expect = {"subs": "Nuclear Subs", "chernobyl": "Chernobyl"}[n]
            if name != expect:
                raise RuntimeError(f"scenario {n!r} names card {SCENARIOS[n].card} = {name!r}, "
                                   f"not {expect!r}: the card table has moved")
        self.scenarios: List[Scenario] = [SCENARIOS[n] for n in names]
        self.names = list(names)
        self.frac = float(frac)
        self.start_step = int(start_step)
        self.rng = np.random.default_rng(seed)
        self.seeded = np.zeros(num_envs, dtype=bool)
        self.done: List[Set[int]] = [set() for _ in range(num_envs)]
        self.stage = np.zeros(num_envs, dtype=np.int8)
        self.card = np.zeros(num_envs, dtype=np.int16)
        self.games = np.zeros(2, dtype=np.int64)                      # (seeded, not) drawn
        self.forced: Dict[str, int] = {n: 0 for n in names}           # forced plays per scenario

    def draw(self, envs: np.ndarray, steps: int) -> None:
        """At a game start: decide afresh whether each of `envs` is seeded."""
        envs = np.asarray(envs, dtype=np.int64)
        if envs.size == 0:
            return
        pick = self.rng.random(envs.size) < self.frac
        if steps < self.start_step:
            pick[:] = False
        self.seeded[envs] = pick
        self.stage[envs] = _IDLE
        self.card[envs] = 0
        for e in envs:
            self.done[int(e)] = set()
        self.games += (int(pick.sum()), int((~pick).sum()))

    def apply(self, actions: torch.Tensor, masks_np: np.ndarray, dp: np.ndarray,
              runner: object, auto_advance: bool) -> np.ndarray:
        """Overwrite the forced rows' actions in place; return a bool mask of the rows forced.

        Whether a card choice is the action round's own card play is asked of the engine, not
        inferred: a clone of the state is stepped with the selection, and the card is forced only if
        that lands on a US play-mode decision offering the event. An action-round card choice with no
        resolving card is not always a card play -- Quagmire's discard is one -- and a heuristic for
        which is which would be a guess.
        """
        forced = np.zeros(actions.shape[0], dtype=bool)
        rows = np.flatnonzero(self.seeded & (np.asarray(dp) == int(ts.Player.US)))
        for r in rows:
            r = int(r)
            legal = np.flatnonzero(masks_np[r])
            stage = int(self.stage[r])
            if stage == _MODE:
                if not masks_np[r][EVENT_SLOT] or legal.max() >= PLAY_MODE_HI or legal.min() < PLAY_MODE_LO:
                    raise RuntimeError(f"env {r}: card {int(self.card[r])} was forced, but the next US "
                                       f"decision is not a play mode offering its event (legal {legal.tolist()})")
                actions[r] = EVENT_SLOT
                sc = next(s for s in self.scenarios if s.card == int(self.card[r]))
                self.stage[r] = _REGION if sc.region else _IDLE
                forced[r] = True
            elif stage == _REGION:
                if legal.size == 0 or legal.min() < REGION_LO:
                    raise RuntimeError(f"env {r}: card {int(self.card[r])}'s event was forced, but the next "
                                       f"US decision is not a region choice (legal {legal.tolist()})")
                actions[r] = int(self.rng.choice(legal))
                self.stage[r] = _IDLE
                forced[r] = True
            else:
                if legal.size == 0 or legal.max() >= PLAY_MODE_LO:
                    continue                                    # not a card choice
                todo = [s for s in self.scenarios
                        if s.card not in self.done[r] and masks_np[r][s.card - 1]]
                if not todo:
                    continue
                st = runner.get_state(r)                        # type: ignore[attr-defined]
                ctx = st.ctx()
                if (st.current_phase != ts.Phase.ACTION_ROUND or ctx.resolving_card != 0
                        or ctx.decision_type != ts.DecisionType.SELECT_CARD):
                    continue
                for s in todo:
                    if not ts.CardHandlers.can_trigger_event(st, s.card, ts.Player.US):
                        continue
                    sim = st.clone()
                    if not ts.Engine.try_step_flat(sim, s.card - 1, auto_advance):
                        continue
                    sc = sim.ctx()
                    if (sc.decision_type != ts.DecisionType.SELECT_PLAY_MODE
                            or sc.decision_player != ts.Player.US
                            or not ts.get_flat_action_mask(sim)[EVENT_SLOT]):
                        continue
                    actions[r] = s.card - 1
                    self.done[r].add(s.card)
                    self.card[r] = s.card
                    self.stage[r] = _MODE
                    self.forced[self.names[self.scenarios.index(s)]] += 1
                    forced[r] = True
                    break
        return forced
