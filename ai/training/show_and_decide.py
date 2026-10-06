"""P31, "show, then let training decide": the behaviour floor (1a) and scenario seeding (1b).

Both put rare situations in front of the learner without telling it what they are worth.

* **The floor (1a)** mixes the learner's behaviour policy with a uniform distribution over the legal
  options at play-mode decisions and at the non-country choices inside events:
  mu = (1 - eps) * pi + eps * uniform(legal). The stored log-prob stays log pi, so PPO's ratio and
  clip are the usual pi_theta / pi_old, and each floor sample's surrogate is weighted by
  pi_old / mu (detached, at most 1 / (1 - eps)) to correct for the extra exploration.
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


def floor_rows(masks: torch.Tensor, which: str = "play_mode_and_events") -> torch.Tensor:
    """The decisions the floor applies to (P31 1a, owner 2026-10-06: play mode plus the
    non-country choices inside events; `event_choices` for the latter only)."""
    if which == "event_choices":
        return event_choice_rows(masks)
    if which != "play_mode_and_events":
        raise ValueError(f"unknown floor rows {which!r}")
    return play_mode_rows(masks) | event_choice_rows(masks)


def apply_floor(actions: torch.Tensor, log_probs_pi: torch.Tensor, masks: torch.Tensor,
                rows: torch.Tensor, eps: float, generator: Optional[torch.Generator] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample from mu = (1 - eps) * pi + eps * uniform(legal) on `rows`, given actions already
    sampled from pi.

    With probability eps a row's action is replaced by a uniform legal one; otherwise it stays the
    pi sample. That is exactly a draw from the mixture. Returns (actions, log pi(action), the
    behaviour weight pi(action) / mu(action) -- 1 off the floor -- and the rows that took the
    uniform draw).

    The weight, not log mu, is what corrects for the floor. Storing log mu as PPO's old log-prob
    made the ratio pi_theta / mu ~ 0.25 for a rare action the floor drew -- far below the clip --
    so a positive advantage raised it and a negative one, clipped, left it alone: every floor draw
    could only push a rare action up (E7-26-44's entropy rose 0.32 -> 0.45 in 95M steps). With the
    ratio on pi_theta / pi_old and the weight applied to the surrogate outside the clip, both signs
    reach the action. The weight is at most 1 / (1 - eps), since mu >= (1 - eps) pi.

    `log_probs_pi` is the full [N, A] canonical log-softmax, because which action a row ends up
    with is only known after the draw.
    """
    n = actions.shape[0]
    rows = rows.bool()
    legal_f = masks.to(torch.float32)
    coin = torch.rand(n, device=actions.device, generator=generator) < eps
    uniform = torch.multinomial(legal_f, 1, generator=generator).squeeze(1)
    took = rows & coin
    actions = torch.where(took, uniform, actions)
    lp_pi = log_probs_pi.gather(1, actions.unsqueeze(1)).squeeze(1)
    n_legal = legal_f.sum(dim=1).clamp(min=1.0)
    lp_mu = torch.log((1.0 - eps) * lp_pi.exp() + eps / n_legal)
    weight = torch.where(rows, (lp_pi - lp_mu).exp(), torch.ones_like(lp_pi))
    return actions, lp_pi, weight, took


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


# ------------------------------------------------------------ applicable-event forcing (owner)

#: Card ids and the condition under which their event is plainly worth playing, from the side that
#: plays it (owner, 2026-10-06): Wargames at DEFCON 2 with a lead of 7+ VP (the event hands the
#: opponent 6 VP and ends the game), Arms Race ahead in military Ops, One Small Step behind in space.
APPLICABLE_EVENTS: Dict[str, int] = {"wargames": 100, "arms_race": 39, "one_small_step": 80,
                                     # Wargames at DEFCON 2 at ANY lead: its event then reaches the
                                     # branch (end the game / pass), where both outcomes are on offer
                                     "wargames_branch": 100}


def event_applicable(name: str, state: object, side: int) -> bool:
    """Whether `name`'s event is applicable for `side` (+1 US, -1 USSR) in `state`."""
    st = state
    us = side == int(ts.Player.US)
    if name == "wargames":
        return int(st.defcon) == 2 and side * int(st.victory_points) >= 7          # type: ignore[attr-defined]
    if name == "wargames_branch":
        return int(st.defcon) == 2                                                 # type: ignore[attr-defined]
    if name == "arms_race":
        mine, theirs = ((st.us_mil_ops, st.ussr_mil_ops) if us                      # type: ignore[attr-defined]
                        else (st.ussr_mil_ops, st.us_mil_ops))                       # type: ignore[attr-defined]
        return int(mine) > int(theirs)
    if name == "one_small_step":
        mine, theirs = ((st.us_space_track, st.ussr_space_track) if us              # type: ignore[attr-defined]
                        else (st.ussr_space_track, st.us_space_track))               # type: ignore[attr-defined]
        return int(mine) < int(theirs)
    raise ValueError(f"unknown applicable event {name!r}; known: {sorted(APPLICABLE_EVENTS)}")


class ApplicableEventForcer:
    """At a learner's play-mode decision for a listed card whose event is legal and applicable, play
    the event with probability `frac` (owner, 2026-10-06). The forced play is trained as the policy's
    own -- learner = 1, its stored log-prob log pi(EVENT) -- so the event rises where its advantage is
    positive and falls where it is not; PPO's clip bounds each step."""

    def __init__(self, names: Sequence[str], frac: float, start_step: int = 0, seed: int = 97531) -> None:
        unknown = [n for n in names if n not in APPLICABLE_EVENTS]
        if unknown:
            raise ValueError(f"unknown applicable event(s) {unknown}; known: {sorted(APPLICABLE_EVENTS)}")
        if not names:
            raise ValueError("applicable-event forcing needs at least one card")
        if not 0.0 < frac <= 1.0:
            raise ValueError("--force-event-frac must be in (0, 1]")
        expect = {"wargames": "Wargames", "arms_race": "Arms Race", "one_small_step": "One Small Step",
                  "wargames_branch": "Wargames"}
        if len({APPLICABLE_EVENTS[n] for n in names}) != len(names):
            raise ValueError(f"{list(names)} name one card twice; give each card one condition")
        for n in names:
            got = str(ts.CardData.get_card_info(APPLICABLE_EVENTS[n])["name"])
            if expect[n].lower() not in got.lower():
                raise RuntimeError(f"{n!r} names card {APPLICABLE_EVENTS[n]} = {got!r}: the card table has moved")
        self.by_card: Dict[int, str] = {APPLICABLE_EVENTS[n]: n for n in names}
        self.frac = float(frac)
        self.start_step = int(start_step)
        self.rng = np.random.default_rng(seed)
        self.applicable: Dict[str, int] = {n: 0 for n in names}   # learner plays, event legal and applicable
        self.forced: Dict[str, int] = {n: 0 for n in names}

    def apply(self, actions: torch.Tensor, masks: torch.Tensor, learner: np.ndarray, runner: object,
              steps: int) -> np.ndarray:
        """Overwrite the forced rows' actions with EVENT in place; return the bool mask of them."""
        forced = np.zeros(actions.shape[0], dtype=bool)
        if steps < self.start_step:
            return forced
        rows = (play_mode_rows(masks) & masks.bool()[:, EVENT_SLOT]).cpu().numpy() & np.asarray(learner, dtype=bool)
        for r in np.flatnonzero(rows):
            r = int(r)
            st = runner.get_state(r)                                        # type: ignore[attr-defined]
            ctx = st.ctx()
            name = self.by_card.get(int(ctx.pending_op_card))
            if name is None or ctx.decision_type != ts.DecisionType.SELECT_PLAY_MODE:
                continue
            if not event_applicable(name, st, int(ctx.decision_player)):
                continue
            self.applicable[name] += 1
            if self.rng.random() < self.frac:
                actions[r] = EVENT_SLOT
                self.forced[name] += 1
                forced[r] = True
        return forced
