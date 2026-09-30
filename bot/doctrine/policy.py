"""Doctrine's decision policy: struggler's value function, searched with ts_engine.

struggler's `StrategicPlayer` prices every option by hand -- a greedy influence plan, a coup's
six rolls, an event run in a sandbox copy of its Python engine. Here the C++ engine is the
sandbox: every option is played out on a clone of the (determinized) state and the resulting
board is priced with the ported evaluator. The shape of the search follows struggler:

* **One action round is the horizon.** An option is played until the next card choice, the
  turn's cleanup, or the end of the game, and the position there is valued.
* **Structural choices are searched; placements are greedy.** A choice of card, of play mode,
  of Ops mode or of an event branch is decided by playing every alternative out. Influence
  points, coup and realignment targets and event placements are chosen one at a time by the
  value right after them -- struggler's point-by-point plan, and what keeps the search
  polynomial.
* **Dice are averaged.** A chance node is expanded over every die face (36 pairs for a
  two-dice roll), so a coup is priced by its expectation, not by a sampled roll.
* **The opponent is played by the same evaluator** from their own side, and minimises ours
  where they make a structural choice inside our round.
* **Hidden cards are shuffled before the search** (`determinize`), so the bot never reads the
  opponent's hand or the deck order.

struggler's DEFCON whole-hand survival search is ported in reduced form (`_defcon_hazard`): a
card choice that leaves fewer safe plays than rounds to fill, with DEFCON at 2 or likely to get
there, is charged the game. What is not ported yet: the one-ply reply look-ahead, the hand-value
terms (holding cards, Ask Not, Missile Envy targets) and the calibrated Military Ops discount.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import ts_engine as ts

from bot.doctrine import evaluator as ev
from bot.doctrine import schedule

DT = ts.DecisionType
CONFIRM_DONE = 208
ROLL_DIE_INDEX = 115
NODE = 116
RESOLUTION = 110
EVENT_RESOLUTION = RESOLUTION + 0

#: struggler's default opening book ("iran"): USSR East Germany +1, Poland +4, Austria +1;
#: US West Germany 4, Italy 4, Iran to 2 -- the last two points are the +2 bonus placement.
_EG, _POLAND, _AUSTRIA, _WG, _ITALY, _IRAN = 14, 15, 13, 7, 10, 25
OPENING_USSR = [_EG, _POLAND, _POLAND, _POLAND, _POLAND, _AUSTRIA]
OPENING_US = [_WG, _WG, _WG, _WG, _ITALY, _ITALY, _ITALY, _ITALY, _IRAN]


#: A card's ways out at DEFCON 2 (`DoctrinePolicy._exits_at_two`).
SAFE, SPACE_ONLY, FATAL = "safe", "space_only", "fatal"
SPACE_MODE = RESOLUTION + 1      # SELECT_PLAY_MODE: Event, Space, Ops (influence, coup, realign)


def space_attempts_left(state: ts.GameState, p: ts.Player) -> int:
    """Space Race attempts `p` has left this turn: one, or two with Animal in Space (the engine's
    `SpaceRace::has_animal_in_space` -- own track at 2 or more, the opponent's below 2)."""
    mine = int(state.us_space_track if p == ts.Player.US else state.ussr_space_track)
    theirs = int(state.ussr_space_track if p == ts.Player.US else state.us_space_track)
    allowed = 2 if mine >= 2 and theirs < 2 else 1
    if mine >= 8:
        return 0
    return max(0, allowed - int(state.get_space_turns_used(p)))


def hand_survives(exits: Sequence[str], space_slots: int, china: bool, rounds: int) -> bool:
    """Whether a hand can fill `rounds` action rounds at DEFCON 2 without a certain loss.

    A SAFE card fills a round; a SPACE_ONLY card fills one only by using a space attempt, of
    which there are `space_slots`; a FATAL card can only be held. The China Card fills one."""
    safe = sum(1 for e in exits if e == SAFE)
    spaced = min(space_slots, sum(1 for e in exits if e == SPACE_ONLY))
    return safe + spaced + (1 if china else 0) >= rounds


def side_index(p: ts.Player) -> int:
    return ev.US if p == ts.Player.US else ev.USSR


def opponent(p: ts.Player) -> ts.Player:
    return ts.Player.USSR if p == ts.Player.US else ts.Player.US


def legal_actions(state: ts.GameState) -> List[int]:
    return [int(i) for i in np.flatnonzero(ts.get_flat_action_mask(state))]


def determinize(state: ts.GameState, me: ts.Player, rng: random.Random) -> ts.GameState:
    """A clone with the cards `me` cannot see reshuffled between the opponent's hand and the
    deck, counts preserved (as `ai.search.dmcts.determinize`)."""
    out = state.clone()
    opp_unknown = ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN
    hand = [c for c in range(1, 111) if state.get_card_location(c) == opp_unknown]
    deck = [c for c in range(1, 111) if state.get_card_location(c) == ts.CardLocation.DRAW_DECK]
    # Sorted before the shuffle: hand-then-deck order would depend on which unseen cards the
    # opponent actually holds, and the sampled world -- so the choice -- would leak it.
    pool = sorted(hand + deck)
    rng.shuffle(pool)
    for c in pool[:len(hand)]:
        out.set_card_location(c, opp_unknown)
    for c in pool[len(hand):]:
        out.set_card_location(c, ts.CardLocation.DRAW_DECK)
    out.rng_state = rng.getrandbits(64)
    return out


def _dice_for(state: ts.GameState, full: bool) -> List[Tuple[int, int, float]]:
    """Forced (actor, opponent) dice for a chance node, with their probabilities.

    Two-dice rolls are decided by the difference alone -- a realignment, Summit, the Olympics --
    so their 36 pairs collapse to the 11 differences, weighted by how many pairs give each. A
    chance node that is not the first in its line (`full` false) takes the middle outcome, as
    struggler's event sandbox does: expanding every roll of every realignment in a sequence is
    36 to the power of the sequence."""
    rt = state.ctx().pending_roll_type
    two = rt in (ts.RollType.REALIGNMENT, ts.RollType.SUMMIT, ts.RollType.OLYMPIC_GAMES)
    if not full:
        return [(4, 4, 1.0)] if two else [(4, 0, 0.5), (3, 0, 0.5)]
    if two:
        return [((1 + d, 1) if d >= 0 else (1, 1 - d)) + ((6 - abs(d)) / 36.0,) for d in range(-5, 6)]
    return [(a, 0, 1.0 / 6.0) for a in range(1, 7)]


@dataclass
class _Context:
    """What one decision is priced against: fixed for every position the search visits."""

    me: ts.Player
    urgency: Tuple[float, ...]
    vp_price: float
    rounds_left: int
    root_key: Tuple[int, int, int, int]
    evaluations: int = 0
    cache: Dict[Tuple[int, ...], float] = field(default_factory=dict)


class DoctrinePolicy:
    """Chooses a flat action for the player to move."""

    def __init__(self, weights: Optional[ev.Weights] = None, seed: int = 0):
        self.w = weights or ev.Weights()
        self.t = ev.terrain()
        self.rng = random.Random(seed)
        self.last_evaluations = 0

    # -- entry point ------------------------------------------------------------------------

    def choose(self, state: ts.GameState) -> int:
        legal = legal_actions(state)
        if not legal:
            return CONFIRM_DONE
        if len(legal) == 1:
            return legal[0]
        me = state.ctx().decision_player
        if state.current_phase == ts.Phase.SETUP:
            return self._opening(state, me, legal)
        legal = self._must_play_scoring(state, me, legal)
        if len(legal) == 1:
            return legal[0]
        world = determinize(state, me, self.rng)
        ctx = self._prepare(world, me)
        if (state.current_phase == ts.Phase.HEADLINE and state.ctx().decision_type == DT.SELECT_CARD
                and int(state.ctx().resolving_card) == 0):
            scored = [(self._headline_value(world, a, ctx), a) for a in legal]
        else:
            structural = self._is_structural(world, root=True)
            scored = []
            for a in legal:
                child = world.clone()
                if not ts.Engine.try_step_flat(child, a):
                    continue
                v = self._settle(child, ctx) if structural else self._shallow(child, ctx, me)
                scored.append((v, a))
            if self._is_card_play(state):
                hazard = self._defcon_hazard(world, ctx, [a for _, a in scored])
                scored = [(v - hazard.get(a, 0.0), a) for v, a in scored]
        self.last_evaluations = ctx.evaluations
        if not scored:
            return legal[0]
        best = max(v for v, _ in scored)
        return min(a for v, a in scored if v == best)

    # -- the hand at DEFCON 2 -------------------------------------------------------------

    #: P(the opponent drops DEFCON from 3 to 2 before our remaining rounds are played) --
    #: struggler's `last_window_guard`, the measured rate of a legal DEFCON-lowering coup.
    DROP_TO_TWO = 0.43

    @staticmethod
    def _is_card_play(state: ts.GameState) -> bool:
        c = state.ctx()
        return (state.current_phase == ts.Phase.ACTION_ROUND and c.decision_type == DT.SELECT_CARD
                and int(c.resolving_card) == 0 and int(state.ctx_stack_depth) == 0)

    def _defcon_hazard(self, world: ts.GameState, ctx: _Context, candidates: List[int]) -> Dict[int, float]:
        """What each card choice risks by the hand it leaves behind.

        struggler's DEFCON whole-hand survival search: at DEFCON 2 some cards cannot be played
        at all without losing -- an opponent's Event that degrades DEFCON or hands them a coup in
        our round -- and some survive only through the Space Race. If the hand we keep cannot
        fill the rounds we still have to play, one of those rounds loses the game. Each decision
        on its own was fine in the game that found this (CIA Created as the USSR's last card at
        DEFCON 2); the loss was in the hand.

        The accounting is joint: a card that survives only by being spaced needs one of the
        turn's space attempts, and one attempt cannot dispose of two such cards
        (`hand_survives`)."""
        defcon = int(world.defcon)
        if defcon > 3:
            return {}
        p_two = 1.0 if defcon <= 2 else self.DROP_TO_TWO
        me = ctx.me
        hand = [c for c in range(1, 111) if ts.in_hand_of(world.get_card_location(c), me)]
        china = int(world.china_card_holder) == int(me) and int(world.china_card_playable) == 1
        rounds_after = max(0, ctx.rounds_left - 1)
        if rounds_after == 0:
            return {}
        exits = {c: self._exits_at_two(world, c, ctx) for c in hand}
        slots = space_attempts_left(world, me)
        swing = ev.GAME_SWING_VP * ctx.vp_price
        out: Dict[int, float] = {}
        for a in candidates:
            played = a + 1 if a < 110 else 0
            kept = [exits[c] for c in hand if c != played]
            # Playing a card that survives only in space spends this turn's attempt on it now.
            left = slots - (1 if played in exits and exits[played] == SPACE_ONLY and defcon <= 2 else 0)
            if not hand_survives(kept, max(0, left), china and played != 6, rounds_after):
                out[a] = p_two * swing
        return out

    def _exits_at_two(self, world: ts.GameState, card: int, ctx: _Context) -> str:
        """How `card` can be played in a later round of ours at DEFCON 2 without certainly
        losing: SAFE (some play other than the Space Race survives), SPACE_ONLY (only spacing it
        does) or FATAL (nothing does)."""
        sim = world.clone()
        sim.defcon = 2
        sim.phasing_player = ctx.me
        sim.ctx().decision_player = ctx.me
        sim.ctx().decision_type = DT.SELECT_CARD
        sim.ctx().resolving_card = 0
        local = _Context(me=ctx.me, urgency=ctx.urgency, vp_price=ctx.vp_price,
                         rounds_left=ctx.rounds_left, root_key=self._round_key(sim))
        if not ts.Engine.try_step_flat(sim, card - 1):
            return SAFE
        try:
            if ts.Engine.is_terminal(sim):
                return FATAL if self._certain_loss(self.static(sim, side_index(ctx.me), local), local) else SAFE
            if sim.ctx().decision_type != DT.SELECT_PLAY_MODE:
                return FATAL if self._certain_loss(self._settle(sim, local), local) else SAFE
            spaced = False
            for mode in legal_actions(sim):
                child = sim.clone()
                if not ts.Engine.try_step_flat(child, mode):
                    continue
                if not self._certain_loss(self._settle(child, local, nest=2), local):
                    if mode != SPACE_MODE:
                        return SAFE
                    spaced = True
            return SPACE_ONLY if spaced else FATAL
        finally:
            ctx.evaluations += local.evaluations

    # -- setup and forced plays ---------------------------------------------------------------

    def _opening(self, state: ts.GameState, me: ts.Player, legal: List[int]) -> int:
        book = OPENING_USSR if me == ts.Player.USSR else OPENING_US
        placed = sum(int(state.get_country(c).ussr_influence if me == ts.Player.USSR
                         else state.get_country(c).us_influence) - self._setup_base(me, c)
                     for c in set(book))
        for cid in book[placed:]:
            if NODE + cid in legal:
                return NODE + cid
        return legal[0]

    @staticmethod
    def _setup_base(me: ts.Player, c: int) -> int:
        fixed_us = {_IRAN: 1}
        fixed_ussr = {_EG: 3}
        return (fixed_us if me == ts.Player.US else fixed_ussr).get(c, 0)

    def _must_play_scoring(self, state: ts.GameState, me: ts.Player, legal: List[int]) -> List[int]:
        """Holding a scoring card at the turn's end loses the game, so once the scoring cards in
        hand fill the rounds left, only they may be played (struggler's `must_play_scoring`)."""
        if state.current_phase != ts.Phase.ACTION_ROUND or state.ctx().decision_type != DT.SELECT_CARD:
            return legal
        if int(state.ctx().resolving_card) != 0:
            return legal
        scoring = [a for a in legal if a < 110 and ts.CardData.get_card_info(a + 1)["is_scoring"]]
        if scoring and len(scoring) >= schedule.rounds_left(state):
            return scoring
        return legal

    # -- pricing --------------------------------------------------------------------------

    def _prepare(self, state: ts.GameState, me: ts.Player) -> _Context:
        urgency = schedule.urgency(state, me, self.t, self.w)
        ctx = _Context(me=me, urgency=urgency, vp_price=1.0, rounds_left=schedule.rounds_left(state),
                       root_key=self._round_key(state))
        one_op = self._one_op_value(state, me, ctx)
        per_vp = self.w.vp_base * self.w.vp_swing ** ((int(state.turn) - 1) / 9)
        ctx.vp_price = per_vp * max(one_op, 1.0)
        return ctx

    def _one_op_value(self, state: ts.GameState, me: ts.Player, ctx: _Context) -> float:
        """What one Op buys on this board: the best single influence point (struggler prices a
        VP as a multiple of this, so VP and Ops stay on one scale)."""
        s = side_index(me)
        pos = ev.Position.of(state, self.t)
        base = self._board(pos, s, state, ctx)
        # Eligibility is read from the root board, never from `pos` inside the loop: each
        # candidate adds and removes a point, and a refresh in between would let the previous
        # candidate's hypothetical influence extend reach to the next country.
        reach = list(pos.reach[s])
        control = list(pos.control)
        best = 0.0
        for i in range(ev.N_COUNTRIES):
            if not reach[i] or control[i] == 1 - s:
                continue
            pos.inf[s][i] += 1
            pos.refresh(self.t)
            best = max(best, self._board(pos, s, state, ctx) - base)
            pos.inf[s][i] -= 1
        pos.refresh(self.t)
        return best

    def _board(self, pos: ev.Position, s: int, state: ts.GameState, ctx: _Context) -> float:
        formosan = state.has_flag(ts.EffectBits.FORMOSAN_RESOLUTION_ACTIVE)
        shuttle = state.has_flag(ts.EffectBits.SHUTTLE_DIPLOMACY_ACTIVE)
        overrides = None
        if formosan or shuttle:
            overrides = {r: ev.scoring_overrides(self.t, pos, r, formosan=formosan, shuttle=shuttle)
                         for r in (ev.ASIA, ev.MIDDLE_EAST)}
        ctx.evaluations += 1
        return ev.board_value(self.t, pos, s, self.w, ctx.urgency, overrides)

    #: An ongoing position's value is squashed strictly inside the terminal values: at most this
    #: share of the game swing, however lopsided the board.
    ONGOING_CAP = 0.999

    def static(self, state: ts.GameState, s: int, ctx: _Context) -> float:
        """The value of a position to side `s`, on one bounded scale.

        A finished game is worth +/- the game swing (40 VP at this decision's VP price). An
        ongoing one is struggler's raw board value -- board, VP and Military Ops -- mapped
        through `swing * ONGOING_CAP * tanh(raw / swing)`. The raw value is unbounded and
        routinely comes near the swing (a measured 5% of positions within 12% of it, extremes
        at 1.3x), so without the map the search could prefer losing outright to continuing a bad
        game, or pass up a win for a good board. The map is monotone, so it never reorders two
        ongoing positions; it only changes how dice outcomes average when values are large, and
        every value stays finite, so expectations over chance remain well defined."""
        swing = ev.GAME_SWING_VP * ctx.vp_price
        if ts.Engine.is_terminal(state):
            vp = int(state.victory_points)
            won = vp > 0 if s == ev.US else vp < 0
            lost = vp < 0 if s == ev.US else vp > 0
            return swing if won else -swing if lost else 0.0
        key = (s,) + self._board_key(state)
        cached = ctx.cache.get(key)
        if cached is not None:
            return cached
        pos = ev.Position.of(state, self.t)
        value = self._board(pos, s, state, ctx)
        vp = int(state.victory_points) * (1 if s == ev.US else -1)
        value += ctx.vp_price * vp
        # Military Operations: a deficit at the turn's end pays the opponent a VP per point.
        # Spread over the rounds still to play, as struggler's `military_credit` does.
        defcon = int(state.defcon)
        mine = int(state.us_mil_ops if s == ev.US else state.ussr_mil_ops)
        theirs = int(state.ussr_mil_ops if s == ev.US else state.us_mil_ops)
        deficit = max(0, defcon - mine) - max(0, defcon - theirs)
        value -= self.w.military * deficit * ctx.vp_price / max(1, ctx.rounds_left)
        value = self.ONGOING_CAP * swing * math.tanh(value / swing)
        ctx.cache[key] = value
        return value

    def _certain_loss(self, value: float, ctx: _Context) -> bool:
        """Whether a backed-up value can only come from lines that all end in our loss. An ongoing
        position is worth more than -ONGOING_CAP x swing, and dice average over outcomes, so only
        a value at the loss itself is certain -- a low board is not."""
        swing = ev.GAME_SWING_VP * ctx.vp_price
        return value <= -swing * (1.0 - 1e-9)

    @staticmethod
    def _board_key(state: ts.GameState) -> Tuple[int, ...]:
        inf = []
        for i in range(ev.N_COUNTRIES):
            c = state.get_country(i)
            inf.append(int(c.us_influence) * 64 + int(c.ussr_influence))
        return tuple(inf) + (int(state.victory_points), int(state.defcon), int(state.us_mil_ops),
                             int(state.ussr_mil_ops),
                             int(state.has_flag(ts.EffectBits.FORMOSAN_RESOLUTION_ACTIVE)),
                             int(state.has_flag(ts.EffectBits.SHUTTLE_DIPLOMACY_ACTIVE)))

    # -- search ---------------------------------------------------------------------------

    @staticmethod
    def _round_key(state: ts.GameState) -> Tuple[int, int, int, int]:
        return (int(state.turn), int(state.current_phase), int(state.action_round), int(state.phasing_player))

    def _at_boundary(self, state: ts.GameState, ctx: _Context) -> bool:
        """The next card choice, the turn's cleanup, or the game's end: where an option stops."""
        if ts.Engine.is_terminal(state):
            return True
        c = state.ctx()
        if c.decision_type == DT.ROLL_DIE and c.pending_roll_type == ts.RollType.TURN_CLEANUP:
            return True
        if (c.decision_type == DT.SELECT_CARD and int(c.resolving_card) == 0
                and int(state.ctx_stack_depth) == 0 and self._round_key(state) != ctx.root_key):
            return True
        return self._round_key(state)[:2] != ctx.root_key[:2]

    #: How many structural choices may nest inside one option before the rest go greedy. Card,
    #: then play mode, then Ops mode, then an event's branch is four; beyond that the tree is an
    #: event chain (Missile Envy into Five Year Plan into ...) where greed costs little.
    MAX_STRUCTURAL_DEPTH = 4

    @staticmethod
    def _is_structural(state: ts.GameState, root: bool = False) -> bool:
        """Searched by playing every alternative out, rather than greedily one step deep.

        A card choice is structural only at the root -- the card to play this round. Inside an
        event a card choice is a discard or a pick (Ask Not asks nine times in a row), and
        searching those exhaustively is exponential for little gain."""
        dt = state.ctx().decision_type
        if dt in (DT.SELECT_PLAY_MODE, DT.SELECT_OP_MODE, DT.CHOOSE_BRANCH):
            return True
        return root and dt == DT.SELECT_CARD

    def _settle(self, state: ts.GameState, ctx: _Context, depth: int = 0, nest: int = 1,
                rolled: bool = False) -> float:
        """Play `state` out to the round's boundary; the expected value there, for `ctx.me`.

        `rolled` is whether this line has already expanded a die: only the first chance node on a
        line is expanded over its faces, the rest take the middle outcome (`_dice_for`)."""
        me_s = side_index(ctx.me)
        guard = 0
        while True:
            guard += 1
            if guard > 400 or depth > 12 or self._at_boundary(state, ctx):
                return self.static(state, me_s, ctx)
            c = state.ctx()
            if c.decision_type == DT.ROLL_DIE and c.decision_player == ts.Player.NONE:
                return self._expect(state, ctx, depth,
                                    lambda child: self._settle(child, ctx, depth + 1, nest, True),
                                    full=not rolled)
            legal = legal_actions(state)
            if not legal:
                return self.static(state, me_s, ctx)
            if len(legal) == 1:
                if not ts.Engine.try_step_flat(state, legal[0]):
                    return self.static(state, me_s, ctx)
                continue
            decider = c.decision_player
            if nest < self.MAX_STRUCTURAL_DEPTH and self._is_structural(state):
                best: Optional[float] = None
                for a in legal:
                    child = state.clone()
                    if not ts.Engine.try_step_flat(child, a):
                        continue
                    v = self._settle(child, ctx, depth + 1, nest + 1, rolled)
                    if best is None or (v > best if decider == ctx.me else v < best):
                        best = v
                return best if best is not None else self.static(state, me_s, ctx)
            a = self._greedy_pick(state, legal, ctx, decider)
            if not ts.Engine.try_step_flat(state, a):
                return self.static(state, me_s, ctx)

    def _greedy_pick(self, state: ts.GameState, legal: List[int], ctx: _Context, decider: ts.Player) -> int:
        best_a, best_v = legal[0], -math.inf
        for a in legal:
            child = state.clone()
            if not ts.Engine.try_step_flat(child, a):
                continue
            v = self._shallow(child, ctx, decider)
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def _shallow(self, state: ts.GameState, ctx: _Context, side: ts.Player) -> float:
        """The value right after one step, a pending die averaged over its faces."""
        s = side_index(side)
        c = state.ctx()
        if (not ts.Engine.is_terminal(state) and c.decision_type == DT.ROLL_DIE
                and c.decision_player == ts.Player.NONE and c.pending_roll_type != ts.RollType.TURN_CLEANUP):
            return self._expect(state, ctx, 0, lambda child: self.static(child, s, ctx))
        return self.static(state, s, ctx)

    def _expect(self, state: ts.GameState, ctx: _Context, depth: int, value_of, full: bool = True) -> float:
        total = 0.0
        mass = 0.0
        for a, b, p in _dice_for(state, full):
            child = state.clone()
            if not ts.Engine.try_step(child, ts.MicroAction(DT.ROLL_DIE, a, b, 0)):
                continue
            total += p * value_of(child)
            mass += p
        if mass == 0.0:
            return self.static(state, side_index(ctx.me), ctx)
        return total / mass

    # -- headline -------------------------------------------------------------------------

    def _headline_value(self, world: ts.GameState, action: int, ctx: _Context) -> float:
        """A headline card priced as its Event played in an action round of ours.

        The opponent's headline is unknown, so it is left out: what is compared is what each of
        our cards' Events does to the board. Played through the ordinary action-round machinery
        on a copy, since that already resolves every Event."""
        if action >= 110:
            return -math.inf
        card = action + 1
        me = ctx.me
        sim = world.clone()
        sim.current_phase = ts.Phase.ACTION_ROUND
        sim.action_round = 1
        sim.phasing_player = me
        sim.headline_stage = 3
        sim.ctx().decision_player = me
        sim.ctx().decision_type = DT.SELECT_CARD
        sim.ctx().resolving_card = 0
        local = _Context(me=me, urgency=ctx.urgency, vp_price=ctx.vp_price, rounds_left=ctx.rounds_left,
                         root_key=self._round_key(sim))
        if not ts.Engine.try_step_flat(sim, action):
            return -math.inf
        if sim.ctx().decision_type == DT.SELECT_PLAY_MODE and int(sim.ctx().resolving_card) == 0:
            if not ts.Engine.try_step_flat(sim, EVENT_RESOLUTION):
                # A card that cannot be an Event here does nothing in the headline.
                return self.static(world, side_index(me), local)
        value = self._settle(sim, local)
        ctx.evaluations += local.evaluations
        return value
