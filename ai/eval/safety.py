"""Detection of decisions that immediately win or lose the game.

:func:`classify_legal_actions` labels each legal action, fast enough to run inside a bot's move
loop. The two directions are found differently.

**Wins** are found by search (:func:`_search_win`). An action is a win when a line from it ends
the game in the mover's favour using nothing but the mover's own choices, steps with a single legal
action, and dice that win on every face -- no opponent decision, no hidden draw, and nothing past
the current card's play and the turn end it reaches. At a choice of the mover's, one winning option
suffices. That one definition covers every way a win hides behind a choice, without a card list:

* Wargames at DEFCON 2 with a lead over 6: the card from hand, its event, its branch;
* an event whose VP reaches 20, from hand;
* Star Wars eventing Wargames or such an event from the discard -- from the mover's own hand, or
  at the pick when the opponent played Star Wars for Ops;
* the card Grain Sales drew, at its play -- whoever played Grain Sales. Choosing Grain Sales
  itself is a win only when the USSR holds one card: otherwise what it draws is chance;
* a battleground coup at DEFCON 2 in the opponent's round (CIA Created or Tear Down This Wall
  played for Ops): DEFCON 1 defeats the phasing opponent, whatever the coup rolls;
* a last action round whose turn end wins: Military Ops made up by a coup, a scoring card the
  opponent still holds, final scoring.

The search tries one option per group of options with the same immediate outcome
(:func:`_same_outcome_groups`): influence, realignment and an event's placements are one group, and
a coup's targets two, battleground or not -- where influence goes can decide the game only at the
final turn's end. A win label always rests on a line the search stepped to the end; where that
reading of placements does not hold, it can miss a win, never invent one.

**Losses** keep their rules plus a forced probe (:func:`_probe_forced`): apply the action, follow
single-option nodes and dice, and report a loss every sampled roll agrees on. A general minimax
for losses would exhaust any sane budget on breadth, and a search that returns "unknown" is
indistinguishable from one that returns "safe" -- which is exactly the failure mode a safety check
must not have.

Key engine facts these rules encode, each verified by stepping to a terminal state:

* The DEFCON-1 loser is always the **phasing player**, whoever's event fired.
* An opponent-associated card cannot be played as an event; playing it for **Ops** fires
  the owner's event unavoidably, on either timing branch.
* At SELECT_PLAY_MODE the card being played is `pending_op_card`, which is what the engine itself
  reads there. `resolving_card` is the card whose event *asked* -- Grain Sales, for the card it drew.
"""

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from ai.eval.positions import PLAY_MODE_ACTION

# Events that degrade DEFCON with no intervening choice. Firing one at DEFCON 2 ends the
# game against whoever is phasing. Verified in both directions (own event, and opponent's
# card played for Ops).
DEFCON_DEGRADING_CARDS = {
    4: "Duck and Cover",
    50: "We Will Bury You",
    89: "Soviets Shoot Down KAL-007",
    46: "How I Learned to Stop Worrying",
}

# Degrades DEFCON only if the opponent picks the boycott branch, so it is decisive only
# under an opponent who takes it.
OLYMPIC_GAMES = 20
WARGAMES = 100

# Events granting the OPPONENT a free coup. At DEFCON 2 that coup can take DEFCON to 1,
# losing the game for the player who played the card -- conditional on the opponent taking
# the line.
OPPONENT_COUP_CARDS = {
    91: "Ortega Elected in Nicaragua",
    62: "Lone Gunman",
    67: "Grain Sales to Soviets",
    96: "Tear Down This Wall",
}

EVENT_ACTION = PLAY_MODE_ACTION["event"]
OPS_ACTION = PLAY_MODE_ACTION["ops"]
NODE_OFFSET = ActionEncoder.NODE_OFFSET   # one source; see bindings/action_encoder.py
CONFIRM_DONE = ActionEncoder.CONFIRM_DONE_INDEX


def _acting(state: ts.GameState) -> ts.Player:
    ctx = state.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else state.phasing_player


def classify_legal_actions(
    state: ts.GameState,
    player: Optional[ts.Player] = None,
) -> Dict[int, str]:
    """Classifies each legal action as 'win', 'loss', 'risky', or 'normal'.

    * ``win``    -- ends the game in this player's favour.
    * ``loss``   -- ends the game against them, with no choice of theirs to avoid it.
    * ``risky``  -- hands the opponent a line that ends the game against them.
    * ``normal`` -- nothing decisive.
    """
    who = player if player is not None else _acting(state)
    ctx = state.ctx()
    legal = _legal_actions(state)
    out: Dict[int, str] = {a: "normal" for a in legal}

    # The card being played. Reading `resolving_card` first named Grain Sales at the play of the
    # card it drew, so that card's own DEFCON loss went unseen and its Ops took Grain Sales' "risky".
    card = int(ctx.pending_op_card)
    defcon = int(state.defcon)

    # --- play-mode decisions on a card already selected ---------------------------------
    # Only for the phasing player, whom DEFCON 1 defeats. The other side reaches a play mode
    # inside the phasing player's round -- the card Grain Sales drew -- and there taking DEFCON to
    # 1 wins, which the search below finds.
    if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and card and who == state.phasing_player:
        if defcon <= 2:
            if card in DEFCON_DEGRADING_CARDS:
                # Firing it loses outright; for an opponent card the only way to fire it is
                # Ops, and for our own card it is the event.
                if EVENT_ACTION in out:
                    out[EVENT_ACTION] = "loss"
                if OPS_ACTION in out and ts.CardData.get_card_info(card)["side"] != _side_name(who):
                    out[OPS_ACTION] = "loss"
            if card == OLYMPIC_GAMES and EVENT_ACTION in out:
                out[EVENT_ACTION] = "loss"
            if card in OPPONENT_COUP_CARDS and OPS_ACTION in out:
                out[OPS_ACTION] = "risky"

    # --- wins: the mover's own line to the end of the game ---------------------------------
    # Wargames needs no rule of its own: event, then branch 0, is such a line exactly when the
    # lead survives the opponent's 6 VP. Two hand-written Wargames rules lived here and both were
    # wrong -- the branch rule compared flat indices against 0, so it never fired, and the
    # play-mode rule missed the Wargames that Grain Sales draws.
    scope = _scope(state)
    candidates = [a for a in legal if out[a] == "normal"]
    for group in _same_outcome_groups(state, candidates, every_target_in_final_turn=True):
        child = state.clone()
        if _step_without_chance(child, group[0]) and _search_win(child, who, scope, [WIN_SEARCH_BUDGET]):
            for a in group:
                out[a] = "win"

    # --- forced losses: VP thresholds, scoring cards, coups ------------------------------
    # Apply each action and follow only FORCED continuations -- chance nodes, and nodes
    # with a single legal action. Anything that ends the game against the mover along that path
    # is decisive no matter what either player would have chosen, so this needs no card list and
    # picks up every VP-threshold crossing (26 sites in the engine), every scoring card that
    # reaches +/-20, and every coup that takes DEFCON to 1.
    #
    # Following only forced steps is what keeps this both cheap and sound: it can miss a
    # decisive line that needed a choice, but it never reports one that does not exist. Its wins
    # are not used: it samples dice, and the search above steps every face. A placement's targets
    # lose alike within a group, as they win alike (`_same_outcome_groups`) -- but stopping early
    # is probed apart: this probe follows no choice, so stopping reaches the turn end (and a held
    # scoring card) where placing a point stops at the next placement. Probed one by one, targets
    # split the same way by cost: with the turn's last 2 Ops to place and a scoring card held, a
    # 2-Op country reached the losing turn end and read "loss" while a 1-Op one read "normal" --
    # and the probe counted an unavoidable loss as avoidable. The group gives them one label.
    candidates = [a for a in legal if out[a] == "normal"]
    for group in _same_outcome_groups(state, candidates, every_target_in_final_turn=True):
        for part in _split_stop(group):
            if _probe_forced(state, part[0], who) == "loss":
                for a in part:
                    out[a] = "loss"

    return out


#: Nodes one top-level action's win search may visit. Every line it follows ends within the
#: current card's play or at the turn end it reaches, so a search that runs out has wandered, and
#: reports no win.
WIN_SEARCH_BUDGET = 400

#: The game's last turn: only its end scores the board, so only there can where influence goes
#: decide the game on the spot.
FINAL_TURN = 10

BATTLEGROUNDS = frozenset(c for c in range(84) if ts.MapData.get_country_info(c)["battleground"])


def _same_outcome_groups(state: ts.GameState, actions: List[int],
                         every_target_in_final_turn: bool = False) -> List[List[int]]:
    """`actions` split into groups whose immediate outcome is the same, so the win search can try
    one of each.

    Every choice that routes a card -- which card, which mode, which branch, which card from the
    discard -- is its own group: that is where a win hides behind a choice. A placement is not:

    * influence, realignment, an event's placement: where it goes changes no VP, no DEFCON and
      nobody's turn, so every target ends the same way -- one group;
    * a coup: the target matters only as battleground or not. A battleground takes DEFCON down,
      to 1 at DEFCON 2 -- defeating the phasing player, which in the opponent's round (CIA Created,
      Tear Down This Wall, Grain Sales' Ops) is a win -- and, above 2, to a level that changes both
      sides' Military Ops at the turn end. Otherwise it only adds Military Ops, whatever the target
      and whatever it rolls -- two groups.

    The exception is the final turn, whose end scores the board: there the top-level decision tries
    every target (`every_target_in_final_turn`), and a search inside a line keeps its groups, which
    can only miss a win.
    """
    if state.ctx().decision_type != ts.DecisionType.POINT_NODE or len(actions) < 2:
        return [[a] for a in actions]
    if every_target_in_final_turn and int(state.turn) >= FINAL_TURN:
        return [[a] for a in actions]
    targets = [a for a in actions if a != CONFIRM_DONE]
    stop = [CONFIRM_DONE] if CONFIRM_DONE in actions else []
    if not targets or not _is_coup_target(state, targets[0]):
        # Stopping early, where allowed, leads first: it reaches whatever follows soonest.
        return [stop + targets]
    battle = [a for a in targets if a - NODE_OFFSET in BATTLEGROUNDS]
    other = [a for a in targets if a - NODE_OFFSET not in BATTLEGROUNDS]
    return [g for g in (other, battle, stop) if g]


def _split_stop(group: List[int]) -> List[List[int]]:
    rest = [a for a in group if a != CONFIRM_DONE]
    return [g for g in ([CONFIRM_DONE] if CONFIRM_DONE in group else [], rest) if g]


def _is_coup_target(state: ts.GameState, action: int) -> bool:
    """Whether choosing `action` rolls a coup. Asked of the engine, not of `op_mode`, which keeps
    its last value: an event's own placement (De-Stalinization, Decolonization) reads COUP after any
    earlier coup, and treating those as coups cost up to 47,000 steps a decision."""
    probe = state.clone()
    try:
        ts.Engine.step_flat(probe, action)
    except Exception:
        return False
    ctx = probe.ctx()
    return ctx.decision_type == ts.DecisionType.ROLL_DIE and ctx.pending_roll_type == ts.RollType.COUP


def _legal_actions(state: ts.GameState) -> List[int]:
    # `.nonzero()` on the mask, not `np.flatnonzero`: the search asks this at every step, and the
    # wrapper's ravel and dispatch were 30% of classifying.
    return ActionEncoder.get_legal_mask(state).nonzero()[0].tolist()


def _scope(state: ts.GameState) -> tuple:
    """The current card's play: a line that leaves it is no longer an immediate win."""
    return (int(state.turn), int(state.current_phase), int(state.action_round), int(state.phasing_player))


def _step_without_chance(probe: ts.GameState, action: int) -> bool:
    """Applies `action`; False if it was refused or drew on the RNG.

    A die is a ROLL_DIE node, but a hidden draw -- Grain Sales' card, Five Year Plan's discard, a
    redraw -- happens inside an ordinary step, and the only trace it leaves is `rng_state`.
    """
    before = int(probe.rng_state)
    try:
        ts.Engine.step_flat(probe, action)
    except Exception:
        return False
    return int(probe.rng_state) == before


def _search_win(probe: ts.GameState, player: ts.Player, scope: tuple, nodes: List[int]) -> bool:
    """True if `player` can end the game in their favour from `probe` by their own choices alone,
    whatever the dice.

    Consumes `probe`. `nodes` is a shared one-element budget.
    """
    while True:
        nodes[0] -= 1
        if nodes[0] < 0:
            return False
        if ts.Engine.is_terminal(probe):
            util = float(ts.Engine.get_terminal_utility(probe))
            return (util if player == ts.Player.US else -util) > 0
        ctx = probe.ctx()
        if ctx.decision_type == ts.DecisionType.ROLL_DIE and ctx.pending_roll_type == ts.RollType.TURN_CLEANUP:
            # The end of the turn the line reached. Military Ops, a scoring card still held and
            # final scoring are settled here with nothing random; a turn that goes on deals, which
            # is chance, and is past the card's play anyway.
            try:
                ts.Engine.step(probe, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            except Exception:
                return False
            if not ts.Engine.is_terminal(probe):
                return False
            continue
        if _scope(probe) != scope:
            return False
        if ctx.decision_type == ts.DecisionType.ROLL_DIE:
            return _every_roll_wins(probe, player, scope, nodes)
        legal = _legal_actions(probe)
        if not legal:
            return False
        if len(legal) > 1:
            if _acting(probe) != player:
                return False                     # an opponent's choice: not the mover's to take
            groups = _same_outcome_groups(probe, legal)
            if len(groups) > 1:
                for group in groups:
                    child = probe.clone()
                    if _step_without_chance(child, group[0]) and _search_win(child, player, scope, nodes):
                        return True
                return False
            legal = groups[0]
        if not _step_without_chance(probe, legal[0]):
            return False


def _every_roll_wins(probe: ts.GameState, player: ts.Player, scope: tuple, nodes: List[int]) -> bool:
    """A die on the line: every face must win. A battleground coup at DEFCON 2 takes DEFCON to 1
    whatever it rolls, so it passes; a war card that wins on 3-6 does not.

    Faces are forced through the ROLL_DIE action (`primary_id` the actor's die, `secondary_id` the
    opponent's), so each is stepped exactly rather than sampled. A roll that uses the second die
    draws it from the RNG when only the first is forced, which sends it to all 36 pairs; one that
    draws even then -- Olympic Games' reroll on a tie -- is chance the line cannot vouch for.
    """
    def wins(actor: int, opponent: int) -> Optional[bool]:
        child = probe.clone()
        before = int(child.rng_state)
        try:
            ts.Engine.step(child, ts.MicroAction(ts.DecisionType.ROLL_DIE, actor, opponent, 0))
        except Exception:
            return False
        if int(child.rng_state) != before:
            return None                          # an unforced die was rolled
        return _search_win(child, player, scope, nodes)

    for actor in range(1, 7):
        result = wins(actor, 0)
        if result is None:
            break
        if not result:
            return False
    else:
        return True
    for actor in range(1, 7):
        for opponent in range(1, 7):
            if not wins(actor, opponent):        # None (still drawing) or False
                return False
    return True


_GOLDEN = 0x9E3779B97F4A7C15
_UINT64 = 1 << 64


def _probe_forced(
    state: ts.GameState,
    action: int,
    player: ts.Player,
    max_forced: int = 6,
    die_samples: int = 6,
    node_budget: int = 64,
) -> Optional[str]:
    """Applies `action`, follows forced continuations, and reports a decisive outcome.

    A chance node is only decisive if *every* die outcome agrees. Stepping the die once and
    trusting the result -- which this did -- reports a win whenever that single sampled roll
    happens to succeed, so a coup or war that wins on 3-6 and leaves the game running on 1-2
    was labelled a forced win. Measured against the engine, 6% of "win" labels were
    die-dependent in that way: Brush War, Lone Gunman and similar VP-on-success cards, at
    9/16 to 14/16 win rates.

    Roll-independent lines survive this unchanged. A coup at DEFCON 2 degrades DEFCON to 1
    whatever the roll, ending the game against the phasing player, so every branch agrees
    and the label stands.

    Die outcomes are varied by mixing the probe's own rng_state, so the classifier stays a
    pure function of the position and remains reproducible.
    """
    probe = state.clone()
    try:
        ts.Engine.step_flat(probe, action)
    except Exception:
        return None
    return _follow_forced(probe, player, max_forced, die_samples, [node_budget])


def _follow_forced(
    probe: ts.GameState,
    player: ts.Player,
    budget: int,
    die_samples: int,
    nodes: List[int],
) -> Optional[str]:
    for _ in range(budget):
        nodes[0] -= 1
        if nodes[0] <= 0:
            return None
        if ts.Engine.is_terminal(probe):
            util = float(ts.Engine.get_terminal_utility(probe))
            mine = util if player == ts.Player.US else -util
            return "win" if mine > 0 else ("loss" if mine < 0 else None)

        pctx = probe.ctx()
        if pctx.decision_type == ts.DecisionType.ROLL_DIE:
            verdicts = set()
            base = int(probe.rng_state)
            for i in range(die_samples):
                branch = probe.clone()
                branch.rng_state = (base + (i + 1) * _GOLDEN) % _UINT64
                try:
                    ts.Engine.step(branch, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
                except Exception:
                    return None
                verdicts.add(_follow_forced(branch, player, budget - 1, die_samples, nodes))
                if len(verdicts) > 1:
                    return None      # outcome depends on the die: not forced
            return verdicts.pop() if len(verdicts) == 1 else None

        forced = _legal_actions(probe)
        if len(forced) != 1:
            return None
        try:
            ts.Engine.step_flat(probe, int(forced[0]))
        except Exception:
            return None

    return None


def _side_name(player: ts.Player) -> str:
    return "US" if player == ts.Player.US else "USSR"


def find_instant_win(state: ts.GameState, player: Optional[ts.Player] = None) -> Optional[int]:
    for action, kind in classify_legal_actions(state, player).items():
        if kind == "win":
            return action
    return None


#: One decision as the win counters see it: (mover, legal actions, winning actions, chose a win).
WinDecision = Tuple[Hashable, int, int, bool]


def fold_win_opportunities(decisions: Sequence[WinDecision]) -> List[Tuple[int, bool]]:
    """One game's decisions, in order, folded into one entry per chance to win: (index of the
    decision that opened it, taken).

    A win can take several of the mover's decisions -- Wargames from hand is the card, its event,
    then its branch -- and every one of them is labelled "win". Counted per decision, taking it
    scored 3 of 3 and playing the card for Ops 1 of 2, and the card counted three times over a win
    that takes one decision. Here the chance opens at the first decision offering it, stays open
    while the same mover keeps taking it, and is taken only if every decision of the line took it.

    A decision at which every legal action wins opens nothing: there is no way to decline it, just
    as a position where every action loses is not an avoidable loss.
    """
    out: List[Tuple[int, bool]] = []
    continuing: Optional[Hashable] = None       # mover of the open entry, while they keep taking it
    for i, (mover, n_legal, n_wins, taken) in enumerate(decisions):
        if n_wins and continuing is not None and continuing == mover:
            out[-1] = (out[-1][0], out[-1][1] and taken)
        elif 0 < n_wins < n_legal:
            out.append((i, taken))
        else:
            continuing = None
            continue
        continuing = mover if taken else None
    return out


def safe_actions(
    state: ts.GameState,
    player: Optional[ts.Player] = None,
    avoid_risky: bool = True,
) -> List[int]:
    """Legal actions that do not lose outright, falling back when every option is bad."""
    kinds = classify_legal_actions(state, player)
    bad = {"loss", "risky"} if avoid_risky else {"loss"}
    safe = [a for a, k in kinds.items() if k not in bad]
    if safe:
        return safe
    safe = [a for a, k in kinds.items() if k != "loss"]
    return safe if safe else list(kinds.keys())
