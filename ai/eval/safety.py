"""Detection of decisions that immediately win or lose the game.

:func:`classify_legal_actions` labels each legal action, fast enough to run inside a bot's move
loop. The two directions are found differently.

**Wins** are found by search (:func:`_search_win`). An action is a win when a line from it ends
the game in the mover's favour using nothing but the mover's own choices and steps with a single
legal action -- no opponent decision, no die, no hidden draw, and nothing past the current card's
play. At a choice of the mover's, one winning option suffices. That one definition covers every
way a win hides behind a choice, without a card list:

* Wargames at DEFCON 2 with a lead over 6: the card from hand, its event, its branch;
* an event whose VP reaches 20, from hand;
* Star Wars eventing Wargames or such an event from the discard -- from the mover's own hand, or
  at the pick when the opponent played Star Wars for Ops;
* the card Grain Sales drew, at its play -- whoever played Grain Sales. Choosing Grain Sales
  itself is never a win: what it draws is chance.

The search is sound but not complete: a win label always rests on a line it stepped to the end.
At a placement decision it follows one fixed option instead of all of them, which can miss a win
that needed a different placement and never invents one.

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

import numpy as np
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
    legal = [int(a) for a in np.flatnonzero(ActionEncoder.get_legal_mask(state))]
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
    for a in legal:
        if out[a] != "normal":
            continue
        child = state.clone()
        if _step_without_chance(child, a) and _search_win(child, who, scope, [WIN_SEARCH_BUDGET]):
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
    # are not used: a win that hangs on a die is not one the mover can take.
    for a in legal:
        if out[a] != "normal":
            continue
        if _probe_forced(state, a, who) == "loss":
            out[a] = "loss"

    return out


#: Nodes one top-level action's win search may visit. Every line it follows ends within the
#: current card's play, so a search that runs out has wandered, and reports no win.
WIN_SEARCH_BUDGET = 400

#: The mover's decisions at which the win search tries every option. These route a card to how
#: it resolves -- which card, which mode, which branch -- and are where a win hides behind a
#: choice: Wargames' branch, Star Wars' pick, the play of the card Grain Sales drew. At any other
#: decision (a placement) the search follows one fixed option: trying them all is combinatorial.
_ROUTING_DECISIONS = frozenset({
    ts.DecisionType.SELECT_CARD,
    ts.DecisionType.SELECT_PLAY_MODE,
    ts.DecisionType.CHOOSE_TIMING_BRANCH,
    ts.DecisionType.SELECT_OP_MODE,
    ts.DecisionType.CHOOSE_BRANCH,
})


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
    """True if `player` can end the game in their favour from `probe` by their own choices alone.

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
        # A die, and the end of the turn, stop the line. The turn end is not random, but what it
        # settles -- Military Ops, a scoring card the opponent still holds, final scoring -- is not
        # the card's play, and is usually reached by every option alike, dice routes included: a
        # win there would label the options that roll a die "declined" beside winning siblings.
        if ctx.decision_type == ts.DecisionType.ROLL_DIE or _scope(probe) != scope:
            return False
        legal = [int(a) for a in np.flatnonzero(ActionEncoder.get_legal_mask(probe))]
        if not legal:
            return False
        if len(legal) > 1:
            if _acting(probe) != player:
                return False                     # an opponent's choice: not the mover's to take
            if ctx.decision_type in _ROUTING_DECISIONS:
                for a in legal:
                    child = probe.clone()
                    if _step_without_chance(child, a) and _search_win(child, player, scope, nodes):
                        return True
                return False
            # A placement: one fixed option, stopping early where that is allowed.
            legal = [CONFIRM_DONE if CONFIRM_DONE in legal else legal[0]]
        if not _step_without_chance(probe, legal[0]):
            return False


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

        forced = np.flatnonzero(ActionEncoder.get_legal_mask(probe))
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
