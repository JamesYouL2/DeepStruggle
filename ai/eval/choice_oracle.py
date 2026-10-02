"""The choice oracle: named alternatives at a named kind of decision, judged by paired playouts.

The doctrine census (`doctrine_census.py`) says where the model departs from a strong player's
rule; this says what the departure costs. Each **scenario** names a kind of decision and the
alternatives to force there:

* `headline` -- a turn's headline for one side, holding both named cards: headline A against
  headline B (and the model's own choice);
* `event` -- a card play of one named card by one side, where the event is legal: event against
  the model's own choice (optionally only when the mover is behind on VP).

Positions come from the model's own greedy self-play. Headline positions are collected by short
games that stop at the headline, so a rare pair of cards is cheap to find. From each position,
pair k redeals the cards the mover cannot see (and forgets an unrevealed opponent headline, which
the engine then asks for again) and uses the same dice in every branch (`branch_oracle._pair_start`),
so branches differ only in the decision. The model plays every continuation on both sides.

As always with this instrument, the verdict is about the alternatives as this model would follow
them up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _pair_start, apply_prefix, play_out
from ai.eval.doctrine_census import N_CARDS, card_context, cards
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
EVENT = MODE_BASE + 0
SPACE = MODE_BASE + 1


@dataclass
class Scenario:
    name: str
    kind: str                      # "headline", "event" or "space"
    side: str                      # "US" or "USSR"
    cards: Tuple[str, ...]         # headline: (A, B); event: (card,)
    turn: int = 1                  # headline only
    behind: bool = False           # event only: mover behind on VP
    before_turn: int = 99          # event/space: only before this turn
    departures: bool = False       # keep only positions where the model does NOT play the alternative
    cond: str = ""                 # "space_ahead" (US ahead in space) or "event_vp5" (the event scores 5+ VP)
    ids: Tuple[int, ...] = field(default=())

    def resolve(self, by_name: Dict[str, int]) -> "Scenario":
        self.ids = tuple(by_name[c] for c in self.cards)
        return self

    def branches(self) -> Dict[str, List[int]]:
        if self.kind == "headline":
            return {"policy": [], self.cards[0]: [self.ids[0] - 1], self.cards[1]: [self.ids[1] - 1]}
        if self.kind == "space":
            return {"policy": [], "space": [SPACE]}
        return {"policy": [], "event": [EVENT]}

    def alternative(self) -> Optional[int]:
        return {"event": EVENT, "space": SPACE}.get(self.kind)


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("USSR T1 headline: Asia Scoring vs Nasser", "headline", "USSR", ("Asia Scoring", "Nasser")),
    Scenario("USSR T1 headline: Asia Scoring vs Socialist Governments", "headline", "USSR",
             ("Asia Scoring", "Socialist Governments")),
    Scenario("USSR T1 headline: Asia Scoring vs Suez Crisis", "headline", "USSR", ("Asia Scoring", "Suez Crisis")),
    Scenario("US T1 headline: Middle East Scoring vs Containment", "headline", "US",
             ("Middle East Scoring", "Containment")),
    Scenario("US T1 headline: Middle East Scoring vs Defectors", "headline", "US",
             ("Middle East Scoring", "Defectors")),
    Scenario("US plays Grain Sales to Soviets: event vs the model's choice", "event", "US", ("Grain Sales to Soviets",)),
    Scenario("USSR plays Aldrich Ames Remix: event vs the model's choice", "event", "USSR", ("Aldrich Ames Remix",)),
    Scenario("US plays Star Wars while ahead in space: event vs the model's choice", "event", "US", ("Star Wars",),
             cond="space_ahead"),
    Scenario("USSR plays OPEC worth 5+ VP: event vs the model's choice", "event", "USSR", ("OPEC",), cond="event_vp5"),
    Scenario("US plays Alliance for Progress worth 5+ VP: event vs the model's choice", "event", "US",
             ("Alliance for Progress",), cond="event_vp5"),
    Scenario("USSR plays Che: event vs the model's choice", "event", "USSR", ("Che",)),
    Scenario("USSR holds The Voice of America and lets it fire: space it instead", "space", "USSR",
             ("The Voice of America",), departures=True),
    Scenario("USSR holds Colonial Rear Guards and lets it fire: space it instead", "space", "USSR",
             ("Colonial Rear Guards",), departures=True),
    Scenario("US holds Decolonization before the Late War and lets it fire: space it instead", "space", "US",
             ("Decolonization",), before_turn=8, departures=True),
)


def scenarios() -> List[Scenario]:
    info = cards()
    by_name = {str(info[c]["name"]): c for c in info}
    return [Scenario(s.name, s.kind, s.side, s.cards, s.turn, s.behind, s.before_turn, s.departures, s.cond)
            .resolve(by_name) for s in SCENARIOS]


def _player(side: str) -> ts.Player:
    return ts.Player.US if side == "US" else ts.Player.USSR


def _hand(st: ts.GameState, p: ts.Player) -> set:
    locs = ((ts.CardLocation.HAND_US_KNOWN, ts.CardLocation.HAND_US_UNKNOWN) if p == ts.Player.US
            else (ts.CardLocation.HAND_USSR_KNOWN, ts.CardLocation.HAND_USSR_UNKNOWN))
    return {c for c in range(1, N_CARDS + 1) if st.get_card_location(c) in locs}


def matches(sc: Scenario, st: ts.GameState, mask: np.ndarray) -> bool:
    ctx = st.ctx()
    if ctx.decision_player != _player(sc.side):
        return False
    if sc.kind == "headline":
        return (st.current_phase == ts.Phase.HEADLINE and st.headline_stage == 0 and int(st.turn) == sc.turn
                and ctx.decision_type == ts.DecisionType.SELECT_CARD
                and all(bool(mask[c - 1]) for c in sc.ids))
    if ctx.decision_type != ts.DecisionType.SELECT_PLAY_MODE or int(ctx.pending_op_card) != sc.ids[0]:
        return False
    alt = sc.alternative()
    if alt is None or not mask[alt] or mask[MODE_BASE:MODE_BASE + 5].sum() < 2 or int(st.turn) >= sc.before_turn:
        return False
    if sc.behind and int(st.victory_points) * (1 if sc.side == "US" else -1) >= 0:
        return False
    if sc.cond == "space_ahead" and int(st.us_space_track) <= int(st.ussr_space_track):
        return False
    if sc.cond == "event_vp5" and card_context(st, sc.ids[0]).get("event_vp", 0) < 5:
        return False
    return True


def collect(act: PolicyFn, scs: Sequence[Scenario], per: int, seed: int, envs: int = 64,
            headline_games: int = 20_000, full_games: int = 3_000, accept: float = 1.0,
            max_steps: int = 3_000_000) -> Dict[str, List[Tuple[ts.GameState, int]]]:
    """Up to `per` positions per scenario, each with the model's greedy action there. Headline
    scenarios use short games reset after the turn-1 headlines; event scenarios full games, each
    position kept with probability `accept` and at most one per scenario per game."""
    out: Dict[str, List[Tuple[ts.GameState, int]]] = {s.name: [] for s in scs}
    rng = np.random.default_rng(seed)
    for phase, budget in (("headline", headline_games), ("event", full_games)):
        todo = [s for s in scs if (s.kind == "headline") == (phase == "headline")]
        if not todo:
            continue
        runner = ts.VectorizedBatchRunner(envs, seed * 7 + (1 if phase == "event" else 0))
        runner.refresh_all()
        started, finished = envs, 0
        taken: List[set] = [set() for _ in range(envs)]
        for _ in range(max_steps):
            if all(len(out[s.name]) >= per for s in todo) or finished >= budget:
                break
            obs = np.asarray(runner.get_observations())
            masks = np.asarray(runner.get_action_masks())
            acts = act(obs, masks)
            reset: List[int] = []
            for i in range(envs):
                m = masks[i]
                if not m.any():
                    continue
                relevant = m[:N_CARDS].any() if phase == "headline" else m[MODE_BASE:MODE_BASE + 5].any()
                if not relevant:
                    continue
                st = runner.get_state(i)
                if phase == "headline" and (st.current_phase != ts.Phase.HEADLINE or int(st.turn) > 1):
                    if st.current_phase == ts.Phase.ACTION_ROUND:
                        reset.append(i)                     # past the turn-1 headlines: next deal
                    continue
                for s in todo:
                    if len(out[s.name]) >= per or s.name in taken[i] or not matches(s, st, m):
                        continue
                    if s.departures and int(acts[i]) == s.alternative():
                        continue
                    if phase == "event" and rng.random() >= accept:
                        continue
                    taken[i].add(s.name)
                    out[s.name].append((st.clone(), int(acts[i])))
            runner.step_flat_all([int(x) for x in acts], auto_advance=True)
            ends = set(np.flatnonzero(np.array(runner.get_terminals())).tolist()) | set(reset)
            for i in ends:
                finished += 1
                runner.reset_game(int(i), seed * 1_000_003 + started + (500_000 if phase == "event" else 0))
                started += 1
                taken[i] = set()
            if ends:
                runner.refresh_all()
    return out


def _loses_now(st: ts.GameState, a: int, mover: ts.Player) -> bool:
    probe = st.clone()
    ts.Engine.step_flat(probe, int(a))
    drain_chance(probe, context="choice_oracle safety probe")
    if not ts.Engine.is_terminal(probe):
        return False
    u = float(ts.Engine.get_terminal_utility(probe))
    return (u < 0) if mover == ts.Player.US else (u > 0)


def resolve_safely(st: ts.GameState, act: PolicyFn, max_steps: int = 40) -> ts.GameState:
    """Finish the mover's current action round from `st` with the model's choices, except that an
    option which loses the game on the spot is never taken while another exists.

    A forced event's inner choices (the card Star Wars retrieves, the card Aldrich Ames discards)
    are decisions the model rarely or never reached in training; left to it, it retrieved a DEFCON
    reducer at DEFCON 2 in 12% of Star Wars events and lost on the spot. A strong player resolves
    the event without suicide; everything after the action round is the model's."""
    s = st.clone()
    mover = s.ctx().decision_player
    turn_ar = (s.turn, s.action_round, s.current_phase)
    for _ in range(max_steps):
        if ts.Engine.is_terminal(s) or (s.turn, s.action_round, s.current_phase) != turn_ar:
            break
        ctx = s.ctx()
        if ctx.decision_player != mover:
            break
        mask = np.asarray(ActionEncoder.get_legal_mask(s)).astype(np.uint8)
        if not mask.any():
            break
        legal = np.flatnonzero(mask)
        safe = [a for a in legal if not _loses_now(s, int(a), mover)]
        if safe and len(safe) < len(legal):
            mask = np.zeros_like(mask)
            mask[safe] = 1
        obs = np.asarray(ts.extract_observation(s, mover), dtype=np.float32)[None]
        a = int(act(obs, mask[None])[0])
        ts.Engine.step_flat(s, a)
        drain_chance(s, context="choice_oracle safe resolution")
    return s


def play(act: PolicyFn, sc: Scenario, positions: Sequence[Tuple[ts.GameState, int]], pairs: int, seed: int,
         chunk: int = 768, label: Optional[Callable[[int], str]] = None) -> List[Dict[str, Any]]:
    """Every position's branches over `pairs` paired playouts; one row per position."""
    names = list(sc.branches())
    prefixes = sc.branches()
    per_pos = len(names) * pairs
    step = max(1, chunk // per_pos)
    rows: List[Dict[str, Any]] = []
    for lo in range(0, len(positions), step):
        group = positions[lo:lo + step]
        starts: List[ts.GameState] = []
        movers: List[ts.Player] = []
        for j, (st, _) in enumerate(group):
            mover = st.ctx().decision_player
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo + j, "resample")
                for nm in names:
                    start = apply_prefix(base, prefixes[nm])
                    if sc.kind != "headline":
                        start = resolve_safely(start, act)     # every branch's action round, without suicide
                    starts.append(start)
                    movers.append(mover)
        ends = play_out(starts, movers, act, seed)
        for j, (st, chosen) in enumerate(group):
            sc_arr = np.array([e["score"] for e in ends[j * per_pos:(j + 1) * per_pos]]).reshape(pairs, len(names))
            row: Dict[str, Any] = {"scenario": sc.name, "turn": int(st.turn), "pairs": pairs,
                                   "vp": int(st.victory_points), "chosen": label(chosen) if label else chosen}
            row.update({nm: float(sc_arr[:, k].mean()) for k, nm in enumerate(names)})
            rows.append(row)
    return rows


def _diff(rows: Sequence[Dict[str, Any]], a: str, b: str) -> Tuple[float, float]:
    d = np.array([r[a] - r[b] for r in rows])
    if len(d) < 2:
        return (float(d.mean()) if len(d) else float("nan")), float("nan")
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def report(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    out = [f"# Choice oracle — {meta.get('model', '?')}", "",
           "Paired playouts from the model's own positions, hidden cards redealt per pair; score = the "
           "mover's win % (draw ½); ± one standard error over positions.", ""]
    summary: Dict[str, Any] = {"meta": meta, "scenarios": {}}
    for sc in scenarios():
        sel = [r for r in rows if r["scenario"] == sc.name]
        if not sel:
            out += [f"## {sc.name}", "", "no positions found", ""]
            continue
        names = list(sc.branches())
        pairs = sel[0]["pairs"]
        out += [f"## {sc.name}", "", f"{len(sel)} positions × {pairs} pairs.", "",
                "| branch | mover's win % | − policy |", "|:---|---:|---:|"]
        entry: Dict[str, Any] = {"n": len(sel), "pairs": pairs}
        for nm in names:
            mean = 100 * float(np.mean([r[nm] for r in sel]))
            if nm == "policy":
                out.append(f"| policy (the model's own choice) | {mean:.1f}% | — |")
            else:
                d, se = _diff(sel, nm, "policy")
                out.append(f"| {nm} | {mean:.1f}% | {100 * d:+.1f} ± {100 * se:.1f} |")
                entry[nm] = {"mean": mean / 100, "minus_policy": (d, se)}
        if sc.kind == "headline":
            a, b = sc.cards
            d, se = _diff(sel, a, b)
            out.append(f"\n**{a} − {b}: {100 * d:+.1f} ± {100 * se:.1f}**")
            entry["a_minus_b"] = (d, se)
        chosen = {}
        for r in sel:
            chosen[r["chosen"]] = chosen.get(r["chosen"], 0) + 1
        out.append("\nThe model's own choice: " + ", ".join(f"{k} {v}" for k, v in
                                                           sorted(chosen.items(), key=lambda x: -x[1])[:5]) + ".")
        out.append("")
        summary["scenarios"][sc.name] = entry
    return "\n".join(out) + "\n", summary
