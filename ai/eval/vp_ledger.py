"""Where do the VP come from? A ledger by source, the model against strong humans; and how
calibrated and how swingy its win estimate is.

The fork owner's hypothesis (a strong player's, 2026-10-02): the model plays the board well and
underweights everything that is not the board -- VP from events, the space race, Military Ops,
auto-wins. A ledger says where each side's VP come from, per game, in the model's self-play and
in the human ts-replayer corpus, so the two can be laid side by side.

**Attribution.** The engine changes VP in many places and records no reason, so each change is
credited to the decision whose step produced it, read from the position before it:

* `scoring: <region>` -- a scoring card played (or headlined);
* `event: <card>` -- an event resolving, including an opponent's card whose event fires after the
  Ops are spent, and the choices inside an event;
* `headline events` -- VP during the headline phase outside an event's own choices: the step that
  resolves the headlines, or the one finishing the first headline's event, which resolves the second;
* `space race` -- a space attempt;
* `turn end: Military Ops` -- the Military Ops shortfall at the end of turns 1-9, the only VP the
  engine awards at a turn's end before the last: read from each side's Military Ops and DEFCON
  before the step that ends the turn, so it is split off a play that scored on the same step (exact
  unless that play itself moved DEFCON or Military Ops);
* `final scoring` -- the end of turn 10 (final scoring, the China Card, the last shortfall);
* `mixed` -- a step that ends turn 10 *and* resolved a VP source itself (a scoring card as the last
  play of the game, say), where final scoring cannot be split off; kept apart rather than guessed at;
* `other` -- anything else that moved VP, listed so an attribution gap shows instead of hiding.

Each source reports the VP it brought each side per game.

**Calibration and swing** (self-play only: the model's value head `v_win`, read from the side to
move at every decision, converted to P(US wins)). Reliability by bucket, against the game's
result; a fitted temperature (below 1: overconfident); and the step-to-step movement of the
estimate, split by whether the side to move changed -- the two sides' readings of nearby
positions should agree, since nothing happens between them but a hand-over.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.doctrine_census import cards
from bindings.action_encoder import ActionEncoder

MODE_BASE = int(ActionEncoder.PLAY_MODE_OFFSET)
SCORING = {1: "Asia", 2: "Europe", 3: "Middle East", 37: "Central America", 38: "Southeast Asia",
           79: "Africa", 81: "South America"}
N_BINS = 20
ERAS = ((1, 3, "Early War"), (4, 7, "Mid War"), (8, 10, "Late War"))


def _name(cid: int) -> str:
    info = cards()
    return str(info[cid]["name"]) if cid in info else f"card {cid}"


def label(st: ts.GameState, action: int) -> str:
    """The VP source a step from `st` playing `action` would be credited to (before turn ends)."""
    ctx = st.ctx()
    dt = ctx.decision_type
    pc, rc = int(ctx.pending_op_card), int(ctx.resolving_card)
    if action < 110 and dt == ts.DecisionType.SELECT_CARD and (action + 1) in SCORING and rc == 0:
        return f"scoring: {SCORING[action + 1]}"
    if pc in SCORING:
        return f"scoring: {SCORING[pc]}"
    if st.current_phase == ts.Phase.HEADLINE and rc == 0:
        # Choosing a headline, or the step that finishes the first headline's event and so resolves
        # the second: either way the VP are the headlines'.
        return "headline events"
    if rc:
        return f"event: {_name(rc)}"
    if dt == ts.DecisionType.SELECT_PLAY_MODE:
        if action == MODE_BASE:
            return f"event: {_name(pc)}"
        if action == MODE_BASE + 1:
            return "space race"
    if pc and pc in cards():
        side = str(cards()[pc]["side"]).upper()
        dec = ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player
        if side == ("USSR" if dec == ts.Player.US else "US"):
            # An opponent's card played for Ops: Ops never move VP, so a change during the play is
            # its event (fired after the Ops, or before them on the event-first branch).
            return f"event: {_name(pc)}"
    return "other"


def shortfall(st: ts.GameState) -> int:
    """The Military Ops penalty (US-positive VP) a turn ending from `st` would award
    (`Scoring::evaluate_military_ops`): each side short of the DEFCON level concedes the gap."""
    req = int(st.defcon)
    us_def = max(0, req - int(st.us_mil_ops))
    ussr_def = max(0, req - int(st.ussr_mil_ops))
    return ussr_def - us_def


def credit(lab: str, vp0: int, vp1: int, turn0: int, turn1: int, over: bool, penalty: int) -> List[Tuple[str, int]]:
    """The sources a step's VP change goes to, now that its outcome is known. A step that ends turns
    1-9 carries the Military Ops shortfall (`penalty`, read before the step) on top of whatever its
    own play scored; the play keeps the rest. At the end of turn 10, final scoring cannot be told
    apart from a play's own VP, so such a step is `mixed`."""
    delta = vp1 - vp0
    ends_turn = turn1 > turn0
    final = over and turn0 >= 10 and abs(vp1) < 20
    if final or (ends_turn and turn0 >= 10):
        return [("final scoring" if lab == "other" else "mixed", delta)]
    if ends_turn:
        if lab == "other":
            return [("turn end: Military Ops", delta)]
        return [("turn end: Military Ops", penalty), (lab, delta - penalty)]
    return [(lab, delta)]


class Ledger:
    """VP by source for one game, each side's gains kept apart."""

    def __init__(self) -> None:
        self.gains: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    def add(self, source: str, delta: int) -> None:
        if delta > 0:
            self.gains[source][0] += delta
        elif delta < 0:
            self.gains[source][1] += -delta

    def row(self, **extra: Any) -> Dict[str, Any]:
        return {"gains": {k: list(v) for k, v in self.gains.items()}, **extra}


# --- the model's self-play: the ledger and the value readings, from the same games -----------

ValueFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]   # -> (actions, v_win)


def _empty_cal() -> Dict[str, Any]:
    return {"bins": {}, "swing": {"same": [0, 0.0, 0.0], "handover": [0, 0.0, 0.0]}}


def _bin_key(side: str, era: str) -> str:
    return f"{side}|{era}"


def _era(turn: int) -> str:
    return next(nm for lo, hi, nm in ERAS if lo <= max(1, min(turn, 10)) <= hi)


def selfplay(policy: ValueFn, n_games: int, seed: int, envs: int = 32,
             max_steps: int = 3_000_000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Greedy self-play: one ledger row per finished game, and the calibration tallies."""
    runner = ts.VectorizedBatchRunner(envs, seed * 29 + 11)
    runner.refresh_all()
    ledgers = [Ledger() for _ in range(envs)]
    pending: List[Optional[Tuple[str, int, int, int]]] = [None] * envs
    readings: List[List[Tuple[float, int, int]]] = [[] for _ in range(envs)]   # (p_us, decider, turn)
    rows: List[Dict[str, Any]] = []
    cal = _empty_cal()
    started, finished = envs, 0
    for _ in range(max_steps):
        if finished >= n_games:
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        acts, v = policy(obs, masks)
        states = [runner.get_state(i) for i in range(envs)]
        for i, st in enumerate(states):
            held = pending[i]
            if held is not None:
                lab, vp0, turn0, pen = held
                for src, d in credit(lab, vp0, int(st.victory_points), turn0, int(st.turn),
                                     bool(ts.Engine.is_terminal(st)), pen):
                    ledgers[i].add(src, d)
                pending[i] = None
            if ts.Engine.is_terminal(st) or not masks[i].any():
                continue
            ctx = st.ctx()
            dec = ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player
            us = dec == ts.Player.US
            vv = float(v[i])
            readings[i].append(((vv if us else -vv) * 0.5 + 0.5, int(us), int(st.turn)))
            pending[i] = (label(st, int(acts[i])), int(st.victory_points), int(st.turn), shortfall(st))
        runner.step_flat_all([int(x) for x in acts], auto_advance=True)
        ends = np.flatnonzero(np.array(runner.get_terminals())).tolist()
        for i in ends:
            st = runner.get_state(int(i))
            held = pending[i]
            if held is not None:
                lab, vp0, turn0, pen = held
                for src, d in credit(lab, vp0, int(st.victory_points), turn0, int(st.turn), True, pen):
                    ledgers[i].add(src, d)
                pending[i] = None
            u = float(ts.Engine.get_terminal_utility(st))
            us_win = 1.0 if u > 0 else (0.5 if u == 0 else 0.0)
            rows.append(ledgers[i].row(turns=int(st.turn), complete=True, vp=int(st.victory_points), us_win=us_win))
            _tally(cal, readings[i], us_win)
            finished += 1
            ledgers[i] = Ledger()
            readings[i] = []
            runner.reset_game(int(i), seed * 1_000_003 + started)
            started += 1
        if ends:
            runner.refresh_all()
    return rows, cal


def _tally(cal: Dict[str, Any], readings: Sequence[Tuple[float, int, int]], us_win: float) -> None:
    prev = None
    for p, us, turn in readings:
        key = _bin_key("US to move" if us else "USSR to move", _era(turn))
        b = cal["bins"].setdefault(key, [[0, 0.0, 0.0] for _ in range(N_BINS)])
        j = min(N_BINS - 1, max(0, int(p * N_BINS)))
        b[j][0] += 1
        b[j][1] += p
        b[j][2] += us_win
        if prev is not None:
            kind = "same" if prev[1] == us else "handover"
            d = abs(p - prev[0])
            s = cal["swing"][kind]
            s[0] += 1
            s[1] += d
            s[2] += d * d
        prev = (p, us)


def merge_cal(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = _empty_cal()
    for c in parts:
        for key, bins in c["bins"].items():
            tgt = out["bins"].setdefault(key, [[0, 0.0, 0.0] for _ in range(N_BINS)])
            for j in range(N_BINS):
                for k in range(3):
                    tgt[j][k] += bins[j][k]
        for kind in ("same", "handover"):
            for k in range(3):
                out["swing"][kind][k] += c["swing"][kind][k]
    return out


# --- the human corpus ------------------------------------------------------------------------

def human_game(game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The ledger of one corpus game, or None if it does not convert. `complete` is whether the
    recording reaches the end of the game (a result), so totals can be restricted to those."""
    from tools.lib.ts_replayer_convert import convert_game

    seq: List[Tuple[str, int, int, int]] = []
    last: List[ts.GameState] = []

    def on_decision(state: ts.GameState, mover: ts.Player, entry: Any, chosen: int) -> None:
        seq.append((label(state, int(chosen)), int(state.victory_points), int(state.turn), shortfall(state)))

    def on_entry(state: ts.GameState, entry: Any) -> None:
        last[:] = [state]

    conv = convert_game(game, on_decision=on_decision, on_entry=on_entry)
    if conv.skipped or not seq or not last:
        return None
    end = last[0]
    led = Ledger()
    over = bool(ts.Engine.is_terminal(end))
    for k, (lab, vp0, turn0, pen) in enumerate(seq):
        if k + 1 < len(seq):
            _, vp1, turn1, _ = seq[k + 1]
            parts = credit(lab, vp0, vp1, turn0, turn1, False, pen)
        else:
            parts = credit(lab, vp0, int(end.victory_points), turn0, int(end.turn), over, pen)
        for src, d in parts:
            led.add(src, d)
    u = float(ts.Engine.get_terminal_utility(end)) if over else 0.0
    return led.row(turns=int(end.turn), complete=over, vp=int(end.victory_points),
                   us_win=(1.0 if u > 0 else (0.5 if u == 0 else 0.0)) if over else None)


# --- the report ------------------------------------------------------------------------------

GROUPS = (("scoring cards", lambda s: s.startswith("scoring:")), ("events", lambda s: s.startswith("event:")),
          ("headline events", lambda s: s == "headline events"), ("space race", lambda s: s == "space race"),
          ("turn end: Military Ops", lambda s: s == "turn end: Military Ops"),
          ("final scoring", lambda s: s == "final scoring"), ("mixed", lambda s: s == "mixed"),
          ("other", lambda s: s == "other"))


def _per_game(rows: Sequence[Dict[str, Any]], pred: Callable[[str], bool]) -> Tuple[float, float]:
    us = sum(v[0] for r in rows for k, v in r["gains"].items() if pred(k))
    ussr = sum(v[1] for r in rows for k, v in r["gains"].items() if pred(k))
    n = max(1, len(rows))
    return us / n, ussr / n


def _temperature(bins: Sequence[Sequence[float]]) -> float:
    """The scale a on logit(p) that best fits the results (log loss over the bins): below 1 the
    estimate is overconfident, above 1 underconfident."""
    pts = [(b[1] / b[0], b[2] / b[0], b[0]) for b in bins if b[0] > 0]
    if not pts:
        return float("nan")

    def loss(a: float) -> float:
        tot = 0.0
        for p, y, n in pts:
            p = min(max(p, 1e-4), 1 - 1e-4)
            q = 1 / (1 + math.exp(-a * math.log(p / (1 - p))))
            q = min(max(q, 1e-6), 1 - 1e-6)
            tot -= n * (y * math.log(q) + (1 - y) * math.log(1 - q))
        return tot

    grid = [x / 100 for x in range(20, 301)]
    return min(grid, key=loss)


def report(bot: Sequence[Dict[str, Any]], human: Sequence[Dict[str, Any]], cal: Dict[str, Any],
           meta: Dict[str, Any], top: int = 25) -> Tuple[str, Dict[str, Any]]:
    hc = [r for r in human if r["complete"]]
    out = [f"# VP ledger and calibration — {meta.get('model', '?')}", "",
           f"{len(bot)} greedy self-play games (mean last turn {np.mean([r['turns'] for r in bot]):.1f}); "
           f"{len(human)} human games converted, {len(hc)} recorded to the end "
           f"(mean last turn {np.mean([r['turns'] for r in hc]) if hc else float('nan'):.1f}). "
           "VP each side gained from each source, per game; human totals over the complete games.", ""]
    summary: Dict[str, Any] = {"meta": meta, "bot_games": len(bot), "human_games": len(hc), "groups": {}}
    out += ["## By kind of source", "",
            "| source | model US | human US | model USSR | human USSR |", "|:---|---:|---:|---:|---:|"]
    for name, pred in GROUPS:
        b, h = _per_game(bot, pred), _per_game(hc, pred)
        out.append(f"| {name} | {b[0]:.2f} | {h[0]:.2f} | {b[1]:.2f} | {h[1]:.2f} |")
        summary["groups"][name] = {"model": b, "human": h}
    out += ["", "Per game, how games end:", "",
            f"* model: US wins {100 * np.mean([r['us_win'] for r in bot]):.0f}%, |final VP| ≥ 20 in "
            f"{100 * np.mean([abs(r['vp']) >= 20 for r in bot]):.0f}%",
            f"* humans: US wins {100 * np.mean([r['us_win'] for r in hc]) if hc else float('nan'):.0f}%, "
            f"|final VP| ≥ 20 in {100 * np.mean([abs(r['vp']) >= 20 for r in hc]) if hc else float('nan'):.0f}%", ""]

    sources = sorted({k for r in list(bot) + list(hc) for k in r["gains"]})
    diffs = []
    for s in sources:
        b, h = _per_game(bot, lambda k, s=s: k == s), _per_game(hc, lambda k, s=s: k == s)
        diffs.append((s, b, h, (h[0] - b[0]) + (h[1] - b[1])))
    for title, keep in (("Scoring cards", lambda s: s.startswith("scoring:")),
                        ("Events: where humans gain more VP than the model, per game", lambda s: s.startswith("event:"))):
        sel = [d for d in diffs if keep(d[0])]
        sel.sort(key=lambda d: -d[3])
        out += [f"## {title}", "", "| source | model US | human US | model USSR | human USSR | human − model |",
                "|:---|---:|---:|---:|---:|---:|"]
        for s, b, h, d in sel[:top]:
            out.append(f"| {s} | {b[0]:.2f} | {h[0]:.2f} | {b[1]:.2f} | {h[1]:.2f} | {d:+.2f} |")
        out.append("")
    ev = sorted([d for d in diffs if d[0].startswith("event:")], key=lambda d: d[3])[:10]
    out += ["### …and where the model gains more", "", "| source | model US | human US | model USSR | human USSR | human − model |",
            "|:---|---:|---:|---:|---:|---:|"]
    for s, b, h, d in ev:
        out.append(f"| {s} | {b[0]:.2f} | {h[0]:.2f} | {b[1]:.2f} | {h[1]:.2f} | {d:+.2f} |")
    out.append("")

    # Calibration.
    if cal["bins"]:
        allb = [[0, 0.0, 0.0] for _ in range(N_BINS)]
        for bins in cal["bins"].values():
            for j in range(N_BINS):
                for k in range(3):
                    allb[j][k] += bins[j][k]
        out += ["## Calibration of the win estimate (self-play)", "",
                "P(US wins) from the value head, read from the side to move at every decision, bucketed, against "
                "how often the US actually won. A fitted temperature below 1 means overconfident.", "",
                "| predicted | readings | mean predicted | US won |", "|:---|---:|---:|---:|"]
        for j, (n, sp, sy) in enumerate(allb):
            if n:
                out.append(f"| {j / N_BINS:.2f}–{(j + 1) / N_BINS:.2f} | {n} | {sp / n:.3f} | {sy / n:.3f} |")
        n_all = sum(b[0] for b in allb)
        brier_proxy = sum(b[0] * (b[1] / b[0] - b[2] / b[0]) ** 2 for b in allb if b[0]) / max(1, n_all)
        out += ["", f"Overall fitted temperature: **{_temperature(allb):.2f}**; binned calibration error "
                f"(mean squared, predicted vs observed): {brier_proxy:.4f}.", "",
                "| side to move | era | readings | temperature | mean predicted | US won |", "|:---|:---|---:|---:|---:|---:|"]
        temps = {}
        for key in sorted(cal["bins"]):
            bins = cal["bins"][key]
            n = sum(b[0] for b in bins)
            side, era = key.split("|")
            t = _temperature(bins)
            temps[key] = t
            out.append(f"| {side} | {era} | {n} | {t:.2f} | {sum(b[1] for b in bins) / n:.3f} | "
                       f"{sum(b[2] for b in bins) / n:.3f} |")
        summary["temperature"] = _temperature(allb)
        summary["temperature_by"] = temps
        sw = cal["swing"]
        out += ["", "### Swing", "",
                "Mean movement of P(US wins) from one decision to the next, in points.", "",
                "| step | steps | mean move | rms move |", "|:---|---:|---:|---:|"]
        for kind, nm in (("same", "same side to move"), ("handover", "side to move changes")):
            n, s1, s2 = sw[kind]
            if n:
                out.append(f"| {nm} | {n} | {100 * s1 / n:.2f} | {100 * math.sqrt(s2 / n):.2f} |")
        summary["swing"] = sw
    return "\n".join(out) + "\n", summary
