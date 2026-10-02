"""Does the net ignore the opponent's reply? Net against search, one action round at a time.

The fork owner's read of the E7 net (a strong player's, 2026-10-01): it "breaks too much,
recommends inefficient ops placements and doesn't defend enough" -- the signature of a policy that
picks moves without looking at what the opponent does next. Search (determinized MCTS on the same
net) is the only part of the system that does look; it beats the plain net ~55%. If that gain is
mostly about replies, the action rounds where the two disagree should show it: the search's round
should leave less exposed, break less, and lose less to the reply.

From positions in the net's own greedy self-play (the start of a mover's action round), two
branches each play the mover's whole action round:

* `net` -- the net's greedy choice at every mover decision;
* `search` -- determinized search (default 64 simulations) at every mover decision.

Decisions the opponent makes inside that round (an event's removal, say) are the net's in both.
Where the two rounds differ, each is measured three ways:

* **the round itself**, from the mover's influence change: points into its own controlled
  countries (defence), into uncontrolled ones, into the opponent's; countries touched; opponent
  influence removed; whether it couped, and where;
* **exposure** at the end of the round, before the reply: for each battleground the mover
  controls in a region open at this DEFCON, the chance a 3-Ops coup breaks control (none at
  DEFCON 2, where a battleground coup loses) and the expected influence one realignment roll by
  the opponent removes, at the engine's net modifier;
* **the reply**: from the end of the round, `pairs` playouts with the cards the mover cannot see
  redealt per pair (the same redeal and dice in both branches), the net on both sides. Measured
  when the opponent's next action round ends -- battlegrounds the mover lost, its influence
  removed, the VP swing -- and the mover's final score.

The verdict on a disagreement is only as good as the net playing the continuation, and search
here is a determinized search: in each sampled world the opponent replies knowing the mover's
hand, so search may defend against threats that are not there.
"""
from __future__ import annotations

import base64
import math
import random
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, play_out
from ai.eval.ops_block import NODE_OFFSET, N_COUNTRIES, controlled, country_table, influence
from ai.eval.realign_spots import net_modifiers
from bindings.action_encoder import ActionEncoder

MODE_BASE = 110
COUP = MODE_BASE + 3
BRANCHES = ("net", "search")
#: The DEFCON at or above which coups and realignments are allowed in a region (rules 8.1.4).
REGION_DEFCON = {"Europe": 5, "Asia": 4, "Middle East": 3}
WORKBENCH = "https://jamesyoul2.github.io/DeepStruggle/?pos="
_UINT64 = 1 << 64


def _region() -> List[str]:
    from ai.eval.doctrine_census import _map
    return [str(r["region"]) for r in _map()]


def _side(p: ts.Player) -> int:
    return 0 if p == ts.Player.US else 1


def _ar_key(st: ts.GameState) -> Tuple[int, int, int, int]:
    return (int(st.turn), int(st.action_round), int(st.phasing_player), int(st.current_phase))


def _decider(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def is_round_start(st: ts.GameState) -> bool:
    """The first decision of an action round: the phasing player choosing a card."""
    ctx = st.ctx()
    return (not ts.Engine.is_terminal(st) and st.current_phase == ts.Phase.ACTION_ROUND
            and ctx.decision_type == ts.DecisionType.SELECT_CARD
            and ctx.decision_player == st.phasing_player and int(ctx.resolving_card) == 0
            and int(ctx.pending_op_card) == 0)


def rounds_left(st: ts.GameState) -> int:
    """The mover's action rounds after this one this turn (6 a turn in the Early War, 7 after;
    the eighth that North Sea Oil or the space race grants is not counted)."""
    return (6 if int(st.turn) <= 3 else 7) - int(st.action_round)


def opponent_replies(st: ts.GameState) -> bool:
    """Does the opponent get an action round after this one before the turn ends? The USSR moves
    first in each round, so only the US's last round goes unanswered."""
    return st.phasing_player == ts.Player.USSR or rounds_left(st) > 0


def position_link(st: ts.GameState) -> str:
    raw = zlib.compress(st.to_save_json().encode("utf-8"))
    return WORKBENCH + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def collect(act: PolicyFn, n: int, seed: int, envs: int = 32, accept: float = 0.08,
            max_steps: int = 2_000_000) -> List[ts.GameState]:
    """Up to `n` round starts from the net's greedy self-play, each kept with probability
    `accept`, at most two per game so one long game cannot dominate."""
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(envs, seed * 13 + 5)
    runner.refresh_all()
    out: List[ts.GameState] = []
    taken = [0] * envs
    started = envs
    for _ in range(max_steps):
        if len(out) >= n:
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        for i in range(envs):
            if taken[i] >= 2 or not masks[i].any() or not masks[i][:110].any():
                continue
            st = runner.get_state(i)
            if is_round_start(st) and int(st.turn) >= 1 and rng.random() < accept:
                out.append(st.clone())
                taken[i] += 1
        runner.step_flat_all([int(x) for x in act(obs, masks)], auto_advance=True)
        ends = np.flatnonzero(np.array(runner.get_terminals())).tolist()
        for i in ends:
            runner.reset_game(int(i), seed * 1_000_003 + started)
            started += 1
            taken[i] = 0
        if ends:
            runner.refresh_all()
    return out[:n]


def play_rounds(starts: Sequence[ts.GameState], act: PolicyFn, search: Any, seed: int,
                max_steps: int = 200) -> List[Dict[str, Any]]:
    """Both branches of every start's action round. Returns per start, per branch, the end state
    and the mover's decisions (flat action, name)."""
    n = len(starts)
    runner = ts.VectorizedBatchRunner(2 * n, seed)
    for i, st in enumerate(starts):
        runner.set_state(2 * i, st)
        runner.set_state(2 * i + 1, st)
    runner.refresh_all()
    movers = [st.phasing_player for st in starts]
    keys = [_ar_key(st) for st in starts]
    done: List[Optional[ts.GameState]] = [None] * (2 * n)
    moves: List[List[Tuple[int, str]]] = [[] for _ in range(2 * n)]
    for _ in range(max_steps):
        states = [runner.get_state(j) for j in range(2 * n)]
        for j, st in enumerate(states):
            if done[j] is None and (ts.Engine.is_terminal(st) or _ar_key(st) != keys[j // 2]):
                done[j] = st.clone()
        if all(d is not None for d in done):
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        live = [j for j in range(2 * n) if masks[j].any() and not ts.Engine.is_terminal(states[j])]
        acts = np.zeros(2 * n, dtype=np.int32)
        if live:
            acts[live] = act(obs[live], masks[live])
        searched = [j for j in live if done[j] is None and j % 2 == 1 and _decider(states[j]) == movers[j // 2]]
        if searched:
            for j, a in zip(searched, search.select_actions_batch([states[j] for j in searched])):
                acts[j] = int(a)
        for j in live:
            if done[j] is None and _decider(states[j]) == movers[j // 2]:
                moves[j].append((int(acts[j]), ActionEncoder.get_action_name(states[j], int(acts[j]))))
        res = runner.step_flat_all(acts.tolist(), auto_advance=True)
        refused = [j for j in live if res[j] == 0]
        if refused:
            raise RuntimeError(f"the engine refused an action in round-branch {refused[0]}")
    for j in range(2 * n):
        if done[j] is None:
            raise RuntimeError(f"start {j // 2}: the action round did not end within {max_steps} steps")
    return [{"start": starts[i], "mover": movers[i],
             **{b: {"end": done[2 * i + k], "moves": moves[2 * i + k]} for k, b in enumerate(BRANCHES)}}
            for i in range(n)]


def first_divergence(a: Sequence[Tuple[int, str]], b: Sequence[Tuple[int, str]]) -> str:
    """Where two rounds first differ: the card, the play mode, a target, or something else."""
    for (x, _), (y, _) in zip(a, b):
        if x != y:
            lo = min(x, y)
            if lo < 110:
                return "card"
            if lo < NODE_OFFSET:
                return "mode"
            if lo < NODE_OFFSET + N_COUNTRIES:
                return "target"
            return "other"
    return "length"


def round_metrics(start: ts.GameState, end: ts.GameState, mover: ts.Player,
                  moves: Sequence[Tuple[int, str]]) -> Dict[str, Any]:
    """The round's own content, from the mover's side."""
    _, bg, _ = country_table()
    s = _side(mover)
    a, b = influence(start), influence(end)
    placed = np.maximum(b[s] - a[s], 0)
    own0, opp0 = controlled(a, s), controlled(a, 1 - s)
    flat = [x for x, _ in moves]
    coup_target = None
    if COUP in flat:
        after = [x for x in flat[flat.index(COUP) + 1:] if NODE_OFFSET <= x < NODE_OFFSET + N_COUNTRIES]
        coup_target = after[0] - NODE_OFFSET if after else None
    return {"placed": int(placed.sum()), "touched": int((placed > 0).sum()),
            "into_own": int(placed[own0].sum()), "into_opp": int(placed[opp0].sum()),
            "into_free": int(placed[~own0 & ~opp0].sum()), "into_bg": int(placed[bg].sum()),
            "opp_removed": int(np.maximum(a[1 - s] - b[1 - s], 0).sum()),
            "opp_bg_broken": int((opp0 & bg & ~controlled(b, 1 - s)).sum()),
            "coup": coup_target is not None, "coup_target": coup_target,
            "coup_bg": bool(coup_target is not None and bg[coup_target])}


def _coup_break(ops: int, stab: int, mine: int, theirs: int) -> float:
    """P(a coup with `ops` by the opponent breaks the mover's control), over the die."""
    broken = 0
    for d in range(1, 7):
        x = d + ops - 2 * stab
        if x <= 0:
            continue
        m, t = mine - min(x, mine), theirs + max(0, x - mine)
        broken += (m - t) < stab
    return broken / 6.0


def _realign_loss(net: int, defender: int) -> float:
    """Expected defender influence removed by one realignment roll at the attacker's `net`."""
    total = 0
    for a in range(1, 7):
        for b in range(1, 7):
            d = a - b + net
            total += min(d, defender) if d > 0 else 0
    return total / 36.0


def exposure(st: ts.GameState, mover: ts.Player, ops: int = 3) -> Dict[str, float]:
    """The mover's controlled battlegrounds open to the opponent at this DEFCON, and how exposed."""
    stab, bg, _ = country_table()
    region = _region()
    s = _side(mover)
    opp = ts.Player.USSR if mover == ts.Player.US else ts.Player.US
    inf = influence(st)
    defcon = int(st.defcon)
    mine = controlled(inf, s) & bg
    open_ = np.array([defcon >= REGION_DEFCON.get(r, 0) for r in region])
    net = net_modifiers(st, opp)
    coup = real = 0.0
    for c in np.flatnonzero(mine & open_):
        if defcon > 2:
            coup += _coup_break(ops, int(stab[c]), int(inf[s, c]), int(inf[1 - s, c]))
        real += _realign_loss(int(net[c]), int(inf[s, c]))
    return {"bg_held": int(mine.sum()), "bg_open": int((mine & open_).sum()),
            "coup_exposure": coup, "realign_exposure": real}


def play_replies(rounds: Sequence[Dict[str, Any]], act: PolicyFn, pairs: int, seed: int,
                 chunk: int = 1024, max_steps: int = 400) -> List[Dict[str, Dict[str, float]]]:
    """Per round, per branch: the reply's damage and the final score, averaged over `pairs`."""
    from ai.search.dmcts import determinize

    _, bg, _ = country_table()
    out: List[Dict[str, Dict[str, float]]] = []
    per = len(BRANCHES) * pairs
    step = max(1, chunk // per)
    for lo in range(0, len(rounds), step):
        group = rounds[lo:lo + step]
        starts: List[ts.GameState] = []
        movers: List[ts.Player] = []
        for gi, r in enumerate(group):
            for k in range(pairs):
                pair_seed = (seed * 1_000_003 + (lo + gi)) * 1_009 + k
                for b in BRANCHES:
                    end = r[b]["end"]
                    rng = random.Random(pair_seed)
                    st = end if ts.Engine.is_terminal(end) else determinize(end, r["mover"], rng)
                    st = st.clone()
                    st.rng_state = rng.getrandbits(64) % _UINT64
                    starts.append(st)
                    movers.append(r["mover"])
        # Step to the end of the opponent's reply round; then play on to the end of the game.
        m = len(starts)
        runner = ts.VectorizedBatchRunner(m, seed + lo)
        for j, st in enumerate(starts):
            runner.set_state(j, st)
        runner.refresh_all()
        keys = [_ar_key(st) for st in starts]
        after: List[Optional[ts.GameState]] = [st.clone() if ts.Engine.is_terminal(st) else None for st in starts]
        for _ in range(max_steps):
            states = [runner.get_state(j) for j in range(m)]
            for j, st in enumerate(states):
                if after[j] is None and (ts.Engine.is_terminal(st) or _ar_key(st) != keys[j]):
                    after[j] = st.clone()
            if all(a is not None for a in after):
                break
            obs = np.asarray(runner.get_observations())
            masks = np.asarray(runner.get_action_masks())
            live = [j for j in range(m) if masks[j].any() and after[j] is None]
            acts = np.zeros(m, dtype=np.int32)
            if live:
                acts[live] = act(obs[live], masks[live])
            runner.step_flat_all(acts.tolist(), auto_advance=True)
        replies = [a if a is not None else runner.get_state(j).clone() for j, a in enumerate(after)]
        ends = play_out(replies, movers, act, seed + lo)
        for gi, r in enumerate(group):
            s = _side(r["mover"])
            sign = 1 if s == 0 else -1
            row: Dict[str, Dict[str, float]] = {}
            for bi, b in enumerate(BRANCHES):
                pre = r[b]["end"]
                inf0 = influence(pre)
                bg0 = int((controlled(inf0, s) & bg).sum())
                lost, removed, vp, score = [], [], [], []
                for k in range(pairs):
                    j = gi * per + k * len(BRANCHES) + bi
                    inf1 = influence(replies[j])
                    lost.append(bg0 - int((controlled(inf1, s) & bg).sum()))
                    removed.append(int(np.maximum(inf0[s] - inf1[s], 0).sum()))
                    vp.append(sign * (int(replies[j].victory_points) - int(pre.victory_points)))
                    score.append(ends[j]["score"])
                row[b] = {"reply_bg_lost": float(np.mean(lost)), "reply_inf_removed": float(np.mean(removed)),
                          "reply_vp": float(np.mean(vp)), "score": float(np.mean(score))}
            out.append(row)
    return out


def summarise(rounds: Sequence[Dict[str, Any]], replies: Sequence[Dict[str, Dict[str, float]]]
              ) -> List[Dict[str, Any]]:
    """One JSON-ready row per disagreement."""
    _, _, names = country_table()
    rows = []
    for r, rep in zip(rounds, replies):
        st, mover = r["start"], r["mover"]
        row: Dict[str, Any] = {"side": "US" if mover == ts.Player.US else "USSR", "turn": int(st.turn),
                               "ar": int(st.action_round), "defcon": int(st.defcon),
                               "vp": int(st.victory_points), "rounds_left": rounds_left(st),
                               "replied": opponent_replies(st),
                               "diverge": first_divergence(r["net"]["moves"], r["search"]["moves"]),
                               "link": position_link(st)}
        for b in BRANCHES:
            m = round_metrics(st, r[b]["end"], mover, r[b]["moves"])
            if m["coup_target"] is not None:
                m["coup_target"] = names[m["coup_target"]]
            row[b] = {"moves": [nm for _, nm in r[b]["moves"]], **m,
                      **exposure(r[b]["end"], mover), **rep[b]}
        rows.append(row)
    return rows


METRICS = (
    ("score", "final score (mover's win %)", 100.0),
    ("reply_bg_lost", "battlegrounds lost to the reply", 1.0),
    ("reply_inf_removed", "own influence removed by the reply", 1.0),
    ("reply_vp", "VP swing in the reply (mover's side)", 1.0),
    ("coup_exposure", "coup exposure (P(3-Ops coup breaks), summed)", 1.0),
    ("realign_exposure", "realign exposure (influence per roll, summed)", 1.0),
    ("bg_held", "battlegrounds held after the round", 1.0),
    ("into_own", "points into own controlled countries (defence)", 1.0),
    ("into_free", "points into uncontrolled countries", 1.0),
    ("into_opp", "points into opponent-controlled countries", 1.0),
    ("touched", "countries touched", 1.0),
    ("opp_removed", "opponent influence removed", 1.0),
    ("opp_bg_broken", "opponent battlegrounds broken", 1.0),
    ("coup", "couped (%)", 100.0),
    ("coup_bg", "couped a battleground (%)", 100.0),
)


def _paired(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float, float, float]:
    a = np.array([float(r["net"][key]) for r in rows])
    b = np.array([float(r["search"][key]) for r in rows])
    d = b - a
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(a.mean()), float(b.mean()), float(d.mean()), se


def _table(rows: Sequence[Dict[str, Any]], keys: Sequence[Tuple[str, str, float]]) -> List[str]:
    out = ["| metric | net | search | search − net |", "|:---|---:|---:|---:|"]
    for key, label, scale in keys:
        a, b, d, se = _paired(rows, key)
        out.append(f"| {label} | {scale * a:.2f} | {scale * b:.2f} | {scale * d:+.2f} ± {scale * se:.2f} |")
    return out


def report(data: Dict[str, Any], meta: Dict[str, Any], top: int = 15) -> Tuple[str, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = data["rows"]
    n_all = int(data["rounds"])
    out = [f"# Reply probe — {meta.get('model', '?')} against {meta.get('search', 'search')}", "",
           f"{n_all} action rounds from the net's greedy self-play; search and net disagree in "
           f"**{len(rows)}** ({100 * len(rows) / max(n_all, 1):.0f}%). Every number below is over the "
           f"disagreements only, paired by round; ± one standard error. Reply and score: "
           f"{meta.get('pairs', '?')} playouts per branch, the mover's unseen cards redealt per pair, "
           f"the net on both sides.", ""]
    summary: Dict[str, Any] = {"meta": meta, "rounds": n_all, "disagreements": len(rows)}
    if len(rows) < 2:
        return "\n".join(out + ["too few disagreements to report"]) + "\n", summary
    out += ["## All disagreements", ""] + _table(rows, METRICS) + [""]
    summary["all"] = {k: _paired(rows, k) for k, _, _ in METRICS}

    out += ["## Is search's gain about the reply?", "",
            "The search branch's score gain, split by whether its round left less, the same or more "
            "exposure (coup + realign) than the net's.", "",
            "| search's exposure vs the net's | rounds | search − net, score | search − net, BGs lost to reply |",
            "|:---|---:|---:|---:|"]

    def exp(r: Dict[str, Any], b: str) -> float:
        return float(r[b]["coup_exposure"] + r[b]["realign_exposure"])

    for label, sel in (("less", [r for r in rows if exp(r, "search") < exp(r, "net") - 1e-9]),
                       ("same", [r for r in rows if abs(exp(r, "search") - exp(r, "net")) <= 1e-9]),
                       ("more", [r for r in rows if exp(r, "search") > exp(r, "net") + 1e-9])):
        if len(sel) < 2:
            out.append(f"| {label} | {len(sel)} | — | — |")
            continue
        _, _, d, se = _paired(sel, "score")
        _, _, d2, se2 = _paired(sel, "reply_bg_lost")
        out.append(f"| {label} | {len(sel)} | {100 * d:+.1f} ± {100 * se:.1f} | {d2:+.2f} ± {se2:.2f} |")
    out.append("")

    short = [m for m in METRICS if m[0] in ("score", "reply_bg_lost", "coup_exposure", "into_own", "coup")]
    for title, field, values in (("By where the rounds first differ", "diverge", ("card", "mode", "target", "other", "length")),
                                 ("By side", "side", ("US", "USSR"))):
        out += [f"## {title}", ""]
        for v in values:
            sel = [r for r in rows if r[field] == v]
            if len(sel) < 2:
                continue
            out += [f"**{v}** — {len(sel)} rounds", ""] + _table(sel, short) + [""]
    out += ["## By the mover's action rounds left this turn", "",
            "Rounds after this one (the eighth not counted). Only the US's last round goes unanswered "
            "before the turn ends.", ""]
    buckets = (("last round, US (no reply this turn)", lambda left, rep: not rep),
               ("last round, USSR (US replies)", lambda left, rep: left <= 0 and rep),
               ("1 left", lambda left, rep: left == 1),
               ("2–3 left", lambda left, rep: 2 <= left <= 3),
               ("4+ left", lambda left, rep: left >= 4))
    every = data.get("all", [])
    for label, inb in buckets:
        sel = [r for r in rows if inb(r["rounds_left"], r["replied"])]
        total = sum(1 for left, rep in every if inb(left, rep))
        rate = f", disagreeing in {100 * len(sel) / total:.0f}% of {total}" if total else ""
        if len(sel) >= 2:
            out += [f"**{label}** — {len(sel)} rounds{rate}", ""] + _table(sel, short) + [""]
    out += ["## By era", ""]
    for label, lo, hi in (("Early War (turns 1–3)", 1, 3), ("Mid War (4–7)", 4, 7), ("Late War (8–10)", 8, 10)):
        sel = [r for r in rows if lo <= r["turn"] <= hi]
        if len(sel) >= 2:
            out += [f"**{label}** — {len(sel)} rounds", ""] + _table(sel, short) + [""]

    gain = sorted(rows, key=lambda r: r["net"]["score"] - r["search"]["score"])
    out += [f"## The {top} rounds where search gains most", "",
            "| side | turn | DEFCON | net's round | search's round | score gain | BGs lost to reply, net → search | |",
            "|:---|---:|---:|:---|:---|---:|:---|:---|"]
    for r in gain[:top]:
        out.append(f"| {r['side']} | {r['turn']}.{r['ar']} | {r['defcon']} | {'; '.join(r['net']['moves'])} | "
                   f"{'; '.join(r['search']['moves'])} | {100 * (r['search']['score'] - r['net']['score']):+.0f} | "
                   f"{r['net']['reply_bg_lost']:.1f} → {r['search']['reply_bg_lost']:.1f} | [position]({r['link']}) |")
    out.append("")
    return "\n".join(out) + "\n", summary
