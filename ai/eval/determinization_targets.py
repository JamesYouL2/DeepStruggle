"""How much of a determinized search target is the world it happened to sample.

Online search distillation (P15-X4b, E4-28; `ai/training/nash_pg.py`) trains the policy toward
the root visit counts of a 64-simulation search that resamples the hidden cards ONCE per search
(`BatchedMCTS`, `determinize=True`) and spends every simulation in that one world. This probe
takes positions from the policy's own training distribution and asks, per decision type:

* **noise** -- do two one-world targets for the same position agree? And how far is a one-world
  target from the average over the other worlds?
* **an equal-budget alternative** -- is 8 worlds x 8 simulations closer to the 8 x 64 average
  than one world x 64 is?
* **world sensitivity** -- how often does the best move change with the sampled world? These are
  the decisions where a target built from sampled worlds is most suspect (strategy fusion,
  `ai/search/dmcts.py`).
* **information value** -- how often does a privileged search (the real hidden cards) disagree
  with the 8-world average?

What it cannot say: the sum over worlds is the expectation of the one-world target, so a
cross-entropy trained on one-world targets converges toward the same average. 1-vs-8 worlds
measures how noisy the target is, never whether the average itself is biased. World sensitivity
is the pointer to where bias could live; it is not a measurement of it.

Search configuration is the training searcher's (`nash_pg.py`, `_searcher`): temperature 0,
auto_advance, advance_root False, every decision type, no root noise. Visits the real legal mask
rejects are dropped before anything is compared, as `_search_targets` drops them.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder

#: Matches the training searcher's subsample: 1 in 8 decisions gets a target.
SAMPLE_PROB = 0.125


@dataclass
class Position:
    save: str            # GameState.to_save_json()
    decision_type: str
    side: str            # "US" / "USSR", the deciding player
    turn: int
    legal: List[int]     # flat indices legal in the real state


def _decider(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def collect_positions(model_path: str, n: int, seed: int, games: int = 64,
                      temperature: float = 1.0) -> List[Position]:
    """`n` positions from games the model plays against itself at the rollout temperature.

    Every non-forced decision is a candidate with probability SAMPLE_PROB, as the training
    searcher's subsample would pick it; whole batches of games are played to the end and `n` are
    drawn uniformly from all candidates, so the sample covers every phase of the game in
    proportion to how often training searches it, not just the opening."""
    from tools.lib.player_agent import OnnxAgent

    agent = OnnxAgent(model_path, seed=seed)
    rng = np.random.default_rng(seed)
    out: List[Position] = []
    base = seed
    while len(out) < 2 * n:
        runner = ts.VectorizedBatchRunner(games, base)
        for i in range(games):
            runner.reset_game(i, base * 1000 + i)
        runner.refresh_all()
        base += 1
        active = np.ones(games, dtype=bool)
        for _ in range(4000):
            terms = np.array(runner.get_terminals())
            active &= ~terms
            if not active.any():
                break
            obs = np.asarray(runner.get_observations())
            masks = np.asarray(runner.get_action_masks())
            rows = np.where(active)[0]
            acts = np.zeros(games, dtype=np.int32)
            acts[rows] = agent.act_batch(obs[rows], masks[rows], temperature, temperature <= 0.05)
            for i in rows:
                legal = np.flatnonzero(masks[i])
                if len(legal) >= 2 and rng.random() < SAMPLE_PROB:
                    st = runner.get_state(int(i))
                    out.append(Position(
                        save=st.to_save_json(),
                        decision_type=ts.DecisionType(int(st.ctx().decision_type)).name,
                        side="US" if _decider(st) == ts.Player.US else "USSR",
                        turn=int(st.turn), legal=[int(a) for a in legal]))
            runner.step_flat_all(acts.tolist(), auto_advance=True)
    keep = rng.choice(len(out), size=n, replace=False)
    return [out[int(i)] for i in sorted(keep)]


def _searcher(model_path: str, sims: int, determinize: bool, seed: int,
              dirichlet_frac: float = 0.0, dirichlet_alpha: float = 1.0):
    from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
    from tools.lib.player_agent import OnnxAgent

    cfg = BatchedMCTSConfig(simulations=sims, temperature=0.0, auto_advance=True,
                            advance_root=False, determinize=determinize, seed=seed,
                            dirichlet_frac=dirichlet_frac, dirichlet_alpha=dirichlet_alpha)
    return BatchedMCTS(OnnxAgent(model_path).as_module(), device="cpu", config=cfg)


def _on_legal(actions: Sequence[int], visits: np.ndarray, legal: Sequence[int]) -> List[float]:
    """Visits re-indexed onto the real legal list; visits on actions illegal there are dropped."""
    pos = {a: k for k, a in enumerate(legal)}
    v = [0.0] * len(legal)
    for a, x in zip(actions, visits):
        if int(a) in pos:
            v[pos[int(a)]] += float(x)
    return v


def search_positions(model_path: str, positions: Sequence[Position], worlds: int = 8,
                     sims: int = 64, small_sims: int = 8, seed: int = 0,
                     chunk: int = 16, dirichlet_frac: float = 0.0, dirichlet_alpha: float = 1.0,
                     indices: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """Per position: `worlds` one-world targets at `sims`, `worlds` at `small_sims` (the
    equal-budget split), a privileged target at `sims`, and the network's prior.

    `dirichlet_frac` > 0 mixes root noise into every determinized search (each world draws its
    own), the blind-spot setting: a low-prior move then gets visited, and only one with a real
    value edge stays on top across worlds. The privileged search stays noise-free.
    `indices` are the positions' places in the full list, recorded so a row can be traced back."""
    from tools.lib.player_agent import OnnxAgent

    det = _searcher(model_path, sims, True, seed, dirichlet_frac, dirichlet_alpha)
    det_small = _searcher(model_path, small_sims, True, seed + 1, dirichlet_frac, dirichlet_alpha)
    priv = _searcher(model_path, sims, False, seed + 2)
    net = OnnxAgent(model_path)
    rows: List[Dict[str, Any]] = []
    for c0 in range(0, len(positions), chunk):
        part = positions[c0:c0 + chunk]
        states = [ts.state_from_save_json(p.save) for p in part]
        rep = [s.clone() for s in states for _ in range(worlds)]
        r_det = det.run(rep)
        r_small = det_small.run([s.clone() for s in rep])
        r_priv = priv.run([s.clone() for s in states])
        for k, (p, st) in enumerate(zip(part, states)):
            me = _decider(st)
            obs = np.asarray(ts.extract_observation(st, me), dtype=np.float32)[None, :]
            mask = np.asarray(ActionEncoder.get_legal_mask(st, False), dtype=np.uint8)[None, :]
            lg = net.logits(obs, mask)[0][p.legal]
            prior = np.exp(lg - lg.max())
            rows.append({
                "pos_index": int(indices[c0 + k]) if indices is not None else c0 + k,
                "decision_type": p.decision_type, "side": p.side, "turn": p.turn,
                "legal": p.legal,
                "prior": (prior / prior.sum()).tolist(),
                "worlds": [_on_legal(*r_det[k * worlds + w], p.legal) for w in range(worlds)],
                "worlds_small": [_on_legal(*r_small[k * worlds + w], p.legal) for w in range(worlds)],
                "privileged": _on_legal(*r_priv[k], p.legal),
            })
    return rows


# ---------------------------------------------------------------------------------- report

def _norm(v: Union[Sequence[float], np.ndarray]) -> Optional[np.ndarray]:
    a = np.asarray(v, dtype=float)
    s = a.sum()
    return a / s if s > 0 else None


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def _row_metrics(r: Dict[str, Any]) -> Optional[Dict[str, float]]:
    W = [_norm(w) for w in r["worlds"]]
    if any(w is None for w in W):
        return None
    Wn: List[np.ndarray] = [w for w in W if w is not None]
    n = len(Wn)
    total = np.sum(Wn, axis=0)
    small = _norm(np.sum(r["worlds_small"], axis=0))
    priv = _norm(r["privileged"])
    avg = total / n
    best = int(avg.argmax())
    pair_agree = np.mean([Wn[i].argmax() == Wn[j].argmax() for i in range(n) for j in range(i + 1, n)])
    loo = [(total - Wn[i]) / (n - 1) for i in range(n)]
    argmaxes = [int(w.argmax()) for w in Wn]
    counts = np.bincount(argmaxes, minlength=len(avg))
    m = {
        "one_world_pair_agree": float(pair_agree),
        "one_world_vs_rest_agree": float(np.mean([Wn[i].argmax() == loo[i].argmax() for i in range(n)])),
        "one_world_vs_rest_tv": float(np.mean([_tv(Wn[i], loo[i]) for i in range(n)])),
        "world_sensitive": float(len(set(argmaxes)) > 1),
        "modal_world_share": float(counts.max() / n),
        "prior_vs_avg_agree": float(int(np.argmax(r["prior"])) == best),
        "prior_vs_avg_tv": _tv(np.asarray(r["prior"]), avg),
        "changes_top": float(int(np.argmax(r["prior"])) != best),
        "blind_spot": float(is_blind_spot(r["prior"], avg, counts, n)),
    }
    if small is not None:
        m["split_vs_avg_agree"] = float(int(small.argmax()) == best)
        m["split_vs_avg_tv"] = _tv(small, avg)
    if priv is not None:
        m["privileged_vs_avg_agree"] = float(int(priv.argmax()) == best)
        m["privileged_vs_avg_tv"] = _tv(priv, avg)
    return m


#: A blind-spot candidate: search's top move has a prior under BLIND_PRIOR, and at least
#: BLIND_WORLDS of the worlds pick it -- noise alone, drawn afresh in each world, rarely does that.
BLIND_PRIOR = 0.05
BLIND_WORLDS = 5 / 8


def is_blind_spot(prior: Sequence[float], avg: np.ndarray, counts: np.ndarray, n: int) -> bool:
    best = int(avg.argmax())
    return bool(prior[best] < BLIND_PRIOR and counts[best] / n >= BLIND_WORLDS
                and int(np.argmax(prior)) != best)


def blind_spots(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The candidate positions, each with the move search found and the prior's own top move."""
    out = []
    for r in rows:
        W = [_norm(w) for w in r["worlds"]]
        if any(w is None for w in W):
            continue
        Wn = [w for w in W if w is not None]
        avg = np.mean(Wn, axis=0)
        counts = np.bincount([int(w.argmax()) for w in Wn], minlength=len(avg))
        if is_blind_spot(r["prior"], avg, counts, len(Wn)):
            best = int(avg.argmax())
            top = int(np.argmax(r["prior"]))
            out.append({"pos_index": r["pos_index"], "decision_type": r["decision_type"],
                        "side": r["side"], "turn": r["turn"],
                        "search_move": r["legal"][best], "prior_of_search_move": r["prior"][best],
                        "prior_move": r["legal"][top], "prior_of_prior_move": r["prior"][top],
                        "worlds_agreeing": int(counts[best])})
    return out


def verify_blind_spots(model_path: str, positions: Sequence[Position],
                       candidates: Sequence[Dict[str, Any]], pairs: int = 256,
                       seed: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
    """Paired playouts at each candidate: the model's own move against the move search found,
    with the hidden cards redealt per pair (`ai/eval/branch_oracle.py`). A blind spot is real only
    if the playouts agree with the search."""
    from ai.eval.branch_oracle import onnx_policy, play_branches
    from ai.eval.branch_oracle import report as branch_report

    act, _ = onnx_policy(model_path)
    out: List[Dict[str, Any]] = []
    lines = ["| position | type | side | turn | search's move | its prior | worlds | search − policy, paired |",
             "|---:|---|---|---:|---|---:|---:|---:|"]
    diffs = []
    for c in candidates:
        st = ts.state_from_save_json(positions[c["pos_index"]].save)
        rows = play_branches(st, {"policy": [], "search": [c["search_move"]]}, range(pairs), act,
                             seed=seed + c["pos_index"])
        _, js = branch_report(rows)
        d = js["search"]
        name = ActionEncoder.get_action_name(st, int(c["search_move"]))
        out.append({**c, "search_move_name": name, "diff": d["diff"], "se": d["se"]})
        diffs.append(d["diff"])
        lines.append(f"| {c['pos_index']} | {c['decision_type']} | {c['side']} | {c['turn']} | {name} | "
                     f"{c['prior_of_search_move']:.3f} | {c['worlds_agreeing']}/8 | "
                     f"{100 * d['diff']:+.1f} ± {100 * d['se']:.1f} |")
    if diffs:
        m = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
        lines.append(f"\nMean over {len(diffs)} candidates: **{100 * m:+.1f} ± {100 * se:.1f}** points for the move search found.")
    return "\n".join(lines) + "\n", out


COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("one_world_pair_agree", "2 one-world targets agree"),
    ("one_world_vs_rest_agree", "one world vs rest, argmax"),
    ("one_world_vs_rest_tv", "one world vs rest, TV"),
    ("split_vs_avg_agree", "8x8 split vs avg, argmax"),
    ("split_vs_avg_tv", "8x8 split vs avg, TV"),
    ("world_sensitive", "best move varies by world"),
    ("modal_world_share", "share of worlds on modal move"),
    ("privileged_vs_avg_agree", "privileged vs avg, argmax"),
    ("prior_vs_avg_agree", "net prior vs avg, argmax"),
    ("changes_top", "search changes the top move"),
    ("blind_spot", "blind-spot candidate"),
)


def report(rows: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Markdown and JSON summaries, overall and by decision type, side and era."""
    groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    dropped = 0
    for r in rows:
        m = _row_metrics(r)
        if m is None:
            dropped += 1
            continue
        era = "early" if r["turn"] <= 3 else ("mid" if r["turn"] <= 7 else "late")
        for g in ("all", f"type:{r['decision_type']}", f"side:{r['side']}", f"era:{era}"):
            groups[g].append(m)

    def mean(ms: List[Dict[str, float]], k: str) -> Optional[float]:
        xs = [m[k] for m in ms if k in m]
        return float(np.mean(xs)) if xs else None

    sizes: Dict[str, int] = {g: len(ms) for g, ms in groups.items()}
    means: Dict[str, Dict[str, Optional[float]]] = {
        g: {k: mean(ms, k) for k, _ in COLUMNS} for g, ms in groups.items()}
    summary: Dict[str, Dict[str, Any]] = {g: {"n": sizes[g], **means[g]} for g in groups}
    order = sorted(groups, key=lambda g: (g != "all", g.split(":")[0], -sizes[g]))
    head = "| group | n | " + " | ".join(lbl for _, lbl in COLUMNS) + " |"
    lines = [
        "# One world against eight: determinized search targets",
        "",
        f"{sum(len(v) for g, v in groups.items() if g == 'all')} positions "
        f"({dropped} dropped: a world whose visits were all on actions illegal in the real state).",
        "Argmax columns are agreement rates; TV is total-variation distance between visit "
        "distributions (0 = identical, 1 = disjoint). \"avg\" is the mean of the 8 one-world targets.",
        "",
        head, "|" + "---|" * (len(COLUMNS) + 2),
    ]
    def cell(k: str, x: Optional[float]) -> str:
        if x is None:
            return "—"
        return f"{x:.3f}" if k.endswith("_tv") else f"{100 * x:.1f}%"

    for g in order:
        cells = [cell(k, means[g][k]) for k, _ in COLUMNS]
        lines.append(f"| {g} | {sizes[g]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", {"groups": summary, "dropped": dropped}


def dump_positions(path: str, positions: Sequence[Position]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in positions:
            f.write(json.dumps(p.__dict__) + "\n")


def load_positions(path: str) -> List[Position]:
    with open(path, "r", encoding="utf-8") as f:
        return [Position(**json.loads(l)) for l in f if l.strip()]
