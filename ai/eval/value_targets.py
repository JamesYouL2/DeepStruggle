"""Does search's value of a position predict the result better than the critic's?

The target-forms probe (research/log/E7_search_target_forms.md) found that search's MOVE choices,
taken one decision at a time, carry no measurable gain over the prior. Search's play-time strength
may come instead from better evaluation -- backing the critic up through lookahead -- which a policy
target cannot carry but a value target could. This probe measures that directly:

* positions from the model's own play (target_forms.collect_positions: rollout temperature, 1 in 8
  non-forced decisions, whole games);
* at each, the critic's value and search's root value (honest search, the mover's side, the root's
  network value and every simulation's backed-up value averaged) at each requested budget;
* the reference: the mean result of `playouts` games played to the end from the position by the
  network (greedy), each with the cards the mover cannot see redealt and its own dice.

Per form: mean squared error against the reference with the reference's own sampling noise
subtracted (so it estimates the error against the true expected result under the network's play),
the error relative to the critic's, the correlation, and the mean bias -- by segment and by turn.

What the reference is: the expected result if the NETWORK plays on, which is what the critic is
trained to predict. Search's value assumes better play below the root; where it differs from the
reference in a way that reflects better play rather than error, this probe counts it as error.
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

import ts_engine as ts
from ai.eval.target_forms import Position, collect_positions
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig
from ai.search.dmcts import determinize

_UINT64 = 1 << 64
_US = int(ts.Player.US)
DEFAULT_FORMS = ("search@16", "search@32", "search@64", "search@128", "search@32,pt1.5")


def _search_values(model, positions: Sequence[Position], form: str, seed: int,
                   device="cpu") -> List[float]:
    """Root value from the mover's side: (network value + backed-up values) / (1 + visits)."""
    sims_s, *opts = form.split("@", 1)[1].split(",")
    pt = 1.0
    for o in opts:
        if o.startswith("pt"):
            pt = float(o[2:])
        else:
            raise ValueError(f"unknown option {o!r} in {form!r}")
    cfg = BatchedMCTSConfig(simulations=int(sims_s), temperature=0.0, auto_advance=True,
                            advance_root=False, determinize=True, node_filter="all",
                            subsample=1.0, seed=seed, prior_temp=pt)
    roots = BatchedMCTS(model, device=device, config=cfg, featurise_capacity=4096)._search(
        [p.state for p in positions])
    out: List[float] = []
    for p, r in zip(positions, roots):
        if r is None:
            out.append(p.value_mover)
            continue
        v_us = (float(r.value_us) + float(sum(r.w))) / (1.0 + float(sum(r.n))) if r.actions \
            else float(r.value_us)
        out.append(v_us if p.mover == _US else -v_us)
    return out


def playout_results(model, positions: Sequence[Position], playouts: int, seed: int,
                    device="cpu", batch: int = 2048, max_steps: int = 4000) -> np.ndarray:
    """(positions, playouts) results in [-1, 1] from each mover's side: the network plays both
    sides greedily to the end, each playout from its own redeal of the mover's unseen cards."""
    starts: List[ts.GameState] = []
    for i, p in enumerate(positions):
        for j in range(playouts):
            r = random.Random((seed * 1_000_003 + i * 7919 + j) % (2 ** 61))
            w = determinize(p.state.clone(), ts.Player(p.mover), r)
            w.rng_state = r.getrandbits(64) % _UINT64
            starts.append(w)
    res = np.zeros(len(starts))
    for lo in range(0, len(starts), batch):
        chunk = starts[lo:lo + batch]
        runner = ts.VectorizedBatchRunner(len(chunk), seed)
        for i, s in enumerate(chunk):
            runner.set_state(i, s)
        runner.refresh_all()
        for _ in range(max_steps):
            terms = np.asarray(runner.get_terminals())
            if terms.all():
                break
            obs = torch.from_numpy(np.array(runner.get_observations(), copy=True)).to(device)
            masks = np.array(runner.get_action_masks(), copy=True)
            with torch.no_grad():
                lg, _v, _ = model(obs, torch.from_numpy(masks).bool().to(device))
            acts = lg.argmax(dim=1).cpu().numpy().tolist()
            runner.step_flat_all([0 if terms[i] else int(a) for i, a in enumerate(acts)], True)
        for i in range(len(chunk)):
            res[lo + i] = float(ts.Engine.get_terminal_utility(runner.get_state(i)))
    res = res.reshape(len(positions), playouts)
    sign = np.array([1.0 if p.mover == _US else -1.0 for p in positions])[:, None]
    return res * sign


def rows(positions: Sequence[Position], values: Dict[str, List[float]], results: np.ndarray
         ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(positions):
        r = results[i]
        out.append({"segment": p.segment, "turn": p.turn, "mover": p.mover,
                    "z_mean": float(r.mean()), "z_var_of_mean": float(r.var(ddof=1) / len(r)),
                    "values": {"critic": p.value_mover, **{f: v[i] for f, v in values.items()}}})
    return out


def _metrics(rs: Sequence[Dict[str, Any]], form: str) -> Dict[str, float]:
    v = np.array([r["values"][form] for r in rs])
    z = np.array([r["z_mean"] for r in rs])
    noise = float(np.mean([r["z_var_of_mean"] for r in rs]))
    mse = float(np.mean((v - z) ** 2)) - noise
    corr = float(np.corrcoef(v, z)[0, 1]) if len(rs) > 2 and v.std() > 0 else float("nan")
    return {"mse": mse, "corr": corr, "bias": float(np.mean(v - z)), "n": float(len(rs))}


def report(all_rows: Sequence[Dict[str, Any]], forms: Sequence[str]) -> str:
    forms = ["critic"] + [f for f in forms if f != "critic"]
    n = len(all_rows)
    base = _metrics(all_rows, "critic")["mse"]
    lines = ["# Value targets: search's root value vs the critic's, against playout results", "",
             f"{n} positions. Reference: mean result of greedy network playouts to the end, each with "
             "the mover's unseen cards redealt; MSE has the reference's own sampling noise subtracted, "
             "so it estimates the error against the expected result. Values from the mover's side, "
             "in [-1, 1].", "",
             "| value | MSE | vs critic | correlation | mean bias |", "|:---|---:|---:|---:|---:|"]
    # Paired bootstrap of the MSE difference against the critic, for an uncertainty.
    rng = np.random.default_rng(0)
    z = np.array([r["z_mean"] for r in all_rows])
    noise = np.array([r["z_var_of_mean"] for r in all_rows])
    vc = np.array([r["values"]["critic"] for r in all_rows])
    for f in forms:
        m = _metrics(all_rows, f)
        if f == "critic":
            rel = "—"
        else:
            vf = np.array([r["values"][f] for r in all_rows])
            d = (vf - z) ** 2 - (vc - z) ** 2
            boots = [d[rng.integers(0, n, n)].mean() for _ in range(400)]
            rel = f"{np.mean(d):+.4f} ± {np.std(boots):.4f}"
        lines.append(f"| {f} | {m['mse']:.4f} | {rel} | {m['corr']:.3f} | {m['bias']:+.3f} |")
    for key, title, groups in (
            ("segment", "By segment: MSE", sorted({r['segment'] for r in all_rows})),
            ("turn", "By turn: MSE", [(1, 3), (4, 7), (8, 10)])):
        lines += ["", f"## {title}", "", "| group | positions | " + " | ".join(forms) + " |",
                  "|:---|---:|" + "---:|" * len(forms)]
        for gr in groups:
            if key == "segment":
                rs = [r for r in all_rows if r["segment"] == gr]
                name = str(gr)
            else:
                lo, hi = gr
                rs = [r for r in all_rows if lo <= r["turn"] <= hi]
                name = f"turns {lo}-{hi}"
            if len(rs) < 3:
                continue
            lines.append(f"| {name} | {len(rs)} | " + " | ".join(f"{_metrics(rs, f)['mse']:.4f}"
                                                               for f in forms) + " |")
    lines += ["", f"Critic MSE {base:.4f} is the bar: a value with lower MSE predicts the result of the "
              "network's own play better than the critic does, so it carries information a value "
              "target could teach."]
    return "\n".join(lines) + "\n"


def dump(rows_: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows_:
            fh.write(json.dumps(r) + "\n")


def build(model, n: int, seed: int, forms: Sequence[str], playouts: int) -> List[Dict[str, Any]]:
    pos = collect_positions(model, n, seed)
    values = {f: _search_values(model, pos, f, seed + 17) for f in forms}
    res = playout_results(model, pos, playouts, seed)
    return rows(pos, values, res)


def finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
