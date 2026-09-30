#!/usr/bin/env python3
"""P30: does the net use card <-> country interactions, and does the game need them?

Three measurements on the same self-play positions (split by game where a fit is involved):

1. **Counterfactual sensitivity (is it used?).** At influence placements, a region's scoring card is
   swapped into the mover's hand in the observation -- in for a non-scoring hand card, which goes
   to the deck, so the hand size is unchanged -- and the policy's probability mass on that region's
   countries is compared with the original. The **placebo** makes the same swap with a random
   non-scoring deck card, so the reported shift is net of "any hand change". The value's shift is
   reported the same way. Only the two cards' location slots change; every other input is as
   the engine wrote it.
2. **Critic residual (is it needed, and missed?).** For each region, "my hand holds its scoring
   card" (and "the opponent is known to hold it") times the observation's own live regional
   scoring margin. Logistic fits of the game's winner on held-out games: the critic alone,
   the critic plus additive terms, and the critic plus additive and interaction terms. If the
   interactions improve on the critic, the outcome depends on something the critic has not
   learned. The same fits without the critic say whether the interactions predict the result at
   all.
3. **Mixed derivatives (how much the function mixes the two).** Hutchinson estimates of the
   Frobenius norms of the value's Hessian blocks: board x cards against board x board and
   cards x cards.

    PYTHONPATH=.:build/release python tools/scripts/card_country_interaction_probe.py \\
        --nets a.pt b.pt --policies a.pt --games 1500 --output-json out.json --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from bindings.action_encoder import ActionEncoder  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.scripts.trunk_ownership_probe import collect  # noqa: E402

N_COUNTRIES, BOARD_W, N_CARDS, CARD_W = 84, 26, 110, 14
BOARD_SIZE = N_COUNTRIES * BOARD_W
CARD_OFFSET = BOARD_SIZE
GLOBAL_OFFSET = CARD_OFFSET + N_CARDS * CARD_W
DECK_OR_HIDDEN, MY_HAND, KNOWN_OPP = 0, 1, 2
ACTIVE_SLOT = 13
POINT_NODE = GLOBAL_OFFSET + 72 + 5          # ctx_slots::DECISION_TYPE + DecisionType::POINT_NODE
OP_INFLUENCE = GLOBAL_OFFSET + 72 + 8        # ctx_slots::OP_MODE + INFLUENCE
REGION_MARGIN = GLOBAL_OFFSET + 64           # + Region: my live scoring margin / 20
CHINA = 6
NODE = ActionEncoder.NODE_OFFSET

#: region -> (scoring card id, countries); Region enum order for the six, then Southeast Asia
REGIONS: Dict[str, Tuple[int, int, List[int]]] = {}


def _regions() -> None:
    info = [ts.MapData.get_country_info(c) for c in range(N_COUNTRIES)]
    by_name = {"Europe": (0, 2), "Asia": (1, 1), "Middle East": (2, 3), "Africa": (3, 79),
               "Central America": (4, 37), "South America": (5, 81)}
    for name, (r, card) in by_name.items():
        REGIONS[name] = (r, card, [c for c in range(N_COUNTRIES) if int(info[c]["region"]) == r])
    REGIONS["Southeast Asia"] = (-1, 38, [c for c in range(N_COUNTRIES) if info[c]["in_southeast_asia"]])
    for name, (_, card, _) in REGIONS.items():
        assert ts.CardData.get_card_info(card)["is_scoring"], (name, card)


def _loc(obs: np.ndarray, card: int) -> np.ndarray:
    """(n, 8) location one-hot of `card` (1-based id)."""
    o = CARD_OFFSET + (card - 1) * CARD_W
    return obs[:, o:o + 8].astype(np.float32)


def _set_loc(obs: np.ndarray, i: int, card: int, slot: int) -> None:
    o = CARD_OFFSET + (card - 1) * CARD_W
    obs[i, o:o + 8] = 0
    obs[i, o + slot] = 1


@torch.no_grad()
def _policy_value(model: Any, obs: np.ndarray, mask: np.ndarray, dev: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    ps, vs = [], []
    for k in range(0, obs.shape[0], 4096):
        o = torch.from_numpy(obs[k:k + 4096]).float().to(dev)
        m = torch.from_numpy(mask[k:k + 4096]).to(dev)
        logits, v, _ = model(o, m)
        ps.append(torch.softmax(logits.float(), -1).cpu().numpy())
        vs.append(v.float().reshape(-1).cpu().numpy())
    return np.concatenate(ps), np.concatenate(vs)


def counterfactuals(d: Dict[str, np.ndarray], seed: int) -> Dict[str, Any]:
    """The swapped observations: for each influence placement and region, (base, with S, placebo)."""
    rng = np.random.default_rng(seed)
    obs = d["obs"]
    infl = np.flatnonzero((obs[:, POINT_NODE] > 0.5) & (obs[:, OP_INFLUENCE] > 0.5)
                          & d["mask"][:, NODE:NODE + N_COUNTRIES].any(1))
    scoring = {card for _, card, _ in REGIONS.values()}
    non_scoring = [c for c in range(1, N_CARDS + 1) if c not in scoring and c != CHINA]
    rows: List[Tuple[int, str]] = []
    with_s: List[np.ndarray] = []
    placebo: List[np.ndarray] = []
    for i in infl:
        o = obs[i]
        active = {c for c in range(1, N_CARDS + 1) if o[CARD_OFFSET + (c - 1) * CARD_W + ACTIVE_SLOT] > 0}
        hand = [c for c in non_scoring if o[CARD_OFFSET + (c - 1) * CARD_W + MY_HAND] > 0.5 and c not in active]
        deck = [c for c in non_scoring if o[CARD_OFFSET + (c - 1) * CARD_W + DECK_OR_HIDDEN] > 0.5 and c not in active]
        if not hand or not deck:
            continue
        for name, (_, card, _) in REGIONS.items():
            if o[CARD_OFFSET + (card - 1) * CARD_W + DECK_OR_HIDDEN] < 0.5:
                continue                              # only a card that could be in hand but is not
            x = int(rng.choice(hand))
            y = int(rng.choice(deck))
            a = o[None].copy()
            _set_loc(a, 0, card, MY_HAND)
            _set_loc(a, 0, x, DECK_OR_HIDDEN)
            b = o[None].copy()
            _set_loc(b, 0, y, MY_HAND)
            _set_loc(b, 0, x, DECK_OR_HIDDEN)
            rows.append((int(i), name))
            with_s.append(a[0])
            placebo.append(b[0])
    return {"rows": rows, "with": np.stack(with_s), "placebo": np.stack(placebo), "n_positions": int(len(infl))}


def sensitivity(model: Any, d: Dict[str, np.ndarray], cf: Dict[str, Any], dev: torch.device) -> Dict[str, Any]:
    idx = np.array([i for i, _ in cf["rows"]])
    names = [n for _, n in cf["rows"]]
    mask = d["mask"][idx]
    p0, v0 = _policy_value(model, d["obs"][idx], mask, dev)
    p1, v1 = _policy_value(model, cf["with"], mask, dev)
    p2, v2 = _policy_value(model, cf["placebo"], mask, dev)
    out: Dict[str, Any] = {}
    for name, (_, _, countries) in REGIONS.items():
        sel = np.array([n == name for n in names])
        if not sel.any():
            continue
        cols = [NODE + c for c in countries]
        m0, m1, m2 = (p[sel][:, cols].sum(1) for p in (p0, p1, p2))
        legal = mask[sel][:, cols].any(1)
        dm = (m1 - m0) - (m2 - m0)
        dv = (v1[sel] - v0[sel]) - (v2[sel] - v0[sel])
        n = int(sel.sum())
        out[name] = {"n": n, "legal_frac": float(legal.mean()), "base_mass": float(m0.mean()),
                     "mass_shift": float(dm.mean()), "mass_shift_se": float(dm.std() / math.sqrt(n)),
                     "mass_shift_legal": float(dm[legal].mean()) if legal.any() else None,
                     "placebo_shift": float((m2 - m0).mean()),
                     "value_shift": float(dv.mean()), "value_shift_se": float(dv.std() / math.sqrt(n))}
    return out


def _features(d: Dict[str, np.ndarray], interact: bool, critic: Optional[np.ndarray]) -> np.ndarray:
    obs = d["obs"].astype(np.float32)
    cols: List[np.ndarray] = []
    if critic is not None:
        cols.append(np.arctanh(np.clip(critic, -0.999, 0.999))[:, None])
    for name, (r, card, _) in REGIONS.items():
        if r < 0:
            continue
        mine = _loc(obs, card)[:, MY_HAND]
        theirs = _loc(obs, card)[:, KNOWN_OPP]
        margin = obs[:, REGION_MARGIN + r]
        cols += [mine[:, None], theirs[:, None], margin[:, None]]
        if interact:
            cols += [(mine * margin)[:, None], (theirs * margin)[:, None]]
    return np.concatenate(cols, 1)


def _logistic(x_fit: np.ndarray, y_fit: np.ndarray, x_test: np.ndarray, dev: torch.device) -> np.ndarray:
    mu, sd = x_fit.mean(0), x_fit.std(0) + 1e-6
    xf = torch.from_numpy((x_fit - mu) / sd).float().to(dev)
    xt = torch.from_numpy((x_test - mu) / sd).float().to(dev)
    yf = torch.from_numpy(y_fit).float().to(dev)
    w = torch.zeros(xf.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xf @ w + b, yf) + 1e-4 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(xt @ w + b).cpu().numpy()


def residual_test(d: Dict[str, np.ndarray], critic: Optional[np.ndarray], test: np.ndarray,
                  dev: torch.device, seed: int) -> Dict[str, Any]:
    decided = d["winner"] != 0
    won = (d["winner"] * d["mover"] == 1).astype(np.float32)
    fit_m, test_m = decided & ~test, decided & test
    games = d["game"][test_m]
    out: Dict[str, Any] = {}
    preds: Dict[str, np.ndarray] = {}
    specs = {"additive": (False, False), "additive + interactions": (True, False)} if critic is None else \
        {"critic": None, "critic + additive": (False, True), "critic + additive + interactions": (True, True)}
    for name, spec in specs.items():
        if spec is None:
            assert critic is not None
            x = np.arctanh(np.clip(critic, -0.999, 0.999))[:, None]
        else:
            x = _features(d, spec[0], critic if spec[1] else None)
        preds[name] = _logistic(x[fit_m], won[fit_m], x[test_m], dev)
    y = won[test_m]
    ll = {k: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))) for k, p in preds.items()}
    names = list(specs)
    rng = np.random.default_rng(seed)
    ug = np.unique(games)
    gidx = {g: np.flatnonzero(games == g) for g in ug}
    for k in names:
        out[k] = {"logloss": float(ll[k].mean())}
    # the last model against the one before it: mean improvement and a bootstrap over games
    base, top = names[-2], names[-1]
    diff = ll[base] - ll[top]
    boots = []
    for _ in range(500):
        pick = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug))])
        boots.append(diff[pick].mean())
    out["interaction_gain"] = {"vs": base, "mean": float(diff.mean()),
                               "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out


def hessian_blocks(model: Any, obs: np.ndarray, mask: np.ndarray, dev: torch.device, n: int = 512,
                   probes: int = 8, seed: int = 0) -> Dict[str, float]:
    """Hutchinson estimates of ||H_bb||_F, ||H_cc||_F, ||H_bc||_F for the value, per position (RMS)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.from_numpy(obs[:n]).float().to(dev).requires_grad_(True)
    m = torch.from_numpy(mask[:n]).to(dev)
    b = slice(0, BOARD_SIZE)
    c = slice(CARD_OFFSET, GLOBAL_OFFSET)
    acc = {"board x board": 0.0, "cards x cards": 0.0, "board x cards": 0.0}

    def rad(shape: Tuple[int, int]) -> torch.Tensor:
        return torch.randint(0, 2, shape, generator=g, device=dev).float() * 2 - 1

    model.eval()
    for _ in range(probes):
        _, v, _ = model(x, m)
        (grad,) = torch.autograd.grad(v.float().sum(), x, create_graph=True)
        u1, u2 = rad((n, BOARD_SIZE)), rad((n, BOARD_SIZE))
        w1, w2 = rad((n, GLOBAL_OFFSET - CARD_OFFSET)), rad((n, GLOBAL_OFFSET - CARD_OFFSET))
        (hu1,) = torch.autograd.grad((grad[:, b] * u1).sum(), x, retain_graph=True)
        (hw1,) = torch.autograd.grad((grad[:, c] * w1).sum(), x, retain_graph=False)
        acc["board x board"] += float(((hu1[:, b] * u2).sum(1) ** 2).mean())
        acc["board x cards"] += float(((hu1[:, c] * w2).sum(1) ** 2).mean())
        acc["cards x cards"] += float(((hw1[:, c] * w2).sum(1) ** 2).mean())
    out = {k: math.sqrt(v / probes) for k, v in acc.items()}
    # per-entry scale: a Frobenius norm grows with the block's size, so compare RMS entries too
    sizes = {"board x board": BOARD_SIZE ** 2, "cards x cards": (GLOBAL_OFFSET - CARD_OFFSET) ** 2,
             "board x cards": BOARD_SIZE * (GLOBAL_OFFSET - CARD_OFFSET)}
    out.update({f"{k} (rms entry)": out[k] / math.sqrt(sizes[k]) for k in sizes})
    out["cross / geometric mean of within (rms entry)"] = out["board x cards (rms entry)"] / math.sqrt(
        out["board x board (rms entry)"] * out["cards x cards (rms entry)"])
    return out


def render(res: Dict[str, Any]) -> str:
    L = ["# Card <-> country interaction", "",
         f"Positions: self-play of {', '.join(res['policies'])} at temperature {res['temperature']}; "
         f"{res['positions']} positions, {res['influence_positions']} of them influence placements.", "",
         "## 1. Counterfactual: a region's scoring card swapped into the hand", "",
         "Shift of the policy's mass on that region's countries (percentage points), net of a placebo "
         "swap with a random non-scoring card; value shift likewise (v_win units, -1..1).", ""]
    nets = list(res["sensitivity"])
    regions = list(next(iter(res["sensitivity"].values())).keys())
    L.append("| region | n | legal | " + " | ".join(f"{n}: mass | {n}: value" for n in nets) + " |")
    L.append("|:---|---:|---:|" + "---:|---:|" * len(nets))
    for r in regions:
        first = res["sensitivity"][nets[0]][r]
        cells = []
        for n in nets:
            s = res["sensitivity"][n][r]
            cells.append(f"{100 * s['mass_shift']:+.2f} ± {100 * s['mass_shift_se']:.2f} (base {100 * s['base_mass']:.1f})")
            cells.append(f"{s['value_shift']:+.4f} ± {s['value_shift_se']:.4f}")
        L.append(f"| {r} | {first['n']} | {100 * first['legal_frac']:.0f}% | " + " | ".join(cells) + " |")
    L += ["", "## 2. Critic residual: do scoring-card × regional-margin terms predict the winner beyond the critic?", "",
          "Held-out log-loss (nats); the gain of adding the interaction terms, with a 95% bootstrap interval over games.", "",
          "| fit | log-loss |", "|:---|---:|"]
    for k, v in res["residual"]["no critic"].items():
        if k != "interaction_gain":
            L.append(f"| {k} (no critic) | {v['logloss']:.4f} |")
    g = res["residual"]["no critic"]["interaction_gain"]
    L.append(f"| **interaction gain, no critic** | {g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] |")
    for n, r in res["residual"].items():
        if n == "no critic":
            continue
        for k, v in r.items():
            if k != "interaction_gain":
                L.append(f"| {n}: {k} | {v['logloss']:.4f} |")
        g = r["interaction_gain"]
        L.append(f"| **{n}: interaction gain beyond its critic** | {g['mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] |")
    L += ["", "## 3. Mixed derivatives of the value", "",
          "RMS Hessian entry per block (Hutchinson); the last column is cross / geometric mean of the within-block ones.", "",
          "| net | board × board | cards × cards | board × cards | ratio |", "|:---|---:|---:|---:|---:|"]
    for n, h in res["hessian"].items():
        L.append(f"| {n} | {h['board x board (rms entry)']:.3e} | {h['cards x cards (rms entry)']:.3e} | "
                 f"{h['board x cards (rms entry)']:.3e} | {h['cross / geometric mean of within (rms entry)']:.3f} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nets", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=1500, help="games per policy")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    _regions()

    parts, policy_names = [], []
    for k, path in enumerate(a.policies):
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        policy_names.append(ag.name)
        parts.append(collect(ag, a.games, a.envs, a.seed + 1000 * k, a.sample_frac, a.temperature, dev,
                             game_offset=k * 10 * a.games))
        print(f"collected {parts[-1]['obs'].shape[0]} positions from {ag.name}", flush=True)
    d = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    games = np.unique(d["game"])
    test = np.isin(d["game"], np.random.default_rng(a.seed).choice(games, int(len(games) * a.test_frac), replace=False))
    cf = counterfactuals(d, a.seed)
    print(f"{len(cf['rows'])} counterfactual pairs from {cf['n_positions']} influence placements", flush=True)

    res: Dict[str, Any] = {"policies": policy_names, "temperature": a.temperature, "positions": int(d["obs"].shape[0]),
                           "influence_positions": cf["n_positions"], "sensitivity": {}, "hessian": {},
                           "residual": {"no critic": residual_test(d, None, test, dev, a.seed)}}
    for path in a.nets:
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        model: Any = ag.model.eval()
        res["sensitivity"][ag.name] = sensitivity(model, d, cf, dev)
        _, v = _policy_value(model, d["obs"], d["mask"], dev)
        res["residual"][ag.name] = residual_test(d, v, test, dev, a.seed)
        res["hessian"][ag.name] = hessian_blocks(model, d["obs"], d["mask"], dev, seed=a.seed)
        print(f"measured {ag.name}", flush=True)
    md = render(res)
    print(md)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(res, f, indent=1)
    if a.output_md:
        with open(a.output_md, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
