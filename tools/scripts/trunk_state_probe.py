#!/usr/bin/env python3
"""P30 stage A: what a trunk's hidden vector holds, and whether the critic is limited by it.

The ownership probes (`research/log/P29_bet2_ownership_probes.md`) found that no trained trunk
encodes more about the future board than its own random initialisation. This asks two follow-ups
on frozen trunks, all on the same self-play positions, split by game:

* **the gate -- value.** Does the game's winner follow from the position better than the net's own
  critic says? Predictors: the critic itself (its `v_win`, rescaled by a one-parameter logistic
  fitted on the fitting set, so calibration is not held against it), a fresh value head on the
  frozen hidden vector (linear and MLP), and an MLP on the raw observation. AUC ranks positions
  and needs no calibration; log-loss is reported beside it. Drawn games are left out.
* **the present.** Can the trunk be read back for the current state -- each country's controller
  (3 classes) and both sides' influence (regression), and every card's location (8 classes, the
  observation's own location one-hot)? These are functions of the observation, so the
  raw-observation probe is a sanity check near 100%; the question is what the trunk keeps.

Every probe standardises its inputs per feature and stops early on held-out games.

    PYTHONPATH=.:build/release python tools/scripts/trunk_state_probe.py \\
        --trunks a.pt b.pt --policies a.pt --games 1500 --output-json out.json --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import ts_engine as ts  # noqa: E402

from tools.lib.player_agent import NeuralAgent  # noqa: E402
from tools.scripts.trunk_ownership_probe import collect  # noqa: E402

BOARD_W = 26
N_COUNTRIES = 84
N_CARDS = 110
CARD_W = 14
CARD_OFFSET = N_COUNTRIES * BOARD_W
MY_INF, OPP_INF, MY_CTRL, OPP_CTRL = 0, 1, 5, 6       # board slots (engine/src/observation.cpp)
N_LOC = 8                                              # card slots 0..7: location one-hot
MY_HAND = 1


def targets_from_obs(obs: np.ndarray) -> Dict[str, np.ndarray]:
    """Present-state targets read out of the observation itself (mover's frame)."""
    b = obs[:, :CARD_OFFSET].astype(np.float32).reshape(-1, N_COUNTRIES, BOARD_W)
    c = obs[:, CARD_OFFSET:CARD_OFFSET + N_CARDS * CARD_W].astype(np.float32).reshape(-1, N_CARDS, CARD_W)
    ctrl = np.where(b[..., MY_CTRL] > 0.5, 0, np.where(b[..., OPP_CTRL] > 0.5, 1, 2)).astype(np.int64)
    infl = np.stack([b[..., MY_INF], b[..., OPP_INF]], -1) * 10.0                 # (n, 84, 2)
    loc = c[..., :N_LOC].argmax(-1).astype(np.int64)                              # (n, 110)
    return {"control": ctrl, "influence": infl.astype(np.float32), "card_loc": loc}


def _standardiser(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    s1 = torch.zeros(x.shape[1], device=x.device, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    for k in range(0, x.shape[0], 8192):
        xb = x[k:k + 8192].double()
        s1 += xb.sum(0)
        s2 += (xb * xb).sum(0)
    mu = s1 / x.shape[0]
    sd = (s2 / x.shape[0] - mu * mu).clamp_min(0).sqrt()
    return mu.float(), torch.where(sd > 1e-6, sd, torch.ones_like(sd)).float()


def fit(make: Callable[[], nn.Module], loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_fit: torch.Tensor, y_fit: torch.Tensor, game_fit: np.ndarray, x_test: torch.Tensor,
        dev: torch.device, seed: int, epochs: int, lr: float = 1e-3) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """A probe fitted with AdamW, stopped on a tenth of the fitting games; returns its test output."""
    torch.manual_seed(seed)
    games = np.unique(game_fit)
    held = np.isin(game_fit, np.random.default_rng(seed + 1).choice(games, max(1, len(games) // 10), replace=False))
    val = torch.from_numpy(np.flatnonzero(held)).to(dev)
    tr = torch.from_numpy(np.flatnonzero(~held)).to(dev)
    mu, sd = _standardiser(x_fit)
    inner = make().to(dev)

    def net(x: torch.Tensor) -> torch.Tensor:
        return inner((x.float() - mu) / sd)

    opt = torch.optim.AdamW(inner.parameters(), lr=lr, weight_decay=1e-4)

    def val_loss() -> float:
        with torch.no_grad():
            tot = 0.0
            for k in range(0, val.numel(), 8192):
                j = val[k:k + 8192]
                tot += float(loss(net(x_fit[j]), y_fit[j])) * j.numel()
            return tot / max(1, val.numel())

    best, best_state, stale, used = float("inf"), None, 0, 0
    for ep in range(epochs):
        inner.train()
        order = tr[torch.randperm(tr.numel(), device=dev)]
        for k in range(0, order.numel(), 1024):
            j = order[k:k + 1024]
            l = loss(net(x_fit[j]), y_fit[j])
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        inner.eval()
        v = val_loss()
        used = ep + 1
        if v < best - 1e-5:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in inner.state_dict().items()}
        else:
            stale += 1
            if stale >= 3:
                break
    assert best_state is not None
    inner.load_state_dict(best_state)
    inner.eval()
    with torch.no_grad():
        out = torch.cat([net(x_test[k:k + 8192]) for k in range(0, x_test.shape[0], 8192)])
    return out, {"epochs": used, "val_loss": best}


def _mlp(d_in: int, d_out: int, width: int = 256) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, d_out))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC of `score` for binary `label`, ties counted half."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / max(1, n_pos * n_neg))


def value_scores(p_win: np.ndarray, won: np.ndarray, turn: np.ndarray) -> Dict[str, Any]:
    p = np.clip(p_win, 1e-6, 1 - 1e-6)
    ll = -(won * np.log(p) + (1 - won) * np.log(1 - p))
    out: Dict[str, Any] = {"auc": auc(p_win, won), "logloss": float(ll.mean()),
                           "accuracy": float(((p_win > 0.5) == (won == 1)).mean())}
    for name, (a, b) in {"turns 1-3": (1, 3), "turns 4-7": (4, 7), "turns 8-10": (8, 10)}.items():
        m = (turn >= a) & (turn <= b)
        out[name] = {"auc": auc(p_win[m], won[m]), "logloss": float(ll[m].mean())} if m.any() else {}
    return out


@torch.no_grad()
def trunk_outputs(model: Any, obs: np.ndarray, dev: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """The hidden vector and the critic's v_win for each position."""
    hs: List[torch.Tensor] = []
    vs: List[np.ndarray] = []
    for k in range(0, obs.shape[0], 4096):
        o = torch.from_numpy(obs[k:k + 4096]).float().to(dev)
        hs.append(model.extract_features(o).float())
        m = torch.ones(o.shape[0], ts.FLAT_ACTION_SIZE if hasattr(ts, "FLAT_ACTION_SIZE") else 220,
                       dtype=torch.bool, device=dev)
        vs.append(model(o, m)[1].float().reshape(-1).cpu().numpy())
    return torch.cat(hs), np.concatenate(vs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trunks", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=1500, help="games per policy")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--concat-raw", action="store_true",
                    help="also fit value heads on the trunk's vector concatenated with the raw observation: "
                         "if the trunk drops state the outcome depends on, this beats the trunk alone")
    ap.add_argument("--skip-present", action="store_true", help="value probes only")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)

    parts: List[Dict[str, np.ndarray]] = []
    policy_names: List[str] = []
    for k, path in enumerate(a.policies):
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        policy_names.append(ag.name)
        parts.append(collect(ag, a.games, a.envs, a.seed + 1000 * k, a.sample_frac, a.temperature, dev,
                             game_offset=k * 10 * a.games))
        print(f"collected {parts[-1]['obs'].shape[0]} positions from {ag.name}", flush=True)
    d = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    games = np.unique(d["game"])
    test_games = np.random.default_rng(a.seed).choice(games, int(len(games) * a.test_frac), replace=False)
    test = np.isin(d["game"], test_games)
    fit_m = ~test
    tg = targets_from_obs(d["obs"])
    won_all = (d["winner"] * d["mover"] == 1).astype(np.float32)
    decided = d["winner"] != 0                                    # draws left out of the value probe
    fv, tv = fit_m & decided, test & decided
    won_fit = torch.from_numpy(won_all[fv]).to(dev)
    won_test, turn_test = won_all[tv], d["turn"][tv]

    def bce(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(out.reshape(-1), y)

    def ce_ctrl(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out.view(-1, 3), y.reshape(-1))

    def mse_inf(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(out.view(-1), y.reshape(-1))

    def ce_loc(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out.view(-1, N_LOC), y.reshape(-1))

    y_ctrl = torch.from_numpy(tg["control"][fit_m]).to(dev)
    y_inf = torch.from_numpy(tg["influence"][fit_m]).to(dev)
    y_loc = torch.from_numpy(tg["card_loc"][fit_m]).to(dev)
    t_ctrl, t_inf, t_loc = tg["control"][test], tg["influence"][test], tg["card_loc"][test]
    hand = t_loc == MY_HAND

    def present(x_fit: torch.Tensor, x_test: torch.Tensor, key: str) -> Dict[str, Any]:
        dim = x_fit.shape[1]
        gf = d["game"][fit_m]
        o, f1 = fit(lambda: _mlp(dim, N_COUNTRIES * 3), ce_ctrl, x_fit, y_ctrl, gf, x_test, dev, a.seed, a.epochs)
        ctrl_acc = float((o.view(-1, N_COUNTRIES, 3).argmax(-1).cpu().numpy() == t_ctrl).mean())
        o, f2 = fit(lambda: _mlp(dim, N_COUNTRIES * 2), mse_inf, x_fit, y_inf, gf, x_test, dev, a.seed, a.epochs)
        pi = o.view(-1, N_COUNTRIES, 2).cpu().numpy()
        o, f3 = fit(lambda: _mlp(dim, N_CARDS * N_LOC), ce_loc, x_fit, y_loc, gf, x_test, dev, a.seed, a.epochs)
        loc_pred = o.view(-1, N_CARDS, N_LOC).argmax(-1).cpu().numpy()
        fits[f"{key}: control"], fits[f"{key}: influence"], fits[f"{key}: cards"] = f1, f2, f3
        return {"control_acc": ctrl_acc,
                "influence_rmse": float(np.sqrt(((pi - t_inf) ** 2).mean())),
                "influence_exact": float((np.rint(pi) == np.rint(t_inf)).mean()),
                "card_loc_acc": float((loc_pred == t_loc).mean()),
                "hand_recall": float((loc_pred[hand] == MY_HAND).mean()) if hand.any() else None,
                "hand_precision": float((t_loc[loc_pred == MY_HAND] == MY_HAND).mean())
                if (loc_pred == MY_HAND).any() else None}

    def value(x_fit: torch.Tensor, x_test: torch.Tensor, key: str, kinds: Sequence[str]) -> Dict[str, Any]:
        dim = x_fit.shape[1]
        out: Dict[str, Any] = {}
        for kind in kinds:
            make: Callable[[], nn.Module] = ((lambda: nn.Linear(dim, 1)) if kind == "linear"
                                             else (lambda: _mlp(dim, 1)))
            o, fits[f"{key}: value {kind}"] = fit(make, bce, x_fit, won_fit, d["game"][fv], x_test, dev,
                                                  a.seed, a.epochs)
            out[kind] = value_scores(torch.sigmoid(o.reshape(-1)).cpu().numpy(), won_test, turn_test)
        return out

    rows: Dict[str, Dict[str, Any]] = {}
    fits: Dict[str, Dict[str, Any]] = {}
    x_all = torch.from_numpy(d["obs"]).to(dev)
    rows["raw observation"] = {"value": value(x_all[torch.from_numpy(np.flatnonzero(fv)).to(dev)],
                                              x_all[torch.from_numpy(np.flatnonzero(tv)).to(dev)],
                                              "raw observation", ["mlp"]),
                               "present": None if a.skip_present else
                               present(x_all[torch.from_numpy(np.flatnonzero(fit_m)).to(dev)],
                                       x_all[torch.from_numpy(np.flatnonzero(test)).to(dev)],
                                       "raw observation")}
    print("fitted raw observation", flush=True)
    del x_all

    for path in a.trunks:
        ag = NeuralAgent.from_checkpoint(path, device=str(dev))
        model: Any = ag.model.eval()
        h, v = trunk_outputs(model, d["obs"], dev)
        hf, ht = h[torch.from_numpy(np.flatnonzero(fv)).to(dev)], h[torch.from_numpy(np.flatnonzero(tv)).to(dev)]
        # the critic, rescaled by a one-parameter-pair logistic fitted on the fitting positions
        vf = torch.from_numpy(v[fv]).to(dev).unsqueeze(1)
        vt = torch.from_numpy(v[tv]).to(dev).unsqueeze(1)
        o, fits[f"{ag.name}: critic calibration"] = fit(lambda: nn.Linear(1, 1), bce, vf, won_fit, d["game"][fv],
                                                        vt, dev, a.seed, a.epochs, lr=1e-2)
        crit = value_scores(torch.sigmoid(o.reshape(-1)).cpu().numpy(), won_test, turn_test)
        crit["auc_raw"] = auc(v[tv], won_test)
        vals = value(hf, ht, ag.name, ["linear", "mlp"])
        if a.concat_raw:
            # half precision: at ~10^6 positions the concatenation would not fit on the GPU in float32
            of = torch.from_numpy(d["obs"][fv]).to(dev)
            ot = torch.from_numpy(d["obs"][tv]).to(dev)
            both = value(torch.cat([hf.half(), of], 1), torch.cat([ht.half(), ot], 1), f"{ag.name} + raw",
                         ["linear", "mlp"])
            vals.update({f"{k} + raw observation": v for k, v in both.items()})
            del of, ot
        rows[ag.name] = {"critic": crit, "value": vals, "hidden_norm": float(h.norm(dim=1).mean()),
                         "present": None if a.skip_present else
                         present(h[torch.from_numpy(np.flatnonzero(fit_m)).to(dev)],
                                 h[torch.from_numpy(np.flatnonzero(test)).to(dev)], ag.name)}
        print(f"fitted {ag.name}", flush=True)
        del h, hf, ht

    res = {"policies": policy_names, "temperature": a.temperature, "fit_positions": int(fit_m.sum()),
           "test_positions": int(test.sum()), "test_games": int(len(test_games)),
           "test_decided_positions": int(tv.sum()), "win_rate_mover_test": float(won_test.mean()),
           "rows": rows, "fits": fits}
    md = render(res)
    print(md)
    if a.output_json:
        with open(a.output_json, "w") as f:
            json.dump(res, f, indent=1)
    if a.output_md:
        with open(a.output_md, "w") as f:
            f.write(md)
    return 0


def _f(x: Any, pct: bool = False, nd: int = 3) -> str:
    if x is None or (isinstance(x, dict) and not x):
        return "—"
    return f"{100 * x:.1f}" if pct else f"{x:.{nd}f}"


def render(res: Dict[str, Any]) -> str:
    L = ["# What the trunk holds, and whether the critic is limited by it", "",
         f"Positions: self-play of {', '.join(res['policies'])} at temperature {res['temperature']}; "
         f"{res['fit_positions']} fitting and {res['test_positions']} test positions from "
         f"{res['test_games']} held-out games. The mover won {100 * res['win_rate_mover_test']:.1f}% of "
         f"the {res['test_decided_positions']} decided test positions.", "",
         "## The gate: predicting the winner", "",
         "AUC ranks positions (0.5 = chance) and needs no calibration; log-loss in nats (0.693 = a coin).", "",
         "| predictor | AUC | log-loss | turns 1-3 AUC | turns 4-7 AUC | turns 8-10 AUC |",
         "|:---|---:|---:|---:|---:|---:|"]

    def vrow(name: str, s: Dict[str, Any]) -> str:
        return (f"| {name} | {_f(s['auc'])} | {_f(s['logloss'])} | {_f(s['turns 1-3'].get('auc'))} | "
                f"{_f(s['turns 4-7'].get('auc'))} | {_f(s['turns 8-10'].get('auc'))} |")

    for name, r in res["rows"].items():
        if "critic" in r:
            L.append(vrow(f"{name}: **its own critic** (rescaled)", r["critic"]))
        for kind, s in r["value"].items():
            label = ("MLP from scratch" if name == "raw observation"
                     else f"fresh {kind} head on the frozen trunk" if "+" not in kind
                     else f"fresh {kind.split(' + ')[0]} head on the frozen trunk **+ raw observation**")
            L.append(vrow(f"{name}: {label}", s))
    L += ["", "## The present: reading the current state back", "",
          "Control accuracy over 84 countries; influence RMSE and exact-after-rounding over both sides; "
          "card location accuracy over 110 cards, and recall / precision of *in my hand*.", "",
          "| source | hidden length | control | influence RMSE | influence exact | card location | hand recall | hand precision |",
          "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in res["rows"].items():
        p = r["present"]
        if p is None:
            continue
        L.append(f"| {name} | {_f(r.get('hidden_norm'), nd=1)} | {_f(p['control_acc'], True)} | "
                 f"{_f(p['influence_rmse'])} | {_f(p['influence_exact'], True)} | {_f(p['card_loc_acc'], True)} | "
                 f"{_f(p['hand_recall'], True)} | {_f(p['hand_precision'], True)} |")
    L += ["", "Probe fits (epochs, held-out loss): " + "; ".join(
        f"{k}: {v['epochs']}, {v['val_loss']:.4f}" for k, v in res["fits"].items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
