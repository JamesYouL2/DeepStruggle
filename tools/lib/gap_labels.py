"""Denoised event-or-not labels: a cross-fitted regression of each bank position's playout gap.

One position's 64 paired playouts measure "event minus the model's own non-event move" with a
standard error of about ten points, while the effects that matter are a few points and confined to
narrow conditions (One Small Step one box behind, KAL-007 with South Korea US-controlled). A label
read off one position therefore mostly carries its noise, and any soft function of it is biased
toward 1/2 -- which taught the first contrastive model to event cards whose event is wrong on
average (research: the -2 to -8 point pooled gaps, labels of 0.22-0.35).

So the gap is regressed on the position's features, and each position is labelled from models that
never saw it:

* **features** -- the bank's generic ones (turn, rounds left, hands, DEFCON, VP lead, Military Ops,
  space, the event's immediate gain, region values ...), the card and side as a category, and the
  conditions the expert review proposed (South Korea US-controlled, The Reformer played, NATO in
  effect with the UK US-controlled). The regression decides whether the proposed conditions carry
  signal; it is not told they do.
* **cross-fitting** -- K folds, repeated over R random splits: a position's prediction always comes
  from a model fitted on other positions, so it cannot fit its own noise.
* **the model** -- gradient-boosted trees, depth 3, leaves of 10 or more positions, early stopping.
  Leaf size was chosen by out-of-fold R² on the E7-04-44 banks (40: +0.111, 20: +0.113, 10: +0.113),
  and only 10 separates KAL-007 without South Korea (26 positions, measured -6.1) from the 323 with it.
* **the label** -- P(event) is the share of the position's R out-of-fold predictions above zero:
  the event where the evidence from *similar positions* says it beats the model's own move, Ops
  where it says it loses, and in between where the fits disagree.

`report()` says, per card and side, whether the regression beats predicting the card's mean gap
(out-of-fold), and how often it labels the event in and out of the review's conditions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EVENT = 110

_NUMERIC = ("turn", "ar", "rounds_left", "opp_rounds_left", "hand", "opp_hand", "surplus", "defcon",
            "lead", "mil", "mil_opp", "short", "short_opp", "space", "space_opp", "ops")
EXPERT = ("south_korea_us", "reformer_played", "nato_uk_us", "space_one_behind_1v2", "space_one_behind_3v4")


def expert_flags(save: str) -> Dict[str, float]:
    """The conditions the expert review proposed, read off the position (mover's perspective)."""
    import ts_engine as ts

    from ai.eval.ops_block import controlled, influence

    st = ts.state_from_save_json(save)
    us = st.ctx().decision_player == ts.Player.US
    ctl_us = controlled(influence(st), 0)
    sk, uk = (int(ts.MapData.get_country_by_name(n)) for n in ("South Korea", "United Kingdom"))
    if max(sk, uk) >= len(ctl_us):   # the lookup answers 255 for a name it does not know
        raise KeyError("a country name in expert_flags is not on the map")
    me, opp = (int(st.us_space_track), int(st.ussr_space_track))[:: 1 if us else -1]
    return {
        "south_korea_us": float(bool(ctl_us[sk])),
        "reformer_played": float(st.has_flag(int(ts.EffectBits.THE_REFORMER_PLAYED))),
        "nato_uk_us": float(st.has_flag(int(ts.EffectBits.NATO_ACTIVE)) and bool(ctl_us[uk])),
        "space_one_behind_1v2": float((me, opp) == (1, 2)),
        "space_one_behind_3v4": float((me, opp) == (3, 4)),
    }


def gap(rec: Dict[str, Any]) -> Optional[float]:
    """Event minus the model's own non-event move, mean of the paired differences (0-1 scale).

    The comparison move is the model's choice when that is not the event, else its likeliest other
    mode -- not the best other mode by measured result, which would pick the luckiest and bias the
    gap downward."""
    cands = rec["candidates"]
    ev = [j for j, c in enumerate(cands) if int(c["prefix"][0]) == EVENT]
    if not ev:
        return None
    e = max(ev, key=lambda j: sum(int(ch) for ch in rec["results"][j]))
    others = [j for j, c in enumerate(cands) if int(c["prefix"][0]) != EVENT]
    if not others:
        return None
    o = 0 if int(cands[0]["prefix"][0]) != EVENT else max(others, key=lambda j: float(cands[j].get("prior", 0.0)))
    d = [(int(a) - int(b)) / 2.0 for a, b in zip(rec["results"][e], rec["results"][o])]
    return sum(d) / len(d)


def design(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[str]]:
    """The feature matrix, its column names, and each row's card-side key."""
    keys = sorted({f"{r['features']['side']}:{r['features']['card']}" for r in records})
    cols = list(_NUMERIC) + ["space_diff", "event_gain", "event_gain_known", "phasing", "opp_replies",
                             "china_own", "china_playable", "n_scoring_held", "n_timing_held"]
    cols += [f"region_value_{i}" for i in range(6)] + [f"region_status_{i}" for i in range(6)]
    cols += list(EXPERT) + [f"cardside={k}" for k in keys]
    rows, rkeys = [], []
    for r in records:
        f = r["features"]
        eg = f.get("event_gain")
        row = [float(f.get(c) or 0) for c in _NUMERIC]
        row += [float((f.get("space") or 0) - (f.get("space_opp") or 0)), float(eg or 0.0), float(eg is not None),
                float(bool(f.get("phasing"))), float(bool(f.get("opp_replies"))), float(f.get("china") == "own"),
                float(bool(f.get("china_playable"))), float(len(f.get("scoring_held") or [])),
                float(len(f.get("timing_held") or []))]
        rv = list(f.get("region_value") or [0] * 6)
        rs = str(f.get("region_status") or "000000")
        row += [float(x) for x in rv[:6]] + [float(int(ch)) for ch in rs[:6]]
        fl = expert_flags(r["save"])
        row += [fl[c] for c in EXPERT]
        k = f"{f['side']}:{f['card']}"
        row += [float(k == kk) for kk in keys]
        rows.append(row)
        rkeys.append(k)
    return np.asarray(rows, dtype=np.float64), cols, rkeys


def cross_fit(x: np.ndarray, y: np.ndarray, folds: int = 5, repeats: int = 10, seed: int = 0) -> np.ndarray:
    """[repeats, n] out-of-fold predictions of y."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    out = np.zeros((repeats, len(y)))
    for rep in range(repeats):
        for tr, te in KFold(folds, shuffle=True, random_state=seed + rep).split(x):
            m = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=10, learning_rate=0.05,
                                              max_iter=300, l2_regularization=1.0, early_stopping=True,
                                              validation_fraction=0.15, n_iter_no_change=20,
                                              random_state=seed + rep)
            m.fit(x[tr], y[tr])
            out[rep, te] = m.predict(x[te])
    return out


def label(records: Sequence[Dict[str, Any]], folds: int = 5, repeats: int = 10, seed: int = 0
          ) -> Tuple[List[Optional[float]], Dict[str, Any]]:
    """P(event) per record (None where the record has no gap), and what the fit found."""
    idx = [i for i, r in enumerate(records) if r["kind"] == "mode" and gap(r) is not None]
    recs = [records[i] for i in idx]
    y = np.asarray([gap(r) for r in recs], dtype=np.float64)
    x, cols, keys = design(recs)
    oof = cross_fit(x, y, folds, repeats, seed)
    p = (oof > 0).mean(axis=0)
    labels: List[Optional[float]] = [None] * len(records)
    for j, i in enumerate(idx):
        labels[i] = float(p[j])
    return labels, {"y": y, "oof": oof, "p": p, "x": x, "cols": cols, "keys": keys, "records": recs}


def report(fit: Dict[str, Any], names: Dict[int, str]) -> str:
    """Per card and side: does the regression beat the card's mean, and where does it say event."""
    y, oof, p, x, cols, keys = fit["y"], fit["oof"], fit["p"], fit["x"], fit["cols"], fit["keys"]
    pred = oof.mean(axis=0)
    ex = {c: x[:, cols.index(c)] > 0.5 for c in EXPERT}
    out = ["| card | side | n | mean gap | out-of-fold R² vs card mean | labelled event (P>½) | in the review's condition | event there | outside it | event there |",
           "|:---|:---|---:|---:|---:|---:|:---|---:|---:|---:|"]
    cond = {"US:80": "space_one_behind_3v4", "USSR:80": "space_one_behind_3v4", "US:89": "south_korea_us",
            "USSR:90": "reformer_played", "US:105": "nato_uk_us"}
    karr = np.asarray(keys)
    for k in sorted(set(keys)):
        m = karr == k
        yk, pk, prk = y[m], pred[m], p[m]
        base = float(((yk - yk.mean()) ** 2).mean())
        r2 = 1.0 - float(((yk - pk) ** 2).mean()) / base if base > 0 else 0.0
        side, card = k.split(":")
        c = cond.get(k)
        if c:
            inn = ex[c][m]
            cin = f"{c} ({int(inn.sum())})"
            ein = f"{(prk[inn] > 0.5).mean():.0%}" if inn.any() else "-"
            eout = f"{(prk[~inn] > 0.5).mean():.0%}" if (~inn).any() else "-"
            nout = str(int((~inn).sum()))
        else:
            cin, ein, eout, nout = "-", "-", "-", "-"
        out.append(f"| {names.get(int(card), card)} | {side} | {int(m.sum())} | {100 * yk.mean():+.1f} | "
                   f"{r2:+.3f} | {(prk > 0.5).mean():.0%} | {cin} | {ein} | {nout} | {eout} |")
    tot = 1.0 - float(((y - pred) ** 2).sum()) / sum(float(((y[karr == k] - y[karr == k].mean()) ** 2).sum())
                                                       for k in set(keys))
    out += ["", f"Pooled out-of-fold R² against each card's mean: {tot:+.4f} "
            f"(positive: the features predict the gap within a card beyond its average)."]
    return "\n".join(out)
