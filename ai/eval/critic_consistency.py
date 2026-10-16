"""Is the critic self-consistent? Four probes on greedy self-play (P33, "something is wrong with the critic").

Deeper rollouts should only add information, yet the rollout root loses when its playouts run past 4
boundaries, and wins when it stops at a turn boundary that comes sooner. Each probe here tests one way
a critic can be wrong *as a leaf evaluator* while being calibrated on average (P33b):

1. **Perspective.** The root reads every leaf from the root mover's side (`rollout_batch.cpp`,
   `fill_leaves`), but the trainer only ever reads a decision from its decider's side. At each decision
   the critic is read both ways: v_dec (the decider's) and v_non (the other side's own-frame value on the
   same state). Zero-sum consistency wants v_dec + v_non = 0; each reading is scored against the result.
2. **Drift (the martingale test).** Under the policy that generated the games, the critic's value in a
   fixed frame should not drift in expectation from one decision to the next. Mean and RMS of the step
   change, by the kind of step: same mover, hand-over, action-round boundary, turn boundary.
3. **Position in the turn.** Log loss against the result by phase and action round, per turn group: is
   the critic sharper at turn starts than mid-turn, beyond the trend of the game?
4. **The deal.** At turn boundaries the next turn's cards are dealt. From the last decision of a turn,
   the same move is replayed under many reseeded deals: the mean value after the deal should equal the
   value before it (from the same side, on-distribution where that side decides).
6. **The turn-end park** (pre-deal): every env's position after a turn's last action, before the deal
   (`VectorizedBatchRunner.set_capture_predeal`), read from both sides -- the rows the pre-deal critic
   (`--predeal-value-coef`) trains on and the root's `stop=predeal` reads: log loss of each side's reading
   against the result, and their sum.
5. **The root's leaves** (`--bank`): per stop rule, the share of rollout-root leaves whose decider is the
   root mover (the reading the critic was trained on), and at the others how far the mover's reading is
   from the decider's sign-corrected one.

    PYTHONPATH=.:build/release python -m ai.eval.critic_consistency --checkpoint swa.pt \\
        [--envs 512] [--batches 2] [--deal-positions 1500] [--deals 16] [--bank bank.jsonl.gz]
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.pimcts import acting_player
from ai.search.rollout_root import STOPS, RolloutConfig, RolloutRoot
from bindings.ts_env import model_obs_features
from tools.lib.corpus_driver import state_from_token
from tools.lib.player_agent import NeuralAgent

_PLAYER = {1: ts.Player.US, -1: ts.Player.USSR}


@dataclass
class Boundary:
    """The last decision of a turn: its state, the move played, the mover and the mover's value."""
    state: ts.GameState
    action: int
    side: int
    v_pre: float
    game: int


@dataclass
class Table:
    g: List[int] = field(default_factory=list)
    side: List[int] = field(default_factory=list)
    turn: List[int] = field(default_factory=list)
    ar: List[int] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)
    v_dec: List[float] = field(default_factory=list)
    v_non: List[float] = field(default_factory=list)
    #: The turn-end parks: (game, turn, v_US from the US side, v_USSR from the USSR side).
    parks: List[Tuple[int, int, float, float]] = field(default_factory=list)


def _values(model: Any, device: Any, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        _, v_win, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
    return v_win.float().squeeze(-1).cpu().numpy()


def play(model: Any, device: Any, num_envs: int, seed: int, max_boundaries: int,
         max_iters: int = 4000) -> Tuple[Table, Dict[int, float], List[Boundary]]:
    """Greedy self-play: every decision read from both sides; the turns' last decisions kept."""
    feats = model_obs_features(model)
    runner = ts.VectorizedBatchRunner(num_envs, seed)
    if feats:
        runner.set_obs_features([feats] * num_envs, [feats] * num_envs)
    runner.set_capture_predeal(True)
    tab = Table()
    done = np.zeros(num_envs, dtype=bool)
    bounds: List[Boundary] = []
    for _ in range(max_iters):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        players = np.asarray(runner.get_decision_players())
        live = [int(i) for i in np.flatnonzero(~done) if int(players[i]) != 0]
        states = {i: runner.get_state(i) for i in live}
        other = np.zeros_like(obs)
        for i in live:
            other[i] = np.asarray(ts.extract_observation_features(states[i], _PLAYER[-int(players[i])], feats))
        with torch.no_grad():
            logits, v_win, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
            v_dec = v_win.float().squeeze(-1).cpu().numpy()
            picks = logits.argmax(-1).cpu().numpy()
        v_non = _values(model, device, other, masks)
        for i in live:
            st = states[i]
            tab.g.append(seed * 100_000 + i)
            tab.side.append(int(players[i]))
            tab.turn.append(int(st.turn))
            tab.ar.append(int(st.action_round))
            tab.phase.append(str(st.current_phase).split(".")[-1])
            tab.v_dec.append(float(v_dec[i]))
            tab.v_non.append(float(v_non[i]))
        runner.step_flat_all([int(a) for a in picks], auto_advance=True)
        flags, pus, pussr = (np.array(x) for x in runner.get_predeal())
        pe = np.flatnonzero(flags)
        if pe.size:
            zero = np.ones((pe.size, masks.shape[1]), dtype=masks.dtype)
            vu = _values(model, device, np.ascontiguousarray(pus[pe], dtype=np.float32), zero)
            vr = _values(model, device, np.ascontiguousarray(pussr[pe], dtype=np.float32), zero)
            for j, i in enumerate(pe):
                tab.parks.append((seed * 100_000 + int(i), int(states[int(i)].turn) if int(i) in states else 0,
                                  float(vu[j]), float(vr[j])))
        if len(bounds) < max_boundaries:
            term = np.asarray(runner.get_terminals(), dtype=bool)
            for i in live:
                if not term[i] and int(runner.get_state(i).turn) != int(states[i].turn):
                    bounds.append(Boundary(states[i].clone(), int(picks[i]), int(players[i]), float(v_dec[i]),
                                           seed * 100_000 + i))
    utils = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
    outcome = {seed * 100_000 + i: float(utils[i]) for i in range(num_envs) if done[i]}
    return tab, outcome, bounds


def _clustered_mean(x: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    """Mean and its game-clustered standard error."""
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), float("nan")
    _, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=x)
    counts = np.bincount(inv)
    m = float(x.mean())
    resid = sums - m * counts
    return m, float(np.sqrt((resid ** 2).sum()) / len(x))


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def _bucket(phase: str, ar: int) -> str:
    if phase == "HEADLINE":
        return "headline"
    if phase == "ACTION_ROUND":
        return f"AR{ar}" if ar <= 6 else "AR7+"
    return phase.lower()


def report(tab: Table, outcome: Dict[int, float], deal_rows: Optional[List[Tuple[int, float, float, float]]]) -> str:
    g = np.asarray(tab.g)
    keep = np.asarray([outcome.get(int(x), 0.0) != 0.0 for x in g])
    side = np.asarray(tab.side, dtype=np.float64)
    vd = np.asarray(tab.v_dec, dtype=np.float64)
    vn = np.asarray(tab.v_non, dtype=np.float64)
    turn = np.asarray(tab.turn)
    ar = np.asarray(tab.ar)
    phase = np.asarray(tab.phase)
    us_win = np.asarray([1.0 if outcome.get(int(x), 0.0) > 0 else 0.0 for x in g])
    y = np.where(side > 0, us_win, 1.0 - us_win)              # the decider won
    buckets = np.asarray([_bucket(str(p), int(a)) for p, a in zip(phase, ar)])
    out: List[str] = [f"{len(set(g.tolist()))} games, {len(g)} decisions ({int(keep.sum())} in decided games)"]

    out.append("\n1. Perspective: the decider's reading against the other side's (own frame) on the same state")
    out.append(f"   {'bucket':10s} {'n':>7s} {'mean v_dec+v_non':>17s} {'mean |sum|':>11s} "
               f"{'LL decider':>11s} {'LL other side':>14s} {'LL avg':>8s}")
    for b in ["all"] + sorted(set(buckets.tolist())):
        sel = keep & ((buckets == b) if b != "all" else np.ones(len(g), dtype=bool))
        if sel.sum() < 200:
            continue
        s = vd[sel] + vn[sel]
        m, se = _clustered_mean(s, g[sel])
        ll_d = _ll((vd[sel] + 1) / 2, y[sel]).mean()
        ll_n = _ll((1 - vn[sel]) / 2, y[sel]).mean()
        ll_a = _ll((vd[sel] - vn[sel] + 2) / 4, y[sel]).mean()
        out.append(f"   {b:10s} {int(sel.sum()):7d} {m:+9.3f} ± {se:.3f} {np.abs(s).mean():11.3f} "
                   f"{ll_d:11.4f} {ll_n:14.4f} {ll_a:8.4f}")

    # Which reading carries the offset, and does a constant remove it? Signed error against the result
    # (2y - 1, the decider's frame) for each reading; then the other side's reading shifted by the mean
    # sum fitted per bucket and decider side on even games, scored on odd games (both readings and
    # their average), against the decider's reading on the same odd games.
    out.append("\n1b. The offset: signed error against the result, and a constant shift (fit on even games, "
               "scored on odd)")
    out.append(f"   {'bucket':10s} {'side':>5s} {'err decider':>12s} {'err other':>10s} {'shift':>7s} "
               f"{'LL decider':>11s} {'LL other+shift':>15s} {'LL avg+shift':>13s}")
    target = 2 * y - 1
    even = (g % 2) == 0
    for b in ["all"] + sorted(set(buckets.tolist())):
        for sd, sname in ((1.0, "US"), (-1.0, "USSR")):
            sel = keep & (side == sd) & ((buckets == b) if b != "all" else np.ones(len(g), dtype=bool))
            if sel.sum() < 400:
                continue
            fit, test = sel & even, sel & ~even
            if fit.sum() < 100 or test.sum() < 100:
                continue
            shift = float((vd[fit] + vn[fit]).mean())
            err_d = float((vd[sel] - target[sel]).mean())
            err_n = float((-vn[sel] - target[sel]).mean())
            vn_s = vn[test] - shift
            ll_d = _ll((vd[test] + 1) / 2, y[test]).mean()
            ll_ns = _ll(np.clip((1 - vn_s) / 2, 0, 1), y[test]).mean()
            ll_as = _ll(np.clip((vd[test] - vn_s + 2) / 4, 0, 1), y[test]).mean()
            out.append(f"   {b:10s} {sname:>5s} {err_d:+12.4f} {err_n:+10.4f} {shift:+7.3f} "
                       f"{ll_d:11.4f} {ll_ns:15.4f} {ll_as:13.4f}")

    out.append("\n2. Drift: the US-frame value from one decision to the next, by kind of step")
    u = side * vd
    kinds: Dict[str, List[Tuple[float, int]]] = {}
    seq = np.argsort(g, kind="stable")        # rows are appended env-interleaved; stable keeps each game's order
    for a, b in zip(seq[:-1], seq[1:]):
        if g[b] != g[a]:
            continue
        if turn[b] != turn[a]:
            kind = "turn boundary"
        elif ar[b] != ar[a] or phase[b] != phase[a]:
            kind = "round boundary"
        elif side[b] != side[a]:
            kind = "hand-over"
        else:
            kind = "same mover"
        kinds.setdefault(kind, []).append((u[b] - u[a], int(g[a])))
    out.append(f"   {'step':15s} {'n':>7s} {'mean change (US frame)':>24s} {'RMS':>7s}")
    for kind in ("same mover", "hand-over", "round boundary", "turn boundary"):
        rows = kinds.get(kind, [])
        if not rows:
            continue
        d = np.asarray([r[0] for r in rows])
        m, se = _clustered_mean(d, np.asarray([r[1] for r in rows]))
        out.append(f"   {kind:15s} {len(d):7d} {m:+14.4f} ± {se:.4f} {np.sqrt((d ** 2).mean()):7.3f}")

    out.append("\n3. Log loss against the result (decider's reading) by position in the turn")
    groups = [("turns 1-3", (1, 3)), ("turns 4-7", (4, 7)), ("turns 8-10", (8, 10))]
    order = ["headline"] + [f"AR{i}" for i in range(1, 7)] + ["AR7+"]
    out.append("   " + f"{'':12s}" + "".join(f"{b:>9s}" for b in order))
    for name, (lo, hi) in groups:
        cells = []
        for b in order:
            sel = keep & (buckets == b) & (turn >= lo) & (turn <= hi)
            cells.append(f"{_ll((vd[sel] + 1) / 2, y[sel]).mean():9.4f}" if sel.sum() >= 200 else f"{'—':>9s}")
        out.append(f"   {name:12s}" + "".join(cells))

    if tab.parks:
        pg = np.asarray([p[0] for p in tab.parks])
        pk = np.asarray([outcome.get(int(x), 0.0) != 0.0 for x in pg])
        vu = np.asarray([p[2] for p in tab.parks])
        vr = np.asarray([p[3] for p in tab.parks])
        uw = np.asarray([1.0 if outcome.get(int(x), 0.0) > 0 else 0.0 for x in pg])
        m, se = _clustered_mean((vu + vr)[pk], pg[pk])
        out.append(f"\n6. The turn-end park (after the last action, before the deal): {int(pk.sum())} positions "
                   f"in decided games")
        out.append(f"   mean v_US + v_USSR {m:+.3f} ± {se:.3f}, mean |sum| {np.abs(vu + vr)[pk].mean():.3f}")
        out.append(f"   log loss: the US side's reading {_ll((vu[pk] + 1) / 2, uw[pk]).mean():.4f}, the USSR side's "
                   f"{_ll((vr[pk] + 1) / 2, 1 - uw[pk]).mean():.4f}, their average "
                   f"{_ll((vu[pk] - vr[pk] + 2) / 4, uw[pk]).mean():.4f}")
    if deal_rows:
        out.append("\n4. The deal: mean value after reseeded deals minus the value before (the last mover's frame)")
        gd = np.asarray([r[0] for r in deal_rows])
        for col, title in ((1, "read from the last mover's side"), (2, "read from the decider's side, sign-corrected")):
            d = np.asarray([r[col] for r in deal_rows])
            m, se = _clustered_mean(d, gd)
            out.append(f"   {title:46s} mean {m:+.4f} ± {se:.4f}   mean |.| {np.abs(d).mean():.3f}")
        spread = np.asarray([r[3] for r in deal_rows])
        out.append(f"   the critic's spread across deals of one position (SD): mean {spread.mean():.3f}")
    return "\n".join(out)


def deal_test(model: Any, device: Any, bounds: Sequence[Boundary], deals: int, seed: int
              ) -> List[Tuple[int, float, float, float]]:
    """Per boundary: (game, mean over deals of v_mover(post) - v_pre, the same read from the post
    decider's side with the sign put in the mover's frame, the SD of v_mover(post) across deals)."""
    feats = model_obs_features(model)
    rows: List[Tuple[int, float, float, float]] = []
    rng = np.random.default_rng(seed)
    for b in bounds:
        mover_obs, dec_obs, signs, masks = [], [], [], []
        for _ in range(deals):
            s = b.state.clone()
            s.rng_state = int(rng.integers(1, 2 ** 63))
            ts.Engine.step_flat(s, b.action, True)
            if ts.Engine.is_terminal(s):
                continue
            dp = int(s.ctx().decision_player) if int(s.ctx().decision_player) != 0 else int(s.phasing_player)
            mover_obs.append(np.asarray(ts.extract_observation_features(s, _PLAYER[b.side], feats)))
            dec_obs.append(np.asarray(ts.extract_observation_features(s, _PLAYER[dp], feats)))
            signs.append(1.0 if dp == b.side else -1.0)
            masks.append(np.asarray(ts.get_flat_action_mask(s, False)))
        if len(mover_obs) < 2:
            continue
        m = np.asarray(masks)
        vm = _values(model, device, np.asarray(mover_obs, dtype=np.float32), m)
        vd = _values(model, device, np.asarray(dec_obs, dtype=np.float32), m) * np.asarray(signs)
        rows.append((b.game, float(vm.mean() - b.v_pre), float(vd.mean() - b.v_pre), float(vm.std())))
    return rows


def leaf_perspective(model: Any, device: Any, states: Sequence[ts.GameState], chunk: int = 64) -> str:
    """Probe 5: the rollout root (k = 4, 16 worlds, 4 boundaries, z2) under each stop rule."""
    feats = model_obs_features(model)
    out = [f"\n5. The rollout root's leaves ({len(states)} positions, k = 4, 16 worlds)",
           f"   {'stop':11s} {'leaves':>7s} {'mover decides':>14s}   mover - decider reading where it does not"]
    for stop in STOPS:
        root = RolloutRoot(model, RolloutConfig(k=4, worlds=16, horizon=4, rule="z2", seed=3, stop=stop), device=device)
        on = off = 0
        diffs: List[float] = []
        for lo in range(0, len(states), chunk):
            part = list(states[lo:lo + chunk])
            lg, masks, _ = root._forward(part)
            lg = np.ascontiguousarray(lg, dtype=np.float32)
            kw = {} if stop == "boundaries" else {"stop": stop}
            batch = ts.RolloutBatch(horizon=4, shared_dice=True, guard=True, features=feats, seed=11 + lo, **kw)
            movers: List[int] = []
            for i, st in enumerate(part):
                c = ts.rollout_candidates(lg[i], masks[i], 4)
                if len(c) >= 2 and not ts.Engine.is_terminal(st):
                    before = batch.rows()
                    batch.add(st, c, 16)
                    movers += [int(acting_player(st))] * (batch.rows() - before)
            if not batch.rows():
                continue
            root._playouts(batch)
            term = batch.row_terminal()
            m_obs, d_obs, msk = [], [], []
            for r in range(batch.rows()):
                if term[r]:
                    continue
                s = batch.row_state(r)
                dp = int(acting_player(s))
                if dp == movers[r]:
                    on += 1
                    continue
                off += 1
                m_obs.append(np.asarray(ts.extract_observation_features(s, _PLAYER[movers[r]], feats)))
                d_obs.append(np.asarray(ts.extract_observation_features(s, _PLAYER[dp], feats)))
                msk.append(np.asarray(ts.get_flat_action_mask(s, False)))
            if m_obs:
                m = np.asarray(msk)
                d = (_values(model, device, np.asarray(m_obs, dtype=np.float32), m)
                     + _values(model, device, np.asarray(d_obs, dtype=np.float32), m))    # v_mover - (-v_decider)
                diffs.extend(d.tolist())
        dd = np.asarray(diffs) if diffs else np.zeros(1)
        out.append(f"   {stop:11s} {on + off:7d} {100 * on / max(on + off, 1):13.1f}%   "
                   f"mean {dd.mean():+.3f}, mean |.| {np.abs(dd).mean():.3f}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5151)
    ap.add_argument("--deal-positions", type=int, default=1500)
    ap.add_argument("--deals", type=int, default=16)
    ap.add_argument("--bank", default=None, help="searched-leak bank.jsonl.gz: run probe 5 on its positions")
    ap.add_argument("--bank-positions", type=int, default=256)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    tab, outcome, bounds = Table(), {}, []
    for k in range(a.batches):
        t, o, b = play(model, dev, a.envs, a.seed + k, a.deal_positions - len(bounds))
        for name in vars(tab):
            getattr(tab, name).extend(getattr(t, name))
        outcome.update(o)
        bounds.extend(b)
    rows = deal_test(model, dev, bounds, a.deals, a.seed) if a.deals > 0 else None
    print(f"checkpoint {a.checkpoint}, greedy self-play")
    print(report(tab, outcome, rows))
    if a.bank:
        states: List[ts.GameState] = []
        with gzip.open(a.bank, "rt") as f:
            for line in f:
                states.append(state_from_token(json.loads(line)["pos"]))
                if len(states) >= a.bank_positions:
                    break
        print(leaf_perspective(model, dev, states))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
