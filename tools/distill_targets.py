#!/usr/bin/env python3
"""P36c's targets: a teacher's improved policy on the network's own entry decisions -- and the ladder of
teachers that reads why the rollout root's corrections did not transfer.

Greedy self-play of the network; every card choice (`SELECT_CARD`, the entry decisions: action-round and
headline cards) with two or more legal moves is a position, and a uniform subset of `--positions` of them
is kept, each tagged with its game so the held-out set is whole games (`--holdout`: one game in ten by
default). At each world count of `--worlds` the root (k 4, stop=earlier, leaf=own, keyed chance -- the
P36b teacher) searches every position once, and the target is R32's improved policy over the legal moves,
softmax(logit + sigma(completed Q)) (`ai.search.rollout_root.RolloutTeacher`), with the root's own z2 pick.
The held-out positions are searched a second time with another seed: the teacher's self-agreement on
exactly those positions, the ceiling any student's held-out agreement is read against (P36c's reading 3).

Output: one `.npz` (`--output`): `states` (N x state_size raw GameStates), `game`, `heldout`, `net_argmax`,
and per teacher T: `acts_T`, `probs_T` (N x 16, padded with -1 / 0, sorted by probability), `pick_T`, and for
the held-out rows `acts2_T`, `probs2_T`, `pick2_T` (the second seed; -1 elsewhere). A root teacher also writes
its statistics, the regression targets of an action-value head: `cand_T` (N x 16, the root's candidates, the
network's own move first, padded with -1), `q_T` (each candidate's mean value), `lead_T` / `lead_se_T` (its
paired lead over the network's move and that lead's standard error; 0 / 0 for the move itself), `worlds_T`
(0 where the root did not search), and the second seed's `cand2_T`, `q2_T`, `lead2_T`, `lead_se2_T`,
`worlds2_T` on the held-out rows; NaN, -1 or 0 elsewhere. A sidecar `.meta.json` records the checkpoint, the
teachers, the configuration and the commit.

**Teachers** (`--teacher NAME=SPEC`, repeatable; `--worlds W` is P36c's `W=root:4:W:4:z2:earlier:own:keyed`):

* `root:k:worlds:horizon:rule[:stop][:own|decider][:keyed][:aligned][:box4]` -- the rollout root on the
  checkpoint (the fields of a `rollout:` player spec, `ai.search.rollout_root.rollout_spec_config`), its
  target R32's improved policy and its pick the root's own decision. The corrected deployed root is
  `root:4:32:4:z2:earlier:own:keyed:aligned:box4`. Horizon 0 with k >= the hand (`root:16:4:0:argmax:...`) is
  the network's own critic one step on: every legal card read at the mover's next decision (the card's play
  mode), so the target is what the critic alone would play -- `aligned` changes nothing at horizon 0.
* `net:<checkpoint>` -- another network's raw policy (softmax of its logits), pick its argmax: a teacher whose
  departures are a function of the observation by construction, so the positive control a student must be
  able to learn. Deterministic: its second copy is itself.

**Splitting the search** (CPU runners): `--collect-only` writes the positions alone; `--states <npz>` searches
an existing position set (teachers added beside those it has); `--shard i/n` searches the i-th of n contiguous
slices of it and writes only those rows (`rows` plus the teacher keys); `--merge <shards...>` puts the shards'
rows back into `--states` and writes `--output`.

    tools/distill_targets.py --checkpoint swa.pt --positions 200000 --worlds 16 64 128 \\
        --output /workspace/data/datasets/p36c/targets.npz
    tools/distill_targets.py --checkpoint swa.pt --states pos.npz --shard 3/40 --output shard3.npz \\
        --teacher h4=root:4:16:4:z2:earlier:own:keyed:aligned:box4 --teacher soup=net:soup.pt
    tools/distill_targets.py --checkpoint swa.pt --states pos.npz --merge shard*.npz --output targets.npz
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ts_engine as ts  # noqa: E402
from ai.search.pimcts import acting_player  # noqa: E402
from ai.search.rollout_root import RolloutConfig, RolloutTeacher, rollout_spec_config, rollout_spec_leaf  # noqa: E402
from bindings.ts_env import model_obs_features  # noqa: E402
from tools.lib.engine_fingerprint import check_raw_states, fingerprint  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402

WIDTH = 16


def collect(model: Any, dev: torch.device, games: int, seed: int) -> Tuple[List[ts.GameState], List[int], List[int]]:
    """Every card choice with two or more legal moves in `games` greedy games: states, games, argmaxes."""
    feats = model_obs_features(model)
    runner = ts.VectorizedBatchRunner(games, seed)
    if feats:
        runner.set_obs_features([feats] * games, [feats] * games)
    done = np.zeros(games, dtype=bool)
    states: List[ts.GameState] = []
    game: List[int] = []
    argmax: List[int] = []
    for _ in range(6000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32)).to(dev)
        masks = np.asarray(runner.get_action_masks())
        with torch.no_grad():
            lg = model(obs, torch.from_numpy(masks).to(dev))[0].float().cpu().numpy()
        acts = np.where(masks > 0, lg, -np.inf).argmax(axis=1)
        acts[done] = 0
        for i in np.flatnonzero(~done & (masks.sum(axis=1) >= 2)):
            st = runner.get_state(int(i))
            if int(st.ctx().decision_type) == int(ts.DecisionType.SELECT_CARD):
                states.append(st)
                game.append(int(i))
                argmax.append(int(acts[i]))
        runner.step_flat_all(acts.tolist(), True)
    return states, game, argmax


def z2_pick(stats: Dict[str, Any]) -> int:
    """The root's own decision from its statistics: the best mean on a lead of 2 SE over the network's move."""
    cands = stats["candidates"]
    if not cands:
        return -1
    base = int(cands[0])
    if stats["worlds"] <= 0:
        return base
    best = max(cands, key=lambda a: stats["q"][a])
    m, s = stats["lead"][best]
    return int(best) if s > 0 and m / s >= 2.0 and best != base else base


Targets = Tuple[np.ndarray, np.ndarray, np.ndarray]
#: A root's per-candidate statistics: candidates, mean, paired lead, its standard error, worlds.
Stats = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
STAT_KEYS = ("cand", "q", "lead", "lead_se", "worlds")


def empty_stats(n: int) -> Stats:
    nan = np.full((n, WIDTH), np.nan, dtype=np.float32)
    return (np.full((n, WIDTH), -1, dtype=np.int16), nan.copy(), nan.copy(), nan.copy(), np.zeros(n, dtype=np.int16))


def _put_stats(out: Stats, row: int, stats: Dict[str, Any]) -> None:
    cand, q, lead, se, worlds = out
    cands = [int(c) for c in stats["candidates"]][:WIDTH]
    cand[row, :len(cands)] = cands
    worlds[row] = int(stats["worlds"])
    if stats["worlds"] <= 0:
        return
    for j, c in enumerate(cands):
        q[row, j] = stats["q"][c]
        lead[row, j], se[row, j] = stats["lead"][c]


def _put(acts: np.ndarray, probs: np.ndarray, row: int, a: Sequence[int], p: np.ndarray) -> None:
    order = np.argsort(-np.asarray(p), kind="stable")[:WIDTH]
    acts[row, :len(order)] = np.asarray(a)[order]
    probs[row, :len(order)] = np.asarray(p)[order]


def search(teacher: RolloutTeacher, states: List[ts.GameState], chunk: int) -> Tuple[Targets, Stats]:
    n = len(states)
    acts = np.full((n, WIDTH), -1, dtype=np.int16)
    probs = np.zeros((n, WIDTH), dtype=np.float32)
    pick = np.full(n, -1, dtype=np.int16)
    st = empty_stats(n)
    for lo in range(0, n, chunk):
        part = states[lo:lo + chunk]
        for k, ((a, p), stats) in enumerate(zip(teacher.improved_policies(part), teacher.root.last_stats)):
            _put(acts, probs, lo + k, a, p)
            pick[lo + k] = z2_pick(stats)
            _put_stats(st, lo + k, stats)
    return (acts, probs, pick), st


def net_policy(model: Any, dev: torch.device, states: List[ts.GameState], chunk: int = 4096) -> Targets:
    """A network's raw policy at each position: softmax of its logits over the legal moves, pick its argmax."""
    feats = model_obs_features(model)
    n = len(states)
    acts = np.full((n, WIDTH), -1, dtype=np.int16)
    probs = np.zeros((n, WIDTH), dtype=np.float32)
    pick = np.full(n, -1, dtype=np.int16)
    for lo in range(0, n, chunk):
        part = states[lo:lo + chunk]
        obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), feats), dtype=np.float32)
                        for s in part])
        masks = np.stack([np.asarray(ts.ActionMask.generate_flat_mask(s), dtype=np.uint8) for s in part])
        with torch.no_grad():
            lg = model(torch.from_numpy(obs).to(dev), torch.from_numpy(masks).to(dev))[0].float().cpu().numpy()
        for k in range(len(part)):
            legal = np.flatnonzero(masks[k])
            z = lg[k, legal].astype(np.float64)
            p = np.exp(z - z.max())
            _put(acts, probs, lo + k, legal.tolist(), p / p.sum())
            pick[lo + k] = int(legal[int(np.argmax(z))])
    return acts, probs, pick


def parse_teacher(arg: str) -> Tuple[str, str]:
    """NAME=SPEC -> (name, spec); the spec checked (a root spec's fields, a net's checkpoint)."""
    name, sep, spec = arg.partition("=")
    if not sep or not name or not spec or not name.replace("-", "").replace(".", "").isalnum():
        raise ValueError(f"--teacher takes NAME=SPEC with an alphanumeric name, got {arg!r}")
    kind, _, rest = spec.partition(":")
    if kind == "root":
        root_config(spec)
    elif kind == "net":
        if not rest:
            raise ValueError(f"{arg!r}: net:<checkpoint>")
    else:
        raise ValueError(f"{arg!r}: a teacher is root:k:worlds:horizon:rule[:...] or net:<checkpoint>")
    return name, spec


def root_config(spec: str) -> RolloutConfig:
    """root:k:worlds:horizon:rule[:tail] -> the rollout root's config (a `rollout:` spec's fields)."""
    parts = spec.split(":")
    if len(parts) < 5:
        raise ValueError(f"{spec!r}: root:k:worlds:horizon:rule[:stop][:own][:keyed][:aligned][:box4]")
    rspec = ":".join(["rollout", "-"] + parts[1:])
    _, cfg, _ = rollout_spec_config(rspec)
    leaf = rollout_spec_leaf(rspec)
    if leaf:
        # A rollout spec reads any trailing field it does not know as the leaf checkpoint; a teacher's leaves are
        # read by --checkpoint, so that field is a typo (or a leaf this tool would silently not use).
        raise ValueError(f"{spec!r}: {leaf!r} is not a field of a root teacher (the leaves are read by --checkpoint)")
    if cfg.belief or cfg.oracle:
        raise ValueError(f"{spec!r}: belief and oracle roots need the game's history or the opponent's hand")
    if cfg.race:
        raise ValueError(f"{spec!r}: a raced root's statistics are not every candidate's over every world "
                         "(RolloutTeacher)")
    return cfg


def teacher_seeds(name: str, spec: str, seed: int, shard: int) -> Tuple[int, int]:
    """The first and second search seeds of a teacher: P36c's for its world-count names (so the record's
    targets rebuild), otherwise from the spec; a shard offsets both."""
    legacy = spec.startswith("root:4:") and spec == f"root:4:{name}:4:z2:earlier:own:keyed"
    base = (seed + int(name)) if legacy else seed + zlib.crc32(spec.encode()) % 1_000_003
    second = (seed + 7919 * int(name)) if legacy else base + 7_919_000
    return base + 104_729 * shard, second + 104_729 * shard


def run_teacher(name: str, spec: str, model: Any, dev: torch.device, states: List[ts.GameState],
                held: np.ndarray, seed: int, shard: int) -> Dict[str, np.ndarray]:
    """One teacher's keys on `states` (held-out rows `held` searched twice)."""
    n = len(states)
    hidx = np.flatnonzero(held)
    hstates = [states[i] for i in hidx]
    if spec.startswith("net:"):
        other = NeuralAgent.from_checkpoint(spec[4:], device=str(dev)).model.eval()
        first = net_policy(other, dev, states)
        second: Targets = (first[0][hidx], first[1][hidx], first[2][hidx])
        stats1: Optional[Stats] = None
        hstats = empty_stats(len(hidx))
    else:
        cfg = root_config(spec)
        chunk = max(32, 16384 // max(1, cfg.worlds * max(1, cfg.k // 4)))
        s1, s2 = teacher_seeds(name, spec, seed, shard)
        first, stats1 = search(RolloutTeacher(model, cfg, device=dev, seed=s1), states, chunk)
        second, hstats = search(RolloutTeacher(model, cfg, device=dev, seed=s2), hstates, chunk)
    out = {f"acts_{name}": first[0], f"probs_{name}": first[1], f"pick_{name}": first[2],
           f"acts2_{name}": np.full((n, WIDTH), -1, dtype=np.int16),
           f"probs2_{name}": np.zeros((n, WIDTH), dtype=np.float32),
           f"pick2_{name}": np.full(n, -1, dtype=np.int16)}
    out[f"acts2_{name}"][hidx], out[f"probs2_{name}"][hidx], out[f"pick2_{name}"][hidx] = second
    if stats1 is not None:
        stats2 = empty_stats(n)
        for key, v1, v2, h in zip(STAT_KEYS, stats1, stats2, hstats):
            v2[hidx] = h
            out[f"{key}_{name}"], out[f"{key}2_{name}"] = v1, v2
    return out


def shard_rows(n: int, shard: str) -> np.ndarray:
    """`i/n` -> the i-th (1-based) of n contiguous slices of range(n)."""
    i, _, k = shard.partition("/")
    if not (i.isdigit() and k.isdigit() and 1 <= int(i) <= int(k)):
        raise ValueError(f"--shard takes i/n with 1 <= i <= n, got {shard!r}")
    return np.array_split(np.arange(n), int(k))[int(i) - 1]


def merge(base: Dict[str, np.ndarray], shards: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """The shards' teacher rows put back into the position set; every row of a teacher must be covered once."""
    out = dict(base)
    n = base["states"].shape[0]
    seen: Dict[str, np.ndarray] = {}
    for sh in shards:
        rows = sh["rows"].astype(np.int64)
        for key, v in sh.items():
            if key == "rows":
                continue
            if key not in out:
                out[key] = (np.full((n,) + v.shape[1:], -1, dtype=v.dtype) if key.startswith(("acts", "pick", "cand"))
                            else np.zeros((n,) + v.shape[1:], dtype=v.dtype))
            out[key][rows] = v
            seen.setdefault(key, np.zeros(n, dtype=np.int32))[rows] += 1
    bad = {k: int((c != 1).sum()) for k, c in seen.items() if (c != 1).any()}
    if bad:
        raise ValueError(f"the shards do not cover every row exactly once: {bad}")
    return out


def merged_specs(prev: Dict[str, Any], shard_metas: Sequence[Tuple[str, Dict[str, Any]]], checkpoint: str,
                 seed: int) -> Dict[str, str]:
    """The teachers of a merge; refuses shards searched with another network or seed, or one teacher's name
    given two specs (by two shards, or by a shard and the position set) -- their rows would pool silently."""
    specs: Dict[str, str] = dict(prev.get("teachers", {}))
    origin = {n: "the position set" for n in specs}
    for path, meta in shard_metas:
        for key, want in (("checkpoint", checkpoint), ("seed", seed)):
            if meta.get(key) != want:
                raise SystemExit(f"{path}: searched with {key} {meta.get(key)!r}, this merge has {want!r}")
        for n, sp in meta.get("teachers", {}).items():
            if specs.setdefault(n, sp) != sp:
                raise SystemExit(f"teacher {n!r} is {specs[n]!r} in {origin[n]} but {sp!r} in {path}")
            origin.setdefault(n, path)
    return specs


def _write(path: str, out: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cast(Any, np.savez)(path, **out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta | {"commit": commit, "compiler": ts.BUILD_COMPILER,
                          # raw states are read only by the build that wrote them (ai/training/policy_fit checks)
                          "engine_fingerprint": fingerprint()}, f, indent=2)


def _load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _meta(path: str) -> Dict[str, Any]:
    meta = path + ".meta.json"
    return json.load(open(meta, encoding="utf-8")) if os.path.exists(meta) else {}


def _report(name: str, out: Dict[str, np.ndarray], rows: np.ndarray, net: np.ndarray, held: np.ndarray, secs: float) -> None:
    pick, top = out[f"pick_{name}"][rows], out[f"acts_{name}"][rows, 0]
    h = held[rows]
    agree = (out[f"pick2_{name}"][rows][h] == pick[h]).mean() if h.any() else float("nan")
    print(f"{name}: {secs:.0f}s; departs (pick) {100 * (pick != net[rows]).mean():.1f}%, (target's top) "
          f"{100 * (top != net[rows]).mean():.1f}%; held-out pick agreement between seeds {100 * agree:.1f}%", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="the network: its games, its argmax, the root's network")
    ap.add_argument("--positions", type=int, default=200000)
    ap.add_argument("--games", type=int, default=2000, help="greedy games to draw the positions from")
    ap.add_argument("--worlds", type=int, nargs="*", default=None,
                    help="P36c's teachers by world count (root:4:W:4:z2:earlier:own:keyed, named W); "
                         "16 64 128 when no --teacher is given")
    ap.add_argument("--teacher", action="append", default=[], metavar="NAME=SPEC",
                    help="root:k:worlds:horizon:rule[:stop][:own][:keyed][:aligned][:box4] or net:<checkpoint>")
    ap.add_argument("--holdout", type=int, default=10, help="one game in this many is held out")
    ap.add_argument("--seed", type=int, default=3637)
    ap.add_argument("--collect-only", action="store_true", help="write the positions, search nothing")
    ap.add_argument("--states", default=None, help="an earlier output: search its positions, keep its teachers")
    ap.add_argument("--shard", default=None, metavar="I/N", help="with --states: search this slice, write its rows")
    ap.add_argument("--merge", nargs="+", default=None, help="with --states: shard outputs to put back")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    teachers = [parse_teacher(t) for t in a.teacher]
    worlds = a.worlds if a.worlds is not None else ([] if teachers or a.states else [16, 64, 128])
    teachers = [(str(w), f"root:4:{w}:4:z2:earlier:own:keyed") for w in worlds] + teachers
    if len({n for n, _ in teachers}) != len(teachers):
        raise SystemExit("two teachers share a name")
    if (a.shard or a.merge) and not a.states:
        raise SystemExit("--shard and --merge take --states")
    meta_base: Dict[str, Any] = {"checkpoint": a.checkpoint, "seed": a.seed}

    if a.merge:
        for p in [a.states] + list(a.merge):
            check_raw_states(p)
        prev: Dict[str, Any] = _meta(a.states)
        specs = merged_specs(prev, [(p, _meta(p)) for p in a.merge], a.checkpoint, a.seed)
        base = _load(a.states)
        shards = [_load(p) for p in a.merge]
        out = merge(base, shards)
        _write(a.output, out, prev | {"teachers": specs, "merged_from": list(a.merge)})
        print(f"merged {len(shards)} shards: {sorted(specs)} -> {a.output}")
        return 0

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    t0 = time.time()
    if a.states:
        check_raw_states(a.states)
        base = _load(a.states)
        prev = _meta(a.states)
        raw = base["states"]
        heldout = base["heldout"].astype(bool)
        net = base["net_argmax"].astype(np.int64)
    else:
        found, game, argmax = collect(model, dev, a.games, a.seed)
        rng = np.random.default_rng(a.seed)
        keep = np.sort(rng.choice(len(found), size=min(a.positions, len(found)), replace=False))
        game_a = np.asarray(game, dtype=np.int32)[keep]
        heldout = (game_a % a.holdout) == 0
        raw = np.stack([np.frombuffer(found[i].raw_bytes(), dtype=np.uint8) for i in keep])
        net = np.asarray(argmax, dtype=np.int64)[keep]
        base = {"states": raw, "game": game_a, "heldout": heldout, "net_argmax": net.astype(np.int16)}
        prev = {"positions": int(keep.size), "games": a.games, "holdout": a.holdout}
        print(f"{keep.size:,} entry decisions from {a.games} games ({heldout.sum():,} held out) in {time.time() - t0:.0f}s",
              flush=True)
    if a.collect_only:
        _write(a.output, base, prev | meta_base | {"teachers": {}})
        return 0
    rows = shard_rows(raw.shape[0], a.shard) if a.shard else np.arange(raw.shape[0])
    shard_no = int(a.shard.partition("/")[0]) if a.shard else 0
    states = [ts.state_from_raw_bytes(raw[int(i)].tobytes()) for i in rows]
    out: Dict[str, np.ndarray] = {"rows": rows} if a.shard else dict(base)
    for name, spec in teachers:
        t1 = time.time()
        got = run_teacher(name, spec, model, dev, states, heldout[rows], a.seed, shard_no)
        out.update(got)
        _report(name, got, np.arange(rows.size), net[rows], heldout[rows], time.time() - t1)
    specs = dict(prev.get("teachers", {})) if not a.shard else {}
    specs.update({n: s for n, s in teachers})
    _write(a.output, out, (prev if not a.shard else {}) | meta_base | {"teachers": specs, "shard": a.shard})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
