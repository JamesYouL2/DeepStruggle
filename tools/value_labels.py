#!/usr/bin/env python3
"""P34's labelling tool: decision positions labelled by greedy playouts to the game's end.

Three stages, each shardable with --part i/n (research/plans/P34_search_centric_evaluator.md):

  positions  Self-play of the generator at --temperature, one decision in --keep kept: the state's
             bytes, its game and ply, the mover, the generator's own game result from the mover's
             side, and the generator's top --candidates legal moves (the sibling slice's
             candidates). Game g is played from --seed and g; games with g % 20 == 0 ... are held out
             by --holdout (by game, never by position).
  label      Every position's --worlds worlds played to the end by the generator on both sides
             under the safe-playout rule (bindings/playout_labels.hpp), and, for the first
             --siblings positions of a fixed shuffle, each candidate in --sibling-worlds shared
             worlds with its leaf (the mover's next decision) kept.
  merge      The parts joined into train / holdout files with the labels as means, a manifest with
             the D code, the generator, the engine, device, seed and sha256 of every file.

    tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/value_labels.py \\
        positions --model <network or .pt> --games 20000 --out data/datasets/p34/<name>
    ... label --dir data/datasets/p34/<name> [--part 0/1]
    ... merge --dir data/datasets/p34/<name>     # the D code is derived (tools/scripts/run_codes.py)

The labeller's network forward runs as a CUDA graph over a fixed batch, from page-locked buffers,
with two labellers alternating so the engine steps one while the GPU evaluates the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from bindings.ts_env import model_obs_features
from tools.lib import leaderboard
from tools.lib.action_view import checkpoint_merged_influence
from tools.lib.checkpoint_id import checkpoint_label
from tools.lib.data_root import data_root
from tools.lib.engine_fingerprint import fingerprint, staleness_reason
from tools.lib.player_agent import NeuralAgent
from tools.scripts.run_codes import labelling_code

A = 220
#: A position's id is game * PLY_SPAN + ply.
PLY_SPAN = 100_000
REGIONS = ("europe", "asia", "middle_east", "africa", "central_america", "south_america", "southeast_asia")


# ---- shared ------------------------------------------------------------------------------------

def _part(text: str) -> Tuple[int, int]:
    i, n = (int(x) for x in text.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise argparse.ArgumentTypeError(f"--part {text}: want i/n with 0 <= i < n")
    return i, n


def resolve_model(spec: str) -> Tuple[str, str]:
    """(path, id) for a registered network name or a checkpoint path."""
    board = leaderboard.load()
    if spec in board.networks:
        return os.path.join(data_root(), str(board.networks[spec]["file"])), spec
    if not os.path.isfile(spec):
        raise SystemExit(f"{spec}: neither a registered network nor a file")
    return spec, checkpoint_label(spec)


def load_model(path: str, device: torch.device) -> Any:
    if checkpoint_merged_influence(path):
        # The labeller steps the E4 view, as the rollout root does.
        raise SystemExit(f"{path} decides in the merged-influence view, which the labeller does not play")
    return NeuralAgent.from_checkpoint(path, device=str(device)).model.eval()


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def engine_checked() -> str:
    why = staleness_reason(str(ts.__file__))
    if why:
        raise SystemExit(why)
    return fingerprint()


class GraphedPolicy:
    """logits = model(obs) over a fixed batch for one labeller: fill(labeller) writes the live rows'
    observations into page-locked memory, launch(n) runs the forward asynchronously, and wait()
    returns the logits.

    With `codes` (CUDA only) each observation float travels as its one-byte code in the labeller's
    codebook and is decoded on the GPU inside the graph: exact, and a quarter of the upload, which
    is what bounds the loop otherwise. Policies built with the same `streams` share one upload
    stream and one compute stream: one policy's upload overlaps another's forward, and no two graphs
    ever replay at once (two graphs replaying concurrently on separate streams hung the GPU,
    2026-10-17)."""

    def __init__(self, model: Any, capacity: int, width: int, device: torch.device, graphs: bool = True,
                 streams: Optional[Tuple[Any, Any]] = None, codes: bool = True) -> None:
        self.model = model
        self.device = device
        cuda = device.type == "cuda"
        self.codes = codes and cuda
        self.buf_t = torch.zeros((capacity, width), dtype=torch.uint8 if self.codes else torch.float32,
                                 pin_memory=cuda)
        self.buf = self.buf_t.numpy()
        self.logits_t = torch.zeros((capacity, A), dtype=torch.float32, pin_memory=cuda)
        self.logits = self.logits_t.numpy()
        self.graph: Optional[Any] = None
        self.event: Optional[Any] = None
        self.book_n = 0
        if not cuda:
            return
        self.buf_dev = torch.zeros((capacity, width), dtype=self.buf_t.dtype, device=device)
        self.book_t = torch.zeros(256, dtype=torch.float32, pin_memory=True)
        self.book_dev = torch.zeros((256, 1), dtype=torch.float32, device=device)
        self.event = torch.cuda.Event()
        self.uploaded = torch.cuda.Event()
        self.copy_stream, self.compute = streams if streams is not None else policy_streams(device)
        if not graphs:
            return
        self.compute.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.compute), torch.no_grad():
            for _ in range(2):
                self._forward()
        torch.cuda.current_stream(device).wait_stream(self.compute)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out_dev = self._forward()
        torch.cuda.synchronize(device)

    def _forward(self, n: Optional[int] = None) -> torch.Tensor:
        x = self.buf_dev if n is None else self.buf_dev[:n]
        if self.codes:
            x = torch.nn.functional.embedding(x.int(), self.book_dev).squeeze(-1)
        return self.model(x, None)[0].float()

    def fill(self, labeller: Any) -> int:
        if not self.codes:
            return int(labeller.fill_live(self.buf))
        n = int(labeller.fill_live_codes(self.buf))
        book = labeller.codebook()
        if book.size != self.book_n:
            self.book_t[:book.size] = torch.from_numpy(book)
            self.book_n = int(book.size)
        return n

    def launch(self, n: int) -> None:
        if n == 0:
            return
        if self.device.type != "cuda":
            with torch.no_grad():
                self.logits_t[:n] = self.model(self.buf_t[:n], None)[0].float()
            return
        assert self.event is not None
        with torch.cuda.stream(self.copy_stream):
            self.buf_dev[:n].copy_(self.buf_t[:n], non_blocking=True)
            if self.codes:
                self.book_dev.view(-1).copy_(self.book_t, non_blocking=True)
            self.uploaded.record(self.copy_stream)
        with torch.cuda.stream(self.compute):
            self.compute.wait_event(self.uploaded)
            if self.graph is not None:
                self.graph.replay()
                out = self.out_dev
            else:
                with torch.no_grad():
                    out = self._forward(n)
            self.logits_t[:n].copy_(out[:n], non_blocking=True)
            self.event.record(self.compute)

    def wait(self) -> np.ndarray:
        if self.event is not None:
            self.event.synchronize()
        return self.logits


def policy_streams(device: torch.device) -> Tuple[Any, Any]:
    """(upload stream, compute stream) for GraphedPolicy objects that alternate."""
    return torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)


# ---- positions ---------------------------------------------------------------------------------

def run_positions(a: argparse.Namespace) -> int:
    engine = engine_checked()
    device = torch.device(a.device)
    path, model_id = resolve_model(a.model)
    model = load_model(path, device)
    features = model_obs_features(model)
    part, parts = a.part
    games = [g for g in range(a.games) if g % parts == part]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    envs = min(a.envs, len(games))
    runner = ts.VectorizedBatchRunner(envs, a.seed)
    runner.set_obs_features([features] * envs, [features] * envs)
    rng = np.random.default_rng([a.seed, part, 1])
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed * 1_000_003 + part)
    nxt = 0
    game_of = np.full(envs, -1, dtype=np.int64)
    ply_of = np.zeros(envs, dtype=np.int64)
    for i in range(envs):
        game_of[i] = games[nxt]
        runner.reset_game(i, a.seed * PLY_SPAN + games[nxt])
        nxt += 1
    kept: Dict[str, List[Any]] = {k: [] for k in ("state", "id", "mover", "turn", "cand")}
    result_of: Dict[int, float] = {}   # game -> US score (1 / 0.5 / 0)
    done_games = 0
    t0 = time.time()
    while done_games < len(games):
        live = game_of >= 0
        obs = torch.from_numpy(np.asarray(runner.get_observations())).to(device, non_blocking=True)
        masks = torch.from_numpy(np.asarray(runner.get_action_masks())).to(device)
        with torch.no_grad():
            logits = model(obs, None)[0].float()
        legal = masks > 0
        legal[:, 0] |= ~legal.any(dim=1)   # a finished env has no legal move; it is not stepped
        z = logits.masked_fill(~legal, float("-inf"))
        probs = torch.softmax(z / a.temperature, dim=-1)
        acts = torch.multinomial(probs, 1, generator=gen).squeeze(1).cpu().numpy()
        top = z.topk(a.candidates, dim=-1)
        top_idx = top.indices.cpu().numpy()
        top_ok = torch.isfinite(top.values).cpu().numpy()
        keep = live & (rng.random(envs) < 1.0 / a.keep)
        movers = runner.get_decision_players()
        for i in np.flatnonzero(keep):
            st = runner.get_state(int(i))
            kept["state"].append(np.frombuffer(st.raw_bytes(), dtype=np.uint8))
            kept["id"].append(int(game_of[i]) * PLY_SPAN + int(ply_of[i]))
            kept["mover"].append(int(movers[i]))
            kept["turn"].append(int(st.turn))
            kept["cand"].append(np.where(top_ok[i], top_idx[i], -1).astype(np.int16))
        res = runner.step_flat_all([int(x) for x in acts], True)
        if any(res[i] == 0 for i in np.flatnonzero(live)):
            raise RuntimeError("the engine refused a sampled self-play action")
        ply_of[live] += 1
        terms = runner.get_terminals()
        utils = runner.get_terminal_utilities()
        for i in np.flatnonzero(live):
            if not terms[i]:
                continue
            result_of[int(game_of[i])] = 0.5 * (1.0 + float(utils[i]))
            done_games += 1
            if nxt < len(games):
                game_of[i] = games[nxt]
                ply_of[i] = 0
                runner.reset_game(int(i), a.seed * PLY_SPAN + games[nxt])
                nxt += 1
            else:
                game_of[i] = -1
        if done_games and done_games % 1000 == 0 and terms.count(True):
            print(f"  {done_games}/{len(games)} games, {len(kept['id'])} positions, {time.time() - t0:.0f}s",
                  flush=True)
    ids = np.asarray(kept["id"], dtype=np.int64)
    game = ids // PLY_SPAN
    us_score = np.asarray([result_of[int(g)] for g in game], dtype=np.float32)
    mover = np.asarray(kept["mover"], dtype=np.int8)
    us = int(ts.Player.US)
    order = np.argsort(ids)
    file = out / f"positions_{part:03d}_of_{parts:03d}.npz"
    np.savez(file,
             state=np.stack(kept["state"])[order] if kept["state"] else np.zeros((0, ts.state_size()), np.uint8),
             id=ids[order],
             mover=mover[order],
             turn=np.asarray(kept["turn"], dtype=np.int8)[order],
             cand=np.stack(kept["cand"])[order] if kept["cand"] else np.zeros((0, a.candidates), np.int16),
             game_score=np.where(mover == us, us_score, 1.0 - us_score)[order].astype(np.float32),
             holdout=(_holdout(game, a.holdout, a.seed))[order])
    meta = {"stage": "positions", "generator": model_id, "generator_file": path,
            "generator_sha256": sha256(pathlib.Path(path)), "engine": engine, "games": a.games,
            "part": f"{part}/{parts}", "seed": a.seed, "temperature": a.temperature, "keep": a.keep,
            "candidates": a.candidates, "holdout": a.holdout, "obs_features": features,
            "device": str(device), "positions": int(ids.size), "seconds": round(time.time() - t0, 1)}
    file.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{file}: {ids.size} positions from {len(games)} games in {time.time() - t0:.0f}s")
    return 0


def _holdout(game: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """A game's split from its index alone, so every part and every stage agrees."""
    h = (game.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed)) >> np.uint64(40)
    return (h.astype(np.float64) / float(1 << 24)) < frac


# ---- label -------------------------------------------------------------------------------------

def _load_positions(d: pathlib.Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    files = sorted(d.glob("positions_*.npz"))
    if not files:
        raise SystemExit(f"{d}: no positions_*.npz")
    metas = [json.loads(f.with_suffix(".json").read_text()) for f in files]
    for m in metas[1:]:
        for k in ("generator_sha256", "engine", "seed", "temperature", "keep"):
            if m[k] != metas[0][k]:
                raise SystemExit(f"positions parts disagree on {k}")
    parts = {int(m["part"].split("/")[1]) for m in metas}
    if len(parts) != 1 or len(files) != parts.pop():
        raise SystemExit(f"{d}: the positions parts are incomplete")
    arrs = [dict(np.load(f)) for f in files]
    cat = {k: np.concatenate([x[k] for x in arrs]) for k in arrs[0]}
    order = np.argsort(cat["id"])
    return {k: v[order] for k, v in cat.items()}, metas[0]


def sibling_ids(ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    """The sibling slice: the first n of a fixed shuffle of the position ids."""
    perm = np.random.default_rng([seed, 2]).permutation(ids.size)
    return np.sort(ids[perm[:n]])


def _drive(labellers: Sequence[Any], policies: Sequence[GraphedPolicy], feed: Any, sink: Any) -> int:
    """Run the labellers to idle, alternating so one steps while the other's batch is on the GPU."""
    n = [0] * len(labellers)
    for j, (pl, pol) in enumerate(zip(labellers, policies)):
        feed(pl)
        n[j] = pol.fill(pl)
        pol.launch(n[j])
    rows = 0
    while any(not pl.idle() for pl in labellers):
        for j, (pl, pol) in enumerate(zip(labellers, policies)):
            if n[j] == 0 and pl.idle():
                feed(pl)
                if pl.idle():
                    continue
            logits = pol.wait()
            if n[j]:
                pl.step(logits)
                rows += n[j]
            sink(pl)
            feed(pl)
            n[j] = pol.fill(pl)
            pol.launch(n[j])
    return rows


def run_label(a: argparse.Namespace) -> int:
    engine = engine_checked()
    d = pathlib.Path(a.dir)
    pos, pmeta = _load_positions(d)
    if pmeta["engine"] != engine:
        raise SystemExit(f"positions were written by engine {pmeta['engine'][:8]}, this is {engine[:8]}")
    device = torch.device(a.device)
    path, model_id = resolve_model(a.model) if a.model else (pmeta["generator_file"], pmeta["generator"])
    if sha256(pathlib.Path(path)) != pmeta["generator_sha256"] and not a.model:
        raise SystemExit("the generator file changed since the positions were written")
    model = load_model(path, device)
    features = model_obs_features(model)
    part, parts = a.part
    mine = np.arange(pos["id"].size) % parts == part
    ids, states = pos["id"][mine], pos["state"][mine]
    sib = np.isin(ids, sibling_ids(pos["id"], a.siblings, a.seed))
    cand = pos["cand"][mine]
    t0 = time.time()

    labellers = [ts.PlayoutLabels(features=features, worlds=a.worlds, capacity=a.capacity, seed=a.seed)
                 for _ in range(2)]
    sib_labellers = [ts.PlayoutLabels(features=features, worlds=a.sibling_worlds, capacity=a.capacity,
                                      seed=a.seed + 1, leaf_horizon=a.leaf_horizon) for _ in range(2)]
    streams = policy_streams(device) if device.type == "cuda" else None
    policies = [GraphedPolicy(model, a.capacity, labellers[0].obs_width(), device, graphs=not a.no_graphs,
                              streams=streams, codes=not a.float_upload) for _ in range(2)]

    main_out: List[Dict[str, np.ndarray]] = []
    sib_out: List[Dict[str, np.ndarray]] = []
    cursor = {"main": 0, "sib": 0}
    chunk = max(1, 2 * a.capacity // a.worlds)
    sib_rows = np.flatnonzero(sib)

    def feed_main(pl: Any) -> None:
        while pl.queued() < chunk and cursor["main"] < ids.size:
            lo = cursor["main"]
            hi = min(ids.size, lo + chunk)
            pl.enqueue_raw(np.ascontiguousarray(states[lo:hi]), np.ascontiguousarray(ids[lo:hi]))
            cursor["main"] = hi

    def feed_sib(pl: Any) -> None:
        k = a.sibling_k
        per = max(1, 2 * a.capacity // (a.sibling_worlds * k))
        while pl.queued() < per and cursor["sib"] < sib_rows.size:
            lo = cursor["sib"]
            take = sib_rows[lo:lo + per]
            cursor["sib"] = lo + take.size
            c = cand[take, :k].astype(np.int32)
            ok = (c >= 0).all(axis=1)   # positions with fewer than k legal moves stay out of the slice
            if ok.any():
                pl.enqueue_raw(np.ascontiguousarray(states[take[ok]]), np.ascontiguousarray(ids[take[ok]]),
                               np.ascontiguousarray(c[ok]))

    last = [time.time()]

    def sink_main(pl: Any) -> None:
        got = pl.take_positions()
        if got["id"].size:
            main_out.append(got)
        if time.time() - last[0] > 60:
            last[0] = time.time()
            done = sum(x["id"].size for x in main_out)
            print(f"  {done}/{ids.size} positions labelled, {time.time() - t0:.0f}s", flush=True)

    def sink_sib(pl: Any) -> None:
        got = pl.take_siblings()
        if got["id"].size:
            sib_out.append(got)

    rows = _drive(labellers, policies, feed_main, sink_main)
    t_main = time.time() - t0
    print(f"main labels: {ids.size} positions x {a.worlds} worlds, {rows} network rows in {t_main:.0f}s "
          f"({rows / max(t_main, 1e-9):,.0f} rows/s)", flush=True)
    srows = _drive(sib_labellers, policies, feed_sib, sink_sib) if sib_rows.size else 0
    t_all = time.time() - t0

    def cat(chunks: List[Dict[str, np.ndarray]], key: str, width: Tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.concatenate([c[key] for c in chunks]) if chunks else np.zeros((0,) + width, dtype)

    file = d / f"labels_{part:03d}_of_{parts:03d}.npz"
    np.savez(file, id=cat(main_out, "id", (), np.int64), count=cat(main_out, "count", (2,), np.int32),
             sums=cat(main_out, "sums", (5,), np.float32), swing_sum=cat(main_out, "swing_sum", (7,), np.float32),
             swing_n=cat(main_out, "swing_n", (7,), np.int32))
    sfile = d / f"siblings_{part:03d}_of_{parts:03d}.npz"
    np.savez(sfile, id=cat(sib_out, "id", (), np.int64), meta=cat(sib_out, "meta", (5,), np.int32),
             cols=cat(sib_out, "cols", (4,), np.float32), swing=cat(sib_out, "swing", (7,), np.int32),
             leaf=cat(sib_out, "leaf", (ts.state_size(),), np.uint8))
    meta = {"stage": "label", "labeller": model_id, "labeller_sha256": sha256(pathlib.Path(path)),
            "engine": engine, "part": f"{part}/{parts}", "seed": a.seed, "worlds": a.worlds, "safe": True,
            "siblings": a.siblings, "sibling_worlds": a.sibling_worlds, "sibling_k": a.sibling_k,
            "leaf_horizon": a.leaf_horizon, "capacity": a.capacity, "device": str(device),
            "torch": torch.__version__, "cuda_graphs": not a.no_graphs, "code_upload": not a.float_upload,
            "network_rows": int(rows + srows), "seconds": round(t_all, 1)}
    file.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{file}: {sum(x['id'].size for x in main_out)} positions; {sfile}: "
          f"{sum(x['id'].size for x in sib_out)} sibling rows; {rows + srows} network rows in {t_all:.0f}s")
    return 0


# ---- merge -------------------------------------------------------------------------------------

def labelling_signature(pmeta: Dict[str, Any], lmeta: Dict[str, Any]) -> Dict[str, Any]:
    """The labelling recipe: every setting that changes what a dataset holds, none of where or how
    fast it was made. Its D code comes from tools/scripts/run_codes.py's `labellings`."""
    return {"temperature": pmeta["temperature"], "keep": pmeta["keep"], "candidates": pmeta["candidates"],
            "holdout": pmeta["holdout"], "worlds": lmeta["worlds"], "stop": "game", "safe": lmeta["safe"],
            "siblings": lmeta["siblings"], "sibling_worlds": lmeta["sibling_worlds"],
            "sibling_k": lmeta["sibling_k"], "leaf_horizon": lmeta["leaf_horizon"],
            "labels": ["score", "final_vp", "turn_vp", "turn_defcon", "region_swing"]}


def run_merge(a: argparse.Namespace) -> int:
    d = pathlib.Path(a.dir)
    pos, pmeta = _load_positions(d)
    lfiles = sorted(d.glob("labels_*.npz"))
    lmetas = [json.loads(f.with_suffix(".json").read_text()) for f in lfiles]
    if not lfiles or len(lfiles) != int(lmetas[0]["part"].split("/")[1]):
        raise SystemExit(f"{d}: the label parts are incomplete")
    for m in lmetas:
        if m["engine"] != pmeta["engine"]:
            raise SystemExit("a label part was written by another engine than the positions")
    lab = [dict(np.load(f)) for f in lfiles]
    L = {k: np.concatenate([x[k] for x in lab]) for k in lab[0]}
    order = np.argsort(L["id"])
    L = {k: v[order] for k, v in L.items()}
    if not np.array_equal(L["id"], pos["id"]):
        raise SystemExit("labels and positions do not cover the same ids")
    n = L["count"][:, 0].astype(np.float64)
    if (n == 0).any():
        raise SystemExit(f"{int((n == 0).sum())} positions have no finished world")
    s = L["sums"].astype(np.float64)
    score = s[:, 0] / n
    var = np.maximum(s[:, 1] / n - score ** 2, 0.0)
    labels = {
        "score": score.astype(np.float32),
        "score_se": np.sqrt(var / np.maximum(n - 1, 1)).astype(np.float32),
        "final_vp": (s[:, 2] / n / 20.0).astype(np.float32),
        "turn_vp": (s[:, 3] / n / 20.0).astype(np.float32),
        "turn_defcon": (s[:, 4] / n).astype(np.float32),
        "swing": np.where(L["swing_n"] > 0, L["swing_sum"] / np.maximum(L["swing_n"], 1), np.nan).astype(np.float32),
        "swing_frac": (L["swing_n"] / n[:, None]).astype(np.float32),
        "worlds": L["count"][:, 0].astype(np.int16),
        "failed": L["count"][:, 1].astype(np.int16),
    }
    files: Dict[str, str] = {}
    for split, sel in (("train", ~pos["holdout"]), ("holdout", pos["holdout"])):
        f = d / f"{split}.npz"
        np.savez(f, **{k: v[sel] for k, v in pos.items()}, **{k: v[sel] for k, v in labels.items()})
        files[f.name] = sha256(f)
    sfiles = sorted(d.glob("siblings_*.npz"))
    if sfiles:
        sib = [dict(np.load(f)) for f in sfiles]
        S = {k: np.concatenate([x[k] for x in sib]) for k in sib[0]}
        hold_games = set((pos["id"][pos["holdout"]] // PLY_SPAN).tolist())
        S["holdout"] = np.isin(S["id"] // PLY_SPAN, list(hold_games))
        f = d / "siblings.npz"
        np.savez(f, id=S["id"], meta=S["meta"], cols=S["cols"], swing=S["swing"], leaf=S["leaf"],
                 holdout=S["holdout"])
        files[f.name] = sha256(f)
    code = labelling_code(labelling_signature(pmeta, lmetas[0]), update=True)
    assert code is not None
    if a.code and a.code != code:
        raise SystemExit(f"the labelling settings are {code}'s, not {a.code}'s (tools/scripts/run_codes.py labellings)")
    name = f"{pmeta['generator']}~{code}(seed={pmeta['seed']},games={pmeta['games']})"
    manifest = {"dataset": name, "code": code, "labelling": labelling_signature(pmeta, lmetas[0]),
                "generator": pmeta["generator"],
                "generator_sha256": pmeta["generator_sha256"], "engine": pmeta["engine"],
                "positions": {k: v for k, v in pmeta.items() if k not in ("part", "positions", "seconds")},
                "label": {k: v for k, v in lmetas[0].items() if k not in ("part", "network_rows", "seconds")},
                "rows": {"positions": int(pos["id"].size), "train": int((~pos["holdout"]).sum()),
                         "holdout": int(pos["holdout"].sum()),
                         "network_rows": int(sum(m["network_rows"] for m in lmetas))},
                "label_seconds": round(sum(m["seconds"] for m in lmetas), 1),
                "regions": list(REGIONS), "files": files,
                "reads": {"mean_score_se": float(labels["score_se"].mean()),
                          "failed_worlds": int(labels["failed"].sum()),
                          "holdout_games_in_train": int(np.isin(pos["id"][~pos["holdout"]] // PLY_SPAN,
                                                               list(set((pos["id"][pos["holdout"]] // PLY_SPAN).tolist()))).sum())}}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["rows"]), json.dumps(manifest["reads"]))
    print(f"| `{name}` | {manifest['rows']['positions']:,} | {manifest['rows']['network_rows']:,} | "
          f"{lmetas[0]['device']} | {', '.join(f'{k} {v[:12]}' for k, v in files.items())} | |")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    po = sub.add_parser("positions", help="self-play decision positions of the generator")
    po.add_argument("--model", required=True, help="a registered network name or a .pt checkpoint")
    po.add_argument("--games", type=int, default=20000)
    po.add_argument("--temperature", type=float, default=0.25)
    po.add_argument("--keep", type=int, default=8, help="keep one decision in this many")
    po.add_argument("--candidates", type=int, default=4, help="the generator's top legal moves recorded")
    po.add_argument("--holdout", type=float, default=0.05, help="fraction of games held out")
    po.add_argument("--seed", type=int, default=44)
    po.add_argument("--envs", type=int, default=2048)
    po.add_argument("--part", type=_part, default=(0, 1))
    po.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    po.add_argument("--out", required=True)
    la = sub.add_parser("label", help="playout labels for the positions in --dir")
    la.add_argument("--dir", required=True)
    la.add_argument("--model", default=None, help="the playout network (default: the generator)")
    la.add_argument("--worlds", type=int, default=32)
    la.add_argument("--siblings", type=int, default=50000, help="positions in the sibling slice")
    la.add_argument("--sibling-worlds", type=int, default=16)
    la.add_argument("--sibling-k", type=int, default=4)
    la.add_argument("--leaf-horizon", type=int, default=0,
                    help="sibling leaves: boundaries before the mover's own decision is the leaf (0: the next one)")
    la.add_argument("--capacity", type=int, default=16384, help="rows per labeller batch")
    la.add_argument("--seed", type=int, default=44)
    la.add_argument("--part", type=_part, default=(0, 1))
    la.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    la.add_argument("--no-graphs", action="store_true", help="plain forwards instead of CUDA graphs")
    la.add_argument("--float-upload", action="store_true",
                    help="upload observations as float32 instead of their exact one-byte codes")
    me = sub.add_parser("merge", help="join the parts into train / holdout files and the manifest")
    me.add_argument("--dir", required=True)
    me.add_argument("--code", default=None,
                    help="the D code expected; refused if the one derived from the settings differs")
    a = ap.parse_args(argv)
    if a.stage == "positions":
        return run_positions(a)
    if a.stage == "label":
        return run_label(a)
    return run_merge(a)


if __name__ == "__main__":
    sys.exit(main())
