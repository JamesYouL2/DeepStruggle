"""The browser workbench's WebAssembly engine is the native engine, bit for bit.

The page plays, displays and saves positions with a WebAssembly build of engine/ + bindings/
(tools/scripts/build_web.sh). It is compiled by a different compiler for a different machine, so
"the same sources" proves nothing by itself: an uninitialised byte, a float contraction or an
integer-width assumption would make the page play a different game while looking right. So the
two builds are run on the same games and compared.

Needs the built engine (web/ui/public/engine/ts_engine.mjs) and node, like the rest of tests/web
needs web/ui/dist. A missing build fails here; it is not skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from typing import Any, Dict, List

import numpy as np
import pytest

import ts_engine as ts
from tools.lib.engine_fingerprint import fingerprint
from tools.lib.game_step import drain_chance

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODULE = os.path.join(REPO, "web", "ui", "public", "engine", "ts_engine.mjs")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wasm_parity.mjs")


def _node(*args: str) -> Dict[str, Any]:
    assert os.path.exists(MODULE), (
        f"no WebAssembly engine at {MODULE}: build it with tools/scripts/build_web.sh --engine")
    node = shutil.which("node")
    assert node, "node is needed to run the WebAssembly engine"
    out = subprocess.run([node, RUNNER, MODULE, *args], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_wasm_engine_was_built_from_these_sources() -> None:
    got = _node("selftest", "1")["fingerprint"]
    assert got == fingerprint(), (
        f"web/ui/public/engine was built from other sources ({got[:12]}) than these "
        f"({fingerprint()[:12]}): rebuild it with tools/scripts/build_web.sh --engine")


def test_whole_games_match_the_native_engine() -> None:
    native_digest, native_steps = ts.selftest_digest(200)
    wasm = _node("selftest", "200")
    assert wasm["steps"] == native_steps
    assert wasm["digest"] == native_digest, (
        "the WebAssembly engine played different games from the native one "
        f"(digest {wasm['digest']:08x} vs {native_digest:08x})")


@pytest.mark.parametrize("seed", [11, 58])
def test_the_page_sees_exactly_what_python_sees(seed: int, tmp_path) -> None:
    """Display JSON, save JSON, both observations and the mask, at every step of a game driven
    through the MicroActions a click sends."""
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state)
    rng = random.Random(seed)
    h = hashlib.sha256()

    def record() -> None:
        h.update(state.to_display_json().encode())
        h.update(state.to_save_json().encode())
        for p in (ts.Player.US, ts.Player.USSR):
            h.update(np.asarray(ts.extract_observation(state, p), dtype=np.float32).tobytes())
        h.update(np.asarray(ts.get_flat_action_mask(state, False), dtype=np.uint8).tobytes())

    record()
    actions: List[List[int]] = []
    while not ts.Engine.is_terminal(state) and len(actions) < 1500:
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(state, False))[0]]
        ma = ts.decode_flat_action(state, rng.choice(legal))
        actions.append([int(ma.decision_type), int(ma.primary_id), int(ma.secondary_id), int(ma.flags)])
        ts.Engine.step(state, ma)
        drain_chance(state)
        record()

    path = tmp_path / "actions.json"
    path.write_text(json.dumps({"seed": seed, "actions": actions}))
    wasm = _node("walk", str(path))
    assert "error" not in wasm, wasm.get("error")
    assert wasm["steps"] == len(actions)
    assert wasm["digest"] == h.hexdigest(), "the page's engine and Python diverged along this game"
    if ts.Engine.is_terminal(state):
        assert wasm["ending"] == ts.game_ending_reason(state)


@pytest.mark.parametrize("seed,plies,horizon,stop,leaf", [
    (21, 70, 2, "boundaries", "mover"), (34, 150, 1, "boundaries", "mover"), (47, 40, 0, "boundaries", "mover"),
    (58, 90, 4, "turn", "mover"), (63, 120, 2, "later", "mover"), (71, 130, 3, "earlier", "mover"),
    (58, 90, 4, "turn", "decider"), (77, 110, 2, "boundaries", "decider"), (58, 90, 4, "turn", "own"),
    (83, 100, 3, "earlier", "own")])
def test_the_rollout_batch_plays_the_same_worlds_in_both_builds(seed: int, plies: int, horizon: int, stop: str,
                                                                 leaf: str, tmp_path) -> None:
    """The rollout root's playouts (bindings/rollout_batch.hpp) under a fixed synthetic policy and
    critic: the same rows, the same worlds, the same observations ply by ply, the same values."""
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    drain_chance(state)
    rng = random.Random(seed)
    actions: List[List[int]] = []
    while not ts.Engine.is_terminal(state) and len(actions) < plies:
        legal = [int(i) for i in np.nonzero(ts.get_flat_action_mask(state, False))[0]]
        ma = ts.decode_flat_action(state, rng.choice(legal))
        actions.append([int(ma.decision_type), int(ma.primary_id), int(ma.secondary_id), int(ma.flags)])
        ts.Engine.step(state, ma)
        drain_chance(state)
    assert not ts.Engine.is_terminal(state)
    k, worlds, batch_seed = 3, 4, 987654321
    mask = np.asarray(ts.get_flat_action_mask(state, False))
    cands = [int(a) for a in np.flatnonzero(mask)[:k]]
    act = int(mask.shape[0])

    batch = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=0, seed=batch_seed, stop=stop,
                            leaf=leaf)
    rows = batch.add(state, cands, worlds)
    h = hashlib.sha256()
    plies_run = 0
    a_idx = np.arange(act, dtype=np.int64)
    while not batch.all_done():
        obs, masks = batch.fill_live()
        h.update(np.ascontiguousarray(obs, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(masks, dtype=np.uint8).tobytes())
        row = (((a_idx * 2654435761) % 4294967296 + plies_run * 40503 + 7) % 1000) / 1000.0
        lg = np.ascontiguousarray(np.tile(row.astype(np.float32), (obs.shape[0], 1)))
        batch.step(lg)
        plies_run += 1
        assert plies_run < 2000
    lobs, _lm = batch.fill_leaves()
    h.update(np.ascontiguousarray(lobs, dtype=np.float32).tobytes())
    leaves = int(lobs.shape[0])
    batch.set_leaf_values(np.asarray([((j * 31) % 17) / 17 - 0.5 for j in range(leaves)], dtype=np.float32))
    native = {
        "candidates": cands, "rows": rows, "worlds_used": batch.worlds_used(0), "plies": plies_run,
        "leaves": leaves, "refused": batch.refused(), "digest": h.hexdigest(),
        "values": [float(v) for v in np.asarray(batch.values(), dtype=np.float32)],
        "table": [[p, w, c, int(d), int(t)] for p, w, c, d, t in zip(
            batch.row_position(), batch.row_world(), batch.row_candidate(), batch.row_done(), batch.row_terminal())],
    }
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({"seed": seed, "actions": actions, "k": k, "worlds": worlds,
                                "horizon": horizon, "batch_seed": batch_seed, "stop": stop, "leaf": leaf}))
    # The root's choice from synthetic root logits, under each rule, and the top-3 pick from them.
    root = np.asarray([((a * 7919) % 97) / 97 for a in range(act)], dtype=np.float32)
    top = ts.rollout_candidates(root, np.ascontiguousarray(mask, dtype=np.uint8), 3)
    decisions: Dict[str, Dict[str, Any]] = {}
    for rule in ("argmax", "z2", "kl0.05"):
        d = ts.rollout_decide(batch, 0, cands, root, np.ascontiguousarray(mask, dtype=np.uint8), rule)
        decisions[rule] = {"pick": d["pick"], "worlds": d["worlds"], "q": [float(x) for x in d["q"]],
                           "lead": [float(x) for x in d["lead"]], "lead_se": [float(x) for x in d["lead_se"]]}
    wasm = _node("rollout", str(path))
    assert "error" not in wasm, wasm.get("error")
    for key in ("candidates", "rows", "worlds_used", "plies", "leaves", "refused", "table"):
        assert wasm[key] == native[key], key
    assert wasm["top"] == top
    assert wasm["digest"] == native["digest"], "the two builds' playouts diverged"
    assert np.array_equal(np.asarray(wasm["values"], dtype=np.float32), np.asarray(native["values"], dtype=np.float32))
    for rule, want in decisions.items():
        got = wasm["decisions"][rule]
        assert (got["pick"], got["worlds"]) == (want["pick"], want["worlds"]), rule
        for key in ("q", "lead", "lead_se"):
            assert np.allclose(np.asarray(got[key], dtype=np.float32), np.asarray(want[key], dtype=np.float32),
                               rtol=1e-6, atol=1e-7), (rule, key)
    assert rows > 0 and native["refused"] == 0
