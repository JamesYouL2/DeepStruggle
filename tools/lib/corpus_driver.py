"""Drive a policy, or the human corpus, one decision at a time into any tracker.

A *tracker* is anything with `observe(state, action)`: it is handed each decision, in order, before
its action is applied. Two sources feed one:

* `selfplay` -- a policy against itself, batched, from logits over observations or from a searcher
  that decides positions;
* `feed_corpus_game` -- a human game of the ts-replayer corpus, through the converter, its
  untrusted turns left out (`untrusted_turn`); `corpus_map` runs a function over every distinct
  game in a process pool.

Positions travel as the workbench's `pos=` token (`position_token` / `state_from_token`), so a
recorded decision opens in the browser as it stood. Every tool here builds E4 masks
(`ActionEncoder.get_legal_mask`), so an agent that decides in the merged-influence view is refused
(`require_e4_view`) rather than handed masks it would misread.
"""

from __future__ import annotations

import base64
import gzip
import json
import multiprocessing
import os
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import ts_engine as ts

from bindings.ts_env import TsVectorizedEnv, model_obs_features
from tools.lib.action_view import checkpoint_merged_influence
from tools.lib.corpus_paths import distinct_corpus_files
from tools.lib.player_agent import NeuralAgent, OnnxAgent, search_spec_config
from tools.lib.ts_replayer_convert import Conversion, convert_game

#: The rules' data files, wherever the tool is run from.
RULES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rules"))


def rules_json(name: str) -> Any:
    with open(os.path.join(RULES_DIR, name)) as f:
        return json.load(f)


def position_token(st: ts.GameState) -> str:
    """The workbench's `pos=` token for a position: the save JSON, zlib-deflated, base64url
    without padding (web/ui/src/game/position.ts)."""
    raw = zlib.compress(st.to_save_json().encode("utf-8"))
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def state_from_token(token: str) -> ts.GameState:
    raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    return ts.state_from_save_json(zlib.decompress(raw).decode("utf-8"))


def require_e4_view(spec: str, agent: Any) -> None:
    """Refuse an agent that decides in the merged-influence (E4.1, P23) view. A searcher carries
    no view of its own, so its checkpoint's is read."""
    merged = bool(getattr(agent, "merged_influence", False))
    if not merged and spec.lower().startswith(("search:", "gumbel:")):
        merged = checkpoint_merged_influence(search_spec_config(spec)[0])
    if merged:
        raise ValueError(f"{spec} decides in the merged-influence view; this tool builds E4 masks")


#: (observations, masks) -> logits, one row per game
LogitsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def load_policy(path: str) -> Tuple[LogitsFn, int]:
    """A checkpoint (.pt) or an ONNX export (.onnx, the form the published models take), as a
    logits function and the observation feature set it reads."""
    if path.endswith(".onnx"):
        agent = OnnxAgent(path)
        require_e4_view(path, agent)
        return agent.logits, 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural = NeuralAgent.from_checkpoint(path, device=str(dev))
    require_e4_view(path, neural)
    model = neural.model
    model.eval()

    def fn(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev),
                        torch.from_numpy(np.asarray(masks)).to(dev))[0]
        return out.float().cpu().numpy()
    return fn, int(model_obs_features(model))


#: positions -> one action each, for a policy that needs the state itself (a searcher)
StatesFn = Callable[[List[ts.GameState]], List[int]]


def selfplay(logits_fn: Optional[LogitsFn], obs_features: int, games: int, seed: int, batch: int,
             temperature: float, factory: Callable[[], Any], states_fn: Optional[StatesFn] = None) -> List[Any]:
    """Self-play of one policy against itself; every decision of game i goes to the i-th object
    `factory` makes (anything with `observe(state, action)`), and those are returned. The policy
    is `logits_fn` over observations, or `states_fn` over the positions themselves (a search agent,
    which decides every game's position in one batch)."""
    rng = np.random.default_rng(seed)
    out: List[Any] = []
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=seed + b0)
        env.set_obs_features(obs_features, obs_features)
        obs, masks, _ = env.reset_all()
        done = [False] * n
        trackers = [factory() for _ in range(n)]
        for _ in range(20_000):
            if all(done):
                break
            if states_fn is not None:
                live = [i for i in range(n) if not done[i]
                        and not ts.Engine.is_terminal(env.runner.get_state(i))]
                # A finished game's slot still takes an action; its first legal one will do.
                actions = np.asarray(masks).argmax(axis=1)
                if live:
                    for i, act in zip(live, states_fn([env.runner.get_state(i) for i in live])):
                        actions[i] = act
            elif logits_fn is None:
                raise ValueError("selfplay needs logits_fn or states_fn")
            elif temperature > 0:
                logits = logits_fn(np.asarray(obs), np.asarray(masks))
                z = np.exp((logits - logits.max(axis=1, keepdims=True)) / temperature)
                cdf = np.cumsum(z, axis=1)
                actions = np.minimum((cdf <= rng.random(len(cdf))[:, None] * cdf[:, -1:]).sum(axis=1),
                                     logits.shape[1] - 1)
            else:
                actions = logits_fn(np.asarray(obs), np.asarray(masks)).argmax(axis=1)
            for i in range(n):
                if done[i]:
                    continue
                st = env.runner.get_state(i)
                if not ts.Engine.is_terminal(st):
                    trackers[i].observe(st, int(actions[i]))
            obs, masks, _, dones, _ = env.step(actions)
            for i, d in enumerate(dones):
                if d:
                    done[i] = True
        out += trackers
    return out


def untrusted_turn(conv: Conversion) -> Optional[int]:
    """The first turn of a converted game whose positions are not to be trusted, or None. Where the
    record stops mid-turn (`truncated_at`) or the conversion fails (`failure`), the converter has
    only a fragment of that turn's hands, and positions from it can hold cards nobody held -- replay
    147, turn 9: the USSR with 14 cards. The converter keeps such a turn out of its training data;
    every consumer of its decisions must do the same. (A turn whose reveals the log contradicts,
    `reveal_conflict_turns`, never reaches a consumer: the converter does not emit it.)"""
    turns = [int(m.turn) for m in (conv.truncated_at, conv.failure) if m is not None]
    return min(turns) if turns else None


def read_corpus_game(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def feed_corpus_game(path: str, tracker: Any) -> str:
    """One corpus game through the converter, each human decision fed to `tracker.observe` --
    those before `untrusted_turn` only, so the decisions are held until the conversion ends:
    "skipped" (not convertible), "complete", or "partial" (the record stops early, or the
    conversion fails part-way, so the game's end is not seen)."""
    held: List[Tuple[ts.GameState, int]] = []
    conv = convert_game(read_corpus_game(path),
                        on_decision=lambda st, mover, entry, chosen: held.append((st, int(chosen))))
    if conv.skipped:
        return "skipped"
    cut = untrusted_turn(conv)
    for st, a in held:
        if cut is None or int(st.turn) < cut:
            tracker.observe(st, a)
    return "complete" if conv.game_ended and conv.failure is None and conv.truncated_at is None else "partial"


def corpus_map(work: Callable[[str], Any], workers: int, limit: int = 0) -> List[Any]:
    """`work(path)` over every distinct game of the ts-replayer corpus, in a process pool; `work`
    must be a module-level function so the pool can send it."""
    files, _ = distinct_corpus_files()
    paths = [str(p) for p in files][: limit or None]
    with multiprocessing.get_context("fork").Pool(max(1, workers)) as pool:
        return list(pool.imap_unordered(work, paths))
