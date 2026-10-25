"""P33c: does the critic rank the searched player's confirmed leaks the way the playouts do?

The searched-leak bank (research/log/E7_searched_leak_R39.md) holds positions where the searched player
(R39 + rollout root) played a move, and a 2,048-pair paired playout confirmed another move as better
or not. For each confirmed position this applies both moves, reads a network's critic at the decision
that follows -- in the mover's frame, averaged over the dice the move may roll -- and sets the critic's
difference (better move minus the player's move) against the playouts' difference.

If the critic ranks the labelled pairs no better than chance where the leak is (DEFCON-2 card choices
above all), the leak is the critic's; if it ranks them right while the search still kept the wrong
move, the fault is the search's horizon or its world noise (research/plans/P33_search_preserved_gains.md).

The confirmed pairs are selected: they are where the search, which reads a critic at its leaves, chose
wrongly, so a critic that is wrong there more often than chance is partly that selection. The
baseline is `--screen`: the bank's screening playouts, every candidate of every bank position against
the player's move at 512 pairs, of which the pairs whose label is signed by |z| >= --min-z are scored
the same way -- the critic's resolution on positions not chosen for being leaks.

    PYTHONPATH=.:build/release python -m ai.eval.leak_critic --bank bank.jsonl.gz \\
        --playouts confirm_playouts.jsonl.gz --checkpoint A.pt [B.pt ...] [--dice 16]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

import ts_engine as ts
from ai.eval.paired_playouts import ARKey, ar_key, decider
from ai.search.rollout_root import RolloutConfig, RolloutRoot
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features
from tools.lib.corpus_driver import state_from_token
from tools.lib.game_step import drain_chance
from tools.lib.player_agent import NeuralAgent

_UINT64 = 1 << 64


@dataclass
class LeakPair:
    """One confirmed bank position: the player's move, the playouts' better move and their label."""
    id: str
    state: ts.GameState
    mover: ts.Player
    played: int
    best: int
    label: float            # playout score of `best` minus that of `played`, mover's side, in [-1, 1]
    weight: float           # the bank's population weight
    decision_type: str
    defcon: int
    turn: int
    side: str


def load_pairs(bank_path: str, playouts_path: str) -> List[LeakPair]:
    bank: Dict[str, Dict[str, Any]] = {}
    with gzip.open(bank_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            bank[r["id"]] = r
    pairs: List[LeakPair] = []
    with gzip.open(playouts_path, "rt") as f:
        for line in f:
            c = json.loads(line)
            b = bank[c["id"]]
            played = int(b["played"])
            moves = [int(k) for k in c["scores"]]
            if played not in moves or len(moves) != 2:
                raise ValueError(f"row {c['id']}: expected the played move and one other, got {moves}")
            best = next(m for m in moves if m != played)
            label = float(np.mean(c["scores"][str(best)]) - np.mean(c["scores"][str(played)]))
            st = state_from_token(b["pos"])
            pairs.append(LeakPair(c["id"], st, decider(st), played, best, label, float(b["weight"]),
                                  str(b["decision_type"]), int(st.defcon), int(st.turn), str(b["side"])))
    return pairs


def load_screen_pairs(bank_path: str, screen_path: str, min_z: float) -> List[LeakPair]:
    """Every screened move against the player's move whose paired label is signed by |z| >= min_z."""
    bank: Dict[str, Dict[str, Any]] = {}
    with gzip.open(bank_path, "rt") as f:
        for line in f:
            r = json.loads(line)
            bank[r["id"]] = r
    pairs: List[LeakPair] = []
    with gzip.open(screen_path, "rt") as f:
        for line in f:
            c = json.loads(line)
            b = bank[c["id"]]
            played = int(b["played"])
            if str(played) not in c["scores"]:
                continue
            base = np.asarray(c["scores"][str(played)], dtype=np.float64)
            st: Optional[ts.GameState] = None
            for k, sc in c["scores"].items():
                if int(k) == played:
                    continue
                d = np.asarray(sc, dtype=np.float64) - base          # paired: the same worlds, pair by pair
                se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
                if se == 0 or abs(float(d.mean())) < min_z * se:
                    continue
                if st is None:
                    st = state_from_token(b["pos"])
                pairs.append(LeakPair(c["id"], st, decider(st), played, int(k), float(d.mean()), float(b["weight"]),
                                      str(b["decision_type"]), int(st.defcon), int(st.turn), str(b["side"])))
    return pairs


def _after(state: ts.GameState, action: int, seed: int) -> ts.GameState:
    """The position the move leads to, its dice drawn from `seed`."""
    st = state.clone()
    st.rng_state = seed % _UINT64
    ts.Engine.step_flat(st, int(action))
    drain_chance(st, context="leak_critic move")
    return st


def mover_values(model: Any, device: Any, states: Sequence[ts.GameState], movers: Sequence[ts.Player]) -> np.ndarray:
    """v_win in [-1, 1] from each mover's side: the critic read from the side about to decide (the
    only frame it is trained in), negated when that is the opponent; a finished game is its result."""
    feats = model_obs_features(model)
    out = np.zeros(len(states), dtype=np.float64)
    live: List[int] = []
    obs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for i, (st, mover) in enumerate(zip(states, movers)):
        if ts.Engine.is_terminal(st):
            us = float(ts.Engine.get_terminal_utility(st))       # from the US's side
            out[i] = us if int(mover) == int(ts.Player.US) else -us
            continue
        d = decider(st)
        obs.append(np.asarray(ts.extract_observation_features(st, d, feats), dtype=np.float32))
        masks.append(np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8))
        live.append(i)
    for lo in range(0, len(live), 2048):
        idx = live[lo:lo + 2048]
        with torch.no_grad():
            _lg, v, _ = model(torch.from_numpy(np.stack(obs[lo:lo + 2048])).to(device),
                              torch.from_numpy(np.stack(masks[lo:lo + 2048])).to(device))
        vals = v.float().squeeze(-1).cpu().numpy()
        for j, i in enumerate(idx):
            same = int(decider(states[i])) == int(movers[i])
            out[i] = float(vals[j]) if same else -float(vals[j])
    return out


def settle(model: Any, device: Any, states: List[ts.GameState], keys: Sequence[ARKey],
           max_steps: int = 64) -> None:
    """Play each state on greedily with `model` (both sides) until its action round, phasing side,
    phase or turn differs from `keys[i]` -- the move's own action round resolved -- or the game ends.
    In place. A card choice's value lies in the play that follows it, which the decision right after
    the choice has not yet seen."""
    feats = model_obs_features(model)
    for _ in range(max_steps):
        live = [i for i, st in enumerate(states) if not ts.Engine.is_terminal(st) and ar_key(st) == keys[i]]
        if not live:
            return
        obs = np.stack([np.asarray(ts.extract_observation_features(states[i], decider(states[i]), feats),
                                   dtype=np.float32) for i in live])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(states[i]), dtype=np.uint8) for i in live])
        with torch.no_grad():
            lg, _v, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
        legal = torch.from_numpy(masks).to(device) > 0
        acts = lg.float().masked_fill(~legal, float("-inf")).argmax(-1).cpu().numpy()
        for j, i in enumerate(live):
            ts.Engine.step_flat(states[i], int(acts[j]))
            drain_chance(states[i], context="leak_critic settle")


def critic_differences(model: Any, device: Any, pairs: Sequence[LeakPair], dice: int, seed: int = 31337,
                       settled: bool = False, settle_model: Any = None) -> np.ndarray:
    """Per pair: the critic's mean value after `best` minus after `played`, mover's side, over `dice`
    draws of the chance the moves roll (the same draws for both moves); with `settled`, read once the
    move's action round has been played out greedily (`settle`), not at the decision that follows it --
    by `settle_model` when given (a value-only network, P34's evaluator, is read but not played)."""
    rng = random.Random(seed)
    seeds = [[rng.getrandbits(64) for _ in range(dice)] for _ in pairs]
    states: List[ts.GameState] = []
    movers: List[ts.Player] = []
    keys: List[ARKey] = []
    for p, ss in zip(pairs, seeds):
        for move in (p.best, p.played):
            for s in ss:
                states.append(_after(p.state, move, s))
                movers.append(p.mover)
                keys.append(ar_key(p.state))
    if settled:
        settle(settle_model if settle_model is not None else model, device, states, keys)
    v = mover_values(model, device, states, movers).reshape(len(pairs), 2, dice).mean(axis=2)
    return v[:, 0] - v[:, 1]


def continuation_differences(model: Any, device: Any, pairs: Sequence[LeakPair], worlds: int, horizon: int,
                             seed: int = 4711) -> np.ndarray:
    """Per pair: the mean leaf value after `best` minus after `played`, mover's side, under the rollout
    root's own continuation -- worlds sampled from the mover's view, the dice shared by the two moves,
    both sides played greedily by `model` through `horizon` action-round boundaries, the critic at the
    stop (a finished game is its result). This is what R49's value target reads, so it says whether
    that target carries the distinctions the critic is missing. NaN where no world kept both moves."""
    root = RolloutRoot(model, RolloutConfig(k=2, worlds=worlds, horizon=horizon, rule="argmax", seed=seed),
                       device=device)
    out = np.full(len(pairs), np.nan)
    for lo in range(0, len(pairs), 256):
        part = list(pairs[lo:lo + 256])
        lg, masks, _ = root._forward([p.state for p in part])
        lg = np.ascontiguousarray(lg, dtype=np.float32)
        batch = ts.RolloutBatch(horizon=horizon, shared_dice=True, guard=True, features=root.features,
                                seed=root._rng.getrandbits(64))
        pos: Dict[int, int] = {}
        for j, p in enumerate(part):
            pos[j] = batch.positions()
            batch.add(p.state, [p.played, p.best], worlds)
        if batch.rows():
            root._playouts(batch)
        for j, p in enumerate(part):
            d = ts.rollout_decide(batch, pos[j], [p.played, p.best], lg[j], masks[j], "argmax")
            if int(d["worlds"]) > 0:
                out[lo + j] = float(d["q"][1]) - float(d["q"][0])
    return out


def _play_to_turn_end(model: Any, device: Any, states: List[ts.GameState], turns: Sequence[int],
                      max_steps: int = 400) -> None:
    """Play each state on greedily with `model` (both sides) until its turn differs from `turns[i]` or
    the game ends. In place."""
    feats = model_obs_features(model)
    for _ in range(max_steps):
        live = [i for i, st in enumerate(states) if not ts.Engine.is_terminal(st) and int(st.turn) == turns[i]]
        if not live:
            return
        for lo in range(0, len(live), 8192):
            part = live[lo:lo + 8192]
            obs = np.stack([np.asarray(ts.extract_observation_features(states[i], decider(states[i]), feats),
                                       dtype=np.float32) for i in part])
            masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(states[i]), dtype=np.uint8) for i in part])
            with torch.no_grad():
                lg, _v, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
            legal = torch.from_numpy(masks).to(device) > 0
            acts = lg.float().masked_fill(~legal, float("-inf")).argmax(-1).cpu().numpy()
            for j, i in enumerate(part):
                ts.Engine.step_flat(states[i], int(acts[j]))
                drain_chance(states[i], context="leak_critic turn end")
    raise RuntimeError(f"positions still in their turn after {max_steps} plies")


def turn_end_differences(model: Any, device: Any, pairs: Sequence[LeakPair], worlds: int, seed: int = 2718) -> np.ndarray:
    """Per pair: the critic's mean value at the turn's end after `best` minus after `played`, mover's
    side -- worlds dealt from the mover's view (`ts.determinize`), the two moves sharing each world's
    dice, both sides greedy until the turn changes (a finished game is its result). R52's premise: a
    value target that stops at the turn's end, where scoring and the China Card resolve, sees what a
    4-boundary one does not. NaN where no world kept both moves legal."""
    rng = random.Random(seed)
    states: List[ts.GameState] = []
    movers: List[ts.Player] = []
    turns: List[int] = []
    owner: List[int] = []                  # the pair each (best, played) couple belongs to
    for k, p in enumerate(pairs):
        for _ in range(worlds):
            world = ts.determinize(p.state, p.mover, rng.getrandbits(64))
            dice = rng.getrandbits(64)
            legal = np.asarray(ActionEncoder.get_legal_mask(world))
            if not (legal[p.best] and legal[p.played]):
                continue
            for move in (p.best, p.played):
                st = world.clone()
                st.rng_state = dice
                ts.Engine.step_flat(st, int(move))
                drain_chance(st, context="leak_critic turn end move")
                states.append(st)
                movers.append(p.mover)
                turns.append(int(p.state.turn))
            owner.append(k)
    _play_to_turn_end(model, device, states, turns)
    v = mover_values(model, device, states, movers).reshape(-1, 2)
    d = v[:, 0] - v[:, 1]
    out = np.full(len(pairs), np.nan)
    own = np.asarray(owner)
    for k in range(len(pairs)):
        sel = own == k
        if sel.any():
            out[k] = float(d[sel].mean())
    return out


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def summarise(name: str, pairs: Sequence[LeakPair], diff: np.ndarray) -> str:
    """The share of pairs whose sign the critic gets wrong (unweighted and population-weighted, with a
    binomial SE), and the correlation of its difference with the label, overall and by slice."""
    label = np.asarray([p.label for p in pairs])
    w = np.asarray([p.weight for p in pairs])
    def row(sel: np.ndarray, title: str) -> str:
        sel = sel & np.isfinite(diff)                  # a pair no world kept both moves of says nothing
        n = int(sel.sum())
        if n < 5:
            return f"  {title:34s} n={n}"
        agree = np.sign(diff[sel]) == np.sign(label[sel])
        tie = label[sel] == 0
        wrong = (~agree & ~tie).astype(float)
        share = float(wrong.mean())
        se = math.sqrt(max(share * (1 - share), 1e-12) / n)
        wshare = _wmean(wrong, w[sel])
        r = float(np.corrcoef(diff[sel], label[sel])[0, 1]) if n > 2 and np.std(diff[sel]) > 0 else float("nan")
        return (f"  {title:34s} n={n:5d}  critic ranks wrong {100*share:5.1f}% ± {100*se:4.1f}  "
                f"(weighted {100*wshare:5.1f}%)  r(critic, label) {r:+.3f}  "
                f"mean label {100*_wmean(label[sel], w[sel]):+.2f} pts, critic {100*_wmean(diff[sel], w[sel])/2:+.2f} pts")
    out = [f"{name}: {len(pairs)} confirmed pairs (chance would rank 50% wrong)"]
    out.append(row(np.ones(len(pairs), dtype=bool), "all"))
    for dt in sorted({p.decision_type for p in pairs}):
        out.append(row(np.asarray([p.decision_type == dt for p in pairs]), dt))
    card = np.asarray([p.decision_type == "SELECT_CARD" for p in pairs])
    for dc in sorted({p.defcon for p in pairs}):
        out.append(row(card & np.asarray([p.defcon == dc for p in pairs]), f"SELECT_CARD at DEFCON {dc}"))
    for side in ("US", "USSR"):
        out.append(row(np.asarray([p.side == side for p in pairs]), f"as {side}"))
    big = np.asarray([p.label >= 0.05 for p in pairs])
    out.append(row(big, "label >= +5 points"))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--playouts", required=True)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--dice", type=int, default=16, help="chance draws per move, shared by the two moves")
    ap.add_argument("--screen", default=None, help="the screening playouts: the unselected baseline")
    ap.add_argument("--min-z", type=float, default=2.0, help="a screened pair counts when its label is this many SEs from 0")
    ap.add_argument("--horizon", type=int, default=0,
                    help="read each move's mean leaf value under the rollout root's continuation through this many "
                         "action-round boundaries (R49's target), in --worlds worlds, instead of the critic after the move")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--turn-end", action="store_true",
                    help="read each move's critic value at the turn's end after greedy play, in --worlds worlds")
    ap.add_argument("--screen-max", type=int, default=0, help="score at most this many baseline pairs (a random draw)")
    ap.add_argument("--settled", action="store_true",
                    help="read the critic once the move's action round is played out greedily, not right after it")
    ap.add_argument("--settle-with", default=None,
                    help="with --settled: the checkpoint whose greedy play settles the action round, for every "
                         "--checkpoint (P34's evaluators are read, not played); default each checkpoint itself")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(a.bank, a.playouts)
    screen = load_screen_pairs(a.bank, a.screen, a.min_z) if a.screen else []
    if a.screen_max and len(screen) > a.screen_max:
        screen = random.Random(7).sample(screen, a.screen_max)
    how = (f" (greedy to the turn's end, {a.worlds} worlds)" if a.turn_end else
           f" (the rollout continuation: {a.horizon} boundaries, {a.worlds} worlds)" if a.horizon > 0 else
           " (settled: the move's action round played out greedily)" if a.settled else "")
    settler = NeuralAgent.from_checkpoint(a.settle_with, device=str(dev)).model.eval() if a.settle_with else None
    if settler is not None:
        how += f", settled by {a.settle_with}"
    for ck in a.checkpoint:
        model = NeuralAgent.from_checkpoint(ck, device=str(dev)).model.eval()
        diff = (turn_end_differences(model, dev, pairs, a.worlds) if a.turn_end else
                continuation_differences(model, dev, pairs, a.worlds, a.horizon) if a.horizon > 0
                else critic_differences(model, dev, pairs, a.dice, settled=a.settled, settle_model=settler))
        print(summarise(f"{ck} -- the confirmed leaks{how}", pairs, diff))
        print()
        if screen:
            sdiff = (turn_end_differences(model, dev, screen, a.worlds) if a.turn_end else
                     continuation_differences(model, dev, screen, a.worlds, a.horizon) if a.horizon > 0
                     else critic_differences(model, dev, screen, a.dice, settled=a.settled, settle_model=settler))
            print(summarise(f"{ck} -- baseline: screened pairs signed at |z| >= {a.min_z}{how}", screen, sdiff))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
