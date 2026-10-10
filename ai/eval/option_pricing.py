"""Option pricing at small decisions: what the policy would be taught to do differently (P32 B4').

The measurement B4' trains on, run as a probe. Self-play games of one network, greedy on both
sides; at a sample of the decisions with a small option set (2 to `max_options` legal moves,
setup excluded), **every** legal option is played out to the end of the current turn over `worlds`
worlds, and each option is scored by its mean value there:

* **A world** redeals the cards the decider cannot see -- the opponent's hand and the deck,
  consistent with what it can see (`ai.search.dmcts.determinize`) -- once, and every option of
  that world faces the same deal. The **dice are not shared**: each option in each world gets its
  own RNG seed, since what is rolled depends on what is played.
* **The continuation** is the same network, greedy, on both sides, to the end of the turn: the
  first position of the next turn, read by the critic from the decider's side -- or the result,
  if the game ends first (on the last turn, always).
* **The difference** between two options is paired by world, and its standard error comes from
  the spread of the per-world differences.

A decision where the option the policy took is beaten by another by more than `z_min` standard
errors is one where B4' would move the policy; the report lists the largest such differences with,
for each, two single-turn replays from the real game (the policy's move and the better one,
continued to the end of the turn in the real deal) and a workbench link to the position.

What a verdict means: an option is valued by how *this network* plays on from it, so a move whose
payoff lies in a follow-up the network does not find reads worse than it is (`paired_playouts`).

    PYTHONPATH=.:build/release python -m ai.eval.option_pricing --model <checkpoint.pt> \\
        --games 200 --worlds 16 --top 12 --report data/reports/option_leaks.md
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.paired_playouts import apply_move, decider
from ai.search.dmcts import determinize
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features
from tools.lib.game_loop import GameLoop, SettlePolicy
from tools.lib.player_agent import NeuralAgent, load_agent
from tools.lib.self_play import generate_self_play_replay
from web.server.replay import replays_dir

_UINT64 = 1 << 64


@dataclass
class Candidate:
    game: int                 # index into the games
    choice: int               # how many of the game's decisions came before this one
    state: Any                # the position, before the move
    options: List[int]        # the legal flat actions
    probs: List[float]        # the policy's probability for each, at temperature 1
    taken: int                # the greedy move the game played
    values: Optional[np.ndarray] = None    # (options, worlds) turn-end values from the decider's side


@dataclass
class Game:
    seed: int
    choices: List[int] = field(default_factory=list)   # every decision's move, forced steps excluded


def _forward(model: Any, obs: np.ndarray, masks: np.ndarray, device: torch.device
             ) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        lg, v, _ = model(torch.from_numpy(obs).to(device, torch.float32), torch.from_numpy(masks).to(device))
    return lg.float().cpu().numpy(), v.float().reshape(-1).cpu().numpy()


class _Recorder:
    """A greedy source for GameLoop that notes every decision's move and samples candidates."""

    def __init__(self, model: Any, device: torch.device, game: int, record: Game,
                 cands: List[Candidate], sample_p: float, max_options: int, rng: random.Random) -> None:
        self.model, self.device, self.game, self.record = model, device, game, record
        self.cands, self.sample_p, self.max_options, self.rng = cands, sample_p, max_options, rng
        self.feats = model_obs_features(model)

    def choose(self, st: ts.GameState, /) -> Optional[ts.MicroAction]:
        mover = decider(st)
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
        obs = np.asarray(ts.extract_observation_features(st, mover, self.feats), dtype=np.float32)
        lg, _ = _forward(self.model, obs[None], mask[None], self.device)
        legal = np.flatnonzero(mask)
        act = int(legal[np.argmax(lg[0][legal])])
        # Single Influence points (POINT_NODE) are left out, as B4' leaves them: one step of a
        # multi-point placement, priced alone, says little about the placement.
        if (2 <= len(legal) <= self.max_options and st.current_phase != ts.Phase.SETUP
                and int(st.ctx().decision_type) != int(ts.DecisionType.POINT_NODE)
                and self.rng.random() < self.sample_p):
            z = lg[0][legal] - lg[0][legal].max()
            p = np.exp(z) / np.exp(z).sum()
            self.cands.append(Candidate(game=self.game, choice=len(self.record.choices),
                                        state=st.clone(), options=[int(a) for a in legal],
                                        probs=[float(x) for x in p], taken=act))
        self.record.choices.append(act)
        return ts.ActionMask.decode_flat_action(st, act)


def play_games(model: Any, device: torch.device, seeds: Sequence[int], sample_p: float,
               max_options: int, rng: random.Random) -> Tuple[List[Game], List[Candidate]]:
    games: List[Game] = []
    cands: List[Candidate] = []
    for g, seed in enumerate(seeds):
        rec = Game(seed=int(seed))
        st = ts.GameState()
        ts.Engine.init_game(st, int(seed))
        src = _Recorder(model, device, g, rec, cands, sample_p, max_options, rng)
        GameLoop(st, {ts.Player.US: src, ts.Player.USSR: src},
                 settle=SettlePolicy.RECORD_FORCED).run()
        games.append(rec)
    return games, cands


def rebuild(seed: int, choices: Sequence[int]) -> ts.GameState:
    """The position before decision len(choices) of the game, rebuilt from its seed and moves the
    way the replay writer rebuilds it (GameLoop, forced steps settled)."""
    st = ts.GameState()
    ts.Engine.init_game(st, int(seed))

    class _Prefix:
        def __init__(self) -> None:
            self.k = 0
            self.at: Optional[ts.GameState] = None

        def choose(self, s: ts.GameState, /) -> Optional[ts.MicroAction]:
            if self.k == len(choices):
                self.at = s.clone()
                return None                     # forfeit: the loop stops here
            self.k += 1
            return ts.ActionMask.decode_flat_action(s, int(choices[self.k - 1]))

    src = _Prefix()
    GameLoop(st, {ts.Player.US: src, ts.Player.USSR: src}, settle=SettlePolicy.RECORD_FORCED).run()
    if src.at is None:
        raise RuntimeError(f"seed {seed}: the game ended before decision {len(choices)}")
    return src.at


def price(model: Any, device: torch.device, cands: Sequence[Candidate], worlds: int, seed: int,
          chunk: int = 2048, max_plies: int = 1500) -> None:
    """Fill each candidate's `values`: every option over `worlds` worlds, to the end of the turn."""
    feats = model_obs_features(model)
    jobs: List[Tuple[int, int, int, ts.GameState]] = []        # (candidate, option, world, start)
    rng = random.Random(seed)
    for ci, c in enumerate(cands):
        c.values = np.full((len(c.options), worlds), np.nan)
        me = decider(c.state)
        for w in range(worlds):
            base = determinize(c.state.clone(), me, random.Random(rng.getrandbits(62)))
            for oi, a in enumerate(c.options):
                st = base.clone()
                st.rng_state = rng.getrandbits(64) % _UINT64        # this option's own dice
                try:
                    jobs.append((ci, oi, w, apply_move(st, a)))
                except ValueError:
                    pass                    # illegal in this deal: the world is dropped below
    for lo in range(0, len(jobs), chunk):
        part = jobs[lo:lo + chunk]
        n = len(part)
        runner = ts.VectorizedBatchRunner(n, seed + lo)
        if feats:
            runner.set_obs_features([feats] * n, [feats] * n)
        start_turn = np.zeros(n, dtype=np.int64)
        sides = []
        for i, (ci, _oi, _w, st) in enumerate(part):
            runner.set_state(i, st)
            start_turn[i] = int(cands[ci].state.turn)
            sides.append(decider(cands[ci].state))
        runner.refresh_all()
        done = np.zeros(n, dtype=bool)
        value = np.zeros(n)
        for _ply in range(max_plies + 1):
            term = np.asarray(runner.get_terminals(), dtype=bool)
            turns = np.asarray(runner.get_turns(), dtype=np.int64)
            new = ~done & (term | (turns > start_turn))
            if new.any():
                util = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
                idx = np.flatnonzero(new)
                rows = [int(i) for i in idx if not term[i]]
                for i in idx:
                    if term[i]:
                        value[i] = util[i] if sides[i] == ts.Player.US else -util[i]
                if rows:
                    sts = [runner.get_state(i) for i in rows]
                    obs = np.stack([np.asarray(ts.extract_observation_features(s, sides[i], feats),
                                               dtype=np.float32) for s, i in zip(sts, rows)])
                    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8)
                                      for s in sts])
                    _lg, v = _forward(model, obs, masks, device)
                    value[rows] = v
                done |= new
            if done.all():
                break
            obs = np.asarray(runner.get_observations(), dtype=np.float32)
            masks = np.asarray(runner.get_action_masks())
            lg, _ = _forward(model, obs, masks, device)
            lg = np.where(masks > 0, lg, -np.inf)
            acts = lg.argmax(axis=1)
            acts[done] = masks[done].argmax(axis=1)     # finished rows: any legal move, ignored
            res = np.asarray(runner.step_flat_all(acts.tolist(), True))
            if int(((res == 0) & ~done).sum()):
                raise RuntimeError("the engine refused a greedy playout action")
        else:
            raise RuntimeError(f"{int((~done).sum())} playouts still running after {max_plies} plies")
        for (ci, oi, w, _st), v in zip(part, value):
            vals = cands[ci].values
            assert vals is not None
            vals[oi, w] = v
    for c in cands:                          # a world with any option missing is dropped whole
        assert c.values is not None
        c.values = c.values[:, ~np.isnan(c.values).any(axis=0)]


@dataclass
class Verdict:
    cand: Candidate
    best: int                 # index into options
    taken: int                # index into options
    diff: float               # mean(best - taken)
    se: float
    means: np.ndarray
    ses: np.ndarray           # each option's own standard error


def verdicts(cands: Sequence[Candidate]) -> List[Verdict]:
    out = []
    for c in cands:
        v = c.values
        if v is None or v.shape[1] < 4:
            continue
        means = v.mean(axis=1)
        ses = v.std(axis=1, ddof=1) / math.sqrt(v.shape[1])
        t = c.options.index(c.taken)
        b = int(np.argmax(means))
        d = v[b] - v[t]
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        out.append(Verdict(c, b, t, float(d.mean()), se, means, ses))
    return out


def position_link(st: ts.GameState, base_url: str) -> str:
    token = base64.urlsafe_b64encode(zlib.compress(st.to_save_json().encode())).decode().rstrip("=")
    return f"{base_url}/?pos={token}&model=off"


def _describe(st: ts.GameState) -> str:
    ctx = st.ctx()
    side = "US" if decider(st) == ts.Player.US else "USSR"
    dt = ts.DecisionType(int(ctx.decision_type)).name
    card = int(getattr(ctx, "resolving_card", 0) or 0)
    cname = f", card {card} {ts.CardData.get_card_info(card)['name']}" if card else ""
    return (f"turn {int(st.turn)} AR {int(st.action_round)} {str(st.current_phase).replace('Phase.', '')}, "
            f"{side} to decide ({dt}{cname}), DEFCON {int(st.defcon)}, VP {int(st.victory_points):+d}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None, help="the network's name, for the report")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=91_000)
    ap.add_argument("--sample", type=float, default=0.15, help="share of eligible decisions priced")
    ap.add_argument("--max-options", type=int, default=12)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--z-min", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--per-kind", type=int, default=2, help="at most this many examples per (decision type, card)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--replay-prefix", default="leak")
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(args.model, device=str(device))
    if not isinstance(agent, NeuralAgent):
        raise SystemExit(f"{args.model} is not a network")
    model = agent.model
    model.eval()
    rng = random.Random(args.seed)
    seeds = [args.seed + i for i in range(args.games)]
    games, cands = play_games(model, device, seeds, args.sample, args.max_options, rng)
    print(f"{len(games)} games, {len(cands)} decisions to price", flush=True)
    price(model, device, cands, args.worlds, args.seed)
    vs = verdicts(cands)
    clear = [v for v in vs if v.best != v.taken and v.se > 0 and v.diff / v.se >= args.z_min]
    clear.sort(key=lambda v: -v.diff)
    picked: List[Verdict] = []
    kinds: Dict[Tuple[str, int], int] = {}
    for v in clear:
        ctx = v.cand.state.ctx()
        k = (ts.DecisionType(int(ctx.decision_type)).name, int(getattr(ctx, "resolving_card", 0) or 0))
        if kinds.get(k, 0) >= args.per_kind:
            continue
        kinds[k] = kinds.get(k, 0) + 1
        picked.append(v)
        if len(picked) >= args.top:
            break

    label = args.label or os.path.basename(args.model)
    lines = [f"# Option pricing: where B4' would move `{label}`", "",
             f"{len(games)} greedy self-play games (seeds {seeds[0]}-{seeds[-1]}); {len(vs)} decisions with "
             f"2-{args.max_options} options priced (single Influence points excluded) ({args.sample:.0%} of the eligible), every option over "
             f"{args.worlds} worlds (a shared redeal of the hidden cards per world, the dice independent per "
             f"option), the network greedy on both sides to the end of the turn, valued by its critic from the "
             f"decider's side there (or the result). **{len(clear)} decisions ({100 * len(clear) / max(1, len(vs)):.1f}%) "
             f"have an option beating the policy's move by >= {args.z_min} SE**; the largest {len(picked)} below "
             f"(at most {args.per_kind} per decision type and card).", "",
             "Each example: the real game's replay from its start, branched at the decision -- the policy's "
             "move (A) and the better one (B) -- and continued to the end of that turn in the real deal (one "
             f"world of the {args.worlds}; the verdict is the average). Values are the decider's turn-end critic value in "
             "[-1, 1].", ""]
    os.makedirs(replays_dir(), exist_ok=True)
    for r, v in enumerate(picked, 1):
        c = v.cand
        g = games[c.game]
        check = rebuild(g.seed, g.choices[:c.choice])
        if check.to_save_json() != c.state.to_save_json():
            raise RuntimeError(f"example {r}: the rebuilt position differs from the priced one")
        names = [ActionEncoder.get_action_name(c.state, a) for a in c.options]
        files = {}
        for tag, oi in (("A", v.taken), ("B", v.best)):
            fn = f"{args.replay_prefix}_{r:02d}_{tag}.tslog.json"
            generate_self_play_replay(args.model, model_name=label, seed=g.seed, temperature=0.0,
                                      output_path=os.path.join(replays_dir(), fn), device=str(device),
                                      verbose=False, prefix=list(g.choices[:c.choice]) + [c.options[oi]],
                                      stop_after_turn=int(c.state.turn))
            files[tag] = fn
        lines += [f"## {r}. {_describe(c.state)}", "",
                  f"Policy plays **{names[v.taken]}** (p = {c.probs[v.taken]:.3f}); "
                  f"**{names[v.best]}** (p = {c.probs[v.best]:.4f}) is better by "
                  f"**{v.diff:+.3f} ± {v.se:.3f}** (z = {v.diff / v.se:.1f}, {c.values.shape[1] if c.values is not None else 0} worlds).", "",
                  "| option | p | turn-end value |", "|:---|---:|---:|"]
        order = np.argsort(-v.means)
        for oi in order:
            mark = " ← policy" if oi == v.taken else (" ← best" if oi == v.best else "")
            lines.append(f"| {names[oi]}{mark} | {c.probs[oi]:.4f} | {v.means[oi]:+.3f} ± {v.ses[oi]:.3f} |")
        lines += ["", f"Seed {g.seed}, decision {c.choice}. Replays: [A, policy]({args.base_url}/?replay={files['A']}) · "
                  f"[B, better]({args.base_url}/?replay={files['B']}) · [position]({position_link(c.state, args.base_url)})", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.report}: {len(picked)} examples, {len(clear)}/{len(vs)} decisions clear", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
