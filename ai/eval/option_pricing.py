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
    target: int = 0           # the target card this decision was recorded for (0: sampled)
    focus: int = -1           # the option of interest for a target decision: its event / its headline


@dataclass
class Game:
    seed: int
    choices: List[int] = field(default_factory=list)   # every decision's move, forced steps excluded


def _forward(model: Any, obs: np.ndarray, masks: np.ndarray, device: torch.device
             ) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        lg, v, _ = model(torch.from_numpy(obs).to(device, torch.float32), torch.from_numpy(masks).to(device))
    return lg.float().cpu().numpy(), v.float().reshape(-1).cpu().numpy()


def priceable(st: ts.GameState, cap: int) -> bool:
    """A decision small enough to price: 2..cap options, past setup, and not one point of a
    multi-point placement -- a coup's target (a POINT_NODE in the COUP op mode) is one choice and is
    priced."""
    if st.current_phase == ts.Phase.SETUP:
        return False
    ctx = st.ctx()
    if int(ctx.decision_type) == int(ts.DecisionType.POINT_NODE) and ctx.op_mode != ts.OpMode.COUP:
        return False
    n = int(np.asarray(ActionEncoder.get_legal_mask(st)).sum())
    return 2 <= n <= cap


def _focus(st: ts.GameState, legal: Sequence[int], targets: Sequence[int],
           headline_targets: Sequence[int]) -> Tuple[int, int]:
    """(target card, index of the option of interest) for a target decision, else (0, -1): the
    play-mode decision of a target card (interest: its event), or a headline choice offering a
    headline target (interest: headlining it)."""
    ctx = st.ctx()
    dt = int(ctx.decision_type)
    if dt == int(ts.DecisionType.SELECT_PLAY_MODE) and int(ctx.pending_op_card) in targets:
        for i, a in enumerate(legal):
            if ActionEncoder.get_action_name(st, int(a)).startswith("Resolution: EVENT"):
                return int(ctx.pending_op_card), i
    if dt == int(ts.DecisionType.SELECT_CARD) and st.current_phase == ts.Phase.HEADLINE and headline_targets:
        for i, a in enumerate(legal):
            cid = int(ts.ActionMask.decode_flat_action(st, int(a)).primary_id)
            if cid in headline_targets:
                return cid, i
    return 0, -1


class _Recorder:
    """A greedy source for GameLoop that notes every decision's move and samples candidates."""

    def __init__(self, model: Any, device: torch.device, game: int, record: Game,
                 cands: List[Candidate], sample_p: float, max_options: int, rng: random.Random,
                 targets: Sequence[int] = (), headline_targets: Sequence[int] = ()) -> None:
        self.model, self.device, self.game, self.record = model, device, game, record
        self.cands, self.sample_p, self.max_options, self.rng = cands, sample_p, max_options, rng
        self.targets, self.headline_targets = list(targets), list(headline_targets)
        self.feats = model_obs_features(model)

    def choose(self, st: ts.GameState, /) -> Optional[ts.MicroAction]:
        mover = decider(st)
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
        obs = np.asarray(ts.extract_observation_features(st, mover, self.feats), dtype=np.float32)
        lg, _ = _forward(self.model, obs[None], mask[None], self.device)
        legal = np.flatnonzero(mask)
        act = int(legal[np.argmax(lg[0][legal])])
        # Single Influence points are left out, as B4' leaves them: one step of a multi-point
        # placement, priced alone, says little about the placement (`priceable`).
        target, focus = (_focus(st, [int(a) for a in legal], self.targets, self.headline_targets)
                         if (self.targets or self.headline_targets) else (0, -1))
        if priceable(st, self.max_options) and (target or self.rng.random() < self.sample_p):
            z = lg[0][legal] - lg[0][legal].max()
            p = np.exp(z) / np.exp(z).sum()
            self.cands.append(Candidate(game=self.game, choice=len(self.record.choices),
                                        state=st.clone(), options=[int(a) for a in legal],
                                        probs=[float(x) for x in p], taken=act,
                                        target=target, focus=focus))
        self.record.choices.append(act)
        return ts.ActionMask.decode_flat_action(st, act)


def play_games(model: Any, device: torch.device, seeds: Sequence[int], sample_p: float,
               max_options: int, rng: random.Random, targets: Sequence[int] = (),
               headline_targets: Sequence[int] = ()) -> Tuple[List[Game], List[Candidate]]:
    games: List[Game] = []
    cands: List[Candidate] = []
    for g, seed in enumerate(seeds):
        rec = Game(seed=int(seed))
        st = ts.GameState()
        ts.Engine.init_game(st, int(seed))
        src = _Recorder(model, device, g, rec, cands, sample_p, max_options, rng, targets, headline_targets)
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


@dataclass
class Nested:
    parent: int               # index of the candidate whose branch it was met in
    option: int               # that branch's option index
    cand: Candidate


def price(model: Any, device: torch.device, cands: Sequence[Candidate], worlds: int, seed: int,
          chunk: int = 2048, max_plies: int = 1500, lookahead: int = 0, look_worlds: int = 4,
          look_cap: int = 12, nested: Optional[List[Nested]] = None, nested_p: float = 0.0) -> None:
    """Fill each candidate's `values`: every option over `worlds` worlds, to the end of the turn.

    `lookahead` > 0 (depth 2): in each branch, the decider's next `lookahead` small decisions of the
    same turn are chosen by pricing them in turn (depth 1, `look_worlds` worlds) instead of greedily;
    the opponent stays greedy. `nested` collects, with probability `nested_p`, the decider's small
    decisions met in the branches of options the policy did not take -- positions its own play does
    not reach -- for pricing on their own."""
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
        look_left = np.full(n, lookahead, dtype=np.int64)
        branch_rng = random.Random(seed + 7 * lo + 1)
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
            if lookahead > 0 or nested is not None:
                dps = np.asarray(runner.get_decision_players(), dtype=np.int64)
                subs: List[Candidate] = []
                owners: List[int] = []
                for i in range(n):
                    if done[i] or turns[i] != start_turn[i] or dps[i] != int(sides[i]):
                        continue
                    ci, oi = part[i][0], part[i][1]
                    off_policy = cands[ci].options[oi] != cands[ci].taken
                    want_nest = nested is not None and off_policy and branch_rng.random() < nested_p
                    if look_left[i] <= 0 and not want_nest:
                        continue
                    st = runner.get_state(i)
                    if not priceable(st, look_cap):
                        continue
                    legal = [int(a) for a in np.flatnonzero(masks[i])]
                    if want_nest and nested is not None:
                        z = lg[i][legal] - lg[i][legal].max()
                        pz = np.exp(z) / np.exp(z).sum()
                        nested.append(Nested(ci, oi, Candidate(game=-1, choice=-1, state=st.clone(), options=legal,
                                                               probs=[float(x) for x in pz], taken=int(acts[i]))))
                    if look_left[i] > 0:
                        subs.append(Candidate(game=-1, choice=-1, state=st, options=legal, probs=[],
                                              taken=int(acts[i])))
                        owners.append(i)
                if subs:
                    price(model, device, subs, look_worlds, branch_rng.getrandbits(30), chunk, max_plies)
                    for i, c in zip(owners, subs):
                        if c.values is not None and c.values.shape[1] > 0:
                            acts[i] = c.options[int(np.argmax(c.values.mean(axis=1)))]
                        look_left[i] -= 1
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


def continue_record(model: Any, device: torch.device, seed: int, prefix: Sequence[int], side: ts.Player,
                    turn: int, lookahead: int, look_worlds: int, cap: int, rseed: int) -> List[int]:
    """The decisions after `prefix` to the end of `turn` in the real game: the opponent greedy, the
    decider's next `lookahead` small decisions priced (depth 1) as in the depth-2 mode -- so a replay
    of prefix + these shows the follow-up the verdict assumed, in the real deal."""
    st = ts.GameState()
    ts.Engine.init_game(st, int(seed))
    feats = model_obs_features(model)
    out: List[int] = []
    rng = random.Random(rseed)

    class _Src:
        def __init__(self) -> None:
            self.k = 0
            self.left = lookahead

        def choose(self, s: ts.GameState, /) -> Optional[ts.MicroAction]:
            if self.k < len(prefix):
                self.k += 1
                return ts.ActionMask.decode_flat_action(s, int(prefix[self.k - 1]))
            mover = decider(s)
            mask = np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8)
            obs = np.asarray(ts.extract_observation_features(s, mover, feats), dtype=np.float32)
            lg, _ = _forward(model, obs[None], mask[None], device)
            legal = [int(a) for a in np.flatnonzero(mask)]
            act = legal[int(np.argmax(lg[0][legal]))]
            if mover == side and int(s.turn) == turn and self.left > 0 and priceable(s, cap):
                c = Candidate(game=-1, choice=-1, state=s.clone(), options=legal, probs=[], taken=act)
                price(model, device, [c], look_worlds, rng.getrandbits(30))
                if c.values is not None and c.values.shape[1] > 0:
                    act = c.options[int(np.argmax(c.values.mean(axis=1)))]
                self.left -= 1
            out.append(act)
            return ts.ActionMask.decode_flat_action(s, act)

    src = _Src()
    GameLoop(st, {ts.Player.US: src, ts.Player.USSR: src}, settle=SettlePolicy.RECORD_FORCED,
             stop=lambda s: src.k >= len(prefix) and int(s.turn) > turn).run()
    return out


def _option_table(c: Candidate, v: Verdict, names: List[str]) -> List[str]:
    lines = ["| option | p | turn-end value |", "|:---|---:|---:|"]
    for oi in np.argsort(-v.means):
        mark = " ← policy" if oi == v.taken else (" ← best" if oi == v.best else "")
        lines.append(f"| {names[oi]}{mark} | {c.probs[oi]:.4f} | {v.means[oi]:+.3f} ± {v.ses[oi]:.3f} |")
    return lines


def _copy(c: Candidate) -> Candidate:
    return Candidate(game=c.game, choice=c.choice, state=c.state, options=list(c.options), probs=list(c.probs),
                     taken=c.taken, target=c.target, focus=c.focus)


def _target_summary(name: str, vs: Sequence[Verdict], z_min: float) -> str:
    """One row: how often the option of interest is the policy's move, the priced best, clearly
    better than the policy's move, and by how much on average where the policy did something else."""
    n = len(vs)
    if not n:
        return f"| {name} | 0 | | | | | |"
    takes = sum(1 for v in vs if v.taken == v.cand.focus)
    best = sum(1 for v in vs if v.best == v.cand.focus)
    clear, deltas = 0, []
    for v in vs:
        vals = v.cand.values
        if v.taken == v.cand.focus or vals is None:
            continue
        d = vals[v.cand.focus] - vals[v.taken]
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.inf
        deltas.append(float(d.mean()))
        if se > 0 and d.mean() / se >= z_min:
            clear += 1
    p_focus = float(np.mean([v.cand.probs[v.cand.focus] for v in vs]))
    dm = float(np.mean(deltas)) if deltas else math.nan
    dse = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else math.nan
    return (f"| {name} | {n} | {p_focus:.3f} | {100 * takes / n:.0f}% | {100 * best / n:.0f}% | "
            f"{100 * clear / n:.0f}% | {dm:+.3f} ± {dse:.3f} |")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None, help="the network's name, for the report")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=91_000)
    ap.add_argument("--sample", type=float, default=0.15, help="share of eligible decisions priced (0: targets only)")
    ap.add_argument("--max-options", type=int, default=12)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--z-min", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--per-kind", type=int, default=2, help="at most this many examples per (decision type, card)")
    ap.add_argument("--targets", type=int, nargs="*", default=[],
                    help="cards whose every play-mode decision is priced (interest: the event)")
    ap.add_argument("--headline-targets", type=int, nargs="*", default=[],
                    help="cards whose every headline choice offering them is priced (interest: headlining it)")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="depth 2 for the target decisions: the decider's next N small decisions of the turn priced")
    ap.add_argument("--look-worlds", type=int, default=4)
    ap.add_argument("--nested-p", type=float, default=0.0,
                    help="share of the decider's small decisions met in the target decisions' untaken branches "
                         "priced on their own (depth 1)")
    ap.add_argument("--examples", type=int, default=3, help="replayed examples per target card")
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
    games, all_cands = play_games(model, device, seeds, args.sample, args.max_options, rng,
                                  args.targets, args.headline_targets)
    tcands = [c for c in all_cands if c.target]
    cands = [c for c in all_cands if not c.target]
    print(f"{len(games)} games, {len(cands)} sampled decisions, {len(tcands)} target decisions", flush=True)
    label = args.label or os.path.basename(args.model)
    os.makedirs(replays_dir(), exist_ok=True)
    lines = [f"# Option pricing: where B4' would move `{label}`", "",
             f"{len(games)} greedy self-play games (seeds {seeds[0]}-{seeds[-1]}). Every option of a priced "
             f"decision is played over {args.worlds} worlds (a shared redeal of the hidden cards per world, the dice "
             f"independent per option), the network greedy on both sides to the end of the turn, valued by its "
             f"critic from the decider's side there (or the result), in [-1, 1].", ""]

    def replay_pair(r: str, c: Candidate, a_opt: int, b_opt: int, b_tail: Sequence[int] = ()) -> Dict[str, str]:
        g = games[c.game]
        if rebuild(g.seed, g.choices[:c.choice]).to_save_json() != c.state.to_save_json():
            raise RuntimeError(f"example {r}: the rebuilt position differs from the priced one")
        files = {}
        for tag, opt, tail in (("A", a_opt, ()), ("B", b_opt, b_tail)):
            fn = f"{args.replay_prefix}_{r}_{tag}.tslog.json"
            generate_self_play_replay(args.model, model_name=label, seed=g.seed, temperature=0.0,
                                      output_path=os.path.join(replays_dir(), fn), device=str(device),
                                      verbose=False, prefix=list(g.choices[:c.choice]) + [opt] + list(tail),
                                      stop_after_turn=int(c.state.turn))
            files[tag] = fn
        return files

    if cands:
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
        lines += ["## Sampled decisions", "",
                  f"{len(vs)} decisions with 2-{args.max_options} options (single Influence points excluded; "
                  f"{args.sample:.0%} of the eligible). **{len(clear)} ({100 * len(clear) / max(1, len(vs)):.1f}%) "
                  f"have an option beating the policy's move by >= {args.z_min} SE**; the largest {len(picked)} "
                  f"(at most {args.per_kind} per decision type and card). Each with the real game's replay "
                  f"branched at the decision -- the policy's move (A), the better one (B) -- to the end of the turn "
                  f"in the real deal (one world; the verdict is the average).", ""]
        for r, v in enumerate(picked, 1):
            c = v.cand
            names = [ActionEncoder.get_action_name(c.state, a) for a in c.options]
            files = replay_pair(f"{r:02d}", c, c.options[v.taken], c.options[v.best])
            lines += [f"### {r}. {_describe(c.state)}", "",
                      f"Policy plays **{names[v.taken]}** (p = {c.probs[v.taken]:.3f}); "
                      f"**{names[v.best]}** (p = {c.probs[v.best]:.4f}) is better by "
                      f"**{v.diff:+.3f} ± {v.se:.3f}** (z = {v.diff / v.se:.1f}).", ""]
            lines += _option_table(c, v, names)
            lines += ["", f"Seed {games[c.game].seed}, decision {c.choice}. Replays: "
                      f"[A, policy]({args.base_url}/?replay={files['A']}) · [B, better]({args.base_url}/?replay={files['B']}) · "
                      f"[position]({position_link(c.state, args.base_url)})", ""]
        print(f"sampled: {len(clear)}/{len(vs)} clear", flush=True)

    if tcands:
        d1 = [_copy(c) for c in tcands]
        nested: List[Nested] = []
        price(model, device, d1, args.worlds, args.seed + 1,
              nested=nested if args.nested_p > 0 else None, nested_p=args.nested_p)
        v1 = {id(v.cand): v for v in verdicts(d1)}
        d2: List[Candidate] = []
        v2: Dict[int, Verdict] = {}
        if args.lookahead > 0:
            d2 = [_copy(c) for c in tcands]
            price(model, device, d2, args.worlds, args.seed + 2, lookahead=args.lookahead,
                  look_worlds=args.look_worlds, look_cap=args.max_options)
            v2 = {id(v.cand): v for v in verdicts(d2)}
        cards = sorted({c.target for c in tcands})
        cname = {cid: str(ts.CardData.get_card_info(cid)["name"]) for cid in cards}
        lines += ["## Target decisions", "",
                  f"Every play-mode decision of {', '.join(cname[c] for c in cards if c in args.targets) or '-'} "
                  f"(the option of interest: its event) and every headline choice offering "
                  f"{', '.join(cname[c] for c in cards if c in args.headline_targets) or '-'} (interest: headlining it). "
                  f"**Depth 1** prices each option with greedy play after it; **depth 2** also chooses the decider's "
                  f"next {args.lookahead} small decisions of the turn by pricing them ({args.look_worlds} worlds), "
                  f"coup targets included -- the follow-up a pair needs.", "",
                  "| card, depth | decisions | policy p(interest) | policy plays it | priced best | clearly better than the policy's move | mean gain where the policy plays otherwise |",
                  "|:---|---:|---:|---:|---:|---:|---:|"]
        for cid in cards:
            for tag, vmap, cs in (("depth 1", v1, d1), ("depth 2", v2, d2)):
                if not cs:
                    continue
                vs_c = [vmap[id(c)] for c in cs if c.target == cid and id(c) in vmap]
                lines.append(_target_summary(f"{cname[cid]}, {tag}", vs_c, args.z_min))
        lines.append("")
        deep = v2 if v2 else v1
        deep_c = d2 if d2 else d1
        for cid in cards:
            ex = [deep[id(c)] for c in deep_c if c.target == cid and id(c) in deep]
            ex = [v for v in ex if v.best == v.cand.focus != v.taken and v.se > 0 and v.diff / v.se >= args.z_min]
            ex.sort(key=lambda v: -v.diff)
            for r, v in enumerate(ex[:args.examples], 1):
                c = v.cand
                names = [ActionEncoder.get_action_name(c.state, a) for a in c.options]
                g = games[c.game]
                tail: List[int] = []
                if args.lookahead > 0:
                    tail = continue_record(model, device, g.seed, list(g.choices[:c.choice]) + [c.options[v.best]],
                                           decider(c.state), int(c.state.turn), args.lookahead, args.look_worlds,
                                           args.max_options, args.seed + r)
                files = replay_pair(f"c{cid}_{r}", c, c.options[v.taken], c.options[v.best], tail)
                lines += [f"### {cname[cid]} {r}. {_describe(c.state)}", "",
                          f"Policy plays **{names[v.taken]}** (p = {c.probs[v.taken]:.3f}); "
                          f"**{names[v.best]}** (p = {c.probs[v.best]:.4f}) is better by "
                          f"**{v.diff:+.3f} ± {v.se:.3f}** ({'depth 2' if v2 else 'depth 1'}).", ""]
                lines += _option_table(c, v, names)
                lines += ["", f"Seed {g.seed}, decision {c.choice}. Replays: [A, policy]({args.base_url}/?replay={files['A']}) · "
                          f"[B, better{', with the priced follow-up' if tail else ''}]({args.base_url}/?replay={files['B']}) · "
                          f"[position]({position_link(c.state, args.base_url)})", ""]
        if nested:
            nc = [n.cand for n in nested]
            price(model, device, nc, args.worlds, args.seed + 3)
            nv = {id(v.cand): v for v in verdicts(nc)}
            lines += ["## Nested decisions", "",
                      f"{len(nested)} of the decider's small decisions met in the branches of target-decision options "
                      f"the policy did not take ({args.nested_p:.0%} sampled) -- positions its own play does not reach, "
                      f"where B4' with nested queueing would train -- priced on their own (depth 1).", "",
                      "| target card | nested decisions | clearly better option than the policy's | largest gain |",
                      "|:---|---:|---:|---:|"]
            per: Dict[int, List[Verdict]] = {}
            for n_ in nested:
                if id(n_.cand) in nv:
                    per.setdefault(tcands[n_.parent].target, []).append(nv[id(n_.cand)])
            for cid, lst in sorted(per.items()):
                cl = [v for v in lst if v.best != v.taken and v.se > 0 and v.diff / v.se >= args.z_min]
                lines.append(f"| {cname[cid]} | {len(lst)} | {len(cl)} ({100 * len(cl) / max(1, len(lst)):.0f}%) | "
                             f"{max((v.diff for v in cl), default=0.0):+.3f} |")
            lines.append("")
            for cid, lst in sorted(per.items()):
                cl = sorted([v for v in lst if v.best != v.taken and v.se > 0 and v.diff / v.se >= args.z_min],
                            key=lambda v: -v.diff)[:args.examples + 2]
                for v in cl:
                    c = v.cand
                    names = [ActionEncoder.get_action_name(c.state, a) for a in c.options]
                    lines += [f"### After {cname[cid]}: {_describe(c.state)}", "",
                              f"Policy plays **{names[v.taken]}**; **{names[v.best]}** better by "
                              f"**{v.diff:+.3f} ± {v.se:.3f}**. [position]({position_link(c.state, args.base_url)})", ""]
        print(f"targets: {len(tcands)} priced; nested {len(nested)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
