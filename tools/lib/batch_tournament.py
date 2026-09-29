# High-Speed Vectorized Tournament Runner & Elo Calculator for Twilight Struggle AI.

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union
import numpy as np
import numpy.typing as npt
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import IllegalActionError
from tools.lib.player_agent import PlayerAgent, NeuralAgent, HeuristicAgent, RandomAgent, load_agent, resolve_device
from ai.game_length import ply as game_ply
from tools.lib.tournament_evaluator import classify_game_ending_reason


def categorize_flat_action_detailed(action_idx: int) -> str:
    """Categorizes a flat action into human-readable semantic categories."""
    if action_idx == ActionEncoder.CONFIRM_DONE_INDEX:
        return "CONFIRM_DONE / PASS"
    elif action_idx < 110:
        card_id = action_idx + 1
        scoring_ids = {1, 2, 3, 37, 38, 39, 81}
        card_type = "Scoring Card" if card_id in scoring_ids else "Normal Card"
        return f"SELECT_CARD ({card_type})"
    elif action_idx < 114:
        modes = ["EVENT", "OPS", "SPACE", "PASS"]
        return f"SELECT_PLAY_MODE ({modes[action_idx - 110]})"
    elif action_idx < 116:
        timings = ["OPS_FIRST", "EVENT_FIRST"]
        return f"CHOOSE_TIMING_BRANCH ({timings[action_idx - 114]})"
    elif action_idx < ActionEncoder.NODE_OFFSET:
        op_modes = ["INFLUENCE", "COUP", "REALIGN"]
        return f"SELECT_OP_MODE ({op_modes[action_idx - 116]})"
    elif action_idx < ActionEncoder.BRANCH_OFFSET:
        return "POINT_NODE (Country)"
    elif action_idx < ActionEncoder.CONFIRM_DONE_INDEX:
        return f"CHOOSE_BRANCH ({action_idx - ActionEncoder.BRANCH_OFFSET})"
    else:
        return "UNKNOWN"



def _assert_width(agent: Any, obs: npt.NDArray[np.float32]) -> None:
    """A model silently misreads an observation of the wrong width; say so instead."""
    want = getattr(agent, "obs_size", None)
    if want is not None and obs.shape[1] != want:
        raise ValueError(
            f"{agent.name} expects an observation of width {want} but the runner produced "
            f"{obs.shape[1]}. Slicing hides this: the network would read the wrong regions and "
            f"play badly rather than fail.")


def _choose_actions(
    agent: PlayerAgent,
    indices: npt.NDArray[np.int64],
    obs: npt.NDArray[np.float32],
    masks: npt.NDArray[Any],
    d_players: npt.NDArray[Any],
    runner: "ts.VectorizedBatchRunner",
    temperature: float,
    greedy: bool,
    dev: torch.device,
    actions: npt.NDArray[np.int32],
) -> None:
    """Fill `actions[indices]` with `agent`'s choices, by the cheapest interface it offers."""
    if isinstance(agent, NeuralAgent):
        sel_obs = obs[indices]
        _assert_width(agent, sel_obs)
        obs_t = torch.from_numpy(sel_obs).float().to(dev)
        mask_t = torch.from_numpy(masks[indices]).to(dev)
        with torch.no_grad():
            act_t, _, _, _, _ = agent.model.sample_action(obs_t, mask_t, temperature=temperature,
                                                          deterministic=greedy)
        actions[indices] = act_t.cpu().numpy()
    elif hasattr(agent, "act_batch"):
        # A network outside torch (OnnxAgent): the same observations and masks, one call.
        sel_obs = obs[indices]
        _assert_width(agent, sel_obs)
        actions[indices] = getattr(agent, "act_batch")(sel_obs, masks[indices], temperature, greedy)
    elif hasattr(agent, "select_actions_batch"):
        # A searcher pays for batching: one call over every game waiting on it,
        # rather than one call per game. Measured at 64 simulations, a batch of
        # 256 roots runs at 103 decisions/s against roughly 0.4/s one at a time.
        sel_states = [runner.get_state(int(idx)) for idx in indices]
        picks = getattr(agent, "select_actions_batch")(sel_states)
        for idx, a in zip(indices, picks):
            actions[idx] = a
    elif hasattr(agent, "select_action"):
        for idx in indices:
            st = runner.get_state(int(idx))
            actions[idx] = agent.select_action(st, ts.Player(int(d_players[idx])),
                                               temperature=temperature)
    else:  # RandomAgent, or anything without a state-based interface
        for idx in indices:
            leg = np.where(masks[idx] > 0)[0]
            actions[idx] = np.random.choice(leg) if len(leg) > 0 else 0


def _choice_stats(
    tot_us: int, single_us: int, tot_ussr: int, single_ussr: int, total_games: int,
    category_counts_us: Dict[str, int], category_counts_ussr: Dict[str, int],
) -> Dict[str, Any]:
    """The `choice_stats` block of a matchup result, from its raw counts."""
    tot_combined = tot_us + tot_ussr
    single_comb = single_us + single_ussr
    return {
        "us_total_micro_actions": tot_us,
        "us_single_choice_micro_actions": single_us,
        "us_single_choice_pct": float((single_us / max(1, tot_us)) * 100.0),
        "ussr_total_micro_actions": tot_ussr,
        "ussr_single_choice_micro_actions": single_ussr,
        "ussr_single_choice_pct": float((single_ussr / max(1, tot_ussr)) * 100.0),
        "overall_total_micro_actions": tot_combined,
        "overall_single_choice_micro_actions": single_comb,
        "overall_single_choice_pct": float((single_comb / max(1, tot_combined)) * 100.0),
        "avg_per_game": {
            "us_total": float(tot_us / max(1, total_games)),
            "us_single": float(single_us / max(1, total_games)),
            "ussr_total": float(tot_ussr / max(1, total_games)),
            "ussr_single": float(single_ussr / max(1, total_games)),
            "combined_total": float(tot_combined / max(1, total_games)),
            "combined_single": float(single_comb / max(1, total_games)),
        },
        "category_counts_us": category_counts_us,
        "category_counts_ussr": category_counts_ussr,
    }


_SUMMED_KEYS = ("total_games", "games_per_side", "a_wins", "b_wins", "draws",
                "a_wins_as_us", "a_losses_as_us", "a_draws_as_us",
                "a_wins_as_ussr", "a_losses_as_ussr", "a_draws_as_ussr")
_MEAN_KEYS = ("avg_steps", "avg_turn", "avg_ply", "avg_vp_margin_a")
_COUNT_DICT_KEYS = ("causes_loss_us", "causes_loss_ussr", "causes_all")


def _add_counts(into: Dict[str, int], more: Dict[str, int]) -> None:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v


def merge_matchup_results(parts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One matchup result from the results of disjoint sets of its game pairs.

    Counts add; the per-game averages are weighted by each part's game count, so the merge equals
    what one call over every pair would have reported. `elapsed_seconds` is the sum over parts --
    compute time, not wall clock, which the caller measures itself.
    """
    if not parts:
        raise ValueError("merge_matchup_results needs at least one part")
    names = {(p["agent_a"], p["agent_b"]) for p in parts}
    if len(names) != 1:
        raise ValueError(f"parts come from different matchups: {sorted(names)}")
    out: Dict[str, Any] = {"agent_a": parts[0]["agent_a"], "agent_b": parts[0]["agent_b"]}
    for k in _SUMMED_KEYS:
        out[k] = int(sum(p[k] for p in parts))
    total = out["total_games"]
    for k in _MEAN_KEYS:
        out[k] = float(sum(p[k] * p["total_games"] for p in parts) / max(1, total))
    for k in _COUNT_DICT_KEYS:
        merged: Dict[str, int] = {}
        for p in parts:
            _add_counts(merged, p[k])
        out[k] = merged
    per_side = max(1, out["games_per_side"])
    out["win_rate_a"] = float(out["a_wins"] / max(1, total))
    out["win_rate_b"] = float(out["b_wins"] / max(1, total))
    out["win_rate_a_as_us"] = float(out["a_wins_as_us"] / per_side)
    out["win_rate_a_as_ussr"] = float(out["a_wins_as_ussr"] / per_side)
    out["elapsed_seconds"] = float(sum(p["elapsed_seconds"] for p in parts))
    stats = [p["choice_stats"] for p in parts if "choice_stats" in p]
    if stats:
        cat_us: Dict[str, int] = {}
        cat_ussr: Dict[str, int] = {}
        for s in stats:
            _add_counts(cat_us, s["category_counts_us"])
            _add_counts(cat_ussr, s["category_counts_ussr"])
        out["choice_stats"] = _choice_stats(
            sum(s["us_total_micro_actions"] for s in stats),
            sum(s["us_single_choice_micro_actions"] for s in stats),
            sum(s["ussr_total_micro_actions"] for s in stats),
            sum(s["ussr_single_choice_micro_actions"] for s in stats),
            total, cat_us, cat_ussr)
    return out

class BatchMatchRunner:
    """Runs 2 * games_per_side games between two PlayerAgents in parallel via C++ VectorizedBatchRunner."""

    @staticmethod
    def play_parallel_matchup(
        agent_a: PlayerAgent,
        agent_b: PlayerAgent,
        games_per_side: int = 1000,
        batch_chunk_size: int = 1000,
        base_seed: int = 10000,
        device: Optional[Union[torch.device, str]] = None,
        max_steps: int = 2500,
        temperature: float = 0.1,
        deterministic: Optional[bool] = None,
        start_states: Optional[Sequence["ts.GameState"]] = None,
        track_choices: bool = False,
        log_games_file: Optional[str] = None,
        auto_advance: bool = True,
        pairs: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Play a matchup batched. Action selection matches NeuralAgent.select_action.

        temperature/deterministic are the same contract as the one-game-at-a-time path in
        TournamentEvaluator.play_matchup: sampling at the given temperature unless it is low
        enough to be indistinguishable from an argmax. This used to be hard-coded to
        deterministic=True here while the sequential path sampled at 0.1, so the two
        disagreed on win rate (0.450 vs 0.610 on one 100-game matchup) and could not be
        swapped for one another.

        `pairs` plays only those game pairs (indices into range(games_per_side)), each with the
        deal it has in the full matchup. That is what lets a matchup be split across processes
        (tools/lib/parallel_tournament.py) and merged with `merge_matchup_results` without
        changing which games are played. The result then counts only those pairs.
        """
        dev = resolve_device(device)
        greedy = (temperature <= 0.05) if deterministic is None else deterministic

        # An agent may pin its own temperature, which then overrides the matchup-wide one. That
        # is what makes a same-weights temperature comparison expressible at all: without it every
        # agent in a tournament shares one setting, so a policy cannot be played against itself at
        # two temperatures. Agents that pin nothing are unaffected.
        def _temp_for(agent):
            t = getattr(agent, "temperature", None)
            if t is None:
                return temperature, greedy
            t = float(t)
            return t, (t <= 0.05) if deterministic is None else deterministic

        temp_a, greedy_a = _temp_for(agent_a)
        temp_b, greedy_b = _temp_for(agent_b)

        # Resume from supplied positions instead of dealing fresh games. Each position is
        # played twice with the sides swapped, which is the same pairing the seeded path
        # uses: both copies resume from one pre-deal state, so they draw the same cards and
        # deal luck cancels between the halves rather than adding variance to the result.
        if start_states is not None:
            games_per_side = len(start_states)
        if games_per_side <= 0:
            # Fail with the reason rather than a ZeroDivisionError from a chunk size of zero
            # further down. Callers that mean "no evaluation" should not call at all.
            raise ValueError(
                f"games_per_side must be positive, got {games_per_side}; "
                f"to skip evaluation, do not call play_parallel_matchup")
        half_per_chunk = min(games_per_side, batch_chunk_size // 2)
        chunk_size = half_per_chunk * 2

        # A pair's deal is fixed by its index in the full matchup and the chunking of the full
        # matchup, never by which subset of pairs this call plays -- the seeds are the ones the
        # unsplit loop has always used.
        def pair_seed(k: int) -> int:
            return base_seed + (k // half_per_chunk) * chunk_size + (k % half_per_chunk)

        def game_index(k: int, second_half: bool) -> int:
            # 1-based, as the unsplit loop numbered it: chunk by chunk, first half then second.
            c = k // half_per_chunk
            full_half = min(games_per_side - c * half_per_chunk, half_per_chunk)
            return c * chunk_size + (k % half_per_chunk) + (full_half if second_half else 0) + 1

        ks = list(range(games_per_side)) if pairs is None else [int(k) for k in pairs]
        bad = [k for k in ks if not 0 <= k < games_per_side]
        if bad or len(set(ks)) != len(ks) or not ks:
            raise ValueError(f"pairs must be distinct indices in range({games_per_side}), got {ks}")
        n_pairs = len(ks)
        total_games = n_pairs * 2

        a_wins = 0
        b_wins = 0
        draws = 0

        a_us_wins = 0
        a_us_losses = 0
        a_us_draws = 0

        a_ussr_wins = 0
        a_ussr_losses = 0
        a_ussr_draws = 0

        all_steps = []
        all_turns = []
        # Length in plies alongside turns. A turn number cannot separate a game abandoned
        # at turn 7 AR1 from one that ran to turn 7 AR7, and it reads 11 for a game that
        # went the distance because finish_end_turn increments before testing its bound.
        all_plies = []
        all_vps = []

        causes_loss_us: Dict[str, int] = {}
        causes_loss_ussr: Dict[str, int] = {}
        causes_all: Dict[str, int] = {}

        total_micro_us = 0
        total_micro_ussr = 0
        single_choice_us = 0
        single_choice_ussr = 0

        category_counts_us: Dict[str, int] = {}
        category_counts_ussr: Dict[str, int] = {}

        if log_games_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_games_file)), exist_ok=True)
            with open(log_games_file, "w", encoding="utf-8") as f_init:
                pass

        t0 = time.time()

        for lo in range(0, n_pairs, half_per_chunk):
            chunk_ks = ks[lo:lo + half_per_chunk]
            cur_half = len(chunk_ks)
            cur_games = cur_half * 2

            runner = ts.VectorizedBatchRunner(cur_games, pair_seed(chunk_ks[0]))
            # Paired deals: env i and env i + cur_half are the same matchup with the sides
            # swapped, so give them the same seed and therefore the same shuffle. Deal luck
            # then cancels between the halves rather than adding variance to the result.
            if start_states is not None:
                for i, k in enumerate(chunk_ks):
                    pos = start_states[k]
                    runner.set_state(i, pos)
                    runner.set_state(i + cur_half, pos)
            else:
                for i, k in enumerate(chunk_ks):
                    paired_seed = pair_seed(k)
                    runner.reset_game(i, paired_seed)
                    runner.reset_game(i + cur_half, paired_seed)
            runner.refresh_all()
            # P23 / E4.1: each side decides in its own agent's action view. Envs < cur_half seat
            # A as USSR and B as US; the rest the reverse. Only neural agents trained in the
            # merged view use it -- bots and searchers act in E4 terms.
            mv_a = bool(getattr(agent_a, "merged_influence", False))
            mv_b = bool(getattr(agent_b, "merged_influence", False))
            if mv_a or mv_b:
                first = np.arange(cur_games) < cur_half
                runner.set_merged_influence([bool(x) for x in np.where(first, mv_b, mv_a)],
                                            [bool(x) for x in np.where(first, mv_a, mv_b)])
            active = np.ones(cur_games, dtype=bool)
            steps = 0

            # Temporary arrays for current chunk
            chunk_utils = np.zeros(cur_games, dtype=np.float32)
            chunk_vps = np.zeros(cur_games, dtype=np.int32)
            chunk_turns = np.zeros(cur_games, dtype=np.int32)
            chunk_plies = np.zeros(cur_games, dtype=np.int32)
            chunk_steps = np.zeros(cur_games, dtype=np.int32)
            chunk_causes = [""] * cur_games

            chunk_ussr_total = np.zeros(cur_games, dtype=np.int32)
            chunk_ussr_single = np.zeros(cur_games, dtype=np.int32)
            chunk_us_total = np.zeros(cur_games, dtype=np.int32)
            chunk_us_single = np.zeros(cur_games, dtype=np.int32)

            while np.any(active) and steps < max_steps:
                obs = runner.get_observations()
                masks = runner.get_action_masks()
                d_players = np.array(runner.get_decision_players())
                terms = np.array(runner.get_terminals())

                newly_finished = active & terms
                if np.any(newly_finished):
                    for idx in np.where(newly_finished)[0]:
                        st = runner.get_state(int(idx))
                        chunk_utils[idx] = float(ts.Engine.get_terminal_utility(st))
                        chunk_vps[idx] = int(st.victory_points)
                        chunk_turns[idx] = int(st.turn)
                        chunk_plies[idx] = game_ply(int(st.turn), int(st.action_round),
                                                    st.phasing_player == ts.Player.US,
                                                    headline_stage=int(st.headline_stage))
                        chunk_steps[idx] = steps
                        chunk_causes[idx] = classify_game_ending_reason(st)
                    active = active & (~terms)

                if not np.any(active):
                    break

                if track_choices:
                    active_indices = np.where(active)[0]
                    if len(active_indices) > 0:
                        active_d_players = d_players[active_indices]
                        valid_counts = np.count_nonzero(masks[active_indices], axis=1)

                        is_ussr = (active_d_players == -1)
                        is_us = (active_d_players == 1)

                        ussr_idxs = active_indices[is_ussr]
                        us_idxs = active_indices[is_us]

                        ussr_vcounts = valid_counts[is_ussr]
                        us_vcounts = valid_counts[is_us]

                        chunk_ussr_total[ussr_idxs] += 1
                        chunk_us_total[us_idxs] += 1

                        ussr_single_mask = (ussr_vcounts == 1)
                        us_single_mask = (us_vcounts == 1)

                        chunk_ussr_single[ussr_idxs[ussr_single_mask]] += 1
                        chunk_us_single[us_idxs[us_single_mask]] += 1

                        for s_idx in ussr_idxs[ussr_single_mask]:
                            act_idx = int(np.argmax(masks[s_idx]))
                            cat = categorize_flat_action_detailed(act_idx)
                            category_counts_ussr[cat] = category_counts_ussr.get(cat, 0) + 1

                        for s_idx in us_idxs[us_single_mask]:
                            act_idx = int(np.argmax(masks[s_idx]))
                            cat = categorize_flat_action_detailed(act_idx)
                            category_counts_us[cat] = category_counts_us.get(cat, 0) + 1

                actions = np.zeros(cur_games, dtype=np.int32)

                # Partition: env 0..cur_half-1 -> (A is USSR, B is US); env cur_half..cur_games-1 -> (B is USSR, A is US)
                is_a_turn = active & (
                    ((np.arange(cur_games) < cur_half) & (d_players == -1)) |
                    ((np.arange(cur_games) >= cur_half) & (d_players == 1))
                )
                is_b_turn = active & (
                    ((np.arange(cur_games) < cur_half) & (d_players == 1)) |
                    ((np.arange(cur_games) >= cur_half) & (d_players == -1))
                )

                if np.any(is_a_turn):
                    _choose_actions(agent_a, np.where(is_a_turn)[0], obs, masks, d_players,
                                    runner, temp_a, greedy_a, dev, actions)
                if np.any(is_b_turn):
                    _choose_actions(agent_b, np.where(is_b_turn)[0], obs, masks, d_players,
                                    runner, temp_b, greedy_b, dev, actions)

                step_results = runner.step_flat_all(actions.tolist(),
                                                    auto_advance=auto_advance)
                # 0 = refused. A refused action here means a game silently did not advance and the
                # matchup's win rate is being computed over a position nobody played.
                if 0 in step_results:
                    bad = step_results.index(0)
                    raise IllegalActionError(
                        f"engine refused flat action {int(actions[bad])} in game {bad} of "
                        f"{len(actions)}; {step_results.count(0)} of the batch refused")
                steps += 1

            # Accumulate Chunk Results
            for idx in range(cur_games):
                a_is_ussr = (idx < cur_half)
                term_util = chunk_utils[idx]
                vp = chunk_vps[idx]
                turn = chunk_turns[idx]
                reason = chunk_causes[idx] or "Early Termination"

                causes_all[reason] = causes_all.get(reason, 0) + 1
                all_steps.append(chunk_steps[idx])
                all_turns.append(turn)
                all_plies.append(chunk_plies[idx])

                vp_for_a = -vp if a_is_ussr else vp
                all_vps.append(vp_for_a)

                a_won = (term_util > 0 and not a_is_ussr) or (term_util < 0 and a_is_ussr)
                b_won = (term_util < 0 and not a_is_ussr) or (term_util > 0 and a_is_ussr)

                if a_won:
                    a_wins += 1
                    if a_is_ussr:
                        a_ussr_wins += 1
                    else:
                        a_us_wins += 1
                elif b_won:
                    b_wins += 1
                    clean_reason = reason
                    if a_is_ussr:
                        a_ussr_losses += 1
                        causes_loss_ussr[clean_reason] = causes_loss_ussr.get(clean_reason, 0) + 1
                    else:
                        a_us_losses += 1
                        causes_loss_us[clean_reason] = causes_loss_us.get(clean_reason, 0) + 1
                else:
                    draws += 1
                    if a_is_ussr:
                        a_ussr_draws += 1
                    else:
                        a_us_draws += 1

            if track_choices:
                total_micro_ussr += int(np.sum(chunk_ussr_total))
                total_micro_us += int(np.sum(chunk_us_total))
                single_choice_ussr += int(np.sum(chunk_ussr_single))
                single_choice_us += int(np.sum(chunk_us_single))

            if log_games_file:
                with open(log_games_file, "a", encoding="utf-8") as f_log:
                    for idx in range(cur_games):
                        a_is_ussr = (idx < cur_half)
                        ussr_agent = agent_a.name if a_is_ussr else agent_b.name
                        us_agent = agent_b.name if a_is_ussr else agent_a.name
                        term_util = chunk_utils[idx]
                        winner = "USSR" if term_util < 0 else ("US" if term_util > 0 else "DRAW")

                        k = chunk_ks[idx % cur_half]
                        g_idx = game_index(k, not a_is_ussr)
                        m_ussr_tot = int(chunk_ussr_total[idx])
                        m_ussr_sgl = int(chunk_ussr_single[idx])
                        m_us_tot = int(chunk_us_total[idx])
                        m_us_sgl = int(chunk_us_single[idx])
                        m_all_tot = m_ussr_tot + m_us_tot
                        m_all_sgl = m_ussr_sgl + m_us_sgl

                        entry = {
                            "game_index": g_idx,
                            "seed": pair_seed(k),
                            "ussr_agent": ussr_agent,
                            "us_agent": us_agent,
                            "winner": winner,
                            "victory_points": int(chunk_vps[idx]),
                            "turn": int(chunk_turns[idx]),
                            "ply": int(chunk_plies[idx]),
                            "steps": int(chunk_steps[idx]),
                            "cause": chunk_causes[idx] or "Early Termination",
                            "ussr_total_micro_actions": m_ussr_tot,
                            "ussr_single_choice_micro_actions": m_ussr_sgl,
                            "ussr_single_choice_pct": round(m_ussr_sgl / max(1, m_ussr_tot) * 100.0, 2),
                            "us_total_micro_actions": m_us_tot,
                            "us_single_choice_micro_actions": m_us_sgl,
                            "us_single_choice_pct": round(m_us_sgl / max(1, m_us_tot) * 100.0, 2),
                            "total_micro_actions": m_all_tot,
                            "total_single_choice_micro_actions": m_all_sgl,
                            "total_single_choice_pct": round(m_all_sgl / max(1, m_all_tot) * 100.0, 2),
                        }
                        f_log.write(json.dumps(entry) + "\n")

        elapsed = time.time() - t0

        res: Dict[str, Any] = {
            "agent_a": agent_a.name,
            "agent_b": agent_b.name,
            "total_games": total_games,
            "games_per_side": n_pairs,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "win_rate_a": float(a_wins / max(1, total_games)),
            "win_rate_b": float(b_wins / max(1, total_games)),
            "a_wins_as_us": a_us_wins,
            "a_losses_as_us": a_us_losses,
            "a_draws_as_us": a_us_draws,
            "win_rate_a_as_us": float(a_us_wins / max(1, n_pairs)),
            "a_wins_as_ussr": a_ussr_wins,
            "a_losses_as_ussr": a_ussr_losses,
            "a_draws_as_ussr": a_ussr_draws,
            "win_rate_a_as_ussr": float(a_ussr_wins / max(1, n_pairs)),
            "avg_steps": float(np.mean(all_steps)) if all_steps else 0.0,
            "avg_turn": float(np.mean(all_turns)) if all_turns else 0.0,
            "avg_ply": float(np.mean(all_plies)) if all_plies else 0.0,
            "avg_vp_margin_a": float(np.mean(all_vps)) if all_vps else 0.0,
            "causes_loss_us": causes_loss_us,
            "causes_loss_ussr": causes_loss_ussr,
            "causes_all": causes_all,
            "elapsed_seconds": elapsed,
        }

        if track_choices:
            res["choice_stats"] = _choice_stats(
                total_micro_us, single_choice_us, total_micro_ussr, single_choice_ussr,
                total_games, category_counts_us, category_counts_ussr)

        return res


def compute_mle_elo(
    model_names: List[str],
    win_matrix: np.ndarray,
    total_matrix: np.ndarray,
    anchor_model: str = "HeuristicBot",
    anchor_elo: float = 1500.0,
    iterations: int = 1000,
    lr: float = 0.01,
) -> Dict[str, float]:
    """Computes Bradley-Terry Maximum Likelihood Elo ratings anchored to a reference model."""
    M = len(model_names)
    ratings = np.ones(M, dtype=np.float64) * 1500.0

    for _ in range(iterations):
        grad = np.zeros(M, dtype=np.float64)
        for i in range(M):
            for j in range(M):
                if i == j or total_matrix[i, j] == 0:
                    continue
                w_ij = win_matrix[i, j]
                w_ji = win_matrix[j, i]
                d_ij = total_matrix[i, j] - w_ij - w_ji
                # Win = 1.0, Draw = 0.5
                score_i = w_ij + 0.5 * d_ij
                n_ij = total_matrix[i, j]

                gamma_i = np.exp(ratings[i] / 400.0 * np.log(10))
                gamma_j = np.exp(ratings[j] / 400.0 * np.log(10))
                e_ij = gamma_i / (gamma_i + gamma_j)

                grad[i] += (score_i - n_ij * e_ij)

        # Update unanchored ratings
        ratings += lr * grad

        # Re-anchor to baseline
        if anchor_model in model_names:
            anchor_idx = model_names.index(anchor_model)
            ratings += (anchor_elo - ratings[anchor_idx])

    return {model_names[i]: float(ratings[i]) for i in range(M)}
