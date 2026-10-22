"""ai/training/active_search.py: P36's gated active search. Closed, it changes nothing; open, the root plays
the learner's entry decisions as environment (learner = 0) with its improved policy filed as the row's
search target; the gate opens on the first check at the threshold, never joins under gate_only, survives a
resume, and the searched snapshot follows the learner on its two-phase schedule."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.active_search import ActiveSearch, GateCheck, child_env, is_entry_decision
from bindings.ts_env import TsVectorizedEnv

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)
SPEC = "2:2:1:z2:earlier:own:keyed:aligned:box4"


def _trainer(envs: int = 2, buffer: int = 96) -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t = NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=envs, base_seed=11),
                      num_envs=envs, buffer_size=buffer, lr=3e-4, eta=0.1, ref_update_freq=500,
                      cuda_graphs=False, device=dev)
    return t


def _search(t: NashPGTrainer, frac: float = 1.0, gate_only: bool = False) -> ActiveSearch:
    return ActiveSearch(t.active_net, SPEC, device=t.device, frac=frac, gate_every=40_000_000, gate_games=200,
                        gate_threshold=0.55, gate_only=gate_only, refresh_every=10_000_000,
                        refresh_every_late=50_000_000, late_from=400_000_000)


def test_closed_it_plays_nothing() -> None:
    t = _trainer()
    t.active_search = _search(t)
    t.search_ce_coef = 0.1
    t.collect_rollouts()
    assert t.active_search.rows_total == 0
    assert float(t.buffer.has_search.sum()) == 0.0


def test_open_the_root_plays_the_learners_entry_decisions_as_environment() -> None:
    t = _trainer()
    a = _search(t)
    a.state.open_step = 0
    t.active_search = a
    t.search_ce_coef = 0.1
    t.collect_rollouts()
    played = t.buffer.has_search > 0.5
    n = int(played.sum())
    assert n > 0 and n == a.rows_total
    assert not bool((t.buffer.learner[played] > 0.5).any())             # the environment's moves
    pi = t.buffer.search_pi[played]
    assert torch.allclose(pi.sum(dim=-1), torch.ones(n), atol=1e-5)
    legal = t.buffer.masks[played].bool()
    assert not bool((pi * (~legal)).abs().sum() > 0)                     # mass only on legal moves
    # Every learner entry decision was searched at frac 1: no other learner row is a card choice entry.
    t.train_step()                                                         # the CE term trains


def test_entry_decisions_are_card_choices_at_a_round_or_headline() -> None:
    env = TsVectorizedEnv(num_envs=1, base_seed=3)
    env.reset_all()
    rng = np.random.default_rng(3)
    kinds = set()
    for _ in range(400):
        st = env.runner.get_state(0)
        if is_entry_decision(st):
            kinds.add(str(st.current_phase))
            assert int(st.ctx().resolving_card) == 0
        m = np.asarray(env.runner.get_action_masks())[0]
        env.step(np.array([int(rng.choice(np.flatnonzero(m)))]))
    assert len(kinds) == 2                                                  # headline and action round


def _finished(tmp_path: Path, step: int, wins: int, games: int = 200) -> GateCheck:
    out = tmp_path / f"gate_{step}.json"
    out.write_text(json.dumps({"matchup": {"total_games": games, "a_wins": wins, "draws": 0,
                                           "win_rate_a_as_us": wins / games, "win_rate_a_as_ussr": wins / games}}))
    proc = subprocess.Popen(["true"])
    proc.wait()
    return GateCheck(step=step, snapshot="", out=str(out), proc=proc)


def test_the_gate_opens_on_the_first_check_at_the_threshold(tmp_path: Path) -> None:
    t = _trainer(envs=1, buffer=4)
    a = _search(t)
    assert not a.gate_due(39_999_999) and a.gate_due(40_000_000)
    a.pending = _finished(tmp_path, 40_000_000, 100)                       # 50%: closed
    rec = a.poll(41_000_000)
    assert rec is not None and rec["open"] == 0.0 and not a.joined
    a.pending = _finished(tmp_path, 80_000_000, 112)                       # 56%: open
    rec = a.poll(81_000_000)
    assert rec is not None and rec["open"] == 1.0 and a.joined and a.state.open_step == 80_000_000
    assert not a.gate_due(120_000_000)                                      # no checks once open
    # A resume carries it.
    b = _search(t)
    b.restore(a.state_dict(), 90_000_000)
    assert b.joined and len(b.state.checks) == 2


def test_gate_only_records_the_opening_but_never_joins(tmp_path: Path) -> None:
    t = _trainer(envs=1, buffer=4)
    a = _search(t, gate_only=True)
    a.pending = _finished(tmp_path, 40_000_000, 140)
    rec = a.poll(41_000_000)
    assert rec is not None and a.state.open_step == 40_000_000 and not a.joined
    # The later resume without gate_only joins at once.
    b = _search(t)
    b.restore(a.state_dict(), 200_000_000)
    assert b.joined


def test_a_resume_continues_the_gate_cadence() -> None:
    t = _trainer(envs=1, buffer=4)
    a = _search(t)
    a.restore({"open_step": None, "checks": [{"step": 120_000_000.0, "rate": 0.5}]}, 150_000_000)
    assert a.next_gate_step == 160_000_000
    a.restore(None, 0)
    assert a.next_gate_step == 40_000_000


def test_the_searched_snapshot_follows_the_learner_on_its_schedule() -> None:
    t = _trainer(envs=1, buffer=4)
    a = _search(t)
    assert not a.maybe_refresh(t.active_net, 0)                             # closed: never
    a.state.open_step = 0
    assert a.maybe_refresh(t.active_net, 5_000_000)                         # first: at once
    assert not a.maybe_refresh(t.active_net, 9_999_999)
    with torch.no_grad():
        for p in t.active_net.parameters():
            p.add_(1.0)
    assert a.maybe_refresh(t.active_net, 10_000_000)
    for p, q in zip(a.snapshot.parameters(), t.active_net.parameters()):
        assert torch.equal(p, q)
    assert a.maybe_refresh(t.active_net, 400_000_000)
    assert not a.maybe_refresh(t.active_net, 440_000_000)                   # late: every 50M
    assert a.maybe_refresh(t.active_net, 450_000_000)


def test_it_refuses_belief_and_bad_gates() -> None:
    t = _trainer(envs=1, buffer=4)
    with pytest.raises(ValueError):
        ActiveSearch(t.active_net, SPEC + ":belief16", device=t.device, frac=0.5, gate_every=10, gate_games=200,
                     gate_threshold=0.55, gate_only=False, refresh_every=1, refresh_every_late=1, late_from=1)
    with pytest.raises(ValueError):
        ActiveSearch(t.active_net, SPEC, device=t.device, frac=0.0, gate_every=10, gate_games=200,
                     gate_threshold=0.55, gate_only=False, refresh_every=1, refresh_every_late=1, late_from=1)


def test_a_gate_check_process_finds_the_trainers_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check runs tools/tournament.py in a child process; it must import the engine the trainer runs,
    whatever PYTHONPATH the run was launched with -- here none, as a run launched with PYTHONPATH=. gives
    the child nothing that points at the build."""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = child_env()
    code = "import ts_engine, sys; sys.stdout.write(ts_engine.__file__)"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd="/", capture_output=True, text=True, check=True)
    assert Path(out.stdout).resolve() == Path(ts.__file__).resolve()
