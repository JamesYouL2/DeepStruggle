"""--seed-env drives every game's seed (owner, 2026-10-27): a training env with a game seed draws each new game's seed
from its own generator, so the deals follow that seed alone -- not the global numpy stream, which --seed-init seeds and
anything may consume -- and the generator's state is saved with the resume state so a continuation draws new games.
Without a game seed (every evaluation tool) the env draws from the global numpy RNG as before."""

from types import SimpleNamespace

import numpy as np

from ai.training.generic_trainer import _env_game_rng_state
from bindings.ts_env import TsVectorizedEnv


def _draws(env: TsVectorizedEnv, n: int = 5) -> list[int]:
    return [env.next_game_seed() for _ in range(n)]


def test_the_game_seed_alone_decides_the_games() -> None:
    np.random.seed(1)
    a = _draws(TsVectorizedEnv(num_envs=2, base_seed=44, game_seed=44))
    np.random.seed(2)                                   # another --seed-init, or another consumer of the global stream
    b = _draws(TsVectorizedEnv(num_envs=2, base_seed=44, game_seed=44))
    c = _draws(TsVectorizedEnv(num_envs=2, base_seed=45, game_seed=45))
    assert a == b and a != c


def test_a_saved_generator_continues_its_stream() -> None:
    env = TsVectorizedEnv(num_envs=2, base_seed=44, game_seed=44)
    _draws(env)
    state = _env_game_rng_state(SimpleNamespace(env=env))
    assert state is not None
    after = _draws(env)
    resumed = TsVectorizedEnv(num_envs=2, base_seed=44, game_seed=44)
    resumed.set_game_rng_state(state)
    assert _draws(resumed) == after


def test_a_resume_without_a_saved_generator_does_not_replay_the_start() -> None:
    start = _draws(TsVectorizedEnv(num_envs=2, base_seed=44, game_seed=44))
    env = TsVectorizedEnv(num_envs=2, base_seed=44, game_seed=44)
    env.reseed_games([44, 13_200_000_000])
    other = TsVectorizedEnv(num_envs=2, base_seed=44, game_seed=44)
    other.reseed_games([44, 13_600_000_000])
    assert _draws(env) != start and _draws(env) != _draws(other)


def test_without_a_game_seed_the_global_stream_is_used_as_before() -> None:
    env = TsVectorizedEnv(num_envs=2, base_seed=44)
    assert env.game_rng_state() is None and _env_game_rng_state(SimpleNamespace(env=env)) is None
    np.random.seed(7)
    a = _draws(env)
    np.random.seed(7)
    assert a == [int(np.random.randint(1, 1_000_000_000)) for _ in range(5)]
