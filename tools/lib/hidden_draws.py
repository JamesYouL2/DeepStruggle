"""The outcomes of a step that draws from the engine's PRNG without asking: enumerated, not sampled.

A die is a decision node (`ROLL_DIE`, a forced face), so a search averages it by stepping the six
faces. A card draw is not: Five Year Plan, Grain Sales, Terrorism, Our Man in Tehran and the
second die of a two-dice event take their randomness from `state.rng_state` inside the step, so a
search that steps the move once gets one draw and treats it as the only one.

The engine's PRNG is SplitMix64 (`engine/include/ts/prng.hpp`): the state advances by a constant
and the output is a bijective mix of it, and every draw is `next_u64 % pool`. Both are invertible
from Python, so a step can be made to draw any output it likes: `state_for_output(u)` is the
`rng_state` under which the engine's next draw is `u`, and `calls_between` counts the draws a step
made. `step_outcomes` uses them to enumerate one draw exactly: the outcome of a draw is a function
of `u % pool` with `pool` unknown, so it steps the move under `u = 0, 1, 2, ...` until the outcome
of `u = 0` recurs, which is `pool`, and checks the next two periods repeat. The outcomes are the
`pool` states, equally likely. A step that draws more than once is not enumerated (`ManyDraws`):
its later draws depend on the first through the generator, and a caller reports it rather than
guess.
"""

from __future__ import annotations

from typing import Callable, List

import ts_engine as ts

MASK = (1 << 64) - 1
#: SplitMix64's increment: the engine's `rng_state` moves by this on every draw.
GAMMA = 0x9E3779B97F4A7C15
_C1 = 0xBF58476D1CE4E5B9
_C2 = 0x94D049BB133111EB
_C1_INV = pow(_C1, -1, 1 << 64)
_C2_INV = pow(_C2, -1, 1 << 64)
_GAMMA_INV = pow(GAMMA, -1, 1 << 64)
#: A step that advances the generator this far or further is not a draw but something else.
MAX_DRAWS = 64
#: Pools are at most the 110 cards; the scan for a draw's period gives up past this.
MAX_POOL = 111


class ManyDraws(Exception):
    """A step drew more than once from the generator, so its outcomes cannot be enumerated."""

    def __init__(self, draws: int) -> None:
        super().__init__(f"a step with {draws} hidden draws")
        self.draws = draws


def mix(z: int) -> int:
    """SplitMix64's output for the advanced state `z`."""
    z = ((z ^ (z >> 30)) * _C1) & MASK
    z = ((z ^ (z >> 27)) * _C2) & MASK
    return z ^ (z >> 31)


def _unxorshift(x: int, shift: int) -> int:
    r = x
    for _ in range(64 // shift + 1):
        r = x ^ (r >> shift)
    return r


def unmix(out: int) -> int:
    """The advanced state whose output is `out` (the inverse of `mix`)."""
    z = _unxorshift(out, 31)
    z = (z * _C2_INV) & MASK
    z = _unxorshift(z, 27)
    z = (z * _C1_INV) & MASK
    return _unxorshift(z, 30)


def next_u64(state: int) -> int:
    """What the engine draws next from `rng_state == state`."""
    return mix((state + GAMMA) & MASK)


def state_for_output(u: int) -> int:
    """The `rng_state` under which the engine's next draw is `u`."""
    return (unmix(u & MASK) - GAMMA) & MASK


def calls_between(before: int, after: int) -> int:
    """How many times the generator was advanced from `before` to `after`; `MAX_DRAWS` where
    `after` is not a small number of steps ahead."""
    k = ((after - before) * _GAMMA_INV) & MASK
    return k if k < MAX_DRAWS else MAX_DRAWS


def _key(st: ts.GameState) -> str:
    return st.to_save_json()


def step_outcomes(st: ts.GameState, step: Callable[[ts.GameState], None]) -> List[ts.GameState]:
    """Every state `step` can reach from `st`, each equally likely, with `rng_state` put back to
    `st`'s so that states equal in every other way stay equal. One state where the step draws
    nothing; the `pool` states of a single draw; `ManyDraws` for more."""
    rng0 = int(st.rng_state)
    first = st.clone()
    step(first)
    draws = calls_between(rng0, int(first.rng_state))
    if draws == 0:
        return [first]
    if draws > 1:
        raise ManyDraws(draws)

    def probe(rng: int) -> ts.GameState:
        child = st.clone()
        child.rng_state = rng
        step(child)
        if calls_between(rng, int(child.rng_state)) != 1:
            raise ManyDraws(calls_between(rng, int(child.rng_state)))
        child.rng_state = rng0
        return child

    outs = [probe(state_for_output(0))]
    keys = [_key(outs[0])]
    pool = 0
    for u in range(1, MAX_POOL + 1):
        c = probe(state_for_output(u))
        k = _key(c)
        if k == keys[0]:
            pool = u
            break
        outs.append(c)
        keys.append(k)
    if pool == 0:
        raise ManyDraws(1)
    for u in range(pool, 3 * pool):
        if _key(probe(state_for_output(u))) != keys[u % pool]:
            raise ManyDraws(1)
    return outs
