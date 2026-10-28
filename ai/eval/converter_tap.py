"""The live position at every decision the human-log converter emits, in step with its samples.

`tools.lib.ts_replayer_convert` calls `ts.extract_observation` exactly once per emitted sample, at
the single line that appends to `Conversion.samples`, so wrapping that binding for the duration of
one conversion yields each sample's position in order. The conversion may later drop a trailing
turn (`_rewind_to_turn_start` deletes a suffix of `samples`), so the taps can outnumber the samples
but never fall short of them -- `ConverterTap.verify` checks exactly that, which is what catches
the converter moving to another binding (`extract_observation_features`), after which a wrapper
would silently see nothing.

One implementation for `human_corpus` and `human_dominance` (and `tools/build_human_dataset.py`,
through `human_dominance._capturing`), which each carried a copy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import ts_engine as ts
from tools.lib.ts_replayer_convert import Conversion


class ConverterTap:
    """How many times the wrapped binding fired during one conversion."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, conv: Conversion) -> None:
        """Raise unless every emitted sample was seen: fewer taps than samples means the converter
        no longer calls the wrapped binding, and every list built from the taps is misaligned."""
        if self.calls < len(conv.samples):
            raise RuntimeError(
                f"the converter emitted {len(conv.samples)} samples but ts.extract_observation was "
                f"called {self.calls} times: it no longer builds its observations through the "
                f"binding this tap wraps, so the captured positions do not line up with the samples")


@contextmanager
def tap_converter_observations(on_sample: Callable[["ts.GameState", "ts.Player"], None]
                               ) -> Iterator[ConverterTap]:
    """Call `on_sample(state, mover)` at every sample the converter emits during the block (the
    state is the converter's live one: clone it to keep it)."""
    original = ts.extract_observation
    tap = ConverterTap()

    def observe(state: "ts.GameState", perspective: "ts.Player") -> Any:
        # Named `perspective` to match the binding it replaces: a caller may pass it by keyword,
        # and a wrapper with a different parameter name would break there and nowhere else.
        tap.calls += 1
        on_sample(state, perspective)
        return original(state, perspective)

    ts.extract_observation = observe
    try:
        yield tap
    finally:
        ts.extract_observation = original
