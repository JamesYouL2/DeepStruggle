"""Native C++ Engine Python Bindings, Vectorized Envs, and Action Codecs."""
import os
import sys

# Fallbacks, appended so that an explicit PYTHONPATH wins: prepending build/release once made every
# process that imported `bindings` before `ts_engine` load the release engine whatever build it was
# given (a match's spawned workers on build/dev ran release's engine, P33). Release still comes first
# among the fallbacks.
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_cur_dir)
for p in [os.path.join(_root_dir, "build", "release"), os.path.join(_root_dir, "build"), _root_dir, _cur_dir]:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

from .action_encoder import ActionEncoder
from .ts_env import TsVectorizedEnv, TsSingleEnv

__all__ = ["ActionEncoder", "TsVectorizedEnv", "TsSingleEnv"]
