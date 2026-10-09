"""Shared evaluation, simulation, and analytics engine for Twilight Struggle AI tools.

Import from the submodules (`from tools.lib.player_agent import load_agent`). This package
deliberately re-exports nothing: importing any `tools.lib` module used to import torch and the
engine first, through re-exports here, which kept the torch-free modules (the leaderboard fit,
tools/lib/leaderboard.py, which CI runs without either) from being importable on their own.
"""
