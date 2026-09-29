"""Brinkman: struggler's strategic bot, ported onto ts_engine. See bot/AGENTS.md."""
from bot.brinkman.agent import BrinkmanAgent, BrinkmanBot
from bot.brinkman.evaluator import Weights
from bot.brinkman.policy import BrinkmanPolicy

__all__ = ["BrinkmanAgent", "BrinkmanBot", "BrinkmanPolicy", "Weights"]
