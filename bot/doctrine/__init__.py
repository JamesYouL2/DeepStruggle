"""Doctrine: struggler's strategic bot, ported onto ts_engine. See bot/AGENTS.md."""
from bot.doctrine.agent import DoctrineAgent, DoctrineBot
from bot.doctrine.evaluator import Weights
from bot.doctrine.policy import DoctrinePolicy

__all__ = ["DoctrineAgent", "DoctrineBot", "DoctrinePolicy", "Weights"]
