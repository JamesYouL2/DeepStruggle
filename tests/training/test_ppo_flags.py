"""--clip-eps and --ppo-epochs (P39 arm 3): their defaults are the trainer's own, so every run launched before the
flags existed keeps its recipe, and launch_flags fills the same values in for runs that never recorded them."""

import inspect

from ai.training.nash_pg import BaseNashPGTrainer
from ai.training.train import build_parser
from tools.scripts import launch_flags


def test_the_flags_default_to_the_trainers_values() -> None:
    sig = inspect.signature(BaseNashPGTrainer.__init__).parameters
    args = build_parser().parse_args([])
    assert args.clip_eps == sig["clip_eps"].default == 0.2
    assert args.ppo_epochs == sig["num_epochs"].default == 4
    assert launch_flags._BEFORE_RECORDED["clip_eps"]({}) == 0.2
    assert launch_flags._BEFORE_RECORDED["ppo_epochs"]({}) == 4


def test_the_flags_parse() -> None:
    args = build_parser().parse_args(["--clip-eps", "0.1", "--ppo-epochs", "2"])
    assert (args.clip_eps, args.ppo_epochs) == (0.1, 2)
