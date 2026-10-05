"""The doctrine playout probe (ai/eval/doctrine_playouts.py) measures exactly the plays that broke a
rule: each kept position applies the rule, the model's mode breaks it, eventing complies, and the
position is the play-mode decision itself, so both branches can be played from it."""

from __future__ import annotations

import numpy as np

import ts_engine as ts
from ai.eval import doctrine_census as D
from ai.eval import doctrine_playouts as P
from bindings.action_encoder import ActionEncoder

INFLUENCE = D.MODE_BASE + D.MODES.index("influence")


def _never_events(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Places influence whenever it may, else the first legal action: breaks every event rule."""
    out = np.argmax(masks, axis=1).astype(np.int32)
    out[masks[:, INFLUENCE].astype(bool)] = INFLUENCE
    return out


def test_every_kept_position_breaks_its_rule_and_offers_the_event() -> None:
    rules = {n: (ap, co) for n, ap, co in D.rules()}
    got = P.collect_breaks(_never_events, 24, seed=3, names=P.DEFAULT_RULES, envs=24)
    assert got["items"], "nothing broke a rule in 24 games of a player that never events"
    for it in got["items"]:
        ap, co = rules[it["rule"]]
        play = it["play"]
        assert ap(play) and not co(play) and co({**play, "mode": "event"})
        assert play["mode"] == "influence"
        mask = np.asarray(ActionEncoder.get_legal_mask(it["state"]))
        assert mask[P.EVENT] and mask[INFLUENCE], "both branches must be legal at the kept position"
        assert it["state"].ctx().decision_player != ts.Player.NONE
    assert sum(got["broken"].values()) >= len(got["items"])
    for n, k in got["broken"].items():
        assert k <= got["applied"][n]


def test_the_cap_keeps_at_most_that_many_per_rule() -> None:
    got = P.collect_breaks(_never_events, 24, seed=3, names=P.DEFAULT_RULES, envs=24, max_per_rule=1)
    per = {}
    for it in got["items"]:
        per[it["rule"]] = per.get(it["rule"], 0) + 1
    assert per and max(per.values()) == 1
