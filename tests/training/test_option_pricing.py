"""ai/eval/option_pricing.py and the replay writer's prefix / excerpt: a game rebuilt from its seed
and moves is the game priced, a branched replay diverges exactly at the branch, and an excerpt
stops at the end of its turn."""

import json
import os
import random

import torch

import ts_engine as ts
from ai.eval import option_pricing as op
from ai.models.ladder_net import create_ladder_net
from tools.lib.self_play import generate_self_play_replay

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)


def test_priced_positions_rebuild_and_branch(tmp_path) -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    net = create_ladder_net(dev, **CFG).eval()
    games, cands = op.play_games(net, dev, [5], sample_p=1.0, max_options=12, rng=random.Random(1))
    turn1 = [c for c in cands if int(c.state.turn) == 1]
    assert turn1, "no small decision on turn 1"
    c = turn1[len(turn1) // 2]
    g = games[c.game]
    # The rebuilt position is the priced one, bit for bit.
    assert op.rebuild(g.seed, g.choices[:c.choice]).to_save_json() == c.state.to_save_json()
    op.price(net, dev, [c], worlds=4, seed=3)
    assert c.values is not None and c.values.shape[0] == len(c.options) and c.values.shape[1] >= 1
    assert bool((abs(c.values) <= 1.0 + 1e-5).all())
    # Two branches of one replay: identical up to the decision, then the forced moves differ, and
    # both stop when turn 1 is over.
    alt = next(a for a in c.options if a != c.taken)
    docs = []
    for tag, a in (("A", c.taken), ("B", alt)):
        path = os.path.join(tmp_path, f"{tag}.tslog.json")
        generate_self_play_replay(net, seed=g.seed, temperature=0.0, output_path=path, device="cpu",
                                  verbose=False, prefix=list(g.choices[:c.choice]) + [a],
                                  stop_after_turn=1)
        docs.append(json.load(open(path)))
    a_steps, b_steps = docs[0]["steps"], docs[1]["steps"]
    k = next(i for i, (x, y) in enumerate(zip(a_steps, b_steps)) if x["action"] != y["action"])
    assert a_steps[k]["action"]["flat_action_idx"] == c.taken
    assert b_steps[k]["action"]["flat_action_idx"] == alt
    for d in docs:
        assert d["metadata"]["result"]["winner"] == "NONE"
        assert d["metadata"]["result"]["end_turn"] == 1
        assert all(s["turn"] <= 2 for s in d["steps"])


def test_lookahead_and_nested_pricing_run() -> None:
    """Depth 2 (the decider's next small decisions priced) and nested collection (decisions met in
    the untaken branches) both run and keep every value in [-1, 1]."""
    torch.manual_seed(0)
    dev = torch.device("cpu")
    net = create_ladder_net(dev, **CFG).eval()
    _games, cands = op.play_games(net, dev, [6], sample_p=0.05, max_options=12, rng=random.Random(2))
    cs = cands[:3]
    nested: list = []
    op.price(net, dev, cs, worlds=4, seed=5, lookahead=2, look_worlds=4, nested=nested, nested_p=1.0)
    for c in cs:
        assert c.values is not None and bool((abs(c.values) <= 1.0 + 1e-5).all())
    assert all(op.priceable(n.cand.state, 12) for n in nested)
    assert all(n.cand.options[0] in n.cand.options for n in nested)
