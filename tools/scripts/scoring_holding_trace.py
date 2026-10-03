#!/usr/bin/env python3
"""Where do scoring-card holdings go when the owner neither headlines nor plays them? (soup, greedy self-play)"""
import json, collections, sys, numpy as np, torch, ts_engine as ts
from bindings.ts_env import TsVectorizedEnv, model_obs_features
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import NeuralAgent
cards = {int(c["id"]): c["name"] for c in json.load(open("rules/cards.json"))}
SCORING = {int(c["id"]) for c in json.load(open("rules/cards.json")) if int(c.get("ops", 0)) == 0}
seed = int(sys.argv[1]); games = int(sys.argv[2])
model = NeuralAgent.from_checkpoint("/workspace/data/checkpoints/_soups/shallow_E7-02+03+04+05_1200M.pt", device="cuda").model.eval()
env = TsVectorizedEnv(num_envs=games, base_seed=seed); env.set_obs_features(0, 0); obs, masks, _ = env.reset_all()
n = games; done = [False]*n; held = [dict() for _ in range(n)]; played = [set() for _ in range(n)]
from typing import Any
prev_ctx: list[Any] = [None]*n; out = collections.Counter(); heads: list[dict] = [dict() for _ in range(n)]; head_disc = collections.Counter(); last_play: list[Any] = [None]*n; examples = collections.defaultdict(list)
total = 0
def side_of(loc):
    return 1 if ts.in_hand_of(loc, ts.Player.US) else (-1 if ts.in_hand_of(loc, ts.Player.USSR) else 0)
for _ in range(20000):
    if all(done): break
    with torch.no_grad():
        acts = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).cuda(), torch.from_numpy(np.asarray(masks)).cuda())[0].float().argmax(-1).cpu().numpy()
    for i in range(n):
        if done[i]: continue
        st = env.runner.get_state(i)
        if ts.Engine.is_terminal(st): continue
        now = {c: side_of(st.get_card_location(c)) for c in SCORING}
        now = {c: s for c, s in now.items() if s}
        for c, s in held[i].items():
            if now.get(c) != s:
                if c not in played[i]:
                    loc = str(st.get_card_location(c)).split(".")[-1]
                    pc = prev_ctx[i]
                    key = (f"left hand -> {loc}", f"during {cards.get(pc[0], 'no event') if pc else '?'}")
                    out[key] += 1
                    if loc == "DISCARD_PILE" and pc and pc[0] == 0:
                        same_turn = heads[i].get("turn") == int(st.turn)
                        opp_head = cards.get(heads[i].get(-s, 0), "none") if same_turn else "none this turn"
                        lp = last_play[i]
                        who = "none" if lp is None else (("holder" if lp[1] == s else "opponent") + " played " + cards.get(lp[0], "?") + " for " + lp[2])
                        head_disc[(pc[2].split()[-1], who)] += 1
                    if len(examples[key]) < 2: examples[key].append((i, cards[c], pc[1:] if pc else None))
                played[i].discard(c)
        for c, s in now.items():
            if held[i].get(c) != s: total += 1
        held[i] = now
        ctx = st.ctx(); a = int(acts[i])
        if ctx.decision_type == ts.DecisionType.SELECT_PLAY_MODE and int(ctx.resolving_card) == 0 and int(ctx.pending_op_card):
            _mv = 1 if (ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player) == ts.Player.US else -1
            last_play[i] = (int(ctx.pending_op_card), _mv, ["EVENT", "SPACE", "OPS", "OPS (coup)", "OPS (realign)"][a - ActionEncoder.PLAY_MODE_OFFSET] if ActionEncoder.PLAY_MODE_OFFSET <= a < ActionEncoder.PLAY_MODE_OFFSET + 5 else "?")
        prev_ctx[i] = (int(ctx.resolving_card), str(ctx.decision_type).split(".")[-1], f"T{int(st.turn)} AR{int(st.action_round)} {str(st.current_phase).split('.')[-1]}")
        if ctx.decision_type == ts.DecisionType.SELECT_CARD and a < ActionEncoder.PLAY_MODE_OFFSET and int(ctx.resolving_card) == 0:
            card = int(ts.decode_flat_action(st, a).primary_id)
            mover = 1 if (ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player) == ts.Player.US else -1
            if card in SCORING and now.get(card) == mover: played[i].add(card)
            if st.current_phase == ts.Phase.HEADLINE:
                if heads[i].get("turn") != int(st.turn): heads[i] = {"turn": int(st.turn)}
                heads[i][mover] = card
    obs, masks, _, dones, info = env.step(acts)
    ends = {int(e["env_idx"]): (e["ending_reason"], e["turn"]) for e in info.get("completed_episodes", [])}
    for i, d in enumerate(dones):
        if d and not done[i]:
            done[i] = True
            for c in held[i]:
                if c not in played[i]:
                    key = ("in hand when the game ended", f"ending: {ends.get(i, ('?',))[0]}")
                    out[key] += 1
print(f"scoring holdings: {total}; not played by the holder: {sum(out.values())} ({sum(out.values())/max(1,total):.1%})")
print("unplayed scoring discards with no resolving event -- (phase, the card play just before):")
for k, v in head_disc.most_common(15): print(f"   {v:5d}  {k}")
for k, v in out.most_common(): print(f"{v:6d}  {k[0]:40s} {k[1]}   e.g. {examples.get(k, [])[:1]}")
