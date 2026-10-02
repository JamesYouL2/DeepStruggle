"""Replay greedy shard-0 games of E7-02-44 to the positions where only classifier v1 sees a win,
and re-run v1 there under different RNG seeds: is its 'win' dice-dependent?"""
import json, sys, numpy as np, torch, ts_engine as ts
from bindings.ts_env import TsVectorizedEnv, model_obs_features
from tools.lib.player_agent import NeuralAgent
from ai.eval.decisive_probe import classify_in_view
O = "/workspace/data/logs/temp/decisive_cost"
r1 = json.load(open(f"{O}/e702_v1_0.json"))["rows"]; r2 = json.load(open(f"{O}/e702_v2_0.json"))["rows"]
targets = {}                                   # (game) -> {decision index: v1 win set}
for gi, (a, b) in enumerate(zip(r1, r2)):
    w2 = {w[0] for w in b["wins"]}
    for w in a["wins"]:
        if w[0] not in w2:
            targets.setdefault(gi, {})[w[0]] = set(w[3])
print(f"{sum(len(v) for v in targets.values())} v1-only positions in {len(targets)} shard-0 games")
model = NeuralAgent.from_checkpoint("/workspace/data/checkpoints/E7-02-44_20260930_222159/snapshot_1200029696steps.pt", device="cuda").model.eval()
n = 512; env = TsVectorizedEnv(num_envs=n, base_seed=77000)
env.set_obs_features(model_obs_features(model), model_obs_features(model)); obs, masks, _ = env.reset_all()
k_dec = [0] * n; counted = [False] * n; res = {"stable": 0, "varies": 0, "never_again": 0}; examples = []
rng = np.random.default_rng(0)
for _ in range(20000):
    if all(counted): break
    with torch.no_grad():
        acts = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).cuda(), torch.from_numpy(np.asarray(masks)).cuda())[0].float().argmax(-1).cpu().numpy()
    for i in range(n):
        if counted[i]: continue
        st = env.runner.get_state(i)
        if ts.Engine.is_terminal(st): continue
        ctx = st.ctx()
        if ctx.decision_type == ts.DecisionType.ROLL_DIE: continue
        player = ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player
        k = k_dec[i]; k_dec[i] += 1
        if i in targets and k in targets[i]:
            base = targets[i][k]
            again = {a for a, kd in classify_in_view(st.clone(), player, False).items() if kd == "win"}
            seen = []
            for _r in range(12):
                c = st.clone(); c.rng_state = int(rng.integers(1, 2**63))
                seen.append({a for a, kd in classify_in_view(c, player, False).items() if kd == "win"})
            if again != base:
                res["never_again"] += 1          # replay drifted -- position not reproduced
            elif all(s == base for s in seen):
                res["stable"] += 1
                # Is it a forced win? Take the first v1-winning action, then let the opponent answer
                # at random (200 tries) while the mover and the dice play on: can the opponent avoid it?
                a0 = sorted(base)[0]; me = int(player); escapes = 0; opp_moved = 0
                for t in range(200):
                    c = st.clone(); c.rng_state = int(rng.integers(1, 2**63))
                    ts.Engine.step_flat(c, a0, True)
                    moved = False
                    for _s in range(400):
                        if ts.Engine.is_terminal(c): break
                        cx = c.ctx(); pl = cx.decision_player if cx.decision_player != ts.Player.NONE else c.phasing_player
                        legal = np.flatnonzero(np.asarray(ts.get_flat_action_mask(c)))
                        if int(pl) != me: moved = True
                        ts.Engine.step_flat(c, int(rng.choice(legal)), True)
                    opp_moved += moved
                    if not (ts.Engine.is_terminal(c) and float(ts.Engine.get_terminal_utility(c)) * me > 0):
                        escapes += 1
                legal_all = [int(x) for x in np.flatnonzero(np.asarray(ts.get_flat_action_mask(st)))]
                others = [x for x in legal_all if x not in base]
                other_wins = 0; other_tries = 0
                for x in others[:5]:
                    for t in range(40):
                        c = st.clone(); c.rng_state = int(rng.integers(1, 2**63)); ts.Engine.step_flat(c, x, True)
                        for _s in range(400):
                            if ts.Engine.is_terminal(c): break
                            legal = np.flatnonzero(np.asarray(ts.get_flat_action_mask(c)))
                            ts.Engine.step_flat(c, int(rng.choice(legal)), True)
                        other_tries += 1
                        other_wins += int(ts.Engine.is_terminal(c) and float(ts.Engine.get_terminal_utility(c)) * me > 0)
                examples.append((f"legal {len(legal_all)}, v1 wins {len(base)}; non-v1-win actions won {other_wins}/{other_tries}",))
                examples.append((i, k, str(ctx.decision_type).split(".")[-1], int(ctx.resolving_card) or int(ctx.pending_op_card),
                                 "mover " + ("US" if me == 1 else "USSR"), f"T{int(st.turn)} AR{int(st.action_round)} DEFCON {int(st.defcon)} VP {int(st.victory_points)}",
                                 f"opponent moved in {opp_moved}/200, mover failed to win in {escapes}/200"))
            else:
                res["varies"] += 1
    obs, masks, _, dones, info = env.step(acts)
    for i, d in enumerate(dones):
        if d: counted[i] = True
print(res)
print("v1-only wins that do NOT vary with the dice -- can the opponent avoid them?")
for e in examples: print("  ", e)
