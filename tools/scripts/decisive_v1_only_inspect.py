"""Replay the v1-only positions (E7-02-44, shard 0, first batch) and trace what the v1-winning action does."""
import json, sys, collections, numpy as np, torch, ts_engine as ts
from bindings.ts_env import TsVectorizedEnv, model_obs_features
from tools.lib.player_agent import NeuralAgent
from ai.eval.decisive_probe import classify_in_view
from bindings.action_encoder import ActionEncoder as A
O = "/workspace/data/logs/temp/decisive_cost"
cards = {int(c["id"]): c["name"] for c in json.load(open("rules/cards.json"))}
countries = {int(c["id"]): c["name"] for c in json.load(open("rules/map.json"))["countries"]}
r1 = json.load(open(f"{O}/e702_v1_0.json"))["rows"]; r2 = json.load(open(f"{O}/e702_v2_0.json"))["rows"]
targets = {}
for gi, (a, b) in enumerate(zip(r1[:512], r2[:512])):
    w2 = {w[0] for w in b["wins"]}
    for w in a["wins"]:
        if w[0] not in w2: targets.setdefault(gi, {})[w[0]] = set(w[3])
def name_action(st, a):
    ma = ts.decode_flat_action(st, int(a)); d = str(ma.decision_type).split(".")[-1]
    if ma.decision_type == ts.DecisionType.POINT_NODE: return f"place in {countries.get(int(ma.primary_id), ma.primary_id)}"
    if ma.decision_type == ts.DecisionType.SELECT_CARD: return f"select {cards.get(int(ma.primary_id), ma.primary_id)}"
    return f"{d} {int(ma.primary_id)}/{int(ma.secondary_id) if hasattr(ma,'secondary_id') else ''} (flat {a})"
def trace(st, a, seed):
    c = st.clone(); c.rng_state = seed; ts.Engine.step_flat(c, int(a), False); steps = []; rolls = []
    for _ in range(40):
        if ts.Engine.is_terminal(c): break
        cx = c.ctx(); legal = np.flatnonzero(np.asarray(ts.get_flat_action_mask(c)))
        dt = str(cx.decision_type).split(".")[-1]
        if cx.decision_type == ts.DecisionType.ROLL_DIE:
            ts.Engine.step(c, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0)); rolls.append(int(c.to_dict().get("last_die_roll", -1))); steps.append("ROLL"); continue
        if len(legal) == 1:
            steps.append(dt); ts.Engine.step_flat(c, int(legal[0]), False); continue
        steps.append(f"CHOICE:{dt}"); break
    end = ts.game_ending_reason(c) if ts.Engine.is_terminal(c) else "not over"
    util = float(ts.Engine.get_terminal_utility(c)) if ts.Engine.is_terminal(c) else 0.0
    return util, end, rolls, steps, int(c.victory_points), int(c.defcon)
model = NeuralAgent.from_checkpoint("/workspace/data/checkpoints/E7-02-44_20260930_222159/snapshot_1200029696steps.pt", device="cuda").model.eval()
n = 512; env = TsVectorizedEnv(num_envs=n, base_seed=77000); env.set_obs_features(0, 0); obs, masks, _ = env.reset_all()
k_dec = [0]*n; counted = [False]*n; rng = np.random.default_rng(1)
for _ in range(20000):
    if all(counted): break
    with torch.no_grad():
        acts = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).cuda(), torch.from_numpy(np.asarray(masks)).cuda())[0].float().argmax(-1).cpu().numpy()
    for i in range(n):
        if counted[i]: continue
        st = env.runner.get_state(i)
        if ts.Engine.is_terminal(st): continue
        cx = st.ctx()
        if cx.decision_type == ts.DecisionType.ROLL_DIE: continue
        pl = cx.decision_player if cx.decision_player != ts.Player.NONE else st.phasing_player
        k = k_dec[i]; k_dec[i] += 1
        if i in targets and k in targets[i]:
            base = sorted(targets[i][k]); me = 1 if pl == ts.Player.US else -1
            card = int(cx.resolving_card) or int(cx.pending_op_card)
            legal = [int(x) for x in np.flatnonzero(np.asarray(ts.get_flat_action_mask(st)))]
            d = st.to_dict()
            print(f"\n=== game {i} decision {k}: {'US' if me==1 else 'USSR'} to move, {str(cx.decision_type).split('.')[-1]}, card {cards.get(card, card)}; "
                  f"T{int(st.turn)} AR{int(st.action_round)} DEFCON {int(st.defcon)} VP {int(st.victory_points)} milops US {d['mil_ops'] if 'mil_ops' in d else '?'}; legal {len(legal)}, v1 wins {len(base)}")
            print("    v1-winning:", ", ".join(name_action(st, a) for a in base[:4]), "..." if len(base) > 4 else "")
            if i == 406:
                E = ts.EffectBits
                flags = {n: bool(st.has_flag(getattr(E, n))) for n in dir(E) if any(t in n for t in ("QUAGMIRE", "RED_SCARE", "NATO", "PURGE"))}
                print("    flags:", flags, "| effective Ops of the US hand:",
                      {cards[x]: __import__("ai.training.card_event_targets", fromlist=["effective_ops"]).effective_ops(st, x, ts.Player.US, False, False) for x in (46, 53, 80)})
                c = st.clone(); ts.Engine.step_flat(c, base[0], False); dd = c.to_dict()
                print("    after placing: UK US/USSR influence", dd["countries"]["United Kingdom"]["us_influence"], dd["countries"]["United Kingdom"]["ussr_influence"],
                      "controlled_by", dd["countries"]["United Kingdom"]["controlled_by"], "VP", int(c.victory_points))
                for _ in range(6):
                    if ts.Engine.is_terminal(c): break
                    cx2 = c.ctx(); lg = np.flatnonzero(np.asarray(ts.get_flat_action_mask(c)))
                    print("     ctx: resolving_card", cards.get(int(cx2.resolving_card), int(cx2.resolving_card)), "phase", str(c.current_phase).split(".")[-1],
                          "phasing", str(c.phasing_player).split(".")[-1], "NATO" if c.has_flag(ts.EffectBits.NATO_ACTIVE) else "no NATO" if hasattr(ts.EffectBits, "NATO_ACTIVE") else "", "milops", c.to_dict()["mil_ops"])
                    print("     next:", str(cx2.decision_type).split(".")[-1], "player", str(cx2.decision_player).split(".")[-1],
                          "AR", int(c.action_round), "legal", [name_action(c, int(x)) for x in lg[:3]], "US hand", c.to_dict()["hands"].get("US") if isinstance(c.to_dict()["hands"], dict) else "?")
                    if cx2.decision_type == ts.DecisionType.ROLL_DIE: ts.Engine.step(c, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
                    else: ts.Engine.step_flat(c, int(lg[0]), False)
                print("     end:", ts.game_ending_reason(c) if ts.Engine.is_terminal(c) else "not over", "VP", int(c.victory_points))
                alt = [int(x) for x in np.flatnonzero(np.asarray(ts.get_flat_action_mask(st))) if int(x) not in base][0]
                c = st.clone(); ts.Engine.step_flat(c, alt, False)
                print("    instead", name_action(st, alt), ": next", str(c.ctx().decision_type).split(".")[-1], "player", str(c.ctx().decision_player).split(".")[-1],
                      "resolving", cards.get(int(c.ctx().resolving_card), 0), "VP", int(c.victory_points), "legal",
                      [name_action(c, int(x)) for x in np.flatnonzero(np.asarray(ts.get_flat_action_mask(c)))[:4]])
                sys.exit(0)
            outcomes = collections.Counter(); ex = None
            for s in range(60):
                util, end, rolls, steps, vp, dc = trace(st, base[0], int(rng.integers(1, 2**63)))
                outcomes[(("mover wins" if util*me > 0 else "mover loses" if util*me < 0 else "game goes on"), end)] += 1
                if ex is None or (util*me <= 0 and ex[0]*me > 0): ex = (util, end, rolls, steps, vp, dc)
            print(f"    after '{name_action(st, base[0])}' over 60 seeds:", dict(outcomes))
            assert ex is not None
            print(f"    a trace: rolls {ex[2]}, forced steps {ex[3][:10]}, VP {ex[4]}, DEFCON {ex[5]}, end: {ex[1]}")
    obs, masks, _, dones, info = env.step(acts)
    for i, dn in enumerate(dones):
        if dn: counted[i] = True
