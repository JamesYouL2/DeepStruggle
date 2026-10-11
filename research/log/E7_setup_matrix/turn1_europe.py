"""Europe on turn 1 from turn1_pricing rows: USSR-controlled European battlegrounds, domination,
and how East European Unrest / Independent Reds leave each hand."""
import collections, glob, json, sys
import ts_engine as ts
from tools.turn1_census import read_rows, option_label
from tools.turn1_pricing import state_from_token

m = json.load(open("rules/map.json"))
C = {int(c["id"]): c for c in m["countries"]}
EURO = [i for i, c in C.items() if c["region"] == "Europe"]
EBG = [i for i in EURO if C[i]["battleground"]]
EEU, IR = 29, 22

def ctrl(st, i, who):
    c = st.get_country(i); u, s = int(c.us_influence), int(c.ussr_influence)
    me, op = (s, u) if who == "USSR" else (u, s)
    return me - op >= C[i]["stability"] and me > 0

def europe(st):
    out = {}
    for who in ("USSR", "US"):
        bg = [i for i in EBG if ctrl(st, i, who)]; all_ = [i for i in EURO if ctrl(st, i, who)]
        out[who] = (bg, all_)
    (sb, sa), (ub, ua) = out["USSR"], out["US"]
    dom = len(sb) > len(ub) and len(sa) > len(ua) and len(sb) >= 1 and len(sa) > len(sb)
    ctl = len(sb) == len(EBG) and len(sa) > len(ua)
    return sb, sa, ub, ua, ("control" if ctl else "domination" if dom else "")

rows = read_rows(sys.argv[1:])
by = collections.defaultdict(list)
for r in rows: by[r["seed"]].append(r)
n = len(by)
last_bg = collections.Counter(); ever3 = 0; dom_last = 0; dom_ever = 0; third = collections.Counter(); first_ar = collections.Counter()
hold = {s: {EEU: collections.Counter(), IR: collections.Counter()} for s in ("USSR", "US")}
ir_target = collections.Counter()
for seed, rs in by.items():
    rs.sort(key=lambda r: r["choice"])
    sts = [(r, state_from_token(r["pos"])) for r in rs]
    got3 = False; gotdom = False
    for r, st in sts:
        sb, sa, ub, ua, d = europe(st)
        if len(sb) >= 3 and not got3:
            got3 = True; first_ar[r["ar"]] += 1
            for i in sb:
                if C[i]["name"] not in ("East Germany", "Poland"): third[C[i]["name"]] += 1
        if d: gotdom = True
    ever3 += got3; dom_ever += gotdom
    sb, sa, ub, ua, d = europe(sts[-1][1]); last_bg[len(sb)] += 1; dom_last += bool(d)
    # card handling
    st0 = sts[0][1]
    for side, P in (("USSR", ts.Player.USSR), ("US", ts.Player.US)):
        for card in (EEU, IR):
            if not ts.in_hand_of(st0.get_card_location(card), P): continue
            how = "held through turn 1"
            for i, (r, st) in enumerate(sts):
                if r["side"] != side: continue
                if r["kind"] in ("headline", "card to play") and option_label(r["names"][r["taken"]]) == ts.CardData.get_card_info(card)["name"]:
                    if r["kind"] == "headline": how = "headlined"; break
                    nxt = next((r2 for r2, _ in sts[i+1:] if r2["side"] == side and r2["kind"] == "event or Ops" and r2["card_name"] == r["names"][r["taken"]].split("(")[1][:-1]), None)
                    how = option_label(nxt["names"][nxt["taken"]]) if nxt else "played (no choice row)"
                    break
            hold[side][card][how] += 1
    # US influence added in Eastern Europe by an Independent Reds event (US side)
    for i, (r, st) in enumerate(sts):
        if r["kind"] == "event placement (first point)" and r["card_name"] == "Independent Reds":
            ir_target[option_label(r["names"][r["taken"]])] += 1
print(f"{n} games")
print("USSR-controlled European battlegrounds at the last decision of turn 1:", {k: f"{100*v/n:.1f}%" for k, v in sorted(last_bg.items())})
print(f"3+ at some decision of turn 1: {100*ever3/n:.1f}%; first reached at AR: {dict(sorted(first_ar.items()))}; third BG: {dict(third.most_common())}")
print(f"USSR Europe domination/control at the last decision: {100*dom_last/n:.1f}%; at some decision: {100*dom_ever/n:.1f}%")
for side in ("USSR", "US"):
    for card, nm in ((EEU, "East European Unrest"), (IR, "Independent Reds")):
        c = hold[side][card]; t = sum(c.values())
        print(f"{side} holds {nm} ({t}):", {k: f"{100*v/t:.1f}%" for k, v in c.most_common()})
print("Independent Reds event, country chosen:", dict(ir_target.most_common()))

# Yugoslavia at the last decision of turn 1, and whether Independent Reds' starred event has fired
yu = collections.Counter(); YUG = 18
for seed, rs in by.items():
    rs.sort(key=lambda r: r["choice"])
    st = state_from_token(rs[-1]["pos"]); c = st.get_country(YUG)
    fired = str(st.get_card_location(IR)).endswith("REMOVED_FROM_GAME")
    yu["US influence in Yugoslavia"] += int(c.us_influence) > 0
    yu["USSR controls Yugoslavia"] += ctrl(st, YUG, "USSR")
    yu["Independent Reds event has fired"] += fired
    yu["fired and US in Yugoslavia"] += fired and int(c.us_influence) > 0
print({k: f"{100*v/n:.1f}%" for k, v in yu.items()})
