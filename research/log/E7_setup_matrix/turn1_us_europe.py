import collections, json, sys
from tools.turn1_census import read_rows
from tools.turn1_pricing import state_from_token
m = json.load(open("rules/map.json")); C = {int(c["id"]): c for c in m["countries"]}
EURO = [i for i, c in C.items() if c["region"] == "Europe"]; EBG = [i for i in EURO if C[i]["battleground"]]
ids = {c["name"]: i for i, c in C.items()}
def ctrl(st, i, who):
    c = st.get_country(i); u, s = int(c.us_influence), int(c.ussr_influence)
    me, op = (u, s) if who == "US" else (s, u)
    return me - op >= C[i]["stability"] and me > 0
def status(st, who):
    opp = "USSR" if who == "US" else "US"
    b, a = [i for i in EBG if ctrl(st, i, who)], [i for i in EURO if ctrl(st, i, who)]
    ob, oa = [i for i in EBG if ctrl(st, i, opp)], [i for i in EURO if ctrl(st, i, opp)]
    if len(b) == len(EBG) and len(a) > len(oa): return "control"
    if len(b) > len(ob) and len(a) > len(oa) and len(b) >= 1 and len(a) > len(b): return "domination"
    return "presence" if a else "none"
import glob
for label in sys.argv[1:]:
    lab, d = label.split("=")
    rows = read_rows(glob.glob(d + "/*/*.jsonl.gz")); by = collections.defaultdict(list)
    for r in rows: by[r["seed"]].append(r)
    n = len(by); cnt = collections.Counter()
    for seed, rs in by.items():
        rs.sort(key=lambda r: r["choice"])
        # after the US's AR1: the USSR's AR2 card choice
        a2 = next((r for r in rs if r["side"] == "USSR" and r["ar"] == 2 and r["kind"] == "card to play"), None)
        last = state_from_token(rs[-1]["pos"])
        for tag, st in (("after US AR1", state_from_token(a2["pos"]) if a2 else None), ("end of T1", last)):
            if st is None: continue
            cnt[(tag, "US " + status(st, "US"))] += 1; cnt[(tag, "USSR " + status(st, "USSR"))] += 1
            for nm in ("West Germany", "France", "Italy"):
                cnt[(tag, f"US controls {nm}")] += ctrl(st, ids[nm], "US")
            cnt[(tag, "US Euro BGs ≥3")] += sum(ctrl(st, i, "US") for i in EBG) >= 3
    print(f"== {lab} ({n} games)")
    for tag in ("after US AR1", "end of T1"):
        print("  " + tag + ": " + "; ".join(f"{k} {100*v/n:.1f}%" for (t, k), v in sorted(cnt.items()) if t == tag and v))
