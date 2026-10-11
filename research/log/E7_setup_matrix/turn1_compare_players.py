"""Turn 1, two players on the same deals (opening_census records): setups, headlines, AR1s."""
import collections, glob, json, sys
from tools.opening_census import read_games
m = json.load(open("rules/map.json")); C = {int(c["id"]): c for c in m["countries"]}; ids = {c["name"]: i for i, c in C.items()}
import ts_engine as ts
_st = ts.GameState(); ts.Engine.init_game(_st, 1)      # the board before any setup decision
FIX = {"US": {i: int(_st.get_country(i).us_influence) for i in range(84)},
       "USSR": {i: int(_st.get_country(i).ussr_influence) for i in range(84)}}
def setup(g, side):
    s = g["snaps"][0]; key = "us" if side == "US" else "ussr"
    add = {C[i]["name"]: s[key][i] - FIX[side].get(i, 0) for i in range(84) if s[key][i] - FIX[side].get(i, 0) > 0}
    return ", ".join(f"{k} {v}" for k, v in sorted(add.items(), key=lambda kv: (-kv[1], kv[0])))
def half(g, ar, side):
    """(card, how, coup target, spread) of `side`'s half of action round `ar` of turn 1."""
    ps = [p for p in g["plays"] if p["turn"] == 1 and p["ar"] == ar and p["side"] == side]
    card = next((p["taken"] for p in ps if p["kind"] == "card to play" and not p["resolving"]), "")
    mode = next((p["taken"] for p in ps if p["kind"] == "event or Ops"), "")
    ops = next((p["taken"] for p in ps if p["kind"] == "Ops mode"), "")
    coup = next((p["taken"] for p in ps if p["kind"] == "coup target"), "")
    sn = g["snaps"]; i = next((k for k, s in enumerate(sn) if s["turn"] == 1 and s["ar"] == ar and s["phasing"] == side), None)
    spread = ""
    if i is not None and i + 1 < len(sn):
        key = "us" if side == "US" else "ussr"
        d = {C[c]["name"]: sn[i + 1][key][c] - sn[i][key][c] for c in range(84) if sn[i + 1][key][c] > sn[i][key][c]}
        spread = ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))
    how = mode + (f"+{ops}" if ops and ops != mode else "")
    return card, how, coup, spread
def hl(g, side): return next((p["taken"] for p in g["plays"] if p["kind"] == "headline" and p["side"] == side), "")
runs = []
for arg in sys.argv[1:]:
    lab, pat = arg.split("=", 1); runs.append((lab, {g["seed"]: g for g in read_games(glob.glob(pat))}))
seeds = sorted(set.intersection(*(set(r) for _, r in runs)))
print(f"{len(seeds)} deals played by every player\n")
feats = {
    "USSR setup": lambda g: setup(g, "USSR"), "US setup": lambda g: setup(g, "US"),
    "USSR headline": lambda g: hl(g, "USSR"), "US headline": lambda g: hl(g, "US"),
    "USSR AR1 (how, coup target)": lambda g: (lambda h: f"{h[1]} {h[2]}".strip())(half(g, 1, "USSR")),
    "USSR AR1 action": lambda g: (lambda h: f"{h[0]} → {h[1]} {h[2]}".strip())(half(g, 1, "USSR")),
    "USSR AR1 spread": lambda g: half(g, 1, "USSR")[3] or "(no influence added)",
    "US AR1 (how, coup target)": lambda g: (lambda h: f"{h[1]} {h[2]}".strip())(half(g, 1, "US")),
    "US AR1 spread": lambda g: half(g, 1, "US")[3] or "(no influence added)",
}
for name, f in feats.items():
    cs = [collections.Counter(f(r[s]) for s in seeds) for _, r in runs]
    keys = sorted(set().union(*cs), key=lambda k: -sum(c[k] for c in cs))[:10]
    same = sum(f(runs[0][1][s]) == f(runs[1][1][s]) for s in seeds) if len(runs) == 2 else 0
    print(f"### {name}" + (f" -- same choice on {100*same/len(seeds):.0f}% of deals" if len(runs) == 2 else ""))
    print("| | " + " | ".join(l for l, _ in runs) + " |\n|:--|" + "--:|" * len(runs))
    for k in keys: print(f"| {k} | " + " | ".join(f"{100*c[k]/len(seeds):.1f}%" for c in cs) + " |")
    print()
def country_share(g, side, nm): return nm in half(g, 1, side)[3]
print("### US AR1: a point in each country (share of deals)")
print("| | " + " | ".join(l for l, _ in runs) + " |\n|:--|" + "--:|" * len(runs))
for nm in ["Pakistan", "Malaysia", "France", "Italy", "West Germany", "Iran", "Egypt", "Lebanon", "Afghanistan", "Thailand"]:
    print(f"| {nm} | " + " | ".join(f"{100*sum(country_share(r[s], 'US', nm) for s in seeds)/len(seeds):.1f}%" for _, r in runs) + " |")
