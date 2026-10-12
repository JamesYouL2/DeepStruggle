import collections, glob, sys
exec(open(sys.argv[1]).read().split("runs = []")[0])
from tools.opening_census import read_games
VN = ids["Vietnam"]; stab = C[VN]["stability"]
def ctl(s):
    u, r = s["us"][VN], s["ussr"][VN]
    return "US" if u - r >= stab else "USSR" if r - u >= stab else "neither"
for lab, pat in (("greedy", sys.argv[2]), ("rollout", sys.argv[3])):
    G = [g for g in read_games(glob.glob(pat)) if hl(g, "USSR") == "Vietnam Revolts" and half(g, 1, "US")[2] == "Vietnam"]
    by = collections.defaultdict(list)
    for g in G:
        sn = g["snaps"]; i2 = next(k for k, s in enumerate(sn) if s["turn"] == 1 and s["ar"] == 2 and s["phasing"] == "USSR")
        by[ctl(sn[i2])].append(g)
    for state in ("US", "neither"):
        H = by[state]; n = len(H)
        if not n: continue
        first = collections.Counter(); how = collections.Counter(); end = collections.Counter()
        for g in H:
            hit = next((ar for ar in range(2, 8) if (lambda h: h[2] == "Vietnam" or "Vietnam" in h[3])(half(g, ar, "USSR"))), None)
            first["AR2" if hit == 2 else "later in turn 1" if hit else "never in turn 1"] += 1
            if hit:
                h = half(g, hit, "USSR"); how["coup" if h[2] == "Vietnam" else "influence"] += 1
            end[ctl(g["snaps"][-1])] += 1
        f = lambda c, m: "; ".join(f"{k} {100*v/m:.0f}%" for k, v in c.most_common())
        print(f"{lab}: after the US AR1 coup Vietnam is {state} ({n} games) -- USSR answers: {f(first, n)}; by: {f(how, max(1, sum(how.values())))}; end of turn 1: {f(end, n)}")
