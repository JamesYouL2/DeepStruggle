import collections, glob, importlib.util, sys
spec = importlib.util.spec_from_file_location("cp", sys.argv[1]); cp = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv; sys.argv = [sys.argv[0]]; spec.loader.exec_module(cp) if False else None
from tools.opening_census import read_games
exec(open(sys.argv_saved[1]).read().split("runs = []")[0])   # setup(), half(), hl() from the comparison script
games = read_games(glob.glob(sys.argv_saved[2]))
HL = ["Vietnam Revolts", "De Gaulle Leads France", "Suez Crisis", "Socialist Governments", "Nasser", "Decolonization"]
def top(c, n, k=6): return "; ".join(f"{x} {100*v/n:.0f}%" for x, v in c.most_common(k))
for h in HL:
    G = [g for g in games if hl(g, "USSR") == h]; n = len(G)
    a1 = collections.Counter(); how = collections.Counter(); sp = collections.Counter(); us = collections.Counter(); ush = collections.Counter(); usp = collections.Counter()
    for g in G:
        c, m, coup, s = half(g, 1, "USSR")
        a1[f"{c} → {m}{' ' + coup if coup else ''}"] += 1
        how["coup " + coup if coup else ("influence" if s else (m or "scoring/other"))] += 1
        if s and not coup: sp[s] += 1
        uc, um, ucoup, us_s = half(g, 1, "US")
        us["coup " + ucoup if ucoup else ("influence" if us_s else (um or "scoring/other"))] += 1
        if us_s and not ucoup:
            for part in us_s.split(", "): usp[part.rsplit(" ", 1)[0]] += 1
        ush[hl(g, "US")] += 1
    print(f"\n### USSR headlines {h} ({n:,} games, {100*n/len(games):.1f}%)")
    print("US headline alongside:", top(ush, n, 5))
    print("USSR AR1, kind:", top(how, n))
    print("USSR AR1, action:", top(a1, n))
    if sp: print("USSR AR1 influence spreads (no coup):", top(sp, sum(sp.values()), 5), f"(of {sum(sp.values())})")
    print("US AR1, kind:", top(us, n))
    print("US AR1 influence, countries (share of all games):", "; ".join(f"{x} {100*v/n:.0f}%" for x, v in usp.most_common(7)))
