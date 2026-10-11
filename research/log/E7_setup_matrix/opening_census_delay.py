import collections, sys, numpy as np
from tools.opening_census import read_games, _fires, USSR_HAND, _card_name
games = read_games(sys.argv[2:]); target = sys.argv[1]
names = {c: _card_name(c) for c in range(1, 111)}; ids = {v: k for k, v in names.items()}
tid = ids[target]
for helper in ["Comecon", "Warsaw Pact Formed", "Romanian Abdication", "Truman Doctrine", "(control) Nuclear Test Ban", "(control) Fidel"]:
    h = helper.replace("(control) ", ""); hid = ids[h]
    res = {True: collections.Counter(), False: collections.Counter()}; ars = {True: [], False: []}
    for g in games:
        starts = {}
        for sn in g["snaps"]: starts.setdefault(sn["turn"], sn)
        ps = g["plays"]
        for t, sn in starts.items():
            if t > g["turns"] or sn["loc"][tid - 1] not in USSR_HAND: continue
            holds = sn["loc"][hid - 1] in USSR_HAND
            i = next((i for i, p in enumerate(ps) if p["turn"] == t and p["side"] == "USSR" and p["kind"] == "card to play" and p["taken"] == target), None)
            if i is None: res[holds]["kept"] += 1
            elif _fires(ps, i, "USSR"): res[holds]["fired"] += 1; ars[holds].append(ps[i]["ar"])
            else: res[holds]["space/UN"] += 1
    def fmt(k):
        c = res[k]; n = sum(c.values())
        return f"n={n:5d} fired {100*c['fired']/n:4.1f}% space/UN {100*c['space/UN']/n:4.1f}% kept {100*c['kept']/n:4.1f}% mean AR when fired {np.mean(ars[k]) if ars[k] else float('nan'):.2f}"
    print(f"{helper:32s} holds: {fmt(True)}\n{'':32s} lacks: {fmt(False)}")
