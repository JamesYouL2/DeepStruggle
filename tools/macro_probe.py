"""Step 0 of the macro-search experiment (`ai/eval/macro_recall.py`): does the network's beam of
complete decisions hold the macro a strong micro-search plays, and what does each method cost?

    # one part: macro starts from 40 greedy self-play games, about 1 in 8 kept
    PYTHONPATH=.:build/release .venv/bin/python tools/macro_probe.py run \\
        --model data/checkpoints/C2_soup+A.pt --games 40 --rate 0.125 --seed 1 --out parts/part-1.json.gz
    # pool the parts
    ... report --parts 'parts/part-*.json.gz' --out-md report.md

`--micro NAME=SPEC` adds a micro-action searcher (a search: spec; the checkpoint is replaced by
--model, so only the settings matter) and `--macro NAME=field=value,...` a macro-search setting
(MacroSearchConfig fields). The defaults are Gumbel k = 8 at 256 simulations against macro search
at depths 1 and 2. On CI: `.github/workflows/macro_probe.yml`.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import gzip
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

DEFAULT_MICRO = ["gumbel256=search:MODEL:256:determinize:all:gumbel_k=8:gumbel_scale=0:fpu_reduction=0.2"]
DEFAULT_MACRO = ["macro_d1=depth=1", "macro_d2=depth=2"]


def macro_config(text: str) -> Any:
    """`field=value,field=value` into a MacroSearchConfig, typed by its own fields."""
    from ai.search.macro_search import MacroSearchConfig
    types = {f.name: f.type for f in dataclasses.fields(MacroSearchConfig)}
    kw: Dict[str, Any] = {}
    for item in [t for t in text.split(",") if t]:
        key, raw = item.split("=", 1)
        t = types.get(key)
        if t is None:
            raise SystemExit(f"--macro: unknown MacroSearchConfig field {key!r} (known: {sorted(types)})")
        kw[key] = (raw.lower() in ("1", "true")) if t == "bool" else int(raw) if t == "int" \
            else float(raw) if t == "float" else raw
    return MacroSearchConfig(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help="a .pt checkpoint")
    r.add_argument("--games", type=int, default=40)
    r.add_argument("--rate", type=float, default=0.125, help="share of macro starts kept")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--micro", nargs="*", default=DEFAULT_MICRO, help="NAME=search:MODEL:... specs")
    r.add_argument("--macro", nargs="*", default=DEFAULT_MACRO, help="NAME=field=value,... settings")
    r.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch's default)")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out-md", required=True)
    a = ap.parse_args()

    if a.cmd == "run":
        from ai.eval.macro_recall import probe, sample_positions
        from ai.search.macro_search import _Net
        from tools.lib.player_agent import NeuralAgent, load_agent
        if a.threads:
            torch.set_num_threads(a.threads)
        torch.manual_seed(a.seed)
        model = NeuralAgent.from_checkpoint(a.model, device="cpu").model.eval()
        positions = sample_positions(_Net(model, torch.device("cpu")), a.games, a.seed, a.rate)
        print(f"{len(positions)} positions from {a.games} games", flush=True)
        micro: Dict[str, Any] = {}
        for item in a.micro:
            name, spec = item.split("=", 1)
            agent = load_agent(spec.replace("MODEL", a.model), device="cpu")
            # One model for every method, so the row counts are comparable.
            setattr(agent, "model", model)
            setattr(getattr(agent, "mcts"), "model", model)
            getattr(agent, "reseed")(a.seed)
            micro[name] = agent
        macro = {}
        for item in a.macro:
            name, text = item.split("=", 1)
            cfg = macro_config(text)
            macro[name] = dataclasses.replace(cfg, seed=a.seed)
        data = probe(model, positions, micro, macro)
        data["meta"] = {"model": os.path.basename(a.model), "seed": a.seed, "games": a.games,
                        "rate": a.rate, "micro": a.micro, "macro": a.macro}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with gzip.open(a.out, "wt") as f:
            json.dump(data, f)
        for name, c in data["cost"].items():
            print(f"{name}: {c['rows'] / max(1, data['n']):.0f} rows/position, "
                  f"{c['seconds'] / max(1, data['n']):.3f} s/position", flush=True)
        return

    from ai.eval.macro_recall import report
    paths: List[str] = sorted(x for pat in a.parts for x in glob.glob(pat))
    parts = []
    for path in paths:
        with gzip.open(path, "rt") as f:
            parts.append(json.load(f))
    if not parts:
        raise SystemExit("no parts")
    meta = parts[0]["meta"]
    micro_names = [m.split("=", 1)[0] for m in meta["micro"]]
    macro_names = [m.split("=", 1)[0] for m in meta["macro"]]
    md = report(parts, micro_names, macro_names)
    md = md.replace("\n", f"\n\nModel `{meta['model']}`; micro: {', '.join(meta['micro'])}; "
                    f"macro: {', '.join(meta['macro'])}.\n", 1)
    with open(a.out_md, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
