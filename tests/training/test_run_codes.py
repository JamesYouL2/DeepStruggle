"""Architecture and recipe codes are derived from what runs recorded, and names from their lineage.

The codes replace attempt numbers, which fixed "observation, architecture, intervention and
recipe" in prose -- the place lineage-critical flags drifted from four times. Here a code is its
flags, so the tests build runs on disk and check the names that come out.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pytest

from ai.training.run_name import parse_label, parse_run_name
from tools.scripts import run_codes as rc

TS = iter(f"20261007_{h:02d}0000" for h in range(24))


def _run(root: str, name: str, parent: Optional[str] = None, resume: str = "resume_state.pt",
         seed: int = 44, steps: int = 200_000_000, **flags: Any) -> str:
    d = os.path.join(root, f"{name}_{next(TS)}")
    os.makedirs(d)
    meta: Dict[str, Any] = {"train_steps": steps, "seed_env": seed, "seed_sampling": seed,
                            "seed_pool": seed, "seed_init": 44, "opponent_frac": 0.3,
                            "opponent_self_pool": True}
    meta.update(flags)
    if parent is not None:
        meta["resumed_from"] = parent if resume == "<dir>" else os.path.join(parent, resume)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(meta, f)
    return d


@pytest.fixture
def lineage(tmp_path: Any) -> Dict[str, str]:
    root = str(tmp_path)
    a = _run(root, "E9-01-44", steps=100_000_000)
    b = _run(root, "E9-02-44", parent=a, steps=300_000_000)                      # continuation
    c = _run(root, "E9-03-44", parent=b, resume="resume_150000000steps.pt", seed=45)
    d = _run(root, "E9-04-44", parent=b, resume="resume_150000000steps.pt", opponent_frac=0.5)
    e = _run(root, "E9-05-44", parent=b, resume="resume_200000000steps.pt", ladder_res_blocks=2)
    f = _run(root, "E9-06-44")                                                   # same as a
    return {k: os.path.basename(v) for k, v in dict(a=a, b=b, c=c, d=d, e=e, f=f).items()} | \
        {"root": root}


def _names(lineage: Dict[str, str]) -> Dict[str, str]:
    runs = rc.load_runs(lineage["root"])
    names, missing = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    assert not missing
    return {k: names[v].text() for k, v in lineage.items() if k != "root"}


def test_names_follow_the_lineage(lineage: Dict[str, str]) -> None:
    n = _names(lineage)
    assert n["a"] == "E9-A1-R1-S44"
    assert n["b"] == n["a"]                       # a continuation that changes nothing keeps it
    assert n["c"] == "E9-A1-R1-S44@150M+S45"      # branch points are absolute lineage steps
    assert n["d"] == "E9-A1-R1-S44@150M+R2"       # a recipe flag is R
    assert n["e"] == "E9-A1-R1-S44@200M+A2"       # an architecture flag is A, not R
    assert n["f"] == "E9-A1-R1-S44-2"             # a second from-scratch run is a replicate


def test_without_update_an_unknown_recipe_is_reported(lineage: Dict[str, str]) -> None:
    runs = rc.load_runs(lineage["root"])
    _, missing = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, False)
    assert missing and all("no " in m for m in missing)


def test_an_inert_setting_does_not_split_a_recipe(tmp_path: Any) -> None:
    """A card-event batch size with the card-event target off trains exactly as without it."""
    root = str(tmp_path)
    _run(root, "E9-01-44")
    _run(root, "E9-02-44", seed=45, aux_card_sample_frac=0.005)
    runs = rc.load_runs(root)
    names, _ = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    assert sorted(p.text() for p in names.values()) == ["E9-A1-R1-S44", "E9-A1-R1-S45"]


@pytest.fixture(scope="module")
def codes() -> Dict[str, Any]:
    with open(rc.CODES_JSON, encoding="utf-8") as f:
        return json.load(f)


class TestTheCommittedRegistry:
    """research/run_codes.json is what names mean; it must be complete and self-consistent."""

    def test_every_code_is_described_and_distinct(self, codes: Dict[str, Any]) -> None:
        for table in ("architectures", "recipes"):
            keys = [json.dumps(e["flags"], sort_keys=True) for e in codes[table].values()]
            assert len(set(keys)) == len(keys), f"two {table} share their flags"
            assert all(e["description"] for e in codes[table].values())

    def test_every_mapped_name_parses_and_uses_known_codes(self, codes: Dict[str, Any]) -> None:
        for d, name in codes["runs"].items():
            f = parse_run_name(name).fields()
            assert "A" + f["A"] in codes["architectures"], d
            assert "R" + f["R"] in codes["recipes"], d

    def test_the_best_soup_is_expressible(self, codes: Dict[str, Any]) -> None:
        """The 4,800M soup's three ingredients are exactly the three mapped runs."""
        lab = parse_label("E7-A4-R1-S44@4390M+(S44,45,46)@4800M")
        mapped = set(codes["runs"].values())
        assert all(r.text() in mapped for r in lab.ingredients)


def test_link_directories_alias_runs_without_becoming_runs(lineage: Dict[str, str]) -> None:
    """`--link` gives each old-scheme run a new-name directory of symlinks. It is an alias: the
    derivation must not count it as a run, and a run resumed through it hangs off the real one."""
    root = lineage["root"]
    runs = rc.load_runs(root)
    names, _ = rc.assign(runs, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    made = rc.link_runs(runs, names, root)
    assert len(made) == 6 and "E9-A1-R1-S44@150M+S45_" + lineage["c"].split("_", 1)[1] in made
    link = os.path.join(root, made[0])
    assert os.path.islink(os.path.join(link, "metadata.json"))
    assert rc.link_runs(runs, names, root) == []                  # idempotent
    # a continuation launched under the new name, resumed through the link directory
    c_link = os.path.join(root, "E9-A1-R1-S44@150M+S45_" + lineage["c"].split("_", 1)[1])
    _run(root, "E9-A1-R1-S44@150M+S45", parent=c_link, resume="<dir>", seed=45, steps=400_000_000)
    runs2 = rc.load_runs(root)
    assert len(runs2) == len(runs) + 1
    names2, _ = rc.assign(runs2, {"architectures": {}, "recipes": {}, "overrides": {}}, True)
    newest = max(runs2, key=lambda r: r.ts)
    assert names2[newest.dir].text() == "E9-A1-R1-S44@150M+S45"   # a continuation keeps the name
