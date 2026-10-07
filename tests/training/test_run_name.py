"""The run-name grammar: E-A-R-S roots, branch points, soups and SWAs.

A name that parses must mean one thing, and a name that could mean two must not parse: the
directory, the registry row and every report quote it, so an ambiguity here becomes an
unattributable number later (research/method/run_nomenclature.md).
"""

from __future__ import annotations

import pytest

from ai.training.generic_trainer import _resolve_run_dir
from ai.training.run_name import (RunNameError, is_run_name, parse_label, parse_run_name,
                                  shares_a_state)
from tools.lib.checkpoint_id import checkpoint_label


def _ingredients(label: str) -> list[str]:
    return [r.text() for r in parse_label(label).ingredients]


class TestRunNames:
    @pytest.mark.parametrize("name", [
        "E7-A4-R1-S44",
        "E7-A4-R1-S44-2",
        "E7-A4-R1-S44@4390M+S45",
        "E7-A4-R1-S44@4390M+A7-R23@4800M+R24@5200M+R1",
        "E6-A1-R1-S44@560M+E7",
        "E6.1-A2-R1-S44",
    ])
    def test_round_trip(self, name: str) -> None:
        assert parse_run_name(name).text() == name
        assert is_run_name(name)

    def test_fields_follow_the_branches(self) -> None:
        f = parse_run_name("E7-A4-R1-S44@4390M+A7-R23@4800M+R24").fields()
        assert f == {"E": "7", "A": "7", "R": "24", "S": "44"}

    @pytest.mark.parametrize("bad, why", [
        ("E7-R1-A4-S44", "in that order"),
        ("E7-A4-R1", "is not E<n>-A<n>-R<n>-S<n>"),
        ("E7-A4-R1-S44@100M+S44", "restates S44"),              # a continuation keeps its name
        ("E7-A4-R1-S44@100M+R2-S44", "restates S44"),
        ("E7-A4-R1-S44@100M+S45-R2", "E, A, R, S order"),
        ("E7-A4-R1-S44@200M+S45@100M+R2", "not after the last branch point"),
        ("E7-A4-R1-S44@100+S45", "absolute steps in millions"),
        ("E7-A4-R1-S44@100M+(S45,S46)", "no group or range"),     # a soup names a model, not a run
        ("E7-A4-R1-S44@4720..4800M", "no group or range"),
        ("E7-A4-R1-S44-2@100M+S45", "a replicate index ends a name"),
    ])
    def test_refused(self, bad: str, why: str) -> None:
        with pytest.raises(RunNameError, match=why):
            parse_run_name(bad)

    def test_legacy_names_stay_valid(self) -> None:
        for n in ("E7-20-44", "E7-20-44-4390M.45", "E4-08-01-2", "E4-17-06-5M.11-90M.07"):
            assert is_run_name(n)

    def test_train_accepts_a_new_name_as_its_directory(self) -> None:
        got = _resolve_run_dir(None, "E7-A4-R1-S44@4390M+S45", "v2", "20261007_000000")
        assert got.endswith("E7-A4-R1-S44@4390M+S45_20261007_000000")


class TestLabels:
    def test_the_owners_example(self) -> None:
        lab = parse_label("E7-A2-R2-S5@800M+R3@1200M+(S6,R4-S7)@1500..1600M")
        assert [r.text() for r in lab.ingredients] == [
            "E7-A2-R2-S5@800M+R3@1200M+S6", "E7-A2-R2-S5@800M+R3@1200M+R4-S7"]
        assert lab.selector is not None and (lab.selector.lo, lab.selector.hi) == ("1500M", "1600M")

    def test_a_restated_item_is_the_parents_own_continuation(self) -> None:
        assert _ingredients("E7-A4-R1-S44@4390M+(S44,45,46)@4800M") == [
            "E7-A4-R1-S44", "E7-A4-R1-S44@4390M+S45", "E7-A4-R1-S44@4390M+S46"]

    def test_recipes_and_seeds_mix_in_one_group(self) -> None:
        """The first shallow soup: the parent, a seed branch and two recipe branches of 870M."""
        assert _ingredients("E7-A4-R1-S44@870M+(S44,S45,R8,R9)@1200M") == [
            "E7-A4-R1-S44", "E7-A4-R1-S44@870M+S45", "E7-A4-R1-S44@870M+R8",
            "E7-A4-R1-S44@870M+R9"]

    def test_groups_nest(self) -> None:
        assert _ingredients("E7-A4-R1-S44@100M+(S45@200M+(R2,R3),R4)@300M") == [
            "E7-A4-R1-S44@100M+S45@200M+R2", "E7-A4-R1-S44@100M+S45@200M+R3",
            "E7-A4-R1-S44@100M+R4"]

    def test_a_top_level_group_lists_whole_runs(self) -> None:
        lab = parse_label("(E7-A4-R1-S44,E7-A4-R10-S44)@1200M")
        assert [r.text() for r in lab.ingredients] == ["E7-A4-R1-S44", "E7-A4-R10-S44"]
        assert not shares_a_state(lab)          # independent runs: averaging them is suspect
        assert shares_a_state(parse_label("E7-A4-R1-S44@4390M+(S44,45)@4800M"))

    @pytest.mark.parametrize("bad, why", [
        ("(E7-A4-R1-S44,E7-A8-R1-S44)@1200M", "cannot be averaged"),
        ("(E6-A4-R1-S44,E7-A4-R1-S44)@1200M", "across engines"),
        ("E7-A4-R1-S44@4390M+(S45,S45)@4800M", "twice"),
        ("E7-A4-R1-S44@4390M+(,S45)@4800M", "empty group item"),
        ("E7-A4-R1-S44@4390M+(R1-S45,S46)@4800M", "restates"),
        ("E7-A4-R1-S44@4800..4720M", "empty"),
        ("E7-A4-R1-S44@4390M+(S45,S46", "unbalanced"),
    ])
    def test_refused(self, bad: str, why: str) -> None:
        with pytest.raises(RunNameError, match=why):
            parse_label(bad)


class TestCheckpointLabels:
    def test_a_new_name_directory_labels_its_snapshots(self) -> None:
        assert checkpoint_label(
            "/d/E7-A4-R1-S44@4390M+S45_20261006_074128/snapshot_4800000000steps.pt"
        ) == "E7-A4-R1-S44@4390M+S45@4800M"

    def test_the_label_parses_back_to_its_run(self) -> None:
        lab = parse_label(checkpoint_label(
            "/d/E7-A4-R1-S44@4390M+S45_20261006_074128/snapshot_4800000000steps.pt"))
        assert [r.text() for r in lab.ingredients] == ["E7-A4-R1-S44@4390M+S45"]

    def test_an_swa_beside_the_snapshots_is_labelled_by_its_range(self) -> None:
        assert checkpoint_label("/d/E7-A8-R1-S44_20261007_092825/swa_1120-1200M.pt") \
            == "E7-A8-R1-S44@1120..1200M"

    def test_a_model_file_named_by_its_label_keeps_it(self) -> None:
        assert checkpoint_label("/d/_models/E7-A4-R1-S44@4390M+(S44,45,46)@4800M.pt") \
            == "E7-A4-R1-S44@4390M+(S44,45,46)@4800M"
