"""The workbench's search grammar (web/ui/src/analysis/rollout_spec.ts): one spelling per search."""
from pathlib import Path

from tests.web.js_runner import run_ts


def test_keyed2_is_one_search_however_it_is_spelled(tmp_path: Path) -> None:
    assert run_ts("rollout_spec.ts", str(tmp_path)) == {"canonical": True}
