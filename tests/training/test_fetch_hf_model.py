"""tools/fetch_hf_model.py picks the model the workbench picks, without the network."""
from __future__ import annotations

from typing import Dict, List

from tools.fetch_hf_model import order_newest_first


def test_newest_onnx_first_ties_by_path_and_other_files_ignored() -> None:
    entries: List[Dict[str, object]] = [
        {"type": "file", "path": "README.md", "lastCommit": {"date": "2026-09-30T00:00:00.000Z"}},
        {"type": "directory", "path": "old.onnx"},
        {"type": "file", "path": "E4-08-36_240M.onnx", "lastCommit": {"date": "2026-09-25T19:46:55.000Z"}},
        {"type": "file", "path": "b.onnx", "lastCommit": {"date": "2026-09-27T21:32:24.000Z"}},
        {"type": "file", "path": "a.onnx", "lastCommit": {"date": "2026-09-27T21:32:24.000Z"}},
        {"type": "file", "path": "undated.onnx"},
    ]
    assert [p for p, _ in order_newest_first(entries)] == [
        "a.onnx", "b.onnx", "E4-08-36_240M.onnx", "undated.onnx"]
