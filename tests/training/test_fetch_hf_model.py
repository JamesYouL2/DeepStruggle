"""tools/fetch_hf_model.py's `torch_twin`: the leak probe's CI plan plays the newest export's torch
checkpoint, and must find it or refuse -- never another network."""

from __future__ import annotations

from tools.fetch_hf_model import torch_twin


def _entries(*paths: str) -> list:
    return [{"type": "file", "path": p} for p in paths] + [{"type": "directory", "path": "runs"}]


def test_the_twin_beside_the_export_wins() -> None:
    e = _entries("runs/A@1M.onnx", "runs/A@1M.pt", "other/A@1M.pt")
    assert torch_twin("runs/A@1M.onnx", e) == "runs/A@1M.pt"


def test_a_twin_elsewhere_under_the_same_name_is_found() -> None:
    e = _entries("E7line_swa_4720-4800M.onnx", "pt/E7line_swa_4720-4800M.pt")
    assert torch_twin("E7line_swa_4720-4800M.onnx", e) == "pt/E7line_swa_4720-4800M.pt"


def test_no_twin_is_none_not_a_lookalike() -> None:
    e = _entries("E7line_swa_4720-4800M.onnx", "E7line_swa_4720-4800M_old.pt", "C2_soup+A.pt")
    assert torch_twin("E7line_swa_4720-4800M.onnx", e) is None
