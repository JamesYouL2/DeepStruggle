"""tools/scripts/weight_soup.py: a uniform average of snapshots, refusing to mix architectures."""

from __future__ import annotations

import pytest
import torch

from tools.scripts.weight_soup import soup


def _save(tmp_path, name: str, w: float, steps: int, shape=(2, 2)) -> str:
    p = str(tmp_path / name)
    torch.save({"lin.weight": torch.full(shape, w), "bn.num_batches_tracked": torch.tensor(steps)}, p)
    return p


def test_floats_are_averaged_and_integer_buffers_come_from_the_last(tmp_path) -> None:
    paths = [_save(tmp_path, "a.pt", 1.0, 10), _save(tmp_path, "b.pt", 3.0, 20)]
    out = soup(paths)
    assert torch.allclose(out["lin.weight"], torch.full((2, 2), 2.0))
    assert int(out["bn.num_batches_tracked"]) == 20


def test_different_shapes_refuse_to_mix(tmp_path) -> None:
    paths = [_save(tmp_path, "a.pt", 1.0, 10), _save(tmp_path, "b.pt", 3.0, 20, shape=(3, 2))]
    with pytest.raises(ValueError):
        soup(paths)
