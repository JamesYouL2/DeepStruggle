"""Read an uncompressed `.npz` without loading it: every member memory-mapped, read-only.

`np.load(path, mmap_mode="r")` does not do this -- for an `.npz` NumPy ignores `mmap_mode`, and
indexing the returned `NpzFile` reads the whole member into memory each time. The fit datasets
(`tools/distill_targets.py`, `tools/value_labels.py`) are written by `np.savez`, which stores each
member uncompressed, so a member's array data sits contiguously in the zip and can be mapped in
place. The fits then read only what they touch, chunk by chunk (`Codes.encode` reads the raw
states 65,536 rows at a time), instead of holding every member, used or not, in memory at once
(key invariant 8). The values are the same arrays `np.load` returns, so results are unchanged.
"""

from __future__ import annotations

import os
import struct
import zipfile
from typing import BinaryIO, Dict, Tuple, Union

import numpy as np

_LOCAL_HEADER = 30
_LOCAL_MAGIC = b"PK\x03\x04"


def _array_header(f: BinaryIO, version: Tuple[int, int]) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    raise ValueError(f"unsupported .npy format version {version}")


def load_npz_mmap(path: Union[str, "os.PathLike[str]"]) -> Dict[str, np.ndarray]:
    """Every array in the `.npz` at `path`, keyed as `np.load` keys them, memory-mapped read-only.

    Refuses (rather than silently loading) a compressed member (`np.savez_compressed`), an object
    array, or anything that is not a `.npy` member: none of those can be mapped.
    """
    fspath = os.fspath(path)
    out: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(fspath) as zf, open(fspath, "rb") as f:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                raise ValueError(f"{fspath}: member {info.filename!r} is not a .npy array")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{fspath}: member {info.filename!r} is compressed and cannot be memory-mapped; "
                                 f"write the dataset with np.savez, not np.savez_compressed")
            f.seek(info.header_offset)
            local = f.read(_LOCAL_HEADER)
            if local[:4] != _LOCAL_MAGIC:
                raise ValueError(f"{fspath}: no zip local header at member {info.filename!r}")
            name_len, extra_len = struct.unpack("<HH", local[26:30])
            f.seek(info.header_offset + _LOCAL_HEADER + name_len + extra_len)
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = _array_header(f, version)
            if dtype.hasobject:
                raise ValueError(f"{fspath}: member {info.filename!r} is an object array and cannot be memory-mapped")
            key = info.filename[:-len(".npy")]
            if int(np.prod(shape, dtype=np.int64)) == 0:
                out[key] = np.zeros(shape, dtype=dtype, order="F" if fortran else "C")
                continue
            out[key] = np.memmap(fspath, dtype=dtype, mode="r", shape=shape, order="F" if fortran else "C",
                                 offset=f.tell())
    return out
