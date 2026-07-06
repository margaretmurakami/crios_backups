"""Binary readers for MITgcm-style big-endian arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DTYPES = {
    "float32": np.dtype(">f4"),
    "float64": np.dtype(">f8"),
}


def read_big_endian(path: str | Path, *, dtype: str = "float32", shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Read a big-endian binary array and optionally reshape it."""

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    data = np.fromfile(Path(path), dtype=DTYPES[dtype])
    if shape is not None:
        data = data.reshape(shape)
    return data


def read_record(path: str | Path, *, record_len: int, record_no: int, dtype: str = "float32") -> np.ndarray:
    """Read one fixed-size record from a big-endian binary file."""

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    dt = DTYPES[dtype]
    with Path(path).open("rb") as file:
        file.seek(record_len * dt.itemsize * record_no)
        raw = file.read(record_len * dt.itemsize)
    return np.frombuffer(raw, dtype=dt).copy()

