from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest

from hxr_analysis.workbench.waveform_display.importers import (
    ImportErrorWithContext,
    load_npz_payload,
)


def _prepare_array(value):
    if isinstance(value, (list, tuple)):
        data = [float(v) for v in value]
        shape = (len(data),)
    else:
        data = [float(value)]
        shape = ()
    return data, shape


def _build_npy_bytes(value):
    data, shape = _prepare_array(value)
    header_dict = f"{{'descr': '<f8', 'fortran_order': False, 'shape': {shape}, }}"
    header_bytes = header_dict.encode("latin1")
    padding = (16 - ((len(header_bytes) + 1 + 10) % 16)) % 16
    padded_header = header_bytes + b" " * padding + b"\n"
    header_len = len(padded_header)
    magic = b"\x93NUMPY"
    version = b"\x01\x00"
    prefix = magic + version + struct.pack("<H", header_len)
    data_bytes = struct.pack("<%dd" % len(data), *data)
    return prefix + padded_header + data_bytes


def create_npz(path: Path, arrays: dict[str, list[float] | float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for key, value in arrays.items():
            zf.writestr(f"{key}.npy", _build_npy_bytes(value))


def test_import_npz_with_t_array(tmp_path: Path):
    path = tmp_path / "with_t.npz"
    time = [0.0, 0.1, 0.2]
    create_npz(path, {"t": time, "A": [1, 2, 3], "B": [3, 2, 1]})

    payload_id, payload = load_npz_payload(path)

    assert payload_id == "with_t"
    assert list(payload["data"]["time"]) == time
    assert set(payload["data"]["channels"].keys()) == {"A", "B"}


def test_import_picoscope_builds_time_and_channels(tmp_path: Path):
    path = tmp_path / "picoscope.npz"
    create_npz(
        path,
        {
            "Tstart": 0.5,
            "Tinterval": 0.1,
            "Length": 3,
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "Version": 1,
        },
    )

    payload_id, payload = load_npz_payload(path)
    time = list(payload["data"]["time"])

    assert payload_id == "picoscope"
    assert time == pytest.approx([0.5, 0.6, 0.7])
    assert set(payload["data"]["channels"].keys()) == {"A", "B"}
    assert payload["meta"]["Tstart"] == pytest.approx(0.5)
    assert payload["meta"]["Tinterval"] == pytest.approx(0.1)


def test_picoscope_length_mismatch_keeps_channels_default(tmp_path: Path):
    path = tmp_path / "length_mismatch.npz"
    create_npz(
        path,
        {
            "Tstart": 0.0,
            "Tinterval": 0.5,
            "Length": 10,
            "A": list(range(14)),
        },
    )

    _, payload = load_npz_payload(path)

    assert len(payload["data"]["time"]) == 14
    assert len(payload["data"]["channels"]["A"]) == 14
    assert payload["meta"]["length_mismatch"] == {
        "declared": 10,
        "channels": 14,
        "chosen": 14,
    }


def test_picoscope_truncate_to_declared_length(tmp_path: Path):
    path = tmp_path / "truncate.npz"
    create_npz(
        path,
        {
            "Tstart": 0.0,
            "Tinterval": 0.5,
            "Length": 10,
            "A": list(range(14)),
        },
    )

    _, payload = load_npz_payload(path, length_policy="truncate_to_declared")

    assert len(payload["data"]["time"]) == 10
    assert len(payload["data"]["channels"]["A"]) == 10


def test_import_rejects_npz_without_time_information(tmp_path: Path):
    path = tmp_path / "no_time.npz"
    create_npz(path, {"A": [1, 2, 3]})

    with pytest.raises(ImportErrorWithContext):
        load_npz_payload(path)
