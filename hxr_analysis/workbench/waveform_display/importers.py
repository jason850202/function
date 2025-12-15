from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

from .models import Payload, create_waveform_payload, payload_id_from_path
from .np_compat import np

Importer = Callable[[Path], Tuple[str, Payload]]


class ImportErrorWithContext(Exception):
    pass


def load_npz_payload(path: Path) -> Tuple[str, Payload]:
    path = path.expanduser().resolve()
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ImportErrorWithContext(f"Failed to load {path}: {exc}") from exc

    payload: Payload
    if "payload" in data:
        payload_obj = data["payload"].item()
        if not isinstance(payload_obj, dict):
            raise ImportErrorWithContext(f"payload entry in {path} is not a dict")
        payload = payload_obj
    else:
        keys = set(data.files)
        time_key = None
        for candidate in ("time", "t"):
            if candidate in keys:
                time_key = candidate
                break
        if not time_key:
            raise ImportErrorWithContext(
                f"{path} missing 'time' or 't' array to build waveform payload"
            )
        time = data[time_key]
        channel_names = [k for k in data.files if k != time_key]
        channels = {name: data[name] for name in channel_names}
        payload = create_waveform_payload(time=time, channels=channels, meta={"source": str(path)})

    payload_id = payload_id_from_path(path)
    return payload_id, payload


_IMPORTERS: Dict[str, Importer] = {
    ".npz": load_npz_payload,
}


def import_files(paths: Iterable[str | Path]):
    payloads: Dict[str, Payload] = {}
    for p in paths:
        path_obj = Path(p)
        ext = path_obj.suffix.lower()
        if ext not in _IMPORTERS:
            raise ImportErrorWithContext(f"Unsupported file type: {ext}")
        payload_id, payload = _IMPORTERS[ext](path_obj)
        payloads[payload_id] = payload
    return payloads


__all__ = ["import_files", "load_npz_payload", "ImportErrorWithContext"]
