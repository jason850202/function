from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

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
        if time_key:
            time = data[time_key]
            channel_names = [k for k in data.files if k != time_key]
            channels = {name: data[name] for name in channel_names}
            payload = create_waveform_payload(
                time=time, channels=channels, meta={"source": str(path)}
            )
        elif "Tstart" in keys and "Tinterval" in keys:
            metadata_keys = {
                "Tstart",
                "Tinterval",
                "ExtraSamples",
                "RequestedLength",
                "Length",
                "Version",
            }

            def _extract_scalar(value: Any) -> Any:
                try:
                    if getattr(value, "shape", None) == ():
                        return value.item()
                except Exception:
                    pass
                try:
                    if hasattr(value, "__len__") and len(value) == 1:
                        return _extract_scalar(value[0])
                except Exception:
                    pass
                try:
                    return value.item()  # type: ignore[call-arg]
                except Exception:
                    return value

            def _is_1d_array(value: Any) -> tuple[bool, int | None]:
                dims = None
                if hasattr(value, "ndim"):
                    dims = getattr(value, "ndim")
                elif hasattr(value, "shape"):
                    try:
                        dims = len(getattr(value, "shape"))
                    except Exception:
                        dims = None
                if dims is not None and dims != 1:
                    return False, None
                try:
                    length = len(value)
                except Exception:
                    return False, None
                return True, length

            def _is_metadata_key(key: str) -> bool:
                return key in metadata_keys or key in {"time", "t", "payload"}

            length_value = data.get("Length") if "Length" in keys else None
            n = None
            if length_value is not None:
                scalar_length = _extract_scalar(length_value)
                try:
                    n_candidate = int(scalar_length)
                    if n_candidate > 0:
                        n = n_candidate
                except Exception:
                    n = None

            if n is None:
                for key in data.files:
                    if _is_metadata_key(key):
                        continue
                    ok, length = _is_1d_array(data[key])
                    if ok:
                        n = length
                        break

            if n is None:
                raise ImportErrorWithContext(
                    "NPZ contains Tstart/Tinterval but cannot infer length; missing Length and no 1D channel arrays found."
                )

            t_start = _extract_scalar(data["Tstart"])
            t_interval = _extract_scalar(data["Tinterval"])
            time = np.asarray([t_start + i * t_interval for i in range(n)])

            channels = {}
            for key in data.files:
                if _is_metadata_key(key):
                    continue
                ok, length = _is_1d_array(data[key])
                if not ok or length != n:
                    continue
                if len(key) == 1 and key.isupper():
                    channels[key] = data[key]
                elif len(key) != 0:
                    channels[key] = data[key]

            meta = {"source": str(path)}
            for key in metadata_keys & keys:
                meta[key] = _extract_scalar(data[key])

            payload = create_waveform_payload(time=time, channels=channels, meta=meta)
        else:
            raise ImportErrorWithContext(
                f"{path} missing 'time'/'t' data and Tstart/Tinterval metadata to build waveform payload"
            )

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
