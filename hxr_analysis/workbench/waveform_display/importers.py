from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

from .models import Payload, create_waveform_payload, payload_id_from_path
from .np_compat import np

Importer = Callable[..., Tuple[str, Payload]]


class ImportErrorWithContext(Exception):
    pass


def load_npz_payload(path: Path, length_policy: str = "use_channels") -> Tuple[str, Payload]:
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

            def _is_metadata_key(key: str) -> bool:
                return key in metadata_keys or key in {"time", "t", "payload"}

            def _is_1d_array(value: Any) -> tuple[bool, int | None]:
                try:
                    arr = np.asarray(value)
                except Exception:
                    return False, None
                dims = getattr(arr, "ndim", None)
                if dims is None:
                    shape = getattr(arr, "shape", None)
                    try:
                        dims = len(shape) if shape is not None else None
                    except Exception:
                        dims = None
                if dims is not None and dims != 1:
                    return False, None
                try:
                    length = len(arr)
                except Exception:
                    return False, None
                return True, length

            candidate_channels = []
            candidate_lengths = []
            for key in data.files:
                if _is_metadata_key(key):
                    continue
                ok, length = _is_1d_array(data[key])
                if ok:
                    candidate_channels.append((key, data[key], length))
                    candidate_lengths.append(length)

            if not candidate_channels:
                raise ImportErrorWithContext(
                    f"{path} contains Tstart/Tinterval but no valid 1D channel arrays. Keys: {sorted(keys)}"
                )

            length_value = data.get("Length") if "Length" in keys else None
            n_declared = None
            if length_value is not None:
                scalar_length = _extract_scalar(length_value)
                try:
                    n_candidate = int(scalar_length)
                    if n_candidate > 0:
                        n_declared = n_candidate
                except Exception:
                    n_declared = None

            length_counts = Counter(candidate_lengths)
            n_channels = length_counts.most_common(1)[0][0] if length_counts else None

            if n_channels is None:
                raise ImportErrorWithContext(
                    f"{path} contains Tstart/Tinterval but cannot infer channel length. Keys: {sorted(keys)}"
                )

            n = n_channels
            meta: Dict[str, Any] = {"source": str(path)}
            if n_declared is not None:
                if length_policy == "truncate_to_declared":
                    n = n_declared
                elif n_declared == n_channels:
                    n = n_declared
                else:
                    n = n_channels
                if n_declared != n_channels:
                    meta["length_mismatch"] = {
                        "declared": n_declared,
                        "channels": n_channels,
                        "chosen": n,
                    }

            t_start = float(_extract_scalar(data["Tstart"]))
            dt = float(_extract_scalar(data["Tinterval"]))
            if hasattr(np, "arange"):
                time = t_start + np.arange(n, dtype=float) * dt
            else:
                time = np.asarray([t_start + i * dt for i in range(n)])

            preferred_keys = [
                key for key, _, _ in candidate_channels if re.fullmatch(r"[A-Z]", key)
            ]
            selected_candidates = (
                [c for c in candidate_channels if c[0] in preferred_keys]
                if preferred_keys
                else candidate_channels
            )

            channels = {}
            truncated = []
            dropped = []
            for key, value, length in selected_candidates:
                if length < n:
                    dropped.append(key)
                    continue
                arr = np.asarray(value)
                if length > n:
                    arr = arr[:n]
                    truncated.append(key)
                channels[key] = arr

            for key in metadata_keys & keys:
                meta[key] = _extract_scalar(data[key])

            if truncated:
                meta["truncated_channels"] = truncated
            if dropped:
                meta["dropped_channels"] = dropped

            if not channels:
                raise ImportErrorWithContext(
                    f"PicoScope NPZ detected in {path} but no channels available after length validation."
                )

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
