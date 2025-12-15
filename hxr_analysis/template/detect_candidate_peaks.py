from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple
import math
import statistics

from hxr_analysis.workbench.waveform_display.np_compat import np
from hxr_analysis.workbench.waveform_display.path import PathResolutionError, resolve_path
from hxr_analysis.workbench.waveform_display.transform import IdSpec, compute_identifier
from hxr_analysis.workbench.waveform_display.transform.errors import SpecError


@dataclass
class DetectCandidatePeaksParams:
    time_path: str = "data.time"
    channels_path: str = "data.channels"
    channel_keys: list[str] | None = None
    polarity: Literal["preserve", "invert", "auto", "normalized"] = "normalized"
    noise_method: Literal["rms", "mad", "std_pretrigger"] = "mad"
    pretrigger_time_range: tuple[float, float] | None = None
    threshold_sigma: float = 5.0
    min_distance_samples: int = 20
    min_width_samples: int = 1
    max_peaks_per_channel: int | None = None
    reject_saturated: bool = False
    saturation_level: float | None = None
    store_regions: bool = True
    store_snr: bool = True
def _ensure_uid(payload: Dict[str, Any]) -> str:
    meta = payload.setdefault("meta", {})
    if "__uid__" not in meta:
        meta["__uid__"] = compute_identifier(payload, IdSpec(["meta.file", "meta.shot", "meta.scope"]))
    return meta["__uid__"]


def _to_list(values: Any) -> List[float]:
    try:
        return [float(v) for v in np.asarray(values)]
    except Exception:
        return [float(v) for v in values]


def _median(values: List[float]) -> float:
    if hasattr(np, "median"):
        try:
            return float(np.median(values))
        except Exception:
            pass
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: List[float], q: float) -> float:
    if hasattr(np, "percentile"):
        try:
            return float(np.percentile(values, q))
        except Exception:
            pass
    if not values:
        return 0.0
    data = sorted(values)
    k = (len(data) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return float(d0 + d1)


def _estimate_polarity(y: Any) -> int:
    arr = _to_list(y)
    median = _median(arr)
    p99 = _percentile(arr, 99)
    p1 = _percentile(arr, 1)
    pos_peakiness = p99 - median
    neg_peakiness = median - p1
    return 1 if pos_peakiness >= neg_peakiness else -1


def _estimate_noise(y: Any, method: str, *, t: Any, pre_range: Tuple[float, float] | None) -> float:
    arr = _to_list(y)
    if method == "mad":
        med = _median(arr)
        deviations = [abs(v - med) for v in arr]
        noise = 1.4826 * _median(deviations)
    elif method == "rms":
        mean = sum(arr) / len(arr) if arr else 0.0
        noise = math.sqrt(sum((v - mean) ** 2 for v in arr) / len(arr)) if arr else 0.0
    elif method == "std_pretrigger":
        if pre_range is None:
            raise ValueError("pretrigger_time_range must be set for std_pretrigger noise estimation")
        t_arr = _to_list(t)
        subset = [val for val, time_val in zip(arr, t_arr, strict=False) if pre_range[0] <= time_val <= pre_range[1]]
        if not subset:
            noise = 0.0
        else:
            mean = sum(subset) / len(subset)
            noise = math.sqrt(sum((v - mean) ** 2 for v in subset) / len(subset))
    else:
        raise SpecError(f"Unsupported noise estimation method: {method}")
    try:
        return float(noise)
    except Exception:
        return 0.0


def _to_array(values: List[Any], dtype: Any | None = None):
    try:
        return np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    except TypeError:
        return np.asarray(values)


def _find_regions(mask: Any) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    in_region = False
    start = 0
    for idx, flag in enumerate(mask):
        if flag and not in_region:
            start = idx
            in_region = True
        elif not flag and in_region:
            regions.append((start, idx))
            in_region = False
    if in_region:
        regions.append((start, len(mask)))
    return regions


def _apply_deadtime(peaks: List[Dict[str, Any]], min_distance: int) -> List[Dict[str, Any]]:
    if not peaks:
        return []
    peaks_sorted = sorted(peaks, key=lambda p: p["i"])
    filtered: List[Dict[str, Any]] = []
    i = 0
    n = len(peaks_sorted)
    while i < n:
        current = peaks_sorted[i]
        j = i + 1
        while j < n and peaks_sorted[j]["i"] - current["i"] < min_distance:
            if peaks_sorted[j]["amp"] > current["amp"]:
                current = peaks_sorted[j]
            j += 1
        filtered.append(current)
        i = j
    return filtered


def detect_candidate_peaks(payload: Dict[str, Any], params: DetectCandidatePeaksParams | None = None) -> Dict[str, Any]:
    params = params or DetectCandidatePeaksParams()

    try:
        t_values = _to_list(resolve_path(payload, params.time_path))
    except PathResolutionError as exc:
        raise SpecError(f"Time path resolution failed: {exc}") from exc

    try:
        channels = resolve_path(payload, params.channels_path)
    except PathResolutionError as exc:
        raise SpecError(f"Channels path resolution failed: {exc}") from exc

    if not isinstance(channels, dict):
        raise ValueError(f"Expected channels to be a dict at {params.channels_path}")

    channel_keys = params.channel_keys or list(channels.keys())

    by_channel: Dict[str, Dict[str, Any]] = {}

    for key in channel_keys:
        if key not in channels:
            continue
        y_raw = _to_list(channels[key])
        if len(y_raw) != len(t_values):
            raise ValueError(
                f"Channel '{key}' length {len(y_raw)} does not match time length {len(t_values)}"
            )

        if params.polarity == "invert":
            y = [-val for val in y_raw]
        elif params.polarity == "preserve":
            y = y_raw
        elif params.polarity == "auto":
            sign = _estimate_polarity(y_raw)
            y = [sign * val for val in y_raw]
        else:  # normalized
            y = y_raw

        noise = _estimate_noise(y, params.noise_method, t=t_values, pre_range=params.pretrigger_time_range)
        if noise <= 0 or noise != noise:  # nan check
            threshold = float("inf")
        else:
            threshold = params.threshold_sigma * noise

        mask = [val > threshold for val in y]
        regions = _find_regions(mask)
        peaks: List[Dict[str, Any]] = []
        for start, end in regions:
            if end - start < params.min_width_samples:
                continue
            local_slice = y[start:end]
            if len(local_slice) == 0:
                continue
            if hasattr(np, "argmax"):
                try:
                    local_max_idx = int(np.argmax(local_slice))
                except Exception:
                    local_max_idx = max(range(len(local_slice)), key=lambda i: local_slice[i])
            else:
                local_max_idx = max(range(len(local_slice)), key=lambda i: local_slice[i])
            i_peak = start + local_max_idx
            amp = float(y[i_peak])
            peak_info: Dict[str, Any] = {
                "i": int(i_peak),
                "t": float(t_values[i_peak]),
                "amp": amp,
            }
            if params.store_regions:
                peak_info["region_start"] = int(start)
                peak_info["region_end"] = int(end)
            if params.store_snr:
                peak_info["snr"] = float(amp / noise) if noise > 0 else float("inf")
            peaks.append(peak_info)

        peaks = _apply_deadtime(peaks, params.min_distance_samples)

        if params.reject_saturated and params.saturation_level is not None:
            peaks = [p for p in peaks if p["amp"] < params.saturation_level]

        if params.max_peaks_per_channel is not None:
            peaks = sorted(peaks, key=lambda p: p["amp"], reverse=True)[: params.max_peaks_per_channel]
            peaks.sort(key=lambda p: p["i"])

        by_channel[key] = {
            "i": _to_array([p["i"] for p in peaks], dtype=int),
            "t": _to_array([p["t"] for p in peaks], dtype=float),
            "amp": _to_array([p["amp"] for p in peaks], dtype=float),
            "noise": noise,
            "threshold": threshold,
        }

        if params.store_snr:
            by_channel[key]["snr"] = _to_array([p.get("snr", 0.0) for p in peaks], dtype=float)
        if params.store_regions:
            by_channel[key]["region_start"] = _to_array([p.get("region_start", 0) for p in peaks], dtype=int)
            by_channel[key]["region_end"] = _to_array([p.get("region_end", 0) for p in peaks], dtype=int)

    new_payload = copy.deepcopy(payload)
    events = new_payload.setdefault("events", {})
    events["candidate_peaks"] = {"by_channel": by_channel, "params": asdict(params)}

    meta = new_payload.setdefault("meta", {})
    output_uid = _ensure_uid(new_payload)
    history = meta.setdefault("__history__", [])
    history.append(
        {
            "op_name": "detect_candidate_peaks",
            "params": asdict(params),
            "timestamp": datetime.utcnow().isoformat(),
            "output_uid": output_uid,
        }
    )

    return new_payload


__all__ = ["DetectCandidatePeaksParams", "detect_candidate_peaks"]
