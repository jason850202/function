from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal

from hxr_analysis.workbench.waveform_display.models import Payload
from hxr_analysis.workbench.waveform_display.np_compat import np
from hxr_analysis.workbench.waveform_display.path import PathResolutionError, _parse_path, resolve_path
from hxr_analysis.workbench.waveform_display.transform import IdSpec, compute_identifier


@dataclass
class BackgroundSubtractParams:
    time_path: str = "data.time"
    channels_path: str = "data.channels"
    match_mode: Literal["by_key", "by_index"] = "by_key"
    missing_channel_policy: Literal["skip", "error"] = "skip"
    bg_scale: float = 1.0
    exp_scale: float = 1.0
    time_align: Literal["require_equal", "interp_bg_to_exp"] = "require_equal"
    interp_kind: Literal["linear"] = "linear"
    result_channel_prefix: str = ""
    store_original: bool = True
    output_field: str = "data.channels"
    record_history: bool = True


def _ensure_dict(obj: Any, context: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict at {context}, got {type(obj).__name__}")
    return obj


def _allclose(a: Any, b: Any) -> bool:
    if hasattr(np, "allclose"):
        try:
            return bool(np.allclose(a, b))
        except Exception:
            pass
    try:
        return list(a) == list(b)
    except Exception:
        return False


def _interp(x: Any, xp: Any, fp: Any):
    if hasattr(np, "interp"):
        return np.interp(x, xp, fp)
    # simple linear interpolation fallback
    x_list = list(x)
    xp_list = list(xp)
    fp_list = list(fp)
    if len(xp_list) != len(fp_list):
        raise ValueError("xp and fp must have same length for interpolation")
    if len(xp_list) < 2:
        return fp_list[: len(x_list)]
    results = []
    for xv in x_list:
        if xv <= xp_list[0]:
            results.append(fp_list[0])
            continue
        if xv >= xp_list[-1]:
            results.append(fp_list[-1])
            continue
        for i in range(1, len(xp_list)):
            if xp_list[i] >= xv:
                x0, x1 = xp_list[i - 1], xp_list[i]
                y0, y1 = fp_list[i - 1], fp_list[i]
                if x1 == x0:
                    results.append(y0)
                else:
                    frac = (xv - x0) / (x1 - x0)
                    results.append(y0 + frac * (y1 - y0))
                break
    return np.asarray(results) if hasattr(np, "asarray") else results


def _clone_with_set(source: Dict[str, Any], tokens: List[str], value: Any) -> Dict[str, Any]:
    new_payload = copy.deepcopy(source)
    current: Any = new_payload
    for token in tokens[:-1]:
        if not isinstance(current, dict):
            raise PathResolutionError(
                f"Cannot traverse into '{token}' because object is of type {type(current).__name__}, expected dict"
            )
        if token not in current:
            current[token] = {}
        elif not isinstance(current[token], dict):
            raise PathResolutionError(
                f"Cannot traverse into '{token}' because object is of type {type(current[token]).__name__}, expected dict"
            )
        current = current[token]
    last = tokens[-1]
    if not isinstance(current, dict):
        raise PathResolutionError(
            f"Cannot set '{last}' because parent object is of type {type(current).__name__}, expected dict"
        )
    current[last] = value
    return new_payload


def _ensure_uid(payload: Payload) -> str:
    meta = payload.setdefault("meta", {})
    if "__uid__" not in meta:
        meta["__uid__"] = compute_identifier(payload, IdSpec(["meta.file", "meta.shot", "meta.scope"]))
    return meta["__uid__"]


def _scaled_subtract(exp_values: Any, bg_values: Any, exp_scale: float, bg_scale: float):
    exp_arr = np.asarray(exp_values)
    bg_arr = np.asarray(bg_values)
    try:
        return exp_scale * exp_arr - bg_scale * bg_arr
    except Exception:
        exp_list = [exp_scale * float(v) for v in exp_arr]
        bg_list = [bg_scale * float(v) for v in bg_arr]
        result = [e - b for e, b in zip(exp_list, bg_list)]
        return np.asarray(result) if hasattr(np, "asarray") else result


def background_subtract_one(exp_payload: Payload, bg_payload: Payload, *, params: BackgroundSubtractParams) -> Payload:
    try:
        t_exp = np.asarray(resolve_path(exp_payload, params.time_path))
        ch_exp = resolve_path(exp_payload, params.channels_path)
    except PathResolutionError as exc:
        raise ValueError(f"Experiment payload missing required data: {exc}") from exc

    try:
        t_bg = np.asarray(resolve_path(bg_payload, params.time_path))
        ch_bg = resolve_path(bg_payload, params.channels_path)
    except PathResolutionError as exc:
        raise ValueError(f"Background payload missing required data: {exc}") from exc

    ch_exp = _ensure_dict(ch_exp, params.channels_path)
    ch_bg = _ensure_dict(ch_bg, params.channels_path)

    if params.time_align == "require_equal":
        if len(t_exp) != len(t_bg) or not _allclose(t_exp, t_bg):
            raise ValueError("Timebases do not match between experiment and background payloads")
    elif params.time_align == "interp_bg_to_exp":
        pass
    else:
        raise ValueError(f"Unknown time_align: {params.time_align}")

    bg_uid = bg_payload.get("meta", {}).get("__uid__") or compute_identifier(
        bg_payload, IdSpec(["meta.file", "meta.shot", "meta.scope"])
    )

    result_channels: Dict[str, Any] = {}
    skipped: List[str] = []

    bg_items = list(ch_bg.items())

    for idx, (exp_key, exp_values) in enumerate(ch_exp.items()):
        match_key: str | None
        if params.match_mode == "by_key":
            match_key = exp_key
        elif params.match_mode == "by_index":
            match_key = bg_items[idx][0] if idx < len(bg_items) else None
        else:
            raise ValueError(f"Unknown match_mode: {params.match_mode}")

        if match_key is None or match_key not in ch_bg:
            if params.missing_channel_policy == "error":
                raise ValueError(f"Missing background channel for experiment channel '{exp_key}'")
            skipped.append(exp_key)
            out_key = f"{params.result_channel_prefix}{exp_key}"
            result_channels[out_key] = copy.deepcopy(exp_values)
            continue

        bg_values = ch_bg[match_key]
        bg_values_arr = np.asarray(bg_values)
        if params.time_align == "interp_bg_to_exp":
            bg_values_arr = _interp(t_exp, t_bg, bg_values_arr)
        y_out = _scaled_subtract(exp_values, bg_values_arr, params.exp_scale, params.bg_scale)
        out_key = f"{params.result_channel_prefix}{exp_key}"
        result_channels[out_key] = y_out

    new_payload = _clone_with_set(exp_payload, _parse_path(params.output_field), result_channels)

    if params.store_original:
        original = {"time": copy.deepcopy(t_exp), "channels": copy.deepcopy(ch_exp)}
        meta_tokens = _parse_path("meta.__original__")
        try:
            new_payload = _clone_with_set(new_payload, meta_tokens, original)
        except PathResolutionError:
            meta = new_payload.setdefault("meta", {})
            meta.setdefault("__original__", original)

    meta = new_payload.setdefault("meta", {})
    out_uid = _ensure_uid(new_payload)
    meta["__background__"] = {
        "uid": bg_uid,
        "source": bg_payload.get("meta", {}).get("source"),
        "channels_skipped": skipped,
    }

    if params.record_history:
        history = meta.setdefault("__history__", [])
        history.append(
            {
                "op_name": "background_subtract",
                "params": asdict(params),
                "background_uid": bg_uid,
                "timestamp": datetime.utcnow().isoformat(),
                "output_uid": out_uid,
            }
        )

    return new_payload


def background_subtract_many(
    exp_payloads: List[Payload], bg_payload: Payload, *, params: BackgroundSubtractParams
) -> List[Payload]:
    return [background_subtract_one(exp, bg_payload, params=params) for exp in exp_payloads]


__all__ = [
    "BackgroundSubtractParams",
    "background_subtract_one",
    "background_subtract_many",
]
