from __future__ import annotations

import copy
import re
from typing import Any, Dict, List

from hxr_analysis.workbench.waveform_display.models import Payload
from hxr_analysis.workbench.waveform_display.path import PathResolutionError, resolve_path, _parse_path

from .errors import MergeCollisionError, SpecError, TransformerError
from .spec import IdSpec, MergeSpec, SplitSpec


_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_part(part: str) -> str:
    return _SANITIZE_PATTERN.sub("_", part).strip("_") or "part"


def compute_identifier(payload: Payload, id_spec: IdSpec) -> str:
    parts: List[str] = []
    for path in id_spec.id_paths:
        try:
            value = resolve_path(payload, path)
        except PathResolutionError:
            continue
        if value is None:
            continue
        parts.append(str(value))

    if not parts:
        parts.append(id_spec.fallback)

    if id_spec.sanitize:
        parts = [_sanitize_part(p) for p in parts]

    return id_spec.joiner.join(parts)


def _clone_with_set(source: Dict[str, Any], tokens: List[str], value: Any) -> Dict[str, Any]:
    if not tokens:
        if isinstance(value, dict):
            return dict(value)
        return value

    token = tokens[0]
    if not isinstance(source, dict):
        raise PathResolutionError(
            f"Cannot traverse into '{token}' because object is of type {type(source).__name__}, expected dict"
        )

    copied = dict(source)
    if token in source:
        copied[token] = _clone_with_set(source[token], tokens[1:], value)
    else:
        if len(tokens) == 1:
            copied[token] = dict(value) if isinstance(value, dict) else value
        else:
            copied[token] = _clone_with_set({}, tokens[1:], value)
    return copied


def _ensure_copy(payload: Payload, policy: str) -> Payload:
    if policy == "deep":
        return copy.deepcopy(payload)
    if policy == "shallow":
        return copy.copy(payload)
    raise SpecError(f"Unknown copy policy: {policy}")


def _ensure_dict(obj: Any, context: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TransformerError(f"Expected dict at {context}, found {type(obj).__name__}")
    return obj


def split_payload(payload: Payload, *, spec: SplitSpec) -> Dict[str, Payload]:
    if spec.split_mode not in {"dict_keys", "list_items"}:
        raise SpecError(f"Unsupported split mode: {spec.split_mode}")

    base_uid = compute_identifier(
        payload,
        spec.id_spec
        or IdSpec([
            "meta.__uid__",
            "meta.file",
            "meta.shot",
            "meta.scope",
        ]),
    )
    source = resolve_path(payload, spec.source_path)
    target_path = spec.child_payload_target_path or spec.source_path
    target_tokens = _parse_path(target_path)

    outputs: Dict[str, Payload] = {}
    seen_ids: Dict[str, int] = {}

    if spec.split_mode == "dict_keys":
        source_dict = _ensure_dict(source, spec.source_path)
        items = source_dict.items()
    else:
        if not isinstance(source, list):
            raise TransformerError(f"Expected list at {spec.source_path}, found {type(source).__name__}")
        items = enumerate(source)

    joiner = spec.id_spec.joiner if spec.id_spec else "__"

    for key, value in items:
        child_payload = _ensure_copy(payload, spec.copy_policy)
        child_key = spec.child_key_template.format(key=key, pid=base_uid)
        child_value_container = {child_key: value}

        child_payload = _clone_with_set(child_payload, target_tokens, child_value_container)
        child_id_base = f"{base_uid}{joiner}{key}"

        count = seen_ids.get(child_id_base, 0)
        child_id = child_id_base if count == 0 else f"{child_id_base}_{count}"
        seen_ids[child_id_base] = count + 1

        meta_tokens = _parse_path(spec.attach_id_to_meta_path)
        child_payload = _clone_with_set(child_payload, meta_tokens, child_id)
        outputs[child_id] = child_payload

    return outputs


def _compare_timebases(reference: Any, current: Any) -> bool:
    try:
        import numpy as np

        return np.array_equal(reference, current)
    except Exception:
        return reference == current


def merge_payloads(payloads: List[Payload], *, spec: MergeSpec) -> Payload:
    if not payloads:
        raise TransformerError("No payloads provided for merge")

    target_tokens = _parse_path(spec.target_path)
    source_map_tokens = _parse_path(spec.source_map_path)
    joiner = spec.id_spec.joiner if spec.id_spec else "__"

    merged_target: Dict[str, Any] = {}
    provenance: List[Dict[str, Any]] = []

    base_payload = copy.deepcopy(payloads[0])
    base_target = resolve_path(base_payload, spec.target_path)
    _ensure_dict(base_target, spec.target_path)

    time_ref = None
    if spec.require_same_timebase:
        time_ref = resolve_path(base_payload, spec.time_path)

    def _apply_collision(key: str, uid: str, existing: Dict[str, Any]) -> str:
        if spec.collision_policy == "error":
            if key in existing:
                raise MergeCollisionError(f"Collision on key '{key}' for uid '{uid}'")
            return key
        if spec.collision_policy == "attach_id":
            new_key = spec.collision_template.format(key=key, uid=uid)
            if new_key in existing:
                raise MergeCollisionError(f"Collision after attachment for key '{key}' and uid '{uid}'")
            return new_key
        if spec.collision_policy == "overwrite":
            return key
        if spec.collision_policy == "suffix_counter":
            counter = 1
            new_key = key
            while new_key in existing:
                new_key = f"{key}{spec.collision_suffix_separator}{counter}"
                counter += 1
            return new_key
        raise SpecError(f"Unknown collision policy: {spec.collision_policy}")

    for payload in payloads:
        payload_uid = compute_identifier(
            payload,
            spec.id_spec
            or IdSpec([
                "meta.__uid__",
                "meta.file",
                "meta.shot",
                "meta.scope",
            ], joiner=joiner),
        )
        source = resolve_path(payload, spec.target_path)
        source_dict = _ensure_dict(source, spec.target_path)

        if spec.require_same_timebase:
            current_time = resolve_path(payload, spec.time_path)
            if not _compare_timebases(time_ref, current_time):
                raise TransformerError("Timebase mismatch across payloads")

        for key, value in source_dict.items():
            final_key = _apply_collision(key, payload_uid, merged_target)
            if spec.collision_policy == "overwrite" and final_key in merged_target:
                provenance.append({"uid": payload_uid, "original_key": key, "final_key": final_key})
                merged_target[final_key] = value
                continue

            if spec.merge_mode == "dict_union":
                merged_target[final_key] = value
            else:
                raise SpecError(f"Unsupported merge mode: {spec.merge_mode}")

            provenance.append({"uid": payload_uid, "original_key": key, "final_key": final_key})

    base_payload = _clone_with_set(base_payload, target_tokens, merged_target)
    try:
        base_payload = _clone_with_set(base_payload, source_map_tokens, provenance)
    except PathResolutionError:
        meta_root = base_payload.get("meta", {})
        if "meta" not in base_payload:
            base_payload["meta"] = meta_root
        base_payload = _clone_with_set(base_payload, source_map_tokens, provenance)

    return base_payload


__all__ = ["split_payload", "merge_payloads", "compute_identifier"]
