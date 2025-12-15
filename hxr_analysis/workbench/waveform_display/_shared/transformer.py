from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from hxr_analysis.workbench.waveform_display.models import Payload
from hxr_analysis.workbench.waveform_display.transform import (
    IdSpec,
    compute_identifier,
    merge_payloads,
    split_payload,
)


def standardize_payload_meta(payload: Payload, op_name: str | None = None, params: Dict[str, Any] | None = None) -> Payload:
    meta = payload.setdefault("meta", {})

    if "__uid__" not in meta:
        meta["__uid__"] = compute_identifier(payload, IdSpec(["meta.file", "meta.shot", "meta.scope"]))

    history = meta.setdefault("__history__", [])
    if op_name:
        history.append({"op_name": op_name, "params": params or {}, "timestamp": datetime.utcnow().isoformat()})

    return payload


__all__ = ["split_payload", "merge_payloads", "compute_identifier", "standardize_payload_meta"]
