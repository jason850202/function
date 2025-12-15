from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

Payload = Dict[str, Any]


@dataclass
class PayloadStore:
    """In-memory registry for imported payloads."""

    payloads: Dict[str, Payload] = field(default_factory=dict)

    def add(self, payload_id: str, payload: Payload) -> None:
        self.payloads[payload_id] = payload

    def remove(self, payload_id: str) -> None:
        self.payloads.pop(payload_id, None)

    def clear(self) -> None:
        self.payloads.clear()

    def ids(self) -> Iterable[str]:
        return self.payloads.keys()

    def get(self, payload_id: str) -> Payload:
        if payload_id not in self.payloads:
            raise KeyError(f"Unknown payload id: {payload_id}")
        return self.payloads[payload_id]


def create_waveform_payload(
    time: Any,
    channels: Dict[str, Any],
    *,
    meta: Dict[str, Any] | None = None,
) -> Payload:
    return {
        "type": "waveform_bundle",
        "data": {"time": time, "channels": channels},
        "meta": meta or {},
    }


def payload_id_from_path(path: str | Path) -> str:
    return Path(path).stem
