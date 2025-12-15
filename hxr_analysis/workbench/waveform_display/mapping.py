from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .models import Payload, PayloadStore
from .np_compat import np
from .path import PathResolutionError, resolve_path


class MappingValidationError(Exception):
    pass


@dataclass
class PlotMapping:
    plot_type: str = "curve"
    x_path: str = ""
    y_path: str = ""
    value_path: str | None = None
    style: str = "{}"
    payload_id: str = ""

    def style_dict(self) -> Dict[str, Any]:
        if not self.style:
            return {}
        try:
            return json.loads(self.style)
        except json.JSONDecodeError as exc:
            raise MappingValidationError(f"Invalid style JSON: {exc}")


@dataclass
class ResolvedMapping:
    mapping: PlotMapping
    x: np.ndarray
    y: np.ndarray
    value: Optional[np.ndarray]
    style: Dict[str, Any] = field(default_factory=dict)


SUPPORTED_TYPES = {"curve", "scatter"}


def validate_and_resolve(mapping: PlotMapping, store: PayloadStore) -> ResolvedMapping:
    if mapping.plot_type not in SUPPORTED_TYPES:
        raise MappingValidationError(f"Unsupported plot type: {mapping.plot_type}")
    if not mapping.payload_id:
        raise MappingValidationError("Mapping missing payload id")

    try:
        payload = store.get(mapping.payload_id)
    except KeyError as exc:
        raise MappingValidationError(str(exc)) from exc

    try:
        x_arr = np.asarray(resolve_path(payload, mapping.x_path))
    except PathResolutionError as exc:
        raise MappingValidationError(f"X path error: {exc}") from exc
    try:
        y_arr = np.asarray(resolve_path(payload, mapping.y_path))
    except PathResolutionError as exc:
        raise MappingValidationError(f"Y path error: {exc}") from exc

    val_arr = None
    if mapping.value_path:
        try:
            val_arr = np.asarray(resolve_path(payload, mapping.value_path))
        except PathResolutionError as exc:
            raise MappingValidationError(f"Value path error: {exc}") from exc

    if x_arr.shape[0] != y_arr.shape[0]:
        raise MappingValidationError(
            f"X and Y lengths differ ({x_arr.shape[0]} vs {y_arr.shape[0]})"
        )

    style_dict = mapping.style_dict()
    return ResolvedMapping(mapping=mapping, x=x_arr, y=y_arr, value=val_arr, style=style_dict)


__all__ = ["PlotMapping", "ResolvedMapping", "validate_and_resolve", "MappingValidationError"]
