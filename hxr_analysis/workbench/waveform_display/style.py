from __future__ import annotations

import ast
import json
from typing import Any, Dict

import pyqtgraph as pg
from PyQt6.QtCore import Qt

DEFAULT_PEN = pg.mkPen("k", width=1)

STYLE_NAME_MAP = {
    "solid": Qt.PenStyle.SolidLine,
    "dash": Qt.PenStyle.DashLine,
    "dot": Qt.PenStyle.DotLine,
    "dashdot": Qt.PenStyle.DashDotLine,
}


def parse_style(style_text: str) -> Dict[str, Any]:
    text = (style_text or "").strip()
    if not text or text == "{}":
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not parse style: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Style must be a dictionary")
    return parsed


def _normalize_alpha(alpha: Any) -> int | None:
    if alpha is None:
        return None
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError):
        return None
    if 0 <= alpha_value <= 1:
        alpha_value *= 255
    alpha_value = max(0, min(255, int(alpha_value)))
    return alpha_value


def _build_color(color_value: Any, alpha: int | None):
    qcolor = pg.mkColor(color_value)
    if alpha is not None:
        qcolor.setAlpha(alpha)
    return qcolor


def style_to_pen(style_dict: Dict[str, Any]) -> pg.QtGui.QPen:
    if not style_dict:
        return DEFAULT_PEN

    pen_kwargs: Dict[str, Any] = {}

    alpha = _normalize_alpha(style_dict.get("alpha"))
    color_value = style_dict.get("color")
    if color_value is not None:
        pen_kwargs["color"] = _build_color(color_value, alpha)
    elif alpha is not None:
        pen_kwargs["color"] = _build_color(DEFAULT_PEN.color(), alpha)

    if style_dict.get("width") is not None:
        pen_kwargs["width"] = style_dict.get("width")

    style_name = style_dict.get("style")
    if isinstance(style_name, str):
        qt_style = STYLE_NAME_MAP.get(style_name.lower())
        if qt_style is not None:
            pen_kwargs["style"] = qt_style

    if not pen_kwargs:
        return DEFAULT_PEN

    return pg.mkPen(**pen_kwargs)


__all__ = ["DEFAULT_PEN", "parse_style", "style_to_pen"]
