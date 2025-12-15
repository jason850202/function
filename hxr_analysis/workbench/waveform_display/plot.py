from __future__ import annotations

from typing import Iterable

import pyqtgraph as pg

from .mapping import ResolvedMapping
from .style import style_to_pen


def render_mappings(plot_widget: pg.PlotWidget, mappings: Iterable[ResolvedMapping]) -> None:
    plot_widget.clear()
    for idx, rm in enumerate(mappings, start=1):
        pen = style_to_pen(rm.style)
        if rm.mapping.plot_type == "scatter":
            brush_color = pen.color() if hasattr(pen, "color") else None
            plot_widget.plot(
                rm.x,
                rm.y,
                pen=pen,
                symbol="o",
                symbolPen=pen,
                symbolBrush=brush_color,
            )
        else:
            plot_widget.plot(rm.x, rm.y, pen=pen)
        print(f"Applied style to row {idx}: {rm.style}")


__all__ = ["render_mappings"]
