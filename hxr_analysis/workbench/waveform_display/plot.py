from __future__ import annotations

from typing import Iterable

import pyqtgraph as pg

from .mapping import ResolvedMapping


def render_mappings(plot_widget: pg.PlotWidget, mappings: Iterable[ResolvedMapping]) -> None:
    plot_widget.clear()
    for rm in mappings:
        kwargs = dict(rm.style)
        if rm.mapping.plot_type == "scatter":
            kwargs.setdefault("pen", None)
            kwargs.setdefault("symbol", "o")
        plot_widget.plot(rm.x, rm.y, **kwargs)


__all__ = ["render_mappings"]
