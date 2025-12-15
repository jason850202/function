# npz_pyqtgraph_viewer.py
# PyQt6 + pyqtgraph NPZ viewer:
# - Load multiple npz
# - Select multiple channels
# - White background
# - Offset per channel
# - Color mode: by CHANNEL or by FILE (fixes same-color across files)
# - Rect zoom (left drag) or Pan toggle

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QListWidget, QFileDialog,
    QLabel, QGroupBox, QCheckBox,
    QMessageBox, QSplitter, QSizePolicy,
    QTableWidget, QTableWidgetItem, QColorDialog, QDoubleSpinBox,
    QComboBox
)

import pyqtgraph as pg


# -----------------------
# Helpers
# -----------------------

def load_npz_np(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def detect_channels(keys: List[str]) -> List[str]:
    ch = [k for k in keys if k.isalpha() and len(k) == 1 and k.upper() == k]
    return sorted(ch)


def make_time_axis(d: Dict[str, np.ndarray], y_len: int) -> Tuple[np.ndarray, str]:
    keys = set(d.keys())
    if {"Tstart", "Tinterval"}.issubset(keys):
        Tstart = float(np.array(d["Tstart"]).ravel()[0])
        Tinterval = float(np.array(d["Tinterval"]).ravel()[0])
        n = y_len
        if "Length" in keys:
            try:
                n = min(n, int(np.array(d["Length"]).ravel()[0]))
            except Exception:
                pass
        idx = np.arange(n, dtype=np.float64)
        x = Tstart + Tinterval * idx
        return x, "Time (s)"
    x = np.arange(y_len, dtype=np.float64)
    return x, "Sample index"


def decimate_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, int]:
    n = min(len(x), len(y))
    if n <= max_points:
        return x[:n], y[:n], 1
    step = int(np.ceil(n / max_points))
    return x[:n:step], y[:n:step], step


# -----------------------
# Channel checklist
# -----------------------

class ChannelChecklist(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)
        self._boxes: Dict[str, QCheckBox] = {}

    def set_channels(self, channels: List[str], default_checked: bool = True):
        for box in self._boxes.values():
            box.setParent(None)
        self._boxes.clear()

        for ch in channels:
            box = QCheckBox(ch)
            box.setChecked(default_checked)
            self.layout.addWidget(box)
            self._boxes[ch] = box

        self.layout.addStretch(1)

    def all_channels(self) -> List[str]:
        return list(self._boxes.keys())

    def checked_channels(self) -> List[str]:
        return [ch for ch, box in self._boxes.items() if box.isChecked()]

    def set_all_checked(self, checked: bool):
        for box in self._boxes.values():
            box.setChecked(checked)


# -----------------------
# Main GUI
# -----------------------

class NpzPyqtgraphViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NPZ Viewer (pyqtgraph)")
        self.resize(1150, 720)

        self.data: Dict[str, Dict[str, np.ndarray]] = {}
        self._style_table_connected = False

        # color cycles
        self._cycle = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ]

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ---------- Left panel ----------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # Files
        g_files = QGroupBox("NPZ files")
        g_files_layout = QVBoxLayout(g_files)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add…")
        self.btn_remove = QPushButton("Remove")
        self.btn_clear_files = QPushButton("Clear")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear_files)
        btn_row.addStretch(1)

        g_files_layout.addWidget(self.file_list)
        g_files_layout.addLayout(btn_row)

        # Channels
        g_ch = QGroupBox("Channels")
        g_ch_layout = QVBoxLayout(g_ch)
        self.channel_checklist = ChannelChecklist()
        g_ch_layout.addWidget(self.channel_checklist)

        # Trace styling table (per channel offset + optional per-channel color)
        g_style = QGroupBox("Trace styling")
        style_layout = QVBoxLayout(g_style)

        top_style_row = QHBoxLayout()
        top_style_row.addWidget(QLabel("Color mode:"))
        self.color_mode = QComboBox()
        self.color_mode.addItems(["By channel", "By file"])  # <- FIX
        top_style_row.addWidget(self.color_mode, 1)
        style_layout.addLayout(top_style_row)

        self.style_table = QTableWidget(0, 3)
        self.style_table.setHorizontalHeaderLabels(["Channel", "Color", "Offset"])
        self.style_table.verticalHeader().setVisible(False)
        self.style_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.style_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.style_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        style_layout.addWidget(self.style_table)

        # Plot controls
        g_ctl = QGroupBox("Plot controls")
        grid = QGridLayout(g_ctl)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.btn_plot = QPushButton("Plot selected")
        self.btn_clear_plot = QPushButton("Clear plot")
        self.btn_autorange = QPushButton("Auto range")
        self.btn_select_all = QPushButton("Select all channels")
        self.btn_select_none = QPushButton("Select none")
        self.btn_pan_mode = QPushButton("Pan mode: OFF (Rect-zoom ON)")

        self.info_label = QLabel("Decimation target: ~200k points/trace")

        grid.addWidget(self.btn_plot, 0, 0, 1, 2)
        grid.addWidget(self.btn_clear_plot, 1, 0, 1, 2)
        grid.addWidget(self.btn_autorange, 2, 0, 1, 2)
        grid.addWidget(self.btn_select_all, 3, 0, 1, 1)
        grid.addWidget(self.btn_select_none, 3, 1, 1, 1)
        grid.addWidget(self.btn_pan_mode, 4, 0, 1, 2)
        grid.addWidget(self.info_label, 5, 0, 1, 2)

        left_layout.addWidget(g_files, 3)
        left_layout.addWidget(g_ch, 2)
        left_layout.addWidget(g_style, 3)
        left_layout.addWidget(g_ctl, 0)

        # ---------- Right panel ----------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")

        axis_pen = pg.mkPen((0, 0, 0))
        for ax in ("bottom", "left"):
            self.plot.getAxis(ax).setPen(axis_pen)
            self.plot.getAxis(ax).setTextPen(axis_pen)

        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "X")
        self.plot.setLabel("left", "Amplitude")

        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 220))

        self._pan_mode = False
        self._apply_mouse_mode()

        right_layout.addWidget(self.plot)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([450, 700])

        # ---------- signals ----------
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_remove.clicked.connect(self.on_remove_files)
        self.btn_clear_files.clicked.connect(self.on_clear_files)
        self.file_list.itemSelectionChanged.connect(self.on_file_selection_changed)

        self.btn_plot.clicked.connect(self.on_plot_selected)
        self.btn_clear_plot.clicked.connect(self.on_clear_plot)
        self.btn_autorange.clicked.connect(self.on_autorange)

        self.btn_select_all.clicked.connect(lambda: self._set_all_channels(True))
        self.btn_select_none.clicked.connect(lambda: self._set_all_channels(False))
        self.btn_pan_mode.clicked.connect(self.on_toggle_pan_mode)

        # Refresh table when color mode changes (optional)
        self.color_mode.currentIndexChanged.connect(lambda: self.style_table.viewport().update())

    # -----------------------
    # Mouse mode
    # -----------------------

    def _apply_mouse_mode(self):
        vb = self.plot.getViewBox()
        if self._pan_mode:
            vb.setMouseMode(pg.ViewBox.PanMode)
            self.btn_pan_mode.setText("Pan mode: ON (Rect-zoom OFF)")
        else:
            vb.setMouseMode(pg.ViewBox.RectMode)
            self.btn_pan_mode.setText("Pan mode: OFF (Rect-zoom ON)")

    def on_toggle_pan_mode(self):
        self._pan_mode = not self._pan_mode
        self._apply_mouse_mode()

    # -----------------------
    # Files
    # -----------------------

    def on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select .npz file(s)", "", "NPZ files (*.npz);;All files (*.*)"
        )
        for f in files:
            f = str(Path(f).expanduser().resolve())
            if f in self.data:
                continue
            try:
                self.data[f] = load_npz_np(f)
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load:\n{f}\n\n{e}")
                continue
            self.file_list.addItem(f)

        if files:
            self.file_list.setCurrentRow(self.file_list.count() - 1)

    def on_remove_files(self):
        for item in self.file_list.selectedItems():
            f = item.text()
            self.data.pop(f, None)
            self.file_list.takeItem(self.file_list.row(item))
        self.on_file_selection_changed()

    def on_clear_files(self):
        self.data.clear()
        self.file_list.clear()
        self.channel_checklist.set_channels([])
        self._refresh_style_table()

    def selected_files(self) -> List[str]:
        return [it.text() for it in self.file_list.selectedItems()]

    def on_file_selection_changed(self):
        files = self.selected_files()
        if not files:
            self.channel_checklist.set_channels([])
            self._refresh_style_table()
            return

        union = set()
        for f in files:
            union.update(detect_channels(list(self.data[f].keys())))

        channels = sorted(union)
        self.channel_checklist.set_channels(channels, default_checked=True)
        self._refresh_style_table()

    # -----------------------
    # Channels + style table
    # -----------------------

    def _set_all_channels(self, checked: bool):
        self.channel_checklist.set_all_checked(checked)
        self._refresh_style_table(keep_existing=True)

    def _refresh_style_table(self, keep_existing: bool = True):
        channels = self.channel_checklist.all_channels()
        if not channels:
            self.style_table.setRowCount(0)
            return

        existing: Dict[str, Tuple[str, float]] = {}
        if keep_existing:
            for r in range(self.style_table.rowCount()):
                ch = self.style_table.item(r, 0).text()
                color_item = self.style_table.item(r, 1)
                offset_widget = self.style_table.cellWidget(r, 2)
                c = color_item.background().color().name() if color_item else "#000000"
                off = float(offset_widget.value()) if offset_widget else 0.0
                existing[ch] = (c, off)

        self.style_table.setRowCount(0)

        for i, ch in enumerate(channels):
            self.style_table.insertRow(i)

            item_ch = QTableWidgetItem(ch)
            self.style_table.setItem(i, 0, item_ch)

            default_c = self._cycle[i % len(self._cycle)]
            c, off = existing.get(ch, (default_c, 0.0))

            item_color = QTableWidgetItem("click")
            item_color.setBackground(QColor(c))
            self.style_table.setItem(i, 1, item_color)

            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12)
            sb.setDecimals(6)
            sb.setSingleStep(0.1)
            sb.setValue(off)
            self.style_table.setCellWidget(i, 2, sb)

        if not self._style_table_connected:
            self.style_table.cellClicked.connect(self._on_style_table_clicked)
            self._style_table_connected = True

        self.style_table.resizeColumnsToContents()

    def _on_style_table_clicked(self, row: int, col: int):
        if col != 1:
            return
        item = self.style_table.item(row, col)
        current = item.background().color()
        c = QColorDialog.getColor(current, self, "Pick trace color (channel)")
        if c.isValid():
            item.setBackground(c)

    def _get_channel_style_map(self) -> Dict[str, Tuple[str, float]]:
        style: Dict[str, Tuple[str, float]] = {}
        for r in range(self.style_table.rowCount()):
            ch = self.style_table.item(r, 0).text()
            color = self.style_table.item(r, 1).background().color().name()
            offset = float(self.style_table.cellWidget(r, 2).value())
            style[ch] = (color, offset)
        return style

    # -----------------------
    # Plotting
    # -----------------------

    def on_clear_plot(self):
        self.plot.clear()
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 220))

    def on_autorange(self):
        self.plot.enableAutoRange()

    def on_plot_selected(self):
        files = self.selected_files()
        if not files:
            QMessageBox.warning(self, "No files selected", "Select one or more NPZ files first.")
            return

        channels = self.channel_checklist.checked_channels()
        if not channels:
            QMessageBox.warning(self, "No channels selected", "Select at least one channel to plot.")
            return

        channel_style = self._get_channel_style_map()
        max_points = 200_000

        # X label
        first = self.data[files[0]]
        chs0 = detect_channels(list(first.keys()))
        if chs0:
            y0 = np.array(first[chs0[0]]).ravel()
            _, xlabel = make_time_axis(first, len(y0))
            self.plot.setLabel("bottom", xlabel)

        color_by = self.color_mode.currentText()

        # File color map (only used if Color mode = By file)
        file_color = {}
        if color_by == "By file":
            for i, f in enumerate(files):
                file_color[f] = self._cycle[i % len(self._cycle)]

        for f in files:
            d = self.data[f]
            fname = Path(f).name

            for ch in channels:
                if ch not in d:
                    continue

                y = np.array(d[ch]).ravel()
                x, _ = make_time_axis(d, len(y))
                x, y, step = decimate_xy(x, y, max_points=max_points)

                ch_color, off = channel_style.get(ch, ("#000000", 0.0))
                y = y + off

                if color_by == "By file":
                    color = file_color.get(f, "#000000")
                else:
                    color = ch_color

                pen = pg.mkPen(color=color, width=1)
                self.plot.plot(x, y, pen=pen, name=f"{fname}:{ch}  off={off:g}  (x{step})")

        self.plot.enableAutoRange()


def main():
    pg.setConfigOptions(antialias=True)

    app = QApplication(sys.argv)
    w = NpzPyqtgraphViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
