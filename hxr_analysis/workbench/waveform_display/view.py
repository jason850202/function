from __future__ import annotations

from typing import List

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .importers import ImportErrorWithContext, import_files
from .mapping import MappingValidationError, PlotMapping, validate_and_resolve
from .models import PayloadStore
from .plot import render_mappings
from .style import parse_style


class WaveformDisplayWindow(QMainWindow):
    COL_TYPE = 0
    COL_PAYLOAD = 1
    COL_X = 2
    COL_Y = 3
    COL_VALUE = 4
    COL_STYLE = 5

    def __init__(self, store: PayloadStore | None = None):
        super().__init__()
        self.setWindowTitle("Waveform Display Workbench")
        self.resize(1200, 700)
        self.store = store or PayloadStore()
        self._setup_ui()

    # ---------- UI setup ----------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self.btn_import = QPushButton("Import Files…")
        self.btn_refresh = QPushButton("Refresh")
        top_row.addWidget(self.btn_import)
        top_row.addWidget(self.btn_refresh)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter, 1)

        # Left: tabs for bundle browser and plot setup
        left_container = QWidget()
        left_container.setMinimumWidth(280)
        left_layout = QVBoxLayout(left_container)
        tabs = QTabWidget()
        left_layout.addWidget(tabs)

        # Bundle browser tab
        bundle_tab = QWidget()
        bundle_layout = QVBoxLayout(bundle_tab)
        bundle_layout.addWidget(QLabel("Bundle Browser"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        bundle_layout.addWidget(self.tree, 1)
        tabs.addTab(bundle_tab, "Bundle Browser")

        # Plot setup tab
        plot_setup_tab = QWidget()
        plot_setup_layout = QVBoxLayout(plot_setup_tab)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Type", "Payload", "X Path", "Y Path", "Value Path", "Style"]
        )
        plot_setup_layout.addWidget(self.table, 1)

        map_btn_row = QHBoxLayout()
        self.btn_add_mapping = QPushButton("Add")
        self.btn_remove_mapping = QPushButton("Remove")
        self.btn_show_plot = QPushButton("Show Plot")
        self.btn_reload_mappings = QPushButton("Reload Mappings")
        map_btn_row.addWidget(self.btn_add_mapping)
        map_btn_row.addWidget(self.btn_remove_mapping)
        map_btn_row.addWidget(self.btn_show_plot)
        map_btn_row.addWidget(self.btn_reload_mappings)
        map_btn_row.addStretch(1)
        plot_setup_layout.addLayout(map_btn_row)
        tabs.addTab(plot_setup_tab, "Plot Setup")

        main_splitter.addWidget(left_container)

        # Right: plot and log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        plot_splitter = QSplitter(Qt.Orientation.Vertical)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self._configure_plot_widget()
        plot_splitter.addWidget(self.plot_widget)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        plot_splitter.addWidget(self.log_box)

        plot_splitter.setStretchFactor(0, 3)
        plot_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(plot_splitter)

        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        # Signals
        self.btn_import.clicked.connect(self.on_import)
        self.btn_refresh.clicked.connect(self.refresh_views)
        self.btn_add_mapping.clicked.connect(self.add_mapping_row)
        self.btn_remove_mapping.clicked.connect(self.remove_selected_rows)
        self.btn_show_plot.clicked.connect(self.on_show_plot)
        self.btn_reload_mappings.clicked.connect(self.refresh_mapping_payloads)

        self.refresh_views()

    def _configure_plot_widget(self):
        self.plot_widget.setBackground("w")
        axis_bottom = self.plot_widget.getAxis("bottom")
        axis_left = self.plot_widget.getAxis("left")
        for axis in (axis_bottom, axis_left):
            axis.setPen(pg.mkPen("k"))
            axis.setTextPen(pg.mkPen("k"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

    # ---------- Payload handling ----------
    def on_import(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select data files", "", "NPZ files (*.npz);;All files (*.*)"
        )
        if not files:
            return
        try:
            imported = import_files(files)
        except ImportErrorWithContext as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        for pid, payload in imported.items():
            self.store.add(pid, payload)
            self.log_box.append(f"Imported {pid} from {payload.get('meta', {}).get('source', '')}")
        self.refresh_views()

    def refresh_views(self):
        self._rebuild_tree()
        self.refresh_mapping_payloads()

    def _rebuild_tree(self):
        self.tree.clear()
        for pid, payload in self.store.payloads.items():
            root_item = QTreeWidgetItem([pid])
            self.tree.addTopLevelItem(root_item)
            self._populate_payload_item(root_item, payload)
        self.tree.expandAll()

    def _populate_payload_item(self, parent: QTreeWidgetItem, obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = QTreeWidgetItem([str(key)])
                parent.addChild(child)
                self._populate_payload_item(child, value)
        elif isinstance(obj, (list, tuple)):
            child = QTreeWidgetItem([f"list[{len(obj)}]"])
            parent.addChild(child)
        elif hasattr(obj, "shape"):
            child = QTreeWidgetItem([f"array{getattr(obj, 'shape', '')}"])
            parent.addChild(child)
        else:
            child = QTreeWidgetItem([str(obj)])
            parent.addChild(child)

    # ---------- Mapping table ----------
    def add_mapping_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        type_combo = QComboBox()
        type_combo.addItems(["curve", "scatter"])
        self.table.setCellWidget(row, self.COL_TYPE, type_combo)

        payload_combo = QComboBox()
        payload_combo.addItems(list(self.store.ids()))
        self.table.setCellWidget(row, self.COL_PAYLOAD, payload_combo)

        for col in (self.COL_X, self.COL_Y, self.COL_VALUE, self.COL_STYLE):
            item = QTableWidgetItem("")
            self.table.setItem(row, col, item)
        self.table.setItem(row, self.COL_STYLE, QTableWidgetItem("{}"))

    def remove_selected_rows(self):
        selected_rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in selected_rows:
            self.table.removeRow(row)

    def refresh_mapping_payloads(self):
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, self.COL_PAYLOAD)
            if isinstance(combo, QComboBox):
                current = combo.currentText()
                combo.clear()
                combo.addItems(list(self.store.ids()))
                idx = combo.findText(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _mapping_from_row(self, row: int) -> PlotMapping:
        type_combo = self.table.cellWidget(row, self.COL_TYPE)
        payload_combo = self.table.cellWidget(row, self.COL_PAYLOAD)
        plot_type = type_combo.currentText() if isinstance(type_combo, QComboBox) else "curve"
        payload_id = payload_combo.currentText() if isinstance(payload_combo, QComboBox) else ""

        def item_text(col: int) -> str:
            item = self.table.item(row, col)
            return item.text() if isinstance(item, QTableWidgetItem) else ""

        mapping = PlotMapping(
            plot_type=plot_type,
            payload_id=payload_id,
            x_path=item_text(self.COL_X),
            y_path=item_text(self.COL_Y),
            value_path=item_text(self.COL_VALUE) or None,
            style=item_text(self.COL_STYLE) or "{}",
        )
        return mapping

    # ---------- Plotting ----------
    def on_show_plot(self):
        resolved: List = []
        errors: List[str] = []
        for row in range(self.table.rowCount()):
            try:
                mapping = self._mapping_from_row(row)
                style_dict = parse_style(mapping.style)
            except ValueError as exc:
                QMessageBox.critical(
                    self,
                    "Style parse error",
                    f"Row {row + 1} style error: {exc}\nInput: {mapping.style}",
                )
                continue
            try:
                resolved_map = validate_and_resolve(
                    mapping, self.store, style_dict=style_dict
                )
                resolved.append(resolved_map)
            except MappingValidationError as exc:
                errors.append(f"Row {row + 1}: {exc}")
        if errors:
            QMessageBox.critical(self, "Mapping errors", "\n".join(errors))
            return
        if not resolved:
            QMessageBox.information(self, "Nothing to plot", "Please add mapping rows first.")
            return
        render_mappings(self.plot_widget, resolved)
        self.log_box.append(f"Plotted {len(resolved)} mapping(s)")


def run():
    app = QApplication.instance() or QApplication([])
    w = WaveformDisplayWindow()
    w.show()
    app.exec()


__all__ = ["WaveformDisplayWindow", "run"]
