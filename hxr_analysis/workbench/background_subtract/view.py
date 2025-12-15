from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import List

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from hxr_analysis.preprocessing.background_subtract import BackgroundSubtractParams, background_subtract_one
from hxr_analysis.workbench.background_subtract.controller import (
    BackgroundSubtractController,
    ImportErrorWithContext,
)


class BackgroundSubtractWindow(QMainWindow):
    def __init__(self, controller: BackgroundSubtractController | None = None):
        super().__init__()
        self.setWindowTitle("Background Subtraction Workbench")
        self.resize(1200, 800)
        self.controller = controller or BackgroundSubtractController()
        self.preview_mode = "subtracted"
        self._setup_ui()
        self.refresh_lists()

    # ---------- UI setup ----------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self.btn_import = QPushButton("Load Payloads…")
        self.btn_import.clicked.connect(self.on_import)
        top_row.addWidget(self.btn_import)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        # Left side selections
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        bg_group = QGroupBox("Background Selection")
        bg_layout = QVBoxLayout(bg_group)
        self.list_background = QListWidget()
        self.btn_set_background = QPushButton("Set as Background")
        self.lbl_background_info = QLabel("No background selected")
        bg_layout.addWidget(self.list_background)
        bg_layout.addWidget(self.btn_set_background)
        bg_layout.addWidget(self.lbl_background_info)
        left_layout.addWidget(bg_group)

        exp_group = QGroupBox("Experiment Payloads")
        exp_layout = QVBoxLayout(exp_group)
        self.list_experiments = QListWidget()
        self.list_experiments.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.btn_select_all = QPushButton("Select All")
        self.btn_clear = QPushButton("Clear")
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_clear)
        exp_layout.addWidget(self.list_experiments)
        exp_layout.addLayout(btn_row)
        left_layout.addWidget(exp_group)

        splitter.addWidget(left_widget)

        # Right side parameters + preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        params_group = QGroupBox("Parameters")
        form = QFormLayout(params_group)
        self.cmb_time_align = QComboBox()
        self.cmb_time_align.addItems(["require_equal", "interp_bg_to_exp"])
        form.addRow("Time align", self.cmb_time_align)

        self.cmb_match_mode = QComboBox()
        self.cmb_match_mode.addItems(["by_key", "by_index"])
        form.addRow("Match mode", self.cmb_match_mode)

        self.cmb_missing_policy = QComboBox()
        self.cmb_missing_policy.addItems(["skip", "error"])
        form.addRow("Missing policy", self.cmb_missing_policy)

        self.spin_bg_scale = QDoubleSpinBox()
        self.spin_bg_scale.setValue(1.0)
        self.spin_bg_scale.setDecimals(3)
        form.addRow("Background scale", self.spin_bg_scale)

        self.spin_exp_scale = QDoubleSpinBox()
        self.spin_exp_scale.setValue(1.0)
        self.spin_exp_scale.setDecimals(3)
        form.addRow("Experiment scale", self.spin_exp_scale)

        self.chk_store_original = QCheckBox()
        self.chk_store_original.setChecked(True)
        form.addRow("Store original", self.chk_store_original)

        right_layout.addWidget(params_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        control_row = QHBoxLayout()
        self.cmb_channel = QComboBox()
        self.cmb_preview_mode = QComboBox()
        self.cmb_preview_mode.addItems(["raw exp", "background", "subtracted"])
        control_row.addWidget(QLabel("Channel"))
        control_row.addWidget(self.cmb_channel)
        control_row.addWidget(QLabel("Show"))
        control_row.addWidget(self.cmb_preview_mode)
        preview_layout.addLayout(control_row)

        self.plot = pg.PlotWidget()
        preview_layout.addWidget(self.plot, 1)
        right_layout.addWidget(preview_group, 2)

        action_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_save = QPushButton("Save Outputs")
        self.btn_close = QPushButton("Close")
        action_row.addWidget(self.btn_run)
        action_row.addWidget(self.btn_save)
        action_row.addStretch(1)
        action_row.addWidget(self.btn_close)
        right_layout.addLayout(action_row)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)

        # Signals
        self.btn_set_background.clicked.connect(self.on_set_background)
        self.btn_select_all.clicked.connect(self.on_select_all)
        self.btn_clear.clicked.connect(self.on_clear_selection)
        self.list_experiments.itemSelectionChanged.connect(self.update_preview_channels)
        self.cmb_channel.currentTextChanged.connect(self.update_preview_plot)
        self.cmb_preview_mode.currentTextChanged.connect(self.update_preview_plot)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_close.clicked.connect(self.close)

    # ---------- Helpers ----------
    def refresh_lists(self):
        self.list_background.clear()
        self.list_experiments.clear()
        for pid in self.controller.store.ids():
            self.list_background.addItem(pid)
            self.list_experiments.addItem(pid)
        self.update_buttons_state()

    def update_buttons_state(self):
        has_bg = self.controller.background_id is not None
        has_exp = bool(self.list_experiments.selectedItems())
        self.btn_run.setEnabled(has_bg and has_exp)
        self.btn_save.setEnabled(has_bg and has_exp)

    def on_import(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select payloads", "", "NPZ files (*.npz)")
        if not files:
            return
        try:
            self.controller.import_payloads(files)
        except ImportErrorWithContext as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.refresh_lists()

    def on_set_background(self):
        item = self.list_background.currentItem()
        if not isinstance(item, QListWidgetItem):
            return
        try:
            self.controller.set_background(item.text())
        except KeyError as exc:
            QMessageBox.critical(self, "Invalid selection", str(exc))
            return
        bg_payload = self.controller.selected_background()
        channels = bg_payload.get("data", {}).get("channels", {})
        time = bg_payload.get("data", {}).get("time", [])
        self.lbl_background_info.setText(
            f"Background: {item.text()} | channels: {list(channels.keys())} | len(time)={len(time)}"
        )
        self.update_buttons_state()
        self.update_preview_channels()

    def on_select_all(self):
        self.list_experiments.selectAll()
        self.update_buttons_state()
        self.update_preview_channels()

    def on_clear_selection(self):
        self.list_experiments.clearSelection()
        self.update_buttons_state()
        self.update_preview_channels()

    def current_params(self) -> BackgroundSubtractParams:
        return BackgroundSubtractParams(
            time_align=self.cmb_time_align.currentText(),
            match_mode=self.cmb_match_mode.currentText(),
            missing_channel_policy=self.cmb_missing_policy.currentText(),
            bg_scale=self.spin_bg_scale.value(),
            exp_scale=self.spin_exp_scale.value(),
            store_original=self.chk_store_original.isChecked(),
        )

    def selected_experiment_ids(self) -> List[str]:
        return [item.text() for item in self.list_experiments.selectedItems()]

    def update_preview_channels(self):
        items = self.list_experiments.selectedItems()
        if not items:
            self.cmb_channel.clear()
            return
        payload_id = items[0].text()
        payload = self.controller.store.get(payload_id)
        channels = payload.get("data", {}).get("channels", {})
        self.cmb_channel.clear()
        self.cmb_channel.addItems(list(channels.keys()))
        self.update_preview_plot()

    def _plot_waveform(self, time, values, label: str):
        self.plot.plot(time, values, pen=pg.mkPen(width=2), name=label)

    def update_preview_plot(self):
        self.plot.clear()
        channel = self.cmb_channel.currentText()
        if not channel:
            return
        bg_payload = None
        try:
            bg_payload = self.controller.selected_background()
        except Exception:
            pass
        items = self.list_experiments.selectedItems()
        if not items:
            return
        exp_payload = self.controller.store.get(items[0].text())
        data = exp_payload.get("data", {})
        time = data.get("time")
        channels = data.get("channels", {})
        if channel in channels:
            self._plot_waveform(time, channels[channel], f"exp:{channel}")
        if bg_payload and channel in bg_payload.get("data", {}).get("channels", {}):
            bg_time = bg_payload.get("data", {}).get("time")
            bg_values = bg_payload.get("data", {}).get("channels", {}).get(channel)
            if bg_time is not None:
                self._plot_waveform(bg_time, bg_values, f"bg:{channel}")
        if bg_payload:
            try:
                params = self.current_params()
                preview_payload = background_subtract_one(exp_payload, bg_payload, params=params)
                result_channels = preview_payload.get("data", {}).get("channels", {})
                if channel in result_channels:
                    self._plot_waveform(time, result_channels[channel], f"sub:{channel}")
            except Exception:
                pass

    def on_run(self):
        params = self.current_params()
        exp_ids = self.selected_experiment_ids()
        if not exp_ids:
            QMessageBox.information(self, "No selection", "Select experiment payloads to run")
            return
        try:
            self.outputs = self.controller.compute(exp_ids, params)
        except Exception as exc:
            QMessageBox.critical(self, "Background subtraction failed", str(exc))
            return
        QMessageBox.information(self, "Completed", f"Processed {len(self.outputs)} payload(s)")
        self.update_buttons_state()
        self.update_preview_plot()

    def on_save(self):
        if not hasattr(self, "outputs") or not self.outputs:
            QMessageBox.information(self, "Nothing to save", "Run subtraction first")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        target = Path(folder) / "background_subtracted"
        manifest = self.controller.save_outputs(self.outputs, target)
        QMessageBox.information(self, "Saved", f"Outputs saved to {target}\nManifest: {manifest}")


def run():
    app = QApplication.instance() or QApplication([])
    w = BackgroundSubtractWindow()
    w.show()
    app.exec()


__all__ = ["BackgroundSubtractWindow", "run"]
