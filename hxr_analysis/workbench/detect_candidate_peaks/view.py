from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hxr_analysis.template.detect_candidate_peaks import DetectCandidatePeaksParams
from hxr_analysis.workbench.detect_candidate_peaks.controller import (
    DetectCandidatePeaksController,
    ImportErrorWithContext,
)
from hxr_analysis.workbench.waveform_display.path import PathResolutionError, resolve_path


class DetectCandidatePeaksWindow(QMainWindow):
    def __init__(self, controller: DetectCandidatePeaksController | None = None):
        super().__init__()
        self.setWindowTitle("Detect Candidate Peaks")
        self.resize(1400, 900)
        self.controller = controller or DetectCandidatePeaksController()
        self.current_payload_id: str | None = None
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("y"))

        self._setup_ui()
        self.refresh_payloads()
        self.update_buttons_state()

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

        # Left pane: payload + channels
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        payload_group = QGroupBox("Payloads")
        payload_layout = QVBoxLayout(payload_group)
        self.list_payloads = QListWidget()
        self.list_payloads.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_payloads.itemSelectionChanged.connect(self.on_payload_selected)
        payload_layout.addWidget(self.list_payloads)
        left_layout.addWidget(payload_group)

        channel_group = QGroupBox("Channels")
        channel_layout = QVBoxLayout(channel_group)
        self.chk_all_channels = QCheckBox("Process all channels")
        self.chk_all_channels.setChecked(True)
        self.chk_all_channels.stateChanged.connect(self.update_buttons_state)
        self.list_channels = QListWidget()
        self.list_channels.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        channel_layout.addWidget(self.chk_all_channels)
        channel_layout.addWidget(self.list_channels)
        left_layout.addWidget(channel_group)

        splitter.addWidget(left_widget)

        # Right pane
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        params_group = QGroupBox("Parameters")
        form = QFormLayout(params_group)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setValue(5.0)
        self.spin_threshold.setRange(0.1, 1_000)
        form.addRow("Threshold sigma", self.spin_threshold)

        self.spin_min_distance = QSpinBox()
        self.spin_min_distance.setRange(1, 100000)
        self.spin_min_distance.setValue(20)
        form.addRow("Dead time (samples)", self.spin_min_distance)

        self.spin_min_width = QSpinBox()
        self.spin_min_width.setRange(1, 100000)
        self.spin_min_width.setValue(1)
        form.addRow("Min width (samples)", self.spin_min_width)

        self.cmb_noise = QComboBox()
        self.cmb_noise.addItems(["mad", "rms", "std_pretrigger"])
        form.addRow("Noise method", self.cmb_noise)

        self.cmb_polarity = QComboBox()
        self.cmb_polarity.addItems(["normalized", "preserve", "invert", "auto"])
        form.addRow("Polarity", self.cmb_polarity)

        self.spin_pre_start = QDoubleSpinBox()
        self.spin_pre_end = QDoubleSpinBox()
        self.spin_pre_start.setDecimals(4)
        self.spin_pre_end.setDecimals(4)
        self.spin_pre_start.setRange(-1e6, 1e6)
        self.spin_pre_end.setRange(-1e6, 1e6)
        pretrigger_row = QHBoxLayout()
        pretrigger_row.addWidget(self.spin_pre_start)
        pretrigger_row.addWidget(QLabel("to"))
        pretrigger_row.addWidget(self.spin_pre_end)
        form.addRow("Pretrigger window (s)", pretrigger_row)

        self.spin_max_peaks = QSpinBox()
        self.spin_max_peaks.setRange(0, 1_000_000)
        self.spin_max_peaks.setValue(0)
        form.addRow("Max peaks/channel (0 = all)", self.spin_max_peaks)

        self.chk_store_regions = QCheckBox()
        self.chk_store_regions.setChecked(True)
        form.addRow("Store regions", self.chk_store_regions)

        self.chk_store_snr = QCheckBox()
        self.chk_store_snr.setChecked(True)
        form.addRow("Store SNR", self.chk_store_snr)

        self.chk_reject_sat = QCheckBox("Reject saturated")
        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setDecimals(3)
        self.spin_saturation.setRange(-1e9, 1e9)
        sat_row = QHBoxLayout()
        sat_row.addWidget(self.chk_reject_sat)
        sat_row.addWidget(QLabel("Level"))
        sat_row.addWidget(self.spin_saturation)
        form.addRow("Saturation", sat_row)

        right_layout.addWidget(params_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        control_row = QHBoxLayout()
        self.cmb_preview_channel = QComboBox()
        self.cmb_preview_channel.currentTextChanged.connect(self.update_preview_plot)
        self.chk_show_threshold = QCheckBox("Show threshold")
        self.chk_show_threshold.setChecked(True)
        self.chk_show_threshold.stateChanged.connect(self.update_preview_plot)
        self.chk_show_regions = QCheckBox("Show regions")
        self.chk_show_regions.setChecked(True)
        self.chk_show_regions.stateChanged.connect(self.update_preview_plot)
        control_row.addWidget(QLabel("Channel"))
        control_row.addWidget(self.cmb_preview_channel)
        control_row.addStretch(1)
        control_row.addWidget(self.chk_show_threshold)
        control_row.addWidget(self.chk_show_regions)
        preview_layout.addLayout(control_row)

        self.plot = pg.PlotWidget()
        self.plot.addItem(self.cursor_line)
        preview_layout.addWidget(self.plot, 2)

        self.lbl_peak_count = QLabel("0 peaks found")
        preview_layout.addWidget(self.lbl_peak_count)
        right_layout.addWidget(preview_group, 2)

        table_group = QGroupBox("Peaks")
        table_layout = QVBoxLayout(table_group)
        self.table_peaks = QTableWidget(0, 4)
        self.table_peaks.setHorizontalHeaderLabels(["Index", "Time", "Amplitude", "SNR"])
        self.table_peaks.cellClicked.connect(self.on_table_clicked)
        table_layout.addWidget(self.table_peaks)
        right_layout.addWidget(table_group, 1)

        action_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_save = QPushButton("Save Output")
        self.btn_close = QPushButton("Close")
        action_row.addWidget(self.btn_run)
        action_row.addWidget(self.btn_save)
        action_row.addStretch(1)
        action_row.addWidget(self.btn_close)
        right_layout.addLayout(action_row)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 2)

        # Signals
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_close.clicked.connect(self.close)

    # ---------- State helpers ----------
    def refresh_payloads(self):
        self.list_payloads.clear()
        for pid in self.controller.payload_ids():
            self.list_payloads.addItem(pid)
        self.update_buttons_state()

    def update_buttons_state(self):
        has_payload = bool(self.list_payloads.selectedItems())
        self.btn_run.setEnabled(has_payload)
        self.btn_save.setEnabled(has_payload and self.controller.get_output(self.current_payload_id or "") is not None)

    def on_import(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select payload", "", "NPZ files (*.npz)")
        if not files:
            return
        try:
            self.controller.import_payloads(files)
        except ImportErrorWithContext as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.refresh_payloads()

    def on_payload_selected(self):
        items = self.list_payloads.selectedItems()
        if not items:
            self.current_payload_id = None
            self.list_channels.clear()
            self.cmb_preview_channel.clear()
            self.update_buttons_state()
            return
        pid = items[0].text()
        self.current_payload_id = pid
        channels = self.controller.available_channels(pid)
        self.list_channels.clear()
        for ch in channels:
            self.list_channels.addItem(ch)
        self.cmb_preview_channel.clear()
        self.cmb_preview_channel.addItems(channels)
        if channels:
            self.cmb_preview_channel.setCurrentIndex(0)
        self.update_buttons_state()
        self.update_preview_plot()

    def _selected_channels(self):
        if self.chk_all_channels.isChecked():
            return None
        selected = [item.text() for item in self.list_channels.selectedItems()]
        return selected or None

    def _collect_params(self) -> DetectCandidatePeaksParams:
        pre_range = None
        if self.cmb_noise.currentText() == "std_pretrigger":
            pre_range = (self.spin_pre_start.value(), self.spin_pre_end.value())
        max_peaks = self.spin_max_peaks.value() or None
        saturation_level = self.spin_saturation.value()
        if not self.chk_reject_sat.isChecked():
            saturation_level = None
        return DetectCandidatePeaksParams(
            threshold_sigma=self.spin_threshold.value(),
            min_distance_samples=self.spin_min_distance.value(),
            min_width_samples=self.spin_min_width.value(),
            noise_method=self.cmb_noise.currentText(),
            polarity=self.cmb_polarity.currentText(),
            pretrigger_time_range=pre_range,
            channel_keys=self._selected_channels(),
            max_peaks_per_channel=max_peaks,
            reject_saturated=self.chk_reject_sat.isChecked(),
            saturation_level=saturation_level,
            store_regions=self.chk_store_regions.isChecked(),
            store_snr=self.chk_store_snr.isChecked(),
        )

    # ---------- Actions ----------
    def on_run(self):
        if not self.current_payload_id:
            QMessageBox.warning(self, "Selection required", "Please choose a payload to process.")
            return
        try:
            params = self._collect_params()
            output = self.controller.run_detection(self.current_payload_id, params)
        except Exception as exc:  # pragma: no cover - GUI feedback path
            QMessageBox.critical(self, "Detection failed", str(exc))
            return

        if output:
            self.update_buttons_state()
            self.update_preview_plot()

    def on_save(self):
        if not self.current_payload_id:
            return
        target_dir = QFileDialog.getExistingDirectory(self, "Save processed payload", "candidate_peaks")
        if not target_dir:
            return
        try:
            path = self.controller.save_output(self.current_payload_id, target_dir)
        except Exception as exc:  # pragma: no cover - GUI feedback path
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Saved to {Path(path).as_posix()}")

    # ---------- Preview + table ----------
    def _load_waveform(self, payload_id: str, channel: str):
        payload = self.controller.get_output(payload_id) or self.controller.get_payload(payload_id)
        try:
            t = resolve_path(payload, "data.time")
            y = resolve_path(payload, f"data.channels.{channel}")
        except PathResolutionError:
            return None, None
        return t, y

    def _candidate_data(self, payload_id: str, channel: str):
        output = self.controller.get_output(payload_id)
        if not output:
            return None
        candidate = output.get("events", {}).get("candidate_peaks", {})
        by_ch = candidate.get("by_channel", {})
        return by_ch.get(channel)

    def update_preview_plot(self):
        self.plot.clear()
        self.plot.addItem(self.cursor_line)
        if not self.current_payload_id:
            return
        channel = self.cmb_preview_channel.currentText()
        if not channel:
            return
        t, y = self._load_waveform(self.current_payload_id, channel)
        if t is None or y is None:
            return
        self.plot.plot(t, y, pen=pg.mkPen("w"))

        peaks = self._candidate_data(self.current_payload_id, channel) or {}
        n_peaks = len(peaks.get("i", [])) if hasattr(peaks.get("i", []), "__len__") else 0
        self.lbl_peak_count.setText(f"{n_peaks} peaks found")

        if self.chk_show_threshold.isChecked() and "threshold" in peaks:
            thr_value = float(peaks.get("threshold", 0.0))
            if thr_value < float("inf"):
                thr_line = pg.InfiniteLine(
                    pos=thr_value, angle=0, pen=pg.mkPen("r", style=Qt.PenStyle.DashLine)
                )
                self.plot.addItem(thr_line)

        if n_peaks:
            scatter = pg.ScatterPlotItem(peaks.get("t", []), peaks.get("amp", []), pen=pg.mkPen("c"), brush=pg.mkBrush("c"))
            self.plot.addItem(scatter)

            if self.chk_show_regions.isChecked() and "region_start" in peaks and "region_end" in peaks:
                for start, end in zip(peaks.get("region_start", []), peaks.get("region_end", []), strict=False):
                    try:
                        x0 = t[int(start)]
                        x1 = t[int(end - 1)] if end > 0 else t[int(start)]
                    except Exception:
                        continue
                    region = pg.LinearRegionItem(values=[x0, x1], brush=pg.mkBrush(100, 100, 150, 50))
                    region.setMovable(False)
                    self.plot.addItem(region)

        self.populate_table(peaks)

    def populate_table(self, peaks: dict | None):
        if not peaks:
            self.table_peaks.setRowCount(0)
            return
        i_vals = peaks.get("i", [])
        t_vals = peaks.get("t", [])
        amp_vals = peaks.get("amp", [])
        snr_vals = peaks.get("snr", [])
        n = len(i_vals)
        self.table_peaks.setRowCount(n)
        for row in range(n):
            self.table_peaks.setItem(row, 0, QTableWidgetItem(str(i_vals[row])))
            self.table_peaks.setItem(row, 1, QTableWidgetItem(f"{t_vals[row]:.6g}"))
            self.table_peaks.setItem(row, 2, QTableWidgetItem(f"{amp_vals[row]:.6g}"))
            snr_val = snr_vals[row] if row < len(snr_vals) else ""
            self.table_peaks.setItem(row, 3, QTableWidgetItem(f"{snr_val}"))

    def on_table_clicked(self, row: int, column: int):  # noqa: ARG002
        channel = self.cmb_preview_channel.currentText()
        if not self.current_payload_id or not channel:
            return
        peaks = self._candidate_data(self.current_payload_id, channel) or {}
        if row >= len(peaks.get("t", [])):
            return
        try:
            t_val = float(peaks.get("t", [])[row])
        except Exception:
            return
        self.cursor_line.setPos(t_val)


__all__ = ["DetectCandidatePeaksWindow"]
