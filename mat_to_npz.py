from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QListWidget, QTextEdit,
    QFileDialog, QLineEdit, QCheckBox, QProgressBar,
    QMessageBox, QGroupBox, QSizePolicy
)


# =========================
# MAT loading + NPZ saving
# =========================

def _is_hdf5_mat(mat_path: Path) -> bool:
    """MATLAB v7.3 files are HDF5 containers."""
    try:
        import h5py
        with h5py.File(mat_path, "r"):
            return True
    except Exception:
        return False


def _mat_struct_to_dict(obj: Any) -> Any:
    """
    Convert scipy.io.loadmat objects into native Python types:
    - MATLAB structs -> dict
    - object arrays/cells -> list (recursively)
    - numeric arrays -> numpy arrays
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_mat_struct_to_dict(x) for x in obj.flat]
        return obj

    if hasattr(obj, "_fieldnames"):  # mat_struct
        out: Dict[str, Any] = {}
        for name in obj._fieldnames:
            out[name] = _mat_struct_to_dict(getattr(obj, name))
        return out

    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: _mat_struct_to_dict(obj[name]) for name in obj.dtype.names}

    return obj


def _load_mat_v7(mat_path: Path, variable_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    from scipy.io import loadmat

    mdict = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=list(variable_names) if variable_names else None,
    )
    for k in ["__header__", "__version__", "__globals__"]:
        mdict.pop(k, None)

    return {k: _mat_struct_to_dict(v) for k, v in mdict.items()}


def _hdf5_group_to_python(h5obj: Any) -> Any:
    import h5py

    if isinstance(h5obj, h5py.Dataset):
        data = h5obj[()]
        if isinstance(data, (bytes, np.bytes_)):
            try:
                return data.decode("utf-8")
            except Exception:
                return data
        return data

    if isinstance(h5obj, h5py.Group):
        out: Dict[str, Any] = {}
        for key in h5obj.keys():
            out[key] = _hdf5_group_to_python(h5obj[key])
        return out

    return None


def _load_mat_v73(mat_path: Path, variable_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    import h5py
    out: Dict[str, Any] = {}
    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())
        if variable_names:
            vset = set(variable_names)
            keys = [k for k in keys if k in vset]
        for k in keys:
            out[k] = _hdf5_group_to_python(f[k])
    return out


def load_mat(mat_path: str | Path, variable_names: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    mat_path = Path(mat_path).expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    if _is_hdf5_mat(mat_path):
        return _load_mat_v73(mat_path, variable_names=variable_names)
    return _load_mat_v7(mat_path, variable_names=variable_names)


def save_npz(data: Dict[str, Any], out_npz_path: str | Path, *, compress: bool = True) -> Path:
    """
    Save a nested dict to .npz.
    - arrays/scalars saved directly
    - everything else saved as dtype=object (requires allow_pickle=True to load)
    """
    out_npz_path = Path(out_npz_path).expanduser().resolve()
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) or np.isscalar(v):
            arrays[k] = v
        else:
            arrays[k] = np.array(v, dtype=object)

    if compress:
        np.savez_compressed(out_npz_path, **arrays)
    else:
        np.savez(out_npz_path, **arrays)

    return out_npz_path


def convert_mat_to_npz(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    out_name: Optional[str] = None,
    variable_names: Optional[Iterable[str]] = None,
    compress: bool = True,
) -> Path:
    mat_path = Path(mat_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = mat_path.stem + ".npz"
    if not out_name.endswith(".npz"):
        out_name += ".npz"

    data = load_mat(mat_path, variable_names=variable_names)
    return save_npz(data, out_dir / out_name, compress=compress)


# =========================
# Worker thread
# =========================

class ConvertWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(bool, str)

    def __init__(
        self,
        mat_files: List[str],
        out_dir: str,
        name_pattern: str,
        vars_text: str,
        compress: bool,
        overwrite: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.mat_files = mat_files
        self.out_dir = out_dir
        self.name_pattern = name_pattern
        self.vars_text = vars_text
        self.compress = compress
        self.overwrite = overwrite
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _make_out_name(self, mat_path: Path) -> str:
        """
        name_pattern supports:
          - {stem}  : filename without suffix
          - {name}  : filename with suffix
          - {idx}   : 1-based index (handled in run)
        """
        # idx is injected in run
        return self.name_pattern.format(stem=mat_path.stem, name=mat_path.name, idx=1)

    def run(self):
        try:
            if not self.mat_files:
                self.done.emit(False, "No input .mat files.")
                return
            if not self.out_dir.strip():
                self.done.emit(False, "No output directory selected.")
                return

            out_dir = Path(self.out_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            variable_names = None
            vars_clean = [v.strip() for v in self.vars_text.split(",") if v.strip()]
            if vars_clean:
                variable_names = vars_clean

            total = len(self.mat_files)
            self.log.emit(f"Starting conversion: {total} file(s)")
            self.log.emit(f"Output directory: {out_dir}")
            self.log.emit(f"Compression: {'ON' if self.compress else 'OFF'}")
            self.log.emit(f"Variable filter: {variable_names if variable_names else '(all variables)'}")
            self.log.emit("")

            for i, f in enumerate(self.mat_files, start=1):
                if self._cancel:
                    self.done.emit(False, "Canceled by user.")
                    return

                mat_path = Path(f).expanduser().resolve()
                if not mat_path.exists():
                    self.log.emit(f"[{i}/{total}] SKIP (missing): {mat_path}")
                    self.progress.emit(int(i / total * 100))
                    continue

                # build output name with idx
                try:
                    out_name = self.name_pattern.format(stem=mat_path.stem, name=mat_path.name, idx=i)
                except Exception:
                    out_name = mat_path.stem + ".npz"

                if not out_name.endswith(".npz"):
                    out_name += ".npz"

                out_path = out_dir / out_name

                if out_path.exists() and not self.overwrite:
                    self.log.emit(f"[{i}/{total}] SKIP (exists): {out_path.name}")
                    self.progress.emit(int(i / total * 100))
                    continue

                # detect type
                is73 = _is_hdf5_mat(mat_path)
                self.log.emit(f"[{i}/{total}] Converting: {mat_path.name}  ({'v7.3' if is73 else 'v7'})")
                saved = convert_mat_to_npz(
                    mat_path,
                    out_dir,
                    out_name=out_name,
                    variable_names=variable_names,
                    compress=self.compress,
                )
                self.log.emit(f"          Saved: {saved.name}")
                self.progress.emit(int(i / total * 100))

            self.log.emit("")
            self.done.emit(True, "All conversions finished.")
        except Exception as e:
            self.done.emit(False, f"Error: {e}")


# =========================
# GUI
# =========================

class MatToNpzGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAT → NPZ Converter")
        self.resize(900, 650)

        self.worker: Optional[ConvertWorker] = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Input group
        g_in = QGroupBox("Input .mat files")
        in_layout = QVBoxLayout(g_in)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(self.file_list.SelectionMode.ExtendedSelection)
        self.file_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add .mat…")
        self.btn_remove = QPushButton("Remove selected")
        self.btn_clear = QPushButton("Clear")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)

        in_layout.addWidget(self.file_list)
        in_layout.addLayout(btn_row)

        # --- Output/options group
        g_out = QGroupBox("Output settings")
        grid = QGridLayout(g_out)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("Choose output folder…")
        self.btn_out_dir = QPushButton("Browse…")

        self.name_pattern_edit = QLineEdit("{stem}.npz")
        self.name_pattern_edit.setToolTip("Use {stem}, {name}, {idx}. Example: {stem}_v1_{idx}.npz")

        self.vars_edit = QLineEdit()
        self.vars_edit.setPlaceholderText("Optional: var1,var2,var3  (leave empty for all)")

        self.chk_compress = QCheckBox("Use compression (np.savez_compressed)")
        self.chk_compress.setChecked(True)

        self.chk_overwrite = QCheckBox("Overwrite existing .npz")
        self.chk_overwrite.setChecked(False)

        grid.addWidget(QLabel("Output folder:"), 0, 0)
        grid.addWidget(self.out_dir_edit, 0, 1)
        grid.addWidget(self.btn_out_dir, 0, 2)

        grid.addWidget(QLabel("Output name pattern:"), 1, 0)
        grid.addWidget(self.name_pattern_edit, 1, 1, 1, 2)

        grid.addWidget(QLabel("Export variables:"), 2, 0)
        grid.addWidget(self.vars_edit, 2, 1, 1, 2)

        grid.addWidget(self.chk_compress, 3, 0, 1, 3)
        grid.addWidget(self.chk_overwrite, 4, 0, 1, 3)

        # --- Run group
        run_row = QHBoxLayout()
        self.btn_convert = QPushButton("Convert")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        run_row.addWidget(self.btn_convert)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.progress, 1)

        # --- Log
        g_log = QGroupBox("Log")
        log_layout = QVBoxLayout(g_log)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log_box)

        layout.addWidget(g_in, 3)
        layout.addWidget(g_out, 0)
        layout.addLayout(run_row)
        layout.addWidget(g_log, 2)

        # signals
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_remove.clicked.connect(self.on_remove_selected)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_out_dir.clicked.connect(self.on_choose_out_dir)
        self.btn_convert.clicked.connect(self.on_convert)
        self.btn_cancel.clicked.connect(self.on_cancel)

    # ---------- UI helpers ----------
    def append_log(self, text: str):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def set_busy(self, busy: bool):
        self.btn_convert.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_add.setEnabled(not busy)
        self.btn_remove.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)
        self.btn_out_dir.setEnabled(not busy)

    # ---------- actions ----------
    def on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .mat file(s)",
            "",
            "MAT files (*.mat);;All files (*.*)",
        )
        for f in files:
            self.file_list.addItem(f)

    def on_remove_selected(self):
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def on_clear(self):
        self.file_list.clear()

    def on_choose_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        if d:
            self.out_dir_edit.setText(d)

    def on_convert(self):
        mat_files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        out_dir = self.out_dir_edit.text().strip()
        name_pattern = self.name_pattern_edit.text().strip() or "{stem}.npz"
        vars_text = self.vars_edit.text().strip()
        compress = self.chk_compress.isChecked()
        overwrite = self.chk_overwrite.isChecked()

        if not mat_files:
            QMessageBox.warning(self, "Missing input", "Please add at least one .mat file.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Missing output", "Please choose an output folder.")
            return

        # quick pattern sanity
        try:
            _ = name_pattern.format(stem="demo", name="demo.mat", idx=1)
        except Exception as e:
            QMessageBox.warning(self, "Bad name pattern", f"Your output name pattern is invalid:\n{e}")
            return

        self.progress.setValue(0)
        self.append_log("====================================================")
        self.append_log("Convert started.")
        self.set_busy(True)

        self.worker = ConvertWorker(
            mat_files=mat_files,
            out_dir=out_dir,
            name_pattern=name_pattern,
            vars_text=vars_text,
            compress=compress,
            overwrite=overwrite,
        )
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def on_cancel(self):
        if self.worker:
            self.worker.cancel()
            self.append_log("Cancel requested…")

    def on_done(self, ok: bool, message: str):
        self.set_busy(False)
        self.append_log(message)
        if ok:
            QMessageBox.information(self, "Done", message)
        else:
            QMessageBox.critical(self, "Stopped", message)
        self.worker = None


def main():
    app = QApplication([])
    w = MatToNpzGUI()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
