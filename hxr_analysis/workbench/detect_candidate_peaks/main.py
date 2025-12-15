from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from .view import DetectCandidatePeaksWindow


def run():
    app = QApplication.instance() or QApplication([])
    window = DetectCandidatePeaksWindow()
    window.show()
    app.exec()


__all__ = ["run", "DetectCandidatePeaksWindow"]
