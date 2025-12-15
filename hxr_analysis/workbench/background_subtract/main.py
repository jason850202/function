from __future__ import annotations

from .view import BackgroundSubtractWindow
from PyQt6.QtWidgets import QApplication


def run():
    app = QApplication.instance() or QApplication([])
    window = BackgroundSubtractWindow()
    window.show()
    app.exec()


__all__ = ["run", "BackgroundSubtractWindow"]
