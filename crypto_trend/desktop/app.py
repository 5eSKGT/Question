"""Desktop GUI entry-point. ``python -m crypto_trend`` launches this."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication

from .main_window import AlphaPulseWindow
from .theme import icon_path


def main() -> int:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AlphaPulse")
    app.setOrganizationName("AlphaPulse")
    app.setWindowIcon(QIcon(str(icon_path())))
    app.setFont(QFont("Segoe UI", 10))

    window = AlphaPulseWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
