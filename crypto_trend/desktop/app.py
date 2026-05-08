"""Desktop GUI entry-point. ``python -m crypto_trend`` launches this."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication

from .main_window import AlphaPulseWindow
from .theme import icon_path


# Used by Windows to group taskbar entries and select the icon shown next to
# them. See: SetCurrentProcessExplicitAppUserModelID. Must be a unique string
# of the form CompanyName.ProductName.SubProduct.VersionInformation.
APP_USER_MODEL_ID = "AlphaPulse.CryptoTrendFollowing.Desktop.1"


def _register_windows_appid() -> None:
    """Tell Windows this is its own application so the taskbar uses our icon.

    Without this call, every ``pythonw.exe`` on the system shares one
    AppUserModelID, and the taskbar therefore falls back to the generic
    Python icon — even though ``app.setWindowIcon(...)`` made the window's
    own title-bar icon correct. This function is the missing piece that
    propagates the AlphaPulse icon to the taskbar, alt-tab switcher, and
    Start menu pin.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID)
    except Exception:                                              # noqa: BLE001
        # Non-fatal: the app still runs, just with the wrong taskbar icon.
        pass


def main() -> int:
    _register_windows_appid()

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AlphaPulse")
    app.setOrganizationName("AlphaPulse")
    app.setDesktopFileName("AlphaPulse")          # affects taskbar grouping on Linux

    icon = QIcon(str(icon_path()))
    app.setWindowIcon(icon)
    app.setFont(QFont("Segoe UI", 10))

    window = AlphaPulseWindow()
    window.setWindowIcon(icon)                     # explicit again — cheap insurance
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
