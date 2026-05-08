"""Embedded documentation viewer.

Loads a bundled .html file (with inline CSS + SVG) into a Qt window,
so the production UI itself stays uncluttered while the user can still
reach detailed explanations from a small ⓘ button.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from .theme import icon_path


class HelpDialog(QDialog):
    def __init__(self, parent, html_path: Path, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(str(icon_path())))
        self.resize(1100, 820)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        if not html_path.exists():
            v.addWidget(QLabel(f"문서 파일을 찾을 수 없습니다:\n{html_path}"))
            return

        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.view = QWebEngineView()
            self.view.load(QUrl.fromLocalFile(str(html_path.resolve())))
            v.addWidget(self.view)
        except ImportError:
            # Fall back to opening the user's default browser
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(html_path.resolve())))
            v.addWidget(QLabel(
                f"Qt WebEngine 미설치 — 기본 브라우저로 문서를 열었습니다:\n{html_path}"))
            self.close()
