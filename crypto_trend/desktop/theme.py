"""Light theme palette + QSS for the AlphaPulse desktop GUI."""
from __future__ import annotations

from pathlib import Path

# ---- semantic colors --------------------------------------------------- #
GREEN = "#1faa59"     # +자산
RED = "#d2474d"       # -자산
GRAY = "#7a8085"      # 평소 / 거래 전
ACCENT = "#2654f0"    # AlphaPulse indigo
ACCENT_DEEP = "#0c248c"
SURFACE = "#ffffff"
SURFACE_ALT = "#f6f8fa"
BORDER = "#e3e7ec"
TEXT = "#1f2329"
SUBTEXT = "#586069"


def color_for_delta(delta: float, neutral: str = GRAY) -> str:
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return neutral


# ---- Qt Style Sheet ---------------------------------------------------- #
QSS = f"""
QMainWindow, QWidget#root {{
    background-color: {SURFACE_ALT};
    color: {TEXT};
}}

QFrame#card {{
    background-color: {SURFACE};
    border-radius: 14px;
    border: 1px solid {BORDER};
}}

QLabel#h1 {{ font-size: 20px; font-weight: 600; color: {TEXT}; }}
QLabel#h2 {{ font-size: 16px; font-weight: 600; color: {TEXT}; }}
QLabel#caption {{ color: {SUBTEXT}; font-size: 12px; }}
QLabel#equity {{ font-size: 32px; font-weight: 700; }}
QLabel#equityDelta {{ font-size: 14px; font-weight: 600; }}
QLabel#bigStatus {{ font-size: 18px; font-weight: 700; }}
QLabel#mode {{
    background-color: #eaf2ff; color: {ACCENT_DEEP};
    padding: 4px 10px; border-radius: 10px; font-weight: 600;
}}
QLabel#modeLive {{
    background-color: #ffe7e9; color: {RED};
    padding: 4px 10px; border-radius: 10px; font-weight: 700;
}}

QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}}
QPushButton:hover  {{ background-color: {ACCENT_DEEP}; }}
QPushButton:disabled {{ background-color: #b9c0cc; color: #f0f2f5; }}

QPushButton#ghost {{
    background-color: transparent; color: {ACCENT};
    border: 1px solid {ACCENT};
}}
QPushButton#ghost:hover {{ background-color: #eef3ff; }}

QPushButton#warn   {{ background-color: #f0a000; color: white; }}
QPushButton#warn:hover {{ background-color: #c98700; }}
QPushButton#danger {{ background-color: {RED}; color: white; }}
QPushButton#danger:hover {{ background-color: #b8383d; }}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background-color: {SURFACE}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 10px; selection-background-color: {ACCENT};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; }}

QListWidget {{
    background-color: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 4px;
}}
QListWidget::item {{ padding: 6px 4px; border-bottom: 1px solid {BORDER}; }}
QListWidget::item:selected {{ background-color: #eaf2ff; color: {TEXT}; }}

QStatusBar {{ background-color: {SURFACE}; border-top: 1px solid {BORDER}; }}

QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #c8ced8; min-height: 24px; border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{ background: #a9b2bf; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


def background_path() -> Path:
    from ..config import ASSETS_DIR
    return ASSETS_DIR / "background.png"


def icon_path() -> Path:
    from ..config import ASSETS_DIR
    p = ASSETS_DIR / "icon.ico"
    return p if p.exists() else ASSETS_DIR / "icon.png"
