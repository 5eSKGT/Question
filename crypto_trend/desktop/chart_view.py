"""QWebEngineView wrapper that renders the same plotly figure used by the
Dash UI — so the A/B/C signal-style invariant defined in ``ui/charts.py`` is
reused verbatim inside the desktop GUI.

WebEngine is imported lazily so the module is importable on minimal systems
(CI, headless test runners) where Qt WebEngine is not provisioned.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from ..strategy.trend_following import Signal
from ..ui.charts import build_signal_chart


class SignalChartView(QWidget):
    """Light-themed embedded chart with consistent A/B/C rendering."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.web = QWebEngineView()
        except ImportError as e:                                       # noqa: BLE001
            from PySide6.QtWidgets import QLabel
            self.web = QLabel(
                f"Qt WebEngine 사용 불가 — 차트를 표시할 수 없습니다.\n{e}")
            self.web.setStyleSheet("color:#7a8085; padding:24px;")
        self.web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.web)
        self._last_html: str | None = None

    # ------------------------------------------------------------------ #
    def render(self, symbol: str, candles: pd.DataFrame,
               signals: Iterable[Signal]) -> None:
        if not hasattr(self.web, "setHtml"):
            return
        import plotly.io as pio
        fig = build_signal_chart(symbol, candles, list(signals))
        fig.update_layout(paper_bgcolor="rgba(255,255,255,0)")
        html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True,
                           config={"displaylogo": False, "responsive": True})
        if html == self._last_html:
            return
        self._last_html = html
        self.web.setHtml(html)

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        if not hasattr(self.web, "setHtml"):
            return
        self.web.setHtml(
            "<html><body style='background:transparent;font-family:sans-serif;"
            "color:#7a8085;display:flex;align-items:center;justify-content:center;"
            "height:100vh;margin:0;'><div>심볼을 선택하면 차트가 표시됩니다</div>"
            "</body></html>"
        )
