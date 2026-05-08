"""AlphaPulse main window — every operation is reachable from here.

No CLI needed: mode toggle, API credentials, strategy parameters, start /
stop / halt / resume, symbol picker, chart view, equity card, OOS card and
trade-message log all live in this single window.
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame,
                                QGraphicsDropShadowEffect, QHBoxLayout,
                                QLabel, QListWidget, QListWidgetItem,
                                QMainWindow, QMessageBox, QPushButton,
                                QSizePolicy, QSpinBox, QStatusBar,
                                QVBoxLayout, QWidget)

from .. import config
from ..config import DOCS_DIR, SETTINGS, TradingMode
from ..portfolio.state import PortfolioState
from ..strategy.trend_following import StrategyParams
from .chart_view import SignalChartView
from .credentials import CredentialStore
from .credentials_dialog import CredentialsDialog
from .help_dialog import HelpDialog
from .theme import (ACCENT, BORDER, GRAY, GREEN, QSS, RED, SUBTEXT, SURFACE,
                    TEXT, background_path, color_for_delta, icon_path)
from .workers import EngineWorker


def _shadow(widget: QWidget) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(18)
    eff.setOffset(0, 2)
    eff.setColor(Qt.GlobalColor.lightGray)
    widget.setGraphicsEffect(eff)


def _card() -> QFrame:
    f = QFrame()
    f.setObjectName("card")
    f.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    _shadow(f)
    return f


# --------------------------------------------------------------------------- #
class AlphaPulseWindow(QMainWindow):
    TITLE = "AlphaPulse · Crypto Trend Following — Bitget"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(QIcon(str(icon_path())))
        self.resize(1480, 920)
        self.setStyleSheet(QSS)

        self.portfolio = PortfolioState()
        self.portfolio.equity_usdt = SETTINGS.base_equity_usdt
        self.portfolio.mode = SETTINGS.mode.value
        self.worker: EngineWorker | None = None
        self.credentials_store = CredentialStore()

        self._set_background()
        self._build_ui()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start(2000)

    # ------------------------------------------------------------------ #
    def _set_background(self) -> None:
        bg = QPixmap(str(background_path()))
        if bg.isNull():
            return
        scaled = bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding,
                           Qt.SmoothTransformation)
        palette = self.palette()
        palette.setBrush(QPalette.Window, scaled)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def resizeEvent(self, event):                                       # noqa: N802
        self._set_background()
        super().resizeEvent(event)

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(12)

        # ---- header strip --------------------------------------------- #
        outer.addLayout(self._header())

        # ---- equity / OOS / mode cards -------------------------------- #
        outer.addLayout(self._top_cards())

        # ---- main grid: left controls | center chart | right log ------ #
        grid = QHBoxLayout()
        grid.setSpacing(12)
        grid.addWidget(self._control_panel(), 0)
        grid.addWidget(self._chart_panel(),   1)
        grid.addWidget(self._messages_panel(), 0)
        outer.addLayout(grid, 1)

        # ---- status bar ----------------------------------------------- #
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("준비됨 — 엔진을 시작하세요")
        sb.addWidget(self.status_label)

    # ------------------------------------------------------------------ #
    def _header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        icon = QLabel()
        pix = QPixmap(str(icon_path())).scaled(36, 36, Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation)
        icon.setPixmap(pix)
        title = QLabel("AlphaPulse")
        title.setObjectName("h1")
        subtitle = QLabel("Crypto Trend Following · Bitget USDT-Perp")
        subtitle.setObjectName("caption")

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.mode_badge = QLabel("PAPER")
        self.mode_badge.setObjectName("mode")
        self.mode_badge.setAlignment(Qt.AlignCenter)

        row.addWidget(icon)
        row.addLayout(title_box)
        row.addStretch(1)
        row.addWidget(self.mode_badge)
        return row

    # ------------------------------------------------------------------ #
    def _top_cards(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        # Equity card
        eq = _card()
        eql = QVBoxLayout(eq)
        eql.setContentsMargins(20, 16, 20, 16)
        cap = QLabel("자산 (USDT)"); cap.setObjectName("caption")
        self.equity_value = QLabel("—"); self.equity_value.setObjectName("equity")
        self.equity_value.setStyleSheet(f"color:{GRAY};")
        self.equity_delta = QLabel("Δ ±0.00"); self.equity_delta.setObjectName("equityDelta")
        self.equity_delta.setStyleSheet(f"color:{GRAY};")
        eql.addWidget(cap); eql.addWidget(self.equity_value); eql.addWidget(self.equity_delta)
        row.addWidget(eq, 1)

        # OOS card
        oos = _card()
        ol = QVBoxLayout(oos)
        ol.setContentsMargins(20, 16, 20, 16)
        cap2 = QLabel("OOS 게이트 상태"); cap2.setObjectName("caption")
        self.oos_status = QLabel("—"); self.oos_status.setObjectName("bigStatus")
        self.oos_detail = QLabel("SR=· PSR=· DSR=·"); self.oos_detail.setObjectName("caption")
        ol.addWidget(cap2); ol.addWidget(self.oos_status); ol.addWidget(self.oos_detail)
        row.addWidget(oos, 1)

        # Halt card
        hb = _card()
        hl = QVBoxLayout(hb)
        hl.setContentsMargins(20, 16, 20, 16)
        cap3 = QLabel("매매 제어"); cap3.setObjectName("caption")
        self.halt_status = QLabel("정상"); self.halt_status.setObjectName("bigStatus")
        self.halt_status.setStyleSheet(f"color:{GREEN};")
        btns = QHBoxLayout()
        self.btn_halt = QPushButton("⏸ 매매 중단"); self.btn_halt.setObjectName("warn")
        self.btn_resume = QPushButton("▶ 재개"); self.btn_resume.setObjectName("ghost")
        self.btn_halt.clicked.connect(self._on_halt)
        self.btn_resume.clicked.connect(self._on_resume)
        btns.addWidget(self.btn_halt); btns.addWidget(self.btn_resume)
        hl.addWidget(cap3); hl.addWidget(self.halt_status); hl.addLayout(btns)
        row.addWidget(hb, 1)

        return row

    # ------------------------------------------------------------------ #
    def _control_panel(self) -> QWidget:
        card = _card()
        card.setMinimumWidth(320)
        card.setMaximumWidth(360)
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(10)

        head_row = QHBoxLayout()
        head = QLabel("⚙  엔진 제어"); head.setObjectName("h2")
        head_row.addWidget(head)
        head_row.addStretch(1)
        head_row.addWidget(self._info_button(
            "engine_pipeline.html",
            "엔진 / 매매 제어 — 작동 원리"))
        v.addLayout(head_row)

        # Mode picker
        v.addWidget(self._caption("거래 모드"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["paper (모의)", "live (실거래)"])
        self.mode_combo.setCurrentIndex(0 if SETTINGS.mode == TradingMode.PAPER else 1)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        v.addWidget(self.mode_combo)

        # Credentials — never displayed in plaintext on the main window.
        v.addWidget(self._caption("API 자격증명 (live 모드 전용)"))
        self.creds_status = QLabel()
        self.creds_status.setWordWrap(True)
        self.creds_status.setStyleSheet(f"color:{GRAY}; padding:4px 0;")
        v.addWidget(self.creds_status)
        creds_row = QHBoxLayout()
        self.btn_set_creds = QPushButton("🔐 설정 / 변경")
        self.btn_set_creds.setObjectName("ghost")
        self.btn_set_creds.clicked.connect(self._on_set_credentials)
        self.btn_clear_creds = QPushButton("삭제")
        self.btn_clear_creds.setObjectName("ghost")
        self.btn_clear_creds.clicked.connect(self._on_clear_credentials)
        creds_row.addWidget(self.btn_set_creds)
        creds_row.addWidget(self.btn_clear_creds)
        v.addLayout(creds_row)
        self._refresh_creds_status()

        # Strategy params
        v.addWidget(self._caption("전략 파라미터"))
        self.lookback_spin = self._spin_int("스크리너 lookback (bars)", 10, 200, 24)
        self.z_spin = self._spin_float("robust-z 임계", 1.0, 6.0, 2.5, 0.1)
        self.cvar_spin = self._spin_float("CVaR 하한 (음수)", -0.30, -0.01,
                                          SETTINGS.cvar_floor_pct, 0.005)
        self.lev_spin = self._spin_float("최대 레버리지", 1.0, 10.0,
                                         SETTINGS.max_gross_leverage, 0.5)
        self.equity_spin = self._spin_float("페이퍼 시작자본 USDT", 100.0, 1_000_000.0,
                                            SETTINGS.base_equity_usdt, 100.0)
        for w in (self.lookback_spin, self.z_spin, self.cvar_spin,
                   self.lev_spin, self.equity_spin):
            v.addWidget(w)

        # Cycle period
        self.period_spin = self._spin_int("사이클 주기 (초)", 60, 86400, 3600)
        v.addWidget(self.period_spin)

        v.addStretch(1)

        self.btn_start = QPushButton("▶ 엔진 시작")
        self.btn_stop = QPushButton("■ 엔진 정지"); self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        v.addWidget(self.btn_start)
        v.addWidget(self.btn_stop)

        return card

    def _caption(self, txt: str) -> QLabel:
        l = QLabel(txt); l.setObjectName("caption"); return l

    def _info_button(self, doc_filename: str, title: str) -> QPushButton:
        """Small ⓘ button that opens a bundled HTML doc in a HelpDialog."""
        btn = QPushButton("ⓘ")
        btn.setObjectName("info")
        btn.setFixedSize(28, 28)
        btn.setToolTip(f"설명 보기 — {title}")
        btn.clicked.connect(lambda: self._open_help(doc_filename, title))
        return btn

    def _open_help(self, doc_filename: str, title: str) -> None:
        path = DOCS_DIR / doc_filename
        dlg = HelpDialog(self, path, title)
        dlg.exec()

    def _spin_int(self, label: str, lo: int, hi: int, val: int) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label); lbl.setObjectName("caption"); lbl.setMinimumWidth(160)
        sp = QSpinBox(); sp.setRange(lo, hi); sp.setValue(val)
        sp.setMinimumWidth(110)
        h.addWidget(lbl); h.addWidget(sp); h.addStretch(1)
        w._spin = sp                                                  # type: ignore[attr-defined]
        return w

    def _spin_float(self, label: str, lo: float, hi: float,
                    val: float, step: float = 0.1) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label); lbl.setObjectName("caption"); lbl.setMinimumWidth(160)
        sp = QDoubleSpinBox(); sp.setRange(lo, hi); sp.setValue(val)
        sp.setSingleStep(step); sp.setDecimals(4)
        sp.setMinimumWidth(110)
        h.addWidget(lbl); h.addWidget(sp); h.addStretch(1)
        w._spin = sp                                                  # type: ignore[attr-defined]
        return w

    @staticmethod
    def _val(widget: QWidget):
        return widget._spin.value()                                  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    def _chart_panel(self) -> QWidget:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel("📈 신호 차트")
        title.setObjectName("h2")
        head.addWidget(title)
        head.addWidget(self._info_button(
            "chart_signals.html",
            "신호 차트 — A/B/C 일관 랜더링"))
        head.addStretch(1)
        head.addWidget(self._caption("심볼"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.setMinimumWidth(220)
        self.symbol_combo.currentTextChanged.connect(lambda _t: self._render_chart())
        head.addWidget(self.symbol_combo)
        v.addLayout(head)

        self.chart = SignalChartView()
        v.addWidget(self.chart, 1)
        self.chart.clear()
        return card

    # ------------------------------------------------------------------ #
    def _messages_panel(self) -> QWidget:
        card = _card()
        card.setMinimumWidth(340)
        card.setMaximumWidth(420)
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(8)

        head = QLabel("📜 거래 메시지"); head.setObjectName("h2")
        v.addWidget(head)
        self.messages = QListWidget()
        v.addWidget(self.messages, 1)
        return card

    # ================================================================== #
    # Event handlers
    # ================================================================== #
    def _on_mode_change(self, idx: int) -> None:
        new_mode = TradingMode.LIVE if idx == 1 else TradingMode.PAPER
        config.apply(mode=new_mode)
        self.portfolio.mode = new_mode.value

    def _gather_settings(self) -> bool:
        """Push GUI values into config.SETTINGS. Returns False if invalid.

        Credentials are *only* loaded from the OS keyring, never from any
        widget on the main window.  They live in ``SETTINGS`` for the
        duration of an engine run and are wiped on stop.
        """
        mode = TradingMode.LIVE if self.mode_combo.currentIndex() == 1 else TradingMode.PAPER
        creds = self.credentials_store.load()
        if mode == TradingMode.LIVE and not creds.is_complete:
            QMessageBox.warning(
                self, "API 키 미설정",
                "live 모드에는 Bitget API 자격증명이 필요합니다. "
                "🔐 설정 / 변경 버튼을 눌러 OS 보안 저장소에 먼저 저장해 주세요.")
            return False
        config.apply(
            mode=mode,
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            api_passphrase=creds.api_passphrase,
            cvar_floor_pct=float(self._val(self.cvar_spin)),
            max_gross_leverage=float(self._val(self.lev_spin)),
            base_equity_usdt=float(self._val(self.equity_spin)),
        )
        # We do NOT keep credentials in the local Credentials object after this.
        creds.wipe()
        self.portfolio.mode = mode.value
        if self.portfolio.equity_usdt <= 0:
            self.portfolio.equity_usdt = SETTINGS.base_equity_usdt
        return True

    # ---- credential handlers ----------------------------------------- #
    def _refresh_creds_status(self) -> None:
        configured = self.credentials_store.is_configured()
        backend = self.credentials_store.backend_name
        if configured:
            self.creds_status.setText(f"✓ 설정됨 ({backend})")
            self.creds_status.setStyleSheet(f"color:{GREEN}; padding:4px 0;")
        else:
            self.creds_status.setText(f"미설정 — live 모드 전 설정 필요\n저장소: {backend}")
            self.creds_status.setStyleSheet(f"color:{GRAY}; padding:4px 0;")

    def _on_set_credentials(self) -> None:
        # Empty prefill — never re-show the saved key, even masked.
        dlg = CredentialsDialog(self, self.credentials_store, prefill=None)
        dlg.exec()
        self._refresh_creds_status()

    def _on_clear_credentials(self) -> None:
        confirm = QMessageBox.question(
            self, "자격증명 삭제",
            "OS 보안 저장소에서 Bitget 자격증명을 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.Cancel)
        if confirm == QMessageBox.Yes:
            self.credentials_store.clear()
            # Also wipe whatever might be lingering in SETTINGS.
            config.apply(api_key="", api_secret="", api_passphrase="")
            self._refresh_creds_status()

    def _on_start(self) -> None:
        if self.worker is not None:
            return
        if not self._gather_settings():
            return
        if SETTINGS.mode == TradingMode.LIVE:
            confirm = QMessageBox.question(
                self, "실거래 모드 확인",
                "live 모드로 전환하면 Bitget 계정에 실제 주문이 전송됩니다.\n계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.Cancel)
            if confirm != QMessageBox.Yes:
                return
        self.worker = EngineWorker(self.portfolio,
                                    period_seconds=int(self._val(self.period_spin)))
        self.worker.cycle_done.connect(lambda eq: self.status_label.setText(
            f"사이클 완료 · 자산 {eq:,.2f} USDT"))
        self.worker.cycle_error.connect(lambda msg: self.status_label.setText(
            f"⚠ 오류: {msg}"))
        self.worker.halted.connect(self._on_engine_halt)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("엔진 가동 중…")

    def _on_stop(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.wait(5000)
        self.worker = None
        # Wipe credentials from process memory once the engine is no longer
        # using them. The keyring still holds them for the next run.
        config.apply(api_key="", api_secret="", api_passphrase="")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("엔진 정지됨")

    def _on_halt(self) -> None:
        self.portfolio.halt("사용자 수동 중단")

    def _on_resume(self) -> None:
        self.portfolio.resume()

    def _on_engine_halt(self, reason: str) -> None:
        QMessageBox.critical(
            self, "⚠ 매매 자동 중단",
            f"OOS 자가보정 실패로 매매가 자동 중단되었습니다.\n\n사유: {reason}\n\n"
            "전략 파라미터를 조정한 뒤 ‘재개’를 눌러주세요.")

    # ================================================================== #
    # Periodic refresh
    # ================================================================== #
    def _refresh(self) -> None:
        # equity
        eq = self.portfolio.equity_usdt
        delta = self.portfolio.equity_delta
        col = color_for_delta(delta, neutral=GRAY)
        self.equity_value.setText(f"{eq:,.2f}")
        self.equity_value.setStyleSheet(f"color:{col};")
        sign = "+" if delta > 0 else ("" if delta < 0 else "±")
        self.equity_delta.setText(f"Δ {sign}{delta:,.2f} USDT")
        self.equity_delta.setStyleSheet(f"color:{col};")

        # mode badge
        if self.portfolio.mode == "live":
            self.mode_badge.setObjectName("modeLive"); self.mode_badge.setText("LIVE")
        else:
            self.mode_badge.setObjectName("mode"); self.mode_badge.setText("PAPER")
        self.mode_badge.setStyleSheet(QSS)
        self.mode_badge.style().unpolish(self.mode_badge)
        self.mode_badge.style().polish(self.mode_badge)

        # OOS card
        oos = self.portfolio.last_oos or {}
        st = oos.get("status", "—").upper()
        oos_color = (GREEN if st == "OK" else
                     RED if st == "HALTED" else
                     "#c08a00" if st == "RECALIBRATED" else
                     "#0c248c" if st == "PENDING" else GRAY)
        self.oos_status.setText(st)
        self.oos_status.setStyleSheet(f"color:{oos_color};")
        self.oos_detail.setText(
            f"SR={oos.get('sharpe', 0):.2f}  PSR={oos.get('psr', 0):.2f}  "
            f"DSR={oos.get('dsr', 0):.2f}  attempts={oos.get('attempts', 0)}")

        # halt card
        if self.portfolio.halted:
            self.halt_status.setText("⚠ 중단됨")
            self.halt_status.setStyleSheet(f"color:{RED};")
        else:
            self.halt_status.setText("정상")
            self.halt_status.setStyleSheet(f"color:{GREEN};")

        # symbol dropdown sync
        syms = sorted({s.symbol for s in self.portfolio.signals if s.symbol})
        cur = self.symbol_combo.currentText()
        existing = [self.symbol_combo.itemText(i) for i in range(self.symbol_combo.count())]
        if syms != existing:
            self.symbol_combo.blockSignals(True)
            self.symbol_combo.clear()
            self.symbol_combo.addItems(syms)
            if cur in syms:
                self.symbol_combo.setCurrentText(cur)
            self.symbol_combo.blockSignals(False)
            self._render_chart()

        # message log
        target_count = len(self.portfolio.messages)
        if target_count != self.messages.count():
            self.messages.clear()
            for m in reversed(self.portfolio.messages[-100:]):
                ts = m.ts.strftime("%H:%M:%S") if isinstance(m.ts, pd.Timestamp) else str(m.ts)
                if m.delta_usdt > 0:
                    color = GREEN
                elif m.delta_usdt < 0:
                    color = RED
                elif m.kind == "halt":
                    color = RED
                elif m.kind == "warning":
                    color = "#c08a00"
                else:
                    color = GRAY
                item = QListWidgetItem(f"[{ts}]  {m.text}")
                item.setForeground(Qt.GlobalColor.darkGray)
                font = QFont(); font.setWeight(QFont.DemiBold)
                item.setFont(font)
                from PySide6.QtGui import QColor
                item.setForeground(QColor(color))
                self.messages.addItem(item)

    # ------------------------------------------------------------------ #
    def _render_chart(self) -> None:
        sym = self.symbol_combo.currentText()
        if not sym:
            self.chart.clear()
            return
        candles = self.worker.candles_for(sym) if self.worker else None
        if candles is None or candles.empty:
            self.chart.clear()
            return
        sigs = [s for s in self.portfolio.signals if s.symbol == sym]
        self.chart.render(sym, candles, sigs)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event):                                       # noqa: N802
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(event)
