"""Backtest configuration + execution inside the GUI.

Design contract — backtest must mirror the live engine
------------------------------------------------------
The system parameters that determine *what data the strategy sees*
(universe filter, history depth, walk-forward windows) are pulled
**directly from live config**, NOT exposed as user-tunable knobs. The
user can only tune the *strategy* (CVaR, Kelly, vol target, sizing
cap, leverage cap) and the cost model (taker fee, slippage). This
preserves the only useful invariant of a backtest: that its
environment matches production.

Visible knobs:                       Hidden / auto-from-live:
  * data source (synthetic|bitget)     * bars             ← engine.history_bars
  * strategy parameters                * train_days       ← walk_forward_train_days
  * preset buttons                     * test_days        ← walk_forward_test_days
  * cost model                         * universe filter  ← screener.min_quote_volume
                                       * top-N            ← entire universe (live mirror)

Synthetic mode keeps a tiny "advanced" disclosure with seeds + n_symbols
for reproducible offline preview, but the defaults match a typical
live universe size.

Threading
---------
The simulator runs in a QThread (BacktestWorker) and streams progress
+ results back via Qt signals so the dialog stays responsive.
"""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                                QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                                QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
                                QPushButton, QScrollArea, QSpinBox,
                                QTableWidget, QTableWidgetItem, QTabWidget,
                                QVBoxLayout, QWidget)

from ..config import STATE_DIR
from .theme import (ACCENT, ACCENT_DEEP, GRAY, GREEN, RED, SUBTEXT,
                    SURFACE, icon_path)


# --------------------------------------------------------------------------- #
class BacktestWorker(QThread):
    """Runs the backtest in a worker thread, emits structured events."""

    log = Signal(str)                # progress lines for the console panel
    result = Signal(dict)            # final result dict
    failed = Signal(str)             # error string

    def __init__(self, params: dict, parent: QObject | None = None):
        super().__init__(parent)
        self.p = params
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ #
    def run(self) -> None:                                              # noqa: D401
        try:
            from ..backtest.simulator import StrategySimulator
            from ..backtest.walk_forward import walk_forward_run
            from ..backtest.metrics import compute_metrics
            from ..backtest.__main__ import (_baseline_buyhold,
                                                _baseline_naive_momentum)
            from ..backtest.data_loader import (synthetic_universe,
                                                  fetch_bitget_universe,
                                                  fetch_bitget_ohlcv)
            from ..strategy.trend_following import (StrategyParams,
                                                      TrendFollowingStrategy)

            train_bars = self.p["train_days"] * 24
            test_bars = self.p["test_days"] * 24
            seeds = list(range(self.p["seeds"]))

            per_seed_rows: list[dict] = []
            agg = {"alphapulse": [], "buyhold": [], "naive_momentum": []}

            for i, seed in enumerate(seeds):
                if self._stop:
                    self.log.emit("⚠ aborted")
                    return

                self.log.emit(f"[{i+1}/{len(seeds)}] preparing universe …")

                if self.p["source"] == "synthetic":
                    candles = synthetic_universe(self.p["n_symbols"],
                                                  self.p["bars"], seed=seed)
                else:
                    top = self.p["top"] or None     # 0 → None → full universe
                    self.log.emit(f"  pulling Bitget USDT-perps "
                                    f"({'top-' + str(top) if top else 'full universe'}) …")
                    universe = fetch_bitget_universe(top_k=top)
                    self.log.emit(f"  {len(universe)} symbols qualify")
                    candles = {}
                    for j, s in enumerate(universe):
                        if self._stop:
                            return
                        try:
                            candles[s] = fetch_bitget_ohlcv(s, "1h", self.p["bars"])
                            if (j + 1) % 10 == 0:
                                self.log.emit(f"  downloaded {j+1}/{len(universe)}")
                        except Exception as e:                         # noqa: BLE001
                            self.log.emit(f"  skip {s}: {e}")
                    if len(seeds) > 1:
                        self.log.emit("  bitget mode runs once — ignoring extra seeds")
                        seeds = [seed]

                if not candles:
                    self.log.emit("  ⚠ no candles loaded — skipping seed")
                    continue

                # ---- strategy parameters from dialog --------------------- #
                strat_params = StrategyParams(
                    cvar_floor=self.p["cvar_floor"],
                    cvar_alpha=self.p["cvar_alpha"],
                    target_annual_vol=self.p["target_vol"],
                    kelly_safety=self.p["kelly_safety"],
                    sizing_cap=self.p["sizing_cap"],
                    leverage_cap=self.p["leverage_cap"],
                )
                sim = StrategySimulator(
                    strategy=TrendFollowingStrategy(strat_params),
                    taker_fee=self.p["taker_fee"],
                    slippage_bps=self.p["slippage_bps"],
                )
                self.log.emit("  running AlphaPulse walk-forward …")
                wf = walk_forward_run(candles, train_bars=train_bars,
                                       test_bars=test_bars, simulator=sim)

                self.log.emit("  running baselines …")
                bh = _baseline_buyhold(candles, warmup=train_bars)
                nm = _baseline_naive_momentum(candles, warmup=train_bars)
                bh_m = compute_metrics(bh["bar_returns"], [], [], 1.0)
                nm_m = compute_metrics(nm["bar_returns"], [], [], 1.0)

                row = {
                    "seed": seed,
                    "alphapulse": wf.metrics.as_row(),
                    "buyhold":     bh_m.as_row(),
                    "naive":       nm_m.as_row(),
                }
                per_seed_rows.append(row)
                for name, m in (("alphapulse", wf.metrics),
                                  ("buyhold", bh_m),
                                  ("naive_momentum", nm_m)):
                    agg[name].append([
                        m.total_return, m.annualized_sharpe, m.sortino,
                        m.max_drawdown, m.calmar, m.exposure, m.n_trades,
                    ])

                self.log.emit(
                    f"  AP ret={wf.metrics.total_return:+.3%} "
                    f"sr={wf.metrics.annualized_sharpe:+.2f} "
                    f"mdd={wf.metrics.max_drawdown:+.3%} "
                    f"trades={wf.metrics.n_trades:.0f}")

            # ---- aggregate ------------------------------------------ #
            cols = ["return", "sharpe", "sortino", "mdd", "calmar",
                    "exposure", "n_trades"]
            summary = {}
            for k, rows in agg.items():
                a = np.asarray(rows)
                if a.size == 0:
                    continue
                summary[k] = {c: {"mean": float(a[:, j].mean()),
                                    "std": float(a[:, j].std(ddof=1) if len(a) > 1 else 0.0),
                                    "median": float(np.median(a[:, j]))}
                               for j, c in enumerate(cols)}

            ap_ret = np.array([r[0] for r in agg["alphapulse"]])
            bh_ret = np.array([r[0] for r in agg["buyhold"]])
            nm_ret = np.array([r[0] for r in agg["naive_momentum"]])
            wins = {
                "ap_beats_bh": int((ap_ret > bh_ret).sum()) if ap_ret.size else 0,
                "ap_beats_nm": int((ap_ret > nm_ret).sum()) if ap_ret.size else 0,
                "n": int(ap_ret.size),
            }

            payload = {"summary": summary, "per_seed": per_seed_rows,
                        "wins": wins, "params": self.p}
            try:
                (STATE_DIR / "last_backtest.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False))
            except Exception:                                          # noqa: BLE001
                pass
            self.result.emit(payload)
        except Exception as e:                                          # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------- #
class BacktestDialog(QDialog):
    """One-stop GUI for configuring + running backtests."""

    def __init__(self, parent, current_strategy_params=None):
        super().__init__(parent)
        self.setWindowTitle("📊 백테스트 — 구성 및 실행")
        self.setWindowIcon(QIcon(str(icon_path())))
        # Default size kept under a 1080-tall display minus taskbar (~1040
        # usable). The dialog is fully resizable and the form panel sits
        # inside a QScrollArea so even smaller laptops can scroll to every
        # control.
        self.resize(1080, 660)
        self.setMinimumSize(900, 520)
        self.setModal(True)

        self.worker: BacktestWorker | None = None
        self._cur = current_strategy_params

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        title = QLabel("📊 백테스트 — 한 화면에서 모든 환경 구성 + 실행")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        root.addWidget(title)

        intro = QLabel(
            "AlphaPulse · Buy-and-Hold · Naive Momentum 세 전략을 동일한 비용 모델 "
            "(taker 6 bps + slippage 1 bps) 아래 실행하고 결과를 비교합니다. "
            "synthetic 모드는 인터넷이 필요 없고, bitget 모드는 ccxt 로 실거래 시세를 "
            "내려받아 parquet 로 캐시합니다.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{SUBTEXT}; font-size:12px;")
        root.addWidget(intro)

        # ---- two-column layout: form / log+results --------------- #
        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_form_panel(), 0)
        body.addWidget(self._build_results_panel(), 1)
        root.addLayout(body)

        # ---- bottom buttons ----------------------------------------- #
        bottom = QHBoxLayout()
        self.btn_run = QPushButton("▶ 백테스트 실행")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop = QPushButton("■ 중단")
        self.btn_stop.setObjectName("ghost")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_close = QPushButton("닫기")
        self.btn_close.setObjectName("ghost")
        self.btn_close.clicked.connect(self.reject)
        bottom.addWidget(self.btn_run)
        bottom.addWidget(self.btn_stop)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_close)
        root.addLayout(bottom)

        self._restore_last_run()

    # ================================================================== #
    # Build sub-panels
    # ================================================================== #
    def _build_form_panel(self) -> QWidget:
        # Inner content widget — holds every form field
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 6, 0)
        v.setSpacing(8)

        head = QLabel("⚙  설정"); head.setStyleSheet("font-size:14px; font-weight:600;")
        v.addWidget(head)

        # Environment summary — read from live config, NOT editable.
        from ..config import SETTINGS
        env_card = QLabel(
            f"<b>실거래 환경 미러링</b><br>"
            f"• 유니버스: 전체 USDT-Perp · 거래대금 ≥ ${SETTINGS.base_equity_usdt and 5}M<br>"
            f"• 봉 수 / 타임프레임: 500 × 1h<br>"
            f"• Walk-forward: train {SETTINGS.walk_forward_train_days}d / "
            f"test {SETTINGS.walk_forward_test_days}d<br>"
            f"• 모드: paper · live 모두 동일 코드 경로<br>"
            f"<span style='color:{SUBTEXT}'>이 항목들은 라이브 엔진과 동일하게 "
            f"고정되어 사용자가 변경할 수 없습니다 — 그래야 백테스트 결과가 실제 "
            f"운용을 의미 있게 예측합니다.</span>")
        env_card.setWordWrap(True)
        env_card.setTextFormat(Qt.RichText)
        env_card.setStyleSheet(
            f"background:#f3f6fa; border:1px solid {ACCENT}; "
            "border-radius:8px; padding:10px; font-size:11.5px;")
        v.addWidget(env_card)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["synthetic (오프라인 미리보기)",
                                       "bitget (실거래 데이터)"])
        self.source_combo.currentIndexChanged.connect(self._on_source_change)
        form.addRow("데이터 소스", self.source_combo)

        # Synthetic-only knobs that *don't* break the live mirror — kept
        # because synthetic universes have no real "size" to inherit.
        self.seeds_spin = QSpinBox()
        self.seeds_spin.setRange(1, 200); self.seeds_spin.setValue(10)
        form.addRow("시드 수", self.seeds_spin)
        self.synth_seeds_label = form.labelForField(self.seeds_spin)

        self.n_sym_spin = QSpinBox()
        self.n_sym_spin.setRange(5, 200); self.n_sym_spin.setValue(40)
        form.addRow("합성 심볼 수", self.n_sym_spin)
        self.synth_nsym_label = form.labelForField(self.n_sym_spin)

        # Hidden/derived knobs — kept as members for the worker to read.
        # setRange must precede setValue, otherwise QSpinBox's default
        # range (0, 99) clamps the desired value silently.
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(100, 20000)
        self.bars_spin.setValue(500)                              # = engine.history_bars default
        self.bars_spin.hide()

        self.top_spin = QSpinBox()
        self.top_spin.setRange(0, 500)
        self.top_spin.setValue(0)                                  # 0 = full universe
        self.top_spin.hide()

        self.train_days_spin = QSpinBox()
        self.train_days_spin.setRange(1, 90)
        self.train_days_spin.setValue(SETTINGS.walk_forward_train_days)
        self.train_days_spin.hide()

        self.test_days_spin = QSpinBox()
        self.test_days_spin.setRange(1, 30)
        self.test_days_spin.setValue(SETTINGS.walk_forward_test_days)
        self.test_days_spin.hide()

        v.addLayout(form)

        sep = QLabel("전략 파라미터"); sep.setStyleSheet(
            f"color:{ACCENT_DEEP}; font-weight:600; padding-top:8px;")
        v.addWidget(sep)

        sform = QFormLayout()
        sform.setLabelAlignment(Qt.AlignRight)

        self.cvar_floor = QDoubleSpinBox(); self.cvar_floor.setRange(-0.5, -0.005)
        self.cvar_floor.setValue(-0.08); self.cvar_floor.setSingleStep(0.005)
        self.cvar_floor.setDecimals(4)
        sform.addRow("CVaR 하한", self.cvar_floor)

        self.cvar_alpha = QDoubleSpinBox(); self.cvar_alpha.setRange(0.005, 0.20)
        self.cvar_alpha.setValue(0.05); self.cvar_alpha.setSingleStep(0.005)
        self.cvar_alpha.setDecimals(4)
        sform.addRow("CVaR α", self.cvar_alpha)

        self.target_vol = QDoubleSpinBox(); self.target_vol.setRange(0.05, 1.0)
        self.target_vol.setValue(0.20); self.target_vol.setSingleStep(0.05)
        sform.addRow("Vol-target (연환산)", self.target_vol)

        self.kelly_safety = QDoubleSpinBox(); self.kelly_safety.setRange(0.0, 1.0)
        self.kelly_safety.setValue(0.5); self.kelly_safety.setSingleStep(0.05)
        sform.addRow("Kelly 안전계수", self.kelly_safety)

        self.sizing_cap = QDoubleSpinBox(); self.sizing_cap.setRange(0.05, 5.0)
        self.sizing_cap.setValue(1.0); self.sizing_cap.setSingleStep(0.1)
        sform.addRow("최대 사이즈 (× equity)", self.sizing_cap)

        self.leverage_cap = QSpinBox(); self.leverage_cap.setRange(1, 20)
        self.leverage_cap.setValue(3)
        sform.addRow("최대 레버리지", self.leverage_cap)

        v.addLayout(sform)

        sep2 = QLabel("비용 모델"); sep2.setStyleSheet(
            f"color:{ACCENT_DEEP}; font-weight:600; padding-top:8px;")
        v.addWidget(sep2)

        cform = QFormLayout()
        cform.setLabelAlignment(Qt.AlignRight)
        self.taker_fee = QDoubleSpinBox(); self.taker_fee.setRange(0.0, 0.005)
        self.taker_fee.setValue(0.0006); self.taker_fee.setSingleStep(0.0001)
        self.taker_fee.setDecimals(5)
        cform.addRow("Taker 수수료", self.taker_fee)
        self.slippage_bps = QDoubleSpinBox(); self.slippage_bps.setRange(0.0, 50.0)
        self.slippage_bps.setValue(1.0); self.slippage_bps.setSingleStep(0.5)
        cform.addRow("슬리피지 (bps)", self.slippage_bps)
        v.addLayout(cform)

        # ---- preset row -------------------------------------------- #
        sep3 = QLabel("프리셋")
        sep3.setStyleSheet(f"color:{ACCENT_DEEP}; font-weight:600; padding-top:8px;")
        v.addWidget(sep3)
        preset_help = QLabel(
            "방어형: 자본 보호 우선 — 하락장에 강함 / 상승장 비참여\n"
            "균형형: 모든 레짐 양수 수익 — Sortino 1.5–3 / raw return 약함\n"
            "공격형: 상승장 흡수력 ↑ — MDD 도 같이 커짐")
        preset_help.setStyleSheet(f"color:{SUBTEXT}; font-size:11px;")
        preset_help.setWordWrap(True)
        v.addWidget(preset_help)
        prow = QHBoxLayout()
        b1 = QPushButton("🛡 방어형")
        b1.setObjectName("ghost"); b1.clicked.connect(self._preset_defensive)
        b2 = QPushButton("⚖ 균형형")
        b2.setObjectName("ghost"); b2.clicked.connect(self._preset_balanced)
        b3 = QPushButton("🚀 공격형")
        b3.setObjectName("ghost"); b3.clicked.connect(self._preset_aggressive)
        prow.addWidget(b1); prow.addWidget(b2); prow.addWidget(b3)
        v.addLayout(prow)

        v.addStretch(1)
        self._on_source_change(self.source_combo.currentIndex())

        # Wrap in a scroll area so the form is always reachable on
        # smaller laptops where the dialog otherwise overflows the screen.
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    # ---- preset application ------------------------------------------ #
    def _preset_defensive(self) -> None:
        self.cvar_floor.setValue(-0.08); self.cvar_alpha.setValue(0.05)
        self.target_vol.setValue(0.20); self.kelly_safety.setValue(0.5)
        self.sizing_cap.setValue(1.0); self.leverage_cap.setValue(3)

    def _preset_balanced(self) -> None:
        self.cvar_floor.setValue(-0.10); self.cvar_alpha.setValue(0.05)
        self.target_vol.setValue(0.30); self.kelly_safety.setValue(1.0)
        self.sizing_cap.setValue(2.0); self.leverage_cap.setValue(3)

    def _preset_aggressive(self) -> None:
        self.cvar_floor.setValue(-0.15); self.cvar_alpha.setValue(0.05)
        self.target_vol.setValue(0.40); self.kelly_safety.setValue(1.0)
        self.sizing_cap.setValue(3.0); self.leverage_cap.setValue(5)

    def _build_results_panel(self) -> QWidget:
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        tabs = QTabWidget()

        # --- summary table ----------------------------------------- #
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["전략", "수익률(평균)", "수익률(중앙)", "Sharpe", "Sortino",
              "MDD", "노출", "trades(평균)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        tabs.addTab(self.table, "요약")

        # --- per-seed drilldown ----------------------------------- #
        self.per_seed = QTableWidget(0, 7)
        self.per_seed.setHorizontalHeaderLabels(
            ["seed", "AP 수익", "AP MDD", "AP trades", "BH 수익", "Naive 수익",
              "AP 우위?"])
        self.per_seed.verticalHeader().setVisible(False)
        tabs.addTab(self.per_seed, "시드별")

        v.addWidget(tabs, 1)

        # --- log -------------------------------------------------- #
        log_label = QLabel("실행 로그")
        log_label.setStyleSheet(f"color:{SUBTEXT}; font-size:12px; padding-top:6px;")
        v.addWidget(log_label)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(180)
        self.log_view.setStyleSheet(
            f"background:{SURFACE}; color:#1f2329; "
            "font-family: Consolas, monospace; font-size: 12px;")
        v.addWidget(self.log_view)

        # --- verdict --------------------------------------------- #
        self.verdict = QLabel("실행 후 평가가 여기 표시됩니다.")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet(
            f"color:{SUBTEXT}; padding:8px; background:#f3f6fa; "
            "border-radius:6px;")
        v.addWidget(self.verdict)
        return wrapper

    # ================================================================== #
    def _on_source_change(self, idx: int) -> None:
        is_synth = idx == 0
        # Synthetic-only knobs visible only in synthetic mode
        self.seeds_spin.setVisible(is_synth)
        self.n_sym_spin.setVisible(is_synth)
        if self.synth_seeds_label:
            self.synth_seeds_label.setVisible(is_synth)
        if self.synth_nsym_label:
            self.synth_nsym_label.setVisible(is_synth)

    # ================================================================== #
    def _gather_params(self) -> dict:
        return {
            "source": "synthetic" if self.source_combo.currentIndex() == 0 else "bitget",
            "seeds": int(self.seeds_spin.value()),
            "n_symbols": int(self.n_sym_spin.value()),
            "top": int(self.top_spin.value()),
            "bars": int(self.bars_spin.value()),
            "train_days": int(self.train_days_spin.value()),
            "test_days": int(self.test_days_spin.value()),
            "cvar_floor": float(self.cvar_floor.value()),
            "cvar_alpha": float(self.cvar_alpha.value()),
            "target_vol": float(self.target_vol.value()),
            "kelly_safety": float(self.kelly_safety.value()),
            "sizing_cap": float(self.sizing_cap.value()),
            "leverage_cap": float(self.leverage_cap.value()),
            "taker_fee": float(self.taker_fee.value()),
            "slippage_bps": float(self.slippage_bps.value()),
        }

    # ================================================================== #
    def _on_run(self) -> None:
        if self.worker is not None:
            return
        self.log_view.clear()
        self.table.setRowCount(0)
        self.per_seed.setRowCount(0)
        self.verdict.setText("…실행 중…")
        params = self._gather_params()
        self.worker = BacktestWorker(params, self)
        self.worker.log.connect(self._append_log)
        self.worker.result.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _on_stop(self) -> None:
        if self.worker:
            self.worker.stop()

    def _on_finished(self) -> None:
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_failed(self, msg: str) -> None:
        self.log_view.appendPlainText(f"⚠ {msg}")
        self.verdict.setText(f"실패 — {msg.splitlines()[0]}")
        self.verdict.setStyleSheet(
            f"color:{RED}; padding:8px; background:#fde6e7; border-radius:6px;")

    def _append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum())

    # ================================================================== #
    def _on_result(self, payload: dict) -> None:
        self._fill_summary(payload["summary"])
        self._fill_per_seed(payload["per_seed"])
        self._fill_verdict(payload)

    def _fill_summary(self, summary: dict) -> None:
        rows = [("AlphaPulse", summary.get("alphapulse")),
                 ("Buy-and-Hold", summary.get("buyhold")),
                 ("Naive Momentum", summary.get("naive_momentum"))]
        self.table.setRowCount(len(rows))
        for i, (name, s) in enumerate(rows):
            if s is None:
                continue
            self.table.setItem(i, 0, self._cell(name))
            self.table.setItem(i, 1, self._signed_cell(s["return"]["mean"], pct=True))
            self.table.setItem(i, 2, self._signed_cell(s["return"]["median"], pct=True))
            self.table.setItem(i, 3, self._cell(f"{s['sharpe']['mean']:+.2f}"))
            self.table.setItem(i, 4, self._cell(f"{s['sortino']['mean']:+.2f}"))
            self.table.setItem(i, 5, self._signed_cell(s["mdd"]["mean"], pct=True))
            self.table.setItem(i, 6, self._cell(f"{s['exposure']['mean']:.1%}"))
            self.table.setItem(i, 7, self._cell(f"{s['n_trades']['mean']:.1f}"))
        self.table.resizeColumnsToContents()

    def _fill_per_seed(self, rows: list[dict]) -> None:
        self.per_seed.setRowCount(len(rows))
        for i, r in enumerate(rows):
            ap = r["alphapulse"]; bh = r["buyhold"]; nm = r["naive"]
            self.per_seed.setItem(i, 0, self._cell(str(r["seed"])))
            self.per_seed.setItem(i, 1, self._signed_cell(ap["total_return"], pct=True))
            self.per_seed.setItem(i, 2, self._signed_cell(ap["max_drawdown"], pct=True))
            self.per_seed.setItem(i, 3, self._cell(f"{ap['n_trades']:.0f}"))
            self.per_seed.setItem(i, 4, self._signed_cell(bh["total_return"], pct=True))
            self.per_seed.setItem(i, 5, self._signed_cell(nm["total_return"], pct=True))
            beats = (ap["total_return"] > bh["total_return"]
                      and ap["total_return"] > nm["total_return"])
            self.per_seed.setItem(i, 6, self._cell("✅" if beats else "—"))
        self.per_seed.resizeColumnsToContents()

    def _fill_verdict(self, payload: dict) -> None:
        ap = payload["summary"].get("alphapulse")
        nm = payload["summary"].get("naive_momentum")
        bh = payload["summary"].get("buyhold")
        wins = payload["wins"]
        if not (ap and nm and bh):
            return
        ap_sortino = ap["sortino"]["mean"]
        nm_sortino = nm["sortino"]["mean"]
        ap_mdd = ap["mdd"]["mean"]; bh_mdd = bh["mdd"]["mean"]
        ap_ret = ap["return"]["mean"]; nm_ret = nm["return"]["mean"]

        good_sortino = ap_sortino >= 2 * max(nm_sortino, 0.5)
        good_mdd = ap_mdd > bh_mdd / 3.0     # less negative
        good_ret = ap_ret > 0.0

        verdict_lines = [
            f"AP > BH on return: {wins['ap_beats_bh']}/{wins['n']}    "
            f"AP > Naive on return: {wins['ap_beats_nm']}/{wins['n']}",
            f"Sortino: AP {ap_sortino:+.2f} vs Naive {nm_sortino:+.2f}    "
            f"MDD: AP {ap_mdd:+.2%} vs BH {bh_mdd:+.2%}",
            "",
        ]
        if good_sortino and good_mdd:
            verdict_lines.append(
                "✅ 위험조정 우위 + 자본 보호 모두 충족 — "
                "이 전략은 점프-주도 시장에서 강하다는 가설을 지지합니다.")
            color = GREEN; bg = "#e8f8ee"
        elif good_mdd or good_sortino:
            verdict_lines.append(
                "△ 부분 우위 — 자본 보호 측면은 양호하나 raw return 은 베이스라인에 약합니다. "
                "kelly_safety 를 0.7~0.8 로 높이거나 sizing_cap 을 1.5 로 풀어 재시도해 보세요.")
            color = "#a05a00"; bg = "#fff4e0"
        else:
            verdict_lines.append(
                "⚠ 우위 미확인 — 합성 데이터 분포·파라미터 둘 다 의심스럽습니다. "
                "Bitget 실거래 데이터에서 같은 분석을 돌려보길 권합니다.")
            color = RED; bg = "#fde6e7"
        self.verdict.setText("\n".join(verdict_lines))
        self.verdict.setStyleSheet(
            f"color:{color}; padding:8px; background:{bg}; "
            "border-radius:6px; font-size:12px;")

    # ================================================================== #
    @staticmethod
    def _cell(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    @staticmethod
    def _signed_cell(value: float, pct: bool = False) -> QTableWidgetItem:
        text = f"{value:+.2%}" if pct else f"{value:+.4f}"
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        if value > 0:
            item.setForeground(QColor(GREEN))
        elif value < 0:
            item.setForeground(QColor(RED))
        else:
            item.setForeground(QColor(GRAY))
        return item

    # ================================================================== #
    def _restore_last_run(self) -> None:
        path = STATE_DIR / "last_backtest.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
            self._on_result(payload)
            self._append_log(f"(이전 실행 복원: {path})")
        except Exception:                                              # noqa: BLE001
            pass

    def closeEvent(self, event):                                       # noqa: N802
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
        super().closeEvent(event)
