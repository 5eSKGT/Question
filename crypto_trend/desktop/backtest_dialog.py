"""Backtest configuration + execution inside the GUI.

Lives behind the "📊 백테스트 실행" button on the engine-control panel.
Runs the same simulator + walk-forward + baselines that
``python -m crypto_trend.backtest`` invokes from the CLI, but here it
is fully wrapped in a dialog so the user never has to touch the shell.

Threading
---------
The simulator is CPU-bound so we run it in a QThread (BacktestWorker)
and stream progress messages + the final result table back to the UI
through Qt signals.  The dialog stays responsive throughout.

Data sources
------------
* **Synthetic**  — Heston-jump universe (no network); deterministic
                   per seed.  Useful for CI / preview / repeated tuning.
* **Bitget**     — pulls real OHLCV via ccxt, parquet-cached locally.
                   Requires only an internet connection (no API key —
                   public market data only).

Outputs
-------
A side-by-side table of {AlphaPulse, Buy-and-Hold, Naive Momentum}
on every metric (return / Sharpe / Sortino / MDD / Calmar / exposure /
n_trades) plus a per-seed drilldown for synthetic mode. Results are
also saved to ``state/last_backtest.json`` so subsequent dialog opens
restore the previous run.
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
                                QHBoxLayout, QLabel, QPlainTextEdit,
                                QPushButton, QSpinBox, QTableWidget,
                                QTableWidgetItem, QTabWidget, QVBoxLayout,
                                QWidget)

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
                    self.log.emit(f"  pulling top-{self.p['top']} Bitget USDT-perps …")
                    universe = fetch_bitget_universe(top_k=self.p["top"])
                    candles = {}
                    for s in universe:
                        try:
                            candles[s] = fetch_bitget_ohlcv(s, "1h", self.p["bars"])
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
        self.resize(1100, 760)
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
        wrapper = QWidget()
        wrapper.setMinimumWidth(360)
        wrapper.setMaximumWidth(420)
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        head = QLabel("⚙  설정"); head.setStyleSheet("font-size:14px; font-weight:600;")
        v.addWidget(head)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["synthetic (오프라인)", "bitget (실거래)"])
        self.source_combo.currentIndexChanged.connect(self._on_source_change)
        form.addRow("데이터 소스", self.source_combo)

        self.seeds_spin = QSpinBox()
        self.seeds_spin.setRange(1, 200); self.seeds_spin.setValue(10)
        form.addRow("시드 수 (synthetic)", self.seeds_spin)

        self.n_sym_spin = QSpinBox()
        self.n_sym_spin.setRange(5, 200); self.n_sym_spin.setValue(30)
        form.addRow("심볼 수 (synthetic)", self.n_sym_spin)

        self.top_spin = QSpinBox()
        self.top_spin.setRange(3, 100); self.top_spin.setValue(20)
        form.addRow("Top-N 종목 (bitget)", self.top_spin)

        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(500, 20000); self.bars_spin.setValue(2000)
        self.bars_spin.setSingleStep(500)
        form.addRow("봉 수 (1h)", self.bars_spin)

        self.train_days_spin = QSpinBox()
        self.train_days_spin.setRange(7, 90); self.train_days_spin.setValue(30)
        form.addRow("Train 기간 (일)", self.train_days_spin)

        self.test_days_spin = QSpinBox()
        self.test_days_spin.setRange(1, 30); self.test_days_spin.setValue(7)
        form.addRow("Test 기간 (일)", self.test_days_spin)

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

        v.addStretch(1)
        self._on_source_change(self.source_combo.currentIndex())
        return wrapper

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
        self.seeds_spin.setEnabled(is_synth)
        self.n_sym_spin.setEnabled(is_synth)
        self.top_spin.setEnabled(not is_synth)

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
