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
from .theme import (ACCENT, ACCENT_DEEP, BORDER, GRAY, GREEN, RED,
                    SUBTEXT, SURFACE, icon_path)


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
                    if i > 0:
                        self.log.emit("  bitget mode runs once — extra seeds skipped")
                        break
                    top = self.p["top"] or None     # 0 → None → full universe
                    self.log.emit(f"  pulling Bitget USDT-perps "
                                    f"({'top-' + str(top) if top else 'full universe'}) …")
                    universe = fetch_bitget_universe(top_k=top)
                    self.log.emit(f"  {len(universe)} symbols qualify — "
                                    f"parallel download (6 workers)")

                    # Parallel OHLCV download — ~5× faster than sequential.
                    from ..backtest.data_loader import fetch_bitget_ohlcv_parallel
                    last_pct = [-1]
                    def _on_progress(done, total, sym):
                        if self._stop:
                            return
                        pct = int(done * 100 / max(total, 1))
                        if pct >= last_pct[0] + 10 or done == total:
                            self.log.emit(f"  downloaded {done}/{total}")
                            last_pct[0] = pct
                    candles = fetch_bitget_ohlcv_parallel(
                        universe, "1h", self.p["bars"],
                        max_workers=6, progress_cb=_on_progress,
                        should_stop=lambda: self._stop)
                    if self._stop:
                        self.log.emit("⚠ aborted during download")
                        return

                    # Drop short-history symbols (new listings)
                    min_history = max(int(self.p["bars"] * 0.7), train_bars + test_bars)
                    short = [s for s, df in candles.items()
                              if len(df) < min_history]
                    for s in short:
                        candles.pop(s)
                    if short:
                        self.log.emit(
                            f"  dropped {len(short)} symbols with < {min_history} "
                            f"bars of history (newer listings)")
                    self.log.emit(f"  {len(candles)} symbols enter walk-forward")

                if not candles:
                    self.log.emit("  ⚠ no candles loaded — skipping seed")
                    continue

                # ---- strategy = canonical defaults (live mirror) -------- #
                # NO user overrides: the same StrategyParams() the live
                # engine constructs at startup is what the simulator runs.
                sim = StrategySimulator(
                    strategy=TrendFollowingStrategy(StrategyParams()),
                    taker_fee=self.p["taker_fee"],
                    slippage_bps=self.p["slippage_bps"],
                )
                self.log.emit(f"  running AlphaPulse walk-forward "
                                f"(universe={len(candles)} symbols) …")
                # Pass the worker's cancellation flag down through the
                # simulator + walk_forward so a "Stop" press is honoured
                # mid-run, not just between seeds.
                wf = walk_forward_run(candles, train_bars=train_bars,
                                       test_bars=test_bars, simulator=sim,
                                       should_stop=lambda: self._stop)
                if self._stop:
                    self.log.emit("⚠ aborted mid walk-forward")
                    return

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
                tele = getattr(wf, "telemetry", {}) or {}
                if tele.get("rescreen_count"):
                    self.log.emit(
                        f"  ▸ universe={tele.get('universe_size', '?')} symbols, "
                        f"rescreen cycles={tele['rescreen_count']}, "
                        f"avg picks/cycle={tele['picks_mean']:.2f}, "
                        f"empty cycles={tele['picks_zero_cycles']}")
                # OOS adaptive — exact mirror of live engine's behaviour
                oos_hist = getattr(wf, "oos_history", []) or []
                if oos_hist:
                    recals = sum(1 for h in oos_hist
                                   if h.get("status") == "recalibrated")
                    halted = wf.halted_at_window
                    self.log.emit(
                        f"  ▸ OOS adaptive: {len(oos_hist)} windows evaluated, "
                        f"{recals} recalibration(s)"
                        + (f", halted at window {halted}" if halted is not None
                            else ""))

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
        # Inner content widget — minimal: data source choice + read-only
        # strategy display. Strategy parameters and cost model are NOT
        # exposed because the backtest must run with the *exact* same
        # settings the live engine will use; allowing user knobs would
        # break the live mirror invariant.
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 6, 0)
        v.setSpacing(8)

        head = QLabel("📊  백테스트")
        head.setStyleSheet("font-size:14px; font-weight:600;")
        v.addWidget(head)

        from ..config import SETTINGS
        from ..strategy.trend_following import StrategyParams
        sp = StrategyParams()                              # canonical defaults
        wf_min_bars = (SETTINGS.walk_forward_train_days * 24
                        + 8 * SETTINGS.walk_forward_test_days * 24)
        bars_default = max(SETTINGS.backtest_bars, wf_min_bars)

        env_card = QLabel(
            f"<b>실거래 환경 100% 미러링</b><br>"
            f"<span style='color:{SUBTEXT}'>"
            f"이 백테스트의 모든 환경 파라미터는 라이브 엔진과 동일합니다. "
            f"전략 파라미터, 사이징 알고리즘, 스크리너 임계치, OOS 자가보정, "
            f"비용 모델 모두 사용자가 변경할 수 없습니다 — 그래야 결과가 "
            f"실제 운용을 의미 있게 예측합니다.</span><br><br>"
            f"<b>유니버스</b> · 전체 USDT-Perp / 24h 거래대금 ≥ $5M<br>"
            f"<b>타임프레임 / 봉 수</b> · 1h × {bars_default} "
            f"(<span style='color:{SUBTEXT}'>≈ 1년, Lo 2002 + López de Prado "
            f"AFML 통계적 최소</span>)<br>"
            f"<b>Walk-forward</b> · train {SETTINGS.walk_forward_train_days}d "
            f"/ test {SETTINGS.walk_forward_test_days}d (≈ 50 windows)<br>"
            f"<b>비용 모델</b> · taker {SETTINGS.taker_fee*1e4:.1f} bps "
            f"+ slippage {SETTINGS.slippage_bps:.1f} bps<br>"
            f"<b>OOS 자가보정</b> · 윈도우마다 AdaptiveOOS.step() · 실패 시 "
            f"전 포지션 시장가 청산 후 HALT<br>"
            f"<b>스크리너 빈도</b> · 매 1봉 (live 사이클과 동일)")
        env_card.setWordWrap(True)
        env_card.setTextFormat(Qt.RichText)
        env_card.setStyleSheet(
            f"background:#f3f6fa; border:1px solid {ACCENT}; "
            "border-radius:8px; padding:10px; font-size:11.5px;")
        v.addWidget(env_card)

        strat_card = QLabel(
            f"<b>적용될 전략 (read-only)</b><br>"
            f"<span style='color:{SUBTEXT}'>이 값들이 라이브 엔진의 매매에 "
            f"그대로 사용됩니다.</span><br><br>"
            f"<b>지표 (OOS 자동보정)</b><br>"
            f"&nbsp;&nbsp;Donchian breakout = {sp.breakout_n}<br>"
            f"&nbsp;&nbsp;ATR period = {sp.atr_n}, mult = {sp.chandelier_mult:.1f}<br>"
            f"&nbsp;&nbsp;Yang-Zhang n = {sp.yz_n}, band = {sp.band_mult:.1f}<br>"
            f"&nbsp;&nbsp;Time stop = {sp.time_stop_bars} bars<br>"
            f"<b>점프 검출 (Hawkes 가설)</b><br>"
            f"&nbsp;&nbsp;LM 임계 = {sp.lm_threshold:.1f}, "
            f"screener 픽이면 inside_band 스킵<br>"
            f"<b>v2 진입 필터 (counter-trend / noise 차단)</b><br>"
            f"&nbsp;&nbsp;TSM lookback = "
            f"{sp.tsm_lookback_bars // 24}d "
            f"<span style='color:{SUBTEXT}'>(Moskowitz-Ooi-Pedersen 2012)</span><br>"
            f"&nbsp;&nbsp;Volume z ≥ {sp.volume_z_threshold:.1f} "
            f"<span style='color:{SUBTEXT}'>(Easley-LdP-O'Hara 2012)</span><br>"
            f"<b>리스크 / 사이징 (Conviction-Power Kelly)</b><br>"
            f"&nbsp;&nbsp;CVaR α = {sp.cvar_alpha:.2f}, floor = {sp.cvar_floor:.2f}<br>"
            f"&nbsp;&nbsp;risk_base = {sp.risk_per_trade:.2%}, "
            f"<b>amp = conf<sup>{sp.confidence_exponent:.0f}</sup></b> "
            f"(quadratic)<br>"
            f"&nbsp;&nbsp;<span style='color:{SUBTEXT}'>약신호 0.02% / 중간 0.5% / 강신호 2-5% 손실</span><br>"
            f"&nbsp;&nbsp;Sizing cap = {sp.sizing_cap:.1f}× equity<br>"
            f"&nbsp;&nbsp;Max leverage = {sp.leverage_cap:.0f}× "
            f"<span style='color:{SUBTEXT}'>(타이트 stop + 강신호에서 5-10× 도달)</span>")
        strat_card.setWordWrap(True)
        strat_card.setTextFormat(Qt.RichText)
        strat_card.setStyleSheet(
            f"background:#fafbfc; border:1px solid {BORDER}; "
            "border-radius:8px; padding:10px; font-size:11.5px;")
        v.addWidget(strat_card)

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
        #
        # NOTE on bars: the live engine's `history_bars=500` is the
        # *per-cycle indicator window*, NOT the total length of a
        # backtest. A walk-forward needs at least
        #   train_bars + N × test_bars ≈ 720 + N × 168
        # bars of history. We default to 2000 so 8 windows fit
        # (≈ 60 days of out-of-sample testing).
        wf_min_bars = (SETTINGS.walk_forward_train_days * 24
                        + 8 * SETTINGS.walk_forward_test_days * 24)
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(100, 20000)
        self.bars_spin.setValue(max(2000, wf_min_bars))
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

    # Strategy parameters are no longer user-tunable — the strategy
    # IS the canonical StrategyParams() committed in the source. This
    # was deliberately removed to enforce live/backtest parity.

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
        # All strategy + cost parameters come from SETTINGS / canonical
        # StrategyParams — NOT from any GUI control. The dialog only
        # selects the data source and (for synthetic) reproducibility seeds.
        from ..config import SETTINGS
        return {
            "source": "synthetic" if self.source_combo.currentIndex() == 0 else "bitget",
            "seeds": int(self.seeds_spin.value()),
            "n_symbols": int(self.n_sym_spin.value()),
            "top": int(self.top_spin.value()),
            "bars": int(self.bars_spin.value()),
            "train_days": int(self.train_days_spin.value()),
            "test_days": int(self.test_days_spin.value()),
            "taker_fee": float(SETTINGS.taker_fee),
            "slippage_bps": float(SETTINGS.slippage_bps),
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
        """Request cooperative cancellation. We do NOT clear
        ``self.worker`` here — that is done by ``_on_finished`` once the
        worker actually exits. Disowning the reference early left the
        QThread orphaned and meant ``finished`` slots never ran, which
        is why the Stop button used to "do nothing"."""
        if self.worker is None:
            return
        self.worker.stop()
        # Visual feedback while the worker drains its current op.
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("중단 중…")
        self.status_label_text("백테스트 중단 요청됨 — 정리 중…")

    def status_label_text(self, msg: str) -> None:
        # Helper so verdict pane shows we're working on the cancel.
        self.verdict.setText(msg)
        self.verdict.setStyleSheet(
            f"color:#a05a00; padding:8px; background:#fff4e0; "
            "border-radius:6px; font-size:12px;")

    def _on_finished(self) -> None:
        # Worker thread exited (either naturally or via stop()).
        was_stopped = bool(self.worker and self.worker._stop)
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Reset the stop button label after a successful cancel.
        self.btn_stop.setText("■ 중단")
        if was_stopped:
            self.log_view.appendPlainText("✓ 중단 완료")
            self.verdict.setText("백테스트가 사용자 요청으로 중단되었습니다.")
            self.verdict.setStyleSheet(
                f"color:{SUBTEXT}; padding:8px; background:#f3f6fa; "
                "border-radius:6px;")

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
        # AlphaPulse is the only strategy actually executed by the live
        # engine. The other two are *baselines* — non-tradeable reference
        # points to validate the strategy's edge.
        rows = [
            ("AlphaPulse  ▸ 실제 매매 전략", summary.get("alphapulse")),
            ("Buy-and-Hold  (baseline)",     summary.get("buyhold")),
            ("Naive Momentum  (baseline)",   summary.get("naive_momentum")),
        ]
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
