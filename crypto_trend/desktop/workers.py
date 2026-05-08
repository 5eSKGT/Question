"""Background QThread that drives the trading engine cycle.

The worker also forwards per-stage progress events from the engine
through Qt signals so the main window's status bar can show "downloading
47/120 symbols…" instead of going dark while a long cycle runs.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from ..execution.engine import TradingEngine
from ..logging_setup import get_logger
from ..portfolio.state import PortfolioState

log = get_logger()


class EngineWorker(QThread):
    """Runs ``engine.run_once()`` on an interval. GUI-thread safe — only emits
    signals; never touches widgets directly."""

    cycle_done = Signal(float)                  # equity after the cycle
    cycle_error = Signal(str)
    halted = Signal(str)
    progress = Signal(str, int, int)            # stage, current, total
    symbols_updated = Signal(list)              # cached candle symbols

    def __init__(self, portfolio: PortfolioState, period_seconds: int = 3600):
        super().__init__()
        self.portfolio = portfolio
        self.period_seconds = period_seconds
        self._stop = False
        self._engine: TradingEngine | None = None

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            self._engine = TradingEngine(self.portfolio)
            # Register a progress callback the engine can call without
            # importing Qt — keeps the engine pure Python.
            self._engine.progress_callback = self._on_engine_progress
        except Exception as e:                                          # noqa: BLE001
            self.cycle_error.emit(f"엔진 초기화 실패: {e}")
            return

        while not self._stop:
            try:
                self._engine.run_once()
                self.cycle_done.emit(self.portfolio.equity_usdt)
                self.symbols_updated.emit(
                    sorted(self._engine._candles_cache.keys()))
                if self.portfolio.halted:
                    self.halted.emit(self.portfolio.halt_reason)
            except Exception as e:                                       # noqa: BLE001
                self.cycle_error.emit(str(e))
                log.exception(f"engine cycle: {e}")
            # poll-friendly sleep so stop() reacts within 1s
            for _ in range(self.period_seconds):
                if self._stop:
                    break
                time.sleep(1)

    # ------------------------------------------------------------------ #
    def _on_engine_progress(self, stage: str, current: int, total: int) -> None:
        self.progress.emit(stage, current, total)

    def candles_for(self, symbol: str):
        if self._engine is None:
            return None
        return self._engine._candles_cache.get(symbol)

    def cached_symbols(self) -> list[str]:
        if self._engine is None:
            return []
        return sorted(self._engine._candles_cache.keys())
