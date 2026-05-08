"""Entry point — runs the engine in a worker thread and the UI in the main thread."""
from __future__ import annotations

import threading
import time

from .config import SETTINGS
from .execution.engine import TradingEngine
from .logging_setup import get_logger
from .portfolio.state import PortfolioState
from .ui.app import run_ui

log = get_logger()


def _engine_loop(engine: TradingEngine, period_seconds: int = 3600) -> None:
    while True:
        try:
            engine.run_once()
        except Exception as e:                       # noqa: BLE001
            log.exception(f"engine cycle error: {e}")
        time.sleep(period_seconds)


def main() -> None:
    portfolio = PortfolioState()
    portfolio.equity_usdt = SETTINGS.base_equity_usdt
    engine = TradingEngine(portfolio)

    t = threading.Thread(target=_engine_loop, args=(engine,), daemon=True)
    t.start()

    run_ui(portfolio, lambda sym: engine._candles_cache.get(sym))


if __name__ == "__main__":
    main()
