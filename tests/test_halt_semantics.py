"""Halt-on-uncalibratable verification tests.

These tests lock the contract that, when AdaptiveOOS cannot find a
passing parameter set:

  1. Every open position is closed via a reduce-only market order.
  2. The halted flag is set.
  3. Subsequent cycles short-circuit and do not open new positions.
  4. No further orders are submitted to the broker after halt.

The tests use a stub broker that records every submitted order plus a
stub adaptor that always raises RecalibrationFailed, so the engine's
halt path is exercised end-to-end.
"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from crypto_trend.exchange.bitget_client import Fill, Order, Position
from crypto_trend.oos.adaptive import RecalibrationFailed
from crypto_trend.portfolio.state import PortfolioState


class _StubBroker:
    """Records every submit call; lets us synthesize positions arbitrarily."""

    def __init__(self):
        self.submitted: list[Order] = []
        self.fake_positions: list[Position] = []

    def fetch_universe(self) -> list[str]:
        return ["BTC/USDT:USDT"]

    def fetch_tickers(self, symbols=None) -> dict:
        return {"BTC/USDT:USDT": {"quoteVolume": 1e9, "last": 60000.0}}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h",
                    limit: int = 500) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=limit, freq="h", tz="UTC")
        # flat synthetic so the screener won't fire — we just want HALT path
        return pd.DataFrame({"open": 60000, "high": 60010, "low": 59990,
                             "close": 60000.0, "volume": 1.0}, index=idx)

    def fetch_balance_usdt(self) -> float:
        return 1000.0

    def fetch_positions(self) -> list[Position]:
        return list(self.fake_positions)

    def submit(self, order: Order) -> Fill:
        self.submitted.append(order)
        return Fill(symbol=order.symbol, side=order.side, qty=order.qty,
                    price=60000.0, fee=0.0, ts=pd.Timestamp.utcnow(),
                    order_id="stub")


def test_halt_closes_all_open_positions(monkeypatch):
    """When OOS recalibration fails, the engine must reduce-only close
    every position before flipping the halt flag."""
    # Avoid building a real broker
    monkeypatch.setattr(
        "crypto_trend.execution.engine.build_broker", _StubBroker)
    # Force the OOS adaptor to fail
    def boom(self, params):                                          # noqa: ARG001
        raise RecalibrationFailed("test halt")
    monkeypatch.setattr(
        "crypto_trend.oos.adaptive.AdaptiveOOS.step", boom)

    from crypto_trend.execution.engine import TradingEngine

    portfolio = PortfolioState()
    portfolio.equity_usdt = 1000.0
    engine = TradingEngine(portfolio)
    # Skip warmup so a single cycle can trigger halt
    engine.oos_warmup_cycles = 0
    # Pretend two positions are already open
    engine.broker.fake_positions = [
        Position(symbol="BTC/USDT:USDT", qty=0.05, avg_price=60000.0),
        Position(symbol="ETH/USDT:USDT", qty=-1.5, avg_price=3500.0),
    ]
    engine._open_positions = {
        "BTC/USDT:USDT": "long",
        "ETH/USDT:USDT": "short",
    }
    portfolio.positions = {
        "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "qty": 0.05,
                           "avg_price": 60000.0, "realized": 0.0},
        "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "qty": -1.5,
                           "avg_price": 3500.0,  "realized": 0.0},
    }

    engine.run_once()

    # Every position closed with reduce_only
    closes = [o for o in engine.broker.submitted if o.reduce_only]
    closed_syms = {o.symbol for o in closes}
    assert closed_syms == {"BTC/USDT:USDT", "ETH/USDT:USDT"}, \
        f"halt failed to close positions, got: {[o.symbol for o in engine.broker.submitted]}"

    # Halt flag set
    assert portfolio.halted is True
    assert "test halt" in portfolio.halt_reason

    # Open-positions tracker emptied
    assert engine._open_positions == {}


def test_halt_blocks_subsequent_cycles(monkeypatch):
    """After halt, every run_once() short-circuits without touching the broker."""
    monkeypatch.setattr(
        "crypto_trend.execution.engine.build_broker", _StubBroker)

    from crypto_trend.execution.engine import TradingEngine

    portfolio = PortfolioState()
    portfolio.equity_usdt = 1000.0
    portfolio.halt("test halt")
    engine = TradingEngine(portfolio)
    before = len(engine.broker.submitted)

    for _ in range(5):
        engine.run_once()

    # Engine never even reached the broker
    assert len(engine.broker.submitted) == before
    assert portfolio.halted is True
