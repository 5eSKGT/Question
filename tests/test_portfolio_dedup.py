"""Verify the contract that PortfolioState dedupes signals."""
import pandas as pd

from crypto_trend.portfolio.state import PortfolioState
from crypto_trend.strategy.trend_following import (Signal, SignalSource,
                                                    SignalType)


def _sig(ts):
    return Signal(ts=ts, symbol="X/USDT:USDT", side="long",
                  type=SignalType.ENTRY, source=SignalSource.HIST,
                  price=100.0, size_fraction=0.1, reason="test")


def test_duplicate_signals_suppressed():
    state = PortfolioState()
    ts = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    state.add_signal(_sig(ts))
    state.add_signal(_sig(ts))   # exact duplicate
    state.add_signal(_sig(ts))   # exact duplicate
    assert len(state.signals) == 1


def test_different_source_kept_separately():
    state = PortfolioState()
    ts = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    base = _sig(ts)
    state.add_signal(base)
    live = Signal(ts=ts, symbol=base.symbol, side=base.side,
                  type=base.type, source=SignalSource.LIVE,
                  price=base.price, size_fraction=base.size_fraction,
                  reason=base.reason)
    state.add_signal(live)
    # Same (sym, ts, side, type) but different source → kept
    assert len(state.signals) == 2
