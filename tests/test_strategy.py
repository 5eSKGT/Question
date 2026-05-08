import numpy as np
import pandas as pd

from crypto_trend.strategy.trend_following import (SignalSource, SignalType,
                                                    StrategyParams,
                                                    TrendFollowingStrategy)


def _df(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="h", tz="UTC")
    df = pd.DataFrame({"open": prices, "high": prices * 1.002,
                      "low": prices * 0.998, "close": prices,
                      "volume": np.ones_like(prices) * 1000.0}, index=idx)
    df.attrs["symbol"] = "T/USDT:USDT"
    return df


def test_strategy_emits_entry_on_breakout():
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 0.05, 200))
    breakout = base.copy()
    breakout[-1] = base[-30:].max() * 1.005        # gentle break, stays inside band

    # band_mult bumped so the YZ band easily covers the tiny break gap
    strat = TrendFollowingStrategy(StrategyParams(
        breakout_n=20, atr_n=14, yz_n=20, band_mult=20.0))
    sigs = strat.generate_signals(_df(breakout), screener_side="long",
                                   source=SignalSource.HIST)
    assert any(s.type == SignalType.ENTRY and s.side == "long" for s in sigs)


def test_strategy_no_signals_on_short_history():
    strat = TrendFollowingStrategy()
    sigs = strat.generate_signals(_df(np.linspace(100, 101, 5)),
                                   screener_side="long",
                                   source=SignalSource.HIST)
    assert sigs == []
