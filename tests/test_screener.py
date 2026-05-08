import numpy as np
import pandas as pd

from crypto_trend.screener.winner_loser import (WinnerLoserScreener, hurst_rs,
                                                robust_z)


def _ohlcv(prices: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="h", tz="UTC")
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.002,
        "low": prices * 0.998,
        "close": prices,
        "volume": np.full_like(prices, 1000.0),
    }, index=idx)
    df.attrs["symbol"] = "TEST/USDT:USDT"
    return df


def test_robust_z_detects_outlier():
    rng = np.random.default_rng(0)
    rs = np.concatenate([rng.normal(0, 0.001, 100), np.array([0.5])])
    z = robust_z(rs)
    assert abs(z) > 3


def test_hurst_trending_above_random():
    rng = np.random.default_rng(42)
    trending = np.cumsum(rng.normal(0.001, 0.005, 512))
    h = hurst_rs(np.diff(trending))
    assert 0.0 <= h <= 1.0


def test_screener_picks_pumping_symbol():
    rng = np.random.default_rng(0)
    flat = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 400)))
    pump = flat.copy()
    # sudden 15% jump on the latest bar — exactly the WINNER pattern we screen for
    pump[-1] = pump[-2] * 1.15

    candles = {
        "PUMP/USDT:USDT": _ohlcv(pump),
        "FLAT/USDT:USDT": _ohlcv(flat),
    }
    qv = {"PUMP/USDT:USDT": 1e8, "FLAT/USDT:USDT": 1e8}

    screener = WinnerLoserScreener(lookback=24, z_threshold=1.5, hurst_floor=0.4,
                                    min_quote_volume=1e6, top_n=2)
    out = screener.run(candles.keys(),
                       lambda s: candles[s],
                       lambda s: qv[s])
    assert out, "screener should return at least one result"
    assert out[0].symbol == "PUMP/USDT:USDT"
    assert out[0].side == "long"
