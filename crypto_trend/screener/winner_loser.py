"""Cross-sectional WINNER / LOSER screener.

Identifies symbols undergoing a sudden directional move that is statistically
extreme *relative to its own history* AND *relative to the universe right now*.

Scoring blends three orthogonal signals — each one chosen for academic
robustness rather than tuning fashion:

1. Robust momentum z-score
   - Numerator   : recent log return over `lookback` bars.
   - Denominator : MAD-scaled volatility (1.4826 * MAD), which is robust to
     fat-tailed returns (Huber 1981; Rousseeuw & Croux 1993).

2. Volume / liquidity dispersion
   - log10(quote-volume) cross-sectional z. Filters out illiquid pumps that
     cannot be traded without slippage cost dominating CVaR.

3. Trend persistence
   - Hurst exponent (Mandelbrot, R/S analysis). Persistent moves (H > 0.55)
     are exploitable by trend followers; mean-reverting (H < 0.45) are not.

The composite score is a convex combination of (1)+(3), gated by (2). The
sign of (1) decides WINNER (positive) vs LOSER (negative).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# Statistics helpers — kept dependency-free for testability
# --------------------------------------------------------------------------- #


def robust_z(returns: np.ndarray) -> float:
    """Median / MAD z-score for the most recent return point."""
    if returns.size < 8:
        return 0.0
    med = np.median(returns)
    mad = np.median(np.abs(returns - med))
    if mad == 0:
        return 0.0
    return float((returns[-1] - med) / (1.4826 * mad))


def hurst_rs(series: np.ndarray, min_chunk: int = 8) -> float:
    """Rescaled-range Hurst estimate. 0.5 = random, >0.5 trending."""
    n = series.size
    if n < min_chunk * 4:
        return 0.5
    chunk_sizes = [c for c in (min_chunk, min_chunk * 2, min_chunk * 4, min_chunk * 8) if c <= n // 2]
    if not chunk_sizes:
        return 0.5
    rs_vals: list[float] = []
    for c in chunk_sizes:
        rs_chunk: list[float] = []
        for i in range(0, n - c + 1, c):
            seg = series[i:i + c]
            mean = seg.mean()
            dev = np.cumsum(seg - mean)
            r = dev.max() - dev.min()
            s = seg.std(ddof=0)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
    if len(rs_vals) < 2:
        return 0.5
    log_c = np.log(chunk_sizes[: len(rs_vals)])
    log_rs = np.log(rs_vals)
    slope, _ = np.polyfit(log_c, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def realized_vol(returns: np.ndarray) -> float:
    if returns.size < 4:
        return 0.0
    return float(returns.std(ddof=1))


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass
class ScreenResult:
    symbol: str
    side: str            # "long" (winner) | "short" (loser)
    z_momentum: float
    hurst: float
    vol: float
    quote_volume: float
    composite: float
    last_price: float

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "z_momentum": round(self.z_momentum, 3),
            "hurst": round(self.hurst, 3),
            "vol": round(self.vol, 4),
            "quote_volume": round(self.quote_volume, 0),
            "composite": round(self.composite, 3),
            "last_price": self.last_price,
        }


# --------------------------------------------------------------------------- #
# Screener
# --------------------------------------------------------------------------- #


class WinnerLoserScreener:
    """Cross-sectional screener for sudden movers.

    Parameters all live on the instance so the OOS adaptor can tune them.
    """

    def __init__(
        self,
        lookback: int = 24,             # bars of return window (e.g. 24h on 1h)
        z_threshold: float = 2.5,       # robust-z floor
        hurst_floor: float = 0.55,      # require persistence
        min_quote_volume: float = 5e6,  # 24h notional in USDT
        top_n: int = 10,
        weight_momentum: float = 0.7,
        weight_persistence: float = 0.3,
    ) -> None:
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.hurst_floor = hurst_floor
        self.min_quote_volume = min_quote_volume
        self.top_n = top_n
        self.w_mom = weight_momentum
        self.w_pers = weight_persistence

    # ------------------------------------------------------------------ #
    def score(self, candles: pd.DataFrame, quote_volume: float) -> tuple[float, float, float, str] | None:
        """Return (composite, z, hurst, side) or None if filtered out."""
        if candles is None or candles.empty or len(candles) < self.lookback + 4:
            return None
        closes = candles["close"].to_numpy(dtype=float)
        rets = np.diff(np.log(closes))
        z = robust_z(rets[-self.lookback:])
        if math.isnan(z) or abs(z) < self.z_threshold:
            return None
        h = hurst_rs(rets[-min(rets.size, 256):])
        if h < self.hurst_floor:
            return None
        if quote_volume < self.min_quote_volume:
            return None
        # composite — sign carries direction
        composite = (self.w_mom * z + self.w_pers * (h - 0.5) * 4.0 * np.sign(z))
        side = "long" if z > 0 else "short"
        return float(composite), float(z), float(h), side

    # ------------------------------------------------------------------ #
    def run(
        self,
        symbols: Iterable[str],
        ohlcv_provider,                  # callable: symbol -> DataFrame
        quote_volume_provider,           # callable: symbol -> float
    ) -> list[ScreenResult]:
        results: list[ScreenResult] = []
        for sym in symbols:
            try:
                candles = ohlcv_provider(sym)
                qv = quote_volume_provider(sym)
            except Exception as e:                      # noqa: BLE001
                log.debug(f"screener: {sym} skipped: {e}")
                continue
            scored = self.score(candles, qv)
            if scored is None:
                continue
            composite, z, h, side = scored
            results.append(ScreenResult(
                symbol=sym,
                side=side,
                z_momentum=z,
                hurst=h,
                vol=realized_vol(np.diff(np.log(candles["close"].to_numpy(dtype=float)))),
                quote_volume=qv,
                composite=composite,
                last_price=float(candles["close"].iloc[-1]),
            ))
        results.sort(key=lambda r: abs(r.composite), reverse=True)
        return results[: self.top_n]
