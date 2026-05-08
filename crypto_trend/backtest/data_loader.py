"""OHLCV data loader for backtesting.

Real-data path (Bitget via ccxt) caches every symbol/timeframe slice as a
parquet file in ``data/cache/<symbol>_<timeframe>.parquet`` so subsequent
runs do not hit the API. Synthetic path generates Heston-style stochastic
volatility paths with rare jumps for offline tests when Bitget is
unreachable.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR
from ..logging_setup import get_logger

log = get_logger()


# --------------------------------------------------------------------------- #
# Real Bitget data via ccxt
# --------------------------------------------------------------------------- #


def _safe_filename(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '_').replace(':', '_')}_{timeframe}.parquet"


def fetch_bitget_ohlcv(symbol: str, timeframe: str = "1h",
                        bars: int = 2000,
                        cache_dir: Path | None = None,
                        force_refresh: bool = False) -> pd.DataFrame:
    """Pull historical OHLCV from Bitget through ccxt with disk caching.

    ``bars`` is the *minimum* number of bars to ensure are present. Bitget
    returns at most 200-1000 bars per call, so we paginate backwards.
    """
    import ccxt
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _safe_filename(symbol, timeframe)

    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        if len(cached) >= bars:
            return cached.tail(bars)

    ex = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    ex.load_markets()
    rows: list[list] = []
    seen: set[int] = set()
    end_ms: int | None = None

    while len(rows) < bars:
        try:
            params = {}
            if end_ms is not None:
                params["until"] = end_ms
            chunk = ex.fetch_ohlcv(symbol, timeframe=timeframe,
                                    limit=1000, params=params)
        except Exception as e:                                   # noqa: BLE001
            log.warning(f"fetch_ohlcv {symbol}: {e}")
            break
        if not chunk:
            break
        new = [r for r in chunk if r[0] not in seen]
        if not new:
            break
        rows = new + rows
        seen.update(r[0] for r in new)
        end_ms = new[0][0] - 1
        time.sleep(ex.rateLimit / 1000.0)

    if not rows:
        raise RuntimeError(f"no data for {symbol}")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    df.attrs["symbol"] = symbol
    df.to_parquet(cache_path)
    return df.tail(bars)


def fetch_bitget_universe(min_quote_volume: float = 5e6,
                           top_k: int | None = None) -> list[str]:
    """Return USDT-perp symbols passing the live-engine quote-volume floor.

    Defaults match the live engine exactly (min_quote_volume = $5M = the
    same screener pre-filter). ``top_k=None`` returns the entire qualifying
    universe — that's the live-mirroring behaviour. Pass an integer
    ``top_k`` only when you explicitly want a smaller universe for a quick
    preview.
    """
    import ccxt
    ex = ccxt.bitget({"enableRateLimit": True,
                       "options": {"defaultType": "swap"}})
    markets = ex.load_markets()
    candidates = [s for s, m in markets.items()
                   if m.get("swap") and m.get("quote") == "USDT"
                   and m.get("active", True)]
    tickers = ex.fetch_tickers(candidates)
    rows: list[tuple[str, float]] = []
    for sym, t in tickers.items():
        qv = float(t.get("quoteVolume") or 0)
        if qv >= min_quote_volume:
            rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    if top_k:
        rows = rows[:top_k]
    return [r[0] for r in rows]


# --------------------------------------------------------------------------- #
# Synthetic data — Heston SV + compound-Poisson jumps
# --------------------------------------------------------------------------- #


def heston_jump_path(
    n_bars: int,
    rng: np.random.Generator,
    mu: float = 0.0,
    kappa: float = 5.0,                    # vol mean-reversion
    theta: float = 0.0009,                 # long-run variance (σ_lr ≈ 3%)
    xi: float = 0.6,                       # vol of vol
    rho: float = -0.4,                     # leverage effect
    jump_intensity: float = 0.005,         # P(jump per bar)
    jump_mean: float = 0.0,                # mean log jump size
    jump_std: float = 0.06,                # std of log jump size
    drift_per_bar: float = 0.0,            # extra deterministic drift
    bar_dt: float = 1.0 / (24.0 * 365.0),  # 1h bar
) -> pd.DataFrame:
    """Stochastic-volatility path with compound-Poisson jumps.

    Uses an Euler discretisation of the Heston model:
        dS/S = μ dt + √v dW₁ + (e^J - 1) dN
        dv   = κ(θ - v) dt + ξ √v dW₂,    Corr(W₁, W₂) = ρ

    The jump component matches the model the LM screener was designed for.
    """
    dt = bar_dt
    v = np.empty(n_bars)
    log_p = np.empty(n_bars)
    v[0] = theta
    log_p[0] = np.log(100.0)

    # Pre-generate correlated normals
    z = rng.standard_normal((n_bars - 1, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(max(1 - rho * rho, 0.0)) * z[:, 1]

    for t in range(1, n_bars):
        v_prev = max(v[t - 1], 1e-12)
        # variance step (Full-truncation scheme avoids negative v)
        v[t] = max(v_prev + kappa * (theta - v_prev) * dt
                    + xi * np.sqrt(v_prev * dt) * z[t - 1, 1], 1e-12)
        ret = (mu * dt + drift_per_bar
                + np.sqrt(v_prev * dt) * z[t - 1, 0])
        # Jump component
        if rng.random() < jump_intensity:
            ret += rng.normal(jump_mean, jump_std)
        log_p[t] = log_p[t - 1] + ret

    closes = np.exp(log_p)
    # Approximate OHL from a Brownian-bridge wiggle around closes
    half = np.maximum(np.abs(np.diff(log_p, prepend=log_p[0])), 1e-4)
    highs = closes * np.exp(0.6 * half)
    lows = closes * np.exp(-0.6 * half)
    opens = np.empty_like(closes)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    volumes = np.exp(rng.normal(15, 0.5, n_bars))     # arbitrary positive vol

    idx = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=idx)


def synthetic_universe(
    n_symbols: int = 40,
    n_bars: int = 1500,
    seed: int = 0,
    factor_loadings_std: float = 0.7,
    regime: str = "neutral",
) -> dict[str, pd.DataFrame]:
    """A correlated cross-section of Heston-jump paths.

    Each symbol is a factor + idiosyncratic mix:
        log_S_i,t = β_i · F_t + ε_i,t
    where F_t and ε_i,t are independent Heston-jump processes. The factor
    introduces realistic cross-symbol correlation.

    ``regime`` controls the macro drift and jump asymmetry of the
    market factor — the single biggest determinant of how a
    trend-following strategy performs:

        * ``"neutral"``   — symmetric jumps, zero drift (default).
        * ``"bull_jump"`` — positive drift + jump distribution skewed up
                            (mean=+1.5%). Models a real crypto bull
                            market: smooth uptrend punctuated by
                            +5-15% upward jumps. AlphaPulse should
                            handle this — its target.
        * ``"bull_diffusion"`` — positive drift, jump intensity HALVED.
                            Models the rare "smooth-drift bull market"
                            (more typical of equity indices). Tests
                            whether AlphaPulse can capture beta with
                            no jumps to grab.
        * ``"bear_jump"`` — negative drift + downward-skewed jumps.
        * ``"high_vol"``  — jump intensity 3× normal, mixed direction.
    """
    rng = np.random.default_rng(seed)

    # ---- Regime → factor process parameters ------------------------ #
    # All drifts calibrated to realistic crypto annualised levels:
    # 0.0001 /h ≈ 0.24%/d ≈ 137%/y. These rough magnitudes match
    # observed BTC bull (≈+60-200%/y) and bear (≈-40-65%/y) cycles.
    if regime == "bull_jump":
        f_drift = 0.00015                   # ≈ +260%/y, real crypto bull
        f_jump_intensity = 0.005            # one jump per ~200 bars
        f_jump_mean = 0.012                 # +1.2% upward-skewed jumps
    elif regime == "bull_diffusion":
        f_drift = 0.00010                   # ≈ +140%/y, smooth drift only
        f_jump_intensity = 0.0008           # rare jumps
        f_jump_mean = 0.0
    elif regime == "bear_jump":
        f_drift = -0.00012                  # ≈ -65%/y
        f_jump_intensity = 0.005
        f_jump_mean = -0.012
    elif regime == "high_vol":
        f_drift = 0.0
        f_jump_intensity = 0.012            # 3× normal, mixed direction
        f_jump_mean = 0.0
    else:                                    # neutral
        f_drift = 0.0
        f_jump_intensity = 0.003
        f_jump_mean = 0.0

    factor = heston_jump_path(n_bars, rng,
                                drift_per_bar=f_drift,
                                jump_intensity=f_jump_intensity,
                                jump_mean=f_jump_mean,
                                jump_std=0.04)
    factor_log = np.log(factor["close"].to_numpy())
    factor_log = factor_log - factor_log[0]

    out: dict[str, pd.DataFrame] = {}
    for i in range(n_symbols):
        beta = rng.normal(0.6, factor_loadings_std)
        idio_rng = np.random.default_rng(seed * 9999 + i)
        idio = heston_jump_path(n_bars, idio_rng,
                                  jump_intensity=0.004,
                                  jump_mean=0.0,
                                  jump_std=0.05)
        idio_log = np.log(idio["close"].to_numpy())
        idio_log = idio_log - idio_log[0]
        combined_log = beta * factor_log + idio_log + np.log(100.0)
        closes = np.exp(combined_log)
        rets = np.diff(combined_log, prepend=combined_log[0])
        half = np.maximum(np.abs(rets), 1e-4)
        highs = closes * np.exp(0.6 * half)
        lows = closes * np.exp(-0.6 * half)
        opens = np.empty_like(closes); opens[0] = closes[0]; opens[1:] = closes[:-1]
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": idio["volume"].to_numpy(),
        }, index=factor.index)
        df.attrs["symbol"] = f"SYN{i:02d}/USDT:USDT"
        out[df.attrs["symbol"]] = df
    return out
