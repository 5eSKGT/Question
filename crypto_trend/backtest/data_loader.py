"""OHLCV data loader for backtesting.

Real-data path (Bitget via ccxt) caches every symbol/timeframe slice as
a parquet file in ``data/cache/<symbol>_<timeframe>.parquet`` so
subsequent runs do not hit the API. The bulk fetch routine
``fetch_bitget_ohlcv_parallel`` parallelises downloads across a
ThreadPoolExecutor — Bitget's rate limit lets ~10 req/s through, and
running 6 worker threads each averaging 0.6 s per call keeps us under
the limit while reducing total wall-clock by ~5×.

Synthetic path generates Heston-style stochastic volatility paths with
rare jumps for offline tests.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def fetch_bitget_ohlcv_parallel(symbols: list[str], timeframe: str = "1h",
                                  bars: int = 2000,
                                  max_workers: int = 6,
                                  cache_dir: Path | None = None,
                                  progress_cb=None,
                                  should_stop=None) -> dict[str, pd.DataFrame]:
    """Pull OHLCV for many symbols concurrently. Returns {symbol: DataFrame}.

    ``should_stop`` is a cooperative cancellation predicate; if it
    returns True we cancel any pending futures and return whatever we
    have so far. Without this the GUI Stop button waited for *every*
    download to finish even after being pressed.
    """
    candles: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_bitget_ohlcv, s, timeframe,
                                 bars, cache_dir): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), start=1):
            if should_stop is not None and should_stop():
                # Cancel anything still pending — already-running
                # futures will continue to completion (Python's
                # Future API cannot interrupt blocking IO mid-call)
                # but no new ones will be started.
                for f in futures:
                    f.cancel()
                break
            sym = futures[fut]
            try:
                candles[sym] = fut.result()
            except Exception as e:                                   # noqa: BLE001
                log.debug(f"skip {sym}: {e}")
            if progress_cb:
                progress_cb(i, len(symbols), sym)
    return candles


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
# Real-data universe loader (parquet cache populated by
# tools/fetch_binance_real.py)
# --------------------------------------------------------------------------- #


def real_universe(cache_dir: Path | None = None,
                   min_bars: int = 4000,
                   min_quote_volume: float = 5e6,
                   max_symbols: int | None = None,
                   tail_bars: int | None = None) -> dict[str, pd.DataFrame]:
    """Load the parquet cache built from real Binance USD-M futures klines.

    The cache schema matches synthetic_universe(): each value is an
    OHLCV DataFrame with a tz-aware DatetimeIndex and a ``symbol`` attr.

    ``min_quote_volume`` mirrors the live screener's liquidity floor.
    Average quote volume per hour is estimated as ``mean(close*volume)``
    and divided by 24 to compare against per-day Bitget figures: a
    symbol with $5M / day average quote turnover passes the live
    pre-filter and is kept; everything below is discarded so the
    backtest universe matches what the live engine actually trades.
    Set to 0 to disable the filter.

    Symbols with fewer than ``min_bars`` rows are skipped (mid-period
    listings) to keep the cross-section comparable.
    """
    cache_dir = cache_dir or CACHE_DIR
    if not cache_dir.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(cache_dir.glob("*_1h.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception as e:                                   # noqa: BLE001
            log.warning(f"real_universe: skip {path.name}: {e}")
            continue
        if len(df) < min_bars:
            continue
        if min_quote_volume > 0:
            qv_hourly = float((df["close"] * df["volume"]).mean())
            qv_daily = qv_hourly * 24.0
            if qv_daily < min_quote_volume:
                continue
        if tail_bars is not None and len(df) > tail_bars:
            df = df.tail(tail_bars).copy()
        sym_token = path.stem.replace("_1h", "")
        symbol = sym_token.replace("_", "/", 1).replace("_", ":", 1)
        df.attrs["symbol"] = symbol
        out[symbol] = df
        if max_symbols and len(out) >= max_symbols:
            break
    return out


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
    jump_intensity: float = 0.005,         # baseline P(jump per bar) — λ₀
    jump_mean: float = 0.0,                # mean log jump size
    jump_std: float = 0.06,                # std of log jump size
    drift_per_bar: float = 0.0,
    bar_dt: float = 1.0 / (24.0 * 365.0),
    # ---- Hawkes self-excitation (Aït-Sahalia, Cacho-Diaz, Laeven 2014) ---- #
    # Setting hawkes_alpha=0 falls back to a constant-intensity Poisson
    # jump process (the original behaviour). Non-zero values make the
    # arrival rate self-exciting:
    #     λ(t) = λ₀ + α · Σ exp(−β(t−t_i))     for prior jumps t_i < t
    # and bias the next jump's mean *toward* the most recent jump's
    # direction with weight ``hawkes_dir_persistence`` ∈ [0, 1].
    # This is the right data-generating model for AlphaPulse: the
    # strategy thesis is "jumps cluster in the same direction", and
    # IID Heston-Poisson jumps actively contradict that thesis.
    hawkes_alpha: float = 0.6,             # excitation jump per past jump
    hawkes_beta:  float = 0.08,            # decay rate (≈ half-life 9 bars)
    hawkes_dir_persistence: float = 0.6,   # 0=symmetric, 1=fully directional
) -> pd.DataFrame:
    """Heston SV with Hawkes self-exciting jumps.

    Captures three crypto-perp empirical regularities the LM-driven
    AlphaPulse design relies on:

      * **Vol clustering** (Engle 1982 ARCH; Heston 1993): high
        variance begets high variance via the v-process.
      * **Jump self-excitation** (Hawkes 1971; Aït-Sahalia et al. 2014):
        a jump at t_i raises the hazard rate of the next jump for
        ~β⁻¹ bars afterwards.
      * **Directional persistence post-jump** (Lee 2012, RFS 25(2)):
        information takes time to diffuse so the next jump is biased
        in the direction of the most recent one.

    Setting ``hawkes_alpha=0`` recovers the original IID Heston-jump
    behaviour for backwards compatibility with older tests.
    """
    dt = bar_dt
    v = np.empty(n_bars)
    log_p = np.empty(n_bars)
    v[0] = theta
    log_p[0] = np.log(100.0)

    z = rng.standard_normal((n_bars - 1, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(max(1 - rho * rho, 0.0)) * z[:, 1]

    hawkes_excitation = 0.0
    last_jump_dir = 0.0      # +1 / -1 / 0

    for t in range(1, n_bars):
        v_prev = max(v[t - 1], 1e-12)
        v[t] = max(v_prev + kappa * (theta - v_prev) * dt
                    + xi * np.sqrt(v_prev * dt) * z[t - 1, 1], 1e-12)
        ret = (mu * dt + drift_per_bar
                + np.sqrt(v_prev * dt) * z[t - 1, 0])
        # Decay Hawkes excitation
        hawkes_excitation *= np.exp(-hawkes_beta)
        # Effective jump probability at this bar — cap at 0.5 to prevent
        # the runaway-feedback regime where successive jumps push
        # `hawkes_excitation` above 1 and every bar produces a jump
        # (causing the synthetic prices to compound to absurd values
        # like 1e9).
        lam = min(0.5, jump_intensity + hawkes_excitation)
        if rng.random() < lam:
            # Bias jump direction toward the most recent jump's sign
            base_mean = jump_mean
            if last_jump_dir != 0 and hawkes_dir_persistence > 0:
                base_mean += last_jump_dir * hawkes_dir_persistence * jump_std * 0.6
            j = rng.normal(base_mean, jump_std)
            ret += j
            hawkes_excitation += hawkes_alpha
            last_jump_dir = float(np.sign(j))
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
    # Calibrated to realistic crypto-perp empirics: BTC bull cycles
    # average ~80%/y, bears ~−40%/y, hourly ATR ≈ 1–2%, jumps ≈ once
    # per few days. The previous values (e.g. f_drift=0.00015 → +260%/y
    # plus +1.2% mean jumps every 200 bars) caused compounded paths to
    # blow up to +1e6%, breaking any meaningful baseline comparison.
    if regime == "bull_jump":
        f_drift = 0.00005                   # ≈ +55%/y
        f_jump_intensity = 0.002            # one jump per ~500 bars
        f_jump_mean = 0.004                 # +0.4% upward-skewed jumps
    elif regime == "bull_diffusion":
        f_drift = 0.00004                   # ≈ +42%/y, no jumps
        f_jump_intensity = 0.0005
        f_jump_mean = 0.0
    elif regime == "bear_jump":
        f_drift = -0.00004
        f_jump_intensity = 0.002
        f_jump_mean = -0.004
    elif regime == "high_vol":
        f_drift = 0.0
        f_jump_intensity = 0.008            # frequent jumps, mixed dir
        f_jump_mean = 0.0
    else:                                    # neutral
        f_drift = 0.0
        f_jump_intensity = 0.002
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
