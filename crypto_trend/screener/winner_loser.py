"""Cross-sectional WINNER / LOSER screener — academically-grounded version.

Problem
-------
Out of 100+ live USDT-perp symbols on Bitget, identify the ones that *just*
started a sudden directional move that is statistically inconsistent with
their own recent volatility. The screener feeds the trend-following
strategy, which then waits for a Donchian breakout in the screener's
direction before risking capital.

Statistical model
-----------------
Treat each symbol's log-return process as a continuous diffusion plus a
rare jump component:

    dlog P_t = μ dt + σ_t dW_t + J_t dN_t

where N_t is a counting process for jumps and J_t the jump size. We want
to detect at time t the event {dN_t = 1} — a freshly-arrived jump.

Three orthogonal estimators are combined:

1. **Bipower variation** (Barndorff-Nielsen & Shephard 2004,
   "Power and bipower variation with stochastic volatility and jumps",
   Journal of Financial Econometrics 2(1), 1-37):

       BV_t = (π/2) · (1/(K-1)) · Σ_{i=t-K+1}^{t-1} |r_{i-1}| · |r_i|

   BV is a consistent estimator of the *integrated diffusion variance*
   that remains finite-valued even when the path contains jumps —
   because the product |r_{i-1}|·|r_i| can carry at most one jump term
   and is therefore o_p(Δt^{1/2}). This makes BV the gold-standard
   denominator for jump tests.

2. **Lee-Mykland jump statistic** (Lee & Mykland 2008,
   "Jumps in financial markets: a new nonparametric test and jump
   dynamics", Review of Financial Studies 21(6), 2535-2563):

       L_t = r_t / sqrt(BV_t)

   Under the no-jump null L_t →^d N(0, 1). Under a jump |L_t| diverges.
   This is mathematically the "right" generalization of robust z that
   accounts for stochastic volatility.

3. **Multi-horizon momentum agreement** — the screened jump is required
   to point in the same direction as the cumulative log-return over at
   least `min_horizons_agree` of the (1, 4, 24)-bar windows. The idea is
   borrowed from time-series momentum (Moskowitz, Ooi, Pedersen 2012,
   "Time series momentum", JFE 104(2), 228-250) — bars whose 24-hour
   trend opposes the jump are very likely noise rather than the start of
   a sustained move.

A persistence proxy (Hurst R/S, Mandelbrot 1969) is then used to drop
mean-reverting tape, and a liquidity floor removes wides-spread perps.

Implementation contract
-----------------------
* Each symbol is scored independently — no cross-sectional standardisation
  inside one cycle, so that the threshold has a stable meaning across runs.
* The jump test point itself (returns[-1]) is excluded from BV, eliminating
  the well-known downward bias of "self-included" volatility estimators.
* Hurst is computed on a separate, longer window AFTER the LM filter has
  passed, so the R/S estimate is not contaminated by the very jump we are
  trying to characterise.
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
# Core statistics
# --------------------------------------------------------------------------- #


def robust_z(returns: np.ndarray) -> float:
    """Median / MAD z-score of the *latest* return.

    Kept exposed for backward compatibility and as a sanity-check fallback.
    The Lee-Mykland statistic (below) is now preferred and is what the
    screener uses internally.
    """
    if returns.size < 8:
        return 0.0
    med = np.median(returns)
    mad = np.median(np.abs(returns - med))
    if mad == 0:
        return 0.0
    return float((returns[-1] - med) / (1.4826 * mad))


def bipower_variation(returns: np.ndarray) -> float:
    """Barndorff-Nielsen & Shephard (2004) bipower variation.

    Returns ``(π/2) · mean_i(|r_{i-1}| · |r_i|)`` — a jump-robust estimator
    of the integrated diffusion variance.
    """
    if returns.size < 4:
        return 0.0
    abs_r = np.abs(returns)
    return float((np.pi / 2.0) * (abs_r[:-1] * abs_r[1:]).mean())


def lee_mykland_statistic(returns: np.ndarray, window: int = 24) -> float:
    """Lee & Mykland (2008) jump test statistic at the most recent bar.

    Formally  L_t = r_t / sqrt(BV_t)  with BV_t computed *strictly before*
    t so the test point does not pollute its own denominator.

    Under the null of no jump in r_t,  L_t →^d N(0, 1).  Under a jump,
    |L_t| → ∞ in probability, regardless of the local diffusion volatility.
    """
    if returns.size < window + 2:
        return 0.0
    prior = returns[-(window + 1):-1]              # exclude the test bar
    bv = bipower_variation(prior)
    if bv <= 0:
        return 0.0
    return float(returns[-1] / np.sqrt(bv))


def multi_horizon_alignment(returns: np.ndarray, sign: int,
                             horizons: tuple[int, ...] = (1, 4, 24)) -> int:
    """Count of pre-jump horizons whose cumulative log-return agrees with `sign`.

    The latest return ``returns[-1]`` is the jump under test, so it must be
    excluded from the alignment sums — otherwise a single 10% jump bar
    would dominate any horizon containing it and the filter would always
    return "agree". By comparing the **pre-jump** cumulative direction
    against the jump's sign, we measure whether the jump *extends* a
    pre-existing trend (informative) or *fights against* it (likely an
    isolated outlier).

    The horizon ``1`` therefore looks at the single bar immediately
    before the jump. Horizon ``h`` looks at the ``h`` bars ending at
    that pre-jump bar.
    """
    if returns.size < 2 or sign == 0:
        return 0
    pre_jump = returns[:-1]
    agree = 0
    for h in horizons:
        if pre_jump.size < h:
            continue
        if np.sign(pre_jump[-h:].sum()) == sign:
            agree += 1
    return agree


def hurst_rs(series: np.ndarray, min_chunk: int = 8) -> float:
    """Rescaled-range Hurst estimate (Mandelbrot 1969).  0.5 = random walk."""
    n = series.size
    if n < min_chunk * 4:
        return 0.5
    chunk_sizes = [c for c in (min_chunk, min_chunk * 2, min_chunk * 4,
                                 min_chunk * 8) if c <= n // 2]
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
    z_momentum: float    # Lee-Mykland statistic L_t
    hurst: float
    vol: float
    quote_volume: float
    composite: float
    last_price: float
    horizons_agree: int = 0

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "L_jump": round(self.z_momentum, 3),
            "hurst": round(self.hurst, 3),
            "vol": round(self.vol, 4),
            "quote_volume": round(self.quote_volume, 0),
            "composite": round(self.composite, 3),
            "last_price": self.last_price,
            "horizons_agree": self.horizons_agree,
        }


# --------------------------------------------------------------------------- #
# Screener
# --------------------------------------------------------------------------- #


class WinnerLoserScreener:
    """Winner/loser screener with a Lee-Mykland-based decision core.

    The constructor accepts the same parameter names as the earlier ad-hoc
    version (``lookback``, ``z_threshold`` …) so existing callers are not
    broken; ``lookback`` is now used as the bipower-variation window.
    """

    def __init__(
        self,
        lookback: int = 24,                       # BV window (bars)
        z_threshold: float = 2.5,                 # |L_t| floor
        hurst_floor: float = 0.55,
        min_quote_volume: float = 5e6,
        top_n: int = 10,
        weight_momentum: float = 0.7,
        weight_persistence: float = 0.3,
        horizons: tuple[int, ...] = (1, 4, 24),
        min_horizons_agree: int = 2,
    ) -> None:
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.hurst_floor = hurst_floor
        self.min_quote_volume = min_quote_volume
        self.top_n = top_n
        self.w_mom = weight_momentum
        self.w_pers = weight_persistence
        self.horizons = horizons
        self.min_horizons_agree = min_horizons_agree

    # ------------------------------------------------------------------ #
    def score(self, candles: pd.DataFrame, quote_volume: float
              ) -> tuple[float, float, float, str, int] | None:
        """Score one symbol. Returns (composite, L, H, side, agree) or None."""
        if (candles is None or candles.empty
                or len(candles) < self.lookback + 4):
            return None
        closes = candles["close"].to_numpy(dtype=float)
        if (closes <= 0).any():
            return None
        rets = np.diff(np.log(closes))

        # 1. Lee-Mykland jump statistic on the latest bar
        L = lee_mykland_statistic(rets, window=self.lookback)
        if math.isnan(L) or abs(L) < self.z_threshold:
            return None
        side_sign = 1 if L > 0 else -1

        # 2. Multi-horizon directional agreement (filters 1-bar noise spikes)
        agree = multi_horizon_alignment(rets, side_sign, self.horizons)
        if agree < self.min_horizons_agree:
            return None

        # 3. Liquidity floor (cheap test, but check after LM so volume isn't
        #    re-evaluated for a trivially-failing candidate)
        if quote_volume < self.min_quote_volume:
            return None

        # 4. Persistence — Hurst R/S over a longer window, computed last so
        #    its estimate isn't dominated by the jump bar itself
        h = hurst_rs(rets[-min(rets.size, 256):])
        if h < self.hurst_floor:
            return None

        # Composite: weighted sum of jump strength and persistence agreement,
        # boosted by an extra term that rewards multi-horizon support so two
        # candidates with similar L break tie on the cleaner trend.
        composite = (
            self.w_mom * L
            + self.w_pers * (h - 0.5) * 4.0 * side_sign
            + 0.25 * agree * side_sign
        )
        side = "long" if side_sign > 0 else "short"
        return float(composite), float(L), float(h), side, int(agree)

    # ------------------------------------------------------------------ #
    def run(
        self,
        symbols: Iterable[str],
        ohlcv_provider,                            # callable: symbol -> DataFrame
        quote_volume_provider,                     # callable: symbol -> float
    ) -> list[ScreenResult]:
        results: list[ScreenResult] = []
        for sym in symbols:
            try:
                candles = ohlcv_provider(sym)
                qv = quote_volume_provider(sym)
            except Exception as e:                        # noqa: BLE001
                log.debug(f"screener: {sym} skipped: {e}")
                continue
            scored = self.score(candles, qv)
            if scored is None:
                continue
            composite, L, h, side, agree = scored
            closes = candles["close"].to_numpy(dtype=float)
            results.append(ScreenResult(
                symbol=sym,
                side=side,
                z_momentum=L,
                hurst=h,
                vol=realized_vol(np.diff(np.log(closes))),
                quote_volume=qv,
                composite=composite,
                last_price=float(closes[-1]),
                horizons_agree=agree,
            ))
        results.sort(key=lambda r: abs(r.composite), reverse=True)
        return results[: self.top_n]
