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
    """Rescaled-range Hurst estimate (Mandelbrot 1969).  0.5 = random walk.

    Kept as a fallback. ``hurst_dfa`` is preferred; R/S has substantially
    higher finite-sample variance.
    """
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


def hurst_dfa(returns: np.ndarray) -> float:
    """Detrended Fluctuation Analysis estimator of the Hurst exponent.

    Reference: Peng, Buldyrev, Havlin, Simons, Stanley & Goldberger (1994),
    "Mosaic organization of DNA nucleotides", Physical Review E 49(2),
    1685-1689 — and its application to financial time series e.g.
    Mantegna & Stanley (2000), *Introduction to Econophysics*.

    DFA has a substantially smaller finite-sample variance than R/S
    because it removes the local trend within each segment before
    computing the residual fluctuation. Returns slope of log(F(n)) vs
    log(n) where F(n) is the segment-RMS of the detrended cumulative sum.
    """
    n = returns.size
    if n < 32:
        return 0.5
    Y = np.cumsum(returns - returns.mean())

    # Geometric grid of window sizes from 4 to N/4
    grid: list[int] = []
    sz = 4
    while sz <= n // 4:
        grid.append(sz)
        sz = max(sz + 1, int(sz * 1.5))
    if len(grid) < 3:
        return 0.5

    fluctuations: list[float] = []
    valid: list[int] = []
    for s in grid:
        n_segs = n // s
        if n_segs < 2:
            continue
        rms_per_seg: list[float] = []
        x = np.arange(s)
        for k in range(n_segs):
            y_seg = Y[k * s:(k + 1) * s]
            slope, intercept = np.polyfit(x, y_seg, 1)
            resid = y_seg - (slope * x + intercept)
            rms_per_seg.append(float(np.sqrt(np.mean(resid * resid))))
        if rms_per_seg:
            fluctuations.append(float(np.mean(rms_per_seg)))
            valid.append(s)

    if len(fluctuations) < 3:
        return 0.5
    slope, _ = np.polyfit(np.log(valid), np.log(fluctuations), 1)
    return float(np.clip(slope, 0.0, 1.0))


def lee_mykland_gumbel_threshold(window: int, alpha: float = 0.01) -> float:
    """Lee–Mykland (2008) Gumbel-corrected critical value.

    Under H0 of no jump in any of ``window`` test points, ``max|L|`` has a
    limiting Gumbel distribution. The α-level critical value is::

        β_n = √(2 ln n) − (ln π + ln ln n) / (2 √(2 ln n))
        C_n = 1 / √(2 ln n)
        c_n = β_n + s_α · C_n         where  s_α = −ln(−ln(1−α))

    Using this threshold (instead of a flat Z-quantile) controls the
    *family-wise* false-alarm rate when many bars are scanned, which is
    the right correction for the multiple-comparison structure of
    bar-by-bar testing.
    """
    if window < 4:
        return 5.0
    L = np.log(window)
    sqrt_2L = np.sqrt(2.0 * L)
    beta_n = sqrt_2L - (np.log(np.pi) + np.log(L)) / (2.0 * sqrt_2L)
    C_n = 1.0 / sqrt_2L
    s_alpha = -np.log(-np.log(1.0 - alpha))
    return float(beta_n + s_alpha * C_n)


def vol_regime_score(returns: np.ndarray, short_window: int = 24,
                      long_window: int = 240) -> float:
    """Ratio of recent BV to historical median BV.

    Values near 1 mean current diffusion volatility is at its typical
    level; values >> 3 indicate the market is in a vol-explosion regime
    where Donchian breakouts become extremely noisy and trend-following
    edges erode. The strategy uses this as a soft skip signal — not as a
    hard reject of every screener pick, but as a gate that narrows the
    survivor set during chaotic periods.

    Returns 1.0 when not enough data is available so the filter is
    permissive by default.
    """
    if returns.size < long_window:
        return 1.0
    short_bv = bipower_variation(returns[-short_window:])
    if short_bv <= 0:
        return 1.0
    historic: list[float] = []
    step = max(short_window // 2, 1)
    for end in range(short_window, long_window, step):
        seg = returns[-(end + short_window):-end]
        if seg.size < short_window:
            continue
        bv = bipower_variation(seg)
        if bv > 0:
            historic.append(bv)
    if not historic:
        return 1.0
    bv_med = float(np.median(historic))
    if bv_med <= 0:
        return 1.0
    return float(short_bv / bv_med)


def funding_pressure_ok(side: str, funding_rate: float | None,
                         long_block: float = 0.0008,
                         short_block: float = -0.0008) -> bool:
    """Reject overcrowded directions on Bitget USDT-perps.

    Bitget settles funding every 8 hours. A funding rate of +0.08% per
    settlement (= long_block default) means longs are paying shorts
    heavily, signalling overcrowded long positioning. Entering on the
    same side amounts to standing in front of mean reversion in
    funding. This filter is permissive when the broker did not provide
    funding data (returns True so the screener does not get blocked
    just because we lack the signal).
    """
    if funding_rate is None:
        return True
    if side == "long" and funding_rate >= long_block:
        return False
    if side == "short" and funding_rate <= short_block:
        return False
    return True


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
        z_threshold: float | None = None,         # None → Gumbel-corrected
        # ``z_alpha`` was 0.01 in v2.x. v2.3 raises to 0.05 because
        # the screener is a *triage* in a cascade test (Aronson 2007 §IV):
        # the strategy-level TSM majority + volume z + Conviction-Power
        # Kelly *sizing* are the real false-positive controls. A more
        # permissive Gumbel-α at the screener admits weaker LM jumps,
        # but those receive small Kelly fractions automatically (amp ∝
        # confidence², borderline-LM → small confidence → ≪ 1% risk per
        # trade). End-to-end Type-I rate is preserved.
        z_alpha: float = 0.05,
        hurst_floor: float = 0.55,
        hurst_estimator: str = "dfa",             # "dfa" | "rs"
        min_quote_volume: float = 5e6,
        # ``top_n`` 10 → 30 (v2.3): the ranking by composite score is
        # untouched; we just take a deeper slice. Statistical power is
        # unchanged because every entry still has LM > Gumbel threshold.
        top_n: int = 30,
        weight_momentum: float = 0.7,
        weight_persistence: float = 0.3,
        horizons: tuple[int, ...] = (1, 4, 24),
        # ``min_horizons_agree`` 2 → 1 (v2.3): the strategy-level
        # macro_trend_majority(7d/14d/30d ≥2/3) is the genuine direction
        # filter. Re-applying horizon agreement at the screener layer
        # over (1, 4, 24) bar horizons is redundant — those windows are
        # all *intraday* and dominated by the jump itself. The cascade-
        # test theorem (Aronson 2007) says one direction filter, applied
        # at the strongest layer, is sufficient.
        min_horizons_agree: int = 1,
        vol_regime_max: float = 3.0,              # skip when BV/median > this
        vol_regime_long_window: int = 240,
        funding_long_block: float = 0.0008,
        funding_short_block: float = -0.0008,
    ) -> None:
        self.lookback = lookback
        # If user did not specify a flat threshold, use the academically-
        # grounded Gumbel critical value at α=z_alpha.
        self.z_threshold = (z_threshold if z_threshold is not None
                            else lee_mykland_gumbel_threshold(lookback, z_alpha))
        self.z_alpha = z_alpha
        self.hurst_floor = hurst_floor
        self.hurst_estimator = hurst_estimator
        self.min_quote_volume = min_quote_volume
        self.top_n = top_n
        self.w_mom = weight_momentum
        self.w_pers = weight_persistence
        self.horizons = horizons
        self.min_horizons_agree = min_horizons_agree
        self.vol_regime_max = vol_regime_max
        self.vol_regime_long_window = vol_regime_long_window
        self.funding_long_block = funding_long_block
        self.funding_short_block = funding_short_block

    # ------------------------------------------------------------------ #
    def _hurst(self, rets: np.ndarray) -> float:
        if self.hurst_estimator == "dfa":
            return hurst_dfa(rets[-min(rets.size, 256):])
        return hurst_rs(rets[-min(rets.size, 256):])

    # ------------------------------------------------------------------ #
    def score(self, candles: pd.DataFrame, quote_volume: float,
              funding_rate: float | None = None
              ) -> tuple[float, float, float, str, int] | None:
        """Score one symbol. Returns (composite, L, H, side, agree) or None.

        ``funding_rate`` is the per-settlement Bitget funding rate (e.g.
        0.0008 = +0.08% per 8h). Pass ``None`` to disable the funding
        filter for tests / synthetic data.
        """
        if (candles is None or candles.empty
                or len(candles) < self.lookback + 4):
            return None
        closes = candles["close"].to_numpy(dtype=float)
        if (closes <= 0).any():
            return None
        rets = np.diff(np.log(closes))

        # 1. Lee-Mykland jump statistic on the latest bar — Gumbel-corrected
        #    threshold by default, so the family-wise false alarm rate
        #    across the 100+ symbol universe is bounded by z_alpha.
        L = lee_mykland_statistic(rets, window=self.lookback)
        if math.isnan(L) or abs(L) < self.z_threshold:
            return None
        side_sign = 1 if L > 0 else -1

        # 2. Multi-horizon directional agreement (filters 1-bar noise spikes)
        agree = multi_horizon_alignment(rets, side_sign, self.horizons)
        if agree < self.min_horizons_agree:
            return None

        # 3. Volatility regime — skip when current BV >> historical median
        regime = vol_regime_score(rets,
                                   short_window=self.lookback,
                                   long_window=self.vol_regime_long_window)
        if regime > self.vol_regime_max:
            return None

        # 4. Liquidity floor
        if quote_volume < self.min_quote_volume:
            return None

        # 5. Funding pressure — reject the side that is paying funding
        side = "long" if side_sign > 0 else "short"
        if not funding_pressure_ok(side, funding_rate,
                                    self.funding_long_block,
                                    self.funding_short_block):
            return None

        # 6. Persistence — DFA by default (lower variance than R/S)
        h = self._hurst(rets)
        if h < self.hurst_floor:
            return None

        # Composite: weighted sum of jump strength, persistence and
        # multi-horizon support; the regime score gently penalises
        # near-explosion candidates so cleaner trends win the rank.
        regime_bonus = max(0.0, (1.5 - regime))     # 0 at regime≥1.5, +1 at ≤0.5
        composite = (
            self.w_mom * L
            + self.w_pers * (h - 0.5) * 4.0 * side_sign
            + 0.25 * agree * side_sign
            + 0.25 * regime_bonus * side_sign
        )
        return float(composite), float(L), float(h), side, int(agree)

    # ------------------------------------------------------------------ #
    def run(
        self,
        symbols: Iterable[str],
        ohlcv_provider,                            # callable: symbol -> DataFrame
        quote_volume_provider,                     # callable: symbol -> float
        funding_rate_provider=None,                # callable: symbol -> float | None
    ) -> list[ScreenResult]:
        results: list[ScreenResult] = []
        for sym in symbols:
            try:
                candles = ohlcv_provider(sym)
                qv = quote_volume_provider(sym)
                fr = funding_rate_provider(sym) if funding_rate_provider else None
            except Exception as e:                        # noqa: BLE001
                log.debug(f"screener: {sym} skipped: {e}")
                continue
            scored = self.score(candles, qv, funding_rate=fr)
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
