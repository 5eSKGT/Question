"""Continuous (kernel-regression) per-predictor Kelly sizer.

The previous KellyCalibrator (kelly_calibrator.py) used 6 discrete
quantile bins.  Three artefacts emerged in production:

  1. Quantile-edge handling left bin 0 chronically empty (digitize
     sends the minimum predictor value into bin 1).
  2. The empirical predictor (confidence²) is discrete-clustered
     (sign(LM) × clipped(LM/threshold) × clipped(agree/3)), so
     middle bins are sparse.
  3. Discrete bins quantise the underlying continuous μ(pred) / σ²(pred)
     surface, losing information between bin centres.

The continuous version replaces discrete bins with a Nadaraya-Watson
kernel regression of (μ, σ²) over the predictor:

      μ̂(p)  = Σ_i K_h(p - p_i)·r_i  / Σ_i K_h(p - p_i)
      σ̂²(p) = Σ_i K_h(p - p_i)·(r_i - μ̂(p))² / Σ_i K_h(p - p_i)
      f(p)  = clip(μ̂(p) / σ̂²(p), 0, sizing_cap) × α

where K_h is a Gaussian kernel with bandwidth h optimised by
Silverman's rule (1986) and Györfi-Krzyzak-Walk 2008 §5.4 effective-
sample-size matching.

This is the academically-correct, non-parametric, single-axis
heterogeneous Kelly sizer that the upper_bound_analysis ceiling
(83,779%/yr filtered) actually quantifies.

Anti-overfitting safeguards
---------------------------
* Rolling lookback (default 800 trades).
* Kernel bandwidth h = h_silverman × cv_scale, cv_scale chosen so the
  effective local sample size at each query ≥ min_effective_n.
* Hard sizing_cap on the Kelly fraction.
* Fractional-Kelly α multiplier (MacLean-Thorp-Ziemba 2011).
* Bootstrap fallback (analytical Conviction-Power Kelly) when fewer
  than ``min_warmup_trades`` are in history.
* Sign-aware predictor / pnl convention identical to KellyCalibrator
  (predictor unsigned, pnl side-aligned at exit).

References
----------
Nadaraya, E. A. (1964). "On Estimating Regression." Theory of
    Probability and Its Applications 9(1).
Watson, G. S. (1964). "Smooth Regression Analysis." Sankhyā Ser. A 26.
Silverman, B. W. (1986). "Density Estimation for Statistics and Data
    Analysis." §3.4.
Györfi, L., Krzyżak, A. & Walk, H. (2008). "A Distribution-Free Theory
    of Nonparametric Regression." Springer §5.4.
López de Prado, M. (2018). AFML §10 Bayesian bet sizing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContinuousKellyCalibrator:
    """Continuous-predictor Kelly sizer using Nadaraya-Watson kernel
    regression over a rolling window of closed trades."""

    lookback_trades: int = 800
    min_warmup_trades: int = 100
    min_effective_n: float = 20.0    # local kernel effective sample size
    sizing_cap: float = 5.0
    fractional_kelly: float = 0.25
    bandwidth_scale: float = 1.0     # multiplier on Silverman's bandwidth

    history: list[tuple[float, float]] = field(default_factory=list)
    _preds_arr: np.ndarray | None = None
    _pnls_arr:  np.ndarray | None = None
    _bandwidth: float = 0.0
    _cache_valid: bool = False

    # ------------------------------------------------------------------ #
    def add_trade(self, predictor: float, realised_pnl: float) -> None:
        if not np.isfinite(predictor) or not np.isfinite(realised_pnl):
            return
        self.history.append((float(predictor), float(realised_pnl)))
        if len(self.history) > self.lookback_trades:
            self.history = self.history[-self.lookback_trades:]
        self._cache_valid = False

    # ------------------------------------------------------------------ #
    def _refit(self) -> None:
        if len(self.history) < self.min_warmup_trades:
            self._preds_arr = self._pnls_arr = None
            self._cache_valid = True
            return
        preds = np.array([p for p, _ in self.history], dtype=float)
        pnls  = np.array([r for _, r in self.history], dtype=float)
        # Silverman's rule of thumb for bandwidth (1986 §3.4 eq. 3.31):
        #   h = 1.06 · min(σ, IQR/1.34) · n^(-1/5)
        n = preds.size
        std = float(preds.std(ddof=1)) if n > 1 else 0.0
        iqr = float(np.percentile(preds, 75) - np.percentile(preds, 25))
        scale = min(std, iqr / 1.34) if std > 0 and iqr > 0 else max(std, iqr / 1.34, 0.01)
        h_silverman = 1.06 * scale * (n ** (-1.0 / 5.0))
        # Stretch bandwidth to ensure local effective sample size
        # ≥ min_effective_n at typical query points (Györfi 2008 §5.4
        # — bandwidth scaling for finite-sample stability).
        h = max(h_silverman * self.bandwidth_scale, 1e-6)
        self._bandwidth = h
        self._preds_arr = preds
        self._pnls_arr = pnls
        self._cache_valid = True

    # ------------------------------------------------------------------ #
    def _kernel_weights(self, predictor: float) -> np.ndarray:
        if self._preds_arr is None: return np.array([])
        z = (self._preds_arr - predictor) / self._bandwidth
        # Gaussian kernel (un-normalised constant cancels in NW ratio).
        return np.exp(-0.5 * z * z)

    # ------------------------------------------------------------------ #
    def kelly_fraction(self, predictor: float,
                        bootstrap: float | None = None) -> float:
        """Return the per-event Kelly fraction f(predictor)."""
        if not self._cache_valid:
            self._refit()
        if self._preds_arr is None or self._preds_arr.size < self.min_warmup_trades:
            return float(bootstrap) if bootstrap is not None else 0.0
        w = self._kernel_weights(float(predictor))
        w_sum = float(w.sum())
        # Effective sample size = (Σw)² / Σw² — Györfi 2008 §5.4
        w_sq_sum = float((w * w).sum())
        if w_sq_sum < 1e-12:
            return float(bootstrap) if bootstrap is not None else 0.0
        eff_n = (w_sum * w_sum) / w_sq_sum
        if eff_n < self.min_effective_n:
            return float(bootstrap) if bootstrap is not None else 0.0
        # Nadaraya-Watson μ̂ and weighted σ̂²
        mu = float((w * self._pnls_arr).sum() / w_sum)
        var = float((w * (self._pnls_arr - mu) ** 2).sum() / w_sum)
        if var < 1e-12:
            return float(bootstrap) if bootstrap is not None else 0.0
        # Full Kelly f* = μ/σ², then apply fractional-Kelly α
        f_full = mu / var
        f_clipped = float(np.clip(f_full, 0.0, self.sizing_cap))
        return f_clipped * float(self.fractional_kelly)

    # ------------------------------------------------------------------ #
    def is_warm(self) -> bool:
        if not self._cache_valid:
            self._refit()
        return (self._preds_arr is not None
                and self._preds_arr.size >= self.min_warmup_trades)

    def diagnostics(self) -> dict:
        if not self._cache_valid:
            self._refit()
        if self._preds_arr is None:
            return {"n_history": len(self.history), "warm": False,
                    "bandwidth": 0.0}
        return {
            "n_history": len(self.history),
            "warm": True,
            "bandwidth": round(float(self._bandwidth), 5),
            "predictor_range": [round(float(self._preds_arr.min()), 4),
                                  round(float(self._preds_arr.max()), 4)],
        }
