"""ContinuousKellyCalibrator contract tests.

Pin the academic behaviour (Nadaraya 1964 / Watson 1964 NW regression;
Silverman 1986 bandwidth; Györfi-Krzyzak-Walk 2008 §5.4 effective
sample size).
"""
from __future__ import annotations

import numpy as np

from crypto_trend.risk.continuous_kelly_calibrator import (
    ContinuousKellyCalibrator)


def test_cold_start_returns_bootstrap():
    cal = ContinuousKellyCalibrator(min_warmup_trades=50)
    # No history → bootstrap.
    assert cal.kelly_fraction(0.5) == 0.0
    assert cal.kelly_fraction(0.5, bootstrap=0.7) == 0.7
    assert cal.is_warm() is False


def test_warm_monotonic_kelly_in_predictor():
    """Synthetic: per-event μ rises linearly with |predictor|, σ fixed.
    Check the Nadaraya-Watson fit recovers monotone Kelly (subject to
    sizing_cap). Tested with a large lookback + a high sizing_cap so
    the underlying NW signal is observable without cap saturation."""
    rng = np.random.default_rng(0)
    cal = ContinuousKellyCalibrator(min_warmup_trades=100,
                                       lookback_trades=4000,
                                       fractional_kelly=1.0,    # raw
                                       min_effective_n=10.0,
                                       sizing_cap=1000.0)        # observe raw
    for _ in range(3000):
        p = rng.uniform(0.1, 2.0)
        mu = 0.0005 + 0.003 * (p - 0.1)
        pnl = rng.normal(mu, 0.05)
        cal.add_trade(p, pnl)
    assert cal.is_warm() is True
    f_weak = cal.kelly_fraction(0.3)
    f_mid  = cal.kelly_fraction(1.0)
    f_high = cal.kelly_fraction(1.8)
    assert 0 <= f_weak < f_mid < f_high, (
        f"NW-fit Kelly should rise with predictor; got {f_weak}, "
        f"{f_mid}, {f_high}")


def test_fractional_kelly_multiplier():
    """fractional_kelly α multiplies the returned f without altering
    the underlying μ̂/σ̂² estimate."""
    rng = np.random.default_rng(1)
    cal_full = ContinuousKellyCalibrator(min_warmup_trades=100,
                                            fractional_kelly=1.0,
                                            min_effective_n=10.0)
    cal_qtr  = ContinuousKellyCalibrator(min_warmup_trades=100,
                                            fractional_kelly=0.25,
                                            min_effective_n=10.0)
    data = [(rng.uniform(0.5, 2.0), rng.normal(0.02, 0.05))
             for _ in range(300)]
    for p, r in data:
        cal_full.add_trade(p, r)
        cal_qtr.add_trade(p, r)
    f_full = cal_full.kelly_fraction(1.0)
    f_qtr  = cal_qtr.kelly_fraction(1.0)
    assert abs(f_qtr - 0.25 * f_full) < 1e-9


def test_low_effective_n_falls_back_to_bootstrap():
    """Querying far from training support → effective n < threshold
    → bootstrap fallback (anti-overfit safeguard)."""
    rng = np.random.default_rng(2)
    cal = ContinuousKellyCalibrator(min_warmup_trades=80,
                                       min_effective_n=200.0,
                                       fractional_kelly=1.0)
    for _ in range(100):
        cal.add_trade(rng.uniform(0.5, 1.0), rng.normal(0.01, 0.05))
    # All training predictors in [0.5, 1.0].  Query at 100.0 — no
    # local support → effective n ≈ 0 → bootstrap.
    assert cal.kelly_fraction(100.0, bootstrap=0.42) == 0.42


def test_rolling_window_forgets_old_regimes():
    rng = np.random.default_rng(3)
    cal = ContinuousKellyCalibrator(min_warmup_trades=50,
                                       lookback_trades=100,
                                       fractional_kelly=1.0,
                                       min_effective_n=10.0)
    # Old regime: negative expectancy at p=1.0
    for _ in range(150):
        cal.add_trade(1.0, rng.normal(-0.05, 0.05))
    # New regime: positive expectancy at p=1.0
    for _ in range(100):
        cal.add_trade(1.0, rng.normal(+0.05, 0.05))
    assert len(cal.history) == 100
    # Lookback forgot the old negative regime → f should be positive.
    f = cal.kelly_fraction(1.0)
    assert f > 0, "rolling-window must forget old non-stationary regime"
