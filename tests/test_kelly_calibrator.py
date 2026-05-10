"""KellyCalibrator contract tests.

Pin the academic-foundation behaviour of the rolling per-bin OOS
Kelly sizer (Cover & Thomas 1991 §16; Lopez de Prado 2018 §10):

  1. Cold-start fallback: with no history, use the bootstrap.
  2. Warm-state behaviour: per-bin f_b = μ_b/σ_b² extracted from
     history, sign-aware.
  3. Anti-overfitting: bin with too few samples falls back to
     bootstrap; sizing_cap respected.
  4. Rolling-window forgetting: old observations drop out.
"""
from __future__ import annotations

import numpy as np

from crypto_trend.risk.kelly_calibrator import KellyCalibrator


def test_cold_start_uses_bootstrap():
    cal = KellyCalibrator(n_bins=4, min_per_bin=10, sizing_cap=5.0)
    # No history → bootstrap (default 0)
    assert cal.kelly_fraction(0.5) == 0.0
    assert cal.kelly_fraction(0.5, bootstrap=0.7) == 0.7


def test_warm_state_extracts_per_bin_kelly():
    """Feed a synthetic history with monotone (μ_b, σ_b²) increasing
    in |predictor|, and verify the fitted per-bin f_b is monotone."""
    rng = np.random.default_rng(0)
    cal = KellyCalibrator(n_bins=4, min_per_bin=20, sizing_cap=5.0,
                            lookback_trades=10000)
    # Continuously-distributed predictor; per-event μ/σ rises with
    # |predictor|.
    for _ in range(400):
        p = rng.uniform(0.1, 2.0)
        # μ rises linearly with |p|, σ stays constant
        mu = 0.001 + 0.02 * (p - 0.1)
        pnl = rng.normal(mu, 0.05)
        cal.add_trade(p, pnl)
    assert cal.is_warm() is True

    f_weak = cal.kelly_fraction(0.2)
    f_mid  = cal.kelly_fraction(1.0)
    f_high = cal.kelly_fraction(1.9)
    # Monotone: stronger predictor → bigger Kelly fraction
    assert f_weak < f_mid <= f_high, (
        f"f_weak={f_weak}, f_mid={f_mid}, f_high={f_high} should be monotone")
    # All non-negative (sign matches predictor sign which is +)
    assert f_weak >= 0 and f_mid > 0 and f_high > 0
    # Cap respected
    assert f_high <= cal.sizing_cap


def test_sign_awareness():
    """Predictor sign flips Kelly sign; magnitude unchanged."""
    rng = np.random.default_rng(1)
    cal = KellyCalibrator(n_bins=2, min_per_bin=10, sizing_cap=5.0,
                            lookback_trades=1000)
    # Two strata, both POSITIVE predictor, different magnitudes
    # (sign-aligned realisations are positive on average).
    for _ in range(40):
        cal.add_trade(1.0, rng.normal(0.005, 0.03))
    for _ in range(40):
        cal.add_trade(2.0, rng.normal(0.02, 0.03))
    f_pos = cal.kelly_fraction(1.5)
    f_neg = cal.kelly_fraction(-1.5)
    # Magnitudes equal, signs opposite
    assert abs(f_pos) == abs(f_neg)
    assert f_pos > 0 > f_neg


def test_sparse_bin_falls_back_to_bootstrap():
    """A bin that has been fitted but ended up with NaN (not enough
    samples after the rolling window applied) must route the
    ``kelly_fraction`` lookup to the bootstrap.  This is the per-
    lookup anti-overfit safeguard: aggregate ``is_warm()`` may be
    True (some bin is fitted) yet a query against a sparse / NaN
    bin still returns the bootstrap.  We construct the calibrator
    state directly to isolate this contract from the quantile-
    binning empirics."""
    cal = KellyCalibrator(n_bins=2, min_per_bin=20, sizing_cap=5.0,
                            lookback_trades=10000)
    cal.history = [(0.2, 0.001)] * 50    # required so cache_valid path is exercised
    cal._bin_edges = np.array([-np.inf, 1.0, np.inf])
    cal._bin_counts = np.array([50, 5])
    cal._bin_kellies = np.array([0.7, np.nan])    # bin 0 fitted, bin 1 NaN
    cal._cache_valid = True
    # Aggregate warm because at least one bin is finite.
    assert cal.is_warm() is True
    # Lookup falls into bin 1 (predictor=2.0 ≥ 1.0) → NaN → bootstrap.
    assert cal.kelly_fraction(2.0, bootstrap=0.42) == 0.42
    # Lookup into bin 0 returns the fitted f_b.
    assert abs(cal.kelly_fraction(0.5) - 0.7) < 1e-9


def test_rolling_window_forgets_old_data():
    cal = KellyCalibrator(n_bins=2, min_per_bin=10, sizing_cap=5.0,
                            lookback_trades=50)
    rng = np.random.default_rng(3)
    # Phase A: 100 trades with negative expectancy (will be forgotten)
    for _ in range(100):
        cal.add_trade(1.0, rng.normal(-0.05, 0.05))
    # Phase B: 50 trades with positive expectancy (current regime)
    for _ in range(50):
        cal.add_trade(1.0, rng.normal(+0.05, 0.05))
    # History trimmed to 50 most recent (positive-expectancy) events
    assert len(cal.history) == 50
    # The fitted f for predictor=1.0 should be POSITIVE despite the
    # earlier negative-regime data which has been forgotten.
    f = cal.kelly_fraction(1.0)
    assert f > 0, (
        "rolling-window calibrator must forget old non-stationary "
        "regimes; otherwise it stale-anchors to early history "
        "(Dyck-Werner-Stambaugh 2005 estimation-risk concern)")
