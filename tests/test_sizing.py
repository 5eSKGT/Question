"""Unit tests for the optimal sizing module."""
import numpy as np

from crypto_trend.risk.sizing import (kelly_fraction, optimal_leverage,
                                        optimal_position, vol_target_fraction)


def test_kelly_zero_for_zero_mean():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 500)
    f = kelly_fraction(rets)
    assert f < 0.05      # close to zero


def test_kelly_positive_for_positive_drift():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0008, 0.005, 500)
    f = kelly_fraction(rets)
    assert f > 0.0


def test_kelly_clipped_to_unity():
    """Even an extreme positive-drift series shouldn't exceed 1.0 unscaled."""
    rng = np.random.default_rng(2)
    rets = rng.normal(0.05, 0.001, 500)     # Sharpe ~50, absurd
    f = kelly_fraction(rets)
    assert 0.0 <= f <= 1.0


def test_vol_target_inverse_with_asset_vol():
    rng = np.random.default_rng(3)
    quiet = rng.normal(0, 0.002, 500)
    loud = rng.normal(0, 0.020, 500)
    f_quiet = vol_target_fraction(quiet, target_vol=0.20)
    f_loud = vol_target_fraction(loud,  target_vol=0.20)
    assert f_quiet > f_loud
    # quiet asset (very low vol) → fraction should be capped near the
    # function's internal absolute cap (5.0)
    assert f_quiet >= 1.0


def test_optimal_leverage_only_above_one():
    assert optimal_leverage(0.5, leverage_cap=3) == 1
    assert optimal_leverage(1.0, leverage_cap=3) == 1
    assert optimal_leverage(1.4, leverage_cap=3) == 3      # ceil(1.4)+1 capped
    assert optimal_leverage(2.6, leverage_cap=3) == 3
    assert optimal_leverage(0.1, leverage_cap=10) == 1


def test_optimal_position_picks_binding_constraint():
    rng = np.random.default_rng(4)
    # high-drift, moderate-vol path → Kelly likely binds before CVaR/vol-target
    rets = rng.normal(0.0006, 0.004, 500)
    decision = optimal_position(
        rets, cvar_floor=-0.20, cvar_alpha=0.05,
        target_vol=2.0, kelly_safety=0.5, fraction_cap=2.0, leverage_cap=3)
    # the binding constraint should be one of the four expected
    assert decision.binding in ("kelly", "vol_target", "cvar", "cap")
    # leverage is integer ≥ 1
    assert isinstance(decision.leverage, int) and decision.leverage >= 1


def test_optimal_position_zero_for_too_few_samples():
    decision = optimal_position(np.array([0.01, -0.02, 0.0]),
                                  cvar_floor=-0.05, cvar_alpha=0.05)
    assert decision.fraction == 0.0
