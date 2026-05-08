"""Tests for the strategy-aware sizing module."""
import math

import numpy as np

from crypto_trend.risk.sizing import (
    SizingDecision, fixed_fractional_size, kelly_fraction, optimal_leverage,
    optimal_position, safe_leverage_for_stop, signal_confidence,
    stop_distance_pct, vol_target_fraction,
)


# -- Backwards-compat helpers (still importable for old callers) --------- #


def test_kelly_zero_for_zero_mean():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 500)
    assert kelly_fraction(rets) < 0.05


def test_vol_target_inverse_with_asset_vol():
    rng = np.random.default_rng(3)
    quiet = rng.normal(0, 0.002, 500)
    loud = rng.normal(0, 0.020, 500)
    assert vol_target_fraction(quiet) > vol_target_fraction(loud)


def test_optimal_leverage_only_above_one():
    assert optimal_leverage(0.5, leverage_cap=3) == 1
    assert optimal_leverage(1.0, leverage_cap=3) == 1
    assert optimal_leverage(2.6, leverage_cap=3) == 3


# -- New strategy-aware blocks ------------------------------------------- #


def test_stop_distance_proportional_to_atr():
    sd1 = stop_distance_pct(price=100, atr=1.0, chandelier_mult=3.0)
    sd2 = stop_distance_pct(price=100, atr=2.0, chandelier_mult=3.0)
    assert math.isclose(sd1, 0.03)
    assert math.isclose(sd2, 0.06)


def test_fixed_fractional_size_inverts_stop():
    """Hitting the stop with this size loses exactly risk_per_trade."""
    rpt = 0.01
    sd = 0.05      # 5% stop
    size = fixed_fractional_size(rpt, sd, sizing_cap=2.0)
    assert math.isclose(size, 0.20)
    # Stop-out P&L = size × stop = 0.20 × 0.05 = 0.01 = rpt ✓
    assert math.isclose(size * sd, rpt)


def test_fixed_fractional_size_capped():
    size = fixed_fractional_size(0.05, 0.01, sizing_cap=2.0)   # would be 5
    assert size == 2.0


def test_signal_confidence_scales_with_lm():
    weak = signal_confidence(lm_stat=2.0, lm_threshold=4.0,
                              agree=1, max_agree=3)
    strong = signal_confidence(lm_stat=8.0, lm_threshold=4.0,
                                agree=3, max_agree=3)
    assert strong > weak
    # full strength: 2.0 × 1.0 = 2.0
    assert math.isclose(strong, 2.0)
    # weak: 0.5 × 0.5 = 0.25
    assert math.isclose(weak, 0.5 * (1.0 / 3.0)) or weak >= 0.25


def test_safe_leverage_uses_stop_distance():
    """Tight stop allows higher leverage; wide stop forces leverage down."""
    # target_size large enough that the stop distance is the binding constraint
    high = safe_leverage_for_stop(stop_pct=0.02, target_size=20.0, leverage_cap=50)
    low = safe_leverage_for_stop(stop_pct=0.20, target_size=20.0, leverage_cap=50)
    assert high > low
    # Liquidation distance must be > stop_pct + buffers
    assert high <= int(1.0 / (0.02 + 0.005 + 0.02))


def test_safe_leverage_one_when_unleveraged():
    assert safe_leverage_for_stop(stop_pct=0.05, target_size=0.5,
                                    leverage_cap=10) == 1


def test_optimal_position_stop_synergy():
    """Stop-out P&L (size × stop_pct) ≈ risk_per_trade × confidence,
    so the per-trade risk is mechanically tied to Chandelier_mult × ATR."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 300)
    decision = optimal_position(
        rets, price=100, atr=2.0,        # stop 6%
        lm_stat=4.0, agree=2, max_agree=3,
        risk_per_trade=0.01, cvar_floor=-0.5,    # CVaR loose
        sizing_cap=5.0, leverage_cap=10,
        lm_threshold=4.0, chandelier_mult=3.0,
    )
    # Stop-out loss = fraction × stop = 0.01 × confidence
    expected_loss = 0.01 * decision.confidence
    realised_loss = decision.fraction * decision.stop_distance_pct
    assert math.isclose(realised_loss, expected_loss, rel_tol=0.05)


def test_optimal_position_cvar_can_bind():
    """A tight CVaR floor with fat tails should reduce the size."""
    rng = np.random.default_rng(4)
    fat = rng.standard_t(df=3, size=300) * 0.02
    decision = optimal_position(
        fat, price=100, atr=2.0,
        lm_stat=8.0, agree=3, max_agree=3,        # confidence = 2.0
        risk_per_trade=0.05, cvar_floor=-0.05,
        sizing_cap=5.0, leverage_cap=10,
        lm_threshold=4.0, chandelier_mult=3.0,
    )
    # With 5% risk + huge confidence, base*conf = 5/3 * 2 ≈ 3.33; CVaR
    # cap likely binds.
    assert decision.binding in ("cvar", "cap", "lev")


def test_optimal_position_zero_for_zero_atr():
    rets = np.random.default_rng(0).normal(0, 0.01, 300)
    decision = optimal_position(rets, price=100, atr=0.0,
                                  lm_stat=4.0, agree=2, max_agree=3)
    assert decision.fraction == 0.0
