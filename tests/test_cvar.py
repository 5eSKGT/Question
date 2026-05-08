import numpy as np

from crypto_trend.risk.cvar import (cvar_cornish_fisher, cvar_historical,
                                    max_size_under_cvar)


def test_historical_cvar_negative_for_losses():
    rs = np.linspace(-0.1, 0.1, 200)
    es = cvar_historical(rs, alpha=0.05)
    assert es < 0


def test_cf_close_to_historical_for_normal():
    rng = np.random.default_rng(0)
    rs = rng.normal(0, 0.01, 5000)
    h = cvar_historical(rs, 0.05)
    cf = cvar_cornish_fisher(rs, 0.05)
    assert abs(h - cf) < 0.005


def test_max_size_zero_for_short_sample():
    rs = np.array([0.01, -0.02, 0.01])
    assert max_size_under_cvar(rs, -0.05) == 0.0


def test_max_size_capped_by_floor():
    rng = np.random.default_rng(1)
    rs = rng.normal(0, 0.02, 1000)
    f = max_size_under_cvar(rs, cvar_floor=-0.05, alpha=0.05,
                            cap=1.0, leverage_cap=3.0)
    # CVaR at 5% on N(0,0.02) is roughly -0.041; floor -0.05 => f ~ 1.2 -> capped at 1.0
    assert 0.0 < f <= 1.0
