import numpy as np
import pytest

from crypto_trend.oos.adaptive import (AdaptiveOOS, AdaptiveStatus,
                                       RecalibrationFailed,
                                       deflated_sharpe_ratio,
                                       probabilistic_sharpe_ratio)
from crypto_trend.strategy.trend_following import StrategyParams


def test_psr_in_unit_interval():
    rng = np.random.default_rng(0)
    rs = rng.normal(1e-4, 1e-3, 500)
    psr = probabilistic_sharpe_ratio(rs)
    assert 0.0 <= psr <= 1.0


def test_dsr_penalises_multiple_trials():
    rng = np.random.default_rng(0)
    rs = rng.normal(1e-4, 1e-3, 500)
    psr = probabilistic_sharpe_ratio(rs)
    dsr = deflated_sharpe_ratio(rs, n_trials=50)
    assert dsr <= psr + 1e-9


def test_adaptive_halts_when_no_grid_passes():
    rng = np.random.default_rng(0)

    def bad_backtest(_p):
        return rng.normal(-1e-4, 1e-3, 200)        # negative SR

    adaptor = AdaptiveOOS(bad_backtest, psr_min=0.99, sharpe_min=2.0,
                          max_attempts=2)
    with pytest.raises(RecalibrationFailed):
        adaptor.step(StrategyParams())


def test_adaptive_passes_when_strategy_is_strong():
    def good_backtest(_p):
        return np.full(500, 0.001)                 # trivially perfect

    adaptor = AdaptiveOOS(good_backtest, psr_min=0.5, sharpe_min=0.1)
    rep = adaptor.step(StrategyParams())
    assert rep.status == AdaptiveStatus.OK


def test_adaptive_pending_when_too_few_samples():
    """No data yet → must NOT halt or recalibrate."""
    def empty_backtest(_p):
        return np.array([])

    adaptor = AdaptiveOOS(empty_backtest, psr_min=0.99, sharpe_min=2.0,
                           max_attempts=1)
    rep = adaptor.step(StrategyParams())
    assert rep.status == AdaptiveStatus.PENDING
