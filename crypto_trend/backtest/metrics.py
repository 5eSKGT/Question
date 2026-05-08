"""Performance metrics for the backtest report."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class BacktestMetrics:
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_return: float
    annualized_sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    avg_trade_duration_hours: float
    bar_pnl_skew: float
    bar_pnl_kurtosis: float
    exposure: float                  # fraction of bars with an open position

    def as_row(self) -> dict:
        return {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in asdict(self).items()}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0, 0.0) else default


def equity_curve_from_bar_returns(bar_returns: np.ndarray) -> np.ndarray:
    """Cumulative product equity curve starting at 1.0."""
    if bar_returns.size == 0:
        return np.array([1.0])
    return np.exp(np.cumsum(bar_returns))


def annualized_sharpe(bar_returns: np.ndarray,
                       periods_per_year: float = 365 * 24) -> float:
    if bar_returns.size < 8:
        return 0.0
    mu = bar_returns.mean()
    sd = bar_returns.std(ddof=1)
    return float(_safe_div(mu, sd) * np.sqrt(periods_per_year))


def sortino_ratio(bar_returns: np.ndarray,
                   periods_per_year: float = 365 * 24) -> float:
    if bar_returns.size < 8:
        return 0.0
    mu = bar_returns.mean()
    downside = bar_returns[bar_returns < 0]
    if downside.size == 0:
        return float("inf") if mu > 0 else 0.0
    ds_std = np.sqrt(np.mean(downside * downside))
    return float(_safe_div(mu, ds_std) * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics(bar_returns: np.ndarray,
                     trade_pnls: list[float],
                     trade_durations: list[float],
                     exposure: float,
                     periods_per_year: float = 365 * 24) -> BacktestMetrics:
    eq = equity_curve_from_bar_returns(bar_returns)
    total_ret = float(eq[-1] - 1.0) if eq.size else 0.0
    sr = annualized_sharpe(bar_returns, periods_per_year)
    so = sortino_ratio(bar_returns, periods_per_year)
    mdd = max_drawdown(eq)
    n = len(trade_pnls)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / n if n else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_factor = (_safe_div(sum(wins), -sum(losses), default=float("inf"))
                     if losses else float("inf") if wins else 0.0)
    avg_dur = float(np.mean(trade_durations)) if trade_durations else 0.0

    if bar_returns.size > 8:
        from scipy import stats
        skew = float(stats.skew(bar_returns, bias=False))
        kurt = float(stats.kurtosis(bar_returns, bias=False))
    else:
        skew = kurt = 0.0

    return BacktestMetrics(
        n_trades=n,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=float(profit_factor),
        total_return=total_ret,
        annualized_sharpe=sr,
        sortino=so,
        max_drawdown=mdd,
        calmar=_safe_div(sr, abs(mdd), default=0.0),
        avg_trade_duration_hours=avg_dur,
        bar_pnl_skew=skew,
        bar_pnl_kurtosis=kurt,
        exposure=exposure,
    )
