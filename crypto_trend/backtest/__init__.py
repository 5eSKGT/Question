"""End-to-end backtester for the AlphaPulse strategy.

Components:
  data_loader  : pull and cache OHLCV (real Bitget via ccxt, or synthetic)
  simulator    : event-driven replay with realistic costs (taker fee + slippage)
  metrics      : Sharpe / Sortino / MDD / Calmar / win rate / profit factor
  walk_forward : rolling train/test orchestrator (walk-forward analysis)
  report       : tabular + markdown summary

CLI:    python -m crypto_trend.backtest --help
"""
from .metrics import compute_metrics, BacktestMetrics
from .simulator import StrategySimulator, Trade
from .walk_forward import walk_forward_run, WalkForwardResult

__all__ = [
    "compute_metrics", "BacktestMetrics",
    "StrategySimulator", "Trade",
    "walk_forward_run", "WalkForwardResult",
]
