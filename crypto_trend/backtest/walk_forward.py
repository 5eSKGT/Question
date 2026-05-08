"""Walk-forward orchestrator.

Splits the global candle history into rolling (train, test) windows. The
*train* window is currently used only as warmup for the indicators (we
do not retune parameters between windows in this backtest — the
production engine handles that via the live OOS adaptor). Each test
window is simulated independently and metrics are aggregated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import BacktestMetrics, compute_metrics
from .simulator import StrategySimulator, Trade


@dataclass
class WalkForwardResult:
    metrics: BacktestMetrics
    per_window: list[BacktestMetrics] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    bar_returns: np.ndarray = field(default_factory=lambda: np.array([]))


def walk_forward_run(
    candles: dict[str, pd.DataFrame],
    train_bars: int = 24 * 30,           # 30 days
    test_bars: int = 24 * 7,             # 7 days
    simulator: StrategySimulator | None = None,
) -> WalkForwardResult:
    sim = simulator or StrategySimulator()
    if not candles:
        return WalkForwardResult(metrics=compute_metrics(np.array([]), [], [], 0.0))

    common_idx = sorted(set.intersection(*(set(df.index) for df in candles.values())))
    n = len(common_idx)
    step = test_bars
    bar_returns_all = []
    trades_all: list[Trade] = []
    per_window: list[BacktestMetrics] = []
    bars_with_position_total = 0
    bars_total = 0

    start = train_bars
    while start + test_bars <= n:
        end = start + test_bars
        slice_idx = common_idx[: end]
        sliced = {s: df.loc[df.index <= slice_idx[-1]] for s, df in candles.items()}
        out = sim.run(sliced, warmup_bars=start)
        rets = out["bar_returns"]
        if rets.size:
            bar_returns_all.append(rets)
            bars_total += rets.size
            bars_with_position_total += int(out["exposure"] * rets.size)
        trades_window = [t for t in out["trades"]
                          if t.entry_ts >= common_idx[start - 1]]
        trades_all.extend(trades_window)
        per_window.append(compute_metrics(
            rets, [t.pnl for t in trades_window],
            [float(t.bars_held) for t in trades_window],
            out["exposure"]))
        start += step

    flat_rets = (np.concatenate(bar_returns_all)
                  if bar_returns_all else np.array([]))
    overall_exposure = (bars_with_position_total / bars_total
                         if bars_total > 0 else 0.0)
    overall = compute_metrics(
        flat_rets,
        [t.pnl for t in trades_all],
        [float(t.bars_held) for t in trades_all],
        overall_exposure)

    return WalkForwardResult(
        metrics=overall,
        per_window=per_window,
        trades=trades_all,
        bar_returns=flat_rets,
    )
