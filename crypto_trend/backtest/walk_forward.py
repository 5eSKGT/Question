"""Walk-forward orchestrator that mirrors the live OOS adaptor.

Splits the global candle history into rolling (train, test) windows. At
each window boundary the same ``AdaptiveOOS`` machinery the live engine
uses is invoked: if the test-window result fails the PSR/Sharpe gate
the strategy parameters are recalibrated for the next window; if the
adaptor cannot find passing parameters, the backtest halts at that
window — exactly what would have happened in production.

The recalibration history is recorded so the user can see when (and
why) the adaptor would have intervened during real operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..oos.adaptive import (AdaptiveOOS, AdaptiveStatus, MIN_SAMPLES,
                              RecalibrationFailed)
from ..strategy.trend_following import StrategyParams
from .metrics import BacktestMetrics, compute_metrics
from .simulator import StrategySimulator, Trade


@dataclass
class WalkForwardResult:
    metrics: BacktestMetrics
    per_window: list[BacktestMetrics] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    bar_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    telemetry: dict = field(default_factory=dict)
    oos_history: list[dict] = field(default_factory=list)   # per-window OOS verdicts
    halted_at_window: int | None = None


def walk_forward_run(
    candles: dict[str, pd.DataFrame],
    train_bars: int = 24 * 30,           # 30 days
    test_bars: int = 24 * 7,             # 7 days
    simulator: StrategySimulator | None = None,
    oos_warmup_windows: int = 3,         # mirror live engine warmup behaviour
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
    telemetry_acc = {
        "universe_size": 0, "rescreen_count": 0, "picks_total": 0,
        "picks_zero_cycles": 0,
    }

    oos_history: list[dict] = []
    halted_at: int | None = None

    start = train_bars
    window_idx = 0
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
        # accumulate screener telemetry from this window
        t_win = out.get("telemetry", {})
        if t_win:
            telemetry_acc["universe_size"] = t_win.get(
                "universe_size", telemetry_acc["universe_size"])
            telemetry_acc["rescreen_count"] += t_win.get("rescreen_count", 0)
            telemetry_acc["picks_total"] += t_win.get("picks_total", 0)
            telemetry_acc["picks_zero_cycles"] += t_win.get("picks_zero_cycles", 0)

        # ---- OOS adaptive recalibration (mirrors live engine) ----------- #
        # The same machinery the production loop uses every cycle.  We
        # build a one-shot backtest function that returns this window's
        # bar returns under a candidate parameter set; the adaptor calls
        # it for the current params and (on failure) for a small grid.
        if rets.size >= MIN_SAMPLES:
            captured_sim = sim                                   # closure capture
            sliced_now = sliced

            def _recalibrate(p: StrategyParams) -> np.ndarray:
                trial = StrategySimulator(
                    strategy=type(captured_sim.strategy)(p),
                    screener=captured_sim.screener,
                    taker_fee=captured_sim.taker_fee,
                    slippage_bps=captured_sim.slippage_bps,
                    rescreen_every=captured_sim.rescreen_every,
                )
                trial_out = trial.run(sliced_now, warmup_bars=start)
                return trial_out["bar_returns"]

            adaptor = AdaptiveOOS(_recalibrate)
            try:
                report = adaptor.step(captured_sim.strategy.p)
                oos_history.append({
                    "window": window_idx,
                    "status": report.status.value,
                    "psr": round(report.psr, 3),
                    "sharpe": round(report.sharpe, 3),
                    "attempts": report.attempts,
                })
                if report.status == AdaptiveStatus.RECALIBRATED:
                    captured_sim.strategy.p = report.params
            except RecalibrationFailed as e:
                # Warmup guard: live engine never halts during the first
                # few cycles either — applying the same here lets the
                # strategy accumulate evidence before the OOS gate
                # binds.
                if window_idx < oos_warmup_windows:
                    oos_history.append({
                        "window": window_idx,
                        "status": "warmup",
                        "message": f"deferred halt during warmup: {e}",
                    })
                else:
                    oos_history.append({
                        "window": window_idx,
                        "status": "halted",
                        "message": str(e),
                    })
                    halted_at = window_idx
                    break

        start += step
        window_idx += 1

    telemetry_acc["picks_mean"] = (
        telemetry_acc["picks_total"] / telemetry_acc["rescreen_count"]
        if telemetry_acc["rescreen_count"] else 0.0)
    telemetry_acc["oos_recalibrations"] = sum(
        1 for h in oos_history if h.get("status") == "recalibrated")
    telemetry_acc["oos_halted_at_window"] = halted_at

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
        telemetry=telemetry_acc,
        oos_history=oos_history,
        halted_at_window=halted_at,
    )
