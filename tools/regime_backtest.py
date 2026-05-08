"""Run AlphaPulse + baselines under different market regimes.

This is the empirical answer to "강한 일방향 상승장에서도 적용되는가?".

We split "bull" into two distinct flavours:

  * ``bull_jump``      — positive drift + upward-skewed jumps (typical
                         of real crypto bull markets — BTC 2017, 2021)
  * ``bull_diffusion`` — positive drift but jumps suppressed (smooth
                         drift, more typical of equity indices)

We also test TWO AlphaPulse configurations side-by-side:

  * **Default**  — kelly_safety=0.5, sizing_cap=1.0 (capital preservation)
  * **Bull-tuned** — kelly_safety=1.0, sizing_cap=2.0,
                     lower z_threshold (more permissive entries)

If "bull-tuned AP" recovers meaningful upside in ``bull_jump``, then the
strategy is fundamentally compatible with bull markets — the question
reduces to a parameter choice the user makes via the GUI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crypto_trend.backtest.data_loader import synthetic_universe       # noqa: E402
from crypto_trend.backtest.simulator import StrategySimulator          # noqa: E402
from crypto_trend.backtest.walk_forward import walk_forward_run        # noqa: E402
from crypto_trend.backtest.metrics import compute_metrics              # noqa: E402
from crypto_trend.backtest.__main__ import (_baseline_buyhold,         # noqa: E402
                                              _baseline_naive_momentum)
from crypto_trend.screener.winner_loser import WinnerLoserScreener     # noqa: E402
from crypto_trend.strategy.trend_following import (StrategyParams,     # noqa: E402
                                                      TrendFollowingStrategy)


REGIMES = ["bull_jump", "bull_diffusion", "bear_jump", "high_vol", "neutral"]


def _ap_default() -> StrategySimulator:
    return StrategySimulator(strategy=TrendFollowingStrategy(StrategyParams()))


def _ap_bull_tuned() -> StrategySimulator:
    """Aggressive AP variant: full Kelly, double cap, permissive screener."""
    params = StrategyParams(
        kelly_safety=1.0,
        sizing_cap=2.0,
        leverage_cap=3.0,
    )
    # Loosened screener threshold: z_threshold=2.5 instead of Gumbel ~3.9
    screener = WinnerLoserScreener(
        z_threshold=2.5,
        min_horizons_agree=1,
        hurst_floor=0.45,
        min_quote_volume=0.0,
    )
    return StrategySimulator(strategy=TrendFollowingStrategy(params),
                              screener=screener)


def _run_regime(regime: str, seeds: int, n_symbols: int, bars: int,
                 train_days: int, test_days: int) -> dict:
    train_bars = train_days * 24
    test_bars = test_days * 24
    rows = {"ap_default": [], "ap_bull": [],
             "buyhold": [], "naive_momentum": []}

    for seed in range(seeds):
        candles = synthetic_universe(n_symbols, bars, seed=seed, regime=regime)

        wf_def = walk_forward_run(candles, train_bars=train_bars,
                                    test_bars=test_bars, simulator=_ap_default())
        wf_bull = walk_forward_run(candles, train_bars=train_bars,
                                     test_bars=test_bars, simulator=_ap_bull_tuned())
        bh = _baseline_buyhold(candles, warmup=train_bars)
        nm = _baseline_naive_momentum(candles, warmup=train_bars)
        bh_m = compute_metrics(bh["bar_returns"], [], [], 1.0)
        nm_m = compute_metrics(nm["bar_returns"], [], [], 1.0)

        for name, m in (("ap_default", wf_def.metrics),
                          ("ap_bull",    wf_bull.metrics),
                          ("buyhold",    bh_m),
                          ("naive_momentum", nm_m)):
            rows[name].append([m.total_return, m.annualized_sharpe,
                                 m.sortino, m.max_drawdown, m.exposure,
                                 m.n_trades])

    summary = {"regime": regime, "seeds": seeds}
    for k, arr in rows.items():
        a = np.asarray(arr)
        summary[k] = {
            "ret_mean":  float(a[:, 0].mean()),
            "ret_med":   float(np.median(a[:, 0])),
            "sortino":   float(a[:, 2].mean()),
            "mdd":       float(a[:, 3].mean()),
            "expo":      float(a[:, 4].mean()),
            "trades":    float(a[:, 5].mean()),
        }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--bars", type=int, default=3000)
    p.add_argument("--n-symbols", type=int, default=20)
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--regimes", nargs="+", default=REGIMES)
    args = p.parse_args()

    print(f"\n{'regime':<18}{'AP-def':>10}{'AP-bull':>10}"
            f"{'BH':>10}{'Naive':>10}"
            f"{'AP-def MDD':>13}{'AP-bull MDD':>13}"
            f"{'AP-def Sortino':>16}{'AP-bull Sortino':>17}")
    print("-" * 110)
    all_results = []
    for r in args.regimes:
        s = _run_regime(r, args.seeds, args.n_symbols, args.bars,
                         args.train_days, args.test_days)
        all_results.append(s)
        print(f"{r:<18}"
                f"{s['ap_default']['ret_mean']:>9.2%} "
                f"{s['ap_bull']['ret_mean']:>9.2%} "
                f"{s['buyhold']['ret_mean']:>9.2%} "
                f"{s['naive_momentum']['ret_mean']:>9.2%} "
                f"{s['ap_default']['mdd']:>12.2%} "
                f"{s['ap_bull']['mdd']:>12.2%} "
                f"{s['ap_default']['sortino']:>15.2f} "
                f"{s['ap_bull']['sortino']:>16.2f}",
                flush=True)
    Path("/tmp/regime_summary.json").write_text(
        json.dumps(all_results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
