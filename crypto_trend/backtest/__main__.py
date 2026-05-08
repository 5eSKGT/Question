"""Command-line entry: ``python -m crypto_trend.backtest --help``.

Examples
--------

Run on real Bitget data (top-15 USDT-perps by 24h turnover, 1-hour
bars, 60 days of history; cached locally on first run):

    python -m crypto_trend.backtest --source bitget --top 15 --bars 1500

Run on synthetic Heston-jump data (no network, deterministic seed):

    python -m crypto_trend.backtest --source synthetic --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_setup import get_logger
from ..screener.winner_loser import WinnerLoserScreener
from ..strategy.trend_following import StrategyParams, TrendFollowingStrategy
from .data_loader import (fetch_bitget_ohlcv, fetch_bitget_universe,
                            synthetic_universe)
from .metrics import compute_metrics
from .simulator import StrategySimulator
from .walk_forward import walk_forward_run

log = get_logger()


# --------------------------------------------------------------------------- #
def _baseline_buyhold(candles: dict[str, pd.DataFrame], warmup: int) -> dict:
    """Equal-weight buy-and-hold benchmark restricted to the test window."""
    common = sorted(set.intersection(*(set(df.index) for df in candles.values())))
    if len(common) <= warmup:
        return {"bar_returns": np.array([]), "trades": [], "exposure": 1.0}
    start_ts = common[warmup]
    rets_per_sym = []
    for sym, df in candles.items():
        c = df.loc[df.index >= start_ts, "close"].to_numpy(dtype=float)
        rets_per_sym.append(np.diff(np.log(c)))
    min_len = min(r.size for r in rets_per_sym)
    stacked = np.stack([r[:min_len] for r in rets_per_sym], axis=0)
    return {"bar_returns": stacked.mean(axis=0), "trades": [], "exposure": 1.0}


def _baseline_naive_momentum(candles: dict[str, pd.DataFrame],
                              warmup: int, lookback: int = 24) -> dict:
    """Simple momentum: long top-quartile by past `lookback`-bar return,
    rebalance daily, no Donchian / CVaR / Hurst gating."""
    common = sorted(set.intersection(*(set(df.index) for df in candles.values())))
    if len(common) <= warmup:
        return {"bar_returns": np.array([]), "trades": [], "exposure": 0.0}
    syms = list(candles.keys())
    rets = []
    for t in range(warmup, len(common)):
        if (t - warmup) % 24 == 0:
            past = {s: float(np.log(candles[s].iloc[t]["close"]
                                      / candles[s].iloc[t - lookback]["close"]))
                     for s in syms}
            ranked = sorted(past.items(), key=lambda kv: kv[1], reverse=True)
            picks = [s for s, _ in ranked[: max(1, len(syms) // 4)]]
        bar_r = []
        for s in picks:
            c0 = candles[s].iloc[t - 1]["close"]
            c1 = candles[s].iloc[t]["close"]
            bar_r.append(np.log(c1 / c0))
        rets.append(float(np.mean(bar_r)) if bar_r else 0.0)
    return {"bar_returns": np.asarray(rets), "trades": [],
            "exposure": 1.0}


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crypto_trend.backtest",
                                  description="AlphaPulse backtester")
    p.add_argument("--source", choices=["bitget", "synthetic"],
                    default="synthetic")
    p.add_argument("--top", type=int, default=15,
                    help="Number of top symbols by 24h turnover (Bitget mode)")
    p.add_argument("--bars", type=int, default=1500,
                    help="Bars per symbol (1h timeframe)")
    p.add_argument("--seed", type=int, default=0,
                    help="RNG seed (synthetic mode)")
    p.add_argument("--n-symbols", type=int, default=40,
                    help="Universe size (synthetic mode)")
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--out", type=str, default=None,
                    help="Optional path to write JSON report")
    args = p.parse_args(argv)

    if args.source == "synthetic":
        log.info(f"generating synthetic universe (n={args.n_symbols}, "
                  f"bars={args.bars}, seed={args.seed})")
        candles = synthetic_universe(args.n_symbols, args.bars, args.seed)
    else:
        log.info(f"pulling top-{args.top} Bitget USDT-perps")
        universe = fetch_bitget_universe(top_k=args.top)
        candles = {}
        for s in universe:
            try:
                candles[s] = fetch_bitget_ohlcv(s, "1h", args.bars)
            except Exception as e:                              # noqa: BLE001
                log.warning(f"skip {s}: {e}")
        if not candles:
            log.error("no usable Bitget data — aborting")
            return 2

    sim = StrategySimulator()
    train_bars = args.train_days * 24
    test_bars = args.test_days * 24
    log.info("running AlphaPulse walk-forward")
    wf = walk_forward_run(candles, train_bars=train_bars,
                           test_bars=test_bars, simulator=sim)

    log.info("running baselines")
    bh = _baseline_buyhold(candles, warmup=train_bars)
    nm = _baseline_naive_momentum(candles, warmup=train_bars)
    bh_metrics = compute_metrics(bh["bar_returns"], [], [], 1.0)
    nm_metrics = compute_metrics(nm["bar_returns"], [], [], 1.0)

    print("\n=== WALK-FORWARD RESULT ===")
    print(json.dumps(wf.metrics.as_row(), indent=2, ensure_ascii=False))
    print("\n=== Buy-and-Hold (equal weight) ===")
    print(json.dumps(bh_metrics.as_row(), indent=2, ensure_ascii=False))
    print("\n=== Naive momentum (top-quartile, daily rebalance) ===")
    print(json.dumps(nm_metrics.as_row(), indent=2, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "alphapulse": wf.metrics.as_row(),
            "buyhold": bh_metrics.as_row(),
            "naive_momentum": nm_metrics.as_row(),
            "n_symbols": len(candles),
            "n_windows": len(wf.per_window),
            "n_trades": len(wf.trades),
        }, indent=2, ensure_ascii=False))
        print(f"\n→ report written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
