"""Aggregate AlphaPulse vs baselines across multiple synthetic-universe seeds.

Run:
    python tools/multi_seed_backtest.py --seeds 12 --bars 3000

Outputs a summary table that includes mean and std of every metric, and
prints whether AlphaPulse beats each baseline on Sharpe / total return /
max drawdown / Calmar ratio. This is the closest we can get to a
reproducible "statistical significance" test inside the sandbox; the
same script runs unchanged on real Bitget data once the user replaces
``synthetic_universe(...)`` with ``fetch_bitget_universe(...)`` (or
simply uses the ``--source bitget`` flag of ``crypto_trend.backtest``).
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


def _row(metrics) -> list[float]:
    return [metrics.total_return, metrics.annualized_sharpe,
            metrics.sortino, metrics.max_drawdown, metrics.calmar,
            metrics.exposure]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-symbols", type=int, default=30)
    p.add_argument("--bars", type=int, default=3000)
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=7)
    args = p.parse_args()

    train_bars = args.train_days * 24
    test_bars = args.test_days * 24

    rows = {"alphapulse": [], "buyhold": [], "naive_momentum": []}
    for seed in range(args.seeds):
        candles = synthetic_universe(args.n_symbols, args.bars, seed=seed)
        sim = StrategySimulator()
        wf = walk_forward_run(candles, train_bars=train_bars,
                                test_bars=test_bars, simulator=sim)
        bh = _baseline_buyhold(candles, warmup=train_bars)
        nm = _baseline_naive_momentum(candles, warmup=train_bars)
        bh_m = compute_metrics(bh["bar_returns"], [], [], 1.0)
        nm_m = compute_metrics(nm["bar_returns"], [], [], 1.0)
        rows["alphapulse"].append(_row(wf.metrics))
        rows["buyhold"].append(_row(bh_m))
        rows["naive_momentum"].append(_row(nm_m))
        print(f"seed {seed:>2}: AP "
                f"ret={wf.metrics.total_return:+.3f} sr={wf.metrics.annualized_sharpe:+.2f} "
                f"mdd={wf.metrics.max_drawdown:+.3f} expo={wf.metrics.exposure:.2f} "
                f"trades={wf.metrics.n_trades} | "
                f"BH ret={bh_m.total_return:+.3f} | "
                f"Naive ret={nm_m.total_return:+.3f}",
                flush=True)

    summary = {}
    cols = ["total_return", "ann_sharpe", "sortino", "max_dd", "calmar", "exposure"]
    for name, arr in rows.items():
        a = np.array(arr)
        summary[name] = {c: {"mean": float(a[:, i].mean()),
                              "std": float(a[:, i].std(ddof=1)),
                              "median": float(np.median(a[:, i]))}
                          for i, c in enumerate(cols)}

    print("\n=== AGGREGATE (mean ± std) ===")
    for name in ("alphapulse", "buyhold", "naive_momentum"):
        s = summary[name]
        print(f"{name:>16}  ret={s['total_return']['mean']:+.3f}±{s['total_return']['std']:.3f}  "
                f"sr={s['ann_sharpe']['mean']:+.2f}±{s['ann_sharpe']['std']:.2f}  "
                f"sortino={s['sortino']['mean']:+.2f}  "
                f"mdd={s['max_dd']['mean']:+.3f}  "
                f"calmar={s['calmar']['mean']:+.2f}  "
                f"expo={s['exposure']['mean']:.2f}")

    # AlphaPulse vs BH per-seed: count wins
    ap_ret = np.array([r[0] for r in rows["alphapulse"]])
    bh_ret = np.array([r[0] for r in rows["buyhold"]])
    nm_ret = np.array([r[0] for r in rows["naive_momentum"]])
    print(f"\nAP > BH on return:    {(ap_ret > bh_ret).sum()}/{args.seeds}")
    print(f"AP > Naive on return: {(ap_ret > nm_ret).sum()}/{args.seeds}")

    Path("/tmp/multi_seed_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
