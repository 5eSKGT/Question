"""Entry timing efficacy study.

The current cascade-test entry rule fires at the SAME bar as the
screener fires:

    fired at bar t  if  close[t] > anchor (= close[t-1])

This study quantifies how much expected return is gained / lost by
shifting entry by k bars (k = 0, 1, 2, 3, 6, 12):

    entry_price = close[t + k]
    exit machinery unchanged (adaptive chandelier from entry_idx)

If forward-shifted entries (k > 0) carry HIGHER expected return, the
strategy is firing too eagerly — the price tends to revert before
continuing, and waiting captures a better entry.  If they carry LOWER
expected return, the strategy is correctly capturing the Hawkes
self-excitation peak.

References
----------
Hasbrouck (2007), *Empirical Market Microstructure*, §6.4 intra-bar
    timing decomposition of expected return.
Lo-MacKinlay (1990) "Short-horizon reversal" RFS 3(2) — the empirical
    motivation for testing delayed entry.
Easley-Lopez de Prado-O'Hara (2012) — VPIN-conditional optimal entry
    timing is a function of liquidity flow, which we cannot measure
    on 1h klines but whose absence we acknowledge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crypto_trend.backtest.data_loader import real_universe
from crypto_trend.screener.winner_loser import (WinnerLoserScreener,
                                                  lee_mykland_statistic,
                                                  multi_horizon_alignment,
                                                  vol_regime_score, hurst_dfa)
from crypto_trend.strategy.trend_following import (atr, macro_trend_majority,
                                                     volume_z_at,
                                                     StrategyParams,
                                                     hawkes_decay_chandelier_mult)


def _scan_with_delay(df, scr, sp, delay_k: int,
                       warmup=240, exit_horizon=48):
    """Return list of side-aligned realised log returns when entry is
    delayed by ``delay_k`` bars after the screener fires."""
    closes = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    rets = np.zeros_like(closes); rets[1:] = np.diff(np.log(closes))
    a_arr = np.nan_to_num(atr(df, 14).to_numpy(dtype=float), nan=0.0)
    n = len(closes); pnls = []
    last_event_bar = -10**9
    for t in range(warmup, n - exit_horizon - delay_k - 1):
        rw = rets[max(0, t - 256): t + 1]
        if rw.size < scr.lookback + 4: continue
        L = lee_mykland_statistic(rw, window=scr.lookback)
        if not np.isfinite(L) or abs(L) < scr.z_threshold: continue
        side_sign = 1 if L > 0 else -1
        if multi_horizon_alignment(rw, side_sign, scr.horizons) < scr.min_horizons_agree:
            continue
        if vol_regime_score(rw, short_window=scr.lookback,
                              long_window=scr.vol_regime_long_window) > scr.vol_regime_max:
            continue
        if hurst_dfa(rw[-min(rw.size, 256):]) < scr.hurst_floor:
            continue
        if t - last_event_bar < 48: continue
        last_event_bar = t
        side = "long" if side_sign > 0 else "short"
        if sp.tsm_majority_lookbacks and not macro_trend_majority(
                rets[: t + 1], side, lookbacks=sp.tsm_majority_lookbacks,
                min_agree=sp.tsm_majority_min_agree):
            continue
        if sp.volume_z_threshold > -10 and not volume_z_at(
                volume, t, threshold=sp.volume_z_threshold):
            continue

        # Cascade-test continuation gate at the candidate entry bar.
        entry_t = t + delay_k
        if entry_t + 1 >= n: continue
        anchor = closes[t - 1]   # SAME anchor (pre-pick close) regardless of delay
        if side_sign > 0 and not (closes[entry_t] > anchor): continue
        if side_sign < 0 and not (closes[entry_t] < anchor): continue

        entry_px = closes[entry_t]; peak = entry_px; trough = entry_px
        a_t = max(a_arr[entry_t], 1e-6)
        exit_idx = entry_t + exit_horizon
        for u in range(entry_t + 1, min(entry_t + exit_horizon + 1, n)):
            c = closes[u]
            mult = hawkes_decay_chandelier_mult(
                sp.chandelier_mult, u - entry_t,
                tau=sp.chandelier_decay_tau,
                width_boost=sp.chandelier_width_boost)
            if side_sign > 0:
                peak = max(peak, c)
                if c < peak - mult * a_t: exit_idx = u; break
            else:
                trough = min(trough, c)
                if c > trough + mult * a_t: exit_idx = u; break
        exit_idx = min(exit_idx, n - 1)
        pnl = side_sign * float(np.log(closes[exit_idx] / entry_px))
        pnls.append(pnl)
    return pnls


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--delays", type=int, nargs="+",
                    default=[0, 1, 2, 3, 6, 12])
    p.add_argument("--max-symbols", type=int, default=None,
                    help="cap symbols (default: all 313)")
    p.add_argument("--out", type=str,
                    default="reports/entry_timing_diagnostic.json")
    args = p.parse_args()

    pool = real_universe()
    if args.max_symbols:
        pool = dict(list(pool.items())[:args.max_symbols])
    print(f"Universe: {len(pool)} symbols")
    scr = WinnerLoserScreener(min_quote_volume=0.0)
    sp = StrategyParams()

    per_delay: dict[int, list[float]] = {k: [] for k in args.delays}
    for i, (sym, df) in enumerate(pool.items(), start=1):
        for k in args.delays:
            try:
                pnls = _scan_with_delay(df, scr, sp, delay_k=k)
            except Exception:
                continue
            per_delay[k].extend(pnls)
        if i % 50 == 0:
            print(f"  scanned {i}/{len(pool)}: "
                  f"events_at_delay_0={len(per_delay[args.delays[0]])}")

    out_results = {}
    print("\n" + "=" * 70)
    print("Entry timing diagnostic (delay = bars after screener fire)")
    print("=" * 70)
    print(f"  {'delay':>5}  {'n':>6}  {'mean_pnl':>10}  {'std':>8}  "
          f"{'win':>6}  {'sharpe_per_evt':>14}")
    for k in args.delays:
        pnls = np.array(per_delay[k])
        if pnls.size < 50: continue
        mu = float(pnls.mean())
        sd = float(pnls.std(ddof=1))
        win = float((pnls > 0).mean())
        sr = mu / sd if sd > 1e-9 else 0.0
        out_results[k] = {
            "n_events":  int(pnls.size),
            "mean_pnl":  round(mu, 5),
            "std_pnl":   round(sd, 5),
            "win_rate":  round(win, 4),
            "sharpe_per_event": round(sr, 4),
            "kelly_f":   round(mu / (sd * sd), 3) if sd > 1e-9 else 0.0,
        }
        print(f"  {k:>5}  {pnls.size:>6}  {mu:+.5f}  {sd:.5f}  "
              f"{win:.3f}  {sr:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_results, indent=2))

    # Pick the optimal delay (max mean PnL, subject to sample size)
    valid = {k: v for k, v in out_results.items() if v["n_events"] >= 200}
    if valid:
        best_k = max(valid, key=lambda k: valid[k]["mean_pnl"])
        current_k_pnl = valid.get(0, {}).get("mean_pnl", float("nan"))
        best_pnl = valid[best_k]["mean_pnl"]
        delta = best_pnl - current_k_pnl
        print(f"\n  optimal delay = {best_k} bars (mean PnL = {best_pnl:+.5f})")
        print(f"  current (delay=0)     = {current_k_pnl:+.5f}")
        print(f"  Δ = {delta:+.5f}")
        if abs(delta) < 0.001:
            print("  → delay-0 (current) is already near-optimal — "
                  "entry-timing is NOT a major lift source.")
        elif delta > 0.005:
            print(f"  → delayed entry (k={best_k}) carries materially "
                  "higher mean PnL.  Cascade-test entry should consider "
                  "a brief wait window.")
        else:
            print("  → marginal lift; not worth complicating entry logic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
