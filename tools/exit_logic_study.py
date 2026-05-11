"""Exit-logic forensics — MFE/MAE / hold-longer / variable-chandelier.

The user identified a high-confidence suspicion: the chandelier exit
is cutting wins too early OR failing to protect large profits.  This
study quantifies that hypothesis on real Binance USDT-Perp data.

Three measurements
------------------
1. **MFE / MAE distribution** (Sweeney 1988 "Where to Take Profits"
   §2): per trade,
       MFE = max favorable excursion (side-signed log return) DURING
             the trade — i.e. the BEST unrealised PnL ever reached
             before exit.
       MAE = min favorable excursion (= max ADVERSE excursion) ditto.
       gap_realised_to_mfe = MFE - realised_pnl  ≥ 0
   A large mean gap means the strategy is GIVING BACK profits at exit.

2. **Exit-reason distribution**: chandelier / time_stop / cvar_breach
   / mark_out. If chandelier dominates AND the gap is large, the
   chandelier is the proximate cause.

3. **Hold-longer counterfactual**: for each chandelier-exited trade,
   simulate "what if we'd held until time_stop instead?" — report
   the cumulative delta PnL.  This directly answers whether the
   current chandelier is OPTIMAL or sub-optimal under the data's
   actual continuation/reversal balance.

References
----------
Sweeney, R. J. (1988). "Some New Filter Rule Tests."
    J. Financial and Quantitative Analysis 23(3) — MFE/MAE analysis.
Bandy, H. (2014). *Mean Reversion Trading Systems*, §5 — chandelier
    width vs realised distribution.
Kestner, L. (1996). "Studies in Stops" — the academic origin of
    Chandelier exits + the trade-off between tightness and
    profit-preservation.
Aït-Sahalia, Cacho-Diaz & Laeven (2014) JFE 117(3) — Hawkes
    cluster decay timescale, the prior used in adaptive chandelier.
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


def _scan_trades(df, scr, sp, time_stop=48, exit_horizon_max=192):
    """For each filter-passed event return a dict with the full trade
    trajectory measurements (exit_reason, realised, MFE, MAE,
    bars_to_chandelier, full-hold realised)."""
    closes = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    rets = np.zeros_like(closes); rets[1:] = np.diff(np.log(closes))
    a_arr = np.nan_to_num(atr(df, 14).to_numpy(dtype=float), nan=0.0)
    n = len(closes); last_event_bar = -10**9
    out = []
    for t in range(240, n - exit_horizon_max - 1):
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

        entry_px = closes[t]
        peak = entry_px; trough = entry_px
        a_t = max(a_arr[t], 1e-6)
        mfe = 0.0; mae = 0.0
        chand_exit_idx = None
        # Run forward bar-by-bar measuring MFE/MAE AND find the
        # chandelier hit if any.
        for u in range(t + 1, min(t + exit_horizon_max + 1, n)):
            c = closes[u]
            unrealised = side_sign * np.log(c / entry_px)
            if unrealised > mfe: mfe = unrealised
            if unrealised < mae: mae = unrealised
            mult = hawkes_decay_chandelier_mult(
                sp.chandelier_mult, u - t,
                tau=sp.chandelier_decay_tau,
                width_boost=sp.chandelier_width_boost)
            if side_sign > 0:
                peak = max(peak, c)
                if c < peak - mult * a_t and chand_exit_idx is None:
                    chand_exit_idx = u
            else:
                trough = min(trough, c)
                if c > trough + mult * a_t and chand_exit_idx is None:
                    chand_exit_idx = u
            # also stop if we've reached time_stop AND chandelier
            # hasn't fired — record exit reason as time_stop
            if (u - t) >= time_stop and chand_exit_idx is None:
                break
        # Determine exit reason within time_stop window
        if chand_exit_idx is not None and (chand_exit_idx - t) <= time_stop:
            exit_idx = chand_exit_idx
            exit_reason = "chandelier"
        else:
            exit_idx = min(t + time_stop, n - 1)
            exit_reason = "time_stop"
        realised = side_sign * float(np.log(closes[exit_idx] / entry_px))
        # Hold-longer counterfactual — always hold until time_stop
        ts_idx = min(t + time_stop, n - 1)
        ts_realised = side_sign * float(np.log(closes[ts_idx] / entry_px))
        # Hold to long horizon (3x time_stop) counterfactual
        lh_idx = min(t + 3 * time_stop, n - 1)
        lh_realised = side_sign * float(np.log(closes[lh_idx] / entry_px))
        out.append({
            "entry_t":    t,
            "exit_t":     exit_idx,
            "bars_held":  exit_idx - t,
            "exit_reason": exit_reason,
            "realised":   realised,
            "mfe":        mfe,
            "mae":        mae,
            "gap_to_mfe": mfe - realised,
            "ts_realised": ts_realised,
            "lh_realised": lh_realised,
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--out", type=str,
                    default="reports/exit_logic_diagnostic.json")
    args = p.parse_args()

    pool = real_universe()
    if args.max_symbols:
        pool = dict(list(pool.items())[: args.max_symbols])
    print(f"Universe: {len(pool)} symbols")
    scr = WinnerLoserScreener(min_quote_volume=0.0)
    sp = StrategyParams()

    all_trades: list[dict] = []
    for i, (sym, df) in enumerate(pool.items(), start=1):
        try:
            ts = _scan_trades(df, scr, sp)
        except Exception:
            continue
        all_trades.extend(ts)
        if i % 50 == 0:
            print(f"  scanned {i}/{len(pool)}: trades={len(all_trades)}")

    if not all_trades:
        print("no trades"); return 2
    print(f"  total trades = {len(all_trades)}")

    realised = np.array([t["realised"] for t in all_trades])
    mfe      = np.array([t["mfe"] for t in all_trades])
    mae      = np.array([t["mae"] for t in all_trades])
    gap      = np.array([t["gap_to_mfe"] for t in all_trades])
    ts_real  = np.array([t["ts_realised"] for t in all_trades])
    lh_real  = np.array([t["lh_realised"] for t in all_trades])
    bars     = np.array([t["bars_held"] for t in all_trades])

    # Exit-reason distribution
    reasons = [t["exit_reason"] for t in all_trades]
    n_chand = sum(1 for r in reasons if r == "chandelier")
    n_ts    = sum(1 for r in reasons if r == "time_stop")
    pct_chand = n_chand / len(reasons)
    pct_ts    = n_ts / len(reasons)

    # Hold-longer lift
    delta_ts = ts_real - realised   # additional pnl if held to time_stop
    delta_lh = lh_real - realised   # additional pnl if held 3× time_stop

    # Conditional analysis: among chandelier-exited trades, what does
    # holding to time_stop give us?
    chand_mask = np.array([r == "chandelier" for r in reasons])
    if chand_mask.any():
        chand_realised = realised[chand_mask]
        chand_ts       = ts_real[chand_mask]
        chand_delta    = chand_ts - chand_realised
    else:
        chand_realised = chand_ts = chand_delta = np.array([])

    # Among winners (realised > 0), what does the chandelier give back?
    win_mask = realised > 0
    win_gap_pct = (gap[win_mask] / np.maximum(mfe[win_mask], 1e-9)).mean() if win_mask.any() else 0.0

    out = {
        "n_trades":                len(all_trades),
        "realised": {
            "mean":   round(float(realised.mean()), 5),
            "median": round(float(np.median(realised)), 5),
            "std":    round(float(realised.std(ddof=1)), 5),
            "p95":    round(float(np.percentile(realised, 95)), 5),
            "p99":    round(float(np.percentile(realised, 99)), 5),
            "win_rate": round(float((realised > 0).mean()), 4),
        },
        "mfe": {
            "mean":   round(float(mfe.mean()), 5),
            "median": round(float(np.median(mfe)), 5),
            "p95":    round(float(np.percentile(mfe, 95)), 5),
            "p99":    round(float(np.percentile(mfe, 99)), 5),
        },
        "mae": {
            "mean":   round(float(mae.mean()), 5),
            "median": round(float(np.median(mae)), 5),
            "p5":     round(float(np.percentile(mae, 5)), 5),
            "p1":     round(float(np.percentile(mae, 1)), 5),
        },
        "gap_realised_to_mfe": {
            "mean":   round(float(gap.mean()), 5),
            "median": round(float(np.median(gap)), 5),
            "fraction_giveback_among_winners": round(float(win_gap_pct), 4),
        },
        "exit_reasons": {
            "n_chandelier":   n_chand,
            "n_time_stop":    n_ts,
            "pct_chandelier": round(pct_chand, 4),
            "pct_time_stop":  round(pct_ts, 4),
        },
        "hold_longer_counterfactual": {
            "delta_time_stop_mean":      round(float(delta_ts.mean()), 5),
            "delta_time_stop_total":     round(float(delta_ts.sum()), 4),
            "delta_3x_time_stop_mean":   round(float(delta_lh.mean()), 5),
            "delta_3x_time_stop_total":  round(float(delta_lh.sum()), 4),
        },
        "chandelier_giveback": {
            "n":                     int(chand_mask.sum()),
            "realised_mean":         round(float(chand_realised.mean()), 5) if chand_realised.size else None,
            "hold_to_time_stop_mean": round(float(chand_ts.mean()), 5) if chand_ts.size else None,
            "delta_mean":            round(float(chand_delta.mean()), 5) if chand_delta.size else None,
            "delta_total":           round(float(chand_delta.sum()), 4) if chand_delta.size else None,
        },
        "hold_time_bars": {
            "mean":   round(float(bars.mean()), 2),
            "median": round(float(np.median(bars)), 2),
            "p95":    round(float(np.percentile(bars, 95)), 2),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print("Exit-logic forensic diagnostic")
    print("=" * 70)
    print(f"  n trades       : {len(all_trades)}")
    print(f"  realised mean  : {realised.mean():+.5f}   median: {np.median(realised):+.5f}")
    print(f"  MFE mean       : {mfe.mean():+.5f}   median: {np.median(mfe):+.5f}")
    print(f"  MAE mean       : {mae.mean():+.5f}   median: {np.median(mae):+.5f}")
    print(f"  gap realised→MFE mean : {gap.mean():+.5f}")
    print(f"    fraction-giveback among winners (gap/MFE) : "
          f"{win_gap_pct:.3f}")
    print()
    print(f"  Exit reasons:")
    print(f"    chandelier : {pct_chand * 100:.1f}%  ({n_chand} trades)")
    print(f"    time_stop  : {pct_ts * 100:.1f}%  ({n_ts} trades)")
    print()
    print(f"  Hold-longer counterfactual (current → hold-to-time_stop):")
    print(f"    Δ mean      = {delta_ts.mean():+.5f}")
    print(f"    Δ total     = {delta_ts.sum():+.4f}")
    print(f"  Hold-very-long (current → 3× time_stop = 6 days):")
    print(f"    Δ mean      = {delta_lh.mean():+.5f}")
    print(f"    Δ total     = {delta_lh.sum():+.4f}")
    print()
    if chand_mask.any():
        print(f"  Among CHANDELIER-exited trades (n={chand_mask.sum()}):")
        print(f"    realised mean             = {chand_realised.mean():+.5f}")
        print(f"    if-held-to-time_stop mean = {chand_ts.mean():+.5f}")
        print(f"    Δ mean (chand → ts)       = {chand_delta.mean():+.5f}")
        print(f"    Δ total                   = {chand_delta.sum():+.4f}")
    print()
    # Verdict
    if win_gap_pct > 0.40 and delta_ts.mean() > 0.005:
        print("  → STRONG evidence chandelier cuts wins too early. "
              "Loosen width_boost OR widen base chandelier_mult OR adopt "
              "trailing-on-tightening profit-only stop.")
    elif delta_ts.mean() > 0.002:
        print("  → MODERATE evidence chandelier cuts wins. Worth "
              "tuning chandelier_decay_tau / width_boost.")
    elif delta_ts.mean() < -0.002:
        print("  → Chandelier is PROTECTIVE — holding longer would "
              "lose money on average. Current exit is well-tuned.")
    else:
        print("  → NEUTRAL — chandelier roughly optimal at the mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
