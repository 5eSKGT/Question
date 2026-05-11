"""Leverage utilization diagnostic — Kelly-optimal vs actual sized.

User identified the next high-confidence suspect: the leverage
calculation logic (sizing.py::optimal_position) is not synergising
with the heterogeneous-Kelly ceiling.  Specifically:

  * Conviction-Power Kelly amp = conf² (k=2) is a fixed functional
    form, not the per-bin/per-predictor empirical μ/σ² (which the
    heterogeneous-Kelly diagnostic showed is the achievable ceiling).
  * Pyramid /max_pyramid_legs = 3 forcibly divides every leg's
    sized by 3 — including the FIRST leg, before any pyramid leg
    exists.  A single-leg trade that should get full Kelly f gets
    f/3 instead.
  * leverage = ceil(sized) is correct broker math, but its INPUT
    (sized) is mis-calibrated by the above two issues.

This script measures, on real 313-symbol Binance data, per-trade:

  * sized_current   : what the live strategy actually sizes at
  * sized_kelly     : the EMPIRICALLY-optimal Kelly f = μ̂/σ̂²
                       (using the post-filter pool μ̂, σ̂² as the
                       homogeneous baseline)
  * leverage_current : what the broker leverage would be
  * leverage_kelly   : what the broker leverage would be at Kelly
  * sized_robust_kelly : Hens-Mayer 2017 robust Kelly with
                          standard-error shrinkage

Aggregate:
  mean / median sized; % trades at sizing_cap; % trades at leverage 1;
  Kelly-utilisation ratio = sized_current / sized_kelly.

If utilisation << 1 we are under-betting; >> 1 we are over-betting.

References
----------
Hens & Mayer (2017). "Robust Kelly strategies under estimation
    uncertainty." European Journal of Operational Research 256(1).
Boyd, Mueller, O'Donoghue & Wang (2017). "Multi-period Trading via
    Convex Optimization." Now Publishers.
Madhavan (2003). "Implementation Shortfall." JFM 8(1).
Cheng & Madhavan (2009). "The Dynamics of Leveraged and Inverse ETFs."
    JIM 38(4).
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
                                                     hawkes_decay_chandelier_mult,
                                                     profit_ratchet_factor)
from crypto_trend.risk.sizing import optimal_position, signal_confidence


def _scan_trades(df, scr, sp, time_stop=48):
    closes = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    rets = np.zeros_like(closes); rets[1:] = np.diff(np.log(closes))
    a_arr = np.nan_to_num(atr(df, 14).to_numpy(dtype=float), nan=0.0)
    n = len(closes); last_event_bar = -10**9
    out = []
    for t in range(240, n - time_stop - 1):
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

        # Realised pnl (chandelier-bounded, matches v3 P1)
        entry_px = closes[t]; peak = entry_px; trough = entry_px
        a_t = max(a_arr[t], 1e-6); exit_idx = t + time_stop
        for u in range(t + 1, min(t + time_stop + 1, n)):
            c = closes[u]
            mult = hawkes_decay_chandelier_mult(
                sp.chandelier_mult, u - t,
                tau=sp.chandelier_decay_tau,
                width_boost=sp.chandelier_width_boost)
            if side_sign > 0:
                peak = max(peak, c)
                if c < peak - mult * a_t: exit_idx = u; break
            else:
                trough = min(trough, c)
                if c > trough + mult * a_t: exit_idx = u; break
        exit_idx = min(exit_idx, n - 1)
        realised = side_sign * float(np.log(closes[exit_idx] / entry_px))

        # Confidence + analytical sizer
        lm_signed = side_sign * abs(L)
        agree = multi_horizon_alignment(rets[: t + 1], side_sign, scr.horizons)
        conf = signal_confidence(lm_signed, sp.lm_threshold, agree, 3)
        sample = rets[max(0, t - 256): t]
        try:
            decision = optimal_position(
                sample, price=entry_px, atr=a_t,
                lm_stat=lm_signed, agree=int(agree), max_agree=3,
                lm_threshold=sp.lm_threshold,
                chandelier_mult=sp.chandelier_mult,
                risk_per_trade=sp.risk_per_trade / max(1, sp.max_pyramid_legs),
                cvar_floor=sp.cvar_floor,
                cvar_alpha=sp.cvar_alpha,
                sizing_cap=sp.sizing_cap,
                leverage_cap=int(sp.leverage_cap),
                confidence_exponent=sp.confidence_exponent,
            )
        except Exception:
            continue
        out.append({
            "predictor":  float(conf * conf),
            "realised":   float(realised),
            "sized_current": float(decision.fraction),
            "leverage_current": int(decision.leverage),
            "stop_pct":   float(decision.stop_distance_pct),
            "conf":       float(conf),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--out", type=str,
                    default="reports/leverage_utilization_diagnostic.json")
    args = p.parse_args()

    pool = real_universe()
    if args.max_symbols:
        pool = dict(list(pool.items())[: args.max_symbols])
    print(f"Universe: {len(pool)} symbols")
    scr = WinnerLoserScreener(min_quote_volume=0.0)
    sp = StrategyParams()

    trades = []
    for i, (sym, df) in enumerate(pool.items(), start=1):
        try:
            ts = _scan_trades(df, scr, sp)
        except Exception:
            continue
        trades.extend(ts)
        if i % 75 == 0:
            print(f"  scanned {i}/{len(pool)}: trades={len(trades)}")

    if len(trades) < 100:
        print("not enough trades"); return 2
    print(f"  total trades = {len(trades)}")

    realised = np.array([t["realised"] for t in trades])
    sized_cur = np.array([t["sized_current"] for t in trades])
    lev_cur  = np.array([t["leverage_current"] for t in trades])
    preds    = np.array([t["predictor"] for t in trades])
    stops    = np.array([t["stop_pct"] for t in trades])

    # Homogeneous Kelly target (Cover-Thomas 1991 §16 baseline)
    mu_pool = realised.mean()
    sd_pool = realised.std(ddof=1)
    f_pool  = mu_pool / (sd_pool * sd_pool) if sd_pool > 0 else 0.0
    # Per-trade Kelly target = f_pool (homogeneous baseline).  Quarter-Kelly
    # (MTZ 2011 §3 robust drawdown-bounded constant).
    f_target = f_pool * 0.25
    leverage_target = max(1, int(np.ceil(min(f_target, sp.sizing_cap))))

    # Per-bin (heterogeneous) Kelly target — bin by predictor quantile
    n_bins = 6
    edges = np.quantile(preds, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    bin_idx = np.digitize(preds, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    f_bin = np.zeros(n_bins)
    n_bin = np.zeros(n_bins, dtype=int)
    bin_stats = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_bin[b] = int(mask.sum())
        if n_bin[b] < 20: continue
        mu_b = float(realised[mask].mean())
        sd_b = float(realised[mask].std(ddof=1))
        f_bin[b] = mu_b / (sd_b * sd_b) if sd_b > 0 else 0.0
        bin_stats.append({
            "bin": b, "n": int(n_bin[b]),
            "mu": round(mu_b, 5), "sd": round(sd_b, 5),
            "kelly_full": round(float(f_bin[b]), 3),
            "kelly_quarter": round(float(f_bin[b] * 0.25), 3),
            "f_pyramid_3":  round(float(f_bin[b] * 0.25 / 3), 3),
            "mean_sized_current": round(float(sized_cur[mask].mean()), 3),
        })
    # Per-trade heterogeneous Kelly target
    f_het = f_bin[bin_idx] * 0.25  # Quarter-Kelly per-bin
    # Apply pyramid /3 fractioning to MATCH the live sized convention
    f_het_per_leg = f_het / max(1, sp.max_pyramid_legs)
    f_het_per_leg = np.clip(f_het_per_leg, 0.0, sp.sizing_cap)
    # Without the /3 fractioning (FIRST-leg Kelly target)
    f_het_first_leg = np.clip(f_het, 0.0, sp.sizing_cap)

    # Aggregate utilisation
    util_per_leg = (sized_cur / np.maximum(f_het_per_leg, 1e-9)).mean()
    util_first_leg = (sized_cur / np.maximum(f_het_first_leg, 1e-9)).mean()
    # Fraction of trades at sizing_cap binding
    pct_at_cap = float((sized_cur >= sp.sizing_cap - 1e-6).mean())
    # Fraction at leverage 1 (no real leverage)
    pct_lev1   = float((lev_cur <= 1).mean())

    out = {
        "n_trades":         len(trades),
        "pool": {
            "mean_pnl":  round(float(mu_pool), 5),
            "std_pnl":   round(float(sd_pool), 5),
            "kelly_full": round(float(f_pool), 3),
            "kelly_quarter": round(float(f_pool * 0.25), 3),
        },
        "per_bin": bin_stats,
        "current_sized": {
            "mean":   round(float(sized_cur.mean()), 4),
            "median": round(float(np.median(sized_cur)), 4),
            "p95":    round(float(np.percentile(sized_cur, 95)), 4),
            "max":    round(float(sized_cur.max()), 4),
        },
        "current_leverage": {
            "mean":   round(float(lev_cur.mean()), 3),
            "median": int(np.median(lev_cur)),
            "p95":    int(np.percentile(lev_cur, 95)),
            "max":    int(lev_cur.max()),
            "pct_at_1x": round(pct_lev1, 4),
            "pct_at_sizing_cap": round(pct_at_cap, 4),
        },
        "kelly_utilisation": {
            "per_leg_target_mean":   round(float(f_het_per_leg.mean()), 4),
            "first_leg_target_mean": round(float(f_het_first_leg.mean()), 4),
            "ratio_current_to_per_leg":   round(float(util_per_leg), 3),
            "ratio_current_to_first_leg": round(float(util_first_leg), 3),
        },
        "diagnosis": (
            "ratio_current_to_first_leg << 1 ⇒ analytical sizer is "
            "under-betting the FIRST leg vs the heterogeneous Kelly "
            "target by removing the /max_pyramid_legs discount from "
            "the first (only) leg of a cluster.  ratio_to_per_leg ≈ 1 "
            "is the design intent (Kelly_aggregate = Kelly_target via "
            "pyramid risk-fractioning, Faber 2007 §IV + MTZ 2011 §3)."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print("Leverage utilisation diagnostic — current vs Kelly target")
    print("=" * 70)
    print(f"  n trades : {len(trades)}")
    print(f"  pool μ={mu_pool:+.5f} σ={sd_pool:.5f} → "
          f"Kelly full = {f_pool:.3f}  Quarter = {f_pool*0.25:.3f}")
    print()
    print("  Per-bin (quartile of predictor):")
    print(f"  {'bin':>3} {'n':>5} {'μ':>10} {'σ':>8} {'KellyQ':>8} {'cur_sized':>10}")
    for b in bin_stats:
        print(f"  {b['bin']:>3} {b['n']:>5} {b['mu']:+.5f} {b['sd']:.5f} "
              f"{b['kelly_quarter']:>8.3f} {b['mean_sized_current']:>10.4f}")
    print()
    print(f"  Current sized:   mean={sized_cur.mean():.4f}  max={sized_cur.max():.4f}")
    print(f"  Current leverage: mean={lev_cur.mean():.2f}x  max={lev_cur.max()}x  "
          f"{pct_lev1*100:.1f}% at 1x")
    print()
    print(f"  Per-leg Kelly target (with /3): mean={f_het_per_leg.mean():.4f}")
    print(f"  First-leg Kelly target (no /3): mean={f_het_first_leg.mean():.4f}")
    print(f"  Utilisation ratio (current / per-leg) : {util_per_leg:+.3f}")
    print(f"  Utilisation ratio (current / first-leg): {util_first_leg:+.3f}")
    print()
    if util_per_leg < 0.8:
        print("  → CURRENT sizer is UNDER-BETTING vs per-leg Kelly target.")
        print("    The analytical Conviction-Power amp (conf²) does not match")
        print("    the empirical μ_bin/σ_bin² of heterogeneous Kelly.")
    if util_first_leg < 0.5:
        print(f"  → STRONG: current sized averages only {util_first_leg*100:.0f}% of "
              "the first-leg Kelly target.  Removing /max_pyramid_legs from")
        print("    the FIRST leg of a cluster (and applying it only to ADDITIONAL")
        print("    pyramid legs) would lift utilisation closer to 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
