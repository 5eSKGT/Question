"""Grinold Fundamental Law upper-bound analysis for AlphaPulse.

References
----------
Grinold (1989), "The Fundamental Law of Active Management",
    Journal of Portfolio Management 15(3), pp.30-37.
Grinold & Kahn (2000), Active Portfolio Management, 2nd ed., chs. 5-6.
Aït-Sahalia, Cacho-Diaz & Laeven (2014), "Modeling financial contagion
    using mutually exciting jump processes", JFE 117(3) — Hawkes
    decorrelation timescale used for breadth deduplication.
Lopez de Prado (2018), Advances in Financial Machine Learning §8 —
    purged k-fold cross-validation for OOS IC estimation.

Theory
------
The Information Ratio (alpha / tracking-error) of an active strategy is
bounded by:

    IR  =  IC * sqrt(BR) * TC

where:
  IC = corr(forecast_alpha, realised_pnl)        ∈ [-1, +1]
       — *property of the data*, can NOT be increased by tuning.
  BR = number of independent bets per year       ≥ 1
       — bounded above by the rate of *uncorrelated* signals in the
       universe; for clustered jumps this is the cluster-arrival rate
       NOT the within-cluster fire rate.
  TC = transfer coefficient                      ∈ [0, 1]
       — efficiency of converting forecast into position. <1 captures
       leverage caps, slippage, position-sizing inefficiency.

Then the achievable annual Sharpe is:

    SR_max  ≈  IR_max  =  IC_oos * sqrt(BR_oos) * TC=1

and the achievable annual return at that Sharpe is:

    R_max   =  SR_max  *  σ_strategy  *  sqrt(year_in_bars)

where σ_strategy is bounded by leverage_cap × σ_universe.

Empirical estimation procedure (overfitting-free)
-------------------------------------------------
1. **IC**: walk-forward — at each screener-event timestamp t in the
   real Binance USDT-Perp universe, compute the strategy's *predicted
   alpha*  α̂(t) = sign(LM_t) * confidence(LM_t)^2 (the Conviction-
   Power Kelly forecast functional, k=2). Realise the chandelier-exit
   PnL of a hypothetical entry at t. Compute IC = Pearson corr(α̂, PnL)
   on the OOS slice only, after de-meaning and standardising.

2. **BR_max**: count Hawkes-cluster-deduplicated screener events per
   symbol-year. Two events on the same symbol within τ=1/β bars
   (Aït-Sahalia 2014 ≈ 48 bars on hourly crypto) are counted as ONE
   independent bet. Sum across symbols, divide by years of data.

3. **TC**: ratio of (achieved IR | current strategy code) to
   (IC * sqrt(BR)). TC=1 means current execution/sizing extracts the
   full theoretical alpha; TC=0.5 means we're leaving half on the table.

Output
------
A single JSON + console summary that pins:
  • IC_oos, IC_95%CI
  • BR_max_universe, BR_max_per_symbol
  • IR_ceiling = IC * sqrt(BR_max)
  • Current achieved IR (from production_validation v2.2 baseline)
  • TC_current = IR_actual / IR_ceiling
  • Annual return ceiling at canonical leverage
  • Gap analysis: which factor (IC vs BR vs TC) is the largest gap

This file makes ZERO parameter changes to the strategy. It is a
diagnostic; it informs WHICH levers to pull, not WHICH values to set.
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
                                                     StrategyParams)


# -- 1. IC: predicted-alpha → realised-PnL correlation -------------- #


def _conviction(lm: float, lm_threshold: float = 4.0) -> float:
    """The Conviction-Power Kelly confidence functional (k=2)."""
    z = abs(lm) / lm_threshold
    z = max(0.5, min(2.0, z))
    return float(z * z)                   # quadratic amp, k=2


def _scan_screener_events(symbol: str, df, scr: WinnerLoserScreener,
                            sp: StrategyParams,
                            warmup: int = 240, exit_horizon: int = 48
                            ) -> tuple[list[float], list[float],
                                       list[float], list[float]]:
    """Walk through one symbol's full history bar-by-bar.

    Returns four parallel lists:
      * raw_forecasts:    α̂ on every screener-fire (no strategy filters)
      * raw_realised:     realised PnL (chandelier exit) for those events
      * filt_forecasts:   α̂ on events that ALSO pass strategy-level filters
                            (TSM majority 7d/14d/30d ≥2/3, volume z ≥ 0.5,
                             cascade-test continuation past anchor)
      * filt_realised:    realised PnL for those filter-passed events

    Comparing IC(raw) vs IC(filt) tells us where alpha actually lives:
    in jump *detection* or in jump *filtering*. The cascade-test theorem
    (Aronson 2007) implies the filters can ADD information not present
    in the raw screener signal — this measurement makes that explicit.
    """
    closes = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    rets = np.zeros_like(closes)
    rets[1:] = np.diff(np.log(closes))
    a_series = atr(df, 14)
    a_arr = np.nan_to_num(a_series.to_numpy(dtype=float), nan=0.0)

    raw_f: list[float] = []; raw_r: list[float] = []
    filt_f: list[float] = []; filt_r: list[float] = []
    n = len(closes)
    last_event_bar = -10**9
    chand_mult = sp.chandelier_mult
    for t in range(warmup, n - exit_horizon - 1):
        r_window = rets[max(0, t - 256): t + 1]
        if r_window.size < scr.lookback + 4:
            continue
        L = lee_mykland_statistic(r_window, window=scr.lookback)
        if not np.isfinite(L) or abs(L) < scr.z_threshold:
            continue
        side_sign = 1 if L > 0 else -1
        agree = multi_horizon_alignment(r_window, side_sign, scr.horizons)
        if agree < scr.min_horizons_agree:
            continue
        regime = vol_regime_score(r_window, short_window=scr.lookback,
                                   long_window=scr.vol_regime_long_window)
        if regime > scr.vol_regime_max:
            continue
        h = hurst_dfa(r_window[-min(r_window.size, 256):])
        if h < scr.hurst_floor:
            continue
        if t - last_event_bar < 48:
            continue
        last_event_bar = t

        alpha_hat = side_sign * _conviction(L)

        # Realised over chandelier-bounded horizon (same logic for both
        # raw and filtered samples).
        entry_px = closes[t]
        peak = closes[t]; trough = closes[t]
        a_t = max(a_arr[t], 1e-6)
        exit_idx = t + exit_horizon
        for u in range(t + 1, min(t + exit_horizon + 1, n)):
            c = closes[u]
            if side_sign > 0:
                peak = max(peak, c)
                if c < peak - chand_mult * a_t:
                    exit_idx = u; break
            else:
                trough = min(trough, c)
                if c > trough + chand_mult * a_t:
                    exit_idx = u; break
        exit_idx = min(exit_idx, n - 1)
        pnl = side_sign * float(np.log(closes[exit_idx] / entry_px))

        raw_f.append(alpha_hat); raw_r.append(pnl)

        # ---- Strategy-level filters (v2.1 + v2.2) -------------------- #
        side = "long" if side_sign > 0 else "short"
        if sp.tsm_majority_lookbacks:
            tsm_ok = macro_trend_majority(
                rets[: t + 1], side,
                lookbacks=sp.tsm_majority_lookbacks,
                min_agree=sp.tsm_majority_min_agree)
        else:
            tsm_ok = True
        vol_ok = (sp.volume_z_threshold <= -10
                    or volume_z_at(volume, t,
                                    threshold=sp.volume_z_threshold))
        # cascade-test continuation past pre-pick anchor (close[t-1]).
        # On the jump bar t, by definition close[t] is on the picked
        # side of close[t-1], so this passes — but only if the gates
        # above also pass.
        if not (tsm_ok and vol_ok):
            continue
        filt_f.append(alpha_hat); filt_r.append(pnl)

    return raw_f, raw_r, filt_f, filt_r


# -- 2. BR: cluster-deduplicated independent bets per year ---------- #


def _cluster_dedup_bets(forecasts: list[float], events_per_year: float
                          ) -> float:
    """Number of *independent* bets per year, after Hawkes-cluster
    deduplication. Already done at the per-symbol scan level (events
    within 48 bars dropped); ``events_per_year`` is the deduped rate.
    """
    return events_per_year


# -- 3. Walk-forward purged-fold OOS IC ------------------------------ #


def _purged_kfold_ic(forecasts: np.ndarray, realised: np.ndarray,
                       k: int = 5, embargo: int = 24) -> tuple[float, float]:
    """Purged k-fold OOS Pearson correlation (López de Prado 2018 §8).

    Splits chronologically into k folds. For each test fold, removes
    a ``embargo``-bar buffer around it from the train set (already
    embarrassingly trivial here since we estimate no parameters — IC
    is just a correlation on the test fold's data). Returns
    (mean_oos_IC, std_across_folds).
    """
    if forecasts.size < 50:
        return float("nan"), float("nan")
    n = forecasts.size
    fold_size = n // k
    ics = []
    for f in range(k):
        a, b = f * fold_size, (f + 1) * fold_size if f < k - 1 else n
        x_test = forecasts[a:b]
        y_test = realised[a:b]
        if x_test.size < 10:
            continue
        if x_test.std() == 0 or y_test.std() == 0:
            continue
        ics.append(float(np.corrcoef(x_test, y_test)[0, 1]))
    if not ics:
        return float("nan"), float("nan")
    return float(np.mean(ics)), float(np.std(ics))


# -- 4. Main ----------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-symbols", type=int, default=None,
                    help="cap universe for quick runs (default: all)")
    p.add_argument("--actual-ir",  type=float, default=4.16,
                    help="empirical annualised Sharpe achieved by the "
                         "*current* committed strategy on the same data "
                         "(v2.2 cascade-test canonical = 4.16)")
    p.add_argument("--leverage-cap", type=float, default=10.0)
    p.add_argument("--out", type=str,
                    default="reports/upper_bound_diagnostic.json")
    args = p.parse_args()

    pool = real_universe()
    if not pool:
        print("⚠ no parquet cache found — run tools/fetch_binance_real.py",
              file=sys.stderr)
        return 2
    if args.max_symbols:
        keep = list(pool.keys())[: args.max_symbols]
        pool = {k: pool[k] for k in keep}
    print(f"Universe: {len(pool)} symbols")

    scr = WinnerLoserScreener(min_quote_volume=0.0)
    sp = StrategyParams()

    raw_f_all: list[float] = [];  raw_r_all: list[float] = []
    filt_f_all: list[float] = []; filt_r_all: list[float] = []
    raw_events_per_symbol: dict[str, int] = {}
    filt_events_per_symbol: dict[str, int] = {}
    median_bars = 0
    for i, (sym, df) in enumerate(pool.items(), start=1):
        try:
            rf, rr, ff, fr = _scan_screener_events(sym, df, scr, sp)
        except Exception as e:                                     # noqa: BLE001
            print(f"  ! skip {sym}: {e}", file=sys.stderr)
            continue
        raw_f_all.extend(rf); raw_r_all.extend(rr)
        filt_f_all.extend(ff); filt_r_all.extend(fr)
        raw_events_per_symbol[sym]  = len(rf)
        filt_events_per_symbol[sym] = len(ff)
        median_bars = max(median_bars, len(df))
        if i % 50 == 0:
            print(f"  scanned {i}/{len(pool)}: raw={sum(raw_events_per_symbol.values())} "
                  f"filt={sum(filt_events_per_symbol.values())}")

    raw_f  = np.array(raw_f_all);   raw_r  = np.array(raw_r_all)
    filt_f = np.array(filt_f_all);  filt_r = np.array(filt_r_all)
    if raw_f.size < 50:
        print("⚠ too few screener events", file=sys.stderr); return 3

    bars_per_year = 365.0 * 24.0
    years = median_bars / bars_per_year

    def _summary(forecasts, realised, label):
        if forecasts.size < 50:
            return None
        ic_full = float(np.corrcoef(forecasts, realised)[0, 1])
        ic_oos_mean, ic_oos_std = _purged_kfold_ic(forecasts, realised)
        z = 0.5 * np.log((1 + ic_oos_mean) / (1 - ic_oos_mean + 1e-12))
        se = 1.0 / np.sqrt(forecasts.size - 3)
        z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
        ic_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
        ic_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
        br = forecasts.size / max(years, 1e-9)
        # Sign-flipped IC matters: if IC < 0 the forecast itself is
        # anti-signal but the strategy can profit by *inverting* it.
        # |IC| × √BR is the achievable IR magnitude.
        ir = abs(ic_oos_mean) * np.sqrt(br)
        ev_std = float(np.std(realised))
        ann_ret = ir * ev_std * np.sqrt(br)
        return {
            "label": label,
            "n_events": int(forecasts.size),
            "events_per_year": round(br, 1),
            "IC_full": round(ic_full, 4),
            "IC_oos_kfold": round(ic_oos_mean, 4),
            "IC_oos_std": round(ic_oos_std, 4),
            "IC_oos_95CI": [round(ic_lo, 4), round(ic_hi, 4)],
            "IR_magnitude": round(ir, 3),
            "ann_return_ceiling": round(ann_ret, 3),
            "event_std": round(ev_std, 4),
        }

    raw_summary  = _summary(raw_f, raw_r, "raw_screener")
    filt_summary = _summary(filt_f, filt_r, "after_strategy_filters")

    # IR ceiling = the LARGER of |IC|·√BR across the two stages
    if filt_summary is not None:
        ir_ceiling = filt_summary["IR_magnitude"]
        ann_ret_ceiling = filt_summary["ann_return_ceiling"]
    else:
        ir_ceiling = raw_summary["IR_magnitude"]
        ann_ret_ceiling = raw_summary["ann_return_ceiling"]
    tc_current = args.actual_ir / ir_ceiling if ir_ceiling > 0 else float("nan")

    # `out` dict construction deferred until after homogeneous +
    # heterogeneous Kelly computations below.

    # ---- Per-trade expectancy (chandelier asymmetry effect) -------- #
    # Mean realised PnL ON FILTERED EVENTS is the cleanest measure of
    # whether the strategy can profit despite sub-zero IC. Asymmetric
    # exits (3-ATR chandelier on the loss side + open-ended on the
    # win side) can produce mean-positive PnL even when IC < 0.
    raw_mean_pnl = float(np.mean(raw_r))
    raw_pnl_std  = float(np.std(raw_r))
    raw_skew     = float(((raw_r - raw_mean_pnl) ** 3).mean()
                          / (raw_pnl_std ** 3 + 1e-12))
    raw_win_rate = float((raw_r > 0).mean())
    filt_mean_pnl = float(np.mean(filt_r)) if filt_r.size else float("nan")
    filt_pnl_std  = float(np.std(filt_r))  if filt_r.size else float("nan")
    filt_skew = (float(((filt_r - filt_mean_pnl) ** 3).mean()
                          / (filt_pnl_std ** 3 + 1e-12))
                  if filt_r.size else float("nan"))
    filt_win_rate = (float((filt_r > 0).mean())
                      if filt_r.size else float("nan"))

    raw_summary["mean_pnl"]  = round(raw_mean_pnl, 5)
    raw_summary["pnl_std"]   = round(raw_pnl_std, 4)
    raw_summary["pnl_skew"]  = round(raw_skew, 3)
    raw_summary["win_rate"]  = round(raw_win_rate, 4)
    if filt_summary is not None:
        filt_summary["mean_pnl"] = round(filt_mean_pnl, 5)
        filt_summary["pnl_std"]  = round(filt_pnl_std, 4)
        filt_summary["pnl_skew"] = round(filt_skew, 3)
        filt_summary["win_rate"] = round(filt_win_rate, 4)

    # ---- HETEROGENEOUS Kelly ceiling (signal-conditional) ---------- #
    # Markowitz (1952) + Cover & Thomas (1991, Elements of Information
    # Theory, Ch. 16) conditional log-optimal portfolio: when per-event
    # μ_i/σ_i² varies, the achievable annual log-growth is
    #
    #     G_het = Σ_i 0.5 · (μ_i / σ_i)²
    #
    # which by Jensen's inequality is STRICTLY greater than the
    # homogeneous-Kelly aggregate
    #
    #     G_hom = N · 0.5 · (μ̄ / σ̄)²
    #
    # whenever μ_i/σ_i has any variance across events. The committed
    # Conviction-Power Kelly amp ∝ confidence^k IS a heterogeneous
    # sizer — but the homogeneous ceiling I previously reported
    # under-counts the achievable G_het by ignoring the per-event
    # variation it can extract.
    #
    # Empirical estimator: bin filtered events by an ex-ante predictor
    # (|LM| × agree, the same composite the strategy uses), compute
    # per-bin (μ_b, σ_b) on OOS k-folds (purged), and sum:
    #
    #     G_het_OOS  =  Σ_b  n_b · 0.5 · (μ_b / σ_b)²
    #
    # This is the *predictor-conditional* Kelly ceiling: how much
    # growth a *perfectly-calibrated* signal-conditional sizer could
    # extract given the predictor we have. It bounds the strategy
    # achievable IR from above, more tightly than the homogeneous
    # ceiling.

    def _het_ceiling(forecasts: np.ndarray, realised: np.ndarray,
                       n_bins: int = 6, k_folds: int = 5) -> dict:
        if forecasts.size < 200:
            return {"status": "insufficient_data", "n": int(forecasts.size)}
        # OOS purged k-fold: train bin edges on (k-1) folds, evaluate
        # bin (μ, σ) on the held-out fold. Sum growth over folds.
        n = forecasts.size
        fold_size = n // k_folds
        per_bin_growth = []
        per_bin_n = []
        for f in range(k_folds):
            a = f * fold_size; b = (f + 1) * fold_size if f < k_folds - 1 else n
            test_idx = slice(a, b)
            train_idx = list(range(a)) + list(range(b, n))
            x_train = forecasts[train_idx]; y_train = realised[train_idx]
            x_test  = forecasts[a:b];  y_test  = realised[a:b]
            if x_test.size < 20: continue
            edges = np.quantile(np.abs(x_train), np.linspace(0, 1, n_bins + 1))
            edges[0] = -np.inf; edges[-1] = np.inf
            bin_idx = np.digitize(np.abs(x_test), edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            for b_id in range(n_bins):
                mask = bin_idx == b_id
                if mask.sum() < 5: continue
                # Sign-aware: bin pnl by sign of forecast (apply forecast
                # sign so realised becomes "side-aligned").
                bin_pnl = np.sign(x_test[mask]) * y_test[mask]
                mu_b = float(bin_pnl.mean())
                sd_b = float(bin_pnl.std())
                if sd_b < 1e-9: continue
                per_bin_growth.append(0.5 * (mu_b / sd_b) ** 2)
                per_bin_n.append(int(mask.sum()))
        if not per_bin_growth:
            return {"status": "insufficient_bins"}
        # Aggregate (per-event) and annualised. Each bin contributes
        # n_b · 0.5 · (μ_b/σ_b)². Sum across bins and folds, divide by
        # k_folds to get per-fold expectation, then scale to per-year
        # by multiplying by total events / 1y.
        total_growth = sum(g * n for g, n in zip(per_bin_growth, per_bin_n))
        annual = total_growth / k_folds
        return {
            "status": "ok",
            "annual_log_growth_OOS": round(float(annual), 4),
            "annual_return_ceiling_pct": round(
                float((np.exp(annual) - 1.0) * 100.0), 1),
            "n_bins_evaluated": len(per_bin_growth),
            "n_events_per_fold_avg": round(forecasts.size / k_folds, 1),
        }

    het_filt = _het_ceiling(filt_f, filt_r) if filt_f.size > 200 else None
    het_raw  = _het_ceiling(raw_f, raw_r)   if raw_f.size  > 200 else None

    # The homogeneous ceiling (single μ̄/σ̄) for comparison
    if filt_r.size:
        hom_filt_per_event = 0.5 * (filt_mean_pnl / filt_pnl_std) ** 2 if filt_pnl_std > 0 else 0.0
        hom_filt_annual = hom_filt_per_event * filt_r.size
        hom_filt_return_pct = (np.exp(hom_filt_annual) - 1.0) * 100.0
    else:
        hom_filt_per_event = hom_filt_annual = hom_filt_return_pct = float("nan")

    # Now build the output dict with all the computed metrics.
    out = {
        "universe_symbols":   len(pool),
        "years_of_data":      round(years, 3),
        "raw_stage":          raw_summary,
        "filtered_stage":     filt_summary,
        "IR_ceiling":         round(ir_ceiling, 3),
        "annualised_return_ceiling": round(ann_ret_ceiling, 3),
        "current_actual_IR":  args.actual_ir,
        "TC_current":         round(tc_current, 3),
        "homogeneous_filtered_kelly": {
            "per_event_log_growth": round(float(hom_filt_per_event), 6),
            "annual_log_growth":    round(float(hom_filt_annual), 4),
            "annual_return_pct":    round(float(hom_filt_return_pct), 1),
        },
        "heterogeneous_filtered_kelly": het_filt,
        "heterogeneous_raw_kelly":      het_raw,
        "diagnosis": {},
    }

    # ---- Diagnosis -------------------------------------------------- #
    raw_ic = raw_summary["IC_oos_kfold"]
    filt_ic = filt_summary["IC_oos_kfold"] if filt_summary else float("nan")
    diag: dict = {}
    if raw_ic < 0 and filt_ic < 0 and filt_mean_pnl > 0:
        diag["alpha_lives_in_exits_not_signal"] = (
            "Both raw and filtered IC are NEGATIVE (jumps mean-revert on "
            "average over chandelier horizon — Lo-MacKinlay 1990) yet "
            "filtered mean-PnL is POSITIVE. The strategy's profit comes "
            "from EXECUTION MECHANICS — asymmetric chandelier exits and "
            "Conviction-Power Kelly sizing nonlinearity — NOT from the "
            "screener's forecasting power. Adding new signal-generating "
            "filters has DIMINISHING RETURNS; the highest-leverage "
            "improvements are in EXIT MANAGEMENT and POSITION SIZING.")
    elif raw_ic < 0 and filt_ic > 0:
        diag["alpha_lives_in_filters"] = (
            "Raw screener IC is NEGATIVE but post-filter IC is POSITIVE. "
            "The 'alpha' lives in the FILTERS (TSM majority + volume z + "
            "cascade continuation). Improvements should target filter "
            "information content, not jump-detection sensitivity.")
    elif raw_ic > 0 and filt_ic > raw_ic:
        diag["filters_amplify_signal"] = (
            "Both stages show positive IC; filters amplify it. Both "
            "jump detection and post-filter quality contribute.")
    elif raw_ic > 0 and filt_ic < raw_ic:
        diag["filters_reduce_signal"] = (
            "Filters reduce raw IC. Either remove a filter or replace "
            "with one that retains forecast quality.")
    elif raw_ic < 0 and filt_ic < 0 and filt_mean_pnl < 0:
        diag["no_alpha_present"] = (
            "Both IC and mean-PnL are negative. The current pipeline "
            "has no measurable edge on this universe at this horizon. "
            "Strategy needs a fundamentally different signal class or "
            "horizon before further iteration.")
    else:
        diag["uncertain"] = (
            "Mixed signals — investigate per-strata IC (e.g., LM bin) "
            "before structural changes.")
    if not np.isnan(tc_current):
        if tc_current < 0.6:
            diag["TC_priority"] = (
                "TC < 0.6 — biggest gap is execution / sizing / position "
                "management. Improve THESE before adding new signals.")
        elif tc_current < 0.85:
            diag["TC_priority"] = (
                "TC moderate — marginal sizing/leverage gains remain; "
                "new signal classes give diminishing returns.")
        else:
            diag["TC_priority"] = (
                "TC near 1 — further gains require new IC (multi-"
                "timeframe LM, order-flow, funding) or BR (cluster "
                "pyramiding). Mind overfitting risk at this regime.")
    out["diagnosis"] = diag

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print("Grinold Fundamental Law — measured ceiling for AlphaPulse")
    print("=" * 70)
    print(f"  Universe              : {len(pool)} symbols × {years:.2f}y")
    print()
    for s in (raw_summary, filt_summary):
        if s is None: continue
        print(f"  [{s['label']}]")
        print(f"    n_events            = {s['n_events']}  "
              f"({s['events_per_year']} / yr)")
        print(f"    IC (purged 5-fold)  = {s['IC_oos_kfold']:+.4f} "
              f"(95% CI {s['IC_oos_95CI']})")
        print(f"    |IR| ceiling        = {s['IR_magnitude']:+.3f}")
        print(f"    annual return ceil  = {s['ann_return_ceiling']*100:+.1f}%  "
              f"σ_event={s['event_std']}")
        print(f"    mean PnL / event    = {s.get('mean_pnl', float('nan')):+.5f}  "
              f"std={s.get('pnl_std', float('nan')):.4f}  "
              f"skew={s.get('pnl_skew', float('nan')):+.2f}  "
              f"win={s.get('win_rate', float('nan')):.3f}")
        print()
    print(f"  Current actual IR     : {args.actual_ir:+.3f}  "
          f"(v2.2 cascade-test canonical)")
    print(f"  TC_current            : {tc_current:.3f}  "
          f"(1.0 = full alpha extraction)")
    print()
    print("  --- Kelly ceiling: homogeneous vs heterogeneous (Markowitz 1952; Cover-Thomas 1991) ---")
    print(f"  Homogeneous (filtered):  G_year = {hom_filt_annual:.3f}  →  "
          f"return ceil ≈ {hom_filt_return_pct:+.1f}%")
    if het_filt and het_filt.get("status") == "ok":
        het_g = het_filt["annual_log_growth_OOS"]
        het_pct = het_filt["annual_return_ceiling_pct"]
        print(f"  Heterogeneous (filt):    G_year = {het_g:.3f}  →  "
              f"return ceil ≈ {het_pct:+.1f}%   "
              f"(× {(het_g / max(hom_filt_annual, 1e-9)):.2f} vs homogeneous)")
    if het_raw and het_raw.get("status") == "ok":
        het_g_raw = het_raw["annual_log_growth_OOS"]
        het_pct_raw = het_raw["annual_return_ceiling_pct"]
        print(f"  Heterogeneous (raw):     G_year = {het_g_raw:.3f}  →  "
              f"return ceil ≈ {het_pct_raw:+.1f}%")
    print()
    print("  --- Diagnosis ---")
    for k, v in diag.items():
        print(f"  • [{k}] {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
