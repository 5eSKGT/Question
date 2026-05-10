"""Event-driven strategy simulator with realistic costs.

The previous version invoked ``strategy.generate_signals`` on a *slice*
of every symbol's history at every bar, which is O(N · T²) and made
even small backtests intractable. This rewrite precomputes all
strategy indicators once per symbol per backtest in O(N · T), then
iterates through bars in O(N · T) for the entry/exit machinery — total
complexity O(N · T) instead of O(N · T²).

Modelled costs:
  * taker_fee   — Bitget USDT-perp default 6 bps per fill
  * slippage    — additional 1 bp on entry and exit (configurable)
  * one position per symbol — matches the production engine
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..screener.winner_loser import WinnerLoserScreener
from ..strategy.trend_following import (StrategyParams, TrendFollowingStrategy,
                                          atr, donchian, yang_zhang_vol)


@dataclass
class Trade:
    symbol: str
    side: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    size_fraction: float
    pnl: float
    bars_held: int
    exit_reason: str


@dataclass
class _SymbolPrecomp:
    """Precomputed indicator arrays for one symbol."""
    closes: np.ndarray
    log_rets: np.ndarray
    atr: np.ndarray
    yz: np.ndarray
    donchian_hi: np.ndarray         # rolling max of high over breakout_n bars
    donchian_lo: np.ndarray


def _precompute(df: pd.DataFrame, p: StrategyParams) -> _SymbolPrecomp:
    closes = df["close"].to_numpy(dtype=float)
    log_p = np.log(closes)
    rets = np.diff(log_p, prepend=log_p[0])
    a = atr(df, p.atr_n).to_numpy(dtype=float)
    yz = yang_zhang_vol(df, p.yz_n).to_numpy(dtype=float)
    hi, lo = donchian(df, p.breakout_n)
    return _SymbolPrecomp(
        closes=closes,
        log_rets=rets,
        atr=np.nan_to_num(a, nan=0.0),
        yz=np.nan_to_num(yz, nan=0.0),
        donchian_hi=np.nan_to_num(hi.to_numpy(dtype=float), nan=np.inf),
        donchian_lo=np.nan_to_num(lo.to_numpy(dtype=float), nan=-np.inf),
    )


@dataclass
class StrategySimulator:
    strategy: TrendFollowingStrategy = field(default_factory=TrendFollowingStrategy)
    screener: WinnerLoserScreener | None = None
    taker_fee: float = 6e-4
    slippage_bps: float = 1.0
    bars_per_year: float = 365 * 24
    # v3 B: persistent KellyCalibrator across walk-forward windows.
    # walk_forward_run.py calls ``sim.run()`` once per OOS test window
    # (default 7 days) — without persistence the calibrator would
    # only see the few trades inside one window and never warm up.
    # Storing it on the simulator instance keeps the rolling 300-trade
    # history alive across windows, so by the time we reach later
    # OOS windows the calibrator has trained on prior-OOS realisations
    # only (strict purging — past trades inform future sizing).
    kelly_calibrator: object | None = None
    # rescreen_every=1 mirrors the live engine, which calls the screener
    # once per bar (= once per cycle with 1h timeframe).
    rescreen_every: int = 1
    # Pick stickiness — once the screener picks a symbol, it stays
    # "active" for ``pick_ttl`` bars. v2.2 sets the default to 48 hrs
    # (2 days) — Aït-Sahalia, Cacho-Diaz & Laeven (2014), *Modeling
    # financial contagion using mutually exciting jump processes*, JFE
    # 117(3) measure the Hawkes self-excitation decay timescale at
    # *days*, not hours, for crypto-like markets. A 24-bar TTL was
    # truncating the cluster window prematurely; 48 bars matches the
    # empirical decay constant 1/β.
    pick_ttl: int = 48
    # Funding-rate provider — None matches live when not available.
    funding_rate_provider: object = None

    # ------------------------------------------------------------------ #
    def run(self, candles: dict[str, pd.DataFrame],
            warmup_bars: int = 256,
            screener_top_n: int | None = None,
            should_stop=None) -> dict:
        """Replay the strategy bar-by-bar.

        ``should_stop`` is an optional zero-arg callable: if it returns
        True at any cooperative checkpoint inside the bar loop, the
        simulation halts and returns whatever results it has so far.
        This is what makes the GUI "Stop" button responsive — without
        it the simulator is opaque to cancellation.
        """
        if not candles:
            return {"bar_returns": np.array([]), "trades": [], "exposure": 0.0}

        idx = sorted(set.intersection(*(set(df.index) for df in candles.values())))
        n_bars = len(idx)
        if n_bars <= warmup_bars + 4:
            return {"bar_returns": np.array([]), "trades": [], "exposure": 0.0}

        symbols = list(candles.keys())
        # Re-index every DataFrame to the common index for safe positional access
        candles_aligned = {s: candles[s].reindex(idx) for s in symbols}

        # ---- pre-compute indicators once per symbol -------------------- #
        p = self.strategy.p
        pre: dict[str, _SymbolPrecomp] = {
            s: _precompute(candles_aligned[s], p) for s in symbols
        }

        # Default screener mirrors the live one — same defaults as
        # ``crypto_trend/screener/winner_loser.py::WinnerLoserScreener``,
        # except we permit a configurable top_n for backtest speed.
        # Default screener mirrors the live one — same defaults as
        # ``crypto_trend/screener/winner_loser.py::WinnerLoserScreener``,
        # except we drop the liquidity floor here because the candle data
        # passed to the simulator has already been pre-filtered by the
        # data loader (live mirror).
        screener = self.screener or WinnerLoserScreener(
            min_quote_volume=0.0,
            top_n=screener_top_n or 10,
            additional_scales=p.multi_scale_lm_factors,
        )

        # v3: multi-leg pyramiding support — positions[sym] is now a
        # list of leg dicts. Each leg carries its own
        # (side, entry_idx, entry_px, size, peak, trough). Total
        # exposure is sum(leg["size"]); chandelier checked per leg.
        positions: dict[str, list[dict]] = {}
        trades: list[Trade] = []
        bar_pnl = np.zeros(n_bars, dtype=float)
        bars_with_position = 0
        # v3 A1: active_picks now carries (side, expiry, anchor, scale).
        # ``scale`` is the screener-detection timeframe multiplier (1 =
        # native 1h; 4 = 4h aggregated). The strategy uses scale to
        # widen the chandelier on entries from longer-scale picks
        # (Bandy 2014 §5: ATR-based stops scale with √timeframe).
        active_picks: dict[str, tuple[str, int, float, int]] = {}
        # v3 tracker — bar at which the current pick episode for `sym`
        # last received a *fresh* (not just TTL-extended) screener fire.
        # Used as the "new cluster pulse" trigger for pyramiding.
        last_fresh_pick_bar: dict[str, int] = {}
        # Track which fresh-pick bars have already been consumed by an
        # entry / pyramid action — so we don't re-fire on the same
        # pulse across consecutive bars.
        consumed_pulse_bar: dict[str, int] = {}
        # v3 A2: leader-jump history per side, used by the cross-asset
        # Hawkes (AS-CD-L 2014) confidence boost on follower picks.
        # Maps side ("long" / "short") → list of (bar_idx, leader_sym)
        # within the rolling decay window.
        leader_jump_history: dict[str, list[tuple[int, str]]] = {"long": [], "short": []}

        # v3 B: OOS-calibrated rolling per-bin Kelly sizer. Lazily
        # created on the FIRST run() call (so the calibrator persists
        # across walk-forward windows — see ``self.kelly_calibrator``
        # field docstring). Disabled if
        # ``StrategyParams.kelly_calibrator_enabled`` is False.
        if p.kelly_calibrator_enabled and self.kelly_calibrator is None:
            from ..risk.kelly_calibrator import KellyCalibrator
            self.kelly_calibrator = KellyCalibrator(
                n_bins=p.kelly_calibrator_bins,
                lookback_trades=p.kelly_calibrator_lookback,
                min_per_bin=p.kelly_calibrator_min_per_bin,
                sizing_cap=p.sizing_cap,
            )
        kelly_calibrator = self.kelly_calibrator if p.kelly_calibrator_enabled else None
        # Map (sym, entry_idx) → predictor used at entry, so we can
        # feed the realised pnl back to the calibrator at exit.
        entry_predictor: dict[tuple[str, int], float] = {}
        rescreen_history: list[tuple[int, int, int]] = []

        cost_per_fill = self.taker_fee + self.slippage_bps * 1e-4

        for t in range(warmup_bars, n_bars):
            # Cooperative cancellation — checked once per bar so a
            # "Stop" button click breaks within ≈ a millisecond.
            # Sampling every 50 bars to keep the predicate cheap; the
            # outer rescreen loop also checks once per rescreen_every.
            if should_stop is not None and (t & 0x3F) == 0 and should_stop():
                break
            ts = idx[t]

            # ---- rescreen ---------------------------------------------- #
            if (t - warmup_bars) % self.rescreen_every == 0:
                # Build a slice provider that doesn't materialise new frames
                def _ohlcv_provider(s, end=t):
                    return candles_aligned[s].iloc[: end + 1]
                results = screener.run(
                    symbols,
                    ohlcv_provider=_ohlcv_provider,
                    quote_volume_provider=lambda s: 1e12,
                    funding_rate_provider=self.funding_rate_provider,
                )
                # Refresh fresh picks (extend expiry). v3: every
                # screener-fire is a "fresh pulse" for pyramiding —
                # we record last_fresh_pick_bar but keep the original
                # anchor.  Also: A2 — record leader (BTC/ETH/...) jumps
                # so follower picks within the decay window get a
                # confidence boost at sizing time.
                for r in results:
                    last_fresh_pick_bar[r.symbol] = t
                    if r.symbol in p.leader_symbols:
                        leader_jump_history[r.side].append((t, r.symbol))
                    pick_scale = getattr(r, "scale", 1)
                    if r.symbol in active_picks:
                        side_old, _expiry_old, anchor_old, scale_old = active_picks[r.symbol]
                        if side_old == r.side:
                            # Keep the larger scale (slower cluster wins).
                            new_scale = max(scale_old, pick_scale)
                            active_picks[r.symbol] = (
                                r.side, t + self.pick_ttl, anchor_old, new_scale)
                            continue
                    pre_pick_close = float(pre[r.symbol].closes[max(0, t - 1)])
                    active_picks[r.symbol] = (
                        r.side, t + self.pick_ttl, pre_pick_close, pick_scale)
                # Drop expired picks
                active_picks = {s: v for s, v in active_picks.items()
                                  if v[1] > t}
                # Trim leader-jump history outside the decay window so
                # the lookup below is O(window) not O(history).
                cutoff = t - p.leader_anchor_decay_bars
                for side in ("long", "short"):
                    leader_jump_history[side] = [
                        (b, s) for (b, s) in leader_jump_history[side]
                        if b >= cutoff]
                rescreen_history.append((
                    (t - warmup_bars) // self.rescreen_every,
                    len(symbols), len(results),
                ))

            # ---- iterate symbols (cheap: only O(1) per symbol now) ----- #
            for sym in symbols:
                pc = pre[sym]
                close = pc.closes[t]
                if close <= 0 or np.isnan(close):
                    continue
                legs = positions.get(sym, [])
                a_i = pc.atr[t]
                yz_i = pc.yz[t]
                band = p.band_mult * yz_i * close

                if legs:
                    from ..strategy.trend_following import (
                        hawkes_decay_chandelier_mult)
                    prev_close = pc.closes[t - 1]
                    surviving: list[dict] = []
                    for leg in legs:
                        side_sign = 1.0 if leg["side"] == "long" else -1.0
                        if prev_close > 0 and not np.isnan(prev_close):
                            bar_pnl[t] += side_sign * (np.log(close / prev_close)
                                                         * leg["size"])
                        # ---- per-leg exit decision (adaptive +
                        # scale-aware chandelier — A1 / Bandy 2014 §5).
                        # 4h-scale picks get a chandelier widened by
                        # √4 = 2× because the natural jump magnitude
                        # at 4h is √scale × 1h-jump magnitude (volatility
                        # scales with √time under no-arbitrage).
                        bars_in_pos = t - leg["entry_idx"]
                        leg_scale = leg.get("scale", 1)
                        scale_mult = float(np.sqrt(max(1, leg_scale)))
                        adapt_mult = hawkes_decay_chandelier_mult(
                            p.chandelier_mult * scale_mult, bars_in_pos,
                            tau=p.chandelier_decay_tau,
                            width_boost=p.chandelier_width_boost)
                        if leg["side"] == "long":
                            leg["peak"] = max(leg["peak"], close)
                            chand = leg["peak"] - adapt_mult * a_i
                            chand_hit = close < chand
                        else:
                            leg["trough"] = min(leg["trough"], close)
                            chand = leg["trough"] + adapt_mult * a_i
                            chand_hit = close > chand
                        time_stop = bars_in_pos >= p.time_stop_bars
                        win = pc.log_rets[max(0, t - 64): t] * side_sign
                        cvar_breach = (win.size > 0
                                        and np.quantile(win, p.cvar_alpha) < p.cvar_floor)
                        if chand_hit or time_stop or cvar_breach:
                            reason = ("chandelier" if chand_hit
                                       else "time_stop" if time_stop
                                       else "cvar_breach")
                            exit_px = close * (1.0 - self.slippage_bps * 1e-4 * side_sign)
                            bar_pnl[t] -= leg["size"] * cost_per_fill
                            pnl = (np.log(exit_px / leg["entry_px"]) * side_sign
                                    * leg["size"] - 2.0 * leg["size"] * cost_per_fill)
                            trades.append(Trade(
                                symbol=sym, side=leg["side"],
                                entry_ts=idx[leg["entry_idx"]], exit_ts=ts,
                                entry_price=float(leg["entry_px"]),
                                exit_price=float(exit_px),
                                size_fraction=float(leg["size"]),
                                pnl=float(pnl),
                                bars_held=bars_in_pos,
                                exit_reason=reason,
                            ))
                            # Feed the calibrator with the closed trade.
                            if kelly_calibrator is not None:
                                key = (sym, leg["entry_idx"])
                                pred = entry_predictor.pop(key, None)
                                if pred is not None:
                                    # Side-aligned realised log return
                                    # (PnL on a unit-size position).
                                    realised_log = side_sign * float(
                                        np.log(exit_px / leg["entry_px"]))
                                    kelly_calibrator.add_trade(
                                        pred, realised_log)
                        else:
                            surviving.append(leg)
                    if surviving:
                        positions[sym] = surviving
                    else:
                        positions.pop(sym, None)

                # ---- entry / pyramid decision ----------------------- #
                # Pyramid permitted iff (Faber 2007 §IV winner-only rule):
                #   - existing legs are all on the picked side
                #   - aggregate unrealised PnL > 0 (only scale into winners)
                #   - leg count < max_pyramid_legs
                #   - total exposure has remaining capacity (sizing_cap)
                #   - the current bar carries a FRESH screener pulse not
                #     yet consumed (so we don't pyramid every bar)
                # If a position exists but pyramid conditions fail → skip.
                # If no position exists → normal cascade-test entry.
                legs_now = positions.get(sym, [])
                if sym not in active_picks:
                    continue
                want, _expiry, anchor, pick_scale = active_picks[sym]
                # v2.2 cascade-test entry: continuation past the
                # pre-pick anchor is the entry trigger. The screener
                # already exhausted the false-positive budget at the
                # universe level; re-detecting the jump via Donchian
                # is statistical double-counting (Aronson 2007 §IV).
                cont_long = close > anchor
                cont_dn = close < anchor
                # Legacy gate kept for ablation (screener_continuation_entry=False).
                hi_prev = pc.donchian_hi[t - 1]
                lo_prev = pc.donchian_lo[t - 1]
                broke_up = close > hi_prev
                broke_dn = close < lo_prev
                if p.screener_continuation_entry:
                    trig_long, trig_short = cont_long, cont_dn
                else:
                    trig_long, trig_short = broke_up, broke_dn
                inside_band = band == 0 or abs(close - pc.closes[t - 1]) <= band

                # AlphaPulse v2: macro TSM + volume confirmation. See
                # strategy/trend_following.py for the rationale; this
                # block is the simulator-side mirror so backtest = live.
                from ..strategy.trend_following import (
                    macro_trend_aligned, macro_trend_majority, volume_z_at)
                if p.tsm_majority_lookbacks:
                    tsm_long_ok = macro_trend_majority(
                        pc.log_rets[: t], "long",
                        lookbacks=p.tsm_majority_lookbacks,
                        min_agree=p.tsm_majority_min_agree)
                    tsm_short_ok = macro_trend_majority(
                        pc.log_rets[: t], "short",
                        lookbacks=p.tsm_majority_lookbacks,
                        min_agree=p.tsm_majority_min_agree)
                else:
                    tsm_long_ok = (p.tsm_lookback_bars <= 0
                                    or macro_trend_aligned(pc.log_rets[: t], "long",
                                                             p.tsm_lookback_bars))
                    tsm_short_ok = (p.tsm_lookback_bars <= 0
                                     or macro_trend_aligned(pc.log_rets[: t], "short",
                                                              p.tsm_lookback_bars))
                vol_arr = candles_aligned[sym]["volume"].to_numpy(dtype=float)
                vol_ok = (p.volume_z_threshold <= -10
                            or volume_z_at(vol_arr, t,
                                            threshold=p.volume_z_threshold))

                fire_long = (want == "long" and trig_long
                              and tsm_long_ok and vol_ok)
                fire_short = (want == "short" and trig_short
                               and tsm_short_ok and vol_ok)
                if not (fire_long or fire_short):
                    continue

                # ---- pyramid gate (Faber 2007 §IV winner-only rule) -- #
                if legs_now:
                    # Existing legs must agree with the picked side.
                    if any(L["side"] != ("long" if fire_long else "short")
                            for L in legs_now):
                        continue
                    # Aggregate unrealised PnL across legs must be > 0.
                    if p.pyramid_in_profit_required:
                        agg_pnl = sum(
                            (1.0 if L["side"] == "long" else -1.0)
                            * np.log(close / L["entry_px"]) * L["size"]
                            for L in legs_now)
                        if agg_pnl <= 0:
                            continue
                    # Leg count cap.
                    if len(legs_now) >= p.max_pyramid_legs:
                        continue
                    # Capacity within sizing_cap.
                    used = sum(L["size"] for L in legs_now)
                    if used >= p.sizing_cap:
                        continue
                    # Fresh screener pulse this bar, not yet consumed.
                    pulse = last_fresh_pick_bar.get(sym)
                    if pulse != t:
                        continue
                    if consumed_pulse_bar.get(sym) == t:
                        continue

                # Strategy-aware sizing — fixed-fractional risk against the
                # Chandelier stop, multiplied by LM/agree confidence,
                # capped by CVaR floor and safe-leverage rule.
                from ..risk.sizing import optimal_position
                from ..screener.winner_loser import (
                    lee_mykland_statistic, multi_horizon_alignment)
                sample = pc.log_rets[max(0, t - 256): t]
                pre_rets = pc.log_rets[: t]
                lm = lee_mykland_statistic(pre_rets, window=24)
                lm_signed = (1 if fire_long else -1) * abs(lm)
                agree = multi_horizon_alignment(
                    pre_rets, 1 if fire_long else -1, horizons=(1, 4, 24))
                # ---- A2: cross-asset Hawkes leader-anchor boost --- #
                # If a same-side leader (BTC/ETH/...) jump fired in the
                # last leader_anchor_decay_bars (AS-CD-L 2014 cross-
                # excitation window τ = 1/β), boost |L| so the
                # downstream Conviction-Power Kelly amp scales the
                # position up. The follower symbol itself can be a
                # leader in disguise (e.g. ETH following BTC); only
                # apply the boost if the picked symbol is NOT the
                # same as the leader that fired.
                want_side = "long" if fire_long else "short"
                lead_present = any(s != sym for (_b, s) in
                                    leader_jump_history[want_side])
                if lead_present:
                    boost = p.leader_anchor_boost
                    lm_signed = lm_signed * boost
                # Pyramid risk fractioning: each leg gets risk_per_trade
                # split across the maximum permitted legs so the
                # aggregate cluster risk respects partial-Kelly
                # (MacLean-Thorp-Ziemba 2011 §3).
                per_leg_risk = (p.risk_per_trade / max(1, p.max_pyramid_legs))
                decision = optimal_position(
                    sample,
                    price=close, atr=a_i,
                    lm_stat=float(lm_signed), agree=int(agree), max_agree=3,
                    lm_threshold=p.lm_threshold,
                    chandelier_mult=p.chandelier_mult,
                    risk_per_trade=per_leg_risk,
                    cvar_floor=p.cvar_floor,
                    cvar_alpha=p.cvar_alpha,
                    sizing_cap=p.sizing_cap,
                    leverage_cap=int(p.leverage_cap),
                    confidence_exponent=p.confidence_exponent,
                )
                size = decision.fraction
                # Predictor used for the calibrator: sign-aware
                # confidence² (matches optimal_position's amp).
                conf = decision.confidence
                pred_sign = 1.0 if fire_long else -1.0
                predictor = pred_sign * conf * conf
                # ---- v3 B: OOS-calibrated rolling per-bin Kelly ---- #
                # When the calibrator is enabled AND warm, override the
                # analytical sizer's |size| with the per-bin Kelly
                # f_b = μ_b/σ_b² estimated from rolling OOS history.
                # When not warm OR not enabled, the analytical
                # Conviction-Power Kelly stays in force as the
                # bootstrap (no behaviour change in cold start).
                if kelly_calibrator is not None and kelly_calibrator.is_warm():
                    f_signed = kelly_calibrator.kelly_fraction(predictor)
                    f_abs = abs(f_signed)
                    # Apply the same per-leg risk fractioning as the
                    # analytical sizer so pyramiding is consistent.
                    f_abs = f_abs / max(1, p.max_pyramid_legs)
                    if f_abs > 0:
                        size = float(min(f_abs, p.sizing_cap))
                if size <= 0:
                    continue
                # Cap so total exposure respects sizing_cap.
                used = sum(L["size"] for L in legs_now)
                size = min(size, max(0.0, p.sizing_cap - used))
                if size <= 0:
                    continue

                side = "long" if fire_long else "short"
                side_sign = 1.0 if fire_long else -1.0
                entry_px = close * (1.0 + self.slippage_bps * 1e-4 * side_sign)
                bar_pnl[t] -= size * cost_per_fill
                positions.setdefault(sym, []).append(dict(
                    side=side, entry_idx=t, entry_px=float(entry_px),
                    size=float(size),
                    peak=float(close),
                    trough=float(close),
                    scale=int(pick_scale),
                ))
                # Record the predictor used so the calibrator can
                # ingest the realised pnl when this leg exits.
                if kelly_calibrator is not None:
                    entry_predictor[(sym, t)] = float(predictor)
                consumed_pulse_bar[sym] = t

            if positions:
                bars_with_position += 1

        # Mark out anything still open (per-leg)
        for sym, legs in list(positions.items()):
            close = pre[sym].closes[-1]
            for leg in legs:
                side_sign = 1.0 if leg["side"] == "long" else -1.0
                pnl = (np.log(close / leg["entry_px"]) * side_sign * leg["size"]
                        - 2.0 * leg["size"] * cost_per_fill)
                trades.append(Trade(
                    symbol=sym, side=leg["side"],
                    entry_ts=idx[leg["entry_idx"]], exit_ts=idx[-1],
                    entry_price=float(leg["entry_px"]),
                    exit_price=float(close),
                    size_fraction=float(leg["size"]),
                    pnl=float(pnl),
                    bars_held=n_bars - 1 - leg["entry_idx"],
                    exit_reason="mark_out",
                ))

        exposure = bars_with_position / max(n_bars - warmup_bars, 1)

        # Telemetry — universe size vs per-cycle targets, so the user can
        # confirm the screener is actually firing (and not just rejecting
        # everything in the candidate pool).
        if rescreen_history:
            picks = [p for _, _, p in rescreen_history]
            telemetry = {
                "universe_size": rescreen_history[0][1],
                "rescreen_count": len(rescreen_history),
                "picks_total": sum(picks),
                "picks_mean":  float(sum(picks) / len(picks)),
                "picks_max":   int(max(picks)),
                "picks_zero_cycles": int(sum(1 for p in picks if p == 0)),
            }
        else:
            telemetry = {"universe_size": len(symbols),
                          "rescreen_count": 0, "picks_total": 0,
                          "picks_mean": 0.0, "picks_max": 0,
                          "picks_zero_cycles": 0}

        return {
            "bar_returns": bar_pnl[warmup_bars:],
            "trades": trades,
            "exposure": float(exposure),
            "telemetry": telemetry,
        }
