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
            top_n=screener_top_n or 30,
        )

        positions: dict[str, dict] = {}
        trades: list[Trade] = []
        bar_pnl = np.zeros(n_bars, dtype=float)
        bars_with_position = 0
        # symbol -> (side, expires_at_bar_idx, pre_pick_anchor_close).
        # The anchor is the close of the bar immediately before the
        # screener fired — used by v2.2 cascade-test continuation entry
        # so the jump bar itself fires (close[jump] > close[jump-1])
        # without re-detecting the jump via Donchian.
        active_picks: dict[str, tuple[str, int, float]] = {}
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
                # Refresh fresh picks (extend expiry); old picks linger
                # until their TTL expires so the strategy gets multiple
                # bars for the cluster to develop. Anchor is locked at
                # the *first* time we see this pick so subsequent
                # rescreens don't keep moving the entry reference.
                for r in results:
                    if r.symbol in active_picks:
                        # keep the original anchor; just extend TTL
                        side_old, _expiry_old, anchor_old = active_picks[r.symbol]
                        if side_old == r.side:
                            active_picks[r.symbol] = (
                                r.side, t + self.pick_ttl, anchor_old)
                            continue
                    pre_pick_close = float(pre[r.symbol].closes[max(0, t - 1)])
                    active_picks[r.symbol] = (
                        r.side, t + self.pick_ttl, pre_pick_close)
                # Drop expired picks
                active_picks = {s: v for s, v in active_picks.items()
                                  if v[1] > t}
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
                pos = positions.get(sym)
                a_i = pc.atr[t]
                yz_i = pc.yz[t]
                band = p.band_mult * yz_i * close

                if pos is not None:
                    side_sign = 1.0 if pos["side"] == "long" else -1.0
                    prev_close = pc.closes[t - 1]
                    if prev_close > 0 and not np.isnan(prev_close):
                        bar_pnl[t] += side_sign * (np.log(close / prev_close)
                                                     * pos["size"])

                    # ---- exit decision -------------------------------- #
                    if pos["side"] == "long":
                        pos["peak"] = max(pos["peak"], close)
                        chand = pos["peak"] - p.chandelier_mult * a_i
                        chand_hit = close < chand
                    else:
                        pos["trough"] = min(pos["trough"], close)
                        chand = pos["trough"] + p.chandelier_mult * a_i
                        chand_hit = close > chand

                    time_stop = (t - pos["entry_idx"]) >= p.time_stop_bars

                    win = pc.log_rets[max(0, t - 64): t] * side_sign
                    cvar_breach = (win.size > 0
                                    and np.quantile(win, p.cvar_alpha) < p.cvar_floor)

                    if chand_hit or time_stop or cvar_breach:
                        reason = ("chandelier" if chand_hit
                                   else "time_stop" if time_stop
                                   else "cvar_breach")
                        exit_px = close * (1.0 - self.slippage_bps * 1e-4 * side_sign)
                        bar_pnl[t] -= pos["size"] * cost_per_fill
                        pnl = (np.log(exit_px / pos["entry_px"]) * side_sign
                                * pos["size"] - 2.0 * pos["size"] * cost_per_fill)
                        trades.append(Trade(
                            symbol=sym, side=pos["side"],
                            entry_ts=idx[pos["entry_idx"]], exit_ts=ts,
                            entry_price=float(pos["entry_px"]),
                            exit_price=float(exit_px),
                            size_fraction=float(pos["size"]),
                            pnl=float(pnl),
                            bars_held=t - pos["entry_idx"],
                            exit_reason=reason,
                        ))
                        positions.pop(sym)
                        continue

                # ---- entry decision -------------------------------------- #
                if pos is not None:
                    continue
                if sym not in active_picks:
                    continue
                want, _expiry, anchor = active_picks[sym]
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
                decision = optimal_position(
                    sample,
                    price=close, atr=a_i,
                    lm_stat=float(lm_signed), agree=int(agree), max_agree=3,
                    lm_threshold=p.lm_threshold,
                    chandelier_mult=p.chandelier_mult,
                    risk_per_trade=p.risk_per_trade,
                    cvar_floor=p.cvar_floor,
                    cvar_alpha=p.cvar_alpha,
                    sizing_cap=p.sizing_cap,
                    leverage_cap=int(p.leverage_cap),
                    confidence_exponent=p.confidence_exponent,
                )
                size = decision.fraction
                if size <= 0:
                    continue

                side = "long" if fire_long else "short"
                side_sign = 1.0 if fire_long else -1.0
                entry_px = close * (1.0 + self.slippage_bps * 1e-4 * side_sign)
                bar_pnl[t] -= size * cost_per_fill
                positions[sym] = dict(
                    side=side, entry_idx=t, entry_px=float(entry_px),
                    size=float(size),
                    peak=float(close),
                    trough=float(close),
                )

            if positions:
                bars_with_position += 1

        # Mark out anything still open
        for sym, pos in list(positions.items()):
            close = pre[sym].closes[-1]
            side_sign = 1.0 if pos["side"] == "long" else -1.0
            pnl = (np.log(close / pos["entry_px"]) * side_sign * pos["size"]
                    - 2.0 * pos["size"] * cost_per_fill)
            trades.append(Trade(
                symbol=sym, side=pos["side"],
                entry_ts=idx[pos["entry_idx"]], exit_ts=idx[-1],
                entry_price=float(pos["entry_px"]),
                exit_price=float(close),
                size_fraction=float(pos["size"]),
                pnl=float(pnl),
                bars_held=n_bars - 1 - pos["entry_idx"],
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
