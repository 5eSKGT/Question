"""Top-level trading engine that wires the parts together.

Cycle (default 1h):
  1. Pull universe + 24h tickers (quote-volume).
  2. Run cross-sectional WINNER/LOSER screener on candles.
  3. For each candidate symbol:
       - replay the strategy on the candle history -> historical signals (group A)
       - emit live signal for the latest bar if strategy fires (group C)
  4. Submit orders for new entries / exits.
  5. Run the OOS adaptor on a recent slice; on RecalibrationFailed -> HALT.
  6. Persist state for the UI.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ..config import SETTINGS
from ..exchange import build_broker
from ..exchange.bitget_client import Order
from ..logging_setup import get_logger
from ..oos.adaptive import AdaptiveOOS, AdaptiveStatus, RecalibrationFailed
from ..portfolio.state import PortfolioState, TradeMessage
from ..screener import WinnerLoserScreener
from ..strategy.trend_following import (
    Signal, SignalSource, SignalType, StrategyParams, TrendFollowingStrategy,
)

log = get_logger()


class TradingEngine:
    def __init__(
        self,
        portfolio: PortfolioState,
        screener: WinnerLoserScreener | None = None,
        strategy: TrendFollowingStrategy | None = None,
        timeframe: str = "1h",
        history_bars: int = 500,
    ) -> None:
        self.broker = build_broker()
        self.portfolio = portfolio
        self.portfolio.mode = SETTINGS.mode.value
        self.screener = screener or WinnerLoserScreener()
        self.strategy = strategy or TrendFollowingStrategy()
        self.timeframe = timeframe
        self.history_bars = history_bars
        self._candles_cache: dict[str, pd.DataFrame] = {}
        self._tickers_cache: dict[str, dict] = {}
        self._open_positions: dict[str, str] = {}    # symbol -> side
        self._cycle_count = 0
        # During the warmup window we never honour an OOS HALT verdict —
        # a freshly-started engine has too little evidence for the
        # calibration logic to be trustworthy.
        self.oos_warmup_cycles = 5
        # Optional progress callback (stage, current, total) — let the
        # GUI render real-time feedback during long-running cycles.
        self.progress_callback = None

    # ------------------------------------------------------------------ #
    # Helpers used as providers by the screener
    # ------------------------------------------------------------------ #
    def _ohlcv(self, symbol: str) -> pd.DataFrame:
        if symbol in self._candles_cache:
            return self._candles_cache[symbol]
        df = self.broker.fetch_ohlcv(symbol, self.timeframe, self.history_bars)
        df.attrs["symbol"] = symbol
        self._candles_cache[symbol] = df
        return df

    def _quote_volume(self, symbol: str) -> float:
        t = self._tickers_cache.get(symbol) or {}
        for k in ("quoteVolume", "quote_volume", "vol_currency", "turnover_24h"):
            if k in t:
                try:
                    return float(t[k])
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _funding_rate(self, symbol: str) -> float | None:
        """Best-effort funding rate fetch for the screener's perp filter."""
        fetcher = getattr(self.broker, "fetch_funding_rate", None)
        if fetcher is None:
            return None
        try:
            return float(fetcher(symbol))
        except Exception:                                                # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # Run one full cycle
    # ------------------------------------------------------------------ #
    def _emit(self, stage: str, current: int = 0, total: int = 0) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(stage, current, total)
            except Exception:                                            # noqa: BLE001
                pass

    def run_once(self) -> None:
        if self.portfolio.halted:
            log.warning(f"engine halted ({self.portfolio.halt_reason}); skipping cycle")
            return

        self._cycle_count += 1
        self._emit("starting_cycle")

        try:
            self._emit("fetching_universe")
            universe = self.broker.fetch_universe()
            self._tickers_cache = self.broker.fetch_tickers(universe)
        except Exception as e:                                       # noqa: BLE001
            log.error(f"market-data fetch failed: {e}")
            return

        # Sync the open-position view with the broker's truth at every cycle.
        # Without this the engine could silently believe a position is still
        # open after the user manually closed it (live mode), or after the
        # paper broker has zeroed it via its own logic.
        try:
            self._open_positions = {
                p.symbol: ("long" if p.qty > 0 else "short")
                for p in self.broker.fetch_positions() if p.qty != 0
            }
        except Exception as e:                                       # noqa: BLE001
            log.debug(f"position sync failed: {e}")

        # Limit OHLCV downloads — only for symbols passing a cheap pre-filter.
        prefiltered = [
            s for s in universe
            if self._quote_volume(s) >= self.screener.min_quote_volume
        ]
        log.info(f"universe={len(universe)} prefiltered={len(prefiltered)}")
        self._emit("prefiltered", len(prefiltered), len(universe))

        self._candles_cache.clear()
        for i, s in enumerate(prefiltered, start=1):
            try:
                self._ohlcv(s)
            except Exception as e:                                   # noqa: BLE001
                log.debug(f"ohlcv fail {s}: {e}")
            if i % 5 == 0 or i == len(prefiltered):
                self._emit("downloading_ohlcv", i, len(prefiltered))

        self._emit("screening", 0, len(prefiltered))
        scored = self.screener.run(
            prefiltered, self._ohlcv, self._quote_volume,
            funding_rate_provider=self._funding_rate)
        log.info(
            f"cycle {self._cycle_count}: universe={len(universe)} "
            f"prefiltered={len(prefiltered)} picks={len(scored)} "
            f"(symbols: {[r.symbol for r in scored[:5]]}{'…' if len(scored)>5 else ''})")
        self._emit("screened", len(scored), len(prefiltered))
        # Mirror to portfolio so the GUI can show universe vs picks live
        self.portfolio.last_screen = {
            "universe": len(universe),
            "prefiltered": len(prefiltered),
            "picks": len(scored),
            "pick_symbols": [r.symbol for r in scored],
        }

        # ---- signal pass per symbol ---------------------------------- #
        for r in scored:
            df = self._candles_cache.get(r.symbol)
            if df is None or df.empty:
                continue

            # full-history replay -> Group A "what would have been" + Group C latest bar
            sigs = self.strategy.generate_signals(df, screener_side=r.side,
                                                  source=SignalSource.HIST)
            if not sigs:
                continue

            last_ts = df.index[-1]
            for s in sigs:
                # the most-recent timestamp belongs to group C
                if s.ts == last_ts:
                    s.source = SignalSource.OOS
                self.portfolio.add_signal(s)

            # ---- act on latest signal ------------------------------- #
            latest = sigs[-1]
            if latest.ts == last_ts:
                self._handle_live_signal(latest, r.last_price)

        # ---- OOS evaluation ----------------------------------------- #
        self._run_oos_check()

        # ---- update equity / state ---------------------------------- #
        try:
            equity = self.broker.fetch_balance_usdt()
            self.portfolio.update_equity(equity)
            self.portfolio.positions = {p.symbol: asdict(p) for p in self.broker.fetch_positions()}
        except Exception as e:                                       # noqa: BLE001
            log.error(f"portfolio refresh failed: {e}")

        self.portfolio.save()
        self._emit("done", self._cycle_count, self._cycle_count)

    # ------------------------------------------------------------------ #
    # Handle a freshly-fired live signal
    # ------------------------------------------------------------------ #
    def _handle_live_signal(self, sig: Signal, last_price: float) -> None:
        if sig.type == SignalType.ENTRY and sig.symbol not in self._open_positions:
            equity = max(self.portfolio.equity_usdt or SETTINGS.base_equity_usdt, 1.0)
            notional = sig.size_fraction * equity
            qty = notional / max(last_price, 1e-9)
            if qty <= 0:
                return
            order = Order(symbol=sig.symbol,
                          side="buy" if sig.side == "long" else "sell",
                          qty=qty, price=None,
                          leverage=int(sig.meta.get("leverage") or 0) or None)
            try:
                fill = self.broker.submit(order)
                self._open_positions[sig.symbol] = sig.side
                self.portfolio.add_message(TradeMessage(
                    ts=fill.ts, symbol=sig.symbol,
                    text=f"ENTRY {sig.side.upper()} {sig.symbol} @ {fill.price:.4f} "
                         f"(size {sig.size_fraction:.2%}, reason={sig.reason})",
                    delta_usdt=0.0, kind="entry",
                ))
                # group B = currently open position
                live_sig = Signal(sig.ts, sig.symbol, sig.side, SignalType.ENTRY,
                                  SignalSource.LIVE, fill.price, sig.size_fraction,
                                  sig.reason, sig.meta)
                self.portfolio.add_signal(live_sig)
            except Exception as e:                                   # noqa: BLE001
                log.error(f"entry order failed {sig.symbol}: {e}")

        elif sig.type == SignalType.EXIT and sig.symbol in self._open_positions:
            held = self.portfolio.positions.get(sig.symbol, {})
            qty = abs(float(held.get("qty", 0.0)))
            if qty <= 0:
                self._open_positions.pop(sig.symbol, None)
                return
            held_side = self._open_positions[sig.symbol]
            order = Order(symbol=sig.symbol,
                          side="sell" if held_side == "long" else "buy",
                          qty=qty, price=None, reduce_only=True)
            try:
                fill = self.broker.submit(order)
                self._open_positions.pop(sig.symbol, None)
                pnl = (fill.price - float(held.get("avg_price", fill.price))) * \
                       (1 if held_side == "long" else -1) * qty
                self.portfolio.add_message(TradeMessage(
                    ts=fill.ts, symbol=sig.symbol,
                    text=f"EXIT {held_side.upper()} {sig.symbol} @ {fill.price:.4f} "
                         f"(reason={sig.reason}, pnl={pnl:+.2f} USDT)",
                    delta_usdt=pnl, kind="exit",
                ))
                live_sig = Signal(sig.ts, sig.symbol, held_side, SignalType.EXIT,
                                  SignalSource.LIVE, fill.price, 0.0, sig.reason, sig.meta)
                self.portfolio.add_signal(live_sig)
            except Exception as e:                                   # noqa: BLE001
                log.error(f"exit order failed {sig.symbol}: {e}")

    # ------------------------------------------------------------------ #
    # OOS gate
    # ------------------------------------------------------------------ #
    def _run_oos_check(self) -> None:
        # Walk-forward window: the most recent K bars are treated as the OOS
        # test set. Older bars are warmup for the indicators only. This is
        # what the spec calls "OOS 데이터에 따라 유동적으로 전략을 자동보정".
        oos_bars = max(SETTINGS.walk_forward_test_days * 24, 24)

        # Prefer evaluating on the symbols we are actually trading; if we have
        # no live positions yet, evaluate on whatever the screener produced
        # this cycle so the OOS decision is grounded in current candidates.
        symbols = list(self._open_positions.keys())
        if not symbols:
            symbols = list(self._candles_cache.keys())[:10]
        if not symbols:
            return

        def _backtest(params: StrategyParams) -> np.ndarray:
            strat = TrendFollowingStrategy(params)
            tail_returns: list[np.ndarray] = []
            for sym in symbols:
                df = self._candles_cache.get(sym)
                if df is None or df.empty:
                    continue
                sigs = strat.generate_signals(df, screener_side=None,
                                              source=SignalSource.HIST)
                eq = self._returns_from_signals(df, sigs)
                # Keep only the OOS tail; the head is warmup / in-sample.
                if eq.size > oos_bars:
                    eq = eq[-oos_bars:]
                if eq.size:
                    tail_returns.append(eq)
            if not tail_returns:
                return np.array([])
            min_len = min(arr.size for arr in tail_returns)
            stacked = np.stack([arr[-min_len:] for arr in tail_returns], axis=0)
            return stacked.mean(axis=0)

        adaptor = AdaptiveOOS(_backtest)
        try:
            report = adaptor.step(self.strategy.p)
            # Apply new params only if the adaptor actually returned a tuned
            # set; for OK / PENDING the existing params are kept.
            if report.status == AdaptiveStatus.RECALIBRATED:
                self.strategy.p = report.params
            self.portfolio.last_oos = {
                "status": report.status.value,
                "sharpe": round(report.sharpe, 3),
                "psr": round(report.psr, 3),
                "dsr": round(report.dsr, 3),
                "attempts": report.attempts,
            }
            if report.status == AdaptiveStatus.RECALIBRATED:
                self.portfolio.add_message(TradeMessage(
                    ts=pd.Timestamp.utcnow(), symbol="*",
                    text=f"OOS recalibrated — new params SR={report.sharpe:.2f} PSR={report.psr:.2f}",
                    kind="info",
                ))
            elif report.status == AdaptiveStatus.PENDING:
                # No log spam — UI's OOS card already shows PENDING.
                pass
        except RecalibrationFailed as e:
            # Warmup guard: do not halt on cold-start cycles, just log.
            if self._cycle_count <= self.oos_warmup_cycles:
                self.portfolio.last_oos = {
                    "status": "warmup",
                    "sharpe": 0.0, "psr": 0.0, "dsr": 0.0, "attempts": 0,
                    "message": f"warmup cycle {self._cycle_count}/{self.oos_warmup_cycles}"
                                f" — OOS judgment deferred ({e})",
                }
                log.info(
                    f"OOS warmup ({self._cycle_count}/{self.oos_warmup_cycles}) — "
                    f"deferring halt: {e}")
                return
            self.portfolio.halt(str(e))
            self.portfolio.add_message(TradeMessage(
                ts=pd.Timestamp.utcnow(), symbol="*",
                text=f"⚠ TRADING HALTED — {e}",
                kind="halt",
            ))

    @staticmethod
    def _returns_from_signals(df: pd.DataFrame, sigs: list[Signal]) -> np.ndarray:
        """Reconstruct bar-level equity returns implied by entry/exit pairs."""
        if not sigs or df.empty:
            return np.array([])
        closes = df["close"].to_numpy(dtype=float)
        ts_to_idx = {ts: i for i, ts in enumerate(df.index)}
        rets = np.zeros(len(df) - 1)
        in_trade = False
        side = 0
        last_idx = 0
        for s in sigs:
            i = ts_to_idx.get(s.ts)
            if i is None:
                continue
            if s.type == SignalType.ENTRY and not in_trade:
                in_trade = True
                side = 1 if s.side == "long" else -1
                last_idx = i
            elif s.type == SignalType.EXIT and in_trade:
                seg = np.diff(np.log(closes[last_idx:i + 1])) * side
                rets[last_idx:last_idx + seg.size] = seg
                in_trade = False
                side = 0
        return rets
