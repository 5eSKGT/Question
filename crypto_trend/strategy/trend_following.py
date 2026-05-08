"""Trend-following strategy designed around extreme movers.

Entry logic (post-screener):
  * Donchian breakout confirmation in the screener-implied direction.
  * Wave-band: enter only if price sits inside the *expected* fluctuation band,
    where the band is built from a Yang–Zhang volatility estimate (drift-free,
    overnight-aware) — Yang & Zhang (2000, J. of Business).

Exit logic:
  * Chandelier exit (Le Beau): trailing ATR(N) * mult from running peak/trough
    in the trade direction.
  * Time stop: K bars without progressing past entry.
  * CVaR breach: rolling realized CVaR of position P&L breaks the floor.

The whole strategy is parameterised so the OOS adaptor can re-tune it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd

from ..risk.sizing import optimal_position
from ..screener.winner_loser import (lee_mykland_statistic,
                                       multi_horizon_alignment)


class SignalType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class SignalSource(str, Enum):
    HIST = "historical"     # group A — what would have been traded historically
    LIVE = "live"           # group B — currently open position
    OOS = "oos_realtime"    # group C — newly triggered by latest OOS bar


@dataclass
class Signal:
    ts: pd.Timestamp
    symbol: str
    side: str               # "long" | "short"
    type: SignalType
    source: SignalSource
    price: float
    size_fraction: float = 0.0
    reason: str = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def yang_zhang_vol(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Drift-independent OHLC variance estimator."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    sigma_oc2 = log_oc.rolling(n).var()
    sigma_co2 = log_co.rolling(n).var()
    rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt(sigma_oc2 + k * sigma_co2 + (1 - k) * rs)


def donchian(df: pd.DataFrame, n: int = 20) -> tuple[pd.Series, pd.Series]:
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #


@dataclass
class StrategyParams:
    """Canonical strategy parameters — the strategy IS this configuration.

    Indicator parameters (breakout_n, atr_n, chandelier_mult) are mutated
    at runtime by ``AdaptiveOOS`` based on live OOS data. Risk and sizing
    parameters are committed defaults that constitute the strategy's
    identity.

    Sizing follows the strategy-aware contract documented in
    ``crypto_trend/risk/sizing.py``: every input maps to a specific
    strategy primitive (Chandelier stop, LM statistic, multi-horizon
    agreement, CVaR distribution). The only knob with $-units is
    ``risk_per_trade``: hitting the Chandelier stop loses exactly that
    fraction of equity, regardless of asset volatility.
    """
    # ---- indicator params (auto-tuned by OOS) -------------------- #
    breakout_n: int = 20
    atr_n: int = 14
    chandelier_mult: float = 3.0
    yz_n: int = 20
    band_mult: float = 1.5
    time_stop_bars: int = 48

    # ---- risk + sizing params (committed strategy identity) ------ #
    cvar_alpha: float = 0.05
    cvar_floor: float = -0.10
    leverage_cap: float = 3.0
    risk_per_trade: float = 0.01      # per-trade risk budget (= 1% of equity)
    sizing_cap: float = 2.0           # absolute fraction-of-equity ceiling
    lm_threshold: float = 4.0         # LM stat reference for confidence multiplier


class TrendFollowingStrategy:
    """Stateless w.r.t. positions — the engine threads position state in."""

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.p = params or StrategyParams()

    # ---- signal generation -------------------------------------------- #
    def generate_signals(
        self,
        df: pd.DataFrame,
        screener_side: str | None,
        source: SignalSource,
    ) -> list[Signal]:
        """Pass over `df` and emit entry+exit events as if trading from scratch.

        For source==HIST/OOS this walks the entire frame so the chart can render
        the would-have-been signals (group A) AND the freshly added bars (group
        C). The engine separates those by timestamp.
        """
        if df.empty or len(df) < max(self.p.breakout_n, self.p.atr_n, self.p.yz_n) + 2:
            return []

        a = atr(df, self.p.atr_n)
        yz = yang_zhang_vol(df, self.p.yz_n)
        hi, lo = donchian(df, self.p.breakout_n)
        out: list[Signal] = []

        position_side: str | None = None
        entry_price: float = 0.0
        entry_idx: int = -1
        peak: float = 0.0
        trough: float = 0.0

        closes = df["close"].to_numpy()
        rets = np.zeros_like(closes)
        rets[1:] = np.diff(np.log(closes))

        for i in range(self.p.breakout_n + 1, len(df)):
            ts = df.index[i]
            close = float(closes[i])
            atr_i = float(a.iloc[i] or 0.0)
            yz_i = float(yz.iloc[i] or 0.0)
            band = self.p.band_mult * yz_i * close

            if position_side is None:
                broke_up = close > float(hi.iloc[i - 1])
                broke_dn = close < float(lo.iloc[i - 1])
                inside_band = band == 0 or abs(close - float(closes[i - 1])) <= band

                want = screener_side
                if want is None:
                    want = "long" if broke_up else ("short" if broke_dn else None)

                fired_long = want == "long" and broke_up and inside_band
                fired_short = want == "short" and broke_dn and inside_band
                if fired_long or fired_short:
                    side = "long" if fired_long else "short"
                    side_sign = 1 if fired_long else -1
                    # Compute the screener's own internals at this bar, so the
                    # sizing decision uses the strategy's actual signal context
                    # (LM jump strength, multi-horizon agreement) rather than
                    # generic bar-level statistics.
                    pre = rets[: i]
                    lm = lee_mykland_statistic(pre, window=24)
                    if (side_sign > 0) != (lm > 0):
                        # For HIST replay we may not have a screener pick at
                        # every bar. Fall back to a neutral confidence so old
                        # backtests still produce signals.
                        lm = side_sign * abs(lm)
                    agree = multi_horizon_alignment(pre, side_sign,
                                                      horizons=(1, 4, 24))
                    sample = rets[max(0, i - 256): i]
                    decision = optimal_position(
                        sample,
                        price=close, atr=atr_i,
                        lm_stat=float(lm), agree=int(agree), max_agree=3,
                        lm_threshold=self.p.lm_threshold,
                        chandelier_mult=self.p.chandelier_mult,
                        risk_per_trade=self.p.risk_per_trade,
                        cvar_floor=self.p.cvar_floor,
                        cvar_alpha=self.p.cvar_alpha,
                        sizing_cap=self.p.sizing_cap,
                        leverage_cap=int(self.p.leverage_cap),
                    )
                    f = decision.fraction
                    reason = "donchian_break_up" if fired_long else "donchian_break_dn"
                    out.append(Signal(ts, df.attrs.get("symbol", ""), side,
                                      SignalType.ENTRY, source, close, f,
                                      reason,
                                      {"atr": atr_i, "yz": yz_i,
                                       "leverage": decision.leverage,
                                       "binding": decision.binding,
                                       "stop_pct": decision.stop_distance_pct,
                                       "confidence": decision.confidence,
                                       "lm": float(lm), "agree": int(agree)}))
                    position_side, entry_price, entry_idx = side, close, i
                    if fired_long:
                        peak = close
                    else:
                        trough = close
            else:
                # update trailing reference
                if position_side == "long":
                    peak = max(peak, close)
                    chandelier = peak - self.p.chandelier_mult * atr_i
                    hit = close < chandelier
                else:
                    trough = min(trough, close)
                    chandelier = trough + self.p.chandelier_mult * atr_i
                    hit = close > chandelier

                time_stop = (i - entry_idx) >= self.p.time_stop_bars
                # rolling realized CVaR check on most recent N bar P&L of the pos
                pnl_window = rets[max(0, i - 64): i] * (1 if position_side == "long" else -1)
                cvar_breach = (np.quantile(pnl_window, self.p.cvar_alpha)
                               if pnl_window.size else 0.0) < self.p.cvar_floor

                if hit or time_stop or cvar_breach:
                    reason = "chandelier" if hit else ("time_stop" if time_stop else "cvar_breach")
                    out.append(Signal(ts, df.attrs.get("symbol", ""), position_side,
                                      SignalType.EXIT, source, close, 0.0, reason,
                                      {"atr": atr_i}))
                    position_side = None
                    entry_price = 0.0
                    entry_idx = -1
                    peak = trough = 0.0

        return out
