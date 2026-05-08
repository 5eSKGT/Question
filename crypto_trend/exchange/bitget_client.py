"""Bitget USDT-perp client with a paper-mode shadow broker.

The live path uses ccxt for REST. WebSocket streaming is intentionally optional —
the system pulls candles via REST so the same code path works for backtesting,
paper trading and live without divergence.

Mode is selected from `Settings.mode`; calling `build_broker()` returns the
correct concrete class.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import SETTINGS, TradingMode
from ..logging_setup import get_logger

log = get_logger()

try:
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover - ccxt is a required runtime dep
    ccxt = None


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class Order:
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    price: float | None  # None for market
    type: str = "market"
    reduce_only: bool = False
    leverage: int | None = None    # passed through to broker.set_leverage
    client_id: str = field(default_factory=lambda: f"ct-{uuid.uuid4().hex[:10]}")


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    ts: pd.Timestamp
    order_id: str


@dataclass
class Position:
    symbol: str
    qty: float = 0.0       # signed: + long, - short
    avg_price: float = 0.0
    realized: float = 0.0


# --------------------------------------------------------------------------- #
# Live client
# --------------------------------------------------------------------------- #


class BitgetClient:
    """Thin ccxt wrapper. REST only — sufficient for hourly/4h trend systems."""

    def __init__(self) -> None:
        if ccxt is None:
            raise RuntimeError("ccxt not installed")
        self._ex = ccxt.bitget({
            "apiKey": SETTINGS.api_key,
            "secret": SETTINGS.api_secret,
            "password": SETTINGS.api_passphrase,
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
        })

    # ---- market data --------------------------------------------------- #
    def fetch_universe(self) -> list[str]:
        """All USDT-margined perpetuals listed on Bitget."""
        markets = self._ex.load_markets()
        return [
            s for s, m in markets.items()
            if m.get("swap") and m.get("quote") == "USDT" and m.get("active", True)
        ]

    def fetch_tickers(self, symbols: list[str] | None = None) -> dict[str, dict]:
        return self._ex.fetch_tickers(symbols)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        rows = self._ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    # ---- account ------------------------------------------------------- #
    def fetch_balance_usdt(self) -> float:
        bal = self._ex.fetch_balance({"type": "swap"})
        return float(bal.get("USDT", {}).get("total", 0.0))

    def fetch_positions(self) -> list[Position]:
        out: list[Position] = []
        for p in self._ex.fetch_positions():
            qty = float(p.get("contracts") or 0)
            if qty == 0:
                continue
            side = (p.get("side") or "").lower()
            signed = qty if side == "long" else -qty
            out.append(Position(
                symbol=p["symbol"],
                qty=signed,
                avg_price=float(p.get("entryPrice") or 0.0),
                realized=float(p.get("realizedPnl") or 0.0),
            ))
        return out

    # ---- leverage ----------------------------------------------------- #
    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Best-effort leverage update on Bitget. Silently ignores failures
        because some symbols default to a non-changeable leverage and
        because this function is called speculatively per entry."""
        try:
            self._ex.set_leverage(int(leverage), symbol)
        except Exception as e:                                  # noqa: BLE001
            log.debug(f"set_leverage {symbol}={leverage}: {e}")

    # ---- order routing ------------------------------------------------- #
    def submit(self, order: Order) -> Fill:
        if getattr(order, "leverage", None):
            self.set_leverage(order.symbol, int(order.leverage))
        params: dict[str, Any] = {"reduceOnly": order.reduce_only}
        resp = self._ex.create_order(
            symbol=order.symbol,
            type=order.type,
            side=order.side,
            amount=order.qty,
            price=order.price,
            params=params,
        )
        price = float(resp.get("average") or resp.get("price") or 0.0)
        return Fill(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=price,
            fee=float((resp.get("fee") or {}).get("cost") or 0.0),
            ts=pd.Timestamp.utcnow(),
            order_id=str(resp.get("id")),
        )


# --------------------------------------------------------------------------- #
# Paper broker — same surface as BitgetClient
# --------------------------------------------------------------------------- #


class PaperBroker:
    """In-memory simulated broker. Uses the live client's market-data surface."""

    TAKER_FEE = 6e-4  # 6 bps — Bitget swap taker

    def __init__(self) -> None:
        self.market = BitgetClient() if ccxt is not None and SETTINGS.api_key else None
        self.equity: float = SETTINGS.base_equity_usdt
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self._last_price: dict[str, float] = {}

    # ---- market data — delegate to live client when available ---------- #
    def fetch_universe(self) -> list[str]:
        if self.market is None:
            return []
        return self.market.fetch_universe()

    def fetch_tickers(self, symbols: list[str] | None = None) -> dict[str, dict]:
        if self.market is None:
            return {}
        tickers = self.market.fetch_tickers(symbols)
        # Keep paper-mode mark-to-market price fresh for every symbol we know
        # about, even if its OHLCV was not downloaded this cycle.
        for sym, t in tickers.items():
            last = t.get("last") or t.get("close") or t.get("info", {}).get("last")
            if last is None:
                continue
            try:
                self._last_price[sym] = float(last)
            except (TypeError, ValueError):
                pass
        return tickers

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        if self.market is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = self.market.fetch_ohlcv(symbol, timeframe, limit)
        if not df.empty:
            self._last_price[symbol] = float(df["close"].iloc[-1])
        return df

    # ---- account ------------------------------------------------------- #
    def fetch_balance_usdt(self) -> float:
        return self.equity + sum(self._mtm(p) for p in self.positions.values())

    def fetch_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.qty != 0.0]

    # ---- leverage (paper mode is a no-op but tracks the request) ------ #
    def set_leverage(self, symbol: str, leverage: int) -> None:
        self._leverage_by_sym = getattr(self, "_leverage_by_sym", {})
        self._leverage_by_sym[symbol] = int(leverage)

    # ---- order routing ------------------------------------------------- #
    def submit(self, order: Order) -> Fill:
        if order.leverage:
            self.set_leverage(order.symbol, order.leverage)
        price = order.price or self._last_price.get(order.symbol)
        if price is None or price <= 0:
            log.warning(f"PaperBroker: no last price for {order.symbol}; rejecting")
            raise RuntimeError(f"no reference price for {order.symbol}")
        fee = abs(order.qty) * price * self.TAKER_FEE
        signed = order.qty if order.side == "buy" else -order.qty
        pos = self.positions.setdefault(order.symbol, Position(order.symbol))

        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):
            new_qty = pos.qty + signed
            if new_qty != 0:
                pos.avg_price = (pos.avg_price * pos.qty + price * signed) / new_qty
            pos.qty = new_qty
        else:
            closing = min(abs(signed), abs(pos.qty))
            sign = 1 if pos.qty > 0 else -1
            pos.realized += sign * closing * (price - pos.avg_price)
            pos.qty += signed
            if pos.qty == 0:
                pos.avg_price = 0.0

        self.equity -= fee
        fill = Fill(order.symbol, order.side, order.qty, price, fee,
                    pd.Timestamp.utcnow(), order.client_id)
        self.fills.append(fill)
        return fill

    def _mtm(self, p: Position) -> float:
        last = self._last_price.get(p.symbol, p.avg_price)
        return (last - p.avg_price) * p.qty + p.realized


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_broker() -> BitgetClient | PaperBroker:
    """Branch by mode — the rest of the system never sees the difference."""
    if SETTINGS.mode == TradingMode.LIVE:
        if not (SETTINGS.api_key and SETTINGS.api_secret and SETTINGS.api_passphrase):
            raise RuntimeError("LIVE mode requires Bitget API credentials")
        log.warning("LIVE trading mode — orders will hit Bitget")
        return BitgetClient()
    log.info("PAPER trading mode — simulated fills only")
    return PaperBroker()
