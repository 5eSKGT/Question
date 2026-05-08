"""Aggregated portfolio state shared between the engine and the UI."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import STATE_DIR
from ..strategy.trend_following import Signal


@dataclass
class TradeMessage:
    ts: pd.Timestamp
    symbol: str
    text: str
    delta_usdt: float = 0.0    # signed P&L attributable to this message
    kind: str = "info"         # info | entry | exit | warning | halt


@dataclass
class PortfolioState:
    equity_usdt: float = 0.0
    prev_equity_usdt: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    mode: str = "paper"
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    messages: list[TradeMessage] = field(default_factory=list)
    last_oos: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- mutators ----------------------------------------------------- #
    def update_equity(self, value: float) -> None:
        with self._lock:
            self.prev_equity_usdt = self.equity_usdt
            self.equity_usdt = value

    def add_signal(self, sig: Signal) -> None:
        """Append a signal, deduplicating by (symbol, ts, type, side, source).

        Each cycle re-replays the strategy on the full candle window, so the
        same (symbol, ts, side, type) HIST entry could appear repeatedly. We
        suppress the duplicate so the chart and any downstream consumers see
        each event exactly once.
        """
        key = (sig.symbol, sig.ts, sig.type.value, sig.side, sig.source.value)
        with self._lock:
            for existing in self.signals:
                if (existing.symbol, existing.ts, existing.type.value,
                        existing.side, existing.source.value) == key:
                    return
            self.signals.append(sig)
            self.signals = self.signals[-2000:]

    def add_message(self, msg: TradeMessage) -> None:
        with self._lock:
            self.messages.append(msg)
            self.messages = self.messages[-200:]

    def halt(self, reason: str) -> None:
        with self._lock:
            self.halted = True
            self.halt_reason = reason

    def resume(self) -> None:
        with self._lock:
            self.halted = False
            self.halt_reason = ""

    # ---- views -------------------------------------------------------- #
    @property
    def equity_delta(self) -> float:
        return self.equity_usdt - self.prev_equity_usdt

    def equity_color(self) -> str:
        d = self.equity_delta
        if d > 0:
            return "#1faa59"     # green
        if d < 0:
            return "#d2474d"     # red
        return "#7a8085"          # gray

    # ---- persistence -------------------------------------------------- #
    def snapshot_path(self) -> Path:
        return STATE_DIR / "portfolio.json"

    def save(self) -> None:
        with self._lock:
            data = {
                "equity_usdt": self.equity_usdt,
                "prev_equity_usdt": self.prev_equity_usdt,
                "halted": self.halted,
                "halt_reason": self.halt_reason,
                "mode": self.mode,
                "positions": self.positions,
                "messages": [
                    {**asdict(m), "ts": m.ts.isoformat()} for m in self.messages[-50:]
                ],
                "last_oos": self.last_oos,
            }
        self.snapshot_path().write_text(json.dumps(data, indent=2, default=str))
