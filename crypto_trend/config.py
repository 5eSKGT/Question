"""Runtime configuration loaded from environment / .env, runtime-mutable.

The desktop GUI overrides fields via :func:`apply` *before* the engine starts —
this is how mode toggling and API-key entry from the UI take effect without
restarting the process.  Fields are typed; setattr enforces nothing at runtime
but the GUI validates upstream.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "state"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ASSETS_DIR = Path(__file__).resolve().parent / "desktop" / "assets"
DOCS_DIR = Path(__file__).resolve().parent / "desktop" / "docs"
for _d in (STATE_DIR, LOG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


def _get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v not in ("", None) else default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:                 # NOTE: not frozen — GUI mutates at runtime
    mode: TradingMode
    api_key: str
    api_secret: str
    api_passphrase: str
    product_type: str
    base_equity_usdt: float
    max_gross_leverage: float
    cvar_alpha: float
    cvar_floor_pct: float
    walk_forward_train_days: int
    walk_forward_test_days: int
    recalibration_max_attempts: int
    log_level: str
    ui_host: str
    ui_port: int

    @property
    def is_live(self) -> bool:
        return self.mode == TradingMode.LIVE

    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret and self.api_passphrase)


def load_settings() -> Settings:
    raw_mode = (_get("TRADING_MODE", "paper") or "paper").lower()
    mode = TradingMode.LIVE if raw_mode == "live" else TradingMode.PAPER
    return Settings(
        mode=mode,
        api_key=_get("BITGET_API_KEY", "") or "",
        api_secret=_get("BITGET_API_SECRET", "") or "",
        api_passphrase=_get("BITGET_API_PASSPHRASE", "") or "",
        product_type=_get("BITGET_PRODUCT_TYPE", "USDT-FUTURES") or "USDT-FUTURES",
        base_equity_usdt=_get_float("BASE_EQUITY_USDT", 1000.0),
        max_gross_leverage=_get_float("MAX_GROSS_LEVERAGE", 3.0),
        cvar_alpha=_get_float("CVAR_ALPHA", 0.05),
        cvar_floor_pct=_get_float("CVAR_FLOOR_PCT", -0.08),
        walk_forward_train_days=_get_int("WALK_FORWARD_TRAIN_DAYS", 30),
        walk_forward_test_days=_get_int("WALK_FORWARD_TEST_DAYS", 7),
        recalibration_max_attempts=_get_int("RECALIBRATION_MAX_ATTEMPTS", 3),
        log_level=_get("LOG_LEVEL", "INFO") or "INFO",
        ui_host=_get("UI_HOST", "127.0.0.1") or "127.0.0.1",
        ui_port=_get_int("UI_PORT", 8050),
    )


SETTINGS = load_settings()


def apply(**overrides) -> Settings:
    """Mutate the singleton from the GUI; unknown keys raise."""
    valid = {f.name for f in fields(SETTINGS)}
    for k, v in overrides.items():
        if k not in valid:
            raise KeyError(f"unknown setting: {k}")
        if k == "mode" and not isinstance(v, TradingMode):
            v = TradingMode.LIVE if str(v).lower() == "live" else TradingMode.PAPER
        setattr(SETTINGS, k, v)
    return SETTINGS
