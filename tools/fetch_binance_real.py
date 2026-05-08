"""Bulk-download real Binance USD-M Futures 1h klines and cache as parquet.

The sandbox can't reach exchange APIs directly, but the public Binance
historical-data S3 bucket (data.binance.vision) is reachable via
its ap-northeast-1 mirror at:

    https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/

Each file is a monthly ZIP of CSV klines. We pull N months across a
list of major USDT-perp symbols and concatenate them into one parquet
per symbol, stored under ``data/cache/<SYMBOL>_1h.parquet``.

Schema produced (matches the synthetic loader):
    index = pd.DatetimeIndex (UTC, hourly)
    columns = open, high, low, close, volume
    .attrs["symbol"] = "BTC/USDT:USDT"

Usage:
    python tools/fetch_binance_real.py --months 12 --symbols BTC ETH SOL ...
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crypto_trend.config import CACHE_DIR

S3_BASE = ("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
           "/data/futures/um/monthly/klines")
S3_BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

DEFAULT_SYMBOLS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX",
    "AVAX", "LINK", "DOT", "MATIC", "LTC", "BCH", "NEAR", "ATOM",
    "APT", "ARB", "OP", "SUI",
]


def discover_all_usdt_symbols() -> list[str]:
    """Enumerate every USD-M futures pair on data.binance.vision and
    keep only ASCII USDT-quoted symbols. Mirrors the live universe.
    """
    import re
    url = (f"{S3_BUCKET}/?prefix=data/futures/um/monthly/klines/"
           f"&delimiter=/&max-keys=1000")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", r.text)
    syms = []
    for p in prefixes:
        if not p.endswith("/"):
            continue
        leaf = p.split("/")[-2]
        if not leaf.endswith("USDT"):
            continue
        if not leaf.isascii() or not leaf.replace("USDT", "").isalnum():
            continue
        base = leaf[:-4]
        if not base:
            continue
        syms.append(base)
    return sorted(set(syms))


def _months(end: date, n: int) -> list[tuple[int, int]]:
    out = []
    y, m = end.year, end.month
    for _ in range(n):
        out.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return list(reversed(out))


def _download_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    pair = f"{symbol}USDT"
    url = f"{S3_BASE}/{pair}/1h/{pair}-1h-{year:04d}-{month:02d}.zip"
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:                                     # noqa: BLE001
        print(f"  ! {pair} {year}-{month:02d}: network {e}", file=sys.stderr)
        return None
    if r.status_code != 200:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                df = pd.read_csv(f, header=0)
    except Exception as e:                                     # noqa: BLE001
        print(f"  ! {pair} {year}-{month:02d}: parse {e}", file=sys.stderr)
        return None

    cols = {c.lower(): c for c in df.columns}
    open_col = cols.get("open_time") or df.columns[0]
    df = df.rename(columns={open_col: "open_time"})
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    ts = df["open_time"].astype("int64")
    if ts.iloc[0] > 10**14:
        idx = pd.to_datetime(ts, unit="us", utc=True)
    else:
        idx = pd.to_datetime(ts, unit="ms", utc=True)
    out = pd.DataFrame({
        "open":   pd.to_numeric(df["open"], errors="coerce").to_numpy(),
        "high":   pd.to_numeric(df["high"], errors="coerce").to_numpy(),
        "low":    pd.to_numeric(df["low"], errors="coerce").to_numpy(),
        "close":  pd.to_numeric(df["close"], errors="coerce").to_numpy(),
        "volume": pd.to_numeric(df["volume"], errors="coerce").to_numpy(),
    }, index=idx).sort_index()
    return out


def fetch_one(symbol: str, months: list[tuple[int, int]],
               cache_dir: Path) -> tuple[str, int]:
    pieces: list[pd.DataFrame] = []
    for y, m in months:
        df = _download_month(symbol, y, m)
        if df is not None and len(df):
            pieces.append(df)
    if not pieces:
        return symbol, 0
    full = pd.concat(pieces).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    full.attrs["symbol"] = f"{symbol}/USDT:USDT"
    out_name = f"{symbol}_USDT_USDT_1h.parquet"
    out_path = cache_dir / out_name
    full.to_parquet(out_path)
    return symbol, len(full)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=12,
                    help="months of history to pull (default 12)")
    p.add_argument("--symbols", nargs="*", default=None,
                    help="base symbols (default: discover ALL USDT-perp "
                         "symbols on data.binance.vision)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-symbols", type=int, default=None,
                    help="cap auto-discovered symbol count for quick runs")
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--end-month", type=int, default=4,
                    help="latest month to include (must be a fully-closed "
                         "month available on data.binance.vision)")
    args = p.parse_args()

    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    end = date(args.end_year, args.end_month, 1)
    months = _months(end, args.months)
    syms = args.symbols
    if syms is None:
        print("Discovering full USDT-perp universe on data.binance.vision …")
        syms = discover_all_usdt_symbols()
        print(f"  discovered {len(syms)} USDT pairs")
        if args.max_symbols:
            syms = syms[: args.max_symbols]
            print(f"  capped to {len(syms)} for this run")
    print(f"Fetching {len(syms)} symbols × {len(months)} months "
          f"from data.binance.vision → {cache_dir}")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s, months, cache_dir): s for s in syms}
        for fut in as_completed(futs):
            sym, n = fut.result()
            done += 1
            if n == 0 or done % 25 == 0:
                print(f"  [{done}/{len(syms)}] {sym}USDT  bars={n}")
    print(f"Done: {done} symbols cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
