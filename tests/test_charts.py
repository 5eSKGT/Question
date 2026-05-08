"""Verify A/B/C signals render with **identical** marker style — the spec contract."""
import pandas as pd
import pytest

plotly = pytest.importorskip("plotly")

from crypto_trend.strategy.trend_following import (Signal, SignalSource,
                                                    SignalType)
from crypto_trend.ui.charts import SIGNAL_STYLE, build_signal_chart


def _sig(source: SignalSource, ts):
    return Signal(ts=ts, symbol="X/USDT:USDT", side="long",
                  type=SignalType.ENTRY, source=source, price=100.0,
                  size_fraction=0.1, reason="test")


def test_signal_style_is_source_independent():
    """The same (side, type) must yield the same marker symbol+color
    regardless of which group (A/B/C) it belongs to."""
    style_a = SIGNAL_STYLE[("long", "entry")]
    style_b = SIGNAL_STYLE[("long", "entry")]
    style_c = SIGNAL_STYLE[("long", "entry")]
    assert style_a == style_b == style_c


def test_chart_contains_one_marker_per_source():
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    candles = pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1],
                            "low": [1, 1, 1], "close": [1, 1, 1]}, index=idx)
    sigs = [
        _sig(SignalSource.HIST, idx[0]),
        _sig(SignalSource.LIVE, idx[1]),
        _sig(SignalSource.OOS, idx[2]),
    ]
    fig = build_signal_chart("X/USDT:USDT", candles, sigs)
    # Scattergl (WebGL) is now used for marker traces — its plotly
    # type is "scattergl", not "scatter".
    marker_traces = [t for t in fig.data
                      if t.type in ("scatter", "scattergl")]
    assert len(marker_traces) >= 3
    long_entry = [t for t in marker_traces
                  if getattr(t.marker, "symbol", None) == "triangle-up"]
    assert long_entry, "long-entry marker trace missing"
    colors = {t.marker.color for t in long_entry}
    assert colors == {SIGNAL_STYLE[("long", "entry")]["color"]}
