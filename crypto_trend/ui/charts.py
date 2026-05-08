"""Unified signal chart rendering.

Critical correctness contract — every Signal, regardless of its `source`
(historical / live / OOS-realtime), is rendered using the **same** marker
shape, the **same** size, and the **same** color rule (long-entry green-up,
short-entry red-down, exits hollow). The only thing that differs across
A/B/C is the legend group, so the user can toggle them but they cannot
look inconsistent.

This is the "오류없이 일관되게" guarantee from the spec: one constant table,
one rendering function, no special cases.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go

from ..strategy.trend_following import Signal, SignalSource, SignalType


# --------------------------------------------------------------------------- #
# Single source of truth — same marker style for A, B and C
# --------------------------------------------------------------------------- #


SIGNAL_STYLE: dict[tuple[str, str], dict] = {
    # (side, signal-type) -> marker style
    ("long",  "entry"): {"symbol": "triangle-up",   "color": "#1faa59", "size": 13},
    ("short", "entry"): {"symbol": "triangle-down", "color": "#d2474d", "size": 13},
    ("long",  "exit"):  {"symbol": "x",             "color": "#1faa59", "size": 12},
    ("short", "exit"):  {"symbol": "x",             "color": "#d2474d", "size": 12},
}


SOURCE_LABEL = {
    SignalSource.HIST: "A · 과거 신호 / 매매 기록",
    SignalSource.LIVE: "B · 현재 매매 중",
    SignalSource.OOS:  "C · OOS 실시간 신규",
}


# --------------------------------------------------------------------------- #
def _trace_for_signals(signals: Iterable[Signal], source: SignalSource) -> list[go.Scatter]:
    """Build one trace per (side, type) so legend toggles are intuitive."""
    buckets: dict[tuple[str, str], list[Signal]] = {}
    for s in signals:
        if s.source != source:
            continue
        buckets.setdefault((s.side, s.type.value), []).append(s)

    traces: list[go.Scatter] = []
    for (side, typ), items in buckets.items():
        style = SIGNAL_STYLE[(side, typ)]
        traces.append(go.Scatter(
            x=[s.ts for s in items],
            y=[s.price for s in items],
            mode="markers",
            name=f"{SOURCE_LABEL[source]} — {side} {typ}",
            legendgroup=source.value,
            legendgrouptitle_text=SOURCE_LABEL[source],
            marker=dict(
                symbol=style["symbol"],
                color=style["color"],
                size=style["size"],
                line=dict(width=1.4, color="#222"),
                # OOS group emphasised with a halo so user spots it instantly
                opacity=1.0 if source != SignalSource.OOS else 1.0,
            ),
            hovertemplate=(
                f"<b>{SOURCE_LABEL[source]}</b><br>"
                f"%{{x|%Y-%m-%d %H:%M}}<br>"
                f"side={side} type={typ}<br>"
                f"price=%{{y:.4f}}<extra></extra>"
            ),
        ))
        # Special-effect: OOS gets an outlined ring for visibility *without*
        # changing the inner marker — keeps style invariant across A/B/C.
        if source == SignalSource.OOS:
            traces.append(go.Scatter(
                x=[s.ts for s in items],
                y=[s.price for s in items],
                mode="markers",
                showlegend=False,
                legendgroup=source.value,
                marker=dict(
                    symbol="circle-open",
                    color="rgba(0,0,0,0)",
                    size=style["size"] + 8,
                    line=dict(width=2, color="#0d6efd"),
                ),
                hoverinfo="skip",
            ))
    return traces


# --------------------------------------------------------------------------- #
def build_signal_chart(symbol: str, candles: pd.DataFrame,
                       signals: Iterable[Signal]) -> go.Figure:
    fig = go.Figure()

    if not candles.empty:
        fig.add_trace(go.Candlestick(
            x=candles.index,
            open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"],
            name=symbol,
            increasing_line_color="#1faa59",
            decreasing_line_color="#d2474d",
            increasing_fillcolor="rgba(31,170,89,0.45)",
            decreasing_fillcolor="rgba(210,71,77,0.45)",
        ))

    # A, B, C — added in order so layering is deterministic
    for source in (SignalSource.HIST, SignalSource.LIVE, SignalSource.OOS):
        for trace in _trace_for_signals(signals, source):
            fig.add_trace(trace)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafbfc",
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text=f"📈 {symbol} · A/B/C 신호 일관 랜더링",
                   font=dict(size=16, color="#333")),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="#eef0f2"),
        yaxis=dict(gridcolor="#eef0f2"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)"),
        height=560,
    )
    return fig
