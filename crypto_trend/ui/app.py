"""Light, intuitive Dash UI.

Design rules from the spec:
  * 밝고 직관적 — bootstrap "FLATLY" theme + white surfaces.
  * 자동매매 전용 — no manual order buttons. Only mode switch, start, halt.
  * 자산 변화 색상 규칙: +녹색 / -빨강 / 평소 회색.
  * A/B/C 신호 차트가 동일 스타일로 일관되게 랜더링 (charts.py).
"""
from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html

from ..config import SETTINGS
from ..portfolio.state import PortfolioState
from ..strategy.trend_following import SignalSource
from .charts import build_signal_chart

GREEN = "#1faa59"
RED = "#d2474d"
GRAY = "#7a8085"
ACCENT = "#0d6efd"


def _equity_color(delta: float) -> str:
    if delta > 0:
        return GREEN
    if delta < 0:
        return RED
    return GRAY


def _format_signed(v: float) -> str:
    sign = "+" if v > 0 else ("" if v < 0 else "±")
    return f"{sign}{v:,.2f}"


# --------------------------------------------------------------------------- #
def build_app(portfolio: PortfolioState,
              candle_provider) -> dash.Dash:
    """`candle_provider` is a callable: symbol -> DataFrame (engine cache)."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Crypto Trend Following · Bitget",
    )

    app.layout = dbc.Container(fluid=True, style={"padding": "24px",
                                                  "backgroundColor": "#f6f8fa",
                                                  "minHeight": "100vh"}, children=[
        # ---- header ---- #
        dbc.Row([
            dbc.Col(html.H3("🌤  Crypto Trend Following — Bitget",
                            style={"color": "#222", "margin": 0}), width="auto"),
            dbc.Col(dbc.Badge(id="mode-badge", color="info", className="ms-2"),
                    width="auto"),
        ], align="center", className="mb-3"),

        # ---- equity strip ---- #
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("자산 (USDT)", style={"color": GRAY, "fontSize": "0.9rem"}),
                html.H2(id="equity-value", style={"color": GRAY, "marginTop": "4px"}),
                html.Div(id="equity-delta", style={"fontSize": "1rem"}),
            ]), style={"borderRadius": "14px", "border": "0",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.05)"}), md=4),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("OOS 상태", style={"color": GRAY, "fontSize": "0.9rem"}),
                html.H4(id="oos-status", style={"marginTop": "4px"}),
                html.Div(id="oos-detail", style={"color": GRAY}),
            ]), style={"borderRadius": "14px", "border": "0",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.05)"}), md=4),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("거래 모드", style={"color": GRAY, "fontSize": "0.9rem"}),
                html.H4(id="mode-text", style={"marginTop": "4px"}),
                dbc.ButtonGroup([
                    dbc.Button("⏸ 매매 중단", id="halt-btn",
                               color="warning", outline=True, size="sm"),
                    dbc.Button("▶ 재개", id="resume-btn",
                               color="success", outline=True, size="sm"),
                ], style={"marginTop": "10px"}),
            ]), style={"borderRadius": "14px", "border": "0",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.05)"}), md=4),
        ], className="mb-3"),

        # ---- alert banner (halt warning) ---- #
        dbc.Alert(id="halt-alert", color="danger", is_open=False,
                  dismissable=False, style={"borderRadius": "10px"}),

        # ---- main grid ---- #
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span("심볼 선택  ", style={"color": GRAY}),
                    dcc.Dropdown(id="symbol-dd", clearable=False,
                                 style={"width": "260px", "display": "inline-block"}),
                ], style={"marginBottom": "10px"}),
                dcc.Graph(id="signal-chart", config={"displaylogo": False}),
            ]), style={"borderRadius": "14px", "border": "0",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.05)"}), md=8),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("📜 거래 메시지", style={"color": "#222"}),
                html.Hr(style={"margin": "8px 0"}),
                html.Div(id="messages", style={"maxHeight": "560px",
                                                "overflowY": "auto"}),
            ]), style={"borderRadius": "14px", "border": "0",
                       "boxShadow": "0 2px 8px rgba(0,0,0,0.05)"}), md=4),
        ]),

        dcc.Interval(id="tick", interval=2000, n_intervals=0),
        dcc.Store(id="symbol-store"),
    ])

    # -------------------- callbacks ---------------------------------- #
    @app.callback(
        Output("equity-value", "children"),
        Output("equity-value", "style"),
        Output("equity-delta", "children"),
        Output("equity-delta", "style"),
        Output("mode-badge", "children"),
        Output("mode-badge", "color"),
        Output("mode-text", "children"),
        Output("oos-status", "children"),
        Output("oos-status", "style"),
        Output("oos-detail", "children"),
        Output("halt-alert", "children"),
        Output("halt-alert", "is_open"),
        Output("symbol-dd", "options"),
        Output("symbol-dd", "value"),
        Output("messages", "children"),
        Input("tick", "n_intervals"),
        State("symbol-dd", "value"),
    )
    def _refresh(_n: int, current_symbol: str | None):
        equity = portfolio.equity_usdt
        delta = portfolio.equity_delta
        color = _equity_color(delta)

        eq_text = f"{equity:,.2f}"
        eq_style = {"color": color, "marginTop": "4px", "fontWeight": 600}
        delta_text = f"Δ {_format_signed(delta)} USDT"
        delta_style = {"color": color, "fontSize": "1rem", "fontWeight": 500}

        mode_label = portfolio.mode.upper()
        mode_color = "danger" if portfolio.mode == "live" else "info"
        oos = portfolio.last_oos or {}
        oos_status_text = oos.get("status", "—").upper()
        oos_color = (GREEN if oos_status_text == "OK"
                     else RED if oos_status_text == "HALTED"
                     else "#c08a00")
        oos_detail = (f"SR={oos.get('sharpe', 0):.2f}  PSR={oos.get('psr', 0):.2f}  "
                      f"DSR={oos.get('dsr', 0):.2f}  attempts={oos.get('attempts', 0)}")

        halt_open = portfolio.halted
        halt_msg = (f"⚠ 매매가 자동 중단되었습니다 — {portfolio.halt_reason}"
                    if halt_open else "")

        # symbol options drawn from the current signal universe
        syms = sorted({s.symbol for s in portfolio.signals if s.symbol})
        options = [{"label": s, "value": s} for s in syms]
        sel = current_symbol if current_symbol in syms else (syms[0] if syms else None)

        # message log — color-coded by sign of delta_usdt
        msgs = []
        for m in reversed(portfolio.messages[-50:]):
            c = (GREEN if m.delta_usdt > 0
                 else RED if m.delta_usdt < 0
                 else (RED if m.kind == "halt"
                       else "#c08a00" if m.kind == "warning"
                       else GRAY))
            ts = m.ts.strftime("%H:%M:%S") if isinstance(m.ts, pd.Timestamp) else str(m.ts)
            msgs.append(html.Div([
                html.Span(f"[{ts}] ", style={"color": GRAY, "fontSize": "0.85rem"}),
                html.Span(m.text, style={"color": c, "fontWeight": 500}),
            ], style={"padding": "4px 0", "borderBottom": "1px solid #eef0f2"}))

        return (eq_text, eq_style, delta_text, delta_style,
                mode_label, mode_color, mode_label,
                oos_status_text, {"color": oos_color, "marginTop": "4px"},
                oos_detail, halt_msg, halt_open,
                options, sel, msgs)

    @app.callback(
        Output("signal-chart", "figure"),
        Input("tick", "n_intervals"),
        Input("symbol-dd", "value"),
    )
    def _chart(_n: int, symbol: str | None):
        if not symbol:
            return build_signal_chart("—", pd.DataFrame(columns=["open", "high", "low", "close"]), [])
        candles = candle_provider(symbol) if candle_provider else pd.DataFrame()
        sigs = [s for s in portfolio.signals if s.symbol == symbol]
        return build_signal_chart(symbol, candles, sigs)

    @app.callback(
        Output("halt-btn", "n_clicks"),
        Input("halt-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _halt(n):
        if n:
            portfolio.halt("사용자 수동 중단")
        return 0

    @app.callback(
        Output("resume-btn", "n_clicks"),
        Input("resume-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _resume(n):
        if n:
            portfolio.resume()
        return 0

    return app


def run_ui(portfolio: PortfolioState, candle_provider) -> None:
    app = build_app(portfolio, candle_provider)
    app.run(host=SETTINGS.ui_host, port=SETTINGS.ui_port, debug=False)
