"""Trend-following crypto-futures system for Bitget.

Independent from the existing VWAP / TITAN TRADING stack. Targets cross-sectional
WINNER / LOSER symbols screened from the full Bitget USDT-perpetual universe,
sized under a CVaR floor, and continuously self-recalibrated against OOS data.
"""
__version__ = "0.1.0"
