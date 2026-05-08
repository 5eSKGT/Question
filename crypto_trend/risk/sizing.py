"""Optimal position sizing — Kelly + volatility targeting + CVaR floor.

Composes three orthogonal sizing principles into one position fraction so
that whichever constraint binds first protects the strategy:

* **Fractional Kelly** (Kelly 1956; Thorp 2006) — ``f_K = μ̂ / σ̂²``
  with a *safety multiplier* (typical practitioner choice: 0.25–0.5).
  Maximises expected log-growth under the empirical return distribution.

* **Volatility targeting** (Moreira & Muir 2017 *J. Finance* 72(4)) —
  ``f_v = σ_target / σ_asset`` keeps the annualised portfolio
  volatility roughly constant across regimes, the standard practice
  for time-series momentum funds.

* **CVaR floor** (Rockafellar & Uryasev 2000) — ``f_c = floor / CVaR_α``,
  bounding the expected shortfall of a position under its empirical tail.

The actual position fraction is the *minimum* of the three (binding
constraint), then capped by an absolute leverage limit. The "optimal
leverage" is then the smallest integer leverage that supports the
implied notional; lower leverage means lower funding cost and a wider
liquidation buffer.

Synergy with the AlphaPulse strategy
------------------------------------
* The screener's Lee-Mykland statistic detects bars where the
  diffusion-vol estimate is small relative to the realised jump. This
  is *exactly* the regime where Kelly says to size up — μ̂ jumps,
  σ̂² stays moderate, so f_K spikes. The fractional safety factor
  prevents over-betting on a single statistic.
* The Yang-Zhang band that gates entries dampens σ̂_asset in chaotic
  markets, naturally pushing f_v down.
* CVaR_α is computed on the very same return window the screener uses,
  so the three pieces share a coherent sample.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .cvar import cvar_cornish_fisher


@dataclass
class SizingDecision:
    fraction: float          # of equity, may exceed 1 only with leverage
    leverage: int            # broker-side leverage to set
    binding: str             # "kelly" | "vol_target" | "cvar" | "cap"
    f_kelly: float
    f_vol_target: float
    f_cvar: float


def kelly_fraction(returns: np.ndarray) -> float:
    """Continuous Kelly fraction f* = μ / σ² on log-returns.

    Returns 0 when fewer than 16 observations or σ=0. The output is
    clipped to [0, 1] — full Kelly without a multiplier is well-known
    to be aggressive; the caller applies the safety factor.
    """
    if returns.size < 16:
        return 0.0
    mu = float(returns.mean())
    var = float(returns.var(ddof=1))
    if var <= 0:
        return 0.0
    return float(np.clip(mu / var, 0.0, 1.0))


def annualized_vol(returns: np.ndarray, periods_per_year: float = 365 * 24) -> float:
    if returns.size < 8:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def vol_target_fraction(returns: np.ndarray, target_vol: float = 0.20) -> float:
    """Position fraction that targets ``target_vol`` annualised."""
    sigma = annualized_vol(returns)
    if sigma <= 0:
        return 0.0
    return float(min(target_vol / sigma, 5.0))   # absolute cap to avoid blow-ups


def cvar_fraction(returns: np.ndarray, cvar_floor: float, alpha: float) -> float:
    """Same as risk.cvar.max_size_under_cvar but signature-symmetric."""
    if returns.size < 16:
        return 0.0
    cvar = cvar_cornish_fisher(returns, alpha)
    if cvar >= 0:
        return 5.0
    return float(max(0.0, cvar_floor / cvar))


def optimal_leverage(target_fraction: float, leverage_cap: int = 3,
                      maintenance_margin: float = 0.005) -> int:
    """Choose the smallest integer leverage that supports `target_fraction`.

    Uses a buffer above the implied notional so a small adverse move does
    not immediately breach maintenance margin. Funding cost grows with
    leverage on Bitget perps, so picking the smallest sufficient value is
    economically meaningful.

    target_fraction <= 1 → leverage = 1 (no margin amplification needed)
    1 < target_fraction <= cap → leverage = ceil(target_fraction) + 1 buffer,
                                  capped
    """
    if target_fraction <= 1.0:
        return 1
    naive = math.ceil(target_fraction) + 1     # one rung of safety
    return int(min(naive, leverage_cap))


def optimal_position(
    returns: np.ndarray,
    cvar_floor: float = -0.08,
    cvar_alpha: float = 0.05,
    target_vol: float = 0.20,
    kelly_safety: float = 0.5,                # half-Kelly default
    fraction_cap: float = 1.0,
    leverage_cap: int = 3,
) -> SizingDecision:
    """End-to-end sizing decision for one symbol given its return history."""
    f_k = kelly_safety * kelly_fraction(returns)
    f_v = vol_target_fraction(returns, target_vol)
    f_c = cvar_fraction(returns, cvar_floor, cvar_alpha)

    # Choose the binding (smallest) constraint
    candidates = {"kelly": f_k, "vol_target": f_v, "cvar": f_c}
    binding, f = min(candidates.items(), key=lambda kv: kv[1])
    if f >= fraction_cap:
        f = fraction_cap
        binding = "cap"

    lev = optimal_leverage(f, leverage_cap=leverage_cap)

    return SizingDecision(
        fraction=float(f),
        leverage=int(lev),
        binding=binding,
        f_kelly=float(f_k),
        f_vol_target=float(f_v),
        f_cvar=float(f_c),
    )
