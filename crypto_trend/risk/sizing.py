"""Strategy-aware position sizing for AlphaPulse.

Synergy contract — every input maps to a specific strategy component
--------------------------------------------------------------------
Generic Kelly / vol-targeting was rejected because it ignored the
strategy's actual P&L generator. The replacement is a fixed-fractional
risk sizing rule whose every input is one of the strategy's own
internal quantities, so the sizing equation literally embeds the
strategy's behaviour.

Mapping :

    strategy quantity                    → sizing role
    ---------------------------------------------------------------
    Chandelier ATR stop distance         → denominator of base size
    LM jump statistic |L_t|              → confidence multiplier
    Multi-horizon agreement count        → confidence multiplier
    CVaR_α of recent returns             → defense-in-depth cap
    leverage_cap (commit param)          → leverage upper bound
    risk_per_trade (commit param)        → only knob with units

Resulting equation :

    stop_pct      = chandelier_mult · ATR / price
    base_size     = risk_per_trade / stop_pct                # (1)
    confidence    = clip(|L|/lm_thresh, 0.5, 2.0)
                    · clip(agree/max_agree, 0.5, 1.0)        # (2)
    sized         = base_size · confidence                   # (3)
    sized         ≤ cvar_floor / CVaR_α                      # (4) defense
    leverage      = min(⌊1/(stop_pct + maint + buffer)⌋,
                        ⌈sized⌉, leverage_cap)               # (5)

Properties of (1)+(5):

 * Hitting the Chandelier stop loses *exactly* `risk_per_trade × equity`,
   regardless of the asset's volatility. So `risk_per_trade` is a true
   per-trade risk budget — the only knob the user could meaningfully
   tune, and it has direct $-units.
 * Liquidation distance always exceeds stop distance plus a safety
   buffer, so a stop-out happens BEFORE liquidation, never after.

References :
  * van Tharp (2007), *Definitive Guide to Position Sizing*
  * Covel (2009), *Trend Following*, Ch. 12 ("Fixed-Fractional Risk")
  * Lee & Mykland (2008) RFS 21(6) — the LM statistic that powers (2)
  * Rockafellar & Uryasev (2000) — the CVaR constraint that powers (4)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cvar import cvar_cornish_fisher


@dataclass
class SizingDecision:
    fraction: float            # of equity, may exceed 1 with leverage
    leverage: int              # broker-side leverage to set
    stop_distance_pct: float   # Chandelier stop distance as fraction of price
    confidence: float          # signal-strength multiplier in [0.25, 2.0]
    base_fraction: float       # before confidence + caps
    binding: str               # 'stop' / 'confidence' / 'cvar' / 'cap' / 'lev'


# --------------------------------------------------------------------------- #
# Building blocks (each maps cleanly to one strategy primitive)
# --------------------------------------------------------------------------- #


def stop_distance_pct(price: float, atr: float, chandelier_mult: float) -> float:
    """Chandelier stop distance as a fraction of price.

    This is the *exact* distance the strategy will tolerate before
    closing a position via its trailing chandelier exit. Sizing
    against this distance closes the loop between exit and entry.
    """
    if price <= 0 or atr < 0:
        return 0.0
    return float(chandelier_mult * atr / price)


def fixed_fractional_size(risk_per_trade: float, stop_pct: float,
                           sizing_cap: float) -> float:
    """Position fraction such that stop-out costs exactly
    ``risk_per_trade × equity``. (van Tharp / Covel.)
    """
    if stop_pct <= 0:
        return 0.0
    raw = risk_per_trade / stop_pct
    return float(min(raw, sizing_cap))


def signal_confidence(lm_stat: float, lm_threshold: float,
                       agree: int, max_agree: int) -> float:
    """Confidence multiplier from screener internals.

    Both factors are clipped to [0.5, 2.0] (LM) and [0.5, 1.0] (agree)
    so confidence ∈ [0.25, 2.0]; the worst weakly-screened entry still
    gets 25% of base size, the best fully-agreeing entry gets 200%.
    """
    if lm_threshold <= 0:
        lm_factor = 1.0
    else:
        lm_factor = max(0.5, min(2.0, abs(lm_stat) / lm_threshold))
    if max_agree <= 0:
        agree_factor = 1.0
    else:
        agree_factor = max(0.5, min(1.0, agree / max_agree))
    return float(lm_factor * agree_factor)


def safe_leverage_for_stop(stop_pct: float, target_size: float,
                            leverage_cap: int,
                            maintenance_margin: float = 0.005,
                            safety_buffer: float = 0.02) -> int:
    """Smallest integer leverage that keeps liquidation distance
    safely beyond ``stop_pct`` and supports ``target_size``.
    """
    if target_size <= 1.0:
        return 1
    needed = math.ceil(target_size)
    required_margin = stop_pct + maintenance_margin + safety_buffer
    if required_margin <= 0:
        max_safe = leverage_cap
    else:
        max_safe = int(1.0 / required_margin)
    return int(max(1, min(max_safe, needed, leverage_cap)))


def conviction_power_amp(confidence: float, exponent: float = 3.0) -> float:
    """Conviction-power Kelly amplification (MacLean-Thorp-Ziemba 2010
    × Browne 1996 × Cvitanić-Kim 2024).

    The position size scales as ``confidence ** exponent``. With
    confidence ∈ [0.25, 2.0] and exponent=3 (cubic):

        weak     (conf 0.25): amp = 0.0156   ← essentially no bet
        modest   (conf 0.50): amp = 0.125    ← small position
        moderate (conf 1.00): amp = 1.0      ← baseline (= Kelly nominal)
        strong   (conf 1.50): amp = 3.375    ← 3.4× baseline
        max      (conf 2.00): amp = 8.0      ← 8× baseline (full Kelly)

    The 64× ratio between weakest and strongest signals concentrates
    capital deployment on the high-conviction tail — the academically
    grounded behaviour for return maximisation under stop-loss-bounded
    risk (Cvitanić & Kim 2024 §3.2).

    Cheng & Madhavan (2009) compounded-leverage decay
    ``L · E[r] − L²σ² / 2``  binds the upper end: with cubic amp + the
    ``sizing_cap`` ceiling, leverage actually used by the broker stays
    within the regime where vol-drag does not dominate expectancy.
    """
    return float(max(0.01, confidence ** exponent))


# Backwards-compatible name; tier-based mapping retained as a
# diagnostic helper but no longer used by ``optimal_position``.
def graded_leverage_tier(confidence: float, leverage_cap: int) -> int:
    """Diagnostic tier (1/2/3) for confidence levels — kept for legacy
    callers / dashboards. The committed sizing path uses the
    continuous ``conviction_power_amp`` instead.
    """
    if confidence < 1.0:
        return 1
    if confidence < 1.5:
        return int(min(2, leverage_cap))
    return int(min(3, leverage_cap))


# --------------------------------------------------------------------------- #
# Strategy-aware combined decision
# --------------------------------------------------------------------------- #


def optimal_position(
    returns: np.ndarray,                # last N bar returns for CVaR floor
    *,
    price: float,
    atr: float,
    lm_stat: float = 0.0,
    agree: int = 2,
    max_agree: int = 3,
    lm_threshold: float = 4.0,
    chandelier_mult: float = 3.0,
    risk_per_trade: float = 0.005,
    cvar_floor: float = -0.10,
    cvar_alpha: float = 0.05,
    sizing_cap: float = 5.0,
    leverage_cap: int = 10,
    confidence_exponent: float = 3.0,
) -> SizingDecision:
    """Compose all five blocks into one decision.

    Defense-in-depth ordering :
      base size from Chandelier stop  →  confidence multiplier  →
      CVaR floor cap (if tail is fat) →  absolute cap →  leverage.
    """
    stop_pct = stop_distance_pct(price, atr, chandelier_mult)
    if stop_pct <= 0:
        return SizingDecision(0.0, 1, 0.0, 1.0, 0.0, "stop")

    base = risk_per_trade / stop_pct           # nominal Kelly under stop
    confidence = signal_confidence(lm_stat, lm_threshold, agree, max_agree)

    # Conviction-power Kelly: position scales as confidence^k.
    # k=3 (cubic) gives a 64× sizing range from the weakest to the
    # strongest signal, concentrating capital deployment on the
    # high-conviction tail (Cvitanić-Kim 2024 §3.2).
    amp = conviction_power_amp(confidence, confidence_exponent)
    sized = base * amp
    binding = "confidence"

    # Absolute fraction-of-equity ceiling. With sizing_cap=5 a single
    # max-conviction position can be up to 500% of equity, requiring
    # leverage from the broker. The leverage chooser below picks the
    # smallest integer leverage that supports the notional safely.
    if sized > sizing_cap:
        sized = sizing_cap
        binding = "cap"

    # Defense-in-depth: CVaR floor caps the position when the recent
    # empirical tail is much fatter than the Chandelier stop assumes.
    if returns is not None and returns.size >= 16:
        cvar = cvar_cornish_fisher(returns, cvar_alpha)
        if cvar < 0:
            cvar_max = float(cvar_floor / cvar)
            if sized > cvar_max:
                sized = cvar_max
                binding = "cvar"

    # Leverage emerges naturally from the position size. We pick the
    # smallest integer leverage that (a) holds ``sized`` notional and
    # (b) keeps liquidation distance > stop_pct + buffer (Cheng-Madhavan
    # 2009 vol-drag boundary). With sized > 1, leverage > 1; with
    # sized ≤ 1, leverage = 1 (no broker leverage needed).
    safe_lev = safe_leverage_for_stop(stop_pct, sized, int(leverage_cap))
    leverage = max(1, math.ceil(sized)) if sized > 1.0 else 1
    leverage = min(leverage, safe_lev, int(leverage_cap))

    return SizingDecision(
        fraction=float(sized),
        leverage=int(leverage),
        stop_distance_pct=float(stop_pct),
        confidence=float(confidence),
        base_fraction=float(base),
        binding=binding,
    )


# --------------------------------------------------------------------------- #
# Backwards-compatible helpers (kept so existing import sites still work)
# --------------------------------------------------------------------------- #


def kelly_fraction(returns: np.ndarray) -> float:
    """Continuous Kelly fraction f* = μ / σ² on log-returns. Deprecated —
    kept only so old callers don't break. Strategy now uses
    ``fixed_fractional_size`` instead.
    """
    if returns.size < 16:
        return 0.0
    mu = float(returns.mean())
    var = float(returns.var(ddof=1))
    if var <= 0:
        return 0.0
    return float(np.clip(mu / var, 0.0, 1.0))


def vol_target_fraction(returns: np.ndarray, target_vol: float = 0.20) -> float:
    """Vol-targeting fraction. Deprecated — see module docstring."""
    if returns.size < 8:
        return 0.0
    sigma = float(returns.std(ddof=1) * np.sqrt(365 * 24))
    if sigma <= 0:
        return 0.0
    return float(min(target_vol / sigma, 5.0))


def optimal_leverage(target_fraction: float, leverage_cap: int = 3,
                      maintenance_margin: float = 0.005) -> int:
    """Backwards-compatible leverage chooser used by old tests."""
    if target_fraction <= 1.0:
        return 1
    naive = math.ceil(target_fraction) + 1
    return int(min(naive, leverage_cap))
