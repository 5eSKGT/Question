"""CVaR (a.k.a. Expected Shortfall) — academic backbone of position sizing.

Two estimators:

* `cvar_historical`        : non-parametric, Acerbi & Tasche (2002).
* `cvar_cornish_fisher`    : 4-moment expansion of Gaussian VaR (Favre & Galeano 2002).
                             Useful when sample size is small and tails matter.

`max_size_under_cvar` solves for the largest position fraction `f` of equity
such that  E[L | L >= VaR_alpha] of `f * R`  >= floor (a negative number, e.g.
-0.08 for an 8% expected-shortfall budget).  Because CVaR scales linearly with
position size for a fixed return distribution, the closed form is trivial — but
the function still bounds against leverage and a hard upper cap.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def cvar_historical(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical CVaR / Expected Shortfall on a *signed* return sample.

    Returns a *signed* number; for losses you'll get a negative value.
    """
    if returns.size == 0:
        return 0.0
    var = np.quantile(returns, alpha)
    tail = returns[returns <= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())


def cvar_cornish_fisher(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Cornish-Fisher 4-moment CVaR. Used when n is small."""
    if returns.size < 16:
        return cvar_historical(returns, alpha)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return 0.0
    s = stats.skew(returns, bias=False)
    k = stats.kurtosis(returns, bias=False)        # excess kurtosis
    z = stats.norm.ppf(alpha)
    z_cf = (z
            + (z ** 2 - 1) * s / 6
            + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * (s ** 2) / 36)
    # ES under Gaussian: -phi(z)/alpha — adjust mean/std and use cf-z
    es_z = -stats.norm.pdf(z_cf) / alpha
    return float(mu + sigma * es_z)


def max_size_under_cvar(
    returns: np.ndarray,
    cvar_floor: float,
    alpha: float = 0.05,
    cap: float = 1.0,
    leverage_cap: float = 3.0,
) -> float:
    """Largest position fraction (of equity) keeping CVaR_alpha >= floor.

    `cvar_floor` is negative (e.g. -0.08). `cap` is an absolute upper bound on
    the fraction (e.g. 1.0 = full equity). `leverage_cap` is gross leverage.
    """
    if returns.size < 16:
        return 0.0
    cvar = cvar_cornish_fisher(returns, alpha)
    if cvar >= 0:
        return min(cap, leverage_cap)
    # f * cvar >= floor   =>   f <= floor / cvar  (both negative, ratio positive)
    f = cvar_floor / cvar
    return float(max(0.0, min(f, cap, leverage_cap)))
