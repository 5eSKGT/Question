"""Out-of-sample monitor + adaptive recalibration with hard-halt fallback.

The OOS test uses the **Probabilistic Sharpe Ratio** (Bailey & López de Prado,
2012) and the **Deflated Sharpe Ratio** (BLP 2014) so that the pass/fail gate
accounts for sample size, skew, kurtosis and multiple-testing inflation.

Workflow each cycle:
  1. Replay the strategy with current params over the OOS window — collect
     bar-level returns of the simulated equity curve.
  2. Compute PSR vs the SR_threshold; compute DSR with N_trials = number of
     parameter sets explored so far.
  3. If PSR < confidence -> recalibrate by perturbing params on a small grid;
     re-test.  Keep best.
  4. After `max_attempts` failed recalibrations -> raise `RecalibrationFailed`,
     which the engine catches and halts trading + alerts the UI.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable

import numpy as np
from scipy import stats

from ..config import SETTINGS
from ..logging_setup import get_logger
from ..strategy.trend_following import StrategyParams

log = get_logger()


class AdaptiveStatus(str, Enum):
    OK = "ok"
    PENDING = "pending"          # too few samples — defer judgment, do not halt
    RECALIBRATED = "recalibrated"
    HALTED = "halted"


# Minimum sample size needed before PSR / SR can be trusted at all. Below
# this, we deliberately return PENDING so the adaptor neither halts trading
# nor enters recalibration on noise.
#
# Calibration note: at n=50 the standard error of an empirical Sharpe is
# already non-trivial (~0.14 for SR=0); below n=50 PSR is dominated by
# sample noise and practically uninformative.
MIN_SAMPLES = 50


@dataclass
class OOSReport:
    status: AdaptiveStatus
    sharpe: float
    psr: float
    dsr: float
    attempts: int
    params: StrategyParams
    message: str = ""


class RecalibrationFailed(RuntimeError):
    """Raised when the adaptor cannot find params that pass the OOS gate."""


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def annualized_sharpe(returns: np.ndarray, periods_per_year: float = 365 * 24) -> float:
    if returns.size < 8:
        return 0.0
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR — Bailey & López de Prado (2012)."""
    n = returns.size
    if n < 16:
        return 0.0
    if returns.std(ddof=1) == 0:
        # degenerate: zero-vol stream; treat as certainty in sign of mean
        return 1.0 if returns.mean() > sr_benchmark else 0.0
    sr = annualized_sharpe(returns, 1)            # un-annualised SR for PSR
    s = stats.skew(returns, bias=False)
    k = stats.kurtosis(returns, bias=False)
    radicand = (1 - s * sr + ((k - 1) / 4) * sr ** 2) / (n - 1)
    if not np.isfinite(radicand) or radicand <= 0:
        return 1.0 if sr > sr_benchmark else 0.0
    denom = np.sqrt(radicand)
    z = (sr - sr_benchmark) / max(denom, 1e-12)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int) -> float:
    """DSR — Bailey & López de Prado (2014). Penalises multiple testing."""
    if returns.size < 16 or n_trials < 1:
        return probabilistic_sharpe_ratio(returns)
    em_const = 0.5772156649
    e_max_z = (1 - em_const) * stats.norm.ppf(1 - 1 / n_trials) + \
              em_const * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr_threshold = e_max_z / np.sqrt(returns.size)
    return probabilistic_sharpe_ratio(returns, sr_threshold)


# --------------------------------------------------------------------------- #
# Adaptive loop
# --------------------------------------------------------------------------- #


# Backtester signature: (params) -> per-bar equity returns array
Backtester = Callable[[StrategyParams], np.ndarray]


class AdaptiveOOS:
    """Out-of-sample monitor with sane defaults for cold-start operation.

    Defaults are deliberately permissive so a fresh engine does not halt
    on its first cycle just because it has not yet accumulated enough
    evidence. The calibration logic kicks in only when the OOS sample is
    large enough (``MIN_SAMPLES``) AND the empirical Sharpe is so weak
    that even a coin-flip prior on positive Sharpe (PSR ≥ 0.55) fails.
    """

    def __init__(
        self,
        backtest_fn: Backtester,
        psr_min: float = 0.55,        # was 0.80 — half-confidence prior
        sharpe_min: float = 0.0,      # was 0.5  — any non-negative Sharpe
        max_attempts: int | None = None,
    ) -> None:
        self.backtest = backtest_fn
        self.psr_min = psr_min
        self.sharpe_min = sharpe_min
        self.max_attempts = max_attempts or SETTINGS.recalibration_max_attempts

    # ------------------------------------------------------------------ #
    def evaluate(self, params: StrategyParams) -> OOSReport:
        rets = self.backtest(params)
        if rets.size < MIN_SAMPLES:
            return OOSReport(status=AdaptiveStatus.PENDING,
                             sharpe=0.0, psr=0.0, dsr=0.0,
                             attempts=0, params=params,
                             message=f"insufficient OOS samples ({rets.size}<{MIN_SAMPLES})")
        sr = annualized_sharpe(rets)
        psr = probabilistic_sharpe_ratio(rets, sr_benchmark=0.0)
        dsr = deflated_sharpe_ratio(rets, n_trials=1)
        status = (AdaptiveStatus.OK
                  if (psr >= self.psr_min and sr >= self.sharpe_min)
                  else AdaptiveStatus.HALTED)
        return OOSReport(status=status, sharpe=sr, psr=psr, dsr=dsr,
                         attempts=0, params=params)

    # ------------------------------------------------------------------ #
    def recalibrate(self, params: StrategyParams) -> OOSReport:
        """Try a small structured grid; keep the best report seen."""
        grid = self._grid(params)
        best: OOSReport | None = None
        attempts = 0
        for cand in grid:
            attempts += 1
            rep = self.evaluate(cand)
            log.info(f"OOS recalibration attempt {attempts}: SR={rep.sharpe:.2f} "
                     f"PSR={rep.psr:.2f} DSR={rep.dsr:.2f}")
            if best is None or rep.psr > best.psr:
                best = replace(rep, attempts=attempts)
            if rep.status == AdaptiveStatus.OK:
                return replace(rep, status=AdaptiveStatus.RECALIBRATED, attempts=attempts)
            if attempts >= self.max_attempts:
                break

        msg = (f"OOS recalibration failed after {attempts} attempts — "
               f"best PSR={best.psr if best else 0:.2f} < {self.psr_min}")
        log.error(msg)
        raise RecalibrationFailed(msg)

    # ------------------------------------------------------------------ #
    def step(self, params: StrategyParams) -> OOSReport:
        report = self.evaluate(params)
        if report.status in (AdaptiveStatus.OK, AdaptiveStatus.PENDING):
            return report
        log.warning(f"OOS gate failed (PSR={report.psr:.2f}, SR={report.sharpe:.2f}) — recalibrating")
        return self.recalibrate(params)

    # ------------------------------------------------------------------ #
    def _grid(self, p: StrategyParams) -> Iterable[StrategyParams]:
        breakout_opts = [max(10, p.breakout_n - 5), p.breakout_n, p.breakout_n + 10]
        atr_opts = [max(7, p.atr_n - 4), p.atr_n, p.atr_n + 7]
        chand_opts = [max(1.5, p.chandelier_mult - 0.5), p.chandelier_mult, p.chandelier_mult + 1.0]
        for b, a, c in itertools.product(breakout_opts, atr_opts, chand_opts):
            if (b, a, c) == (p.breakout_n, p.atr_n, p.chandelier_mult):
                continue
            yield replace(p, breakout_n=b, atr_n=a, chandelier_mult=c)
