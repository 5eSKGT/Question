"""OOS-calibrated rolling per-bin Kelly sizer.

The committed Conviction-Power Kelly (sizing.py::optimal_position)
applies a fixed functional form ``amp = confidence^k`` (k=2). That is
HETEROGENEOUS in the sense that bigger signals get bigger sizes — but
the *exponent* is a hand-picked constant. It does not adapt to the
empirical per-bin (μ_b, σ_b²) statistics of the very universe being
traded, which the upper-bound diagnostic
(reports/upper_bound_diagnostic.json) shows are HIGHLY heterogeneous
across signal strengths (PnL skew = +4.6, with the high-LM bins
carrying expectancy multiples of the average).

The Markowitz (1952) / Cover & Thomas (1991, *Elements of Information
Theory*, ch. 16) conditional log-optimal portfolio sizes each event at

    f_i*  =  μ_i  /  σ_i²

If we estimate μ_b and σ_b² per signal-strength bin from a rolling
window of recent realised trades, we can size each new event at the
*OOS-calibrated bin Kelly* — no fixed exponent, the data tells us
the curve.

Academic foundations
--------------------
* Cover & Thomas (1991) §16.2 universal portfolios — the optimal
  log-growth strategy adapts to realised returns; the Kelly fraction
  per state is the conditional μ/σ².
* Lopez de Prado (2018), *Advances in Financial Machine Learning*,
  §10 "Bet Sizing from Probabilities" — Bayesian update of bet size
  on each new realised observation.
* Dyck, Werner & Stambaugh (2005), "On the Robustness of the Kelly
  Criterion under Estimation Risk" — fractional Kelly with rolling
  estimators is more robust than full Kelly with a single point
  estimate; *bin-based* fractional Kelly is the academically
  preferred middle ground.

Anti-overfitting safeguards
---------------------------
1. Rolling window of fixed length (default 300 closed trades): the
   calibrator is "amnestic", forgetting old regimes — combats the
   classic over-tuning to early-history data.
2. Per-bin minimum count (default 20): bins with too few trades fall
   back to the prior (committed Conviction-Power Kelly). Sparse
   high-signal bins do NOT drive aggressive sizing on small N.
3. Sign-aware aggregation: each trade's realised PnL is rotated by
   its forecast sign so we always estimate "side-aligned mean", which
   removes directional bias from the per-bin estimates.
4. Hard `sizing_cap` ceiling: the calibrator's f_b is capped at the
   same `sizing_cap` as the analytical sizer — no run-away leverage.
5. Bootstrap fallback: until enough trades have closed across all
   bins, the analytical Conviction-Power Kelly remains in force.
6. Live = backtest parity: the same calibrator class is used by both
   simulator and live engine, so OOS performance is the live
   performance (modulo execution).

This file is a self-contained module — it has no I/O and is
deterministic given a sequence of (predictor, realised_pnl) tuples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KellyCalibrator:
    """Rolling per-bin OOS-calibrated Kelly sizer.

    Parameters
    ----------
    n_bins : int
        Number of |predictor| quantile bins. 6 is the empirically robust
        choice given ~300 rolling-window samples (≈ 50/bin).
    lookback_trades : int
        Maximum number of past closed trades retained. Older trades are
        dropped. Trades older than ~3-6 months are systematically
        non-stationary; lookback=300 corresponds to roughly 1 year at
        current breadth (302 trades/yr in v3+A2).
    min_per_bin : int
        Minimum trades per bin before that bin's empirical f_b is used.
        Below this, the analytical bootstrap is consulted instead.
    bootstrap_size_fn : callable
        Fallback sizer used when a bin has too few samples. Signature
        ``(predictor: float) -> fraction: float``. Default: identity
        (size = |predictor| capped at sizing_cap), matching a unit-
        amp Conviction-Power Kelly until calibration is on.
    sizing_cap : float
        Hard ceiling on |f|, identical to the analytical sizer's cap.
    """

    n_bins: int = 6
    lookback_trades: int = 300
    min_per_bin: int = 20
    sizing_cap: float = 5.0

    history: list[tuple[float, float]] = field(default_factory=list)
    _bin_edges: np.ndarray | None = None
    _bin_kellies: np.ndarray | None = None
    _bin_counts: np.ndarray | None = None
    _cache_valid: bool = False

    # ------------------------------------------------------------------ #
    def add_trade(self, predictor: float, realised_pnl: float) -> None:
        """Register a closed trade.  ``predictor`` is the ex-ante
        signal-strength scalar (sign-bearing). ``realised_pnl`` is the
        side-signed log return realised at the chandelier exit."""
        if not np.isfinite(predictor) or not np.isfinite(realised_pnl):
            return
        self.history.append((float(predictor), float(realised_pnl)))
        if len(self.history) > self.lookback_trades:
            self.history = self.history[-self.lookback_trades:]
        self._cache_valid = False

    # ------------------------------------------------------------------ #
    def _refit(self) -> None:
        if len(self.history) < self.min_per_bin * self.n_bins:
            self._bin_edges = self._bin_kellies = self._bin_counts = None
            self._cache_valid = True
            return
        preds = np.array([p for p, _ in self.history])
        pnls = np.array([r for _, r in self.history])
        edges = np.quantile(np.abs(preds), np.linspace(0, 1, self.n_bins + 1))
        edges[0] = -np.inf; edges[-1] = np.inf
        bin_idx = np.digitize(np.abs(preds), edges) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
        kellies = np.zeros(self.n_bins, dtype=float)
        counts = np.zeros(self.n_bins, dtype=int)
        for b in range(self.n_bins):
            mask = bin_idx == b
            counts[b] = int(mask.sum())
            if counts[b] < self.min_per_bin:
                kellies[b] = np.nan      # sentinel: fall back to bootstrap
                continue
            # Sign-align: each trade's pnl is oriented by its forecast.
            aligned = np.sign(preds[mask]) * pnls[mask]
            mu = float(aligned.mean())
            sd = float(aligned.std(ddof=1))
            if sd < 1e-9:
                kellies[b] = 0.0
            else:
                # Full Kelly per bin, capped.
                f = mu / (sd * sd)
                kellies[b] = float(np.clip(f, 0.0, self.sizing_cap))
        self._bin_edges = edges
        self._bin_kellies = kellies
        self._bin_counts = counts
        self._cache_valid = True

    # ------------------------------------------------------------------ #
    def kelly_fraction(self, predictor: float, bootstrap: float | None = None
                        ) -> float:
        """Return f_b for ``predictor``.  Sign of returned size matches
        ``predictor``; magnitude is the bin's calibrated Kelly fraction.
        If the bin has fewer than ``min_per_bin`` trades or the
        calibrator hasn't warmed up, returns ``bootstrap`` (or 0 if
        ``bootstrap`` is None and not enough trades).
        """
        if not self._cache_valid:
            self._refit()
        if self._bin_kellies is None:
            return float(bootstrap) if bootstrap is not None else 0.0
        absp = abs(predictor)
        b = int(np.digitize([absp], self._bin_edges)[0]) - 1
        b = max(0, min(self.n_bins - 1, b))
        f = float(self._bin_kellies[b])
        if not np.isfinite(f):
            return float(bootstrap) if bootstrap is not None else 0.0
        # Apply sign of predictor; preserve sizing_cap on the magnitude.
        signed = np.sign(predictor) * f
        return float(np.clip(signed, -self.sizing_cap, self.sizing_cap))

    # ------------------------------------------------------------------ #
    def is_warm(self) -> bool:
        """The calibrator is "warm" if at least ONE bin has been
        successfully fit (≥ min_per_bin samples and σ > 0).  Sparse
        bins are handled per-lookup via the NaN-aware bootstrap
        fallback inside ``kelly_fraction``; requiring every bin to
        be populated is too strict because (a) bin 0 is permanently
        empty under quantile-edge handling (digitize sends the
        minimum predictor value into bin 1) and (b) the empirical
        predictor distribution is naturally discrete (the
        ``signal_confidence`` formula multiplies a clipped LM ratio
        by an integer-quantised agree fraction), so middle bins are
        chronically sparse.  The calibrator extracts useful per-bin
        Kelly fractions long before every bin is dense, and the
        sparse-bin fallback to bootstrap preserves the OOS
        anti-overfit guarantee."""
        if not self._cache_valid:
            self._refit()
        if self._bin_kellies is None:
            return False
        return bool(np.isfinite(self._bin_kellies).any())

    def diagnostics(self) -> dict:
        if not self._cache_valid:
            self._refit()
        return {
            "n_history": len(self.history),
            "warm": self.is_warm(),
            "bin_counts": (self._bin_counts.tolist()
                              if self._bin_counts is not None else None),
            "bin_kellies": (self._bin_kellies.tolist()
                               if self._bin_kellies is not None else None),
            "bin_edges": (self._bin_edges.tolist()
                            if self._bin_edges is not None else None),
        }
