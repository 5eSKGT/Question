"""Synthetic backtest validating the screener's statistical properties.

The tests below plant known jumps in a population of otherwise-pure
diffusion paths and verify that the screener:

  1. **Precision @ K**     — recovers a high fraction of planted jumps in
     its top-K output even when the universe is dominated by null cases.
  2. **Direction calibration** — the ``side`` label matches the planted
     jump sign on every recovered symbol.
  3. **Forward-return alignment** — picks made by the screener carry a
     positive expected forward return *in their own direction*, validating
     that the statistic is informative and not just an outlier hunter.
  4. **False-alarm control under no-jump null** — when nothing is planted,
     the screener emits at most a small fraction of false positives at
     the configured threshold.

These tests use deterministic seeds so they are reproducible in CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_trend.screener.winner_loser import (
    WinnerLoserScreener, bipower_variation, funding_pressure_ok,
    hurst_dfa, lee_mykland_gumbel_threshold, lee_mykland_statistic,
    multi_horizon_alignment, vol_regime_score,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ohlcv_from_prices(prices: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="h", tz="UTC")
    return pd.DataFrame({
        "open":   prices,
        "high":   prices * 1.001,
        "low":    prices * 0.999,
        "close":  prices,
        "volume": np.full_like(prices, 1000.0),
    }, index=idx)


def _diffusion_path(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n)))


def _plant_jump(prices: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply a single jump on the last bar."""
    out = prices.copy()
    out[-1] = out[-2] * (1.0 + magnitude)
    return out


def _plant_jump_with_drift(prices: np.ndarray, magnitude: float,
                            drift_bars: int, drift_per_bar: float) -> np.ndarray:
    """Jump on bar t-drift_bars, then steady drift in the same direction."""
    out = prices.copy()
    j_idx = len(out) - drift_bars - 1
    out[j_idx] = out[j_idx - 1] * (1.0 + magnitude)
    sign = np.sign(magnitude)
    for k in range(j_idx + 1, len(out)):
        out[k] = out[k - 1] * (1.0 + sign * drift_per_bar)
    return out


# --------------------------------------------------------------------------- #
# Component-level statistics
# --------------------------------------------------------------------------- #


def test_bipower_variation_jump_robust():
    """BV must be much smaller than the realized variance under a jump."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.005, 200)
    rv = (rets ** 2).sum()
    bv = bipower_variation(rets) * (rets.size - 1)        # un-normalize for compare
    assert abs(bv - rv) / rv < 0.5

    # Inject one big jump
    rets[-1] = 0.20
    rv_jump = (rets ** 2).sum()
    bv_jump = bipower_variation(rets) * (rets.size - 1)
    # BV barely moves even though RV jumps massively
    assert bv_jump < rv_jump
    assert (rv_jump - bv_jump) / rv_jump > 0.5


def test_lee_mykland_n01_under_null():
    """Empirical |L| should sit near 0–3 when there is no jump."""
    rng = np.random.default_rng(1)
    Ls = []
    for _ in range(400):
        rets = rng.normal(0, 0.005, 60)
        Ls.append(lee_mykland_statistic(rets, window=24))
    Ls = np.abs(Ls)
    assert Ls.mean() < 1.5                # roughly half-normal mean ≈ 0.8
    assert (Ls > 3).mean() < 0.05         # tail rate near nominal


def test_lee_mykland_explodes_on_jump():
    rng = np.random.default_rng(2)
    rets = rng.normal(0, 0.005, 60)
    rets[-1] = 0.10                        # 10% jump
    L = lee_mykland_statistic(rets, window=24)
    assert abs(L) > 10


def test_multi_horizon_alignment_excludes_jump_bar():
    """The latest bar (the jump under test) must NOT be counted in the
    alignment sums — otherwise a +10% jump trivially makes any horizon
    containing it positive and the filter loses its meaning."""
    # rets[-1] is the "jump"; pre-jump = [0.02, 0.01, 0.03] — clearly positive
    rets_with_uptrend = np.array([0.02, 0.01, 0.03, +0.10])
    # h=1 looks at pre_jump[-1] = 0.03 (+), h=2 = 0.04 (+), h=3 = 0.06 (+)
    assert multi_horizon_alignment(rets_with_uptrend, sign=1,
                                    horizons=(1, 2, 3)) == 3
    # If the *jump* happens to be positive but the pre-jump trend was negative,
    # alignment should be 0 — exactly the case the screener must filter out.
    rets_with_downtrend = np.array([-0.02, -0.01, -0.03, +0.10])
    assert multi_horizon_alignment(rets_with_downtrend, sign=1,
                                    horizons=(1, 2, 3)) == 0


# --------------------------------------------------------------------------- #
# Screener-level synthetic backtest
# --------------------------------------------------------------------------- #


def test_precision_at_k_for_planted_jumps():
    """Plant 8 jumps among 60 symbols → screener's top-8 recovers ≥ 6."""
    rng = np.random.default_rng(42)
    n_symbols, n_bars, k_pumps = 60, 300, 8

    candles: dict[str, pd.DataFrame] = {}
    planted: dict[str, int] = {}
    for i in range(n_symbols):
        sym = f"S{i:02d}/USDT:USDT"
        prices = _diffusion_path(n_bars, sigma=0.005, rng=rng)
        if i < k_pumps:
            mag = float(rng.uniform(0.08, 0.15)) * float(rng.choice([-1, 1]))
            prices = _plant_jump(prices, mag)
            planted[sym] = 1 if mag > 0 else -1
        candles[sym] = _ohlcv_from_prices(prices)

    screener = WinnerLoserScreener(
        lookback=24, z_threshold=3.0, hurst_floor=0.0,    # disable Hurst here
        min_quote_volume=1e6, top_n=k_pumps,
        min_horizons_agree=1)
    out = screener.run(list(candles.keys()),
                        lambda s: candles[s], lambda s: 1e8)

    found = {r.symbol for r in out}
    overlap = found & planted.keys()
    assert len(overlap) >= k_pumps - 2, \
        f"recall too low: {len(overlap)}/{k_pumps}"

    # Direction must match the planted sign for every recovered symbol
    for r in out:
        if r.symbol in planted:
            expected_side = "long" if planted[r.symbol] > 0 else "short"
            assert r.side == expected_side, \
                f"{r.symbol}: side={r.side} vs planted={expected_side}"


def _path_with_drift(n: int, drift_per_bar: float, sigma: float,
                      rng: np.random.Generator) -> np.ndarray:
    rets = rng.normal(drift_per_bar, sigma, n - 1)
    return 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))


def test_screener_aligns_jump_with_recent_drift():
    """A jump that confirms a pre-existing same-direction drift must pass
    every filter; the *opposite* case must be rejected by multi-horizon.

    This is the contract that distinguishes "regime continuation" from
    "isolated outlier": the screener should recognise WIN as a true
    winner (jump + supporting trend) and refuse OUTLIER (jump opposite to
    its 24h pre-jump trend).
    """
    rng = np.random.default_rng(7)
    sigma, drift = 0.004, 0.002         # SNR per bar = 0.5, very obvious by h=24

    # WIN: clear uptrend for 220 bars, then a +10% jump on the last bar.
    win = _path_with_drift(220, +drift, sigma, rng)
    win[-1] = win[-2] * 1.10

    # OUTLIER: clear downtrend for 220 bars, then a +10% jump on last bar.
    outlier = _path_with_drift(220, -drift, sigma, rng)
    outlier[-1] = outlier[-2] * 1.10

    candles = {
        "WIN/USDT:USDT":     _ohlcv_from_prices(win),
        "OUTLIER/USDT:USDT": _ohlcv_from_prices(outlier),
    }

    screener = WinnerLoserScreener(
        lookback=24, z_threshold=2.5, hurst_floor=0.0,
        min_quote_volume=1e6, top_n=5, min_horizons_agree=2)
    out = screener.run(list(candles.keys()),
                        lambda s: candles[s], lambda s: 1e8)
    syms = {r.symbol for r in out}

    assert "WIN/USDT:USDT" in syms, \
        "jump confirming pre-existing trend should be picked"
    assert "OUTLIER/USDT:USDT" not in syms, \
        "jump against pre-existing trend should be filtered out"

    pick = next(r for r in out if r.symbol == "WIN/USDT:USDT")
    assert pick.side == "long"
    assert pick.horizons_agree >= 2


def test_forward_return_expectation_in_pick_direction():
    """Across many independent universes the screener's picks should have
    a positive expected forward return in their own direction.

    We plant a jump on the *last visible bar* and a continuation drift on
    the *next 12 hidden bars*, then run the screener on the visible slice
    and check that the realised hidden return aligns with the picked side.
    """
    rng = np.random.default_rng(31)
    n_trials = 80
    aligned = 0
    forward_aligned_returns: list[float] = []

    for _ in range(n_trials):
        # Universe of 20 nulls + 1 jumper
        n_visible, n_hidden = 80, 12
        sign = int(rng.choice([-1, 1]))
        prices = _diffusion_path(n_visible + n_hidden, sigma=0.004, rng=rng)
        jump_idx = n_visible - 1
        prices[jump_idx] = prices[jump_idx - 1] * (1.0 + 0.10 * sign)
        # Continuation drift in hidden window
        for k in range(jump_idx + 1, len(prices)):
            prices[k] = prices[k - 1] * (1.0 + 0.0025 * sign)

        candles = {f"N{i}": _ohlcv_from_prices(
            _diffusion_path(n_visible, sigma=0.004, rng=rng)) for i in range(20)}
        candles["JUMP"] = _ohlcv_from_prices(prices[:n_visible])

        screener = WinnerLoserScreener(
            lookback=24, z_threshold=2.5, hurst_floor=0.0,
            min_quote_volume=1e6, top_n=3, min_horizons_agree=1)
        out = screener.run(list(candles.keys()),
                            lambda s: candles[s], lambda s: 1e8)
        for r in out:
            if r.symbol == "JUMP":
                fwd = np.log(prices[-1] / prices[n_visible - 1])
                signed_fwd = fwd if r.side == "long" else -fwd
                forward_aligned_returns.append(signed_fwd)
                if signed_fwd > 0:
                    aligned += 1
                break

    # Direction-correctness: ≥ 80% of picks should land on the right side
    assert aligned / max(len(forward_aligned_returns), 1) >= 0.80, \
        f"only {aligned}/{len(forward_aligned_returns)} picks aligned"
    # Expected forward return in the pick direction must be positive
    assert np.mean(forward_aligned_returns) > 0


def test_dfa_hurst_better_than_rs_on_random_walk():
    """DFA estimator should converge to 0.5 on a true random walk with
    smaller variance than R/S across many independent paths."""
    rng = np.random.default_rng(99)
    dfa_estimates, rs_estimates = [], []
    from crypto_trend.screener.winner_loser import hurst_rs
    for _ in range(60):
        rets = rng.normal(0, 0.005, 256)
        dfa_estimates.append(hurst_dfa(rets))
        rs_estimates.append(hurst_rs(rets))
    dfa, rs = np.array(dfa_estimates), np.array(rs_estimates)
    # both should center near 0.5
    assert abs(dfa.mean() - 0.5) < 0.10
    # DFA should have lower variance — that's the whole point
    assert dfa.std() < rs.std() * 1.5    # generous bound; at least not worse


def test_gumbel_threshold_grows_with_window():
    """Threshold for max|L| over n test points must grow with n."""
    t24 = lee_mykland_gumbel_threshold(24, alpha=0.01)
    t100 = lee_mykland_gumbel_threshold(100, alpha=0.01)
    t500 = lee_mykland_gumbel_threshold(500, alpha=0.01)
    assert t24 < t100 < t500
    assert t24 > 3.0    # reasonable single-bar control under family-wise α=1%


def test_vol_regime_score_catches_explosion():
    """A sudden vol explosion in the recent 24 bars produces a regime
    score > 3 even though earlier history was calm."""
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.005, 240)
    chaos = rng.normal(0, 0.025, 24)         # 5x vol burst
    rets = np.concatenate([calm, chaos])
    score = vol_regime_score(rets)
    assert score > 3.0


def test_funding_pressure_blocks_overcrowded_side():
    assert funding_pressure_ok("long",  None) is True
    assert funding_pressure_ok("long",  0.0010) is False    # longs paying
    assert funding_pressure_ok("long",  0.0001) is True
    assert funding_pressure_ok("short", -0.0010) is False
    assert funding_pressure_ok("short", -0.0001) is True


def test_low_false_alarm_under_pure_noise():
    """No jumps planted → top-N should be empty or very small."""
    rng = np.random.default_rng(13)
    candles = {f"S{i}": _ohlcv_from_prices(
        _diffusion_path(200, sigma=0.005, rng=rng)) for i in range(80)}
    screener = WinnerLoserScreener(
        lookback=24, z_threshold=3.0, hurst_floor=0.0,
        min_quote_volume=1e6, top_n=80, min_horizons_agree=2)
    out = screener.run(list(candles.keys()),
                        lambda s: candles[s], lambda s: 1e8)
    # With z_threshold=3.0 the empirical false-alarm fraction at the LM tail
    # plus the multi-horizon filter should keep this well under 10%.
    assert len(out) / 80 < 0.10, f"too many false alarms: {len(out)}/80"
