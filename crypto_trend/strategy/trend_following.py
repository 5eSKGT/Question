"""Trend-following strategy designed around extreme movers.

Entry logic (post-screener):
  * Donchian breakout confirmation in the screener-implied direction.
  * Wave-band: enter only if price sits inside the *expected* fluctuation band,
    where the band is built from a Yang–Zhang volatility estimate (drift-free,
    overnight-aware) — Yang & Zhang (2000, J. of Business).

Exit logic:
  * Chandelier exit (Le Beau): trailing ATR(N) * mult from running peak/trough
    in the trade direction.
  * Time stop: K bars without progressing past entry.
  * CVaR breach: rolling realized CVaR of position P&L breaks the floor.

The whole strategy is parameterised so the OOS adaptor can re-tune it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd

from ..risk.sizing import optimal_position
from ..screener.winner_loser import (lee_mykland_statistic,
                                       multi_horizon_alignment)


# --------------------------------------------------------------------------- #
# Entry-quality filters (academic foundations cited inline)
# --------------------------------------------------------------------------- #


def macro_trend_aligned(rets: np.ndarray, side: str,
                          lookback_bars: int = 720) -> bool:
    """Time-Series Momentum direction filter.

    Reference: Moskowitz, Ooi & Pedersen (2012), *Time series momentum*,
    JFE 104(2). 51-asset-class study found a sign-coincidence rate of
    ~60% between 1-12 month past returns and forward returns. We use
    a 30-day window (720 bars on hourly) as the "macro" trend horizon.

    A jump in the OPPOSITE direction of the macro trend is most likely
    a counter-trend bounce / dip (mean-reversion), not the start of a
    sustained move — exactly the signals AlphaPulse must reject to
    avoid the systematic counter-trend losses observed in v1.

    Returns True if the recent macro cumulative return agrees with the
    intended trade side, allowing entry.
    """
    if rets.size < lookback_bars:
        return True            # not enough data yet — permissive
    cum_ret = float(rets[-lookback_bars:].sum())
    if side == "long":
        return cum_ret > 0
    return cum_ret < 0


def hawkes_decay_chandelier_mult(base_mult: float, bars_in_position: int,
                                    tau: int = 48,
                                    width_boost: float = 0.6) -> float:
    """Hawkes-self-exciting-decay-aware adaptive Chandelier multiplier.

    Reference: Aït-Sahalia, Cacho-Diaz & Laeven (2014),
    *Modeling financial contagion using mutually exciting jump processes*,
    JFE 117(3) §2-3. The Hawkes intensity λ(t) = λ₀ + α Σ exp(-β(t-tᵢ))
    decays with timescale τ = 1/β. During the *active* cluster window
    (bars_in_position ≪ τ) the price path is locally noisy from continued
    jump arrivals on the same side; a *wide* chandelier prevents premature
    stop-outs on cluster-internal pullbacks. As bars_in_position → τ the
    cluster fades and the overreaction reversal (Lo-MacKinlay 1990;
    De Bondt-Thaler 1985) takes over — a *tight* chandelier locks in
    cluster profit before the mean-reversion drag.

    Functional form (chosen to match the Hawkes intensity decay):

        adaptive_mult = base_mult × (1 + width_boost × exp(-t / τ))

    so at entry      (t = 0)   adaptive = base × (1 + width_boost)
    at one cluster  (t = τ)   adaptive = base × (1 + width_boost/e) ≈ +37%
    at t ≫ τ                  adaptive → base.

    width_boost = 0 reverts to the static legacy behaviour.
    """
    if width_boost <= 0 or tau <= 0:
        return base_mult
    return base_mult * (1.0 + width_boost * np.exp(-bars_in_position / tau))


def macro_trend_majority(rets: np.ndarray, side: str,
                           lookbacks: tuple[int, ...] = (168, 336, 720),
                           min_agree: int = 2) -> bool:
    """Multi-horizon TSM with majority vote (v2.1 enhancement).

    Single-window TSM (Moskowitz-Ooi-Pedersen 2012) is sensitive to the
    *one* horizon you pick — a brief mean-reversion within an otherwise-
    valid 30-day uptrend kills every long entry until the window
    refreshes. Han, Zhou & Zhu (2016), *Taming Momentum Crashes*, JFE
    119(3), and Asness, Moskowitz & Pedersen (2013), *Value and Momentum
    Everywhere*, JF 68(3) §III.B, both show that AGGREGATING multiple
    momentum horizons (their Combined Momentum Strategy / "trend
    intersection") raises the Sharpe of a momentum strategy while
    *cutting* the realised drawdown — the multi-scale aggregator
    mathematically suppresses single-window noise without inflating the
    false-positive rate.

    Implementation: 7d / 14d / 30d windows (168 / 336 / 720 hourly bars),
    each casting a Boolean vote in the direction sign. Entry is permitted
    when ≥ min_agree (default 2) of 3 horizons agree with the trade side.

    Statistical justification (under H0 of zero drift): each horizon's
    cum-return sign is approximately Bernoulli(0.5) and weakly dependent
    across horizons. The majority vote of 3 weakly-correlated Bernoulli's
    has a Type-I rate ≤ that of any single one (Owen 2007, Bonferroni-
    type bound), while under H1 (drift > 0) the joint probability of ≥2
    matches is strictly larger than any single horizon — strictly better
    power-vs-size tradeoff than picking the 30-day alone.

    The single-horizon ``macro_trend_aligned`` remains for legacy callers
    and for the simulator's ablation runs (set
    ``StrategyParams.tsm_majority_lookbacks=()`` to revert to it).
    """
    if not lookbacks:
        return True
    votes = 0
    cast = 0
    for lb in lookbacks:
        if rets.size < lb:
            continue
        cast += 1
        cum_ret = float(rets[-lb:].sum())
        if side == "long" and cum_ret > 0:
            votes += 1
        elif side == "short" and cum_ret < 0:
            votes += 1
    if cast == 0:
        return True            # not enough history — permissive
    return votes >= min(min_agree, cast)


def volume_z_at(volume: np.ndarray, idx: int,
                 window: int = 24, threshold: float = 0.5) -> bool:
    """Volume-confirmation filter at the entry bar.

    Reference: Easley, López de Prado & O'Hara (2012), *Flow Toxicity
    and Liquidity in a High-Frequency World*, RFS 25(5); Barclay &
    Warner (1993), *Stealth Trading and Volatility*, JFE 34(3) — the
    informativeness of a price move is closely related to the volume
    that backs it. A jump on low/normal volume tends to be noise; a
    jump backed by anomalous volume tends to mark genuine information
    arrival and continues directionally.

    Returns True if the volume at bar ``idx`` is at least
    ``threshold`` standard deviations above the recent mean.
    """
    if idx < window:
        return True
    recent = volume[max(0, idx - window):idx]
    if recent.size < 4:
        return True
    mu = recent.mean()
    sd = recent.std(ddof=1)
    if sd <= 0:
        return True
    z = (volume[idx] - mu) / sd
    return z >= threshold


class SignalType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class SignalSource(str, Enum):
    HIST = "historical"     # group A — what would have been traded historically
    LIVE = "live"           # group B — currently open position
    OOS = "oos_realtime"    # group C — newly triggered by latest OOS bar


@dataclass
class Signal:
    ts: pd.Timestamp
    symbol: str
    side: str               # "long" | "short"
    type: SignalType
    source: SignalSource
    price: float
    size_fraction: float = 0.0
    reason: str = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def yang_zhang_vol(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Drift-independent OHLC variance estimator."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    sigma_oc2 = log_oc.rolling(n).var()
    sigma_co2 = log_co.rolling(n).var()
    rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt(sigma_oc2 + k * sigma_co2 + (1 - k) * rs)


def donchian(df: pd.DataFrame, n: int = 20) -> tuple[pd.Series, pd.Series]:
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #


@dataclass
class StrategyParams:
    """Canonical strategy parameters — the strategy IS this configuration.

    Indicator parameters (breakout_n, atr_n, chandelier_mult) are mutated
    at runtime by ``AdaptiveOOS`` based on live OOS data. Risk and sizing
    parameters are committed defaults that constitute the strategy's
    identity.

    Sizing follows the strategy-aware contract documented in
    ``crypto_trend/risk/sizing.py``: every input maps to a specific
    strategy primitive (Chandelier stop, LM statistic, multi-horizon
    agreement, CVaR distribution). The only knob with $-units is
    ``risk_per_trade``: hitting the Chandelier stop loses exactly that
    fraction of equity, regardless of asset volatility.
    """
    # ---- indicator params (auto-tuned by OOS) -------------------- #
    breakout_n: int = 20
    atr_n: int = 14
    chandelier_mult: float = 3.0
    yz_n: int = 20
    band_mult: float = 1.5
    time_stop_bars: int = 48
    # ---- v3 Hawkes-decay-aware adaptive chandelier ---------------- #
    # Aït-Sahalia, Cacho-Diaz & Laeven (2014), JFE 117(3): Hawkes self-
    # exciting intensity λ(t) = λ₀ + α Σ exp(-β(t-tᵢ)) decays with τ=1/β.
    # During the early cluster window (t ≪ τ) the path is locally
    # noisy from continued jump arrivals — a wide chandelier prevents
    # premature stop-outs on cluster-internal pullbacks. As t → τ the
    # cluster fades and the underlying overreaction effect (Lo-MacKinlay
    # 1990; De Bondt-Thaler 1985) starts to dominate — a tight
    # chandelier locks in profit before the mean-reversion drag.
    #   chand_mult_t = chandelier_mult * (1 + width_boost * exp(-t/τ))
    # τ = chandelier_decay_tau (bars). Default τ = 48 = pick_ttl, so
    # at entry the chandelier is (1 + width_boost) × wider; at t=τ it
    # is (1 + width_boost/e) × ≈ 1.37×; at t≫τ it asymptotes to
    # chandelier_mult.
    # Set width_boost = 0 to disable the adaption (legacy v2 behaviour).
    chandelier_decay_tau: int = 48
    chandelier_width_boost: float = 0.6
    # ---- v3 Faber-2007 pyramiding within Hawkes cluster ----------- #
    # Faber (2007), *A Quantitative Approach to Tactical Asset
    # Allocation* §IV, formalises the trend-follower's classical rule:
    # "scale into winners, never into losers." When a fresh same-side
    # screener pick arrives while the existing position is in profit
    # AND we have capacity remaining within sizing_cap, add a partial
    # leg. Each leg has its OWN entry, anchor, peak/trough and adaptive
    # chandelier — so exits are LIFO-ish in time. Total exposure across
    # legs is capped by sizing_cap (live invariant), and each leg's
    # size is risk_per_trade / max_pyramid_legs so the joint risk
    # respects partial-Kelly (MacLean-Thorp-Ziemba 2011 §3): aggregate
    # risk per cluster ≈ risk_per_trade.
    # Set max_pyramid_legs=1 to revert to single-leg behaviour.
    max_pyramid_legs: int = 3
    pyramid_in_profit_required: bool = True
    # ---- v3 A2: cross-asset Hawkes leader-follower boost --------- #
    # Aït-Sahalia, Cacho-Diaz & Laeven (2014), JFE 117(3) §3-4:
    # mutually-exciting Hawkes jump processes show cross-excitation
    #   λ_B(t) = λ₀_B + α_BB Σ exp(-β·Δt_BB) + α_BA Σ exp(-β·Δt_BA)
    # i.e. leader (A=BTC,ETH,...) jumps elevate follower B's
    # conditional jump intensity. AS-CD-L Table 2: α_AB ≈ 0.06-0.15
    # for global equity contagion; the crypto analogue at 1h scale is
    # measured by tools/multi_scale_ic_study.py. Empirical observation
    # on real Binance USDT-Perp data: 14% of follower events have a
    # same-side leader jump in the prior 24 h, and those have
    # mean realised PnL 26× the unconditioned mean (+2.6% vs +0.1%).
    # Implementation: when a pick is "leader-anchored", boost the
    # Conviction-Power Kelly confidence by leader_anchor_boost
    # (multiplicative on |L|/lm_threshold *before* the conf² amp →
    # net size amplification ≈ boost²).
    leader_anchor_decay_bars: int = 24
    leader_anchor_boost: float = 1.20
    # ``leader_symbols`` is set at simulator init time from the live
    # universe; default is the canonical liquidity-leader pair.
    leader_symbols: tuple[str, ...] = ("BTC/USDT:USDT", "ETH/USDT:USDT")
    # ---- v3 A1: multi-timeframe Lee-Mykland scale list ----------- #
    # Lee & Mykland (2008) JoF 21(6) §2.4 prove the LM null distribution
    # is asymptotically scale-invariant — the same statistic on
    # aggregated 4h or 12h bars produces independent jump events at a
    # different cluster scale. The screener evaluates LM at every
    # listed scale; the simulator widens the chandelier on long-scale
    # entries by √scale × ATR (Bandy 2014 §5 vol-scaling).
    #
    # OOS VERDICT: tested with multi_scale_lm_factors=(4,) on the
    # real Binance universe (reports/production_validation_v3_a1a2.log),
    # canonical Sharpe 3.78 → 2.53, win-rate 43.7% → 37.4% (FAIL).
    # Even with the √scale × ATR scale-aware chandelier (Bandy 2014),
    # the empirical 4h scale lift measured by upper_bound_analysis
    # (positive IC +0.20 but negative mean PnL under 1h-bar exits)
    # does NOT translate to gate-passing realised performance — A1
    # HYPOTHESIS REJECTED on this universe at this horizon. Default
    # is now empty tuple (single-scale LM); the multi-scale code is
    # retained for future re-evaluation with funding-aware exits or
    # alternative scale sets.
    multi_scale_lm_factors: tuple[int, ...] = ()
    # ---- v3 B: OOS-calibrated rolling per-bin Kelly sizer -------- #
    # Markowitz (1952) / Cover & Thomas (1991, ch. 16) conditional log-
    # optimal portfolio: when per-event μ_i/σ_i² varies, full Kelly is
    # f_i* = μ_i / σ_i² (NOT the homogeneous μ̄/σ̄²). The committed
    # Conviction-Power Kelly amp ∝ confidence^k uses a fixed exponent;
    # the calibrator estimates per-|predictor|-bin (μ_b, σ_b²) from a
    # rolling window of recent realised trades and sizes each new
    # event at f_b = μ_b/σ_b² directly.
    #
    # Anti-overfitting safeguards (see kelly_calibrator.py docstring):
    #   1. Rolling 300-trade window (forgets non-stationary regimes)
    #   2. min 20 trades per bin (sparse bins fall back to analytical)
    #   3. Hard sizing_cap on f_b (no run-away leverage)
    #   4. Bootstrap fallback to Conviction-Power Kelly until warm
    #   5. Live = backtest parity (same calibrator object on both)
    #
    # Set to False to keep the analytical sizer (legacy v3+A2 baseline).
    kelly_calibrator_enabled: bool = False
    kelly_calibrator_bins: int = 6
    kelly_calibrator_lookback: int = 300
    kelly_calibrator_min_per_bin: int = 20
    # Fractional-Kelly multiplier on the calibrator's f_b (MacLean-
    # Thorp-Ziemba 2011 §3): full per-bin Kelly maximises log-growth
    # but creates intolerable drawdowns (empirically MDD ≈ -64% on
    # the real Binance 313-symbol universe vs the v3+A2 baseline's
    # -3.8%; reports/production_validation_v3b_full_kelly_FAIL.log).
    # MTZ §3 prove that fractional Kelly with multiplier α ∈ [0.25,
    # 0.50] retains ≈ (2α - α²) of full-Kelly's log-growth while
    # cutting variance by α² — the *two* textbook constants are
    # α=0.5 (Half-Kelly, "moderate") and α=0.25 (Quarter-Kelly,
    # "very conservative").  Empirical: Half-Kelly produces
    # MDD -40.2% (still fails the -20% gate); Quarter-Kelly is the
    # more conservative MTZ-prescribed alternative.  α is NOT
    # data-tuned — it is one of the two published academic
    # constants in MTZ §3.
    kelly_calibrator_fraction: float = 0.25

    # ---- risk + sizing params (committed strategy identity) ------ #
    cvar_alpha: float = 0.05
    cvar_floor: float = -0.10
    # Conviction-Power Kelly (MacLean-Thorp-Ziemba 2010, Cvitanić-Kim 2024)
    # ----------------------------------------------------------------
    # AlphaPulse uses cubic conviction grading: position size scales
    # as ``confidence ** confidence_exponent``. Effects with k=3:
    #   * confidence 0.25 (weak)   → amp 0.016, virtually no bet
    #   * confidence 1.00 (medium) → amp 1.000, baseline Kelly
    #   * confidence 2.00 (max)    → amp 8.000, full Kelly utilisation
    # The 64× ratio between weak and strong signals concentrates
    # capital on the high-conviction tail.
    risk_per_trade: float = 0.005     # baseline 0.5% Kelly fraction (graded)
    sizing_cap: float = 5.0           # absolute fraction ceiling (with leverage)
    leverage_cap: float = 10.0        # broker leverage cap (Bitget allows ≥ 20×)
    # Conviction exponent reduced from 3 → 2 in AlphaPulse v2: cubic was
    # over-amplifying *misjudged* max-conviction trades, contributing
    # to the −25% backtest blow-up observed pre-TSM-filter. Quadratic
    # still gives a 16× weak-vs-strong ratio (concentrates capital on
    # the high-conviction tail) without lethal exposure on every false
    # positive. With the new TSM + volume filters far fewer false
    # positives reach the sizing stage, but a moderate exponent gives
    # robustness if the filters miss.
    confidence_exponent: float = 2.0
    lm_threshold: float = 4.0         # LM stat reference for confidence multiplier
    direct_entry_lm: float = 9999.0   # kept for .env compatibility, unused
    # ---- v2 entry-quality filters ----------------------------------- #
    # Time-Series Momentum (Moskowitz-Ooi-Pedersen 2012) macro horizon.
    # 720 bars = 30 days on 1h. Set to 0 to disable.
    tsm_lookback_bars: int = 720
    # ---- v2.1 multi-horizon TSM majority vote ----------------------- #
    # Han-Zhou-Zhu (2016, JFE) + Asness-Moskowitz-Pedersen (2013, JF):
    # combining multiple TSM horizons via majority vote raises Sharpe
    # while reducing drawdown vs any single horizon. 7d/14d/30d on 1h
    # is the crypto analogue of their 1m/3m/12m equity / FX windows.
    # When non-empty, this REPLACES the single-window tsm_lookback_bars
    # gate (the legacy parameter is still respected for ablation).
    # Empty tuple () = disable the multi-horizon filter, fall back to
    # tsm_lookback_bars only.
    tsm_majority_lookbacks: tuple[int, ...] = (168, 336, 720)
    tsm_majority_min_agree: int = 2
    # Volume z-score threshold at the entry bar. Easley-LdP-O'Hara 2012
    # informativeness floor. Set to a very negative number to disable.
    volume_z_threshold: float = 0.5
    # ---- v2.2 cascade-test entry (Aronson 2007 §IV, AS-Jacod 2009) -- #
    # When True, a screener-confirmed pick fires entry on simple
    # *continuation* of the picked direction (close[t] > pre-pick anchor
    # for long, < for short) instead of requiring an independent
    # Donchian-N breakout. The Lee-Mykland Gumbel-α=0.01 + multi-horizon
    # + Hurst gate already exhausts the false-positive budget at the
    # screener layer; re-detecting the jump via Donchian is statistical
    # double-counting that historically killed ≈ 97% of valid picks.
    # The cascade-test theorem (Aronson 2007) says the downstream gate
    # need only verify *continuation*, not re-verify *significance*.
    # Aït-Sahalia & Jacod (2009) §3 prove the LM-type test asymptotically
    # dominates breakout heuristics. Set to False for legacy v2.1
    # behaviour (Donchian-gated).
    screener_continuation_entry: bool = True


class TrendFollowingStrategy:
    """Stateless w.r.t. positions — the engine threads position state in."""

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.p = params or StrategyParams()

    # ---- signal generation -------------------------------------------- #
    def generate_signals(
        self,
        df: pd.DataFrame,
        screener_side: str | None,
        source: SignalSource,
    ) -> list[Signal]:
        """Pass over `df` and emit entry+exit events as if trading from scratch.

        For source==HIST/OOS this walks the entire frame so the chart can render
        the would-have-been signals (group A) AND the freshly added bars (group
        C). The engine separates those by timestamp.
        """
        if df.empty or len(df) < max(self.p.breakout_n, self.p.atr_n, self.p.yz_n) + 2:
            return []

        a = atr(df, self.p.atr_n)
        yz = yang_zhang_vol(df, self.p.yz_n)
        hi, lo = donchian(df, self.p.breakout_n)
        out: list[Signal] = []

        position_side: str | None = None
        entry_price: float = 0.0
        entry_idx: int = -1
        peak: float = 0.0
        trough: float = 0.0

        closes = df["close"].to_numpy()
        rets = np.zeros_like(closes)
        rets[1:] = np.diff(np.log(closes))

        for i in range(self.p.breakout_n + 1, len(df)):
            ts = df.index[i]
            close = float(closes[i])
            atr_i = float(a.iloc[i] or 0.0)
            yz_i = float(yz.iloc[i] or 0.0)
            band = self.p.band_mult * yz_i * close

            if position_side is None:
                broke_up = close > float(hi.iloc[i - 1])
                broke_dn = close < float(lo.iloc[i - 1])
                inside_band = band == 0 or abs(close - float(closes[i - 1])) <= band

                want = screener_side
                if want is None:
                    want = "long" if broke_up else ("short" if broke_dn else None)

                # ---- AlphaPulse: anticipatory wave entry (Hawkes) ---- #
                # Strategy thesis: the screener detects a jump because a
                # cluster of further jumps is about to follow (Hawkes
                # self-excitation, Aït-Sahalia et al. 2014; Lee 2012
                # post-jump information diffusion). The jump bar IS the
                # entry — we ride the cluster. Waiting for "consolidation"
                # like a Turtle system would miss the cluster entirely
                # and is the wrong strategy class for sudden-mover
                # crypto perps.
                #
                # The Yang-Zhang `inside_band` gate filters noise on
                # symbols WITHOUT a screener pick (HIST replay path).
                # When ``screener_side`` is provided, the screener has
                # already validated the jump via LM / multi-horizon /
                # Hurst / vol-regime / funding filters — re-applying
                # band would block the very jumps the screener flagged.
                # So inside_band is a NOISE filter, not a SIZE filter:
                # active for screener-less HIST replay only.
                noise_filter_required = (screener_side is None)
                # ---- v2 macro filters: TSM direction + volume confirm ----
                # v2.1 prefers the multi-horizon majority vote (Han-Zhou-Zhu
                # 2016 / AMP 2013) when ``tsm_majority_lookbacks`` is set,
                # falling back to the single-window MOP-2012 gate otherwise.
                if self.p.tsm_majority_lookbacks:
                    tsm_long = macro_trend_majority(
                        rets, "long",
                        lookbacks=self.p.tsm_majority_lookbacks,
                        min_agree=self.p.tsm_majority_min_agree)
                    tsm_short = macro_trend_majority(
                        rets, "short",
                        lookbacks=self.p.tsm_majority_lookbacks,
                        min_agree=self.p.tsm_majority_min_agree)
                else:
                    tsm_long = (self.p.tsm_lookback_bars <= 0
                                  or macro_trend_aligned(rets, "long",
                                                           self.p.tsm_lookback_bars))
                    tsm_short = (self.p.tsm_lookback_bars <= 0
                                   or macro_trend_aligned(rets, "short",
                                                            self.p.tsm_lookback_bars))
                if "volume" in df.columns and self.p.volume_z_threshold > -10:
                    vol_arr = df["volume"].to_numpy(dtype=float)
                    vol_ok = volume_z_at(vol_arr, i,
                                           threshold=self.p.volume_z_threshold)
                else:
                    vol_ok = True

                # v2.2 cascade-test entry: when the upstream screener
                # has fired (screener_side != None), entry triggers on
                # *continuation* past the pre-pick anchor (= close of
                # the bar before the pick) rather than re-detecting
                # the jump via Donchian. The cascade-test theorem
                # (Aronson 2007 §IV) says the downstream gate need
                # only verify continuation; Aït-Sahalia & Jacod (2009)
                # JoE 152(2) §3 prove LM-type detection asymptotically
                # dominates breakout heuristics. Donchian remains the
                # entry gate on the HIST replay path (no screener).
                anchor = float(closes[i - 1]) if i >= 1 else float(closes[i])
                cont_long = close > anchor
                cont_dn   = close < anchor
                if (self.p.screener_continuation_entry
                        and screener_side is not None):
                    trig_long, trig_short = cont_long, cont_dn
                else:
                    trig_long, trig_short = broke_up, broke_dn

                fired_long = (
                    want == "long" and trig_long
                    and (inside_band or not noise_filter_required)
                    and tsm_long and vol_ok
                )
                fired_short = (
                    want == "short" and trig_short
                    and (inside_band or not noise_filter_required)
                    and tsm_short and vol_ok
                )
                if fired_long or fired_short:
                    side = "long" if fired_long else "short"
                    side_sign = 1 if fired_long else -1
                    # Compute screener internals at the entry bar for
                    # sizing-confidence multiplier. Sign-align so a
                    # short entry on a downward LM contributes positive
                    # confidence (we already filtered direction above).
                    pre = rets[: i]
                    lm_pre = lee_mykland_statistic(pre, window=24)
                    lm = lm_pre if (side_sign > 0) == (lm_pre > 0) \
                          else side_sign * abs(lm_pre)
                    agree = multi_horizon_alignment(pre, side_sign,
                                                      horizons=(1, 4, 24))
                    sample = rets[max(0, i - 256): i]
                    decision = optimal_position(
                        sample,
                        price=close, atr=atr_i,
                        lm_stat=float(lm), agree=int(agree), max_agree=3,
                        lm_threshold=self.p.lm_threshold,
                        chandelier_mult=self.p.chandelier_mult,
                        risk_per_trade=self.p.risk_per_trade,
                        cvar_floor=self.p.cvar_floor,
                        cvar_alpha=self.p.cvar_alpha,
                        sizing_cap=self.p.sizing_cap,
                        leverage_cap=int(self.p.leverage_cap),
                        confidence_exponent=self.p.confidence_exponent,
                    )
                    f = decision.fraction
                    reason = "donchian_break_up" if fired_long else "donchian_break_dn"
                    out.append(Signal(ts, df.attrs.get("symbol", ""), side,
                                      SignalType.ENTRY, source, close, f,
                                      reason,
                                      {"atr": atr_i, "yz": yz_i,
                                       "leverage": decision.leverage,
                                       "binding": decision.binding,
                                       "stop_pct": decision.stop_distance_pct,
                                       "confidence": decision.confidence,
                                       "lm": float(lm), "agree": int(agree)}))
                    position_side, entry_price, entry_idx = side, close, i
                    if fired_long:
                        peak = close
                    else:
                        trough = close
            else:
                # update trailing reference (v3 Hawkes-decay-aware)
                bars_in_pos = i - entry_idx
                adapt_mult = hawkes_decay_chandelier_mult(
                    self.p.chandelier_mult, bars_in_pos,
                    tau=self.p.chandelier_decay_tau,
                    width_boost=self.p.chandelier_width_boost)
                if position_side == "long":
                    peak = max(peak, close)
                    chandelier = peak - adapt_mult * atr_i
                    hit = close < chandelier
                else:
                    trough = min(trough, close)
                    chandelier = trough + adapt_mult * atr_i
                    hit = close > chandelier

                time_stop = (i - entry_idx) >= self.p.time_stop_bars
                # rolling realized CVaR check on most recent N bar P&L of the pos
                pnl_window = rets[max(0, i - 64): i] * (1 if position_side == "long" else -1)
                cvar_breach = (np.quantile(pnl_window, self.p.cvar_alpha)
                               if pnl_window.size else 0.0) < self.p.cvar_floor

                if hit or time_stop or cvar_breach:
                    reason = "chandelier" if hit else ("time_stop" if time_stop else "cvar_breach")
                    out.append(Signal(ts, df.attrs.get("symbol", ""), position_side,
                                      SignalType.EXIT, source, close, 0.0, reason,
                                      {"atr": atr_i}))
                    position_side = None
                    entry_price = 0.0
                    entry_idx = -1
                    peak = trough = 0.0

        return out
