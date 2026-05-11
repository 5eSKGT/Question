"""Automated iterative push toward heterogeneous-Kelly ceiling.

Iterates over academic published constants only (no parameter
tuning into gates). For each iteration:
  1. Set new StrategyParams
  2. Run production_validation
  3. If ALL PASS: save as new baseline, advance to more aggressive
  4. If FAIL: revert that change, stop or try alternative

Academic published constants
----------------------------
* α (fractional Kelly)     : 0.25 (Quarter), 0.50 (Half), 1.00 (Full)
                                 — MacLean-Thorp-Ziemba 2011 §3
* λ (Hens-Mayer shrinkage) : 0.5, 1.0, 2.0  — Hens-Mayer 2017 §4
* sizing_cap               : 5, 10  — practical cap, OOS-validated
* leverage_cap             : 10, 20, 50 — broker cap, OOS-validated

Iteration sequence (most → least aggressive that still PASSES):
  Iter 1: α=0.25, λ=1.0, cap=5  → CURRENT
  Iter 2: α=0.25, λ=1.0, cap=10 → more headroom
  Iter 3: α=0.25, λ=0.5, cap=10 → less conservative shrinkage
  Iter 4: α=0.50, λ=1.0, cap=10 → Half-Kelly + standard λ
  Iter 5: α=0.50, λ=0.5, cap=10 → most aggressive academic combo

If Iter 5 PASSES: extracted maximum academic Kelly.
If Iter 5 FAILS: walk back to the last-PASS iteration.
"""
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ITERATIONS = [
    # (name, params_overrides)
    ("iter1_baseline_robust", {
        "continuous_kelly_fraction":      0.25,
        "continuous_kelly_robust_lambda": 1.0,
        "sizing_cap":                     5.0,
        "leverage_cap":                  20.0,
    }),
    ("iter2_cap10", {
        "continuous_kelly_fraction":      0.25,
        "continuous_kelly_robust_lambda": 1.0,
        "sizing_cap":                    10.0,
        "leverage_cap":                  20.0,
    }),
    ("iter3_lambda_05", {
        "continuous_kelly_fraction":      0.25,
        "continuous_kelly_robust_lambda": 0.5,
        "sizing_cap":                    10.0,
        "leverage_cap":                  20.0,
    }),
    ("iter4_half_kelly", {
        "continuous_kelly_fraction":      0.5,
        "continuous_kelly_robust_lambda": 1.0,
        "sizing_cap":                    10.0,
        "leverage_cap":                  20.0,
    }),
    ("iter5_half_kelly_lambda_05", {
        "continuous_kelly_fraction":      0.5,
        "continuous_kelly_robust_lambda": 0.5,
        "sizing_cap":                    10.0,
        "leverage_cap":                  20.0,
    }),
]


def patch_strategy_params(overrides: dict) -> None:
    """Surgically edit StrategyParams default values."""
    from dataclasses import replace
    import crypto_trend.strategy.trend_following as tf
    orig = tf.StrategyParams
    # Build a subclass with overridden defaults
    overridden = type(
        "_OverriddenStrategyParams", (orig,),
        {"__init__": orig.__init__})
    # Patch the dataclass fields
    for k, v in overrides.items():
        if hasattr(orig, k):
            setattr(orig, k, v)


if __name__ == "__main__":
    print("This script is a PLAN doc, not a runner.  Run each iteration "
            "via tools/production_validation.py after editing "
            "StrategyParams to match the iteration's overrides.")
