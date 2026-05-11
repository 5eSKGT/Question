"""Wrapper to run production_validation with patched StrategyParams defaults.

Used by the parallel iter_push driver to spawn multiple validation
runs with different parameter overrides simultaneously.

Usage:
    python tools/run_validation_with_params.py --params '<json>' --out <log>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=str, required=True,
                    help="JSON dict of StrategyParams field overrides")
    p.add_argument("--out",    type=str, required=True,
                    help="output log file")
    p.add_argument("--seeds",  type=int, default=1)
    args = p.parse_args()

    overrides = json.loads(args.params)

    # Patch StrategyParams defaults BEFORE importing production_validation.
    from dataclasses import fields
    import crypto_trend.strategy.trend_following as tf
    valid = {f.name for f in fields(tf.StrategyParams)}
    for k, v in overrides.items():
        if k not in valid:
            print(f"⚠ unknown param: {k}", file=sys.stderr); continue
        # Patch the dataclass field default for new instances.
        tf.StrategyParams.__dataclass_fields__[k].default = v
        setattr(tf.StrategyParams, k, v)

    # Redirect stdout to the log file
    log_fh = open(args.out, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    # Now import + run production_validation
    sys.argv = ["production_validation.py",
                 "--seeds", str(args.seeds),
                 "--data-source", "real"]
    from tools import production_validation
    return production_validation.main()


if __name__ == "__main__":
    sys.exit(main())
