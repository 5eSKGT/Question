"""One-command builder for the AlphaPulse executable.

Usage:
    python build/build_exe.py

Steps:
  1. Regenerate the icon + background (idempotent).
  2. Run PyInstaller against ``build/AlphaPulse.spec``.
  3. Print the resulting binary path.

Works on Windows, macOS and Linux. Requires:
    pip install -r requirements.txt
    pip install pyinstaller
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "build" / "AlphaPulse.spec"


def _run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    # 1. assets
    _run([sys.executable, str(ROOT / "tools" / "generate_assets.py")])

    # 2. PyInstaller
    if shutil.which("pyinstaller") is None:
        print("PyInstaller not found — installing into the current interpreter")
        _run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.6"])

    _run([sys.executable, "-m", "PyInstaller",
          str(SPEC), "--clean", "--noconfirm",
          "--distpath", str(ROOT / "dist"),
          "--workpath", str(ROOT / "build" / "_work")])

    # 3. report
    if sys.platform.startswith("win"):
        binary = ROOT / "dist" / "AlphaPulse.exe"
    else:
        binary = ROOT / "dist" / "AlphaPulse"
    if binary.exists():
        print(f"\n✓ built: {binary}  ({binary.stat().st_size / 1e6:.1f} MB)")
        return 0
    print("✗ build artifact not found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
