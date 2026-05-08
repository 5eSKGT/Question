"""Builder for the AlphaPulse executable.

Modes:
    python build/build_exe.py            # default: onedir (fast, recommended)
    python build/build_exe.py --onefile  # single-file .exe (slow, distro-only)

For everyday iteration you do NOT need to run this at all — use the
``run_alphapulse.bat`` launcher (or the desktop shortcut installed by
``tools/install_shortcut.ps1``).  Updates then take effect immediately
after ``git pull``.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_ONEDIR = ROOT / "build" / "AlphaPulse_dir.spec"
SPEC_ONEFILE = ROOT / "build" / "AlphaPulse.spec"


def _run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build AlphaPulse executable.")
    ap.add_argument("--onefile", action="store_true",
                    help="Single-file build (10-15 min, slower startup). "
                         "Default is onedir (1-2 min on rebuild, instant startup).")
    args = ap.parse_args()

    spec = SPEC_ONEFILE if args.onefile else SPEC_ONEDIR
    workdir = ROOT / "build" / ("_work" if args.onefile else "_work_dir")

    # 1. assets
    _run([sys.executable, str(ROOT / "tools" / "generate_assets.py")])

    # 2. PyInstaller
    if shutil.which("pyinstaller") is None:
        print("PyInstaller not found — installing")
        _run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.6"])

    # NOTE we deliberately do NOT pass --clean by default for onedir, so the
    # cache in workdir survives between builds and incremental analyses run
    # in 30-90 seconds. Add --clean=force if you really want a fresh build.
    cmd = [sys.executable, "-m", "PyInstaller", str(spec), "--noconfirm",
           "--distpath", str(ROOT / "dist"),
           "--workpath", str(workdir)]
    if args.onefile:
        cmd.append("--clean")
    _run(cmd)

    # 3. report
    if args.onefile:
        binary = ROOT / "dist" / ("AlphaPulse.exe"
                                   if sys.platform.startswith("win")
                                   else "AlphaPulse")
    else:
        folder = ROOT / "dist" / "AlphaPulse"
        binary = folder / ("AlphaPulse.exe"
                            if sys.platform.startswith("win")
                            else "AlphaPulse")

    if binary.exists():
        size_mb = binary.stat().st_size / 1e6
        print(f"\n✓ built: {binary}  ({size_mb:.1f} MB)")
        if not args.onefile:
            print(f"  (start by double-clicking {binary} — folder must stay together)")
        return 0
    print("✗ build artifact not found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
