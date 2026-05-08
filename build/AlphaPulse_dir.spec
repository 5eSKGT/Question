# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — ONEDIR variant.

Difference vs AlphaPulse.spec (onefile):
  * Output is ``dist/AlphaPulse/`` (a folder containing the .exe + DLLs).
  * Cold launch is INSTANT — no self-extraction at startup.
  * Incremental rebuilds reuse the cache in ``build/_work_dir`` and
    typically finish in 30–90 seconds rather than 10–15 minutes,
    because PyInstaller only re-analyzes what changed.

Build:
    python build\\build_exe.py --dir
or directly:
    pyinstaller build\\AlphaPulse_dir.spec --clean --noconfirm \\
        --distpath dist --workpath build/_work_dir

For day-to-day code edits prefer the .bat / shortcut launcher — onedir
is for when you want to ship the app to someone who doesn't have Python.
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(os.getcwd()).resolve()
ASSETS = ROOT / "crypto_trend" / "desktop" / "assets"
DOCS = ROOT / "crypto_trend" / "desktop" / "docs"

block_cipher = None

datas = [
    (str(ASSETS / "icon.png"),               "crypto_trend/desktop/assets"),
    (str(ASSETS / "icon.ico"),               "crypto_trend/desktop/assets"),
    (str(ASSETS / "background.png"),         "crypto_trend/desktop/assets"),
    (str(DOCS / "chart_signals.html"),       "crypto_trend/desktop/docs"),
    (str(DOCS / "engine_pipeline.html"),     "crypto_trend/desktop/docs"),
]
datas += collect_data_files("ccxt")

hiddenimports = (
    collect_submodules("ccxt")
    + collect_submodules("plotly")
    + collect_submodules("PySide6.QtWebEngineWidgets")
    + collect_submodules("keyring")
    + ["scipy.special.cython_special"]
)

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "notebook",
              "jupyter", "PyQt5", "PyQt6"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Note exclude_binaries=True — that's the onedir signal.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AlphaPulse",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(ASSETS / "icon.ico"),
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AlphaPulse",
)
