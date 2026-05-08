# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AlphaPulse — build with:

    pyinstaller build/AlphaPulse.spec --clean --noconfirm

The result is one self-contained executable:
    dist/AlphaPulse.exe          (Windows)
    dist/AlphaPulse              (macOS/Linux)
The window title is set inside the application, the executable's icon
is the bundled assets/icon.ico, and the resources/assets needed at
runtime are folded into the binary via PyInstaller's ``datas``.
"""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

ROOT = Path(os.getcwd()).resolve()
ASSETS = ROOT / "crypto_trend" / "desktop" / "assets"

block_cipher = None

datas = [
    (str(ASSETS / "icon.png"),        "crypto_trend/desktop/assets"),
    (str(ASSETS / "icon.ico"),        "crypto_trend/desktop/assets"),
    (str(ASSETS / "background.png"),  "crypto_trend/desktop/assets"),
]
datas += collect_data_files("ccxt")           # ccxt ships symbol metadata

hiddenimports = (
    collect_submodules("ccxt")
    + collect_submodules("plotly")
    + collect_submodules("PySide6.QtWebEngineWidgets")
    + ["scipy.special.cython_special"]
)

a = Analysis(
    ["../crypto_trend/desktop/app.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "notebook",
        "jupyter",
        "PyQt5", "PyQt6",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AlphaPulse",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                       # GUI app → no console window
    icon=str(ASSETS / "icon.ico"),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
