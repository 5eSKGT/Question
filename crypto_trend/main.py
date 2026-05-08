"""Backwards-compatible alias for the desktop entry point.

Old launcher (Dash) lives in ``crypto_trend.ui.app`` and can still be invoked
manually for diagnostics. The user-facing executable runs the native GUI.
"""
from __future__ import annotations

from .desktop.app import main

if __name__ == "__main__":
    main()
