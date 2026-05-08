"""Top-level launcher for the AlphaPulse executable.

PyInstaller treats the entry-point script as ``__main__`` and strips its
package context. That is fatal for a script that uses relative imports
(``from .main_window import ...``), which is why bundling
``crypto_trend/desktop/app.py`` directly produced::

    ImportError: attempted relative import with no known parent package

This module sits at the project root and uses *absolute* imports — so
PyInstaller can run it as ``__main__`` while still resolving the rest of
the codebase through the bundled ``crypto_trend`` package.
"""
from __future__ import annotations

import sys


def main() -> int:
    # Absolute import — works whether we are bundled by PyInstaller or run
    # directly from a checkout (``python launcher.py``).
    from crypto_trend.desktop.app import main as _main
    return _main()


if __name__ == "__main__":
    sys.exit(main())
