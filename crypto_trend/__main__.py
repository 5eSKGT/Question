"""``python -m crypto_trend`` → launches the AlphaPulse desktop GUI."""
from __future__ import annotations

import sys

from .desktop.app import main

if __name__ == "__main__":
    sys.exit(main())
