"""Loguru-based logging shared by every module."""
from __future__ import annotations

from loguru import logger

from .config import LOG_DIR, SETTINGS

_initialized = False


def get_logger():
    global _initialized
    if not _initialized:
        logger.remove()
        logger.add(
            lambda m: print(m, end=""),
            level=SETTINGS.log_level,
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}",
        )
        logger.add(
            LOG_DIR / "crypto_trend.log",
            level=SETTINGS.log_level,
            rotation="10 MB",
            retention="14 days",
            enqueue=True,
        )
        _initialized = True
    return logger
