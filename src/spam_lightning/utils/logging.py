"""Logging helpers for consistent script output."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure a shared console logger for the demo scripts."""

    logger = logging.getLogger("spam_lightning")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
