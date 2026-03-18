"""Logging helpers for the teleoperation data pipeline."""

from __future__ import annotations

import logging
import os


def configure_data_logger() -> logging.Logger:
    """Configure the shared pipeline logger once and return it."""

    logger = logging.getLogger("deoxys.data")
    if logger.handlers:
        return logger

    level_name = os.environ.get("DEOXYS_DATA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Deoxys Data %(levelname)s] %(message)s"))

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_data_logger(name: str) -> logging.Logger:
    """Return a child logger of the shared pipeline logger."""

    return configure_data_logger().getChild(name)
