"""Simple logging helper wrapping Python's logging module."""

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure root logger with optional file handler."""

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
