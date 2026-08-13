"""Process-wide progress logging so long research runs are never silent."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("FEDCRG_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def log_stage(logger: logging.Logger, message: str, **fields: object) -> Generator[None]:
    """Log a start/finish/elapsed pair around one coarse-grained stage."""
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("start %s %s", message, suffix)
    started = time.monotonic()
    try:
        yield
    except Exception:
        logger.exception("failed %s after %.1fs", message, time.monotonic() - started)
        raise
    else:
        logger.info("done %s in %.1fs", message, time.monotonic() - started)
