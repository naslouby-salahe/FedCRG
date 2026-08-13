"""Structured contextual logging persisted under ``outputs/logs/``."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(logs_root: Path | None = None, level: str | None = None) -> None:
    """Configure root logging with a stderr console handler and an optional file handler.

    The file handler appends to ``outputs/logs/fedcrg.log`` so long-running research
    workloads persist structured logs without depending on the console.
    """
    resolved_level = (level or os.environ.get("FEDCRG_LOG_LEVEL", "INFO")).upper()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logs_root is not None:
        logs_root.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_root / "fedcrg.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        handlers.append(file_handler)
    logging.basicConfig(
        level=resolved_level,
        format=_FORMAT,
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


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
