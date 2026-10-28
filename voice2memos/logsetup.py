"""Logging config and excepthooks.

Rotating file handler at ``logs/voice2memos.log`` beside the config, level from
the .cfg (INFO by default), plus a console handler when run from a terminal so
bench.py and the app log the same way. ``sys.excepthook`` and
``threading.excepthook`` both route to the logger and the status line.

Every stage logs start, end, elapsed, and the memo stem; the stem is the
correlation id. DEBUG captures prompts and responses verbatim. WARNING is for
every fallback the user saw in the status line.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

log = logging.getLogger(__name__)

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_NAME = "voice2memos.log"
MAX_BYTES = 2_000_000
BACKUPS = 3
UNEXPECTED = "Unexpected error, see log"


def setup_logging(log_dir: Path, level: str = "INFO", console: bool | None = None) -> Path | None:
    """Install the file handler (and a console one when on a terminal). Returns the log path.

    Safe to call again: handlers from an earlier call are replaced. If the
    folder can't be written, logging stays on the console and the path is None.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "voice2memos", False):
            root.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter(FORMAT)
    if console is None:
        console = sys.stderr is not None and sys.stderr.isatty()
    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        stream.voice2memos = True  # type: ignore[attr-defined]
        root.addHandler(stream)
    path = log_dir / LOG_NAME
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError as e:
        set_level(level)
        log.warning("log file not available at %s: %s", path, e.strerror)
        return None
    file.setFormatter(formatter)
    file.voice2memos = True  # type: ignore[attr-defined]
    root.addHandler(file)
    set_level(level)
    return path


def set_level(level: str) -> None:
    """The root level from the .cfg; applied live when Settings change it."""
    logging.getLogger().setLevel(level.upper())


def install_excepthooks(report: Callable[[str], None]) -> None:
    """Route uncaught exceptions on any thread to the log and to ``report``.

    ``report`` gets the one-line status text. It may be called from any thread,
    so the app's version posts through ``root.after``.
    """

    def on_exception(exc_type, exc, tb) -> None:
        log.error("uncaught exception", exc_info=(exc_type, exc, tb))
        report(UNEXPECTED)

    def on_thread_exception(args: threading.ExceptHookArgs) -> None:
        log.error("uncaught exception in thread %s", args.thread.name if args.thread else "?", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        report(UNEXPECTED)

    sys.excepthook = on_exception
    threading.excepthook = on_thread_exception


@contextmanager
def timed(logger: logging.Logger, stage: str, stem: str) -> Iterator[None]:
    """Log a stage's start, and its end with elapsed seconds, under the memo's stem."""
    logger.info("%s: %s start", stem, stage)
    started = time.perf_counter()
    try:
        yield
    except Exception as e:
        logger.info("%s: %s failed after %.1fs: %s", stem, stage, time.perf_counter() - started, e)
        raise
    logger.info("%s: %s done in %.1fs", stem, stage, time.perf_counter() - started)
