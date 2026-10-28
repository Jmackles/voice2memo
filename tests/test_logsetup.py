"""Rotating log file beside the config, live level, excepthooks reaching the status line."""

from __future__ import annotations

import logging
import sys
import threading

import pytest

from voice2memos import logsetup


@pytest.fixture
def clean_root():
    root = logging.getLogger()
    before = (list(root.handlers), root.level, sys.excepthook, threading.excepthook)
    yield
    for handler in list(root.handlers):
        if getattr(handler, "voice2memos", False):
            root.removeHandler(handler)
            handler.close()
    root.handlers[:], root.level, sys.excepthook, threading.excepthook = before[0], before[1], before[2], before[3]


def test_setup_writes_a_file_in_the_log_folder(tmp_path, clean_root):
    path = logsetup.setup_logging(tmp_path / "logs", "DEBUG", console=False)
    logging.getLogger("voice2memos.test").debug("hello file")
    assert path == tmp_path / "logs" / "voice2memos.log"
    assert "DEBUG voice2memos.test: hello file" in path.read_text(encoding="utf-8")
    assert logging.getLogger().level == logging.DEBUG


def test_setup_twice_keeps_one_handler_set(tmp_path, clean_root):
    logsetup.setup_logging(tmp_path / "logs", console=False)
    logsetup.setup_logging(tmp_path / "logs", console=False)
    ours = [h for h in logging.getLogger().handlers if getattr(h, "voice2memos", False)]
    assert len(ours) == 1


def test_an_unwritable_log_folder_falls_back_to_the_console(tmp_path, clean_root, caplog):
    blocker = tmp_path / "logs"
    blocker.write_text("not a folder", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert logsetup.setup_logging(blocker, console=False) is None
    assert "log file not available" in caplog.text


def test_set_level_applies_live(clean_root):
    logsetup.set_level("warning")
    assert logging.getLogger().level == logging.WARNING


def test_excepthooks_log_and_report(clean_root, caplog):
    reports = []
    logsetup.install_excepthooks(reports.append)
    with caplog.at_level(logging.ERROR):
        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        thread = threading.Thread(target=lambda: 1 / 0, name="worker")
        thread.start()
        thread.join()
    assert reports == [logsetup.UNEXPECTED] * 2
    assert "ValueError: boom" in caplog.text
    assert "uncaught exception in thread worker" in caplog.text
    assert "ZeroDivisionError" in caplog.text


def test_timed_logs_start_end_and_failure(caplog):
    log = logging.getLogger("voice2memos.test")
    with caplog.at_level(logging.INFO):
        with logsetup.timed(log, "cleanup", "20261028-1726"):
            pass
        with pytest.raises(RuntimeError), logsetup.timed(log, "entities", "20261028-1726"):
            raise RuntimeError("nope")
    assert caplog.messages[0] == "20261028-1726: cleanup start"
    assert caplog.messages[1].startswith("20261028-1726: cleanup done in 0.")
    assert caplog.messages[3].startswith("20261028-1726: entities failed after 0.") and caplog.messages[3].endswith(": nope")
