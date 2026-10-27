"""Window shell: every control is there, the stubs log their name, lock and status line work."""

from __future__ import annotations

import logging
import tkinter as tk

import pytest

from voice2memos import __version__
from voice2memos.ui.app import BANNER, ICONS, STATUS_COLORS, App
from voice2memos.ui.panes import replace_text

BANNER_NAMES = [name for name, _ in BANNER]


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"tkinter has no display: {e}")
    root.withdraw()
    yield App(root)
    root.destroy()


def test_title_carries_the_version(app):
    assert app.root.title() == f"voice2memos {__version__}"


def test_banner_buttons_in_order_with_tooltips(app):
    assert list(app.buttons) == BANNER_NAMES
    assert [app.tooltips[name].text for name in BANNER_NAMES] == [
        "Open audio file",
        "Save (right-click for Save as)",
        "Record",
        "Discard",
        "Choose vault",
        "Settings",
        "Log",
    ]


@pytest.mark.parametrize("name", BANNER_NAMES)
def test_banner_stub_logs_its_name(app, caplog, name):
    with caplog.at_level(logging.INFO):
        app.buttons[name].invoke()
    assert f"stub: {name}" in caplog.messages


def test_save_right_click_menu_offers_save_as(app, caplog):
    assert app.save_menu.entrycget(0, "label") == "Save as…"
    with caplog.at_level(logging.INFO):
        app.save_menu.invoke(0)
    assert "stub: save_as" in caplog.messages


def test_left_pane_starts_locked_and_toggles(app):
    text = app.left.text
    assert text.cget("state") == "disabled"

    app.left.toggle.invoke()
    assert text.cget("state") == "normal"
    assert app.left.toggle.cget("text") == ICONS["unlocked"]
    assert text.cget("background") != app.right.text.cget("background")

    app.left.toggle.invoke()
    assert text.cget("state") == "disabled"
    assert app.left.toggle.cget("text") == ICONS["locked"]
    assert text.cget("background") == app.right.text.cget("background")


def test_replace_text_writes_through_the_lock(app):
    replace_text(app.left.text, "first")
    replace_text(app.left.text, "second")
    assert app.left.text.get("1.0", "end-1c") == "second"
    assert app.left.text.cget("state") == "disabled"


def test_right_pane_is_editable_and_starts_in_paint_mode(app):
    assert app.right.text.cget("state") == "normal"
    assert app.paint_mode
    app.right.toggle.invoke()
    assert not app.paint_mode
    assert app.right.toggle.cget("text") == ICONS["type"]
    app.right.toggle.invoke()
    assert app.right.toggle.cget("text") == ICONS["paint"]


@pytest.mark.parametrize("level, log_level", [("info", logging.INFO), ("warning", logging.WARNING), ("error", logging.ERROR)])
def test_status_is_colored_by_level_and_logged(app, caplog, level, log_level):
    with caplog.at_level(logging.INFO):
        app.set_status("Ollama not running", level)
    assert app.status_label.cget("text") == "Ollama not running"
    assert str(app.status_label.cget("foreground")) == STATUS_COLORS[level]
    assert (caplog.records[-1].levelno, caplog.records[-1].getMessage()) == (log_level, "status: Ollama not running")


def test_status_history_keeps_the_last_ten(app):
    for i in range(15):
        app.set_status(f"message {i}")
    assert [text for _, _, text in app.status_history] == [f"message {i}" for i in range(5, 15)]


def test_retry_link_shows_only_with_a_callback(app):
    calls = []
    app.set_status("Cleanup failed", "error", retry=lambda: calls.append("retried"))
    assert app.retry_label.winfo_manager() == "pack"
    app.retry_label.event_generate("<Button-1>")
    assert calls == ["retried"]

    app.set_status("Ready")
    assert app.retry_label.winfo_manager() == ""


def test_clicking_the_status_lists_the_history(app):
    app.set_status("one")
    app.set_status("two", "error")
    app.status_label.event_generate("<Button-1>")
    body = app.history_window.winfo_children()[0]
    lines = [label.cget("text").split("  ", 1)[1] for label in body.winfo_children()]
    assert lines == ["Ready", "one", "two"]
