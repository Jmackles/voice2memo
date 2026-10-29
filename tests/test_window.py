"""Window shell: every control is there, the stubs log their name, lock, status line, key log."""

from __future__ import annotations

import logging
import re
import time
import tkinter as tk
from tkinter import ttk

import pytest

from voice2memos import __version__
from voice2memos.ui.app import BANNER, ICONS, STATUS_COLORS, App
from voice2memos.ui.panes import KeyLog, describe_key, replace_text

BANNER_NAMES = [name for name, _ in BANNER]
STUB_NAMES = ["open_log"]  # the log viewer is step 13


def key_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "voice2memos.keys"]


def test_title_carries_the_version(app):
    assert app.root.title() == f"voice2memos {__version__}"


def test_banner_buttons_in_order_with_tooltips(app):
    assert list(app.buttons) == BANNER_NAMES
    assert [app.tooltips[name].text for name in BANNER_NAMES] == [
        "Open audio file",
        "Save (right-click for Save as)",
        "Record",
        "Choose vault",
        "Settings",
        "Log",
    ]


@pytest.mark.parametrize("name", STUB_NAMES)
def test_banner_stub_logs_its_name(app, caplog, name):
    with caplog.at_level(logging.INFO):
        app.buttons[name].invoke()
    assert f"stub: {name}" in caplog.messages


def test_save_right_click_menu_offers_save_as(app):
    assert app.save_menu.entrycget(0, "label") == "Save as…"
    app.save_menu.invoke(0)
    assert app.status_label.cget("text") == "Nothing to save"


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


@pytest.mark.parametrize(
    "keysym, char, state, logged",
    [
        ("a", "a", 0, "a"),
        ("A", "A", 0x1, "A"),
        ("space", " ", 0, " "),
        ("BackSpace", "\x08", 0, "⌫"),
        ("Return", "\r", 0, "⏎"),
        ("Left", "", 0, "←"),
        ("z", "\x1a", 0x4, "[Ctrl+Z]"),
        ("S", "\x13", 0x5, "[Ctrl+S]"),
        ("BackSpace", "\x7f", 0x4, "[Ctrl+⌫]"),
        ("at", "@", 0x20004, "@"),  # AltGr on Windows holds Control too
        ("F5", "", 0, "[F5]"),
        ("Shift_L", "", 0, ""),
        ("Control_L", "", 0x4, ""),
    ],
)
def test_describe_key(keysym, char, state, logged):
    assert describe_key(keysym, char, state) == logged


def test_key_log_holds_a_burst_until_flushed(app, caplog):
    keys = KeyLog(app.root)
    with caplog.at_level(logging.INFO):
        for key in "hi there":
            keys.record("note", key)
        assert key_lines(caplog) == []
        keys.flush("note")
    [line] = key_lines(caplog)
    assert re.fullmatch(r'note, \d+\.\ds: "hi there"', line)


def typed(caplog) -> list[str]:
    """Just the quoted keys from each burst line."""
    return [line.split(": ", 1)[1] for line in key_lines(caplog)]


def pending_timers(app) -> tuple[str, ...]:
    """Every scheduled ``after`` except the app's own poll for posted work."""
    return tuple(t for t in app.root.tk.splitlist(app.root.tk.call("after", "info")) if t != app.drain_timer)


def wait_for_key_lines(app, caplog) -> None:
    deadline = time.monotonic() + 2
    while not key_lines(caplog) and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)


def test_key_log_breaks_the_line_at_max_chars(app, caplog):
    keys = KeyLog(app.root, max_chars=5)
    with caplog.at_level(logging.INFO):
        for key in "abcdefgh":
            keys.record("note", key)
        keys.flush_all()
    assert typed(caplog) == ['"abcde"', '"fgh"']
    assert pending_timers(app) == ()


def test_enter_ends_the_line(app, caplog):
    keys = KeyLog(app.root)
    with caplog.at_level(logging.INFO):
        for key in ["h", "i", describe_key("Return", "\r", 0), "x"]:
            keys.record("note", key)
        keys.flush_all()
    assert typed(caplog) == ['"hi⏎"', '"x"']


def test_key_log_posts_after_a_pause(app, caplog):
    keys = KeyLog(app.root, idle_seconds=0.05)
    with caplog.at_level(logging.INFO):
        keys.record("note", "a")
        wait_for_key_lines(app, caplog)
    assert typed(caplog) == ['"a"']
    assert keys.bursts == {}
    assert pending_timers(app) == ()


def test_each_key_restarts_the_pause(app):
    keys = KeyLog(app.root)
    keys.record("note", "a")
    first = keys.bursts["note"].idle
    keys.record("note", "b")
    assert keys.bursts["note"].idle != first
    assert first not in pending_timers(app)


def test_key_log_breaks_the_line_after_max_seconds(app, caplog):
    keys = KeyLog(app.root, max_seconds=0.05)
    with caplog.at_level(logging.INFO):
        keys.record("note", "a")
        wait_for_key_lines(app, caplog)
    assert typed(caplog) == ['"a"']
    assert keys.bursts == {}


def test_click_is_logged_after_any_unfinished_typing(app, caplog):
    replace_text(app.right.text, "hello")
    app.keylog.record("transcript", "t")
    with caplog.at_level(logging.INFO):
        app.right.text.event_generate("<Button-1>", x=1, y=1)
    burst, click = key_lines(caplog)
    assert burst.startswith("transcript, ") and burst.endswith('"t"')
    assert click == "note: clicked at line 1, char 1"


def test_key_log_keeps_fields_apart_and_skips_modifiers(app, caplog):
    keys = KeyLog(app.root)
    with caplog.at_level(logging.INFO):
        keys.record("note", "")
        assert keys.bursts == {}
        keys.record("note", "n")
        keys.record("transcript", "t")
        keys.record("note", "o")
        keys.flush_all()
    assert {line.split(",")[0]: line.split(": ", 1)[1] for line in key_lines(caplog)} == {
        "note": '"no"',
        "transcript": '"t"',
    }


def test_closing_the_window_writes_out_the_last_burst(app, caplog):
    app.keylog.record("note", "x")
    with caplog.at_level(logging.INFO):
        app.close()
    assert typed(caplog) == ['"x"']


def test_key_log_refuses_a_masked_field(app):
    with pytest.raises(ValueError, match="masked"):
        KeyLog(app.root).watch(ttk.Entry(app.root, show="•"), "api key")


def test_saving_a_new_log_level_applies_without_restart(app, caplog, monkeypatch):
    seen = []
    monkeypatch.setattr("voice2memos.logsetup.set_level", seen.append)
    from voice2memos import config as cfg

    values = cfg.values(app.config)
    with caplog.at_level(logging.INFO):
        app.apply_settings(values)
        assert seen == []
        values[("logging", "level")] = "DEBUG"
        app.apply_settings(values)
    assert seen == ["DEBUG"]
    assert "log level now DEBUG" in caplog.messages
    assert app.status_label.cget("text") == "Settings saved"


def test_a_tk_callback_exception_lands_in_the_log_and_status_line(app, caplog):
    with caplog.at_level(logging.ERROR):
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            app.root.report_callback_exception(type(e), e, e.__traceback__)
    assert "RuntimeError: boom" in caplog.text
    assert app.status_label.cget("text") == "Unexpected error, see log"


def test_a_broken_config_shows_in_the_status_line(tmp_path):
    path =tmp_path / "voice2memos.cfg"
    path.write_text("model = llama3\n", encoding="utf-8")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"tkinter has no display: {e}")
    root.withdraw()
    try:
        app = App(root, path)
        assert app.status_label.cget("text") == "Config error on line 1, using defaults"
        assert str(app.status_label.cget("foreground")) == STATUS_COLORS["warning"]
        assert path.with_name("voice2memos.cfg.bad").exists()
    finally:
        root.destroy()


def test_every_click_is_logged_even_when_nothing_follows(app, caplog, monkeypatch):
    from voice2memos.ui import app as app_module

    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: "")
    with caplog.at_level(logging.INFO):
        app.buttons["open_file"].invoke()  # then cancelled
        app.save_menu.invoke(0)
        app.set_status("Cleanup failed", "error", retry=lambda: None)
        app.retry_label.event_generate("<Button-1>")
    assert [m for m in caplog.messages if m.startswith("click: ")] == ["click: open_file", "click: save_as", "click: retry"]


def test_hotkeys_are_logged(fake_app, caplog):
    with caplog.at_level(logging.INFO):
        fake_app.hotkey_pressed()
        fake_app.hotkey_released()
    assert "hotkey: push-to-talk down" in caplog.messages and "hotkey: push-to-talk up" in caplog.messages
