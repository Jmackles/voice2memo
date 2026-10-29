"""Settings dialog: fields match the .cfg, secrets never reach the log, hotkeys capture."""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

import pytest

from voice2memos.config import SECRET_FIELDS
from voice2memos.ui.settings import (
    CAPTURE_PROMPT,
    HOTKEY,
    NOTE_HOTKEY,
    PROMPT,
    default_prompt,
    describe_changes,
    format_hotkey,
)

REPO = Path(__file__).resolve().parents[1]
SECRET = "sk-test-0123456789abcdefSECRET"


@pytest.fixture
def dialog(app):
    app.open_settings()
    return app.settings_dialog


def test_every_field_is_a_key_in_the_example_config(dialog):
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(REPO / "voice2memos.example.cfg", encoding="utf-8")
    known = {(section, key) for section in cfg.sections() for key in cfg[section]}
    assert set(dialog.values()) <= known


def test_secret_fields_are_masked_and_never_watched(dialog):
    for field in SECRET_FIELDS:
        widget = dialog.widgets[field]
        assert widget.cget("show")
        assert widget.bind("<KeyPress>") == ""
        assert field not in dialog.watched
        with pytest.raises(ValueError):
            dialog.keylog.watch(widget, "anything")


def test_typed_fields_are_watched(dialog):
    assert {PROMPT, ("tags", "pool"), ("whisper", "glossary"), ("vault", "path"), ("cleanup", "model")} <= dialog.watched


def test_a_saved_secret_never_reaches_the_log(app, dialog, caplog):
    with caplog.at_level(logging.DEBUG):
        dialog.vars[("keys", "openai")].set(SECRET)
        dialog.vars[("cleanup", "model")].set("llama3")
        dialog.save()
        app.open_settings()
        app.settings_dialog.vars[("keys", "openai")].set(SECRET + "2")
        app.settings_dialog.save()
    assert app.config["keys"]["openai"] == SECRET + "2"
    assert "SECRET" not in caplog.text
    assert "setting keys.openai: set" in caplog.messages
    assert "setting keys.openai: changed" in caplog.messages
    assert "setting cleanup.model: 'gemma4:e2b' -> 'llama3'" in caplog.messages
    # Stored in the .cfg by design: the environment is checked first, the config second.
    assert SECRET + "2" in app.config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "old, new, logged",
    [
        ({}, {("cleanup", "model"): "llama3"}, ["cleanup.model: '' -> 'llama3'"]),
        ({("cleanup", "model"): "a"}, {("cleanup", "model"): "a"}, []),
        ({}, {("keys", "openai"): SECRET}, ["keys.openai: set"]),
        ({("keys", "openai"): "x"}, {("keys", "openai"): SECRET}, ["keys.openai: changed"]),
        ({("keys", "openai"): SECRET}, {("keys", "openai"): ""}, ["keys.openai: cleared"]),
        ({}, {("tags", "pool"): "idea\ntodo"}, ["tags.pool: edited, now 2 lines"]),
    ],
)
def test_describe_changes(old, new, logged):
    assert describe_changes(old, new) == logged


def test_saving_keeps_values_for_next_time(app, dialog):
    dialog.vars[("whisper", "model_size")].set("medium")
    dialog.vars[("cleanup", "temperature")].set(0.3)
    dialog.save()
    app.open_settings()
    reopened = app.settings_dialog
    assert reopened is not dialog
    assert reopened.values()[("whisper", "model_size")] == "medium"
    assert reopened.values()[("cleanup", "temperature")] == "0.3"


def test_cancel_saves_nothing(app, dialog):
    before = app.config_path.read_text(encoding="utf-8")
    dialog.vars[("whisper", "model_size")].set("medium")
    dialog.cancel()
    assert not dialog.window.winfo_exists()
    assert app.config["whisper"]["model_size"] == "small"
    assert app.config_path.read_text(encoding="utf-8") == before


def test_opening_twice_keeps_one_dialog(app, dialog):
    app.open_settings()
    assert app.settings_dialog is dialog


def test_equal_hotkeys_are_refused(app, dialog):
    dialog.vars[HOTKEY].set("<f9>")
    dialog.vars[NOTE_HOTKEY].set("<f9>")
    dialog.save()
    assert dialog.window.winfo_exists()
    assert "different keys" in dialog.error.cget("text")
    assert app.config["capture"]["note_hotkey"] == "<f10>"


def test_blank_prompt_shows_the_default_and_saves_blank(dialog):
    text = dialog.texts[PROMPT]
    assert text.get("1.0", "end-1c") == default_prompt()
    assert dialog.values()[PROMPT] == ""
    text.insert("end", "\nAlways use British spelling.")
    assert dialog.values()[PROMPT].endswith("British spelling.")
    dialog.reset_prompt()
    assert dialog.values()[PROMPT] == ""


@pytest.mark.parametrize(
    "keysym, state, hotkey",
    [
        ("F9", 0, "<f9>"),
        ("F12", 0x4, "<ctrl>+<f12>"),
        ("s", 0x4, "<ctrl>+s"),
        ("S", 0x5, "<ctrl>+<shift>+s"),
        ("1", 0x4, "<ctrl>+1"),
        ("space", 0x4, "<ctrl>+<space>"),
        ("Prior", 0, "<page_up>"),
        ("Control_L", 0x4, ""),
        ("exclam", 0x1, ""),
    ],
)
def test_format_hotkey(keysym, state, hotkey):
    assert format_hotkey(keysym, state) == hotkey


def test_hotkey_capture_waits_for_a_real_key_and_escape_restores(dialog):
    var = dialog.vars[HOTKEY]
    var.set("<f9>")

    dialog.start_capture(HOTKEY)
    assert var.get() == CAPTURE_PROMPT
    assert dialog.capture(HOTKEY, "Control_L", 0x4) == "break"
    assert var.get() == CAPTURE_PROMPT
    assert dialog.capture(HOTKEY, "F10", 0x4) == "break"
    assert var.get() == "<ctrl>+<f10>"

    dialog.start_capture(HOTKEY)
    dialog.capture(HOTKEY, "Escape", 0)
    assert var.get() == "<ctrl>+<f10>"

    dialog.start_capture(HOTKEY)
    assert dialog.capture(HOTKEY, "Tab", 0) is None
    dialog.end_capture(HOTKEY)
    assert var.get() == "<ctrl>+<f10>"


@pytest.mark.parametrize("typed, starts_in", [("~/recs", "home"), ("recs", "beside the cfg"), ("", None)])
def test_browse_starts_where_the_folder_field_points(app, dialog, monkeypatch, tmp_path, typed, starts_in):
    from voice2memos.ui import settings

    asked = []
    monkeypatch.setattr(settings.filedialog, "askdirectory", lambda **kw: asked.append(kw["initialdir"]) or "")
    var = dialog.vars[("capture", "save_dir")]
    var.set(typed)
    dialog._browse(var)
    expected = {"home": Path.home() / "recs", "beside the cfg": tmp_path / "recs", None: None}[starts_in]
    assert asked == [expected]
    assert var.get() == typed  # cancelling the folder picker changes nothing
