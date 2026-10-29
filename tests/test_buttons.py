"""Every button does what it says, against the fakes: record, open, discard, vault, save."""

from __future__ import annotations

import logging
import math
import shutil
import tkinter as tk
import wave
from pathlib import Path

import pytest

from voice2memos import config, note
from voice2memos.ui import app as app_module
from voice2memos.ui.app import ICONS

from .conftest import FAKE_CFG, record_one

EXPECTED_NOTE = "I was thinking about the database thing again.\n\nMaybe Bookinnit should just have fields instead of tables."
EXPECTED_RAW = (
    "um so I was thinking about the uh database thing again\n\nlike maybe Bookinnit should just have fields instead of tables you know"
)


def pane(app, side: str) -> str:
    return getattr(app, side).text.get("1.0", "end-1c")


# --- record ------------------------------------------------------------------


def test_record_writes_the_wav_then_fills_both_panes(fake_app, finish, caplog):
    with caplog.at_level(logging.INFO):
        memo = record_one(fake_app, finish)
    assert memo.folder.parent == fake_app.save_dir
    assert sorted(p.name for p in memo.folder.iterdir()) == ["cleaned.json", "entities.json", "memo.wav", "segments.json", "state.json", "tags.json"]
    assert pane(fake_app, "left") == EXPECTED_RAW
    assert pane(fake_app, "right") == EXPECTED_NOTE
    assert memo.entities == ["database", "Bookinnit", "fields", "tables"] and memo.tags == ["idea", "bookinnit"]
    assert fake_app.status_label.cget("text") == "Ready"
    assert fake_app.buttons["record"].cget("text") == ICONS["record"]
    assert not fake_app.dirty and fake_app.root.title() == f"voice2memos {app_module.__version__}"
    stem = memo.stem
    for stage in ("transcribe (fake whisper)", "cleanup (fake)", "entities (fake)", "tags (fake)"):
        assert f"{stem}: {stage} done in" in caplog.text


def test_buttons_are_disabled_while_the_worker_runs(fake_app, finish):
    fake_app.buttons["record"].invoke()
    assert fake_app.buttons["record"].cget("text") == ICONS["stop"]
    fake_app.buttons["record"].invoke()
    assert fake_app.busy
    busy = [fake_app.buttons["open_file"], fake_app.buttons["record"], fake_app.player.discard_button]
    assert all(button.instate(["disabled"]) for button in busy)
    finish(fake_app)
    assert not any(button.instate(["disabled"]) for button in busy)


def test_a_dead_microphone_shows_in_the_status_line(app_factory):
    app = app_factory(FAKE_CFG + "fail = capture\n")
    app.buttons["record"].invoke()
    assert not app.recording
    assert app.status_label.cget("text") == "Microphone unavailable"


def test_hotkey_press_and_release_record_too(fake_app, finish):
    fake_app.hotkey_pressed()
    assert fake_app.recording
    fake_app.hotkey_released()
    finish(fake_app)
    assert fake_app.memo is not None and pane(fake_app, "right") == EXPECTED_NOTE


def test_hotkeys_are_bound_from_the_config_and_rebound_on_save(fake_app, no_global_hotkeys):
    [hotkeys] = no_global_hotkeys.instances
    assert hotkeys.binds == [("<f9>", "<f10>")]
    values = config.values(fake_app.config)
    values[("capture", "hotkey")] = "<ctrl>+<f9>"
    fake_app.apply_settings(values)
    assert hotkeys.binds[-1] == ("<ctrl>+<f9>", "<f10>")
    values[("whisper", "model_size")] = "small"
    fake_app.apply_settings(values)
    assert len(hotkeys.binds) == 2  # nothing hotkey-related changed
    fake_app.close()
    assert hotkeys.stopped


# --- degrade and retry --------------------------------------------------------


def test_cleanup_failure_shows_raw_text_with_a_retry_that_reruns_only_cleanup(app_factory, finish, caplog):
    app = app_factory(FAKE_CFG + "fail = cleanup\n")
    with caplog.at_level(logging.INFO):
        record_one(app, finish)
    assert app.status_label.cget("text") == "Cleanup failed, showing original"
    assert app.retry_label.winfo_manager() == "pack"
    assert pane(app, "right") == EXPECTED_RAW  # raw, a paragraph per segment, like the Transcript
    assert app.memo.entities == ["database", "Bookinnit", "fields", "tables"]  # later stages still ran

    app.config["fake"]["fail"] = ""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        app.retry_label.event_generate("<Button-1>")
        finish(app)
    assert pane(app, "right") == EXPECTED_NOTE
    assert app.status_label.cget("text") == "Ready"
    assert "transcribe (fake whisper) start" not in caplog.text and "cleanup (fake) start" in caplog.text


def test_transcription_failure_keeps_the_wav_and_stops_there(app_factory, finish):
    app = app_factory(FAKE_CFG + "fail = transcribe\n")
    memo = record_one(app, finish)
    assert app.status_label.cget("text") == "Transcription failed"
    assert memo.wav.exists() and not (memo.folder / "segments.json").exists()
    assert pane(app, "left") == "" and app.retry_callback is not None


def test_the_default_config_degrades_to_raw_text_until_the_models_exist(app_factory, finish):
    app = app_factory("[capture]\ninput_device = fake\n[whisper]\nmodel_size = fake\n[fake]\nfixture = tests/fixtures/sample\n")
    record_one(app, finish)
    assert pane(app, "left") == EXPECTED_RAW
    assert app.status_label.cget("text") in ("Cleanup failed, showing original", "Entities failed", "Tags failed")
    assert app.memo.entities == [] and app.memo.tags == []


def test_a_crash_in_the_worker_lands_in_the_status_line(fake_app, finish, monkeypatch, caplog):
    monkeypatch.setattr(app_module.transcribe, "make_transcriber", lambda *_: 1 / 0)
    with caplog.at_level(logging.ERROR):
        record_one(fake_app, finish)
    assert fake_app.status_label.cget("text") == "Unexpected error, see log"
    assert "pipeline crashed" in caplog.text and "ZeroDivisionError" in caplog.text
    assert not fake_app.busy


# --- open file ------------------------------------------------------------------


def test_open_file_copies_the_wav_and_remembers_the_folder(fake_app, finish, monkeypatch, fixtures_dir, tmp_path):
    source = tmp_path / "elsewhere" / "talk.wav"
    source.parent.mkdir()
    shutil.copy(fixtures_dir / "sample" / "memo.wav", source)
    asked = []
    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: asked.append(kw) or str(source))
    fake_app.buttons["open_file"].invoke()
    finish(fake_app)
    assert fake_app.memo.wav.read_bytes() == source.read_bytes() and source.exists()
    assert fake_app.memo.source == source and note.Memo.open(fake_app.memo.folder).source == source
    assert pane(fake_app, "right") == EXPECTED_NOTE
    assert config.load(fake_app.config_path)[0]["capture"]["open_dir"] == str(source.parent)
    assert asked[0]["initialdir"] == str(Path.home())
    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: asked.append(kw) or "")
    fake_app.buttons["open_file"].invoke()
    assert asked[1]["initialdir"] == str(source.parent)


def test_other_audio_is_decoded_on_the_worker_and_the_original_left_alone(fake_app, finish, monkeypatch, tmp_path, caplog):
    # An 8-bit stereo WAV stands in for an mp3: PyAV decodes both, and the test needs no encoder.
    source = tmp_path / "odd.wav"
    with wave.open(str(source), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(1)
        out.setframerate(22050)
        out.writeframes(bytes(128 + int(100 * math.sin(i / 10)) for i in range(22050 * 2)))
    before = source.read_bytes()
    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: str(source))
    with caplog.at_level(logging.INFO):
        fake_app.buttons["open_file"].invoke()
        finish(fake_app)
    with wave.open(str(fake_app.memo.wav)) as decoded:
        assert (decoded.getnchannels(), decoded.getsampwidth(), decoded.getframerate()) == (1, 2, 22050)
    assert source.read_bytes() == before
    assert f"{fake_app.memo.stem}: decode done in" in caplog.text
    assert fake_app.player.playback.path == fake_app.memo.wav
    assert pane(fake_app, "right") == EXPECTED_NOTE


def test_unreadable_audio_says_so_and_retry_tries_the_decode_again(fake_app, finish, monkeypatch, tmp_path):
    source = tmp_path / "song.mp3"
    source.write_bytes(b"not audio at all")
    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: str(source))
    fake_app.buttons["open_file"].invoke()
    finish(fake_app)
    assert fake_app.status_label.cget("text") == "Can't read that audio file"
    assert not fake_app.memo.wav.exists() and not fake_app.player.playback.total
    assert source.read_bytes() == b"not audio at all"
    fake_app.retry_label.event_generate("<Button-1>")
    finish(fake_app)
    assert fake_app.status_label.cget("text") == "Can't read that audio file"


# --- save ----------------------------------------------------------------------


def test_first_save_asks_where_and_renders_the_note_into_the_vault(fake_app, finish, monkeypatch, vault, fixtures_dir):
    fake_app.config["vault"]["path"] = str(vault)
    memo = record_one(fake_app, finish)
    asked = []
    target = vault / f"{memo.stem}.md"
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: asked.append(kw) or str(target))
    fake_app.buttons["save"].invoke()
    assert asked[0]["initialdir"] == vault and asked[0]["initialfile"] == f"{memo.stem}.md"
    text = target.read_text(encoding="utf-8")
    expected = (fixtures_dir / "sample" / "expected.md").read_text(encoding="utf-8").replace("sample", memo.stem)
    assert text.splitlines()[2:] == expected.splitlines()[2:]  # all but the date line
    assert (vault / "attachments" / f"{memo.stem}.wav").exists()
    assert (memo.folder / "note.md").read_text(encoding="utf-8") == text
    assert note.Memo.open(memo.folder).saved_to == target
    assert fake_app.status_label.cget("text") == f"Saved {target.name}"


def test_second_save_of_an_edited_note_writes_in_place_and_a_clean_one_opens_it(fake_app, finish, monkeypatch, vault):
    memo = record_one(fake_app, finish)
    target = vault / "n.md"
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: str(target))
    fake_app.buttons["save"].invoke()

    fake_app.right.text.insert("end", " Extra.")
    fake_app.root.update()
    assert fake_app.dirty and fake_app.root.title().endswith("*")
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: pytest.fail("asked again"))
    fake_app.buttons["save"].invoke()
    assert "[[tables]]. Extra. ^" in target.read_text(encoding="utf-8")
    assert not fake_app.dirty

    opened = []
    monkeypatch.setattr(app_module.note, "open_note", lambda path, registered: opened.append((path, registered)) or "Opened in Obsidian")
    fake_app.buttons["save"].invoke()
    assert opened == [(target, False)]
    assert fake_app.status_label.cget("text") == "Opened in Obsidian"


def test_save_as_always_asks(fake_app, finish, monkeypatch, vault):
    record_one(fake_app, finish)
    targets = iter([vault / "a.md", vault / "b.md"])
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: str(next(targets)))
    fake_app.buttons["save"].invoke()
    fake_app.save_menu.invoke(0)
    assert (vault / "a.md").exists() and (vault / "b.md").exists()
    assert fake_app.memo.saved_to == vault / "b.md"


def test_a_failed_save_keeps_the_note_dirty(fake_app, finish, monkeypatch, tmp_path):
    record_one(fake_app, finish)
    fake_app.right.text.insert("end", "!")
    fake_app.root.update()
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: str(blocker / "n.md"))
    fake_app.buttons["save"].invoke()
    assert fake_app.dirty and fake_app.memo.saved_to is None
    assert fake_app.status_label.cget("text").startswith("Could not save")


def test_closing_with_edits_asks_first(fake_app, finish, monkeypatch):
    record_one(fake_app, finish)
    fake_app.right.text.insert("end", "!")
    fake_app.root.update()
    monkeypatch.setattr(app_module.messagebox, "askyesnocancel", lambda *a, **k: None)
    fake_app.close()
    assert fake_app.root.winfo_exists()
    monkeypatch.setattr(app_module.messagebox, "askyesnocancel", lambda *a, **k: False)
    fake_app.close()
    with pytest.raises(tk.TclError, match="destroyed"):
        fake_app.root.winfo_exists()


# --- discard ---------------------------------------------------------------------


def test_discard_moves_the_memo_to_the_trash_and_clears_the_panes(fake_app, finish):
    memo = record_one(fake_app, finish)
    fake_app.player.discard_button.invoke()
    assert not memo.folder.exists()
    assert (fake_app.trash_dir / memo.stem / "memo.wav").exists()
    assert pane(fake_app, "left") == "" and pane(fake_app, "right") == ""
    assert fake_app.memo is None and fake_app.status_label.cget("text") == f"Discarded {memo.stem}"
    assert fake_app.player.discard_button.winfo_manager() == "" and fake_app.player.play_button.instate(["disabled"])
    fake_app.discard()
    assert fake_app.status_label.cget("text") == "Nothing to discard"


def test_a_blank_trash_folder_means_one_inside_the_recordings_folder(app_factory, tmp_path):
    app = app_factory("[capture]\nsave_dir = recs\ntrash_dir =\n")
    assert app.save_dir == tmp_path / "recs"
    assert app.trash_dir == tmp_path / "recs" / "trash"


def test_old_trash_is_purged_at_startup(app_factory, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(note, "purge_trash", lambda trash: calls.append(trash) or 0)
    app = app_factory(FAKE_CFG)
    assert calls == [app.trash_dir] and app.trash_dir == tmp_path / "memos" / "trash"


# --- vault ---------------------------------------------------------------------------


def test_choose_vault_validates_and_persists(fake_app, monkeypatch, vault, tmp_path):
    monkeypatch.setattr(app_module.filedialog, "askdirectory", lambda **kw: str(tmp_path))
    warned = []
    monkeypatch.setattr(app_module.messagebox, "showwarning", lambda *a, **k: warned.append(a))
    fake_app.buttons["choose_vault"].invoke()
    assert warned and fake_app.status_label.cget("text") == "Not an Obsidian vault"
    assert fake_app.config["vault"]["path"] == ""

    monkeypatch.setattr(app_module.filedialog, "askdirectory", lambda **kw: str(vault))
    monkeypatch.setattr(app_module.note, "is_registered", lambda path: False)
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: False)
    fake_app.buttons["choose_vault"].invoke()
    assert fake_app.vault == vault
    assert config.load(fake_app.config_path)[0]["vault"]["path"] == str(vault)
    assert fake_app.status_label.cget("text") == "Open this vault in Obsidian once"

    monkeypatch.setattr(app_module.note, "is_registered", lambda path: True)
    fake_app.buttons["choose_vault"].invoke()
    assert fake_app.status_label.cget("text") == f"Vault: {vault.name}"
