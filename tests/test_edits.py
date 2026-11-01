"""The Transcript and the Note: Clean up rebuilds the Note from the Transcript; Note edits never touch the Transcript."""

from __future__ import annotations

import json
import logging

import pytest

from voice2memos import cleanup, config
from voice2memos.ui import app as app_module

from .conftest import record_one
from .test_buttons import EXPECTED_NOTE, EXPECTED_RAW, pane

EDITED_SECOND = "maybe Bookinnit should have fields instead of tables"


def type_into(app, side: str, text: str, where: str = "end") -> None:
    """Typing as far as the panes can tell: an insert, then the event loop delivers <<Modified>>."""
    getattr(app, side).text.insert(where, text)
    app.root.update()


def replace_transcript(app, text: str) -> None:
    app.left.text.delete("1.0", "end")
    app.left.text.insert("1.0", text)
    app.root.update()


@pytest.fixture
def recorded(fake_app, finish):
    record_one(fake_app, finish)
    fake_app.toggle_lock()  # unlocked, as a user would before typing
    return fake_app


def test_the_panes_are_titled(app):
    assert app.left.title.cget("text") == "Transcript"
    assert app.right.title.cget("text") == "Note"


def test_the_note_follows_the_transcript_as_it_is_typed(recorded):
    app = recorded
    type_into(app, "left", " really")
    assert pane(app, "right") == EXPECTED_RAW + " really"
    type_into(app, "left", "\n\nA new thought")
    assert pane(app, "right") == EXPECTED_RAW + " really\n\nA new thought"
    assert pane(app, "left") == pane(app, "right")
    assert not app.note_hand_edited  # the program wrote it, not the user


def test_clean_up_shows_with_a_transcript_and_says_whether_it_was_edited(fake_app, finish, caplog):
    app = fake_app
    assert app.commit_button.winfo_manager() == ""  # nothing to clean up yet
    record_one(app, finish)
    assert app.commit_button.winfo_manager() == "pack" and app.commit_button.cget("text") == app_module.ICONS["commit"]
    assert app.tooltips["commit"].text == app_module.CLEAN_UP_TIP
    app.toggle_lock()
    with caplog.at_level(logging.INFO):
        type_into(app, "left", " really")
    assert app.tooltips["commit"].text == app_module.CLEAN_UP_EDITED_TIP
    assert "transcript edited; the note follows it until it's cleaned up" in caplog.messages
    app.left.text.delete("1.0", "end")
    app.root.update()
    assert app.commit_button.winfo_manager() == ""


def test_typing_back_to_the_original_gives_the_note_its_cleanup_back(recorded):
    app = recorded
    type_into(app, "left", " really")
    assert app.tooltips["commit"].text == app_module.CLEAN_UP_EDITED_TIP
    app.left.text.delete("end-8c", "end-1c")  # typed back to what it was
    app.root.update()
    assert pane(app, "left") == EXPECTED_RAW
    assert pane(app, "right") == EXPECTED_NOTE
    assert app.tooltips["commit"].text == app_module.CLEAN_UP_TIP


def test_clean_up_on_an_unchanged_transcript_runs_the_cleanup_again(recorded, finish, caplog):
    app = recorded
    memo = app.memo
    with caplog.at_level(logging.INFO):
        app.commit_button.invoke()
        assert pane(app, "right") == EXPECTED_RAW  # raw until the cleanup lands
        finish(app)
    assert pane(app, "right") == EXPECTED_NOTE and pane(app, "left") == EXPECTED_RAW
    assert f"{memo.stem}: cleaning up again, 2 segment(s)" in caplog.messages
    assert not (memo.folder / "segments.original.json").exists()  # nothing was edited, so nothing set aside
    assert f"{memo.stem}: cleanup (fake) start" in caplog.messages
    assert "transcribe (fake whisper) start" not in caplog.text


def test_plus_rebuilds_the_note_and_keeps_the_transcript_as_written(recorded, finish, caplog):
    app = recorded
    memo = app.memo
    first = memo.segments[0]
    replace_transcript(app, f"{first.text.strip()}\n\n{EDITED_SECOND}")
    with caplog.at_level(logging.INFO):
        app.commit_button.invoke()
        finish(app)
    # The unchanged paragraph comes back cleaned from the fixture; the fake echoes the edited one.
    assert pane(app, "right") == f"I was thinking about the database thing again.\n\n{EDITED_SECOND}"
    assert pane(app, "left") == f"{first.text.strip()}\n\n{EDITED_SECOND}"
    assert app.tooltips["commit"].text == app_module.CLEAN_UP_TIP  # nothing left to commit
    assert [(s.start, s.end) for s in memo.segments] == [(0.0, 3.2), (3.4, 7.1)]
    assert memo.segments[0].words and not memo.segments[1].words  # stale word timings dropped
    saved = json.loads((memo.folder / "segments.json").read_text(encoding="utf-8"))
    assert saved[1]["text"] == EDITED_SECOND
    original = json.loads((memo.folder / "segments.original.json").read_text(encoding="utf-8"))
    assert original[1]["text"].strip() == "like maybe Bookinnit should just have fields instead of tables you know"
    assert "click: commit" in caplog.messages
    assert f"{memo.stem}: transcript edited by hand, 2 segment(s); cleaning it up" in caplog.messages
    assert f"{memo.stem}: transcribe (fake whisper) start" not in caplog.text  # only cleanup onward


def test_editing_the_note_never_touches_the_transcript(recorded):
    app = recorded
    type_into(app, "right", " And a thought of my own.")
    assert pane(app, "left") == EXPECTED_RAW
    assert app.tooltips["commit"].text == app_module.CLEAN_UP_TIP  # the Transcript wasn't edited
    assert app.note_hand_edited and app.dirty


def test_note_edits_are_kept_if_the_user_says_so_asked_once(recorded, finish, monkeypatch):
    app = recorded
    type_into(app, "right", " Mine.")
    asked = []
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: asked.append(a) or False)
    type_into(app, "left", " really")
    type_into(app, "left", " truly")
    assert len(asked) == 1  # not on every key
    assert pane(app, "right") == EXPECTED_NOTE + " Mine."
    assert app.tooltips["commit"].text == app_module.CLEAN_UP_EDITED_TIP  # the edit is still there to clean up

    app.commit_button.invoke()  # Clean up would replace those edits too, so it asks as well
    assert len(asked) == 2 and not app.busy
    assert pane(app, "right") == EXPECTED_NOTE + " Mine."

    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: True)
    app.commit_button.invoke()
    finish(app)
    assert not pane(app, "right").endswith("Mine.")
    assert not app.note_hand_edited


def test_note_edits_give_way_to_the_transcript_if_the_user_says_so(recorded, monkeypatch):
    app = recorded
    type_into(app, "right", " Mine.")
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: True)
    type_into(app, "left", " really")
    assert pane(app, "right") == EXPECTED_RAW + " really"
    monkeypatch.setattr(app_module.messagebox, "askyesno", lambda *a, **k: pytest.fail("asked again"))
    type_into(app, "left", " truly")
    assert pane(app, "right") == EXPECTED_RAW + " really truly"


def test_saving_a_hand_edited_note_learns_examples_only_when_asked_to(recorded, monkeypatch, vault, caplog):
    app = recorded
    target = vault / "n.md"
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: str(target))
    type_into(app, "right", " Mine.")
    app.buttons["save"].invoke()
    assert app.status_label.cget("text") == "Saved n.md"
    assert "Example input:" not in app.config_path.read_text(encoding="utf-8")  # off by default

    app.config, _ = config.update_file(app.config_path, {("cleanup", "learn_from_edits"): "true"})
    app.dirty = True
    with caplog.at_level(logging.INFO):
        app.buttons["save"].invoke()
    assert app.status_label.cget("text") == "Saved n.md, 1 cleanup example(s) learned"
    assert f"{app.memo.stem}: 1 cleanup example(s) learned from the edited note" in caplog.messages
    learned = app.config["cleanup"]["examples"]
    assert learned == (
        "Example input:\nlike maybe Bookinnit should just have fields instead of tables you know\n"
        "Example output:\nMaybe Bookinnit should just have fields instead of tables. Mine."
    )
    assert config.load(app.config_path)[0]["cleanup"]["examples"] == learned
    assert cleanup.rewrite_prompt(app.config).endswith(learned)

    app.dirty = True
    app.buttons["save"].invoke()  # the same edit again adds nothing
    assert app.status_label.cget("text") == "Saved n.md"
    assert app.config["cleanup"]["examples"].count("Example input:") == 1


def test_a_saved_note_rebuilt_from_the_transcript_needs_saving_again(recorded, finish, monkeypatch, vault):
    app = recorded
    target = vault / "n.md"
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: str(target))
    app.buttons["save"].invoke()
    assert not app.dirty
    type_into(app, "left", " really")
    app.commit_button.invoke()
    finish(app)
    assert app.dirty and app.root.title().endswith("*")
    app.buttons["save"].invoke()
    assert "you know really" in target.read_text(encoding="utf-8")


def test_more_paragraphs_than_segments_each_become_one(recorded, finish):
    app = recorded
    replace_transcript(app, "One thought.\n\nAnother.\n\nA third.")
    app.commit_button.invoke()
    finish(app)
    assert [s.text for s in app.memo.segments] == ["One thought.", "Another.", "A third."]
    assert {(s.start, s.end) for s in app.memo.segments} == {(0.0, 7.1)}
    assert pane(app, "right") == "One thought.\n\nAnother.\n\nA third."


def test_typing_a_transcript_from_scratch_starts_a_memo_without_audio(fake_app, finish):
    app = fake_app
    app.toggle_lock()
    type_into(app, "left", "Remember the milk.")
    assert pane(app, "right") == "Remember the milk." and app.memo is None  # shown at once, no memo yet
    app.commit_button.invoke()
    finish(app)
    assert app.memo is not None and not app.memo.has_audio
    assert pane(app, "right") == "Remember the milk."
    assert not app.player.playback.total and app.player.discard_button.winfo_manager() == ""


def test_plus_is_disabled_while_the_worker_runs(recorded, finish):
    app = recorded
    type_into(app, "left", " really")
    app.commit_button.invoke()
    assert app.busy and app.commit_button.instate(["disabled"])
    finish(app)
    assert not app.commit_button.instate(["disabled"])
