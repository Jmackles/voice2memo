"""The footer under the panes: tag chips a click strikes, and the preview of the note as Save writes it."""

from __future__ import annotations

import json
import logging

from voice2memos import config, note
from voice2memos.ui import app as app_module
from voice2memos.ui.chips import CLICK_HINT, NO_TAGS
from voice2memos.ui.footer import FOLDED, OPEN

from .conftest import FAKE_CFG, record_one
from .test_buttons import pane


def chips(app) -> list[str]:
    return list(app.footer.chips.chips)


def frontmatter_tags(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("tags:"))


def test_the_footer_starts_empty_with_the_preview_open(app):
    footer = app.footer
    assert footer.frame.winfo_manager() == "pack" and footer.shown
    assert chips(app) == [] and footer.chips.hint.cget("text") == NO_TAGS
    assert footer.preview_text() == "" and footer.preview.cget("state") == "disabled"


def test_a_memo_fills_the_chips_and_the_preview_is_what_save_would_write(fake_app, finish):
    memo = record_one(fake_app, finish)
    footer = fake_app.footer
    assert chips(fake_app) == ["idea", "bookinnit"] and footer.chips.hint.cget("text") == CLICK_HINT
    assert not any(footer.chips.is_struck(t) for t in chips(fake_app))
    assert footer.preview_text() == note.render(memo, pane(fake_app, "right"))
    assert "[[database]]" in footer.preview_text() and "tags: [idea, bookinnit]" in footer.preview_text()


def test_striking_a_chip_drops_the_tag_and_a_second_click_puts_it_back(fake_app, finish, caplog):
    memo = record_one(fake_app, finish)
    footer = fake_app.footer
    assert not fake_app.dirty
    with caplog.at_level(logging.INFO):
        footer.chips.chips["idea"].event_generate("<Button-1>")
    assert footer.chips.is_struck("idea") and memo.rejected_tags == ["idea"]
    assert json.loads((memo.folder / "tags.json").read_text(encoding="utf-8"))["rejected"] == ["idea"]
    assert frontmatter_tags(footer.preview_text()) == "tags: [bookinnit]"
    assert fake_app.dirty and fake_app.root.title().endswith("*")
    assert f"{memo.stem}: tag idea struck" in caplog.messages
    footer.chips.toggle("idea")
    assert not footer.chips.is_struck("idea") and memo.rejected_tags == []
    assert frontmatter_tags(footer.preview_text()) == "tags: [idea, bookinnit]"


def test_a_struck_tag_stays_out_of_the_saved_note(fake_app, finish, monkeypatch, vault):
    record_one(fake_app, finish)
    fake_app.footer.chips.toggle("bookinnit")
    target = vault / "n.md"
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: str(target))
    fake_app.buttons["save"].invoke()
    saved = target.read_text(encoding="utf-8")
    assert frontmatter_tags(saved) == "tags: [idea]"
    assert saved == fake_app.footer.preview_text()  # the preview never lies


def test_typing_in_the_note_updates_the_preview(fake_app, finish):
    record_one(fake_app, finish)
    fake_app.right.text.insert("end", " And more.")
    fake_app.root.update()
    assert "[[tables]]. And more.\n" in fake_app.footer.preview_text()


def test_cleaning_up_again_clears_the_chips_until_new_ones_arrive(fake_app, finish):
    record_one(fake_app, finish)
    fake_app.footer.chips.toggle("idea")
    fake_app.commit_button.invoke()
    assert chips(fake_app) == []  # proposals on their way
    finish(fake_app)
    assert chips(fake_app) == ["idea", "bookinnit"]
    assert fake_app.footer.chips.is_struck("idea")  # the decision outlives the rerun


def test_discard_clears_the_footer(fake_app, finish):
    record_one(fake_app, finish)
    fake_app.player.discard_button.invoke()
    assert chips(fake_app) == [] and fake_app.footer.preview_text() == ""


def test_the_preview_folds_and_is_remembered(app_factory, tmp_path, caplog):
    app = app_factory(FAKE_CFG)
    footer = app.footer
    assert footer.fold_button.cget("text") == OPEN and footer.body.winfo_manager() == "pack"
    with caplog.at_level(logging.INFO):
        footer.fold_button.invoke()
    assert footer.fold_button.cget("text") == FOLDED and footer.body.winfo_manager() == ""
    assert "click: preview" in caplog.messages and "note preview folded" in caplog.messages
    app.close()
    assert config.load(tmp_path / "voice2memos.cfg")[0]["window"]["preview"] == "false"
    again = app_factory(FAKE_CFG + "[window]\npreview = false\n")
    assert not again.footer.shown


def test_a_struck_chip_cannot_be_toggled_once_cleared(fake_app, finish):
    record_one(fake_app, finish)
    fake_app.footer.chips.clear()
    fake_app.footer.chips.toggle("idea")  # nothing to strike; no crash, no change
    assert fake_app.memo.rejected_tags == []
