"""Naming notes and saving them into the vault on their own.

The title stage names the note; the file takes that name, or the name and the
memo's date and time when the vault already has a note called that, with an
alias and links to the others. With auto_save on, a finished memo is in the
vault without anyone pressing Save.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from voice2memos import cleanup, config, note
from voice2memos.note import Memo
from voice2memos.ui import app as app_module

from .conftest import FAKE_CFG, record_one

NAME = "Bookinnit database fields"  # tests/fixtures/sample/title.json
CREATED = datetime.fromisoformat("2026-11-01T14:43:31-05:00")


# --- the name itself ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, name",
    [
        ("Dogs waiting for dinner", "Dogs waiting for dinner"),
        ('  "Dogs: dinner / time?"  ', "Dogs dinner time"),
        ("Atlas [[Ares]] #dogs ^x | y", "Atlas Ares dogs x y"),
        ("...hidden.", "hidden"),
        ("line\nbreak\ttab", "line break tab"),
        ("CON", ""),
        ("nul.txt", ""),
        ("", ""),
        ("x" * 100, "x" * 80),
        ("word " * 30, ("word " * 16).strip()),
    ],
)
def test_sanitize_title(raw, name):
    assert cleanup.sanitize_title(raw) == name
    assert len(cleanup.sanitize_title(raw)) <= cleanup.MAX_TITLE


def test_the_title_stage_asks_the_model_and_cleans_up_its_answer(fixtures_dir):
    provider = cleanup.FakeProvider(fixtures_dir / "sample")
    assert cleanup.suggest_title(provider, "anything", 5, "s") == NAME

    class Answers(cleanup.Provider):
        name = "answers"

        def __init__(self, answer):
            self.answer = answer

        def complete(self, system, user, *, json_mode=False, timeout=60):
            assert json_mode and '"title"' in system
            return self.answer

    assert cleanup.suggest_title(Answers('{"title": "Atlas: dinner?"}'), "t", 5, "s") == "Atlas dinner"
    assert cleanup.suggest_title(Answers('"Bare string"'), "t", 5, "s") == "Bare string"
    assert cleanup.suggest_title(Answers("not json"), "t", 5, "s") == ""
    assert cleanup.suggest_title(Answers('{"title": 3}'), "t", 5, "s") == ""


def test_a_failing_title_call_is_a_naming_error():
    with pytest.raises(cleanup.EntityError, match="Naming failed"):
        cleanup.suggest_title(cleanup.FakeProvider(fail={"title"}), "t", 5, "s")


def test_the_memo_title_prefers_the_user_then_the_saved_name_then_the_model(tmp_path):
    memo = Memo.create(tmp_path / "20261101-144331")
    assert memo.title == "20261101-144331"  # nothing yet: the stem
    memo.proposed_title = "Dogs at dinner"
    assert memo.title == "Dogs at dinner"
    memo.saved_to, memo.saved_title = tmp_path / "Dogs at dinner.md", "Dogs at dinner"
    memo.proposed_title = "Something else"  # a rerun's new idea
    assert memo.title == "Dogs at dinner"  # a saved note keeps its name
    memo.chosen_title = "My: name"
    assert memo.title == "My name"  # the user's wins, cleaned up the same way
    memo.chosen_title = "   "
    assert memo.title == "Dogs at dinner"


def test_a_note_saved_under_its_stem_takes_the_model_name_when_it_comes(tmp_path):
    memo = Memo.create(tmp_path / "20261101-144331")
    memo.saved_to, memo.saved_title = tmp_path / "20261101-144331.md", "20261101-144331"
    memo.proposed_title = "Dogs at dinner"
    assert memo.title == "Dogs at dinner"


def test_title_and_saved_name_survive_reopening(tmp_path):
    memo = Memo.create(tmp_path / "m")
    memo.proposed_title, memo.chosen_title = "Model name", "Mine"
    memo.save_title()
    memo.saved_to, memo.saved_title = tmp_path / "Mine.md", "Mine"
    memo.save_state()
    again = Memo.open(memo.folder)
    assert (again.proposed_title, again.chosen_title, again.saved_to, again.saved_title) == (
        "Model name", "Mine", tmp_path / "Mine.md", "Mine",
    )


# --- where it goes ------------------------------------------------------------------


def memo_named(tmp_path: Path, title: str) -> Memo:
    memo = Memo.create(tmp_path / "memos" / "20261101-144331")
    memo.created = CREATED
    memo.proposed_title = title
    return memo


def test_vault_notes_lists_every_note_but_hidden_folders(vault):
    (vault / "Atlas.md").write_text("x", encoding="utf-8")
    (vault / "dogs").mkdir()
    (vault / "dogs" / "Ares.MD").write_text("x", encoding="utf-8")
    (vault / ".trash").mkdir()
    (vault / ".trash" / "Gone.md").write_text("x", encoding="utf-8")
    (vault / ".obsidian" / "workspace.md").write_text("x", encoding="utf-8")
    (vault / "attachments").mkdir(exist_ok=True)
    (vault / "attachments" / "a.wav").write_bytes(b"")
    assert note.vault_notes(vault) == {"atlas": "Atlas", "ares": "Ares"}
    assert note.vault_notes(vault / "missing") == {}


def test_a_free_name_is_used_as_it_is(tmp_path, vault):
    placement = note.place(memo_named(tmp_path, "Dogs at dinner"), vault, {"atlas": "Atlas"})
    assert placement == note.Placement(vault / "Dogs at dinner.md", "Dogs at dinner", [])
    assert not placement.alias


def test_a_taken_name_gets_the_date_an_alias_and_links_to_the_others(tmp_path, vault):
    names = {
        "dogs at dinner": "Dogs at dinner",
        "dogs at dinner 2026-10-30 0912": "Dogs at dinner 2026-10-30 0912",
        "dogs at dinner party": "Dogs at dinner party",  # a different name that starts the same
    }
    memo = memo_named(tmp_path, "dogs at dinner")  # case doesn't make it a different note to Obsidian
    placement = note.place(memo, vault, names)
    assert placement.path == vault / "dogs at dinner 2026-11-01 1443.md"
    assert placement.alias and placement.see_also == ["Dogs at dinner", "Dogs at dinner 2026-10-30 0912"]
    text = note.render(memo, None, placement)
    assert '\naliases: ["dogs at dinner"]\n---\n# dogs at dinner\n' in text
    assert "\n---\n\nSee also: [[Dogs at dinner]], [[Dogs at dinner 2026-10-30 0912]]\n" in text


def test_two_memos_in_one_minute_still_get_their_own_files(tmp_path, vault):
    names = {"dogs": "Dogs", "dogs 2026-11-01 1443": "Dogs 2026-11-01 1443"}
    placement = note.place(memo_named(tmp_path, "Dogs"), vault, names)
    assert placement.path == vault / "Dogs 2026-11-01 1443-01.md"
    assert placement.see_also == ["Dogs", "Dogs 2026-11-01 1443"]


def test_a_saved_note_stays_put_and_never_counts_as_its_own_twin(tmp_path, vault):
    memo = memo_named(tmp_path, "Dogs")
    memo.saved_to, memo.saved_title = vault / "Dogs.md", "Dogs"
    placement = note.place(memo, vault, {"dogs": "Dogs"})
    assert placement == note.Placement(vault / "Dogs.md", "Dogs", [])


def test_a_new_name_moves_a_saved_note(tmp_path, vault):
    memo = memo_named(tmp_path, "Dogs")
    memo.saved_to, memo.saved_title = vault / "Dogs.md", "Dogs"
    memo.chosen_title = "Cats"
    assert note.place(memo, vault, {"dogs": "Dogs"}).path == vault / "Cats.md"


def test_a_picked_file_is_used_and_still_links_the_notes_of_that_name(tmp_path, vault):
    memo = memo_named(tmp_path, "Dogs")
    placement = note.place(memo, tmp_path, {"dogs": "Dogs"}, path=tmp_path / "elsewhere.md")
    assert placement.path == tmp_path / "elsewhere.md" and placement.alias and placement.see_also == ["Dogs"]


def test_remove_old_copy_leaves_a_case_only_rename_alone(tmp_path):
    old = tmp_path / "dogs.md"
    old.write_text("x", encoding="utf-8")
    note.remove_old_copy(old, tmp_path / "DOGS.md")  # the same file where case is ignored
    assert old.exists() == (tmp_path / "DOGS.md").exists()
    other = tmp_path / "Cats.md"
    other.write_text("y", encoding="utf-8")
    note.remove_old_copy(old, other)
    assert not old.exists() and other.exists()


# --- in the window ------------------------------------------------------------------


def vault_cfg(vault: Path, auto_save: str = "true", fake: str = "") -> str:
    """FAKE_CFG ends inside [fake], so ``fake`` lines go straight after it."""
    return FAKE_CFG + fake + f"[vault]\npath = {vault}\nauto_save = {auto_save}\n"


def test_auto_save_is_on_by_default_and_checked():
    assert config.DEFAULTS["vault"]["auto_save"] == "true"
    assert config.CHECKS[("vault", "auto_save")][0]("False")
    assert not config.CHECKS[("vault", "auto_save")][0]("yes")


def test_a_finished_memo_saves_itself_into_the_vault(app_factory, finish, vault, monkeypatch, caplog):
    app = app_factory(vault_cfg(vault))
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: pytest.fail("asked where"))
    with caplog.at_level(logging.INFO):
        memo = record_one(app, finish)
    target = vault / f"{NAME}.md"
    assert target.read_text(encoding="utf-8") == app.footer.preview_text()
    assert (vault / "attachments" / f"{memo.stem}.wav").exists()
    assert memo.saved_to == target and not app.dirty and not app.root.title().endswith("*")
    assert app.status_label.cget("text") == f"Saved {target.name}"
    assert app.footer.name_hint.cget("text") == f"saved as {target.name}"
    assert f"{memo.stem}: saved to {target}" in caplog.messages


def test_the_next_memo_of_the_same_name_links_back_without_touching_the_first(app_factory, finish, vault):
    app = app_factory(vault_cfg(vault))
    first = record_one(app, finish)
    first_text = first.saved_to.read_text(encoding="utf-8")
    second = record_one(app, finish)
    assert second.saved_to.name.startswith(f"{NAME} ") and second.saved_to != first.saved_to
    text = second.saved_to.read_text(encoding="utf-8")
    assert f'\naliases: ["{NAME}"]\n' in text and f"\nSee also: [[{NAME}]]\n" in text
    assert first.saved_to.read_text(encoding="utf-8") == first_text
    assert app.footer.name_hint.cget("text") == f"saved as {second.saved_to.name}, linked to 1 other note of that name"


def test_with_auto_save_off_nothing_is_written_until_save(app_factory, finish, vault):
    app = app_factory(vault_cfg(vault, "false"))
    memo = record_one(app, finish)
    assert memo.saved_to is None and not list(vault.glob("*.md"))
    assert app.status_label.cget("text") == "Ready"
    assert app.footer.name_hint.cget("text") == f"saves as {NAME}.md"
    app.buttons["save"].invoke()
    assert (vault / f"{NAME}.md").exists()


def test_with_no_vault_a_finished_memo_says_it_was_not_saved(app_factory, finish):
    app = app_factory(FAKE_CFG + "[vault]\nauto_save = true\n")
    memo = record_one(app, finish)
    assert memo.saved_to is None
    assert app.status_label.cget("text") == "Done, not saved: choose your vault and memos save themselves"


def test_a_vault_that_is_not_there_is_never_created(app_factory, finish, tmp_path):
    missing = tmp_path / "unplugged"
    app = app_factory(vault_cfg(missing))
    record_one(app, finish)
    assert not missing.exists() and app.memo.saved_to is None


def test_a_run_with_a_problem_still_saves_but_keeps_its_retry(app_factory, finish, vault):
    app = app_factory(vault_cfg(vault, fake="fail = cleanup\n"))
    memo = record_one(app, finish)
    assert memo.saved_to == vault / f"{NAME}.md"  # a worse note, never no note
    assert "um so I was thinking" in memo.saved_to.read_text(encoding="utf-8")
    assert app.status_label.cget("text") == "Cleanup failed, showing original"
    assert app.retry_callback is not None


def test_a_failed_name_saves_under_the_stem_and_the_retry_renames_it(app_factory, finish, vault):
    app = app_factory(vault_cfg(vault, fake="fail = title\n"))
    memo = record_one(app, finish)
    stemmed = vault / f"{memo.stem}.md"
    assert memo.saved_to == stemmed and app.status_label.cget("text") == "Naming failed"
    app.config["fake"]["fail"] = ""
    app.retry_label.event_generate("<Button-1>")
    finish(app)
    assert memo.saved_to == vault / f"{NAME}.md" and not stemmed.exists()
    assert app.footer.name_var.get() == NAME


def test_cleaning_up_again_keeps_the_saved_name(app_factory, finish, vault, monkeypatch):
    app = app_factory(vault_cfg(vault))
    memo = record_one(app, finish)
    memo_title = json.loads((memo.folder / "title.json").read_text(encoding="utf-8"))
    assert memo_title == {"proposed": NAME, "chosen": ""}
    monkeypatch.setattr(cleanup, "suggest_title", lambda *a: "A different idea")
    app.commit_button.invoke()
    finish(app)
    assert memo.proposed_title == "A different idea"
    assert memo.saved_to == vault / f"{NAME}.md" and [p.name for p in vault.glob("*.md")] == [f"{NAME}.md"]
    assert app.footer.name_var.get() == NAME


def test_typing_a_name_renames_the_saved_note(app_factory, finish, vault, caplog):
    app = app_factory(vault_cfg(vault))
    memo = record_one(app, finish)
    old = memo.saved_to
    entry = app.footer.name_entry
    entry.delete(0, "end")
    entry.insert(0, "Dogs: dinner")
    app.root.update()
    assert memo.chosen_title == "Dogs: dinner" and app.dirty
    assert json.loads((memo.folder / "title.json").read_text(encoding="utf-8"))["chosen"] == "Dogs: dinner"
    assert app.footer.name_hint.cget("text") == f"Save renames {old.name} to Dogs dinner.md"
    assert "\n# Dogs dinner\n" in app.footer.preview_text()
    with caplog.at_level(logging.INFO):
        app.buttons["save"].invoke()
    assert not old.exists() and memo.saved_to == vault / "Dogs dinner.md"
    assert memo.saved_to.read_text(encoding="utf-8") == app.footer.preview_text()
    assert f"{memo.stem}: renamed {old.name} to Dogs dinner.md" in caplog.messages
    assert app.footer.name_hint.cget("text") == "saved as Dogs dinner.md"


def test_a_typed_name_outlives_the_title_stage(app_factory, finish):
    app = app_factory(FAKE_CFG + "delay = 0.3\n")
    app.buttons["record"].invoke()
    app.buttons["record"].invoke()
    app.root.update()
    app.footer.name_entry.delete(0, "end")
    app.footer.name_entry.insert(0, "Mine")
    finish(app)
    assert app.footer.name_var.get() == "Mine" and app.memo.title == "Mine"
    assert app.memo.proposed_title == NAME


def test_the_name_field_is_locked_without_a_memo_and_clears_on_discard(fake_app, finish):
    footer = fake_app.footer
    assert footer.name_entry.instate(["disabled"]) and footer.name_var.get() == ""
    record_one(fake_app, finish)
    assert not footer.name_entry.instate(["disabled"]) and footer.name_var.get() == NAME
    fake_app.player.discard_button.invoke()
    assert footer.name_entry.instate(["disabled"]) and footer.name_var.get() == ""
    assert footer.name_hint.cget("text") == ""


def test_the_name_field_is_key_logged(fake_app):
    assert fake_app.footer.name_entry.bind("<KeyPress>") != ""


def test_save_as_keeps_the_earlier_copy(app_factory, finish, vault, monkeypatch, tmp_path):
    app = app_factory(vault_cfg(vault))
    memo = record_one(app, finish)
    first = memo.saved_to
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename", lambda **kw: str(tmp_path / "copy.md"))
    app.save_menu.invoke(0)
    assert first.exists() and memo.saved_to == tmp_path / "copy.md"
