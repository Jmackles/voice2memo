"""The working folder, the serializer against expected.md, vault checks, discard and trash purge."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pytest

from voice2memos import note
from voice2memos.errors import SaveError
from voice2memos.note import Memo
from voice2memos.transcribe import Segment

SAMPLE_DATE = datetime.fromisoformat("2026-10-28T17:26:00-04:00")


@pytest.fixture
def sample(fixtures_dir, tmp_path) -> Memo:
    """The sample fixture as a working folder named like its expected note."""
    folder = tmp_path / "memos" / "sample"
    shutil.copytree(fixtures_dir / "sample", folder)
    memo = Memo.open(folder)
    memo.created = SAMPLE_DATE
    return memo


def test_create_then_open_round_trips_the_state(tmp_path):
    memo = Memo.create(tmp_path / "20261028-172600")
    memo.segments = [Segment(0, 0, 1, " raw", "Clean.")]
    memo.entities, memo.tags, memo.rejected_tags = ["raw"], ["idea", "todo"], ["todo"]
    memo.saved_to = tmp_path / "vault" / "x.md"
    memo.save_segments()
    memo.save_cleaned()
    memo.save_entities()
    memo.save_tags()
    memo.save_state()
    again = Memo.open(memo.folder)
    assert again.segments == memo.segments
    assert (again.entities, again.tags, again.rejected_tags, again.saved_to) == (["raw"], ["idea", "todo"], ["todo"], memo.saved_to)
    assert again.created == memo.created
    assert again.stem == "20261028-172600" and again.wav == memo.folder / "memo.wav" and not again.has_audio


def test_open_prefers_cleaned_over_segments_and_tolerates_an_empty_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    assert Memo.open(tmp_path / "empty").segments == []
    memo = Memo.create(tmp_path / "m")
    memo.segments = [Segment(0, 0, 1, " raw")]
    memo.save_segments()
    memo.segments[0].cleaned = "Clean."
    memo.save_cleaned()
    assert Memo.open(memo.folder).segments[0].cleaned == "Clean."


def test_render_matches_the_expected_note(sample, fixtures_dir):
    assert note.render(sample) == (fixtures_dir / "sample" / "expected.md").read_text(encoding="utf-8")


def test_render_uses_the_edited_body_paragraph_per_segment(sample):
    body = "First edited.\n\nSecond, about Bookinnit."
    text = note.render(sample, body)
    assert "**0:00** First edited. ^sample-0\n\n**0:03** Second, about [[Bookinnit]]. ^sample-1\n" in text


def test_render_keeps_an_edited_body_whole_when_paragraphs_do_not_pair_up(sample):
    text = note.render(sample, "One paragraph only about tables.")
    assert "**0:00** One paragraph only about [[tables]]. ^sample-0\n" in text
    assert "^sample-1" not in text


def test_render_without_audio_or_segments_leaves_those_parts_out(tmp_path):
    memo = Memo.create(tmp_path / "typed")
    memo.created = SAMPLE_DATE
    text = note.render(memo, "Just typed.")
    assert text == "---\ndate: 2026-10-28T17:26:00-04:00\ntags: []\n---\n# typed\n\nJust typed. ^typed-0\n"


def test_a_typed_note_has_no_time_stamps_but_keeps_its_original_words(tmp_path):
    memo = Memo.create(tmp_path / "typed")
    memo.created = SAMPLE_DATE
    memo.segments = [Segment(0, 0.0, 0.0, "typed it", "Typed it.")]
    text = note.render(memo)
    assert text.endswith("# typed\n\nTyped it. ^typed-0\n\n---\n\n> [!quote]- Original\n> typed it\n")
    assert "**" not in text and "![[" not in text


def test_rejected_tags_stay_out_of_the_frontmatter(sample):
    sample.rejected_tags = ["idea"]
    assert "tags: [bookinnit]" in note.render(sample)


@pytest.mark.parametrize(
    "paragraphs, entities, linked",
    [
        (["Python and python."], ["python"], ["[[Python]] and python."]),
        (["a [[Python]] b Python"], ["Python"], ["a [[Python]] b Python"]),
        (["none here", "Python here"], ["Python"], ["none here", "[[Python]] here"]),
        (["Pythonic"], ["Python"], ["Pythonic"]),
        (["Python, yes"], ["Python"], ["[[Python]], yes"]),
    ],
)
def test_link_entities(paragraphs, entities, linked):
    assert note.link_entities(paragraphs, entities) == linked


def test_write_note_goes_through_a_temp_file(tmp_path):
    path = tmp_path / "vault" / "n.md"
    note.write_note(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"
    assert list(path.parent.iterdir()) == [path]


def test_write_note_failure_is_a_save_error(tmp_path):
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(SaveError, match="Could not save"):
        note.write_note(tmp_path / "file" / "n.md", "hello")


def test_copy_audio_creates_the_attachments_folder_and_names_by_stem(sample, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    target = note.copy_audio(sample, vault, "files")
    assert target == vault / "files" / "sample.wav"
    assert target.read_bytes() == sample.wav.read_bytes()
    assert note.copy_audio(sample, vault, "files") == target  # a second save is fine


def test_unique_path_counts_up(tmp_path):
    path = tmp_path / "n.md"
    assert note.unique_path(path) == path
    path.write_text("x", encoding="utf-8")
    assert note.unique_path(path) == tmp_path / "n-01.md"


def test_vault_checks(tmp_path, vault):
    assert note.is_vault(vault) and not note.is_vault(tmp_path)
    registry = tmp_path / "obsidian.json"
    assert note.registered_vaults(registry) == []
    registry.write_text(json.dumps({"vaults": {"abc": {"path": str(vault), "ts": 1}, "gone": {"path": str(tmp_path / "gone")}}}), encoding="utf-8")
    assert note.registered_vaults(registry) == [vault, tmp_path / "gone"]
    assert note.is_registered(vault, registry)
    assert not note.is_registered(tmp_path, registry)


def test_discard_moves_the_folder_and_the_saved_copies(sample, vault, tmp_path):
    saved = vault / "sample.md"
    saved.write_text("note", encoding="utf-8")
    note.copy_audio(sample, vault, "attachments")
    sample.saved_to = saved
    trash = tmp_path / "trash"
    moved = note.discard(sample, trash, vault, "attachments")
    assert not sample.folder.exists() and not saved.exists() and not (vault / "attachments" / "sample.wav").exists()
    assert (trash / "sample" / "memo.wav").exists()
    assert (trash / "sample" / "sample.md").exists() and (trash / "sample" / "sample.wav").exists()
    assert moved[0] == trash / "sample"


def test_discard_twice_keeps_both(sample, tmp_path):
    trash = tmp_path / "trash"
    note.discard(sample, trash, None, "attachments")
    again = Memo.create(tmp_path / "memos" / "sample")
    note.discard(again, trash, None, "attachments")
    assert sorted(p.name for p in trash.iterdir()) == ["sample", "sample-01"]


def test_purge_trash_removes_only_old_entries(tmp_path):
    trash = tmp_path / "trash"
    old, new = trash / "old", trash / "new"
    old.mkdir(parents=True)
    (old / "memo.wav").write_bytes(b"x")
    new.mkdir()
    stale = time.time() - 31 * 86400
    os.utime(old, (stale, stale))
    assert note.purge_trash(trash) == 1
    assert not old.exists() and new.exists()
    assert note.purge_trash(tmp_path / "missing") == 0


def test_open_note_prefers_obsidian_when_registered(monkeypatch):
    opened = []
    monkeypatch.setattr(note.webbrowser, "open", opened.append)
    path = Path("C:/v/n.md")
    assert note.open_note(path, registered=True) == "Opened in Obsidian"
    assert opened == [f"obsidian://open?path={quote(str(path))}"]


def test_keep_original_copies_once_and_never_again(sample):
    original = sample.folder / note.ORIGINAL_SEGMENTS_FILE
    first = (sample.folder / "segments.json").read_text(encoding="utf-8")
    sample.keep_original()
    assert original.read_text(encoding="utf-8") == first
    (sample.folder / "segments.json").write_text("[]", encoding="utf-8")
    sample.keep_original()
    assert original.read_text(encoding="utf-8") == first
