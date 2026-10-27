"""Config loader: defaults, partial files, round trips, bad files, and nothing quoted back."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from voice2memos import config

REPO = Path(__file__).resolve().parents[1]
SECRET = "sk-test-0123456789abcdefSECRET"

# Every awkward thing a hand-written prompt might hold.
PROMPT = """Rewrite the note.

# Rules
; keep this line too
[not a section]
key = value, still prompt
100% of the ideas stay"""


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "voice2memos.cfg"


def defaults() -> dict[config.Field, str]:
    return config.values(config.new_parser())


def test_a_missing_file_gives_defaults_and_is_written(path):
    cfg, warnings = config.load(path)
    assert warnings == []
    assert config.values(cfg) == defaults()
    assert path.read_text(encoding="utf-8").startswith(config.HEADER)
    assert config.values(config.load(path)[0]) == defaults()


def test_a_list_written_below_its_key_has_no_blank_first_line(path):
    path.write_text("[tags]\npool =\n    idea\n    todo\n", encoding="utf-8")
    assert config.load(path)[0]["tags"]["pool"] == "idea\ntodo"


def test_a_partial_file_gets_the_missing_keys(path):
    path.write_text("[whisper]\nmodel_size = medium\n", encoding="utf-8")
    cfg, warnings = config.load(path)
    assert warnings == []
    assert cfg["whisper"]["model_size"] == "medium"
    assert cfg["whisper"]["device"] == "auto"
    assert cfg["tags"]["pool"] == config.DEFAULTS["tags"]["pool"]


def test_the_multi_line_prompt_round_trips(path):
    cfg, _ = config.load(path)
    cfg["cleanup"]["prompt"] = PROMPT
    config.save(cfg, path)
    assert config.load(path)[0]["cleanup"]["prompt"] == PROMPT


def test_comments_at_the_left_edge_are_ignored(path):
    path.write_text(
        "﻿# A comment, after the byte-order mark Notepad adds\n"
        "[cleanup]\n"
        "; another comment\n"
        "model = llama3\n"
        "prompt = First line.\n"
        "    # Indented, so part of the prompt\n",
        encoding="utf-8",
    )
    cfg, warnings = config.load(path)
    assert warnings == []
    assert cfg["cleanup"]["model"] == "llama3"
    assert cfg["cleanup"]["prompt"] == "First line.\n# Indented, so part of the prompt"


def test_the_example_config_is_exactly_the_defaults(path):
    # A copy: load() renames a broken file, and the repo's own example must never move.
    shutil.copy(REPO / "voice2memos.example.cfg", path)
    cfg, warnings = config.load(path)
    assert warnings == []
    assert config.values(cfg) == defaults()


def test_every_secret_field_has_a_default():
    for section, key in config.SECRET_FIELDS:
        assert config.DEFAULTS[section][key] == ""


def test_a_broken_file_is_set_aside_and_never_quoted(path, caplog):
    path.write_text(f"[keys]\nopenai = fine\n{SECRET}\n", encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        cfg, warnings = config.load(path)
    assert warnings == ["Config error on line 3, using defaults"]
    assert config.values(cfg) == defaults()
    assert SECRET in path.with_name("voice2memos.cfg.bad").read_text(encoding="utf-8")
    assert config.values(config.load(path)[0]) == defaults()
    assert "SECRET" not in caplog.text


def test_a_file_without_sections_names_line_one(path):
    path.write_text("model = llama3\n", encoding="utf-8")
    assert config.load(path)[1] == ["Config error on line 1, using defaults"]


def test_a_bad_value_falls_back_without_being_quoted(path, caplog):
    path.write_text("[cleanup]\ntemperature = hot\ntimeout = 0\n[window]\ngeometry = big\n", encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        cfg, warnings = config.load(path)
    assert warnings == [
        "Config cleanup.temperature should be a number from 0 to 1; using 0",
        "Config cleanup.timeout should be a number of seconds; using 60",
        "Config window.geometry should be width x height, like 960x600; using 960x600",
    ]
    assert cfg["cleanup"]["temperature"] == "0"
    assert "hot" not in caplog.text and "big" not in caplog.text
    assert "temperature = hot" in path.read_text(encoding="utf-8")  # the file is left alone


def test_update_file_keeps_edits_made_by_hand_meanwhile(path):
    cfg, _ = config.load(path)
    text = path.read_text(encoding="utf-8").replace("model_size = small", "model_size = medium")
    path.write_text(text + "\n[mine]\nnote = kept\n", encoding="utf-8")

    updated, warnings = config.update_file(path, {("window", "geometry"): "1200x700"})
    assert warnings == []
    assert updated["window"]["geometry"] == "1200x700"
    assert updated["whisper"]["model_size"] == "medium"
    reread, _ = config.load(path)
    assert reread["whisper"]["model_size"] == "medium"
    assert reread["mine"]["note"] == "kept"
    assert not path.with_name(path.name + ".tmp").exists()


def test_update_file_writes_nothing_when_the_file_is_broken(path):
    path.write_text("[keys]\nnot a key line\n", encoding="utf-8")
    with pytest.raises(config.configparser.Error):
        config.update_file(path, {("window", "geometry"): "1200x700"})
    assert path.read_text(encoding="utf-8") == "[keys]\nnot a key line\n"


def test_relative_folders_are_beside_the_config(path, tmp_path):
    assert config.resolve_folder("memos", path) == tmp_path / "memos"
    assert config.resolve_folder(str(tmp_path / "elsewhere"), path) == tmp_path / "elsewhere"
