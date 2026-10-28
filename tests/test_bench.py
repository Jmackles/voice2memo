"""bench.py runs each built stage against the sample fixture and diffs."""

from __future__ import annotations

import json
import shutil

import pytest

import bench


@pytest.fixture
def cfg_path(tmp_path):
    return tmp_path / "bench.cfg"


def run(capsys, *args) -> tuple[int, str, str]:
    status = bench.main([str(a) for a in args])
    out = capsys.readouterr()
    return status, out.out, out.err


def test_fake_transcribe_matches_the_fixture(capsys, fixtures_dir, cfg_path):
    status, out, err = run(capsys, "--config", cfg_path, "--fake", "transcribe", fixtures_dir / "sample" / "memo.wav")
    assert status == 0 and err == ""
    assert "-- matches segments.json" in out
    assert "-- 0." in out  # elapsed seconds


def test_fake_clean_prints_pairs_and_matches(capsys, fixtures_dir, cfg_path):
    status, out, _ = run(capsys, "--config", cfg_path, "--fake", "clean", fixtures_dir / "sample" / "segments.json")
    assert status == 0
    assert "[0] um so I was thinking about the uh database thing again" in out
    assert "  > I was thinking about the database thing again." in out
    assert "-- matches cleaned.json" in out


def test_fake_clean_all_runs_every_fixture(capsys, cfg_path):
    status, out, _ = run(capsys, "--config", cfg_path, "--fake", "clean", "--all")
    assert status == 0
    assert out.count("== clean ") >= 1


def test_fake_entities_matches_both_files(capsys, fixtures_dir, cfg_path):
    status, out, _ = run(capsys, "--config", cfg_path, "--fake", "entities", fixtures_dir / "sample" / "cleaned.json")
    assert status == 0
    assert "-- matches entities.json" in out and "-- matches tags.json" in out


def test_a_differing_result_shows_a_diff_and_fails(capsys, fixtures_dir, tmp_path, cfg_path):
    # The fake answers from the real fixture (via the config); the expected file is a trimmed copy.
    copy = tmp_path / "copy"
    shutil.copytree(fixtures_dir / "sample", copy)
    expected = json.loads((copy / "entities.json").read_text(encoding="utf-8"))
    (copy / "entities.json").write_text(json.dumps(expected[:-1]), encoding="utf-8")
    cfg_path.write_text("[entities]\nprovider = fake\n[fake]\nfixture = tests/fixtures/sample\n", encoding="utf-8")
    status, out, _ = run(capsys, "--config", cfg_path, "entities", copy / "cleaned.json")
    assert status == 1
    assert '+  "tables"' in out
    assert "-- matches tags.json" in out


def test_without_fakes_the_default_config_reports_the_missing_backend(capsys, fixtures_dir, cfg_path):
    status, _, err = run(capsys, "--config", cfg_path, "clean", fixtures_dir / "sample" / "segments.json")
    assert status == 1
    assert "Cleanup failed, showing original: segment 0: RuntimeError: ollama provider arrives" in err


def test_fake_backends_can_come_from_the_config_too(capsys, fixtures_dir, cfg_path):
    cfg_path.write_text("[whisper]\nmodel_size = fake\n[fake]\nfixture = tests/fixtures/sample\n", encoding="utf-8")
    status, out, _ = run(capsys, "--config", cfg_path, "transcribe", fixtures_dir / "sample" / "memo.wav")
    assert status == 0 and "-- matches segments.json" in out


def test_note_is_not_built_yet(capsys, fixtures_dir, cfg_path):
    status, _, err = run(capsys, "--config", cfg_path, "note", fixtures_dir / "sample")
    assert status == 2 and "not implemented" in err
