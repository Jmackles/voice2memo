"""Segments round-trip through JSON; the fake transcriber answers from a fixture; the real one says it's not built."""

from __future__ import annotations

import logging

import pytest

from voice2memos import config, transcribe
from voice2memos.errors import TranscribeError
from voice2memos.transcribe import FakeWhisper, Segment, WhisperTranscriber, make_transcriber


def test_segments_round_trip_through_json(tmp_path):
    segments = [Segment(0, 0.0, 1.5, " hello", words=[{"word": " hello", "start": 0.0, "end": 1.5}]), Segment(1, 1.6, 2.0, " there", "There.")]
    path = tmp_path / "segments.json"
    transcribe.save_segments(segments, path)
    assert transcribe.load_segments(path) == segments


def test_joined_text_prefers_cleaned_when_asked():
    segments = [Segment(0, 0, 1, " um hello ", "Hello."), Segment(1, 1, 2, " there ")]
    assert transcribe.joined_text(segments) == "um hello there"
    assert transcribe.joined_text(segments, cleaned=True) == "Hello. there"


@pytest.mark.parametrize("content, message", [(None, "missing"), ("not json", "unreadable"), ('[{"nope": 1}]', "unreadable")])
def test_load_segments_names_the_problem_without_quoting(tmp_path, content, message):
    path = tmp_path / "segments.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(TranscribeError) as info:
        transcribe.load_segments(path)
    assert message in info.value.message
    assert "nope" not in info.value.message


def test_fake_whisper_reads_the_folder_beside_the_wav(fixtures_dir, caplog):
    with caplog.at_level(logging.INFO):
        segments = FakeWhisper().transcribe(fixtures_dir / "sample" / "memo.wav")
    assert [s.index for s in segments] == [0, 1]
    assert all(s.cleaned == "" for s in segments)
    assert any("sample: transcribe (fake whisper) start" in m for m in caplog.messages)
    assert any("sample: transcribe (fake whisper) done in" in m for m in caplog.messages)


def test_fake_whisper_reads_a_named_fixture_for_any_wav(fixtures_dir, tmp_path):
    wav = tmp_path / "20261028-172600" / "memo.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"RIFF")
    segments = FakeWhisper(fixtures_dir / "sample").transcribe(wav)
    assert len(segments) == 2


def test_fake_whisper_needs_the_wav_to_exist(tmp_path):
    with pytest.raises(TranscribeError, match="Audio file missing"):
        FakeWhisper().transcribe(tmp_path / "gone.wav")


def test_fake_whisper_can_be_told_to_fail(fixtures_dir, caplog):
    with caplog.at_level(logging.INFO), pytest.raises(TranscribeError, match="Transcription failed"):
        FakeWhisper(fail=True).transcribe(fixtures_dir / "sample" / "memo.wav")
    assert any("transcribe (fake whisper) failed after" in m for m in caplog.messages)


def test_make_transcriber_follows_the_config(fixtures_dir):
    cfg = config.new_parser()
    assert isinstance(make_transcriber(cfg, config.APP_DIR), WhisperTranscriber)
    cfg["whisper"]["model_size"] = "fake"
    cfg["fake"]["fixture"] = "tests/fixtures/sample"
    cfg["fake"]["fail"] = "cleanup transcribe"
    fake = make_transcriber(cfg, config.APP_DIR)
    assert isinstance(fake, FakeWhisper)
    assert fake.fixture == fixtures_dir / "sample"
    assert fake.fail


def test_the_real_transcriber_says_it_is_not_built(fixtures_dir):
    with pytest.raises(TranscribeError, match="not built yet"):
        make_transcriber(config.new_parser(), config.APP_DIR).transcribe(fixtures_dir / "sample" / "memo.wav")
