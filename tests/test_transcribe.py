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


SEGMENTS = [
    Segment(0, 0.0, 1.0, " first part", "First part.", [{"word": " first", "start": 0.0, "end": 0.5}]),
    Segment(1, 1.2, 2.0, " second part", "Second part.", [{"word": " second", "start": 1.2, "end": 1.6}]),
]


def test_display_text_is_a_paragraph_per_segment():
    assert transcribe.display_text(SEGMENTS) == "first part\n\nsecond part"


def test_an_edit_with_the_same_paragraphs_keeps_timing_and_untouched_words():
    edited = transcribe.apply_edit(SEGMENTS, "first part\n\nsecond   part, changed\n")
    assert [(s.index, s.start, s.end, s.text, s.cleaned) for s in edited] == [
        (0, 0.0, 1.0, "first part", ""),
        (1, 1.2, 2.0, "second part, changed", ""),
    ]
    assert edited[0].words == SEGMENTS[0].words and edited[1].words == []


def test_an_edit_with_other_paragraphs_spans_the_whole_memo():
    edited = transcribe.apply_edit(SEGMENTS, "all\none paragraph")
    assert [(s.index, s.start, s.end, s.text) for s in edited] == [(0, 0.0, 2.0, "all one paragraph")]
    edited = transcribe.apply_edit([], "typed\n\nfrom scratch")
    assert [(s.start, s.end, s.text) for s in edited] == [(0.0, 0.0, "typed"), (0.0, 0.0, "from scratch")]


# --- the real wrapper, over a fake WhisperModel --------------------------------------

from .conftest import FakeWhisperModel, whisper_segment  # noqa: E402


@pytest.mark.parametrize(
    "device, compute, plan",
    [
        ("auto", "auto", [("cuda", "float16"), ("cpu", "int8")]),
        ("cpu", "auto", [("cpu", "int8")]),
        ("cuda", "int8_float16", [("cuda", "int8_float16")]),
        ("auto", "int8", [("cuda", "int8"), ("cpu", "int8")]),
    ],
)
def test_resolve_whisper_plans_devices_and_compute_types(device, compute, plan):
    assert transcribe.resolve_whisper(device, compute) == plan


def wrapper(**overrides) -> WhisperTranscriber:
    args = {"model_size": "small", "device": "auto", "compute_type": "auto", "language": "en", "glossary": ""}
    args.update(overrides)
    return WhisperTranscriber(**args)


def test_whisper_segments_come_out_rounded_with_their_words(fixtures_dir, caplog):
    w = wrapper(glossary="Bookinnit\n\n  Oakwillow  \n")
    with caplog.at_level(logging.INFO):
        segments = w.transcribe(fixtures_dir / "sample" / "memo.wav", "stem")
    assert [(s.index, s.start, s.end, s.text, s.cleaned) for s in segments] == [
        (0, 0.0, 1.2, " hello there", ""),
        (1, 1.4, 2.4, " again", ""),
    ]
    assert segments[0].words == [{"word": " hello", "start": 0.0, "end": 0.5}, {"word": " there", "start": 0.6, "end": 1.2}]
    assert segments[1].words == []
    [call] = w.model.calls
    assert call["language"] == "en" and call["word_timestamps"] and call["vad_filter"]
    assert call["initial_prompt"] == "Bookinnit, Oakwillow"
    assert w.loaded_on == ("cuda", "float16") and w.note == ""
    assert "stem: 2.4s of audio on cuda float16" in caplog.messages


def test_the_model_is_loaded_once_and_shared(fixtures_dir):
    a, b = wrapper(), wrapper()
    a.load()
    b.load()
    assert FakeWhisperModel.made == [("small", "cuda", "float16")]
    assert a.model is b.model
    wrapper(model_size="tiny").load()
    assert FakeWhisperModel.made[-1] == ("tiny", "cuda", "float16")


def test_no_cuda_falls_back_to_cpu_int8_with_a_note(caplog):
    FakeWhisperModel.fail_on = {"cuda"}
    w = wrapper()
    with caplog.at_level(logging.WARNING):
        w.load()
    assert FakeWhisperModel.made == [("small", "cpu", "int8")]
    assert w.loaded_on == ("cpu", "int8") and w.note == "GPU unavailable, using CPU"
    assert "won't load on cuda float16" in caplog.text


def test_an_explicit_cuda_that_fails_is_an_error_not_a_fallback():
    FakeWhisperModel.fail_on = {"cuda"}
    with pytest.raises(TranscribeError, match="Whisper model unavailable"):
        wrapper(device="cuda").load()
    assert FakeWhisperModel.made == []


def test_nothing_heard_is_its_own_error(fixtures_dir):
    FakeWhisperModel.answer = []
    with pytest.raises(TranscribeError, match="Nothing heard"):
        wrapper().transcribe(fixtures_dir / "sample" / "memo.wav")


def test_a_decode_failure_is_a_transcribe_error(fixtures_dir, monkeypatch):
    w = wrapper()
    w.load()
    monkeypatch.setattr(w.model, "transcribe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad file")))
    with pytest.raises(TranscribeError, match="Transcription failed") as info:
        w.transcribe(fixtures_dir / "sample" / "memo.wav")
    assert "RuntimeError: bad file" in info.value.detail


def test_make_transcriber_builds_the_real_one_from_the_config():
    cfg = config.new_parser()
    cfg["whisper"]["model_size"] = "medium"
    cfg["whisper"]["glossary"] = "Bookinnit"
    w = make_transcriber(cfg, config.APP_DIR)
    assert isinstance(w, WhisperTranscriber)
    assert (w.model_size, w.device, w.compute_type, w.language, w.glossary) == ("medium", "auto", "auto", "en", "Bookinnit")
