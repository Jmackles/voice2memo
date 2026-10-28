"""Stems, WAV writing, the fake recorder, and the hotkey listener's binding rules."""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from voice2memos import capture, config
from voice2memos.capture import FakeRecorder, HotkeyListener, MicRecorder, make_recorder
from voice2memos.errors import CaptureError

WHEN = datetime(2026, 10, 28, 17, 26, 0)


def test_new_stem_is_the_time_with_a_counter_only_on_collision(tmp_path):
    assert capture.new_stem(tmp_path, WHEN) == "20261028-172600"
    (tmp_path / "20261028-172600").mkdir()
    assert capture.new_stem(tmp_path, WHEN) == "20261028-172600-01"
    (tmp_path / "20261028-172600-01").mkdir()
    assert capture.new_stem(tmp_path, WHEN) == "20261028-172600-02"


def test_wav_round_trips_with_its_own_rate(tmp_path):
    frames = bytes(range(256)) * 4
    capture.write_wav(tmp_path / "deep" / "memo.wav", frames, 44100)
    assert capture.read_wav(tmp_path / "deep" / "memo.wav") == (frames, 44100)


def test_fake_recorder_answers_with_the_fixture_audio(fixtures_dir):
    recorder = FakeRecorder(fixtures_dir / "sample")
    recorder.start()
    assert recorder.recording
    frames, rate = recorder.stop()
    assert not recorder.recording
    assert (frames, rate) == capture.read_wav(fixtures_dir / "sample" / "memo.wav")


def test_fake_recorder_without_a_fixture_gives_a_second_of_silence():
    recorder = FakeRecorder()
    recorder.start()
    frames, rate = recorder.stop()
    assert rate == 16000 and len(frames) == 32000 and not any(frames)


def test_fake_recorder_can_fail():
    with pytest.raises(CaptureError, match="Microphone unavailable"):
        FakeRecorder(fail=True).start()


def test_make_recorder_follows_the_config(fixtures_dir):
    cfg = config.new_parser()
    mic = make_recorder(cfg, config.APP_DIR)
    assert isinstance(mic, MicRecorder) and mic.device is None
    cfg["capture"]["input_device"] = "Some Mic"
    assert make_recorder(cfg, config.APP_DIR).device == "Some Mic"
    cfg["capture"]["input_device"] = "fake"
    cfg["fake"]["fixture"] = "tests/fixtures/sample"
    cfg["fake"]["fail"] = "capture"
    fake = make_recorder(cfg, config.APP_DIR)
    assert isinstance(fake, FakeRecorder) and fake.fixture == fixtures_dir / "sample" and fake.fail


def test_a_bad_microphone_name_is_a_capture_error():
    with pytest.raises(CaptureError, match="Microphone unavailable"):
        MicRecorder("no such device, surely").start()


@pytest.fixture
def listener(no_global_hotkeys, monkeypatch):
    # The autouse fixture swapped the class out; this file tests the real one.
    monkeypatch.setattr(capture, "HotkeyListener", HotkeyListener.__mro__[0])
    errors = []
    made = HotkeyListener(lambda: None, lambda: None, lambda: None, errors.append)
    made.errors = errors
    yield made
    made.stop()


def test_an_unusable_chord_is_reported_and_the_other_still_binds(listener, caplog):
    caplog.set_level(logging.INFO)
    listener.bind("<f9>", "<nope>+x")
    assert listener.errors == ["Push-to-note hotkey not usable"]
    assert listener.listener is not None  # the push-to-talk chord is live
    assert "hotkeys bound: push-to-talk <f9>" in caplog.text


def test_blank_chords_bind_nothing(listener):
    listener.bind("", " ")
    assert listener.listener is None and listener.errors == []


def test_rebinding_replaces_the_listener(listener):
    listener.bind("<f9>", "<f10>")
    first = listener.listener
    listener.bind("<f8>", "<f10>")
    assert listener.listener is not first and not first.running
    listener.stop()
    assert listener.listener is None
