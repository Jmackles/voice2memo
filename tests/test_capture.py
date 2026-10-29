"""Stems, WAV writing, recorders, audio devices, and the hotkey listener's binding rules."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pytest

from voice2memos import capture, config
from voice2memos.capture import FakeRecorder, HotkeyListener, MicRecorder, make_recorder
from voice2memos.errors import CaptureError

from .conftest import FakeInputStream

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
    assert isinstance(mic, MicRecorder) and mic.device == ""
    cfg["capture"]["input_device"] = "Some Mic"
    assert make_recorder(cfg, config.APP_DIR).device == "Some Mic"
    cfg["capture"]["input_device"] = "fake"
    cfg["fake"]["fixture"] = "tests/fixtures/sample"
    cfg["fake"]["fail"] = "capture"
    fake = make_recorder(cfg, config.APP_DIR)
    assert isinstance(fake, FakeRecorder) and fake.fixture == fixtures_dir / "sample" and fake.fail


# --- devices -----------------------------------------------------------------


def test_each_device_is_listed_once_by_its_whole_name():
    assert capture.list_devices("input") == ["Microphone (USB Headset)"]
    assert capture.list_devices("output") == [
        "Speakers (Realtek(R) Audio)",
        "PHL 346E2C (NVIDIA High Definition Audio)",  # MME cut it short; DirectSound had it whole
        "Headphones (USB Headset)",
    ]


def test_the_system_defaults_by_name():
    assert capture.default_device_name("input") == "Microphone (USB Headset)"
    assert capture.default_device_name("output") == "Speakers (Realtek(R) Audio)"


def test_no_default_device_is_a_blank_name(no_sound):
    no_sound.table.default["input"] = None
    assert capture.default_device_name("input") == ""


def test_a_name_resolves_to_its_device_on_the_default_host_api():
    assert capture.resolve_device("", "output") == (None, "")
    assert capture.resolve_device("Headphones (USB Headset)", "output") == (5, "")
    assert capture.resolve_device("PHL 346E2C (NVIDIA High Definition Audio)", "output") == (4, "")


def test_a_device_that_is_not_connected_falls_back_to_the_default(caplog):
    with caplog.at_level(logging.WARNING):
        assert capture.resolve_device("Gone Headset", "input") == (None, "Gone Headset isn't connected, using the default")
    assert "input device 'Gone Headset' isn't connected" in caplog.text


def test_the_microphone_opens_the_chosen_device_at_its_own_rate():
    mic = MicRecorder("Microphone (USB Headset)")
    mic.start()
    [stream] = FakeInputStream.opened
    assert (stream.device, stream.rate, stream.started) == (1, 48000, True)
    assert mic.note == ""
    stream.feed(bytes(range(10)))
    assert mic.stop() == (bytes(range(10)), 48000)
    assert stream.closed


def test_a_missing_microphone_records_from_the_default_with_a_note():
    mic = MicRecorder("Gone Headset")
    mic.start()
    assert FakeInputStream.opened[0].device is None
    assert mic.note == "Gone Headset isn't connected, using the default"


def test_no_microphone_at_all_is_a_capture_error(no_sound):
    no_sound.table.default["input"] = None
    with pytest.raises(CaptureError, match="No microphone found"):
        MicRecorder().start()


def test_a_microphone_that_will_not_open_is_a_capture_error():
    FakeInputStream.fail = True
    with pytest.raises(CaptureError, match="Microphone unavailable"):
        MicRecorder().start()


@pytest.mark.parametrize("amplitude, level", [(0, 0.0), (32767, 1.0), (328, 1 / 3)])
def test_the_level_meter_reads_decibels(amplitude, level):
    meter = capture.LevelMeter("")
    assert meter.start() == ""
    FakeInputStream.opened[0].feed(np.full(1600, amplitude, "<i2").tobytes())
    assert meter.level == pytest.approx(level, abs=0.01)
    meter.stop()
    assert meter.level == 0 and FakeInputStream.opened[0].closed


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
