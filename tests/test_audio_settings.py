"""Settings → Audio: pick the microphone and the speakers, test both, rescan; the app follows."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from voice2memos import config

from .conftest import FAKE_CFG, FakeInputStream, FakeStream, fake_device, record_one

MIC = ("capture", "input_device")
SPEAKERS = ("playback", "output_device")


@pytest.fixture
def dialog(app):
    app.open_settings()
    return app.settings_dialog


def shown(dialog, field) -> list[str]:
    return list(dialog.widgets[field].cget("values"))


def pick(dialog, field, text: str) -> None:
    dialog.vars[field].set(text)
    dialog.widgets[field].event_generate("<<ComboboxSelected>>")


def test_the_lists_offer_the_system_default_then_each_device(dialog):
    assert shown(dialog, MIC) == ["System default (Microphone (USB Headset))", "Microphone (USB Headset)"]
    assert shown(dialog, SPEAKERS) == [
        "System default (Speakers (Realtek(R) Audio))",
        "Speakers (Realtek(R) Audio)",
        "PHL 346E2C (NVIDIA High Definition Audio)",
        "Headphones (USB Headset)",
    ]
    assert dialog.vars[MIC].get().startswith("System default")
    assert dialog.values()[MIC] == "" and dialog.values()[SPEAKERS] == ""


def test_a_picked_device_is_saved_by_name_and_the_player_follows_it(app, dialog):
    pick(dialog, SPEAKERS, "Headphones (USB Headset)")
    pick(dialog, MIC, "Microphone (USB Headset)")
    dialog.save()
    saved = config.load(app.config_path)[0]
    assert saved["playback"]["output_device"] == "Headphones (USB Headset)"
    assert saved["capture"]["input_device"] == "Microphone (USB Headset)"
    assert app.player.playback.output_device == "Headphones (USB Headset)"


def test_a_device_that_is_not_connected_stays_selected_and_saved_as_is(app_factory):
    app = app_factory("[capture]\ninput_device = Gone Headset\n[playback]\noutput_device = fake\n")
    app.open_settings()
    dialog = app.settings_dialog
    assert dialog.vars[MIC].get() == "Gone Headset (not connected)"
    assert shown(dialog, MIC)[-1] == "Gone Headset (not connected)"
    assert dialog.values()[MIC] == "Gone Headset"


def test_the_fake_recorder_shows_as_such(app_factory):
    app = app_factory(FAKE_CFG)
    app.open_settings()
    assert app.settings_dialog.vars[MIC].get() == "Fake recorder, for testing"
    assert app.settings_dialog.values()[MIC] == "fake"


def test_the_microphone_test_shows_the_level_until_stopped(dialog, caplog):
    pick(dialog, MIC, "Microphone (USB Headset)")
    with caplog.at_level(logging.INFO):
        dialog.mic_test_button.invoke()
    [stream] = FakeInputStream.opened
    assert stream.device == 1 and stream.started
    assert dialog.mic_test_button.cget("text") == "Stop"
    stream.feed(np.full(1600, 32767, "<i2").tobytes())
    dialog._tick_test()
    assert float(dialog.level_bar["value"]) == pytest.approx(1.0, abs=0.01)
    dialog.mic_test_button.invoke()
    assert stream.closed and float(dialog.level_bar["value"]) == 0
    assert dialog.mic_test_button.cget("text") == "Test"
    assert "settings: microphone test on Microphone (USB Headset)" in caplog.messages


def test_picking_another_microphone_mid_test_carries_on_with_it(dialog):
    dialog.mic_test_button.invoke()
    first = FakeInputStream.opened[0]
    assert first.device is None  # the system default
    pick(dialog, MIC, "Microphone (USB Headset)")
    assert first.closed and FakeInputStream.opened[1].device == 1
    assert dialog.meter is not None


def test_a_microphone_that_will_not_open_says_so(dialog):
    FakeInputStream.fail = True
    dialog.mic_test_button.invoke()
    assert dialog.meter is None and dialog.audio_note.cget("text") == "Microphone unavailable"


def test_the_tone_plays_on_the_picked_speakers_and_closes_after(dialog):
    pick(dialog, SPEAKERS, "Headphones (USB Headset)")
    dialog.tone_button.invoke()
    [stream] = FakeStream.opened
    assert stream.device == 5 and stream.rate == 44100
    assert dialog.audio_note.cget("text") == "You should hear a short beep."
    stream.pull(44100)  # past the end
    dialog._tick_test()
    assert stream.closed and dialog.test_tone is None


def test_closing_the_dialog_stops_both_tests(dialog):
    dialog.mic_test_button.invoke()
    dialog.tone_button.invoke()
    dialog.cancel()
    assert FakeInputStream.opened[0].closed and FakeStream.opened[0].closed


def test_rescan_finds_a_device_plugged_in_meanwhile_and_keeps_the_picks(app, dialog, no_sound):
    pick(dialog, SPEAKERS, "Headphones (USB Headset)")
    rescans = no_sound.rescans
    no_sound.table.devices.append(fake_device("Speakers (Bluetooth)", 0, outputs=2))
    dialog.rescan()
    assert no_sound.rescans == rescans + 1
    assert "Speakers (Bluetooth)" in shown(dialog, SPEAKERS)
    assert dialog.vars[SPEAKERS].get() == "Headphones (USB Headset)"
    assert dialog.audio_note.cget("text") == "Device list updated."


def test_opening_settings_rescans_but_not_while_recording(fake_app, no_sound):
    fake_app.open_settings()
    assert no_sound.rescans == 1
    fake_app.settings_dialog.cancel()
    fake_app.buttons["record"].invoke()
    fake_app.open_settings()
    assert no_sound.rescans == 1
    fake_app.settings_dialog.rescan()
    assert fake_app.settings_dialog.audio_note.cget("text") == "Stop recording to rescan"


# --- the app, with a device that isn't there -------------------------------------


def test_playing_on_missing_speakers_uses_the_default_and_says_so(app_factory, finish):
    app = app_factory(FAKE_CFG + "[playback]\noutput_device = Gone Headset\n")
    record_one(app, finish)
    app.player.play_button.invoke()
    assert FakeStream.opened[-1].device is None
    assert app.status_label.cget("text") == "Gone Headset isn't connected, using the default"


def test_switching_speakers_mid_play_carries_on_there(fake_app, finish):
    record_one(fake_app, finish)
    fake_app.player.play()
    FakeStream.opened[-1].pull(4000)
    fake_app.player.set_output_device("Headphones (USB Headset)")
    old, new = FakeStream.opened[-2:]
    assert old.closed and new.device == 5 and new.started
    assert fake_app.player.playback.seconds == 0.25


def test_recording_from_a_missing_microphone_uses_the_default_and_says_so(app_factory):
    app = app_factory("[capture]\ninput_device = Gone Headset\nsave_dir = memos\n")
    app.buttons["record"].invoke()
    assert app.recording and FakeInputStream.opened[0].device is None
    assert app.status_label.cget("text") == "Recording… Gone Headset isn't connected, using the default"
