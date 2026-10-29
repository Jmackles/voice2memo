"""The banner's media player: playback through a fake stream, then the bar inside the real window."""

from __future__ import annotations

import logging
import tkinter as tk
import wave
from tkinter import ttk

import numpy as np
import pytest

from voice2memos import config
from voice2memos.ui.app import ICONS
from voice2memos.ui.player import Playback, click_to_jump, format_time

from .conftest import FAKE_CFG, FakeStream, record_one


@pytest.fixture
def wav(fixtures_dir):
    return fixtures_dir / "sample" / "memo.wav"  # one second, 16 kHz mono


@pytest.fixture
def loaded(wav):
    playback = Playback()
    playback.load(wav)
    return playback


@pytest.mark.parametrize("seconds, shown", [(0, "0:00"), (7.9, "0:07"), (61, "1:01"), (600, "10:00")])
def test_format_time(seconds, shown):
    assert format_time(seconds) == shown


def test_load_reads_the_whole_file(loaded):
    assert (loaded.rate, loaded.channels, loaded.duration, loaded.seconds) == (16000, 1, 1.0, 0)
    assert not loaded.playing


def test_load_refuses_anything_but_16_bit(tmp_path):
    path = tmp_path / "eight.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(1)
        out.setframerate(8000)
        out.writeframes(bytes(800))
    with pytest.raises(ValueError, match="16-bit"):
        Playback().load(path)


def test_play_feeds_frames_in_order_and_moves_the_position(loaded):
    loaded.play()
    [stream] = FakeStream.opened
    assert (stream.rate, stream.channels, stream.started) == (16000, 1, True)
    assert stream.pull(8000) == loaded.frames[:16000]
    assert loaded.seconds == 0.5
    assert stream.pull(10) == loaded.frames[16000:16020]


def test_the_end_pads_with_silence_and_finishes(loaded):
    loaded.seek(0.9)
    loaded.play()
    out = FakeStream.opened[0].pull(2000)
    assert out[:3200] == loaded.frames[-3200:] and not any(out[3200:])
    assert loaded.finished and loaded.seconds == 1.0


def test_seek_clamps_and_playback_resumes_there(loaded):
    loaded.seek(-1)
    assert loaded.position == 0
    loaded.seek(5)
    assert loaded.position == loaded.total
    loaded.seek(0.25)
    loaded.play()
    assert FakeStream.opened[0].pull(10) == loaded.frames[8000:8020]


def test_play_at_the_end_starts_over(loaded):
    loaded.seek(1.0)
    loaded.play()
    assert loaded.position == 0


def test_pause_closes_the_stream_and_keeps_the_place(loaded):
    loaded.play()
    stream = FakeStream.opened[0]
    stream.pull(4000)
    loaded.pause()
    assert stream.stopped and stream.closed and not loaded.playing
    assert loaded.seconds == 0.25


def test_a_failed_start_closes_the_stream_and_raises(loaded):
    FakeStream.fail = True
    with pytest.raises(OSError):
        loaded.play()
    assert FakeStream.opened[0].closed and not loaded.playing


# --- the bar in the window -----------------------------------------------------


def test_the_player_starts_empty_and_disabled(fake_app):
    bar = fake_app.player
    assert bar.play_button.instate(["disabled"]) and bar.scale.instate(["disabled"])
    assert bar.discard_button.winfo_manager() == ""
    assert bar.frame.winfo_manager() == "pack"


def test_a_recording_loads_with_its_discard_button(fake_app, finish):
    memo = record_one(fake_app, finish)
    bar = fake_app.player
    assert bar.playback.path == memo.wav
    assert not bar.play_button.instate(["disabled"])
    assert bar.discard_button.winfo_manager() == "grid"
    assert bar.time_label.cget("text") == "0:00 / 0:01"


def test_an_opened_file_has_no_discard_button(fake_app, finish, monkeypatch, wav):
    from voice2memos.ui import app as app_module

    monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **kw: str(wav))
    fake_app.buttons["open_file"].invoke()
    finish(fake_app)
    assert fake_app.player.playback.path == fake_app.memo.wav
    assert fake_app.player.discard_button.winfo_manager() == ""


def test_play_pauses_and_shows_the_state(fake_app, finish):
    record_one(fake_app, finish)
    bar = fake_app.player
    bar.play_button.invoke()
    assert bar.playback.playing and bar.play_button.cget("text") == ICONS["pause"]
    FakeStream.opened[-1].pull(8000)
    bar._tick()
    assert bar.scale.get() == 0.5 and bar.time_label.cget("text") == "0:00 / 0:01"
    bar.play_button.invoke()
    assert not bar.playback.playing and bar.play_button.cget("text") == ICONS["play"]


def test_reaching_the_end_stops_on_the_next_tick(fake_app, finish):
    record_one(fake_app, finish)
    bar = fake_app.player
    bar.play()
    FakeStream.opened[-1].pull(20000)
    bar._tick()
    assert not bar.playback.playing and bar.play_button.cget("text") == ICONS["play"]
    assert bar.timer is None


def test_letting_go_of_the_slider_seeks(fake_app, finish):
    record_one(fake_app, finish)
    bar = fake_app.player
    bar.scale.event_generate("<ButtonPress-1>")
    assert bar.dragging
    bar.scale.set(0.75)
    bar._tick()  # playback's own position doesn't fight the drag
    assert bar.scale.get() == 0.75
    bar.scale.event_generate("<ButtonRelease-1>")
    assert not bar.dragging and bar.playback.seconds == 0.75


@pytest.fixture
def laid_out(app):
    """A real slider in a window that's mapped (off screen), so pixels mean something."""
    window = tk.Toplevel(app.root)
    window.geometry("+-3000+-3000")
    scale = ttk.Scale(window, from_=0, to=100, length=200)
    scale.pack()
    window.update()
    yield scale
    window.destroy()


def at(scale: ttk.Scale, value: float) -> int:
    """The x pixel ttk itself puts ``value`` at."""
    return next(x for x in range(scale.winfo_width()) if scale.get(x, 5) >= value)


def test_a_click_on_the_line_jumps_there_and_holds_for_a_drag(laid_out):
    scale = laid_out
    events = []
    click_to_jump(scale, lambda: events.append("press"), lambda value: events.append(round(value)))
    scale.event_generate("<ButtonPress-1>", x=at(scale, 75), y=5)
    assert round(scale.get()) == 75 and events == ["press"]  # straight there, not one step toward it
    scale.event_generate("<B1-Motion>", x=at(scale, 25), y=5)
    assert round(scale.get()) == 25 and events == ["press"]  # following, not yet let go
    scale.event_generate("<B1-Motion>", x=scale.winfo_width() + 300, y=5)
    assert scale.get() == 100  # dragged past the end stops at the end
    scale.event_generate("<ButtonRelease-1>", x=scale.winfo_width() + 300, y=5)
    assert events == ["press", 100]


def test_moving_without_a_press_or_on_a_disabled_line_does_nothing(laid_out):
    scale = laid_out
    events = []
    click_to_jump(scale, lambda: events.append("press"), events.append)
    scale.set(10)
    scale.event_generate("<B1-Motion>", x=at(scale, 90), y=5)
    scale.event_generate("<ButtonRelease-1>", x=at(scale, 90), y=5)
    scale.state(["disabled"])
    scale.event_generate("<ButtonPress-1>", x=at(scale, 90), y=5)
    scale.event_generate("<ButtonRelease-1>", x=at(scale, 90), y=5)
    assert scale.get() == 10 and events == []


def test_recording_silences_and_locks_the_player(fake_app, finish):
    record_one(fake_app, finish)
    bar = fake_app.player
    bar.play()
    fake_app.buttons["record"].invoke()
    assert not bar.playback.playing and bar.play_button.instate(["disabled"])
    fake_app.buttons["record"].invoke()
    finish(fake_app)
    assert not bar.play_button.instate(["disabled"])


def test_no_output_device_shows_in_the_status_line(fake_app, finish):
    record_one(fake_app, finish)
    FakeStream.fail = True
    fake_app.player.play_button.invoke()
    assert fake_app.status_label.cget("text") == "Can't play audio, see log"
    assert fake_app.player.play_button.cget("text") == ICONS["play"]


def test_closing_the_window_stops_playback(app_factory, finish):
    app = app_factory(FAKE_CFG)
    record_one(app, finish)
    app.player.play()
    stream = FakeStream.opened[-1]
    app.close()
    assert stream.closed


# --- volume and the log ----------------------------------------------------------


def test_volume_scales_the_samples(loaded):
    loaded.volume = 0.5
    loaded.play()
    out = FakeStream.opened[0].pull(100)
    expected = (np.frombuffer(loaded.frames[:200], "<i2") * 0.5).astype("<i2").tobytes()
    assert out == expected and out != loaded.frames[:200]
    loaded.volume = 0
    assert not any(FakeStream.opened[0].pull(100))


def test_the_volume_slider_sets_playback_live_and_logs_where_it_lands(fake_app, caplog):
    bar = fake_app.player
    bar._volume_moved("40")
    assert bar.playback.volume == 0.4
    bar.set_volume(140)
    assert bar.volume_percent == 100 and bar.playback.volume == 1.0
    with caplog.at_level(logging.INFO):
        bar.volume.event_generate("<ButtonPress-1>")
        bar.volume.set(35)  # as a drag would
        assert bar.playback.volume == 0.35  # live, before it's let go
        bar.volume.event_generate("<ButtonRelease-1>")
    assert "volume 35%" in caplog.messages


def test_the_volume_comes_back_next_time(app_factory, tmp_path):
    app = app_factory(FAKE_CFG + "[window]\nvolume = 60\n")
    assert app.player.volume_percent == 60 and app.player.playback.volume == 0.6
    app.player.set_volume(25)
    app.close()  # withdrawn, so no size is saved, but the volume is
    assert config.load(tmp_path / "voice2memos.cfg")[0]["window"]["volume"] == "25"


def test_each_player_action_logs_one_line(fake_app, finish, caplog):
    memo = record_one(fake_app, finish)
    bar = fake_app.player
    with caplog.at_level(logging.INFO):
        bar.play_button.invoke()
        FakeStream.opened[-1].pull(8000)
        bar.play_button.invoke()
        bar._release(0.75)
        bar.play()
        FakeStream.opened[-1].pull(20000)
        bar._tick()
    assert [m for m in caplog.messages if m.startswith(memo.stem)] == [
        f"{memo.stem}: play from 0:00",
        f"{memo.stem}: pause at 0:00",
        f"{memo.stem}: seek to 0:00",
        f"{memo.stem}: play from 0:00",
        f"{memo.stem}: played to the end",
    ]
