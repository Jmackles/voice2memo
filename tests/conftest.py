"""Shared pytest fixtures."""

from __future__ import annotations

import gc
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path

import pytest

from voice2memos import capture, config
from voice2memos.ui import player

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

# Everything a test window needs to run a memo end to end on fakes.
FAKE_CFG = """\
[capture]
input_device = fake
save_dir = memos
trash_dir = memos/trash
[whisper]
model_size = fake
[cleanup]
provider = fake
[entities]
provider = fake
[fake]
fixture = tests/fixtures/sample
"""


class FakeHotkeys:
    """Stands in for capture.HotkeyListener so no test installs a global keyboard hook."""

    instances: list[FakeHotkeys] = []

    def __init__(self, on_press, on_release, on_note, on_error) -> None:
        self.on_press, self.on_release, self.on_note, self.on_error = on_press, on_release, on_note, on_error
        self.binds: list[tuple[str, str]] = []
        self.stopped = False
        FakeHotkeys.instances.append(self)

    def bind(self, ptt: str, note: str) -> None:
        self.binds.append((ptt, note))

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture(autouse=True)
def no_global_hotkeys(monkeypatch):
    FakeHotkeys.instances.clear()
    monkeypatch.setattr(capture, "HotkeyListener", FakeHotkeys)
    return FakeHotkeys


class FakeStream:
    """Stands in for a sounddevice output stream: makes no sound; a test pulls frames by hand."""

    opened: list[FakeStream] = []
    fail = False

    def __init__(self, rate: int, channels: int, callback, device: int | None = None) -> None:
        self.rate, self.channels, self.callback, self.device = rate, channels, callback, device
        self.started = self.stopped = self.closed = False
        FakeStream.opened.append(self)

    def start(self) -> None:
        if FakeStream.fail:
            raise OSError("no output device")
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def pull(self, frames: int) -> bytes:
        """What PortAudio would ask for next: ``frames`` frames through the callback."""
        out = bytearray(frames * 2 * self.channels)
        self.callback(out, frames, None, None)
        return bytes(out)


class FakeInputStream:
    """Stands in for a sounddevice input stream: hears nothing until a test feeds it."""

    opened: list[FakeInputStream] = []
    fail = False

    def __init__(self, device: int | None, rate: int, callback) -> None:
        self.device, self.rate, self.callback = device, rate, callback
        self.started = self.stopped = self.closed = False
        FakeInputStream.opened.append(self)

    def start(self) -> None:
        if FakeInputStream.fail:
            raise OSError("device unavailable")
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def feed(self, frames: bytes) -> None:
        self.callback(frames, len(frames) // 2, None, None)


def fake_device(name: str, hostapi: int, inputs: int = 0, outputs: int = 0, rate: float = 44100.0) -> dict:
    return {"name": name, "hostapi": hostapi, "max_input_channels": inputs, "max_output_channels": outputs,
            "default_samplerate": rate}


def fake_device_table() -> capture.DeviceTable:
    """Shaped like a real Windows machine: MME (host API 0) cuts names and lists its mapper,
    DirectSound (1) lists the same devices again with whole names."""
    return capture.DeviceTable(
        [
            fake_device("Microsoft Sound Mapper - Input", 0, inputs=2),
            fake_device("Microphone (USB Headset)", 0, inputs=1, rate=48000.0),
            fake_device("Microsoft Sound Mapper - Output", 0, outputs=2),
            fake_device("Speakers (Realtek(R) Audio)", 0, outputs=8),
            fake_device("PHL 346E2C (NVIDIA High Definit", 0, outputs=2),
            fake_device("Headphones (USB Headset)", 0, outputs=2),
            fake_device("Microphone (USB Headset)", 1, inputs=1),
            fake_device("Speakers (Realtek(R) Audio)", 1, outputs=8),
            fake_device("PHL 346E2C (NVIDIA High Definition Audio)", 1, outputs=2),
            fake_device("Headphones (USB Headset)", 1, outputs=2),
        ],
        hostapi=0,
        default={"input": 1, "output": 3},
    )


class FakeAudio:
    """The fake audio system every test runs against; a test can swap the table or count rescans."""

    def __init__(self) -> None:
        self.table = fake_device_table()
        self.rescans = 0

    def rescan(self) -> None:
        self.rescans += 1


@pytest.fixture(autouse=True)
def no_sound(monkeypatch):
    """No test plays a sound, opens a microphone, or depends on this machine's devices."""
    FakeStream.opened.clear()
    FakeStream.fail = False
    FakeInputStream.opened.clear()
    FakeInputStream.fail = False
    audio = FakeAudio()
    monkeypatch.setattr(player, "open_output_stream", FakeStream)
    monkeypatch.setattr(capture, "open_input_stream", FakeInputStream)
    monkeypatch.setattr(capture, "device_table", lambda: audio.table)
    monkeypatch.setattr(capture, "rescan_devices", audio.rescan)
    return audio


class FakeModels:
    """What each provider and faster-whisper say they have, as a test sets it. Counts the asking."""

    def __init__(self) -> None:
        self.lists: dict[str, tuple[list[str], str]] = {
            "ollama": (["gemma4:e2b", "llama3.2:latest"], ""),
            "openai": ([], "No OpenAI API key: add one under API keys, or set OPENAI_API_KEY."),
            "anthropic": ([], "No Anthropic API key: add one under API keys, or set ANTHROPIC_API_KEY."),
            "fake": ([], "The fake provider answers from a fixture; the model is ignored."),
            "whisper": (["tiny", "small", "large-v3"], "Downloaded: small. Any other downloads the first time it's used."),
        }
        self.asked: list[tuple[str, str]] = []

    def available(self, provider: str, key: str = "", timeout: float = 5) -> tuple[list[str], str]:
        self.asked.append((provider, key))
        return self.lists[provider]

    def whisper(self) -> tuple[list[str], str]:
        self.asked.append(("whisper", ""))
        return self.lists["whisper"]


@pytest.fixture(autouse=True)
def no_model_servers(monkeypatch):
    """No test reaches Ollama, OpenAI or Anthropic, or waits for faster-whisper to import."""
    from voice2memos import cleanup, transcribe

    models = FakeModels()
    monkeypatch.setattr(cleanup, "available_models", models.available)
    monkeypatch.setattr(transcribe, "whisper_models", models.whisper)
    return models


def record_one(app, finish):
    """Click record twice and wait for the memo to finish; returns it."""
    app.buttons["record"].invoke()
    assert app.recording
    app.buttons["record"].invoke()
    finish(app)
    return app.memo


@pytest.fixture
def app_factory(tmp_path: Path, monkeypatch):
    """Build the real window, withdrawn, around a config written from ``text``.

    The default recordings folder is pinned inside pytest's temp folder, so a
    test that leaves save_dir out never writes into the real Documents.
    """
    from voice2memos.ui.app import App

    monkeypatch.setitem(config.DEFAULTS["capture"], "save_dir", str(tmp_path / "memos"))
    apps = []

    def make(text: str = "") -> App:
        try:
            root = tk.Tk()
        except tk.TclError as e:
            pytest.skip(f"tkinter has no display: {e}")
        root.withdraw()
        path = tmp_path / "voice2memos.cfg"
        if text:
            path.write_text(text, encoding="utf-8")
        app = App(root, path)
        apps.append(app)
        return app

    yield make
    for app in apps:
        try:
            if app.settings_dialog is not None and app.settings_dialog.window.winfo_exists():
                app.settings_dialog.close()
            app.root.after_cancel(app.drain_timer)
            app.root.after_cancel(app.check_timer)
            app.player.close()
            app.root.destroy()
        except tk.TclError:
            pass  # the test closed the window itself
    # Free the window here, on the main thread. Left to the garbage collector it can
    # be freed on a memo's worker thread, where every tk object waits a second for a
    # main loop that never runs under pytest.
    apps.clear()
    gc.collect()


@pytest.fixture
def app(app_factory):
    """The real window with default settings in its own config file."""
    return app_factory()


@pytest.fixture
def fake_app(app_factory):
    """The real window with fake recorder, transcriber and providers reading the sample fixture."""
    return app_factory(FAKE_CFG)


@pytest.fixture
def finish() -> Callable:
    """Pump the event loop until the worker has posted its last result."""

    def wait(app, seconds: float = 10) -> None:
        deadline = time.monotonic() + seconds
        while app.busy and time.monotonic() < deadline:
            app.root.update()
            time.sleep(0.01)
        app.root.update()
        assert not app.busy, "the pipeline never finished"

    return wait


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A throwaway Obsidian vault, fresh for every test that writes a note.

    It lives in pytest's temp folder rather than tests/vault/ because this repo
    can sit inside a sync folder like OneDrive, which locks freshly written
    folders and makes wiping them in place unreliable.
    """
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / "attachments").mkdir()
    return vault


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
