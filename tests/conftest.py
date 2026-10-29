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

    def __init__(self, rate: int, channels: int, callback) -> None:
        self.rate, self.channels, self.callback = rate, channels, callback
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


@pytest.fixture(autouse=True)
def no_sound(monkeypatch):
    FakeStream.opened.clear()
    FakeStream.fail = False
    monkeypatch.setattr(player, "open_output_stream", FakeStream)
    return FakeStream


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
            app.root.after_cancel(app.drain_timer)
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
