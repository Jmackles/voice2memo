"""Media player in the banner: play, pause, scrub and set the volume of the current memo's audio.

``Playback`` holds the WAV and a sounddevice output stream, and knows nothing
of tk. The stream's callback runs on PortAudio's thread; it copies frames,
scales them by the volume, and advances the position, nothing else.
``PlayerBar`` is the widgets: it polls the position with ``after`` while
playing. The discard button shows only for a recording; an opened file is the
user's own and has nothing to discard.

Each thing the user does here logs one line (play, pause, seek, volume, the
end reached); dragging a slider logs where it was let go, not every pixel.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
import wave
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk

import numpy as np

from voice2memos import capture
from voice2memos.transcribe import format_time  # noqa: F401  (the bar's clock; tests import it from here too)

log = logging.getLogger(__name__)

TICK_MS = 50


def open_output_stream(rate: int, channels: int, callback: Callable, device: int | None = None) -> object:
    """A sounddevice stream of 16-bit frames fed by ``callback``; None is the default device. Tests swap this out."""
    import sounddevice as sd

    return sd.RawOutputStream(samplerate=rate, channels=channels, dtype="int16", callback=callback, device=device)


def click_to_jump(scale: ttk.Scale, on_press: Callable[[], None], on_release: Callable[[float], None]) -> None:
    """Clicking anywhere on the line puts the knob there and holds it, so the same press can drag.

    Replaces ttk's own press, which steps the value one unit toward the
    click and repeats while held: on a scrub bar in seconds, a crawl. Setting
    the value fires the scale's ``command`` as usual, so whatever follows the
    knob still does. ``on_release`` gets the value where it was let go.
    """
    held = False

    def press(event: tk.Event) -> str:
        nonlocal held
        if scale.instate(["disabled"]):
            return "break"
        held = True
        on_press()
        scale.set(scale.get(event.x, event.y))  # get(x, y) clamps to the ends, so off-line drags stop there
        return "break"

    def drag(event: tk.Event) -> str:
        if held:
            scale.set(scale.get(event.x, event.y))
        return "break"

    def release(_event: tk.Event) -> str:
        nonlocal held
        if held:
            held = False
            on_release(scale.get())
        return "break"

    scale.bind("<ButtonPress-1>", press)
    scale.bind("<B1-Motion>", drag)
    scale.bind("<ButtonRelease-1>", release)


def tone(seconds: float = 0.6, rate: int = 44100, hz: float = 440, loudness: float = 0.25) -> bytes:
    """A short sine with soft edges, as mono 16-bit frames: the speaker test in Settings."""
    t = np.arange(int(seconds * rate)) / rate
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.03)  # no click at either end
    return (np.sin(2 * np.pi * hz * t) * fade * loudness * 32767).astype("<i2").tobytes()


class Playback:
    """One loaded WAV, its position in frames, the volume, and the stream playing it when there is one."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.frames = b""
        self.rate = 0
        self.frame_bytes = 2
        self.channels = 1
        self.position = 0
        self.finished = False
        self.volume = 1.0  # 0 to 1, read by the callback on every block
        self.output_device = ""  # a name from capture.list_devices, blank for the system default
        self.note = ""  # set by play when the chosen device wasn't there
        self.stream = None
        self.lock = threading.Lock()

    @property
    def total(self) -> int:
        return len(self.frames) // self.frame_bytes

    @property
    def duration(self) -> float:
        return self.total / self.rate if self.rate else 0.0

    @property
    def seconds(self) -> float:
        return self.position / self.rate if self.rate else 0.0

    @property
    def playing(self) -> bool:
        return self.stream is not None

    def load(self, path: Path) -> None:
        """Raises OSError, wave.Error, or ValueError for anything but 16-bit PCM."""
        self.unload()
        with wave.open(str(path), "rb") as src:
            if src.getsampwidth() != 2:
                raise ValueError(f"{path.name} is not 16-bit PCM")
            rate, channels, frames = src.getframerate(), src.getnchannels(), src.readframes(src.getnframes())
        self.load_frames(frames, rate, channels, path)

    def load_frames(self, frames: bytes, rate: int, channels: int = 1, path: Path | None = None) -> None:
        self.unload()
        self.path, self.rate, self.channels, self.frames = path, rate, channels, frames
        self.frame_bytes = 2 * channels

    def unload(self) -> None:
        self.pause()
        self.path, self.frames, self.rate, self.position, self.finished = None, b"", 0, 0, False

    def play(self) -> None:
        """From the current position, or from the start once it reached the end.

        Raises whatever the output device raises; nothing is left open then.
        """
        if self.playing or not self.total:
            return
        with self.lock:
            if self.position >= self.total:
                self.position = 0
            self.finished = False
        device, self.note = capture.resolve_device(self.output_device, "output")
        stream = open_output_stream(self.rate, self.channels, self._callback, device)
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self.stream = stream

    def pause(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def seek(self, seconds: float) -> None:
        with self.lock:
            self.position = max(0, min(self.total, round(seconds * self.rate)))
            self.finished = False

    def _callback(self, outdata, frames, time_info, status) -> None:
        # PortAudio's thread: copy the next frames, scale them, move the position, nothing else.
        with self.lock:
            start = self.position * self.frame_bytes
            chunk = self.frames[start : start + frames * self.frame_bytes]
            self.position += len(chunk) // self.frame_bytes
            if len(chunk) < len(outdata):
                self.finished = True  # the UI thread closes the stream on its next tick
        volume = self.volume
        if volume < 1 and chunk:
            chunk = (np.frombuffer(chunk, "<i2") * volume).astype("<i2").tobytes()
        outdata[: len(chunk)] = chunk
        outdata[len(chunk) :] = bytes(len(outdata) - len(chunk))


class PlayerBar:
    """Play/pause, a scrub slider, the time, a volume slider, and a discard button for recordings."""

    def __init__(
        self,
        parent: tk.Misc,
        icons: dict[str, str],
        on_discard: Callable[[], None],
        on_status: Callable[[str, str], None],
        icon_style: str = "",
    ) -> None:
        """``on_status(text, level)`` reaches the status line. ``icon_style`` names a ttk
        style prefix with .TButton and .TLabel variants for the icons."""
        button_style = f"{icon_style}.TButton" if icon_style else "TButton"
        label_style = f"{icon_style}.TLabel" if icon_style else "TLabel"
        self.icons = icons
        self.on_status = on_status
        self.playback = Playback()
        self.dragging = False
        self.timer: str | None = None

        self.frame = ttk.Frame(parent)
        self.play_button = ttk.Button(self.frame, text=icons["play"], style=button_style, command=self.toggle)
        # No tk variables: each scale holds its own value, so nothing here needs tk to tidy up on collection.
        self.scale = ttk.Scale(self.frame, from_=0, to=1, command=self._moved)
        self.time_label = ttk.Label(self.frame, width=11, anchor="center")
        self.volume_label = ttk.Label(self.frame, text=icons["volume"], style=label_style)
        self.volume = ttk.Scale(self.frame, from_=0, to=100, length=70, command=self._volume_moved)
        self.volume.set(100)
        self.discard_button = ttk.Button(self.frame, text=icons["discard"], style=button_style, command=on_discard)
        self.play_button.grid(row=0, column=0)
        self.scale.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.time_label.grid(row=0, column=2)
        self.volume_label.grid(row=0, column=3, padx=(8, 2))
        self.volume.grid(row=0, column=4)
        self.discard_button.grid(row=0, column=5, padx=(8, 0))
        self.frame.columnconfigure(1, weight=1, minsize=80)
        # A click lands the knob where it was clicked and holds it; the seek waits for the release.
        click_to_jump(self.scale, self._grab, self._release)
        click_to_jump(self.volume, lambda: None, self._volume_released)
        self.clear()

    @property
    def stem(self) -> str:
        """The memo the loaded audio belongs to, for log lines."""
        return self.playback.path.parent.name if self.playback.path else "-"

    @property
    def volume_percent(self) -> int:
        return round(self.volume.get())

    def set_volume(self, percent: int) -> None:
        self.volume.set(max(0, min(100, percent)))
        self.playback.volume = self.volume.get() / 100

    def load(self, path: Path, discardable: bool) -> None:
        """Show ``path`` ready to play from the start. ``discardable`` shows the discard button."""
        try:
            self.playback.load(path)
        except (OSError, wave.Error, ValueError) as e:
            log.warning("can't load %s for playback: %s", path, e)
            self.clear()
            self.on_status("Can't play this audio, see log", "error")
            return
        log.info("%s: player loaded, %s long", self.stem, format_time(self.playback.duration))
        self.scale.configure(to=max(self.playback.duration, 0.01))
        self.scale.set(0)
        self._show_time(0)
        self.set_enabled(True)
        if discardable:
            self.discard_button.grid()
        else:
            self.discard_button.grid_remove()

    def clear(self) -> None:
        self._cancel_tick()
        self.playback.unload()
        self.play_button.configure(text=self.icons["play"])
        self.scale.configure(to=1)
        self.scale.set(0)
        self.time_label.configure(text="–:–– / –:––")
        self.discard_button.grid_remove()
        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        """Play and scrub; off while recording and while nothing is loaded."""
        state = ["!disabled"] if enabled and self.playback.total else ["disabled"]
        self.play_button.state(state)
        self.scale.state(state)

    def set_busy(self, busy: bool) -> None:
        """No discarding a memo the worker is still writing into."""
        self.discard_button.state(["disabled"] if busy else ["!disabled"])

    def toggle(self) -> None:
        if self.playback.playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        try:
            self.playback.play()
        except Exception as e:  # sounddevice raises its own classes, plus OSError without a device
            log.warning("can't play audio: %s: %s", type(e).__name__, e)
            self.on_status("Can't play audio, see log", "error")
            return
        log.info("%s: play from %s", self.stem, format_time(self.playback.seconds))
        if self.playback.note:
            self.on_status(self.playback.note, "warning")
        self.play_button.configure(text=self.icons["pause"])
        self._schedule_tick()

    def set_output_device(self, name: str) -> None:
        """Play through ``name`` from now on; if playing, carry on there from the same spot."""
        playing = self.playback.playing
        if playing:
            self.playback.pause()
        self.playback.output_device = name
        if playing:
            self.play()

    def pause(self) -> None:
        """Pause if playing; harmless otherwise, so recording can call it unconditionally."""
        self._cancel_tick()
        if self.playback.playing:
            self.playback.pause()
            log.info("%s: pause at %s", self.stem, format_time(self.playback.seconds))
        self.play_button.configure(text=self.icons["play"])
        if self.playback.total:
            self._show_position()

    def close(self) -> None:
        self._cancel_tick()
        self.playback.unload()

    # --- internals ---------------------------------------------------------

    def _tick(self) -> None:
        self.timer = None
        if self.playback.finished:
            self.playback.pause()
            log.info("%s: played to the end", self.stem)
            self.play_button.configure(text=self.icons["play"])
            self._show_position()
            return
        self._show_position()
        self._schedule_tick()

    def _schedule_tick(self) -> None:
        self._cancel_tick()
        self.timer = self.frame.after(TICK_MS, self._tick)

    def _cancel_tick(self) -> None:
        if self.timer is not None:
            self.frame.after_cancel(self.timer)
            self.timer = None

    def _show_position(self) -> None:
        if not self.dragging:
            self.scale.set(self.playback.seconds)
            self._show_time(self.playback.seconds)

    def _show_time(self, seconds: float) -> None:
        self.time_label.configure(text=f"{format_time(seconds)} / {format_time(self.playback.duration)}")

    def _grab(self) -> None:
        self.dragging = True  # the play position stops moving the knob until it's let go

    def _release(self, seconds: float) -> None:
        self.dragging = False
        if not self.playback.total:
            return
        self.playback.seek(seconds)
        self._show_time(self.playback.seconds)
        log.info("%s: seek to %s", self.stem, format_time(self.playback.seconds))

    def _moved(self, value: str) -> None:
        self._show_time(float(value))

    def _volume_moved(self, value: str) -> None:
        self.playback.volume = float(value) / 100  # live, so dragging is heard as it moves

    def _volume_released(self, _percent: float) -> None:
        log.info("volume %d%%", self.volume_percent)
