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

log = logging.getLogger(__name__)

TICK_MS = 50


def open_output_stream(rate: int, channels: int, callback: Callable) -> object:
    """A sounddevice stream of 16-bit frames fed by ``callback``. Tests swap this out."""
    import sounddevice as sd

    return sd.RawOutputStream(samplerate=rate, channels=channels, dtype="int16", callback=callback)


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


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
        stream = open_output_stream(self.rate, self.channels, self._callback)
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
        on_error: Callable[[str], None],
        icon_style: str = "",
    ) -> None:
        """``icon_style`` names a ttk style prefix with .TButton and .TLabel variants for the icons."""
        button_style = f"{icon_style}.TButton" if icon_style else "TButton"
        label_style = f"{icon_style}.TLabel" if icon_style else "TLabel"
        self.icons = icons
        self.on_error = on_error
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
        # Seek when the slider is let go, not on every pixel of a drag.
        self.scale.bind("<ButtonPress-1>", self._grab, add="+")
        self.scale.bind("<ButtonRelease-1>", self._release, add="+")
        self.volume.bind("<ButtonRelease-1>", self._volume_released, add="+")
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
            self.on_error("Can't play this audio, see log")
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
            self.on_error("Can't play audio, see log")
            return
        log.info("%s: play from %s", self.stem, format_time(self.playback.seconds))
        self.play_button.configure(text=self.icons["pause"])
        self._schedule_tick()

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

    def _grab(self, _event: tk.Event | None = None) -> None:
        self.dragging = True

    def _release(self, _event: tk.Event | None = None) -> None:
        self.dragging = False
        if not self.playback.total:
            return
        self.playback.seek(self.scale.get())
        self._show_time(self.playback.seconds)
        log.info("%s: seek to %s", self.stem, format_time(self.playback.seconds))

    def _moved(self, value: str) -> None:
        self._show_time(float(value))

    def _volume_moved(self, value: str) -> None:
        self.playback.volume = float(value) / 100  # live, so dragging is heard as it moves

    def _volume_released(self, _event: tk.Event | None = None) -> None:
        log.info("volume %d%%", self.volume_percent)
