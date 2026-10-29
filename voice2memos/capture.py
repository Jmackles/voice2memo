"""Capture: global hotkey, mic stream, WAV writer, socket trigger.

Rule above all: the WAV is on disk before anything downstream runs.

The sounddevice callback runs on PortAudio's thread. It may append to the
buffer and feed the level meter's RMS; it never touches a tk widget. The
pynput listener has its own thread too; its callbacks post through ``root.after``.

``input_device = fake`` gives a recorder that answers with the fixture's
memo.wav, or a second of silence, so the record button works without a mic.
The socket trigger arrives with push-to-note in step 14.
"""

from __future__ import annotations

import logging
import threading
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from voice2memos.errors import CaptureError

log = logging.getLogger(__name__)

WAV_NAME = "memo.wav"
SILENCE_RATE = 16000


def new_stem(save_dir: Path, now: datetime | None = None) -> str:
    """YYYYMMDD-HHMMSS in local time, with a two-digit counter only on collision."""
    stem = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = stem
    counter = 1
    while (save_dir / candidate).exists():
        candidate = f"{stem}-{counter:02d}"
        counter += 1
    return candidate


def write_wav(path: Path, frames: bytes, samplerate: int, channels: int = 1, sampwidth: int = 2) -> None:
    """16-bit PCM through the stdlib wave module; the header carries the true rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sampwidth)
        out.setframerate(samplerate)
        out.writeframes(frames)


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as src:
        return src.readframes(src.getnframes()), src.getframerate()


def is_pcm16_wav(path: Path) -> bool:
    """A WAV the app can use as is: anything else gets decoded."""
    try:
        with wave.open(str(path), "rb") as src:
            return src.getsampwidth() == 2
    except (OSError, EOFError, wave.Error):
        return False


def decode_to_wav(source: Path, target: Path) -> None:
    """Any audio PyAV reads (mp3, m4a, odd WAVs) as a mono 16-bit WAV at its own rate.

    PyAV directly rather than faster-whisper's decode_audio: that one breaks on
    PyAV 19 and imports all of faster-whisper first. Whisper resamples on load,
    so keeping the source rate only makes playback sound better.
    Raises CaptureError; the source is only read.
    """
    try:
        import av
    except ImportError as e:
        raise CaptureError("Can't read that audio file", f"PyAV not installed: {e}") from e
    try:
        with av.open(str(source)) as container:
            stream = container.streams.audio[0]
            rate = stream.rate
            resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
            chunks = []
            for frame in [*container.decode(stream), None]:  # None flushes the resampler
                for out in resampler.resample(frame):
                    chunks.append(bytes(out.planes[0])[: out.samples * 2])
    except (IndexError, av.FFmpegError, OSError, ValueError) as e:
        raise CaptureError("Can't read that audio file", f"{source}: {type(e).__name__}: {e}") from e
    write_wav(target, b"".join(chunks), rate)


# --- recorders ------------------------------------------------------------


class Recorder:
    """``start`` opens the input, ``stop`` returns (frames, samplerate). One at a time."""

    name = "recorder"

    def __init__(self) -> None:
        self.recording = False

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> tuple[bytes, int]:
        raise NotImplementedError


class MicRecorder(Recorder):
    """sounddevice InputStream, mono int16 at the device's own rate; chunks pile up while held."""

    name = "microphone"

    def __init__(self, device: str = "") -> None:
        super().__init__()
        self.device = device or None
        self.chunks: list[bytes] = []
        self.samplerate = 0
        self.stream = None
        self.lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:
        # PortAudio's thread: copy the chunk and nothing else.
        if status:
            log.debug("input stream status: %s", status)
        if self.recording:
            with self.lock:
                self.chunks.append(bytes(indata))

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise CaptureError("Microphone unavailable", f"sounddevice not installed: {e}") from e
        try:
            self.samplerate = int(sd.query_devices(self.device, "input")["default_samplerate"])
            self.stream = sd.RawInputStream(
                samplerate=self.samplerate, channels=1, dtype="int16", device=self.device, callback=self._callback
            )
            self.stream.start()
        except Exception as e:  # sounddevice raises its own classes plus ValueError for a bad name
            self.stream = None
            raise CaptureError("Microphone unavailable", f"{type(e).__name__}: {e}") from e
        with self.lock:
            self.chunks = []
        self.recording = True
        log.info("recording from %s at %d Hz", self.device or "the default input", self.samplerate)

    def stop(self) -> tuple[bytes, int]:
        self.recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self.lock:
            frames = b"".join(self.chunks)
            self.chunks = []
        log.info("recorded %.1fs", len(frames) / 2 / max(self.samplerate, 1))
        return frames, self.samplerate


class FakeRecorder(Recorder):
    """Answers with the fixture's memo.wav, or one second of silence without one."""

    name = "fake recorder"

    def __init__(self, fixture: Path | None = None, fail: bool = False) -> None:
        super().__init__()
        self.fixture = fixture
        self.fail = fail

    def start(self) -> None:
        if self.fail:
            raise CaptureError("Microphone unavailable", "fake recorder set to fail")
        self.recording = True

    def stop(self) -> tuple[bytes, int]:
        self.recording = False
        if self.fixture is not None and (self.fixture / WAV_NAME).exists():
            return read_wav(self.fixture / WAV_NAME)
        return bytes(SILENCE_RATE * 2), SILENCE_RATE


def make_recorder(cfg, repo_dir: Path) -> Recorder:
    device = cfg["capture"]["input_device"].strip()
    if device.lower() == "fake":
        fixture = cfg["fake"]["fixture"].strip()
        return FakeRecorder(repo_dir / fixture if fixture else None, fail="capture" in cfg["fake"]["fail"].split())
    return MicRecorder(device)


# --- hotkeys ----------------------------------------------------------------


class HotkeyListener:
    """Two pynput chords on one Listener: push-to-talk (press and release) and push-to-note.

    ``bind`` stops the old listener and starts a new one, which is how Settings
    apply without a restart. Callbacks run on pynput's thread. A chord that
    won't parse is reported through ``on_error`` and skipped, the other stays.
    """

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        on_note: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.on_note = on_note
        self.on_error = on_error
        self.listener = None
        self.bound: tuple[str, str] = ("", "")

    def bind(self, ptt: str, note: str) -> None:
        self.stop()
        self.bound = (ptt, note)
        try:
            from pynput import keyboard
        except ImportError as e:
            log.warning("hotkeys off: pynput not installed (%s)", e)
            self.on_error("Hotkeys off: pynput not installed")
            return
        chords: dict[str, keyboard.HotKey] = {}
        for spec, name, on_activate in ((ptt, "push-to-talk", self._pressed), (note, "push-to-note", self.on_note)):
            if not spec.strip():
                continue
            try:
                chords[name] = keyboard.HotKey(keyboard.HotKey.parse(spec), on_activate)
            except ValueError:
                log.warning("%s hotkey not usable: %s", name, spec)
                self.on_error(f"{name.capitalize()} hotkey not usable")
        if not chords:
            return
        self.held = False
        ptt_keys = chords["push-to-talk"]._keys if "push-to-talk" in chords else set()

        def on_press(key):
            key = listener.canonical(key)
            for chord in chords.values():
                chord.press(key)

        def on_release(key):
            key = listener.canonical(key)
            for chord in chords.values():
                chord.release(key)
            if self.held and key in ptt_keys:  # releasing any key of the chord ends the recording
                self.held = False
                self.on_release()

        # Closed over, not read from self: a key arriving after a rebind still has its own listener.
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        self.listener = listener
        log.info("hotkeys bound: %s", ", ".join(f"{name} {spec}" for name, spec in (("push-to-talk", ptt), ("push-to-note", note)) if name in chords))

    def _pressed(self) -> None:
        if not self.held:
            self.held = True
            self.on_press()

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
