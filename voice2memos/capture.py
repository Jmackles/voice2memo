"""Capture: global hotkey, mic stream, WAV writer, socket trigger.

Rule above all: the WAV is on disk before anything downstream runs.

The sounddevice callback runs on PortAudio's thread. It may append to the
buffer and feed the level meter's RMS; it never touches a tk widget. The
pynput listener has its own thread too; its callbacks go through ``App.post``.
Devices are chosen by name in Settings → Audio; see the devices section.

``input_device = fake`` gives a recorder that answers with the fixture's
memo.wav, or a second of silence, so the record button works without a mic.
The socket trigger arrives with push-to-note in step 14.
"""

from __future__ import annotations

import logging
import math
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

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


# --- devices --------------------------------------------------------------
#
# Every device is listed once per host API (MME, DirectSound, WASAPI, WDM-KS on
# Windows), so a name alone is ambiguous to sounddevice. The app lists and
# opens devices on the default host API only: MME on Windows, which shows what
# Windows shows as connected and resamples, so a 16 kHz WAV plays anywhere.
# MME cuts names to 31 characters; the full name comes from another host API.

MME_NAME_LIMIT = 31
SOUND_MAPPER = "Microsoft Sound Mapper"  # MME's stand-in for the default, not a device of its own
KINDS = ("input", "output")


@dataclass
class DeviceTable:
    """What PortAudio sees: every device under every host API, and the defaults."""

    devices: list[dict]
    hostapi: int  # the default host API
    default: dict[str, int | None]  # kind -> device index, None when there is none


def device_table() -> DeviceTable:
    """Ask PortAudio. Tests swap this out."""
    import sounddevice as sd

    inp, out = sd.default.device
    devices = [dict(d) for d in sd.query_devices()]
    return DeviceTable(devices, sd.default.hostapi, {"input": inp if inp >= 0 else None, "output": out if out >= 0 else None})


def rescan_devices() -> None:
    """Make PortAudio look again, for a device plugged in since launch. Only with no stream open."""
    import sounddevice as sd

    sd._terminate()
    sd._initialize()


def _has(device: dict, kind: str) -> bool:
    return device[f"max_{kind}_channels"] > 0


def _full_name(table: DeviceTable, device: dict, kind: str) -> str:
    name = device["name"]
    if len(name) < MME_NAME_LIMIT:
        return name
    longer = {d["name"] for d in table.devices if _has(d, kind) and d["name"].startswith(name) and d["name"] != name}
    return longer.pop() if len(longer) == 1 else name


def _listed(table: DeviceTable, kind: str) -> list[tuple[int, str]]:
    """(index, full name) of each device of ``kind`` on the default host API."""
    return [
        (index, _full_name(table, d, kind))
        for index, d in enumerate(table.devices)
        if d["hostapi"] == table.hostapi and _has(d, kind) and not d["name"].startswith(SOUND_MAPPER)
    ]


def list_devices(kind: str, table: DeviceTable | None = None) -> list[str]:
    """The names to offer in Settings, each once."""
    return [name for _, name in _listed(table or device_table(), kind)]


def default_device_name(kind: str, table: DeviceTable | None = None) -> str:
    """The system default's name; blank when there is none."""
    table = table or device_table()
    index = table.default[kind]
    return "" if index is None else _full_name(table, table.devices[index], kind)


def resolve_device(name: str, kind: str, table: DeviceTable | None = None) -> tuple[int | None, str]:
    """A saved device name as a PortAudio index; None means the system default.

    A device that isn't connected falls back to the default rather than
    failing, with a note for the status line saying so.
    """
    if not name:
        return None, ""
    for index, full in _listed(table or device_table(), kind):
        if full == name:
            return index, ""
    log.warning("%s device %r isn't connected, using the default", kind, name)
    return None, f"{name} isn't connected, using the default"


def open_input_stream(device: int | None, rate: int, callback: Callable) -> object:
    """A sounddevice stream of mono 16-bit frames into ``callback``. Tests swap this out."""
    import sounddevice as sd

    return sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", device=device, callback=callback)


class LevelMeter:
    """A microphone opened only to show how loud it hears, for the test in Settings.

    The callback runs on PortAudio's thread and only stores a number; the
    dialog reads ``level`` from its own ``after`` loop.
    """

    FLOOR_DB = -60  # quieter than this shows as nothing

    def __init__(self, device: str) -> None:
        self.device = device
        self.level = 0.0  # 0 to 1, on a decibel scale
        self.stream = None

    def start(self) -> str:
        """Open the microphone; returns a note if it fell back to the default. Raises CaptureError."""
        try:
            table = device_table()
            index, note = resolve_device(self.device, "input", table)
            chosen = index if index is not None else table.default["input"]
            if chosen is None:
                raise CaptureError("No microphone found", "the system has no input device")
            self.stream = open_input_stream(index, int(table.devices[chosen]["default_samplerate"]), self._callback)
            self.stream.start()
        except CaptureError:
            raise
        except Exception as e:
            self.stream = None
            raise CaptureError("Microphone unavailable", f"{type(e).__name__}: {e}") from e
        return note

    def stop(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        self.level = 0.0

    def _callback(self, indata, frames, time_info, status) -> None:
        samples = np.frombuffer(bytes(indata), "<i2").astype(np.float32) / 32768
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        db = 20 * math.log10(rms) if rms > 0 else self.FLOOR_DB
        self.level = min(1.0, max(0.0, 1 - db / self.FLOOR_DB))


# --- recorders ------------------------------------------------------------


class Recorder:
    """``start`` opens the input, ``stop`` returns (frames, samplerate). One at a time.

    ``note`` is set by ``start`` when it had to settle for something other
    than what was asked, for the status line.
    """

    name = "recorder"

    def __init__(self) -> None:
        self.recording = False
        self.note = ""

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> tuple[bytes, int]:
        raise NotImplementedError


class MicRecorder(Recorder):
    """sounddevice InputStream, mono int16 at the device's own rate; chunks pile up while held."""

    name = "microphone"

    def __init__(self, device: str = "") -> None:
        super().__init__()
        self.device = device  # a name from list_devices, blank for the system default
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
            table = device_table()
            index, self.note = resolve_device(self.device, "input", table)
            chosen = index if index is not None else table.default["input"]
            if chosen is None:
                raise CaptureError("No microphone found", "the system has no input device")
            self.samplerate = int(table.devices[chosen]["default_samplerate"])
            self.stream = open_input_stream(index, self.samplerate, self._callback)
            self.stream.start()
        except CaptureError:
            raise
        except Exception as e:  # ImportError without sounddevice, its own classes for a device that won't open
            self.stream = None
            raise CaptureError("Microphone unavailable", f"{type(e).__name__}: {e}") from e
        with self.lock:
            self.chunks = []
        self.recording = True
        log.info("recording from %s at %d Hz", table.devices[chosen]["name"], self.samplerate)

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
