"""Transcribe: faster-whisper wrapper, Segment dataclass, JSON in/out.

Segment contract (one per Whisper segment, serialized beside the WAV):
    index    int    position in the memo
    start    float  seconds
    end      float  seconds
    text     str    raw transcript for this span
    cleaned  str    rewritten text, empty until cleanup runs
    words    list   Whisper's word timestamps, kept for the clipper

Every transcriber has one method, ``transcribe(wav) -> list[Segment]``, and
raises TranscribeError for anything it can't do. ``make_transcriber`` picks one
from the config: ``model_size = fake`` gives FakeWhisper, which answers from a
fixture folder so the whole app runs without a model; anything else is a
faster-whisper model name for WhisperTranscriber.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import sysconfig
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from voice2memos.errors import TranscribeError
from voice2memos.logsetup import timed

log = logging.getLogger(__name__)

SEGMENTS_FILE = "segments.json"


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    cleaned: str = ""
    words: list[dict] = field(default_factory=list)

    @property
    def shown(self) -> str:
        """What the note pane shows for this segment: cleaned if it exists, else raw."""
        return self.cleaned or self.text


def joined_text(segments: list[Segment], cleaned: bool = False) -> str:
    """The segments as one string for a prompt, raw or cleaned."""
    return " ".join((s.shown if cleaned else s.text).strip() for s in segments).strip()


def display_text(segments: list[Segment]) -> str:
    """The raw transcript as the Transcript pane shows it: one paragraph per segment."""
    return "\n\n".join(s.text.strip() for s in segments)


def apply_edit(segments: list[Segment], text: str) -> list[Segment]:
    """The Transcript pane's text back into segments, one per paragraph.

    With as many paragraphs as segments, each keeps its timing, and its word
    timestamps too if its text didn't change. Otherwise the timing is no
    longer known, so each paragraph spans the whole memo. Cleaned text always
    goes: it belonged to the old words.
    """
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) == len(segments):
        return [
            Segment(s.index, s.start, s.end, p, "", s.words if p == " ".join(s.text.split()) else [])
            for s, p in zip(segments, paragraphs)
        ]
    start = segments[0].start if segments else 0.0
    end = segments[-1].end if segments else 0.0
    return [Segment(i, start, end, p) for i, p in enumerate(paragraphs)]


def save_segments(segments: list[Segment], path: Path) -> None:
    path.write_text(json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_segments(path: Path) -> list[Segment]:
    """Raises TranscribeError if the file is missing or not a segment list."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [Segment(**item) for item in raw]
    except OSError as e:
        raise TranscribeError("Transcript file missing", f"{path}: {e.strerror}") from e
    except (ValueError, TypeError) as e:
        raise TranscribeError("Transcript file unreadable", f"{path}: {e}") from e


class Transcriber:
    """WAV in, segments out. Subclasses implement ``_transcribe``.

    ``note`` is set when the transcriber settled for less than asked (CPU
    instead of CUDA), for the status line.
    """

    name = "transcriber"
    note = ""
    stem = ""

    def transcribe(self, wav: Path, stem: str = "") -> list[Segment]:
        self.stem = stem or wav.parent.name
        with timed(log, f"transcribe ({self.name})", self.stem):
            segments = self._transcribe(wav)
        log.info("%s: %d segments", self.stem, len(segments))
        return segments

    def _transcribe(self, wav: Path) -> list[Segment]:
        raise NotImplementedError


class FakeWhisper(Transcriber):
    """Answers with the ``segments.json`` from a fixture folder.

    ``fixture`` blank means the folder the WAV is in, which is what bench.py
    wants. ``fail`` raises instead, and ``delay`` sleeps first, for the error
    states.
    """

    name = "fake whisper"

    def __init__(self, fixture: Path | None = None, fail: bool = False, delay: float = 0) -> None:
        self.fixture = fixture
        self.fail = fail
        self.delay = delay

    def _transcribe(self, wav: Path) -> list[Segment]:
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise TranscribeError("Transcription failed", "fake transcriber set to fail")
        if not wav.exists():
            raise TranscribeError("Audio file missing", str(wav))
        folder = self.fixture or wav.parent
        return load_segments(folder / SEGMENTS_FILE)


def prepare_cuda_dlls() -> None:
    """On Windows, put the pip-installed cuBLAS and cuDNN DLL folders where ctranslate2 can find them.

    Must run before faster-whisper is first imported. Harmless elsewhere, or
    when the CUDA wheels aren't installed.
    """
    if sys.platform != "win32":
        return
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    folders = [str(p) for p in nvidia.glob("*/bin") if p.is_dir()]
    for folder in folders:
        os.add_dll_directory(folder)
    if folders:
        os.environ["PATH"] = os.pathsep.join(folders + [os.environ.get("PATH", "")])


# Loaded models, by (size, device, compute type). A model takes seconds to load
# and hundreds of MB, so each is loaded once and shared by every memo after.
_MODELS: dict[tuple[str, str, str], object] = {}
_MODELS_LOCK = threading.Lock()


def load_whisper_model(model_size: str, device: str, compute_type: str):
    """``faster_whisper.WhisperModel``, downloading on first use. Tests swap this out.

    A model already in the Hugging Face cache is loaded from there without
    asking the hub for a newer revision: the app is offline unless the user
    points it at an API provider, and that check is a network request on
    every launch.
    """
    prepare_cuda_dlls()
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    try:
        source = download_model(model_size, local_files_only=True)
    except Exception:  # not cached yet: let WhisperModel download it
        source = model_size
        log.info("whisper %s isn't downloaded yet, fetching it", model_size)
    return WhisperModel(source, device=device, compute_type=compute_type)


def resolve_whisper(device: str, compute_type: str) -> list[tuple[str, str]]:
    """The (device, compute type) pairs to try, in order. ``auto`` means CUDA first, then CPU int8."""
    devices = ["cuda", "cpu"] if device == "auto" else [device]
    plan = []
    for dev in devices:
        if compute_type == "auto":
            plan.append((dev, "float16" if dev == "cuda" else "int8"))
        else:
            plan.append((dev, compute_type))
    return plan


class WhisperTranscriber(Transcriber):
    """faster-whisper: word timestamps, VAD, the language pinned, the glossary as the initial prompt.

    ``load`` picks the device: with ``device = auto`` it tries CUDA and falls
    back to CPU int8, setting ``note`` ("GPU unavailable, using CPU") for the
    status line. Loading is slow, so the app warms it up on a thread at startup.
    """

    name = "whisper"

    def __init__(self, model_size: str, device: str, compute_type: str, language: str, glossary: str) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.glossary = glossary
        self.note = ""
        self.model = None
        self.loaded_on: tuple[str, str] | None = None

    def load(self) -> None:
        """Have the model in memory, loading it if no memo has needed it yet. Raises TranscribeError."""
        if self.model is not None:
            return
        plan = resolve_whisper(self.device, self.compute_type)
        last: Exception | None = None
        for device, compute_type in plan:
            key = (self.model_size, device, compute_type)
            with _MODELS_LOCK:
                try:
                    if key not in _MODELS:
                        log.info("loading whisper %s on %s %s", *key)
                        started = time.perf_counter()
                        _MODELS[key] = load_whisper_model(*key)
                        log.info("whisper %s loaded on %s %s in %.1fs", *key, time.perf_counter() - started)
                    self.model = _MODELS[key]
                except Exception as e:  # ctranslate2 raises RuntimeError for missing CUDA, ValueError for bad types
                    last = e
                    log.warning("whisper %s won't load on %s %s: %s: %s", *key, type(e).__name__, e)
                    continue
            self.loaded_on = (device, compute_type)
            if device == "cpu" and self.device == "auto" and plan[0][0] == "cuda":
                self.note = "GPU unavailable, using CPU"
            return
        raise TranscribeError("Whisper model unavailable", f"{type(last).__name__}: {last}" if last else "no device to try")

    def _transcribe(self, wav: Path) -> list[Segment]:
        self.load()
        glossary = [line.strip() for line in self.glossary.splitlines() if line.strip()]
        try:
            raw, info = self.model.transcribe(
                str(wav),
                language=self.language or None,
                word_timestamps=True,
                vad_filter=True,
                initial_prompt=", ".join(glossary) or None,
            )
            segments = [
                Segment(
                    index,
                    round(s.start, 2),
                    round(s.end, 2),
                    s.text,
                    words=[{"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)} for w in (s.words or [])],
                )
                for index, s in enumerate(raw)  # the generator runs here; faster-whisper is lazy
            ]
        except Exception as e:  # PyAV decode errors, ctranslate2 runtime errors
            raise TranscribeError("Transcription failed", f"{type(e).__name__}: {e}") from e
        log.info("%s: %.1fs of audio on %s", self.stem, info.duration, " ".join(self.loaded_on or ()))
        if not segments:
            raise TranscribeError("Nothing heard in the recording", f"{info.duration:.1f}s of audio, no speech found")
        return segments


def whisper_models() -> tuple[list[str], str]:
    """The model sizes the installed faster-whisper knows, and a note saying which are downloaded.

    Asks the library rather than keeping a list, so a new release's models
    just show up. Importing faster-whisper takes seconds: call this off the
    UI thread.
    """
    try:
        prepare_cuda_dlls()
        import faster_whisper
        from faster_whisper.utils import download_model
    except Exception as e:  # ImportError, or a CUDA or PyAV install that breaks on import
        log.warning("faster-whisper won't import: %s: %s", type(e).__name__, e)
        return [], "faster-whisper isn't installed; run python install.py"
    names = list(faster_whisper.available_models())
    downloaded = []
    for name in names:
        try:
            download_model(name, local_files_only=True)  # only looks in the Hugging Face cache
            downloaded.append(name)
        except Exception:
            pass
    if downloaded:
        return names, f"Downloaded: {', '.join(downloaded)}. Any other downloads the first time it's used."
    return names, "None downloaded yet; the chosen one downloads the first time it's used."


def make_transcriber(cfg, repo_dir: Path) -> Transcriber:
    """The transcriber the config asks for. ``repo_dir`` is where a relative fake fixture lives."""
    whisper = cfg["whisper"]
    if whisper["model_size"].strip().lower() == "fake":
        fake = cfg["fake"]
        fixture = fake["fixture"].strip()
        return FakeWhisper(
            fixture=repo_dir / fixture if fixture else None,
            fail="transcribe" in fake["fail"].split(),
            delay=float(fake["delay"] or 0),
        )
    return WhisperTranscriber(
        whisper["model_size"], whisper["device"], whisper["compute_type"], whisper["language"], whisper["glossary"]
    )
