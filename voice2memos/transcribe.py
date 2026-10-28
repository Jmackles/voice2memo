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
fixture folder so the whole app runs without a model. The real wrapper is the
grey box step 7 fills in.
"""

from __future__ import annotations

import json
import logging
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
    """The segments as one string for a pane, raw or cleaned."""
    return " ".join((s.shown if cleaned else s.text).strip() for s in segments).strip()


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
    """WAV in, segments out. Subclasses implement ``_transcribe``."""

    name = "transcriber"

    def transcribe(self, wav: Path, stem: str = "") -> list[Segment]:
        stem = stem or wav.parent.name
        with timed(log, f"transcribe ({self.name})", stem):
            segments = self._transcribe(wav)
        log.info("%s: %d segments", stem, len(segments))
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


class WhisperTranscriber(Transcriber):
    """The real one. Step 7 fills this in; until then it says so."""

    name = "whisper"

    def __init__(self, model_size: str, device: str, compute_type: str, language: str, glossary: str) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.glossary = glossary

    def _transcribe(self, wav: Path) -> list[Segment]:
        raise TranscribeError("Whisper not built yet", "WhisperTranscriber arrives in build step 7")


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
