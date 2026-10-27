"""Exception classes, one per stage, and the StageResult the worker posts back.

The worker never lets an exception escape. Each stage runs in its own try
block; on failure it logs, posts a StageResult through ``root.after``, and
continues to the next stage with whatever it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class VoiceMemoError(Exception):
    """Base for every stage error.

    ``message`` is the short text shown in the status line; ``detail`` is the
    longer string that goes to the log.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class CaptureError(VoiceMemoError):
    """Hotkey, microphone, or WAV writing failed."""


class TranscribeError(VoiceMemoError):
    """Whisper failed to load or produced nothing."""


class CleanupError(VoiceMemoError):
    """The rewrite call failed, timed out, or returned rejected output."""


class EntityError(VoiceMemoError):
    """Entity or tag extraction returned bad JSON or nothing."""


class SaveError(VoiceMemoError):
    """The note could not be written into the vault."""


@dataclass
class StageResult:
    """What the worker posts to the UI thread after each stage."""

    stage: str
    ok: bool
    message: str = ""
    payload: Any = None
