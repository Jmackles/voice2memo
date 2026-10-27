from voice2memos import __version__
from voice2memos.errors import (
    CaptureError,
    CleanupError,
    EntityError,
    SaveError,
    StageResult,
    TranscribeError,
    VoiceMemoError,
)


def test_version_is_a_string():
    assert isinstance(__version__, str) and __version__


def test_stage_errors_share_a_base():
    for cls in (CaptureError, TranscribeError, CleanupError, EntityError, SaveError):
        err = cls("short", detail="long")
        assert isinstance(err, VoiceMemoError)
        assert err.message == "short"
        assert err.detail == "long"
        assert str(err) == "short"


def test_stage_result_defaults():
    r = StageResult(stage="cleanup", ok=False, message="Ollama not running")
    assert r.payload is None


def test_vault_fixture_is_an_obsidian_vault(vault):
    assert (vault / ".obsidian").is_dir()
