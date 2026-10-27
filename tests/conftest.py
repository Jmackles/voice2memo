"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


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
