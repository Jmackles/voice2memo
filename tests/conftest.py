"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
VAULT_DIR = TESTS_DIR / "vault"


@pytest.fixture
def vault() -> Path:
    """A throwaway Obsidian vault, wiped and recreated for every test that writes a note."""
    shutil.rmtree(VAULT_DIR, ignore_errors=True)
    (VAULT_DIR / ".obsidian").mkdir(parents=True)
    (VAULT_DIR / "attachments").mkdir()
    return VAULT_DIR


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
