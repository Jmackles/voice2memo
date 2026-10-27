"""Config: ``.cfg`` load, defaults dict, apply.

Sections mirror ``voice2memos.example.cfg``: capture, whisper, cleanup,
entities, tags, vault, keys, logging, window. The defaults dict is the single
source of truth for shipped values; the installer writes the initial .cfg from it.

Notes from the design doc:
- construct ``configparser.ConfigParser(interpolation=None)``, or any ``%`` in
  the rewrite prompt raises; it also lowercases keys, so never rely on key case
- missing file gives defaults, partial file gets missing keys filled, the
  multi-line prompt must round-trip

Comments are whole lines starting with # or ; at the left edge. Lines of a
multi-line value are indented, so a prompt line starting with # stays part of
the prompt. configparser strips each line's own indentation, so that part of a
prompt doesn't survive.

Nothing here puts a value from the file into a message: errors name the line
or the key, so a secret on a broken line can't reach the status line or the log.
"""

from __future__ import annotations

import configparser
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".voice2memos"
CONFIG_PATH = CONFIG_DIR / "voice2memos.cfg"

Field = tuple[str, str]  # (section, key)

DEFAULTS: dict[str, dict[str, str]] = {
    "capture": {
        "hotkey": "<f9>",
        "note_hotkey": "<f10>",
        "input_device": "",
        "save_dir": "memos",
        "trash_dir": "memos/trash",
    },
    "whisper": {
        "model_size": "small",
        "device": "auto",
        "compute_type": "auto",
        "language": "en",
        "glossary": "",
    },
    "cleanup": {
        "provider": "ollama",
        "model": "gemma4:e2b",
        "temperature": "0",
        "timeout": "60",
        "keep_alive": "30m",
        "prompt": "",
    },
    "entities": {
        "provider": "ollama",
        "model": "gemma4:e2b",
        "temperature": "0",
        "timeout": "30",
    },
    "tags": {"pool": "idea\nquestion\nrant\nhomework\nbookinnit\ntodo"},
    "vault": {"path": "", "attachments_subdir": "attachments"},
    "keys": {"openai": "", "anthropic": ""},
    "logging": {"level": "INFO"},
    "window": {"geometry": "960x600", "sash": ""},
}

# Masked in Settings, never watched by the key log, and never logged by value.
SECRET_FIELDS: set[Field] = {("keys", "openai"), ("keys", "anthropic")}

PROVIDERS = ("ollama", "openai", "anthropic", "fake")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

HEADER = """\
# voice2memos settings. Edit here or in Settings; a key left out uses its default.
# Every key is explained in voice2memos.example.cfg in the app folder. The app
# rewrites this file when Settings are saved, so comments other than these go.
"""


def _number(low: float, high: float | None = None) -> Callable[[str], bool]:
    def check(value: str) -> bool:
        try:
            number = float(value)
        except ValueError:
            return False
        return number >= low and (high is None or number <= high)

    return check


# Values the app can't run with. Each falls back to its default with a warning.
CHECKS: dict[Field, tuple[Callable[[str], bool], str]] = {
    ("whisper", "device"): (lambda v: v in ("auto", "cpu", "cuda"), "auto, cpu, or cuda"),
    ("cleanup", "provider"): (lambda v: v in PROVIDERS, ", ".join(PROVIDERS)),
    ("entities", "provider"): (lambda v: v in PROVIDERS, ", ".join(PROVIDERS)),
    ("cleanup", "temperature"): (_number(0, 1), "a number from 0 to 1"),
    ("entities", "temperature"): (_number(0, 1), "a number from 0 to 1"),
    ("cleanup", "timeout"): (_number(1), "a number of seconds"),
    ("entities", "timeout"): (_number(1), "a number of seconds"),
    ("logging", "level"): (lambda v: v.upper() in LOG_LEVELS, ", ".join(LOG_LEVELS)),
    ("window", "geometry"): (lambda v: re.fullmatch(r"\d+x\d+", v) is not None, "width x height, like 960x600"),
    ("window", "sash"): (lambda v: v == "" or v.isdigit(), "blank or a number of pixels"),
}


def new_parser() -> configparser.ConfigParser:
    """A parser holding the defaults. Comments are blanked before parsing, so none are configured."""
    cfg = configparser.ConfigParser(interpolation=None, comment_prefixes=(), inline_comment_prefixes=None)
    cfg.read_dict(DEFAULTS)
    return cfg


def _read_over(cfg: configparser.ConfigParser, path: Path) -> None:
    """Read the file over ``cfg``. Raises OSError, UnicodeDecodeError, or configparser.Error."""
    # utf-8-sig: Notepad may save a byte-order mark, which would hide the first section.
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    # Blank whole-line comments rather than dropping them, so error line numbers stay true.
    cfg.read_string("\n".join("" if line.startswith(("#", ";")) else line for line in lines), source=str(path))
    for section in cfg.sections():
        for key, value in cfg[section].items():
            if value.startswith("\n"):  # "pool =" with the list on the lines below it
                cfg[section][key] = value.lstrip("\n")


def describe_error(error: Exception) -> str:
    """A status line for a file that won't parse. Never quotes the file."""
    line = getattr(error, "lineno", None)
    if line is None and isinstance(error, configparser.ParsingError) and error.errors:
        line = error.errors[0][0]
    return f"Config error on line {line}" if line else "Config file unreadable"


def check(cfg: configparser.ConfigParser) -> list[str]:
    """Put defaults back for values the app can't use; one warning per value, naming only the key."""
    warnings = []
    for (section, key), (ok, wanted) in CHECKS.items():
        if not ok(cfg[section][key].strip()):
            default = DEFAULTS[section][key]
            cfg[section][key] = default
            warnings.append(f"Config {section}.{key} should be {wanted}; using {default or 'blank'}")
    for warning in warnings:
        log.warning("%s", warning)
    return warnings


def load(path: Path) -> tuple[configparser.ConfigParser, list[str]]:
    """The config at ``path`` over the defaults, plus warnings for the status line.

    A missing file is written from the defaults. One that won't parse is renamed
    to .cfg.bad and replaced with defaults. A value the app can't use falls back
    to its default in memory; the file itself is left alone.
    """
    cfg = new_parser()
    if not path.exists():
        try:
            save(cfg, path)
            log.info("wrote a default config to %s", path)
        except OSError as e:
            log.warning("couldn't write a default config to %s: %s", path, e.strerror)
        return cfg, []
    try:
        _read_over(cfg, path)
    except (OSError, UnicodeDecodeError, configparser.Error) as e:
        message = f"{describe_error(e)}, using defaults"
        cfg = new_parser()
        bad = path.with_name(path.name + ".bad")
        try:
            os.replace(path, bad)
            save(cfg, path)
            log.warning("%s; the old file is now %s", message, bad.name)
        except OSError as move_error:
            log.warning("%s; couldn't set the file aside: %s", message, move_error.strerror)
        return cfg, [message]
    return cfg, check(cfg)


def save(cfg: configparser.ConfigParser, path: Path) -> None:
    """Write the whole config through a temp file, so a crash never leaves half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        cfg.write(f)
    os.replace(temp, path)


def update_file(path: Path, changes: dict[Field, str]) -> tuple[configparser.ConfigParser, list[str]]:
    """Write ``changes`` into the file, keeping everything else in it as it is now.

    The file is re-read first, so edits made by hand while the app runs survive.
    Raises OSError, UnicodeDecodeError, or configparser.Error if it can't be
    read; nothing is written then.
    """
    cfg = new_parser()
    if path.exists():
        _read_over(cfg, path)
    for (section, key), value in changes.items():
        cfg[section][key] = value
    save(cfg, path)
    return cfg, check(cfg)


def resolve_folder(value: str, config_path: Path) -> Path:
    """A folder named in the config. A relative one is relative to the folder the .cfg is in."""
    return config_path.parent / Path(value).expanduser()


def values(cfg: configparser.ConfigParser) -> dict[Field, str]:
    return {(section, key): value for section in cfg.sections() for key, value in cfg[section].items()}
