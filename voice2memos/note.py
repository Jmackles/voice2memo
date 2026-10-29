"""Note: the memo's working folder, the serializer, vault paths, obsidian.json check.

Working folder, one per memo under the save directory, named by the stem. It
is the app's source of truth; the vault only ever receives a rendered copy::

    20261027-093400/
      memo.wav          as captured, never modified
      segments.json     raw transcript, rewritten when the Transcript is edited
      segments.original.json   Whisper's own, kept on the first edit, never touched again
      cleaned.json      the same segments with cleaned text
      entities.json     link terms
      tags.json         proposed and rejected tags
      note.md           draft rendered from the above, regenerated on save
      state.json        saved_to path if saved, created time, source file if opened

Note file layout, written here and nowhere else::

    ---
    date: 2026-10-27T09:34:00-04:00
    audio: 20261027-093400.wav
    tags: [idea, database]
    ---
    # 20261027-093400

    <cleaned text, one paragraph per segment ending in ^stem-N, links in [[ ]]>

    > [!quote]- Original
    > <raw transcript, one callout line per segment>

    ![[20261027-093400.wav]]

The note is written to a temp file beside the target and ``os.replace``d into
place, so a sync client never sees a half-written file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from voice2memos.capture import WAV_NAME
from voice2memos.cleanup import CLEANED_FILE, ENTITIES_FILE, TAGS_FILE
from voice2memos.errors import SaveError
from voice2memos.transcribe import SEGMENTS_FILE, Segment, display_text, load_segments, save_segments

log = logging.getLogger(__name__)

NOTE_FILE = "note.md"
STATE_FILE = "state.json"
ORIGINAL_SEGMENTS_FILE = "segments.original.json"
TRASH_DAYS = 30


@dataclass
class Memo:
    """One memo's working folder and what has been worked out about it so far."""

    folder: Path
    segments: list[Segment] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    rejected_tags: list[str] = field(default_factory=list)
    saved_to: Path | None = None
    created: datetime = field(default_factory=lambda: datetime.now().astimezone())
    source: Path | None = None  # the file it was opened from; None for a recording

    @property
    def stem(self) -> str:
        return self.folder.name

    @property
    def is_recording(self) -> bool:
        return self.source is None

    @property
    def wav(self) -> Path:
        return self.folder / WAV_NAME

    @property
    def has_audio(self) -> bool:
        return self.wav.exists()

    @property
    def raw_text(self) -> str:
        """What the Transcript pane shows: one paragraph per segment."""
        return display_text(self.segments)

    @property
    def cleaned_text(self) -> str:
        """What the Note pane shows: one paragraph per segment."""
        return "\n\n".join(s.shown.strip() for s in self.segments)

    @classmethod
    def create(cls, folder: Path, source: Path | None = None) -> Memo:
        folder.mkdir(parents=True, exist_ok=False)
        memo = cls(folder, source=source)
        memo.save_state()
        return memo

    @classmethod
    def open(cls, folder: Path) -> Memo:
        """Whatever the folder holds; missing files just mean that stage hasn't run."""
        memo = cls(folder)
        if (folder / STATE_FILE).exists():
            state = json.loads((folder / STATE_FILE).read_text(encoding="utf-8"))
            memo.saved_to = Path(state["saved_to"]) if state.get("saved_to") else None
            memo.source = Path(state["source"]) if state.get("source") else None
            if state.get("created"):
                memo.created = datetime.fromisoformat(state["created"])
        for name in (CLEANED_FILE, SEGMENTS_FILE):
            if (folder / name).exists():
                memo.segments = load_segments(folder / name)
                break
        if (folder / ENTITIES_FILE).exists():
            memo.entities = json.loads((folder / ENTITIES_FILE).read_text(encoding="utf-8"))
        if (folder / TAGS_FILE).exists():
            tags = json.loads((folder / TAGS_FILE).read_text(encoding="utf-8"))
            memo.tags, memo.rejected_tags = tags.get("proposed", []), tags.get("rejected", [])
        return memo

    def save_state(self) -> None:
        state = {
            "saved_to": str(self.saved_to) if self.saved_to else None,
            "created": self.created.isoformat(),
            "source": str(self.source) if self.source else None,
        }
        (self.folder / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def save_segments(self) -> None:
        save_segments(self.segments, self.folder / SEGMENTS_FILE)

    def keep_original(self) -> None:
        """Before the first hand edit, set Whisper's transcript aside so it's never lost."""
        original = self.folder / ORIGINAL_SEGMENTS_FILE
        if not original.exists() and (self.folder / SEGMENTS_FILE).exists():
            shutil.copy2(self.folder / SEGMENTS_FILE, original)

    def save_cleaned(self) -> None:
        save_segments(self.segments, self.folder / CLEANED_FILE)

    def save_entities(self) -> None:
        (self.folder / ENTITIES_FILE).write_text(json.dumps(self.entities, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def save_tags(self) -> None:
        tags = {"proposed": self.tags, "rejected": self.rejected_tags}
        (self.folder / TAGS_FILE).write_text(json.dumps(tags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- serializer -------------------------------------------------------------


def render(memo: Memo, body: str | None = None) -> str:
    """The note as markdown. ``body`` is the note pane's text when the user edited it.

    One paragraph per segment with a block id; when the body has as many
    paragraphs as there are segments they pair up, otherwise the whole body is
    one paragraph. Links wrap the first occurrence of each entity.
    """
    paragraphs = [p.strip() for p in (body if body is not None else memo.cleaned_text).split("\n\n") if p.strip()]
    if len(paragraphs) != len(memo.segments):
        paragraphs = ["\n\n".join(paragraphs)] if paragraphs else []
    linked = link_entities(paragraphs, memo.entities)
    lines = ["---", f"date: {memo.created.isoformat(timespec='seconds')}"]
    if memo.has_audio:
        lines.append(f"audio: {memo.stem}.wav")
    lines += [f"tags: [{', '.join(t for t in memo.tags if t not in memo.rejected_tags)}]", "---", f"# {memo.stem}", ""]
    for index, paragraph in enumerate(linked):
        lines += [f"{paragraph} ^{memo.stem}-{index}", ""]
    if memo.segments:
        lines += ["> [!quote]- Original"] + [f"> {s.text.strip()}" for s in memo.segments] + [""]
    if memo.has_audio:
        lines += [f"![[{memo.stem}.wav]]", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


LINK = re.compile(r"\[\[([^\]|]*)(?:\|[^\]]*)?\]\]")


def link_entities(paragraphs: list[str], entities: list[str]) -> list[str]:
    """Wrap the first occurrence of each entity, across the paragraphs in order.

    An entity the text already links (the user painted it, or an earlier save
    did) is left alone, and nothing inside an existing link is touched.
    """
    already = {m.group(1).casefold() for p in paragraphs for m in LINK.finditer(p)}
    pending = [e for e in entities if e.casefold() not in already]
    out = []
    for paragraph in paragraphs:
        for entity in list(pending):
            pattern = re.compile(r"(?<!\w)" + re.escape(entity) + r"(?!\w)(?![^\[]*\]\])", re.IGNORECASE)
            paragraph, count = pattern.subn(lambda m: f"[[{m.group(0)}]]", paragraph, count=1)
            if count:
                pending.remove(entity)
        out.append(paragraph)
    return out


# --- vault ------------------------------------------------------------------


def is_vault(path: Path) -> bool:
    return (path / ".obsidian").is_dir()


def obsidian_registry() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "obsidian" / "obsidian.json"


def registered_vaults(registry: Path | None = None) -> list[Path]:
    """The vault folders Obsidian has opened before; empty when it has never run."""
    registry = registry or obsidian_registry()
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [Path(v["path"]) for v in data.get("vaults", {}).values() if isinstance(v, dict) and v.get("path")]


def is_registered(vault: Path, registry: Path | None = None) -> bool:
    try:
        target = vault.resolve()
    except OSError:
        return False
    return any(p.resolve() == target for p in registered_vaults(registry) if p.exists())


def unique_path(path: Path) -> Path:
    """``path`` itself, or the first ``stem-NN`` beside it that doesn't exist yet."""
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{counter:02d}{path.suffix}")
        counter += 1
    if candidate != path:
        log.info("%s exists, using %s", path.name, candidate.name)
    return candidate


def write_note(path: Path, text: str) -> None:
    """Through a temp file beside the target, so a sync client never sees half a note."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    except OSError as e:
        raise SaveError(f"Could not save: {e.strerror or e}", str(path)) from e


def copy_audio(memo: Memo, vault: Path, attachments_subdir: str) -> Path | None:
    """memo.wav into the vault's attachments folder under the memo's stem; None without audio."""
    if not memo.has_audio:
        return None
    folder = vault / attachments_subdir
    try:
        if not folder.exists():
            folder.mkdir(parents=True)
            log.info("created attachments folder %s", folder)
        target = folder / f"{memo.stem}.wav"
        if not target.exists() or target.stat().st_size != memo.wav.stat().st_size:
            shutil.copy2(memo.wav, target)
    except OSError as e:
        raise SaveError(f"Could not copy audio: {e.strerror or e}", str(folder)) from e
    return target


def open_note(path: Path, registered: bool) -> str:
    """Open in Obsidian when the vault is registered, else with the OS handler. Returns what happened."""
    if registered:
        webbrowser.open(f"obsidian://open?path={quote(str(path))}")
        return "Opened in Obsidian"
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except OSError as e:
        raise SaveError("Could not open the note", str(e)) from e
    return "Opened with the system handler, open this vault in Obsidian once to use Obsidian"


# --- trash ------------------------------------------------------------------


def discard(memo: Memo, trash_dir: Path, vault: Path | None, attachments_subdir: str) -> list[Path]:
    """Move the working folder to the trash, and a saved note plus its audio copy too. Returns what moved."""
    moved = []
    try:
        trash_dir.mkdir(parents=True, exist_ok=True)
        target = unique_path(trash_dir / memo.stem)
        extras = []
        if memo.saved_to is not None and memo.saved_to.exists():
            extras.append(memo.saved_to)
        if vault is not None and (vault / attachments_subdir / f"{memo.stem}.wav").exists():
            extras.append(vault / attachments_subdir / f"{memo.stem}.wav")
        shutil.move(str(memo.folder), str(target))
        moved.append(target)
        for extra in extras:
            shutil.move(str(extra), str(unique_path(target / extra.name)))
            moved.append(extra)
    except OSError as e:
        raise SaveError(f"Could not discard: {e.strerror or e}", str(memo.folder)) from e
    log.info("%s: discarded to %s%s", memo.stem, target, f" with {len(extras)} vault file(s)" if extras else "")
    return moved


def purge_trash(trash_dir: Path, days: int = TRASH_DAYS, now: float | None = None) -> int:
    """Remove trash entries older than ``days`` by modification time. Returns how many went."""
    if not trash_dir.exists():
        return 0
    cutoff = (now or time.time()) - days * 86400
    removed = 0
    for entry in trash_dir.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                removed += 1
        except OSError as e:
            log.warning("couldn't purge %s: %s", entry, e.strerror)
    if removed:
        log.info("purged %d trash entries older than %d days", removed, days)
    return removed
