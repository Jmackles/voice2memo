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
      title.json        the model's name for the note, and the user's if they typed one
      note.md           draft rendered from the above, regenerated on save
      state.json        saved_to path and the name saved under, created time, source file if opened

Note file layout, written here and nowhere else::

    ---
    date: 2026-10-27T09:34:00-04:00
    audio: 20261027-093400.wav
    tags: [idea, database]
    aliases: ["Database thoughts"]        only when the file name isn't the title
    ---
    # Database thoughts

    **0:00** <cleaned text, one paragraph per segment, links in [[ ]]>

    **0:31** <the next paragraph>

    ---

    See also: [[Database thoughts]]            only when other notes have this name

    > [!quote]- Original
    > **0:00** <raw transcript, one callout line per segment>
    > **0:31** <...>

    ![[20261027-093400.wav]]

A segment is a paragraph of speech (see ``transcribe.group_segments``). The
bold stamp is where it starts in the audio, left out of a typed note. No block
ids: an Obsidian ``^id`` can only sit at the end of its paragraph, where it
reads as clutter, so they wait for the subnotes step that would use them.

The note is named for its title (``Memo.title``: the user's, else the name it
was saved under, else the model's, else the stem). ``place`` picks the file:
the bare title when no note in the vault has that name, otherwise the title
with the memo's date and time after it. Obsidian resolves links by file name
across the whole vault, so the check is vault-wide. Such a note also carries
the bare title as an alias, which Obsidian's quick switcher and link
suggestions match, and a See also line linking the notes of that name. The
older notes then list it in their backlinks without the app touching them.

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
from voice2memos.cleanup import CLEANED_FILE, ENTITIES_FILE, TAGS_FILE, TITLE_FILE, sanitize_title
from voice2memos.errors import SaveError
from voice2memos.transcribe import SEGMENTS_FILE, Segment, display_text, format_time, load_segments, save_segments

log = logging.getLogger(__name__)

NOTE_FILE = "note.md"
STATE_FILE = "state.json"
ORIGINAL_SEGMENTS_FILE = "segments.original.json"
TRASH_DAYS = 30
NAME_STAMP = "%Y-%m-%d %H%M"  # after a title the vault already has: "Groceries 2026-11-01 1443"
STAMPED = r"( \d{4}-\d{2}-\d{2} \d{4}(-\d{2})?)?"


@dataclass
class Memo:
    """One memo's working folder and what has been worked out about it so far."""

    folder: Path
    segments: list[Segment] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    rejected_tags: list[str] = field(default_factory=list)
    proposed_title: str = ""  # the title stage's
    chosen_title: str = ""  # typed into the Name field; wins over everything
    saved_to: Path | None = None
    saved_title: str = ""  # the title saved_to was written under
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

    @property
    def linked_text(self) -> str:
        """The Note pane once the entities are in: the cleaned text with each entity's first mention in [[ ]]."""
        return "\n\n".join(link_entities(self.cleaned_text.split("\n\n"), self.entities))

    @property
    def title(self) -> str:
        """The note's name: the user's, else what it was saved as, else the model's, else the stem.

        A saved note keeps its name through a Clean up rerun, so the vault file isn't
        renamed behind the user's back; only typing a new one renames it. A note saved
        under the stem, because naming hadn't worked, takes the model's name when it comes."""
        saved = self.saved_title if self.saved_to is not None and self.saved_title != self.stem else ""
        for candidate in (self.chosen_title, saved, self.proposed_title):
            name = sanitize_title(candidate)
            if name:
                return name
        return self.stem

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
            memo.saved_title = state.get("saved_title") or ""
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
        if (folder / TITLE_FILE).exists():
            title = json.loads((folder / TITLE_FILE).read_text(encoding="utf-8"))
            memo.proposed_title, memo.chosen_title = title.get("proposed", ""), title.get("chosen", "")
        return memo

    def save_state(self) -> None:
        state = {
            "saved_to": str(self.saved_to) if self.saved_to else None,
            "saved_title": self.saved_title,
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

    def save_title(self) -> None:
        title = {"proposed": self.proposed_title, "chosen": self.chosen_title}
        (self.folder / TITLE_FILE).write_text(json.dumps(title, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def save_tags(self) -> None:
        tags = {"proposed": self.tags, "rejected": self.rejected_tags}
        (self.folder / TAGS_FILE).write_text(json.dumps(tags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- serializer -------------------------------------------------------------


@dataclass
class Placement:
    """Where a note goes and how it sits among the notes already in the vault."""

    path: Path
    title: str
    see_also: list[str] = field(default_factory=list)  # other notes of this title, by file name

    @property
    def alias(self) -> bool:
        """The file name isn't the bare title, so the title goes in as an alias."""
        return self.path.stem != self.title


def vault_notes(folder: Path) -> dict[str, str]:
    """Every note name under ``folder``, casefolded, to its own spelling. Hidden folders
    (.obsidian, .trash) are skipped; a folder that can't be read gives what was found."""
    names: dict[str, str] = {}
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.lower().endswith(".md"):
                stem = name[:-3]
                names.setdefault(stem.casefold(), stem)
    return names


def place(memo: Memo, folder: Path, names: dict[str, str], path: Path | None = None) -> Placement:
    """Where ``memo`` is saved, given the note names already in the vault.

    ``path`` is a file the user picked; otherwise a saved note stays where it is
    until its title changes, and a new or renamed one goes into ``folder`` under
    the bare title, or the title and the memo's date and time when that name is
    taken. The memo's own file never counts as taken.
    """
    title = memo.title
    own = memo.saved_to.stem.casefold() if memo.saved_to is not None else None
    others = {k: v for k, v in names.items() if k != own}
    if path is None and memo.saved_to is not None and memo.saved_title == title:
        path = memo.saved_to
    if path is None:
        path = folder / f"{title}.md"
        stamp = memo.created.strftime(NAME_STAMP)
        counter = 0
        while path.stem.casefold() in others or (path.exists() and path != memo.saved_to):
            counter += 1
            path = folder / (f"{title} {stamp}.md" if counter == 1 else f"{title} {stamp}-{counter - 1:02d}.md")
    same = re.compile(re.escape(title.casefold()) + STAMPED)
    see_also = sorted(v for k, v in others.items() if same.fullmatch(k) and k != path.stem.casefold())
    return Placement(path, title, see_also)


def render(memo: Memo, body: str | None = None, placement: Placement | None = None) -> str:
    """The note as markdown. ``body`` is the note pane's text when the user edited it;
    ``placement`` is where it's going, which decides the alias and the See also line.

    One paragraph per segment, each stamped with where it starts in the audio;
    when the body has as many paragraphs as there are segments they pair up, otherwise the whole body is one paragraph from
    the first segment's start. Links wrap the first occurrence of each entity.
    A rule closes the body; the footer under it holds the links to notes of
    the same name, the original words, collapsed, and the audio.
    """
    paragraphs = [p.strip() for p in (body if body is not None else memo.cleaned_text).split("\n\n") if p.strip()]
    if len(paragraphs) != len(memo.segments):
        paragraphs = ["\n\n".join(paragraphs)] if paragraphs else []
    linked = link_entities(paragraphs, memo.entities)
    lines = ["---", f"date: {memo.created.isoformat(timespec='seconds')}"]
    if memo.has_audio:
        lines.append(f"audio: {memo.stem}.wav")
    lines.append(f"tags: [{', '.join(t for t in memo.tags if t not in memo.rejected_tags)}]")
    title = placement.title if placement is not None else memo.title
    if placement is not None and placement.alias:
        lines.append(f'aliases: ["{title}"]')
    lines += ["---", f"# {title}", ""]
    for index, paragraph in enumerate(linked):
        lines += [f"{_stamp(memo, index)}{paragraph}", ""]
    see_also = placement.see_also if placement is not None else []
    if memo.segments or memo.has_audio or see_also:
        lines += ["---", ""]
    if see_also:
        lines += ["See also: " + ", ".join(f"[[{name}]]" for name in see_also), ""]
    if memo.segments:
        lines += ["> [!quote]- Original"] + [f"> {_stamp(memo, s.index)}{s.text.strip()}" for s in memo.segments] + [""]
    if memo.has_audio:
        lines += [f"![[{memo.stem}.wav]]", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _stamp(memo: Memo, index: int) -> str:
    """Where paragraph ``index`` starts in the audio, as **m:ss**; nothing for a typed note."""
    if not memo.has_audio or not memo.segments:
        return ""
    segment = memo.segments[min(index, len(memo.segments) - 1)]
    return f"**{format_time(segment.start)}** "


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


def remove_old_copy(old: Path, new: Path) -> None:
    """After a rename the note is at ``new``; ``old`` goes, unless it is the same file
    (a change of case only, on a file system that ignores case)."""
    try:
        if old.exists() and not os.path.samefile(old, new):
            old.unlink()
    except OSError as e:
        raise SaveError(f"Saved as {new.name}, but {old.name} is still there: {e.strerror or e}", str(old)) from e


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
