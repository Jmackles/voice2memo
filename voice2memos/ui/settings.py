"""Settings dialog, applied without restart.

Refuses to save the PTT and push-to-note hotkeys equal.

Built ahead of the config file so the rules for secrets existed before any
secret was stored. A field in config.SECRET_FIELDS is masked, never handed to the
key log (which refuses masked fields anyway), and describe_changes() never logs
its value.
"""

from __future__ import annotations

import logging
import re
import sys
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, ttk

from voice2memos.config import SECRET_FIELDS, Field
from voice2memos.ui.panes import KeyLog, replace_text

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

HOTKEY = ("capture", "hotkey")
NOTE_HOTKEY = ("capture", "note_hotkey")
PROMPT = ("cleanup", "prompt")

# "fake" on both lists runs that stage from a fixture, see [fake] in the example cfg.
PROVIDERS = ["ollama", "openai", "anthropic", "fake"]
WHISPER_SIZES = ["tiny", "base", "small", "medium", "large", "fake"]
DEVICES = ["auto", "cpu", "cuda"]
CAPTURE_PROMPT = "Press a key…"

# Tk modifier bits as pynput names, in the order pynput chords are written.
if sys.platform == "win32":
    MODIFIERS = [(0x4, "<ctrl>"), (0x20000, "<alt>"), (0x1, "<shift>")]
elif sys.platform == "darwin":
    MODIFIERS = [(0x4, "<ctrl>"), (0x10, "<alt>"), (0x8, "<cmd>"), (0x1, "<shift>")]
else:
    MODIFIERS = [(0x4, "<ctrl>"), (0x8, "<alt>"), (0x1, "<shift>")]

# Tk keysyms pynput spells differently. Tab and Escape are left out: Tab moves
# between fields and Escape cancels a capture.
PYNPUT_KEYS = {
    "Return": "enter",
    "BackSpace": "backspace",
    "Delete": "delete",
    "Insert": "insert",
    "Home": "home",
    "End": "end",
    "Prior": "page_up",
    "Next": "page_down",
    "Left": "left",
    "Right": "right",
    "Up": "up",
    "Down": "down",
    "space": "space",
    "Pause": "pause",
    "Scroll_Lock": "scroll_lock",
    "Print": "print_screen",
}


def format_hotkey(keysym: str, state: int) -> str:
    """A Tk keypress as a pynput hotkey like ``<ctrl>+<f9>``; empty for a bare modifier or unusable key."""
    if re.fullmatch(r"F\d{1,2}", keysym):
        key = f"<{keysym.lower()}>"
    elif keysym in PYNPUT_KEYS:
        key = f"<{PYNPUT_KEYS[keysym]}>"
    elif len(keysym) == 1 and keysym.isalnum():
        key = keysym.lower()
    else:
        return ""
    return "+".join([name for mask, name in MODIFIERS if state & mask] + [key])


def default_prompt() -> str:
    return (PROMPTS_DIR / "rewrite.txt").read_text(encoding="utf-8").strip()


def describe_changes(old: dict[Field, str], new: dict[Field, str]) -> list[str]:
    """One log line per changed field. A secret's line says only that it changed, never its value."""
    lines = []
    for field in sorted(new):
        before, after = old.get(field, ""), new[field]
        if before == after:
            continue
        name = ".".join(field)
        if field in SECRET_FIELDS:
            lines.append(f"{name}: {'cleared' if not after else 'changed' if before else 'set'}")
        elif "\n" in before or "\n" in after:
            lines.append(f"{name}: edited, now {len(after.splitlines())} lines")
        else:
            lines.append(f"{name}: {before!r} -> {after!r}")
    return lines


class SettingsDialog:
    """The Settings window. Every field is a (section, key) in the .cfg."""

    def __init__(
        self,
        parent: tk.Misc,
        values: dict[Field, str],
        keylog: KeyLog,
        on_save: Callable[[dict[Field, str]], None],
    ) -> None:
        self.keylog = keylog
        self.on_save = on_save
        self.vars: dict[Field, tk.Variable] = {}
        self.texts: dict[Field, tk.Text] = {}
        self.widgets: dict[Field, tk.Widget] = {}
        self.watched: set[Field] = set()
        self.before_capture: dict[Field, str] = {}

        self.window = tk.Toplevel(parent)
        self.window.title("Settings")
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)
        self.window.bind("<Escape>", lambda _e: self.cancel())

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        notebook.add(self._general_tab(notebook), text="General")
        notebook.add(self._models_tab(notebook), text="Models")
        notebook.add(self._prompt_tab(notebook), text="Prompt")
        notebook.add(self._words_tab(notebook), text="Words")

        bottom = ttk.Frame(self.window, padding=10)
        bottom.pack(fill="x")
        self.error = ttk.Label(bottom, foreground="#c0392b")
        self.error.pack(side="left")
        ttk.Button(bottom, text="Save", command=self.save).pack(side="right")
        ttk.Button(bottom, text="Cancel", command=self.cancel).pack(side="right", padx=(0, 6))
        self.set_values(values)

    # --- tabs -------------------------------------------------------------

    def _general_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=10)
        keys = self._group(tab, "Hotkeys")
        self._hotkey(keys, HOTKEY, "Push to talk")
        self._hotkey(keys, NOTE_HOTKEY, "Push to note")
        folders = self._group(tab, "Folders")
        self._entry(folders, ("vault", "path"), "Obsidian vault", browse=True)
        self._entry(folders, ("capture", "save_dir"), "Recordings", browse=True)
        self._entry(folders, ("capture", "trash_dir"), "Trash", browse=True)
        whisper = self._group(tab, "Transcription")
        self._choice(whisper, ("whisper", "model_size"), "Whisper model", WHISPER_SIZES)
        self._choice(whisper, ("whisper", "device"), "Device", DEVICES)
        return tab

    def _models_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=10)
        for section, title in (("cleanup", "Rewrite"), ("entities", "Entities and tags")):
            group = self._group(tab, title)
            self._choice(group, (section, "provider"), "Provider", PROVIDERS)
            self._choice(group, (section, "model"), "Model", [], editable=True)
            self._temperature(group, (section, "temperature"), "Temperature")
        keys = self._group(tab, "API keys")
        self._entry(keys, ("keys", "openai"), "OpenAI")
        self._entry(keys, ("keys", "anthropic"), "Anthropic")
        ttk.Label(keys, text="Blank uses OPENAI_API_KEY or ANTHROPIC_API_KEY from the environment.").grid(
            column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        return tab

    def _prompt_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=10)
        ttk.Label(tab, text="Rewrite prompt and its examples. Blank or unchanged uses prompts/rewrite.txt.").pack(
            anchor="w"
        )
        self._text(tab, PROMPT, height=16).pack(fill="both", expand=True, pady=(4, 6))
        ttk.Button(tab, text="Reset to default", command=self.reset_prompt).pack(anchor="e")
        return tab

    def _words_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=10)
        tab.columnconfigure((0, 1), weight=1, uniform="words")
        tab.rowconfigure(1, weight=1)
        ttk.Label(tab, text="Tag pool, one per line").grid(row=0, column=0, sticky="w")
        ttk.Label(tab, text="Glossary for Whisper, one term per line").grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._text(tab, ("tags", "pool"), height=14).grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self._text(tab, ("whisper", "glossary"), height=14).grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(4, 0))
        return tab

    # --- field builders ---------------------------------------------------

    def _group(self, tab: ttk.Frame, title: str) -> ttk.LabelFrame:
        group = ttk.LabelFrame(tab, text=title, padding=8)
        group.pack(fill="x", pady=(0, 8))
        group.columnconfigure(1, weight=1)
        return group

    def _label(self, group: ttk.Frame, text: str) -> int:
        row = group.grid_size()[1]
        ttk.Label(group, text=text).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        return row

    def _watch(self, widget: tk.Text | ttk.Entry, field: Field) -> None:
        self.keylog.watch(widget, "settings " + ".".join(field))
        self.watched.add(field)

    def _entry(self, group: ttk.Frame, field: Field, label: str, browse: bool = False) -> None:
        row = self._label(group, label)
        var = self.vars[field] = tk.StringVar()
        secret = field in SECRET_FIELDS
        entry = ttk.Entry(group, textvariable=var, width=44, show="•" if secret else "")
        entry.grid(row=row, column=1, sticky="ew")
        self.widgets[field] = entry
        if not secret:
            self._watch(entry, field)
        if browse:
            ttk.Button(group, text="Browse…", command=lambda: self._browse(var)).grid(row=row, column=2, padx=(6, 0))

    def _choice(self, group: ttk.Frame, field: Field, label: str, choices: list[str], editable: bool = False) -> None:
        row = self._label(group, label)
        var = self.vars[field] = tk.StringVar()
        combo = ttk.Combobox(group, textvariable=var, values=choices, state="normal" if editable else "readonly")
        combo.grid(row=row, column=1, sticky="ew")
        self.widgets[field] = combo
        if editable:
            self._watch(combo, field)

    def _temperature(self, group: ttk.Frame, field: Field, label: str) -> None:
        row = self._label(group, label)
        var = self.vars[field] = tk.DoubleVar()
        shown = ttk.Label(group, width=5)
        var.trace_add("write", lambda *_: shown.configure(text=f"{var.get():.2f}"))
        # Snap to steps of 0.05 as the slider moves.
        scale = ttk.Scale(group, from_=0, to=1, variable=var, command=lambda v: var.set(round(float(v) * 20) / 20))
        scale.grid(row=row, column=1, sticky="ew")
        shown.grid(row=row, column=2, padx=(6, 0))
        self.widgets[field] = scale

    def _hotkey(self, group: ttk.Frame, field: Field, label: str) -> None:
        row = self._label(group, label)
        var = self.vars[field] = tk.StringVar()
        # Readonly: keys pressed here are captured, never typed.
        entry = ttk.Entry(group, textvariable=var, width=24, state="readonly")
        entry.grid(row=row, column=1, sticky="w")
        entry.bind("<FocusIn>", lambda _e: self.start_capture(field))
        entry.bind("<KeyPress>", lambda e: self.capture(field, e.keysym, e.state))
        entry.bind("<FocusOut>", lambda _e: self.end_capture(field))
        self.widgets[field] = entry

    def _text(self, parent: ttk.Frame, field: Field, height: int) -> ttk.Frame:
        frame = ttk.Frame(parent)
        text = tk.Text(frame, wrap="word", width=40, height=height, undo=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.texts[field] = text
        self.widgets[field] = text
        self._watch(text, field)
        return frame

    # --- behavior ---------------------------------------------------------

    def _browse(self, var: tk.StringVar) -> None:
        folder = filedialog.askdirectory(parent=self.window, initialdir=var.get() or None)
        if folder:
            var.set(folder)

    def start_capture(self, field: Field) -> None:
        self.before_capture[field] = self.vars[field].get()
        self.vars[field].set(CAPTURE_PROMPT)

    def capture(self, field: Field, keysym: str, state: int) -> str | None:
        """One keypress in a hotkey field. Returns "break" so Tk does nothing else with it."""
        if keysym in ("Tab", "ISO_Left_Tab"):
            return None  # let focus move on; end_capture puts the old value back
        if keysym == "Escape":
            self.vars[field].set(self.before_capture.pop(field, ""))
        else:
            hotkey = format_hotkey(keysym, state)
            if not hotkey:
                return "break"  # a bare modifier, or a key pynput can't bind: keep waiting
            self.before_capture.pop(field, None)
            self.vars[field].set(hotkey)
            log.info("%s captured %s", ".".join(field), hotkey)
        self.window.focus_set()
        return "break"

    def end_capture(self, field: Field) -> None:
        if self.vars[field].get() == CAPTURE_PROMPT:
            self.vars[field].set(self.before_capture.pop(field, ""))

    def reset_prompt(self) -> None:
        replace_text(self.texts[PROMPT], default_prompt())

    def set_values(self, values: dict[Field, str]) -> None:
        for field, var in self.vars.items():
            value = values.get(field, "")
            var.set(float(value or 0) if isinstance(var, tk.DoubleVar) else value)
        for field, text in self.texts.items():
            replace_text(text, values.get(field, "") or (default_prompt() if field == PROMPT else ""))

    def values(self) -> dict[Field, str]:
        values = {
            field: f"{var.get():g}" if isinstance(var, tk.DoubleVar) else var.get().strip()
            for field, var in self.vars.items()
        }
        for field, text in self.texts.items():
            values[field] = text.get("1.0", "end-1c").strip()
        if values[PROMPT] == default_prompt():
            values[PROMPT] = ""  # blank in the .cfg means use prompts/rewrite.txt
        return values

    def save(self) -> None:
        values = self.values()
        if values[HOTKEY] and values[HOTKEY] == values[NOTE_HOTKEY]:
            self.error.configure(text="Push to talk and push to note need different keys.")
            return
        self.close()
        self.on_save(values)

    def cancel(self) -> None:
        log.info("settings closed without saving")
        self.close()

    def close(self) -> None:
        self.keylog.flush_all()
        self.window.destroy()
