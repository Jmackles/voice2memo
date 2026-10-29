"""Settings dialog, applied without restart.

Refuses to save the PTT and push-to-note hotkeys equal.

No model name is written down here. The Models tab offers what each provider
says it has when the dialog opens (Ollama: what's pulled; OpenAI, Anthropic:
what the key can use), the Whisper list is faster-whisper's own, and a model
that isn't on its list is flagged under the field.

The Audio tab picks the microphone and the speakers by name, from the list
capture.list_devices gives, with "System default" first. It can test both: a
level meter on the microphone, a short tone on the speakers. Both stop when the
dialog closes, and a device that isn't connected stays selected as such.

Built ahead of the config file so the rules for secrets existed before any
secret was stored. A field in config.SECRET_FIELDS is masked, never handed to the
key log (which refuses masked fields anyway), and describe_changes() never logs
its value.
"""

from __future__ import annotations

import functools
import logging
import queue
import re
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, ttk

from voice2memos import capture, cleanup, config, transcribe
from voice2memos.config import SECRET_FIELDS, Field
from voice2memos.errors import CaptureError
from voice2memos.ui.panes import KeyLog, replace_text
from voice2memos.ui.player import Playback, tone

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

HOTKEY = ("capture", "hotkey")
NOTE_HOTKEY = ("capture", "note_hotkey")
PROMPT = ("cleanup", "prompt")
INPUT_DEVICE = ("capture", "input_device")
OUTPUT_DEVICE = ("playback", "output_device")
DEVICE_KINDS = {INPUT_DEVICE: "input", OUTPUT_DEVICE: "output"}
TEST_TICK_MS = 50
AUDIO_LABEL_WIDTH = 11  # "Microphone" and "Speakers" alike, so the two lists line up

# The providers the app has code for. "fake" runs that stage from a fixture, see
# [fake] in the example cfg. Model names are never listed here: the Models tab
# asks each provider, and the Whisper list comes from faster-whisper itself.
PROVIDERS = ["ollama", "openai", "anthropic", "fake"]
FAKE_WHISPER = "fake"
DEVICES = ["auto", "cpu", "cuda"]
WHISPER_MODEL = ("whisper", "model_size")
NOTE_COLOR = "#666666"
WARNING_COLOR = "#a35c00"
LIST_POLL_MS = 100
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


def _answer_into(answers: queue.Queue, source: str, ask: Callable[[], tuple[list[str], str]]) -> None:
    """A model-list thread: ask, and leave the answer for the dialog to pick up. Never raises."""
    try:
        answer = ask()
    except Exception:
        log.exception("listing %s models failed", source)
        answer = ([], "Couldn't list the models, see log")
    answers.put((source, answer))


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
        config_path: Path | None = None,
        on_rescan: Callable[[], str] | None = None,
    ) -> None:
        """``on_rescan`` has PortAudio look for devices again and returns why it couldn't, or blank."""
        self.keylog = keylog
        self.on_save = on_save
        self.on_rescan = on_rescan
        self.device_choices: dict[Field, dict[str, str]] = {}  # field -> shown text -> value in the .cfg
        self.meter: capture.LevelMeter | None = None
        self.test_tone: Playback | None = None
        self.test_timer: str | None = None
        # Model lists, fetched off the UI thread: source ("whisper" or a provider) -> (names, note).
        self.model_notes: dict[str, ttk.Label] = {}
        self.model_lists: dict[str, tuple[list[str], str]] = {}
        self.listing: set[str] = set()
        self.lists_ready: queue.Queue[tuple[str, tuple[list[str], str]]] = queue.Queue()
        self.list_timer: str | None = None
        self.config_path = config_path or config.CONFIG_PATH  # relative folders are relative to its folder
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
        notebook.add(self._audio_tab(notebook), text="Audio")
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
        self.list_models("whisper")
        for section in ("cleanup", "entities"):
            self.list_models(section)

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
        ttk.Label(folders, text="Blank trash keeps it inside the recordings folder. ~ is your home folder.").grid(
            column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        whisper = self._group(tab, "Transcription")
        self._choice(whisper, WHISPER_MODEL, "Whisper model", [])
        self.model_notes["whisper"] = self._note(whisper)
        self._choice(whisper, ("whisper", "device"), "Device", DEVICES)
        paragraphs = self._group(tab, "Note")
        self._entry(paragraphs, ("note", "paragraph_seconds"), "Paragraph length")
        self._note(paragraphs).configure(
            text="Seconds of speech per paragraph at most; a pause breaks one sooner. 0 keeps Whisper's own sentences."
        )
        return tab

    def _audio_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=10)
        mic = self._group(tab, "Input")
        self._device(mic, INPUT_DEVICE, "Microphone")
        self.mic_test_button = ttk.Button(mic, text="Test", width=12, command=self.toggle_mic_test)
        self.mic_test_button.grid(row=0, column=2, padx=(6, 0))
        row = self._label(mic, "Level", width=AUDIO_LABEL_WIDTH)
        self.level_bar = ttk.Progressbar(mic, maximum=1.0, mode="determinate")
        self.level_bar.grid(row=row, column=1, sticky="ew")
        speakers = self._group(tab, "Output")
        self._device(speakers, OUTPUT_DEVICE, "Speakers")
        self.tone_button = ttk.Button(speakers, text="Play a tone", width=12, command=self.play_tone)
        self.tone_button.grid(row=0, column=2, padx=(6, 0))
        bottom = ttk.Frame(tab)
        bottom.pack(fill="x", pady=(4, 0))
        ttk.Button(bottom, text="Rescan devices", command=self.rescan).pack(side="left")
        self.audio_note = ttk.Label(bottom, text="Plugged something in? Rescan to list it.")
        self.audio_note.pack(side="left", padx=(10, 0))
        self._load_devices()
        return tab

    def _models_tab(self, notebook: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(notebook, padding=10)
        for section, title in (("cleanup", "Rewrite"), ("entities", "Entities and tags")):
            group = self._group(tab, title)
            self._choice(group, (section, "provider"), "Provider", PROVIDERS)
            self._choice(group, (section, "model"), "Model", [], editable=True)
            self.model_notes[section] = self._note(group)
            self._temperature(group, (section, "temperature"), "Temperature")
            self.widgets[(section, "provider")].bind(
                "<<ComboboxSelected>>", lambda _e, s=section: self.list_models(s), add="+"
            )
            for event in ("<<ComboboxSelected>>", "<FocusOut>"):
                self.widgets[(section, "model")].bind(event, lambda _e, s=section: self._show_model_note(s), add="+")
        keys = self._group(tab, "API keys")
        self._entry(keys, ("keys", "openai"), "OpenAI")
        self._entry(keys, ("keys", "anthropic"), "Anthropic")
        for provider in ("openai", "anthropic"):
            # A key typed or changed here: ask that provider again, with it.
            self.widgets[("keys", provider)].bind("<FocusOut>", lambda _e, p=provider: self._key_changed(p), add="+")
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

    def _label(self, group: ttk.Frame, text: str, width: int | None = None) -> int:
        """``width`` in characters lines the fields up across groups, which each have their own grid."""
        row = group.grid_size()[1]
        ttk.Label(group, text=text, width=width).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
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

    def _note(self, group: ttk.Frame) -> ttk.Label:
        """A line of small print under the field above it, in the second column."""
        note = ttk.Label(group, foreground=NOTE_COLOR, wraplength=420)
        note.grid(row=group.grid_size()[1], column=1, columnspan=2, sticky="w", pady=(0, 4))
        return note

    def _device(self, group: ttk.Frame, field: Field, label: str) -> None:
        row = self._label(group, label, width=AUDIO_LABEL_WIDTH)
        var = self.vars[field] = tk.StringVar()
        combo = ttk.Combobox(group, textvariable=var, state="readonly", width=44)
        combo.grid(row=row, column=1, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._device_picked(field), add="+")
        self.widgets[field] = combo

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
        # Start where the field points, which for "~/..." or a relative folder isn't the text itself.
        current = var.get().strip()
        initial = config.resolve_folder(current, self.config_path) if current else None
        folder = filedialog.askdirectory(parent=self.window, initialdir=initial)
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
        for field in DEVICE_KINDS:
            self._show_device(field, values.get(field, ""))

    def values(self) -> dict[Field, str]:
        values = {
            field: f"{var.get():g}" if isinstance(var, tk.DoubleVar) else var.get().strip()
            for field, var in self.vars.items()
        }
        for field, text in self.texts.items():
            values[field] = text.get("1.0", "end-1c").strip()
        for field in DEVICE_KINDS:
            values[field] = self.device_value(field)
        if values[PROMPT] == default_prompt():
            values[PROMPT] = ""  # blank in the .cfg means use prompts/rewrite.txt
        return values

    def save(self) -> None:
        values = self.values()
        if values[HOTKEY] and values[HOTKEY] == values[NOTE_HOTKEY]:
            self.error.configure(text="Push to talk and push to note need different keys.")
            log.info("settings not saved: both hotkeys are %s", values[HOTKEY])
            return
        self.close()
        self.on_save(values)

    def cancel(self) -> None:
        log.info("settings closed without saving")
        self.close()

    def close(self) -> None:
        self.stop_mic_test()
        self._stop_tone()
        for timer in (self.test_timer, self.list_timer):
            if timer is not None:
                self.window.after_cancel(timer)
        self.keylog.flush_all()
        self.window.destroy()

    # --- model lists -----------------------------------------------------------

    def list_models(self, target: str) -> None:
        """Fill ``target``'s model list ("whisper", "cleanup" or "entities") from what exists right now.

        Each source is asked once per dialog, on a thread of its own, since
        faster-whisper takes seconds to import and the API providers are a
        network call away. The list fills in when the answer arrives.
        """
        source = self._source(target)
        if source in self.model_lists:
            self._show_models(target)
            return
        self.model_notes[target].configure(text="Looking for models…", foreground=NOTE_COLOR)
        if source in self.listing:
            return
        self.listing.add(source)
        if source == "whisper":
            ask = transcribe.whisper_models
        else:
            key = self.vars[("keys", source)].get() if ("keys", source) in self.vars else ""
            ask = functools.partial(cleanup.available_models, source, key)
        # The thread gets the queue and the question, nothing of tk's.
        threading.Thread(target=_answer_into, args=(self.lists_ready, source, ask), name=f"models-{source}", daemon=True).start()
        if self.list_timer is None:
            self.list_timer = self.window.after(LIST_POLL_MS, self._lists_arrived)

    def _lists_arrived(self) -> None:
        self.list_timer = None
        while True:
            try:
                source, answer = self.lists_ready.get_nowait()
            except queue.Empty:
                break
            self.listing.discard(source)
            self.model_lists[source] = answer
            log.info("settings: %s lists %d model(s)", source, len(answer[0]))
            for target in ("whisper", "cleanup", "entities"):
                if self._source(target) == source:
                    self._show_models(target)
        if self.listing:
            self.list_timer = self.window.after(LIST_POLL_MS, self._lists_arrived)

    def _source(self, target: str) -> str:
        return "whisper" if target == "whisper" else self.vars[(target, "provider")].get().strip()

    def _show_models(self, target: str) -> None:
        names, _ = self.model_lists[self._source(target)]
        if target == "whisper":
            self.widgets[WHISPER_MODEL].configure(values=names + [FAKE_WHISPER])
        else:
            self.widgets[(target, "model")].configure(values=names)
        self._show_model_note(target)

    def _show_model_note(self, target: str) -> None:
        """Say what the list is, or that the model typed in isn't on it."""
        source = self._source(target)
        if source not in self.model_lists:
            return
        names, note = self.model_lists[source]
        field = WHISPER_MODEL if target == "whisper" else (target, "model")
        model = self.vars[field].get().strip()
        if names and model and model != FAKE_WHISPER and not any(cleanup.same_model(model, n) for n in names):
            where = "faster-whisper" if source == "whisper" else cleanup.PROVIDER_NAMES.get(source, source)
            have = "pulled in" if source == "ollama" else "offered by"
            text, color = f"{model} isn't {have} {where}; pick one from the list.", WARNING_COLOR
        elif note:
            text, color = note, WARNING_COLOR if not names and source != "fake" else NOTE_COLOR
        else:
            have = "pulled in Ollama" if source == "ollama" else "available"
            text, color = f"{len(names)} models {have}.", NOTE_COLOR
        self.model_notes[target].configure(text=text, foreground=color)

    def _key_changed(self, provider: str) -> None:
        self.model_lists.pop(provider, None)
        for section in ("cleanup", "entities"):
            if self._source(section) == provider:
                self.list_models(section)

    # --- audio devices -------------------------------------------------------

    def _load_devices(self) -> None:
        """Fill both device lists from PortAudio, keeping whatever each one shows now."""
        current = {field: self.device_value(field) for field in self.device_choices}
        try:
            table = capture.device_table()
        except Exception as e:  # no sounddevice, or PortAudio failing
            log.warning("couldn't list audio devices: %s: %s", type(e).__name__, e)
            table = None
        for field, kind in DEVICE_KINDS.items():
            default = capture.default_device_name(kind, table) if table else ""
            choices = {f"System default ({default})" if default else "System default": ""}
            for name in capture.list_devices(kind, table) if table else []:
                choices[name] = name
            self.device_choices[field] = choices
            self.widgets[field].configure(values=list(choices))
        for field, value in current.items():
            self._show_device(field, value)

    def _show_device(self, field: Field, value: str) -> None:
        """Select ``value`` in its list. One that isn't connected is added as such, so saving keeps it."""
        choices = self.device_choices[field]
        for shown, choice in choices.items():
            if choice == value:
                self.vars[field].set(shown)
                return
        shown = "Fake recorder, for testing" if value == "fake" else f"{value} (not connected)"
        choices[shown] = value
        self.widgets[field].configure(values=list(choices))
        self.vars[field].set(shown)

    def device_value(self, field: Field) -> str:
        """The .cfg value behind what the list shows: a device name, or blank for the default."""
        return self.device_choices[field].get(self.vars[field].get(), "")

    def _device_picked(self, field: Field) -> None:
        if field == INPUT_DEVICE and self.meter is not None:
            self.stop_mic_test()
            self.toggle_mic_test()  # carry on testing, on the new microphone

    def toggle_mic_test(self) -> None:
        """Open the chosen microphone and show its level, until pressed again or the dialog closes."""
        if self.meter is not None:
            self.stop_mic_test()
            return
        name = self.device_value(INPUT_DEVICE)
        meter = capture.LevelMeter(name)
        try:
            note = meter.start()
        except CaptureError as e:
            log.warning("microphone test: %s: %s", e.message, e.detail)
            self.audio_note.configure(text=e.message)
            return
        self.meter = meter
        self.mic_test_button.configure(text="Stop")
        self.audio_note.configure(text=note or "Say something: the level should move.")
        log.info("settings: microphone test on %s", name or "the system default")
        self._tick_test()

    def stop_mic_test(self) -> None:
        if self.meter is None:
            return
        self.meter.stop()
        self.meter = None
        self.level_bar["value"] = 0
        self.mic_test_button.configure(text="Test")
        log.info("settings: microphone test off")

    def play_tone(self) -> None:
        """A short beep on the chosen speakers."""
        self._stop_tone()
        name = self.device_value(OUTPUT_DEVICE)
        beep = Playback()
        beep.load_frames(tone(), 44100)
        beep.output_device = name
        try:
            beep.play()
        except Exception as e:  # sounddevice raises its own classes, plus OSError without a device
            log.warning("speaker test: %s: %s", type(e).__name__, e)
            self.audio_note.configure(text="Can't play on that device, see log")
            return
        self.test_tone = beep
        self.audio_note.configure(text=beep.note or "You should hear a short beep.")
        log.info("settings: test tone on %s", name or "the system default")
        self._tick_test()

    def _stop_tone(self) -> None:
        if self.test_tone is not None:
            self.test_tone.unload()
            self.test_tone = None

    def _tick_test(self) -> None:
        """While a test runs: move the level bar, and close the tone's stream once it has played."""
        if self.test_timer is not None:
            self.window.after_cancel(self.test_timer)
            self.test_timer = None
        if self.meter is not None:
            self.level_bar["value"] = self.meter.level
        if self.test_tone is not None and self.test_tone.finished:
            self._stop_tone()
        if self.meter is not None or self.test_tone is not None:
            self.test_timer = self.window.after(TEST_TICK_MS, self._tick_test)

    def rescan(self) -> None:
        """Look for devices plugged in since the list was made."""
        self.stop_mic_test()
        self._stop_tone()
        problem = self.on_rescan() if self.on_rescan is not None else ""
        if problem:
            self.audio_note.configure(text=problem)
            return
        self._load_devices()
        self.audio_note.configure(text="Device list updated.")
        log.info("settings: audio devices rescanned")
