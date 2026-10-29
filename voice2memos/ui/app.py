"""Root window, banner, panes, worker wiring.

The banner buttons do what they say against whatever backends the config
names, fakes included. A memo runs through the pipeline on a worker thread,
one stage at a time, each posting a StageResult back through ``root.after``;
a failed stage degrades to a worse note and offers Retry for that stage only.
The window's size and the split between the panes come from the config and go
back to it on close.
"""

from __future__ import annotations

import configparser
import logging
import queue
import shutil
import threading
import tkinter as tk
import webbrowser
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
from tkinter import ttk

from voice2memos import __version__, capture, cleanup, config, logsetup, note, transcribe
from voice2memos.config import Field
from voice2memos.errors import StageResult, VoiceMemoError
from voice2memos.note import Memo
from voice2memos.ui.panes import KeyLog, make_pane, replace_text
from voice2memos.ui.platform import set_dpi_awareness
from voice2memos.ui.player import PlayerBar
from voice2memos.ui.settings import SettingsDialog, describe_changes

log = logging.getLogger(__name__)

AUDIO_TYPES = [("Audio", "*.wav *.mp3 *.m4a"), ("WAV", "*.wav"), ("MP3", "*.mp3"), ("M4A", "*.m4a")]
HOTKEY_FIELDS = {("capture", "hotkey"), ("capture", "note_hotkey")}
OUTPUT_DEVICE = ("playback", "output_device")
BUSY_BUTTONS = ("open_file", "record")  # and the player's discard, see set_busy
STAGES = ("transcribe", "cleanup", "entities", "tags")
POLL_MS = 50  # how often the UI thread looks for work posted by other threads
STARTUP_CHECK_MS = 300  # after the window is up, so a slow answer never holds it back
MIN_SIZE = (640, 400)  # the design's; the width grows to whatever the banner needs, see __init__

# tkinter ships no usable icons, so Unicode symbols stand in for the PNGs the
# design calls for: no files, no license, drawn by fonts every OS already has.
# Moving to PNGs later only touches this table and the button builders here
# and in player.py. Each toggle shows its pane's current state: the lock is
# read-only, the keyboard is typing, the brush is paint mode. Not 🔓, which
# Windows' monochrome emoji font draws exactly like 🔒.
ICONS = {
    "open_file": "📂",
    "save": "💾",
    "record": "⏺",
    "stop": "⏹",
    "choose_vault": "💎",
    "open_settings": "⚙",
    "open_log": "☰",
    "locked": "🔒",
    "unlocked": "⌨",
    "paint": "🖌",
    "type": "⌨",
    "play": "▶",
    "pause": "⏸",
    "volume": "🔈",
    "discard": "🗑",
    "commit": "+",
}

# Banner buttons left to right: handler name and tooltip. Discard lives in
# the player, beside the recording it throws away.
BANNER = [
    ("open_file", "Open audio file"),
    ("save", "Save (right-click for Save as)"),
    ("record", "Record"),
    ("choose_vault", "Choose vault"),
    ("open_settings", "Settings"),
    ("open_log", "Log"),
]

STATUS_COLORS = {"info": "#666666", "warning": "#a35c00", "error": "#c0392b"}
STATUS_LOG_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
UNLOCKED_TINT = "#fff8e1"


class Tooltip:
    """A small borderless window under ``widget`` while the pointer is over it."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def show(self, _event: tk.Event | None = None) -> None:
        if self.window is not None:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        tk.Label(self.window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, padx=6, pady=2).pack()
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.window.wm_geometry(f"+{x}+{y}")

    def hide(self, _event: tk.Event | None = None) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


class App:
    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.config, config_warnings = config.load(config_path)
        self.left_locked = True
        self.paint_mode = True
        self.status_history: deque[tuple[datetime, str, str]] = deque(maxlen=10)
        self.retry_callback: Callable[[], None] | None = None
        self.history_window: tk.Toplevel | None = None
        self.settings_dialog: SettingsDialog | None = None
        self.buttons: dict[str, ttk.Button] = {}
        self.tooltips: dict[str, Tooltip] = {}
        self.memo: Memo | None = None
        self.busy = False
        self.whisper_loading = False
        self.recording = False
        self.recorder: capture.Recorder | None = None
        self.dirty = False  # the Note has changes the saved copy doesn't
        self.note_hand_edited = False  # the user typed in the Note since the program last filled it
        self.mirror_declined = False  # the user kept those edits rather than let the Note follow the Transcript
        self.note_built_from = ""  # the Transcript text the Note's cleanup ran (or is running) on
        self.filling = False  # the program is writing a pane, not the user
        # Other threads never touch a widget: they put a call here and the UI thread runs it.
        self.posted: queue.Queue[tuple[Callable, tuple]] = queue.Queue()

        root.title(f"voice2memos {__version__}")
        root.geometry(self.config["window"]["geometry"])
        root.protocol("WM_DELETE_WINDOW", self.close)
        # Exceptions inside tk callbacks go here, not to sys.excepthook.
        root.report_callback_exception = self._callback_exception
        self._make_fonts_and_styles()
        self._build_banner()
        self._build_panes()
        # No narrower than the banner, or the player or an icon gets cut off.
        # Measured with the discard button showing, as it does for a recording.
        self.player.discard_button.grid()
        root.update_idletasks()
        root.minsize(max(MIN_SIZE[0], self.banner.winfo_reqwidth()), MIN_SIZE[1])
        self.player.discard_button.grid_remove()
        # Both pane headers as tall as the +, so it appearing doesn't push the Transcript down.
        for pane in (self.left, self.right):
            pane.frame.rowconfigure(0, minsize=self.commit_button.winfo_reqheight() + 3)
        self.player.set_volume(int(self.config["window"]["volume"]))
        self.player.set_output_device(self.config["playback"]["output_device"])
        self.keylog = KeyLog(root)
        self.keylog.watch(self.left.text, "transcript")
        self.keylog.watch(self.right.text, "note")
        self.left.text.bind("<<Modified>>", self._transcript_modified, add="+")
        self.right.text.bind("<<Modified>>", self._note_modified, add="+")
        sash = self.config["window"]["sash"]
        self.sash_to_restore = int(sash) if sash else None
        self.panes.bind("<Configure>", self._restore_sash, add="+")
        for warning in config_warnings:
            self.set_status(warning, "warning")
        if not config_warnings:
            self.set_status("Ready")
        self.hotkeys = capture.HotkeyListener(
            on_press=lambda: self.post(self.hotkey_pressed),
            on_release=lambda: self.post(self.hotkey_released),
            on_note=lambda: self.post(self.push_to_note),
            on_error=lambda text: self.report(text, "warning"),
        )
        self.hotkeys.bind(self.config["capture"]["hotkey"], self.config["capture"]["note_hotkey"])
        note.purge_trash(self.trash_dir)
        self._drain_posted()
        self.check_timer = root.after(STARTUP_CHECK_MS, self.check_models)
        self.warm_up_whisper()

    def warm_up_whisper(self) -> None:
        """Load the Whisper model on a thread now, so the first memo doesn't wait for it.

        Record and Open stay off until it's in: the first run downloads
        hundreds of MB and would otherwise look hung. A failure still lets
        them on, since a recording is kept whether or not it transcribes.
        """
        if self.config["whisper"]["model_size"].strip().lower() == "fake" or self.whisper_loading:
            return
        self.whisper_loading = True
        self._update_capture_buttons()
        self.set_status("Loading Whisper model… (downloads on first use)")
        transcriber = transcribe.make_transcriber(self.config, config.APP_DIR)

        def work() -> None:
            try:
                transcriber.load()
                self.post(self._whisper_ready, transcriber.note)
            except VoiceMemoError as e:
                self.post(self._whisper_ready, e.message, True)
            except Exception:
                log.exception("whisper warm-up crashed")
                self.post(self._whisper_ready, logsetup.UNEXPECTED, True)

        threading.Thread(target=work, name="whisper-load", daemon=True).start()

    def _whisper_ready(self, note: str, failed: bool = False) -> None:
        self.whisper_loading = False
        self._update_capture_buttons()
        if failed:
            self.set_status(f"{note}; recordings are kept, transcription will fail", "error")
        elif note:
            self.set_status(note, "warning")
        else:
            self.set_status("Ready")

    def check_models(self) -> None:
        """Is Ollama up, with the models the config names? Says so in the status line if not.

        Only Ollama: it's local, so asking costs milliseconds and sends nothing
        anywhere. The API providers are checked when Settings lists their models.
        """
        wanted = sorted({
            self.config[section]["model"].strip()
            for section in ("cleanup", "entities")
            if self.config[section]["provider"].strip() == "ollama"
        })
        if not wanted:
            return
        names, problem = cleanup.available_models("ollama", timeout=2)
        if problem:
            self.set_status(f"{problem.split('.')[0]}; cleanup will show the raw transcript", "warning")
            return
        missing = [m for m in wanted if not any(cleanup.same_model(m, n) for n in names)]
        if missing:
            self.set_status(f"Ollama has no {', '.join(missing)}; pick an installed model in Settings > Models", "warning")
        else:
            log.info("ollama has %s", ", ".join(wanted))

    def post(self, func: Callable, *args) -> None:
        """Run ``func(*args)`` on the UI thread soon. Safe from any thread."""
        self.posted.put((func, args))

    def _drain_posted(self) -> None:
        while True:
            try:
                func, args = self.posted.get_nowait()
            except queue.Empty:
                break
            func(*args)
        self.drain_timer = self.root.after(POLL_MS, self._drain_posted)

    # --- folders from the config -------------------------------------------

    @property
    def save_dir(self) -> Path:
        return config.resolve_folder(self.config["capture"]["save_dir"], self.config_path)

    @property
    def trash_dir(self) -> Path:
        value = self.config["capture"]["trash_dir"].strip()
        return config.resolve_folder(value, self.config_path) if value else self.save_dir / "trash"

    @property
    def vault(self) -> Path | None:
        path = self.config["vault"]["path"].strip()
        return config.resolve_folder(path, self.config_path) if path else None

    def _restore_sash(self, event: tk.Event) -> None:
        # Once, on the first resize with a real width: as the window appears.
        # Any earlier and the panes are 1px wide, so the sash lands at 0.
        if self.sash_to_restore is not None and event.width > 1:
            self.panes.sashpos(0, self.sash_to_restore)
            self.sash_to_restore = None

    def close(self) -> None:
        """Offer to save unsaved edits, write out any half-finished key log lines, remember the window, close."""
        if self.dirty and self.memo is not None:
            answer = messagebox.askyesnocancel("voice2memos", "Save the note before closing?", parent=self.root)
            if answer is None:
                return
            if answer:
                self.save()
                if self.dirty:
                    return  # the save didn't happen; stay open
        self.keylog.flush_all()
        self.hotkeys.stop()
        self.player.close()
        if self.recording and self.recorder is not None:
            self.recorder.stop()
        self._remember_window()
        self.root.after_cancel(self.drain_timer)
        self.root.after_cancel(self.check_timer)
        self.root.destroy()

    def _remember_window(self) -> None:
        window = {("window", "volume"): str(self.player.volume_percent)}
        if self.root.winfo_ismapped():  # a hidden window has no size worth keeping
            window[("window", "sash")] = str(self.panes.sashpos(0))
            if self.root.state() != "zoomed":  # keep the normal size, not the maximized one
                window[("window", "geometry")] = f"{self.root.winfo_width()}x{self.root.winfo_height()}"
        if all(self.config[section][key] == value for (section, key), value in window.items()):
            return
        try:
            self.config, _ = config.update_file(self.config_path, window)
        except (OSError, UnicodeDecodeError, configparser.Error) as e:
            log.warning("window size not saved: %s", config.describe_error(e))

    def _make_fonts_and_styles(self) -> None:
        # Held on self: tkinter deletes a named font when its Font object is collected.
        base = tkfont.nametofont("TkDefaultFont")
        self.title_font = base.copy()
        self.title_font.configure(size=14, weight="bold")
        self.icon_font = base.copy()
        self.icon_font.configure(size=14)
        self.toggle_font = base.copy()
        self.toggle_font.configure(size=11)
        self.link_font = base.copy()
        self.link_font.configure(underline=True)
        self.pane_font = tkfont.nametofont("TkTextFont").copy()
        self.pane_font.configure(size=11)
        self.pane_title_font = base.copy()
        self.pane_title_font.configure(weight="bold")

        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=self.title_font)
        style.configure("PaneTitle.TLabel", font=self.pane_title_font)
        style.configure("Icon.TButton", font=self.icon_font, width=3, padding=2)
        style.configure("Icon.TLabel", font=self.icon_font)
        style.configure("Toggle.TButton", font=self.toggle_font, width=2, padding=0)

    def _clicked(self, name: str, handler: Callable[[], None]) -> Callable[[], None]:
        """A button command that logs the click first, so the log shows it even when nothing follows."""

        def command() -> None:
            log.info("click: %s", name)
            handler()

        return command

    def _build_banner(self) -> None:
        banner = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        banner.pack(side="top", fill="x")
        self.banner = banner

        top = ttk.Frame(banner)
        top.pack(fill="x")
        ttk.Label(top, text="voice2memos", style="Title.TLabel").pack(side="left")
        icons = ttk.Frame(top)
        icons.pack(side="right")
        for name, tip in BANNER:
            command = self._clicked(name, getattr(self, name))
            button = ttk.Button(icons, text=ICONS[name], style="Icon.TButton", command=command)
            button.pack(side="left", padx=2)
            self.buttons[name] = button
            self.tooltips[name] = Tooltip(button, tip)
        # Packed last, so it takes whatever width the title and icons leave. Its
        # buttons log what they did themselves, with the position and the memo.
        discard = self._clicked("discard", self.discard)
        self.player = PlayerBar(top, ICONS, discard, self.set_status, "Icon")
        self.player.frame.pack(side="left", fill="x", expand=True, padx=(16, 12))
        self.tooltips["play"] = Tooltip(self.player.play_button, "Play / pause")
        self.tooltips["volume"] = Tooltip(self.player.volume_label, "Volume")
        self.tooltips["discard"] = Tooltip(self.player.discard_button, "Discard this recording")

        self.save_menu = tk.Menu(self.root, tearoff=False)
        self.save_menu.add_command(label="Save as…", command=self._clicked("save_as", self.save_as))
        self.buttons["save"].bind("<Button-3>", lambda e: self.save_menu.tk_popup(e.x_root, e.y_root))

        status_row = ttk.Frame(banner)
        status_row.pack(fill="x", pady=(6, 0))
        self.status_label = ttk.Label(status_row, cursor="hand2")
        self.status_label.pack(side="left")
        self.status_label.bind("<Button-1>", self.show_status_history)
        self.retry_label = ttk.Label(status_row, text="Retry", font=self.link_font, cursor="hand2")
        self.retry_label.bind("<Button-1>", self._retry)

    def _build_panes(self) -> None:
        panes = ttk.PanedWindow(self.root, orient="horizontal")
        panes.pack(side="top", fill="both", expand=True, padx=10, pady=(6, 10))
        self.left = make_pane(panes, self.pane_font, "Transcript", ICONS["locked"], self.toggle_lock)
        self.right = make_pane(panes, self.pane_font, "Note", ICONS["paint"], self.toggle_paint)
        panes.add(self.left.frame, weight=1)
        panes.add(self.right.frame, weight=1)
        self.panes = panes
        self.left.text.configure(state="disabled")
        self.left_background = self.left.text.cget("background")
        self.tooltips["lock"] = Tooltip(self.left.toggle, "Unlock to edit the Transcript")
        self.tooltips["paint"] = Tooltip(self.right.toggle, "Paint mode: click words to link them")
        # Shown only while the Transcript holds edits the Note hasn't been rebuilt from.
        self.commit_button = ttk.Button(
            self.left.header, text=ICONS["commit"], style="Toggle.TButton",
            command=self._clicked("commit", self.commit_transcript),
        )
        self.tooltips["commit"] = Tooltip(self.commit_button, "Clean up the Note from this Transcript, as if just transcribed")

    # --- status line -----------------------------------------------------

    def set_status(self, text: str, level: str = "info", retry: Callable[[], None] | None = None) -> None:
        """The one way to show a message: colors it by level, keeps it in the history, and logs it."""
        self.status_history.append((datetime.now(), level, text))
        log.log(STATUS_LOG_LEVELS[level], "status: %s", text)
        self.status_label.configure(text=text, foreground=STATUS_COLORS[level])
        self.retry_callback = retry
        if retry is None:
            self.retry_label.pack_forget()
        else:
            self.retry_label.configure(foreground=STATUS_COLORS[level])
            self.retry_label.pack(side="left", padx=(8, 0))

    def _retry(self, _event: tk.Event | None = None) -> None:
        if self.retry_callback is not None:
            log.info("click: retry")
            self.retry_callback()

    def report(self, text: str, level: str = "error") -> None:
        """set_status from any thread: the hotkey thread and the excepthooks come through here."""
        self.post(self.set_status, text, level)

    def _callback_exception(self, exc_type, exc, tb) -> None:
        log.error("uncaught exception in a tk callback", exc_info=(exc_type, exc, tb))
        self.set_status(logsetup.UNEXPECTED, "error")

    def show_status_history(self, _event: tk.Event | None = None) -> None:
        """A small window listing the last ten status messages with their times."""
        if self.history_window is not None:
            self.history_window.destroy()
        window = tk.Toplevel(self.root)
        window.title("Recent messages")
        window.transient(self.root)
        window.resizable(False, False)
        body = ttk.Frame(window, padding=10)
        body.pack(fill="both", expand=True)
        for when, level, text in self.status_history:
            ttk.Label(body, text=f"{when:%H:%M:%S}  {text}", foreground=STATUS_COLORS[level]).pack(anchor="w")
        x = self.status_label.winfo_rootx()
        y = self.status_label.winfo_rooty() + self.status_label.winfo_height() + 4
        window.geometry(f"+{x}+{y}")
        window.bind("<Escape>", lambda _e: window.destroy())
        window.focus_set()
        self.history_window = window

    # --- pane toggles ----------------------------------------------------

    def toggle_lock(self) -> None:
        """Flip the transcript pane between read-only and editable."""
        self.left_locked = not self.left_locked
        if self.left_locked:
            self.left.text.configure(state="disabled", background=self.left_background)
            self.left.toggle.configure(text=ICONS["locked"])
            self.tooltips["lock"].text = "Unlock to edit the Transcript"
        else:
            self.left.text.configure(state="normal", background=UNLOCKED_TINT)
            self.left.toggle.configure(text=ICONS["unlocked"])
            self.tooltips["lock"].text = "Lock the Transcript"
        log.info("transcript pane %s", "locked" if self.left_locked else "unlocked")

    def toggle_paint(self) -> None:
        """Flip the note pane between paint and type. The click-to-link bindings arrive in step 12."""
        self.paint_mode = not self.paint_mode
        if self.paint_mode:
            self.right.toggle.configure(text=ICONS["paint"])
            self.tooltips["paint"].text = "Paint mode: click words to link them"
        else:
            self.right.toggle.configure(text=ICONS["type"])
            self.tooltips["paint"].text = "Type mode: edit the note as text"
        log.info("note pane in %s mode", "paint" if self.paint_mode else "type")

    # --- settings --------------------------------------------------------

    def open_settings(self) -> None:
        if self.settings_dialog is not None and self.settings_dialog.window.winfo_exists():
            self.settings_dialog.window.lift()
            return
        self.rescan_audio()  # so a headset plugged in since launch is on the list; skipped while recording
        self.settings_dialog = SettingsDialog(
            self.root, config.values(self.config), self.keylog, self.apply_settings, self.config_path,
            on_rescan=self.rescan_audio,
        )

    def rescan_audio(self) -> str:
        """Have PortAudio look for devices again. Returns why it couldn't, or blank.

        PortAudio can't rescan with a stream open, so playback pauses first,
        and nothing happens while recording.
        """
        if self.recording:
            return "Stop recording to rescan"
        self.player.pause()
        try:
            capture.rescan_devices()
        except Exception as e:  # no sounddevice, or PortAudio itself failing
            log.warning("couldn't rescan audio devices: %s: %s", type(e).__name__, e)
            return "Couldn't rescan audio devices, see log"
        log.info("audio devices rescanned")
        return ""

    def apply_settings(self, values: dict[Field, str]) -> None:
        """Write the dialog's values into the .cfg, log what changed (secrets by name only), apply."""
        try:
            new_config, warnings = config.update_file(self.config_path, values)
        except (OSError, UnicodeDecodeError, configparser.Error) as e:
            message = f"Settings not saved: {config.describe_error(e)}"
            log.warning("%s", message)
            self.set_status(message, "error")
            return
        old = config.values(self.config)
        changes = describe_changes(old, values)
        for line in changes:
            log.info("setting %s", line)
        self.config = new_config
        self.apply({field for field, value in values.items() if old.get(field) != value})
        self.set_status("Settings saved" if changes else "Settings saved, nothing changed")
        for warning in warnings:
            self.set_status(warning, "warning")

    def apply(self, changed: set[Field]) -> None:
        """Make the running app match ``self.config``, no restart.

        Each memo builds its backends from ``self.config`` when it runs, so
        model and provider changes need nothing here. What does: anything held
        between memos. Today that is the log level, the hotkey listener and
        the player's output device; the cached Whisper model joins in step 7.
        The microphone is picked up by the next recording.
        """
        if ("logging", "level") in changed:
            logsetup.set_level(self.config["logging"]["level"])
            log.info("log level now %s", self.config["logging"]["level"].upper())
        if OUTPUT_DEVICE in changed:
            self.player.set_output_device(self.config["playback"]["output_device"])
        if any(section == "whisper" for section, _ in changed):
            self.warm_up_whisper()  # a model already loaded comes straight from the cache
        if changed & HOTKEY_FIELDS:
            self.hotkeys.bind(self.config["capture"]["hotkey"], self.config["capture"]["note_hotkey"])

    # --- capture -----------------------------------------------------------

    def record(self) -> None:
        """The button: one click starts, the next stops and runs the memo."""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def hotkey_pressed(self) -> None:
        if isinstance(self.root.focus_get(), tk.Text):
            log.info("hotkey: push-to-talk ignored, typing in a pane")
            return
        log.info("hotkey: push-to-talk down")
        if not self.recording:
            self.start_recording()

    def hotkey_released(self) -> None:
        if self.recording:
            log.info("hotkey: push-to-talk up")
            self.stop_recording()

    def push_to_note(self) -> None:
        """Step 14 fills this in; for now the chord raises the window."""
        self.root.deiconify()
        self.root.lift()
        log.info("stub: push_to_note")

    def start_recording(self) -> None:
        if self._cannot_capture():
            self.set_status(self._cannot_capture(), "warning")
            return
        self.recorder = capture.make_recorder(self.config, config.APP_DIR)
        try:
            self.recorder.start()
        except VoiceMemoError as e:
            log.warning("%s: %s", e.message, e.detail)
            self.set_status(e.message, "error")
            return
        self.recording = True
        # Silent while the mic is open, or the recording would hear the last one.
        self.player.pause()
        self.player.set_enabled(False)
        self.buttons["record"].configure(text=ICONS["stop"])
        self.tooltips["record"].text = "Stop"
        if self.recorder.note:
            self.set_status(f"Recording… {self.recorder.note}", "warning")
        else:
            self.set_status("Recording…")

    def stop_recording(self) -> None:
        self.recording = False
        self.buttons["record"].configure(text=ICONS["record"])
        self.tooltips["record"].text = "Record"
        self.player.set_enabled(True)
        frames, samplerate = self.recorder.stop()
        try:
            memo = Memo.create(self.save_dir / capture.new_stem(self.save_dir))
            capture.write_wav(memo.wav, frames, samplerate)
        except OSError as e:
            log.warning("couldn't write the recording: %s", e)
            self.set_status(f"Could not save the recording: {e.strerror or e}", "error")
            return
        log.info("%s: wav written, %.1fs", memo.stem, len(frames) / 2 / max(samplerate, 1))
        self.run_pipeline(memo)

    def open_file(self) -> None:
        """Pick a wav, mp3 or m4a; it becomes a memo of its own and runs like a recording.

        A 16-bit WAV is copied in; anything else is decoded to one on the
        worker. Either way the original is only ever read.
        """
        if self._cannot_capture():
            self.set_status(self._cannot_capture(), "warning")
            return
        initial = self.config["capture"]["open_dir"] or str(Path.home())
        picked = filedialog.askopenfilename(parent=self.root, title="Open audio file", filetypes=AUDIO_TYPES, initialdir=initial)
        if not picked:
            return
        source = Path(picked)
        self._remember(("capture", "open_dir"), str(source.parent))
        try:
            memo = Memo.create(self.save_dir / capture.new_stem(self.save_dir), source=source)
            if capture.is_pcm16_wav(source):
                shutil.copy2(source, memo.wav)
        except OSError as e:
            log.warning("couldn't copy %s: %s", source, e)
            self.set_status(f"Could not open: {e.strerror or e}", "error")
            return
        log.info("%s: opened %s", memo.stem, source)
        self.run_pipeline(memo)

    # --- pipeline ----------------------------------------------------------

    def run_pipeline(self, memo: Memo, start: str = "transcribe") -> None:
        """Hand the memo to the worker thread from ``start`` onward."""
        if memo is not self.memo or self.player.playback.path != memo.wav:
            if memo.wav.exists():
                self.player.load(memo.wav, discardable=memo.is_recording)
            else:
                self.player.clear()  # an opened file still decoding; it loads when that's done
        self.memo = memo
        self.set_busy(True)
        self.run_issue = False  # set when a stage fails or warns, so "done" doesn't say Ready over it
        if start == "transcribe":
            self._fill("", "")
            self.dirty = False
            self._update_title()
        self.set_status("Transcribing…" if start == "transcribe" else f"Running {start}…")
        cfg = config.new_parser()
        cfg.read_dict({s: dict(self.config[s]) for s in self.config.sections()})  # the worker gets its own copy
        thread = threading.Thread(target=self._pipeline, args=(memo, cfg, start), name=f"memo-{memo.stem}", daemon=True)
        thread.start()

    def _pipeline(self, memo: Memo, cfg, start: str) -> None:
        """Worker thread. Never touches a widget; never lets an exception escape."""
        stem = memo.stem
        post = lambda result: self.post(self._on_stage, memo, result)  # noqa: E731
        stages = STAGES[STAGES.index(start):]
        try:
            if not memo.wav.exists() and memo.source is not None:
                try:
                    with logsetup.timed(log, "decode", stem):
                        capture.decode_to_wav(memo.source, memo.wav)
                    post(StageResult("decode", True))
                except VoiceMemoError as e:
                    log.warning("%s: %s: %s", stem, e.message, e.detail)
                    post(StageResult("decode", False, e.message))
                    return
            if "transcribe" in stages:
                try:
                    transcriber = transcribe.make_transcriber(cfg, config.APP_DIR)
                    memo.segments = transcriber.transcribe(memo.wav, stem)
                    memo.save_segments()
                    post(StageResult("transcribe", True, transcriber.note))  # e.g. "GPU unavailable, using CPU"
                except VoiceMemoError as e:
                    log.warning("%s: %s: %s", stem, e.message, e.detail)
                    post(StageResult("transcribe", False, e.message))
                    return  # nothing downstream without a transcript
            text = memo.raw_text
            if "cleanup" in stages:
                prompt = cfg["cleanup"]["prompt"].strip() or cleanup.load_prompt("rewrite")
                try:
                    provider = cleanup.make_provider(cfg, "cleanup", config.APP_DIR)
                    warnings = cleanup.rewrite(provider, memo.segments, prompt, float(cfg["cleanup"]["timeout"]), stem)
                    memo.save_cleaned()
                    post(StageResult("cleanup", True, "; ".join(warnings)))
                except VoiceMemoError as e:
                    log.warning("%s: %s: %s", stem, e.message, e.detail)
                    memo.save_cleaned()
                    post(StageResult("cleanup", False, e.message))
                text = memo.cleaned_text
            timeout = float(cfg["entities"]["timeout"])
            if "entities" in stages:
                try:
                    provider = cleanup.make_provider(cfg, "entities", config.APP_DIR)
                    memo.entities = cleanup.extract_entities(provider, text, timeout, stem)
                    memo.save_entities()
                    post(StageResult("entities", True))
                except VoiceMemoError as e:
                    log.warning("%s: %s: %s", stem, e.message, e.detail)
                    post(StageResult("entities", False, e.message))
            if "tags" in stages:
                try:
                    provider = cleanup.make_provider(cfg, "entities", config.APP_DIR)
                    memo.tags = cleanup.choose_tags(provider, text, cfg["tags"]["pool"].split(), timeout, stem)
                    memo.save_tags()
                    post(StageResult("tags", True))
                except VoiceMemoError as e:
                    log.warning("%s: %s: %s", stem, e.message, e.detail)
                    post(StageResult("tags", False, e.message))
        except Exception:
            log.exception("%s: pipeline crashed", stem)
            post(StageResult("pipeline", False, logsetup.UNEXPECTED))
        finally:
            post(StageResult("done", True))

    def _on_stage(self, memo: Memo, result: StageResult) -> None:
        """UI thread: show what a stage produced, or what went wrong and how to retry it."""
        if memo is not self.memo:
            return  # discarded or replaced while the worker ran
        if result.stage == "done":
            self.set_busy(False)
            if not self.run_issue:
                self.set_status("Ready")
            return
        if result.stage == "decode" and result.ok:
            self.player.load(memo.wav, discardable=memo.is_recording)
        elif result.stage == "transcribe" and result.ok:
            self._fill(memo.raw_text, memo.cleaned_text)
        elif result.stage == "cleanup":
            self._fill(None, memo.cleaned_text)
        if result.ok:
            log.info("%s: %s shown", memo.stem, result.stage)
            if result.message:
                self.run_issue = True
                self.set_status(result.message, "warning")
            return
        self.run_issue = True
        # A failed decode retries from the top, which decodes again since there's still no WAV.
        rerun = "transcribe" if result.stage == "decode" else result.stage
        retry = (lambda: self.run_pipeline(memo, rerun)) if rerun in STAGES else None
        self.set_status(result.message, "error", retry=retry)

    def _fill(self, transcript: str | None, note_text: str | None) -> None:
        """The program writing the panes. ``None`` leaves that pane as it is."""
        self.filling = True
        try:
            if transcript is not None:
                replace_text(self.left.text, transcript)
                self.left.text.edit_modified(False)
                # Whatever the program puts in the Transcript, the Note is (being) built from.
                self.note_built_from = transcript
            if note_text is not None:
                replace_text(self.right.text, note_text)
                self.right.text.edit_modified(False)
                self.note_hand_edited = self.mirror_declined = False
                # A saved Note the program just rebuilt no longer matches the copy in the vault.
                self.dirty = self.memo is not None and self.memo.saved_to is not None
                self._update_title()
        finally:
            self.filling = False
        self._update_commit()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._update_capture_buttons()
        self.player.set_busy(busy)
        self.commit_button.state(["disabled"] if busy else ["!disabled"])

    def _update_capture_buttons(self) -> None:
        """Record and Open are off while a memo runs and while the Whisper model loads."""
        off = self.busy or self.whisper_loading
        for name in BUSY_BUTTONS:
            self.buttons[name].state(["disabled"] if off else ["!disabled"])

    def _cannot_capture(self) -> str:
        """Why a recording or an open can't start right now, or blank."""
        if self.busy:
            return "Busy with the last memo"
        if self.whisper_loading:
            return "Whisper model still loading"
        return ""

    # --- pane edits ------------------------------------------------------------

    def _note_modified(self, _event: tk.Event | None = None) -> None:
        # The event is delivered later than the edit, so a program write that
        # already cleared the flag shows up here with it False: not an edit.
        if not self.right.text.edit_modified():
            return
        self.right.text.edit_modified(False)  # or tk fires this exactly once
        if self.filling or self.memo is None:
            return
        self.note_hand_edited = True
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def _transcript_modified(self, _event: tk.Event | None = None) -> None:
        if not self.left.text.edit_modified():
            return
        self.left.text.edit_modified(False)
        self._update_commit()
        self._mirror_transcript()

    def transcript_edited(self) -> bool:
        """The Transcript pane holds text the Note wasn't built from."""
        text = self.left.text.get("1.0", "end-1c").strip()
        return bool(text) and text != self.note_built_from.strip()

    def _mirror_transcript(self) -> None:
        """The Note follows the Transcript as it's typed, as it would look straight after transcription.

        While the Transcript differs from what the Note was built from, the
        Note shows it as written; typed back to that, the Note gets its
        cleaned text back. Edits typed into the Note are only overwritten
        after asking, once.
        """
        if self.note_hand_edited:
            if self.mirror_declined:
                return
            if not messagebox.askyesno(
                "voice2memos",
                "The Note has edits of your own. Let it follow the Transcript, replacing them?",
                parent=self.root,
            ):
                self.mirror_declined = True
                log.info("note kept as edited; it stops following the transcript")
                return
            log.info("note's own edits replaced; it follows the transcript")
        if self.transcript_edited():
            text = self.left.text.get("1.0", "end-1c")
        else:
            text = self.memo.cleaned_text if self.memo is not None else ""
        if text == self.right.text.get("1.0", "end-1c"):
            return
        top = self.right.text.yview()[0]  # typing far down the Transcript shouldn't scroll the Note to the top
        self._fill(None, text)
        self.right.text.yview_moveto(top)

    def _update_commit(self) -> None:
        """Show + while there are Transcript edits to clean up; hide it once there aren't."""
        if self.transcript_edited():
            if not self.commit_button.winfo_manager():
                self.commit_button.pack(side="right")
                log.info("transcript edited; the note follows it, + cleans it up")
        elif self.commit_button.winfo_manager():
            self.commit_button.pack_forget()

    def commit_transcript(self) -> None:
        """The +: the rest of the pipeline, as if the edited Transcript had just been transcribed.

        The Transcript becomes the memo's segments, and cleanup, entities and
        tags run on them, filling the Note. Saving it stays the Save button's
        job, as for any memo. The Note is replaced, so if it still holds edits
        of its own (the user declined to let it follow), it asks first. With
        no memo yet, the typed text starts one, without audio.
        """
        if self.busy or not self.transcript_edited():
            return
        if self.note_hand_edited and not messagebox.askyesno(
            "voice2memos",
            "The Note has edits of your own. Replace it with a fresh cleanup of the Transcript?",
            parent=self.root,
        ):
            log.info("transcript not sent; the note's own edits kept")
            return
        text = self.left.text.get("1.0", "end-1c")
        memo = self.memo
        try:
            if memo is None:
                memo = Memo.create(self.save_dir / capture.new_stem(self.save_dir))
            memo.keep_original()
            memo.segments = transcribe.apply_edit(memo.segments, text)
            memo.save_segments()
        except OSError as e:
            log.warning("couldn't save the edited transcript: %s", e)
            self.set_status(f"Could not save the transcript: {e.strerror or e}", "error")
            return
        log.info("%s: transcript edited by hand, %d segment(s); cleaning it up", memo.stem, len(memo.segments))
        self.run_pipeline(memo, "cleanup")
        self._fill(memo.raw_text, memo.raw_text)  # tidied to a paragraph per segment, which hides the +

    def _update_title(self) -> None:
        self.root.title(f"voice2memos {__version__}{' *' if self.dirty else ''}")

    # --- save, discard, vault ---------------------------------------------------

    def save(self) -> None:
        """Three behaviors: first save asks where; a dirty note writes again; a clean one opens in Obsidian."""
        if self.memo is None:
            self.set_status("Nothing to save", "warning")
            return
        if self.memo.saved_to is None:
            self.save_as()
        elif self.dirty:
            self._write_note(self.memo.saved_to)
        else:
            try:
                self.set_status(note.open_note(self.memo.saved_to, self.vault is not None and note.is_registered(self.vault)))
            except VoiceMemoError as e:
                log.warning("%s: %s", e.message, e.detail)
                self.set_status(e.message, "error")

    def save_as(self) -> None:
        if self.memo is None:
            self.set_status("Nothing to save", "warning")
            return
        initial_dir = self.vault if self.vault is not None and self.vault.exists() else Path.home()
        picked = filedialog.asksaveasfilename(
            parent=self.root, title="Save note", initialdir=initial_dir, initialfile=f"{self.memo.stem}.md",
            defaultextension=".md", filetypes=[("Markdown", "*.md")],
        )
        if picked:
            self._write_note(Path(picked))

    def _write_note(self, path: Path) -> None:
        memo = self.memo
        text = note.render(memo, self.right.text.get("1.0", "end-1c"))
        try:
            note.write_note(path, text)
            (memo.folder / note.NOTE_FILE).write_text(text, encoding="utf-8")
            if self.vault is not None:
                note.copy_audio(memo, self.vault, self.config["vault"]["attachments_subdir"])
        except VoiceMemoError as e:
            log.warning("%s: %s", e.message, e.detail)
            self.set_status(e.message, "error")
            return  # dirty stays set
        memo.saved_to = path
        memo.save_state()
        self.dirty = False
        self._update_title()
        log.info("%s: saved to %s", memo.stem, path)
        self.set_status(f"Saved {path.name}")

    def discard(self) -> None:
        """The player's 🗑: the recording's working folder, and a saved note with its
        audio copy, move to the trash; the panes and the player clear."""
        if self.memo is None:
            self.set_status("Nothing to discard", "warning")
            return
        memo, self.memo = self.memo, None
        self.player.clear()
        try:
            note.discard(memo, self.trash_dir, self.vault, self.config["vault"]["attachments_subdir"])
        except VoiceMemoError as e:
            log.warning("%s: %s", e.message, e.detail)
            self.memo = memo
            self.player.load(memo.wav, discardable=memo.is_recording)
            self.set_status(e.message, "error")
            return
        self._fill("", "")
        self.set_status(f"Discarded {memo.stem}")

    def choose_vault(self) -> None:
        picked = filedialog.askdirectory(parent=self.root, title="Choose your Obsidian vault", mustexist=True)
        if not picked:
            return
        vault = Path(picked)
        if not note.is_vault(vault):
            messagebox.showwarning("voice2memos", "That folder has no .obsidian inside it, so it isn't an Obsidian vault.", parent=self.root)
            self.set_status("Not an Obsidian vault", "warning")
            return
        self._remember(("vault", "path"), str(vault))  # right away, so it survives a crash
        if not note.is_registered(vault):
            self._ask_to_register(vault)
        else:
            self.set_status(f"Vault: {vault.name}")

    def _ask_to_register(self, vault: Path) -> None:
        self.set_status("Open this vault in Obsidian once", "warning")
        if messagebox.askyesno(
            "voice2memos",
            "Obsidian hasn't opened this folder as a vault yet, so notes will open with the system handler instead.\n\n"
            "Launch Obsidian now? Open this folder as a vault there, then come back.",
            parent=self.root,
        ):
            webbrowser.open("obsidian://")

    def _remember(self, field: Field, value: str) -> None:
        if self.config[field[0]][field[1]] == value:
            return
        try:
            self.config, _ = config.update_file(self.config_path, {field: value})
        except (OSError, UnicodeDecodeError, configparser.Error) as e:
            log.warning("%s not saved to the config: %s", ".".join(field), config.describe_error(e))

    def open_log(self) -> None:
        log.info("stub: open_log")


def main() -> None:
    log_path = logsetup.setup_logging(config.LOG_DIR)
    log.info("voice2memos %s starting, log at %s", __version__, log_path)
    set_dpi_awareness()
    root = tk.Tk()
    app = App(root, config.CONFIG_PATH)
    logsetup.set_level(app.config["logging"]["level"])
    logsetup.install_excepthooks(app.report)
    root.mainloop()
