"""Root window, banner, panes, worker wiring.

Build-order step 2: every control is present and clickable. The banner buttons
are stubs that log their own name until step 6 makes them do what they say,
except Settings, which opens the real dialog. The window's size and the split
between the panes come from the config and go back to it on close.
"""

from __future__ import annotations

import configparser
import logging
import tkinter as tk
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

from voice2memos import __version__, config
from voice2memos.config import Field
from voice2memos.ui.panes import KeyLog, make_pane
from voice2memos.ui.platform import set_dpi_awareness
from voice2memos.ui.settings import SettingsDialog, describe_changes

log = logging.getLogger(__name__)

# tkinter ships no usable icons, so Unicode symbols stand in for the PNGs the
# design calls for: no files, no license, drawn by fonts every OS already has.
# Moving to PNGs later only touches this table and the two button builders.
# Each toggle shows its pane's current state: the lock is read-only, the
# keyboard is typing, the brush is paint mode. Not 🔓, which Windows'
# monochrome emoji font draws exactly like 🔒.
ICONS = {
    "open_file": "📂",
    "save": "💾",
    "record": "⏺",
    "discard": "🗑",
    "choose_vault": "💎",
    "open_settings": "⚙",
    "open_log": "☰",
    "locked": "🔒",
    "unlocked": "⌨",
    "paint": "🖌",
    "type": "⌨",
}

# Banner buttons left to right: handler name and tooltip.
BANNER = [
    ("open_file", "Open audio file"),
    ("save", "Save (right-click for Save as)"),
    ("record", "Record"),
    ("discard", "Discard"),
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

        root.title(f"voice2memos {__version__}")
        root.geometry(self.config["window"]["geometry"])
        root.minsize(640, 400)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._make_fonts_and_styles()
        self._build_banner()
        self._build_panes()
        self.keylog = KeyLog(root)
        self.keylog.watch(self.left.text, "transcript")
        self.keylog.watch(self.right.text, "note")
        sash = self.config["window"]["sash"]
        self.sash_to_restore = int(sash) if sash else None
        self.panes.bind("<Configure>", self._restore_sash, add="+")
        for warning in config_warnings:
            self.set_status(warning, "warning")
        if not config_warnings:
            self.set_status("Ready")

    def _restore_sash(self, event: tk.Event) -> None:
        # Once, on the first resize with a real width: as the window appears.
        # Any earlier and the panes are 1px wide, so the sash lands at 0.
        if self.sash_to_restore is not None and event.width > 1:
            self.panes.sashpos(0, self.sash_to_restore)
            self.sash_to_restore = None

    def close(self) -> None:
        """Write out any half-finished key log lines, remember the window, then close.

        Step 9 adds the unsaved-changes prompt.
        """
        self.keylog.flush_all()
        self._remember_window()
        self.root.destroy()

    def _remember_window(self) -> None:
        if not self.root.winfo_ismapped():
            return  # a hidden window has no size worth keeping
        window = {("window", "sash"): str(self.panes.sashpos(0))}
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

        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=self.title_font)
        style.configure("Icon.TButton", font=self.icon_font, width=3, padding=2)
        style.configure("Toggle.TButton", font=self.toggle_font, width=2, padding=0)

    def _build_banner(self) -> None:
        banner = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        banner.pack(side="top", fill="x")

        top = ttk.Frame(banner)
        top.pack(fill="x")
        ttk.Label(top, text="voice2memos", style="Title.TLabel").pack(side="left")
        icons = ttk.Frame(top)
        icons.pack(side="right")
        for name, tip in BANNER:
            button = ttk.Button(icons, text=ICONS[name], style="Icon.TButton", command=getattr(self, name))
            button.pack(side="left", padx=2)
            self.buttons[name] = button
            self.tooltips[name] = Tooltip(button, tip)

        self.save_menu = tk.Menu(self.root, tearoff=False)
        self.save_menu.add_command(label="Save as…", command=self.save_as)
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
        self.left = make_pane(panes, self.pane_font, ICONS["locked"], self.toggle_lock)
        self.right = make_pane(panes, self.pane_font, ICONS["paint"], self.toggle_paint)
        panes.add(self.left.frame, weight=1)
        panes.add(self.right.frame, weight=1)
        self.panes = panes
        self.left.text.configure(state="disabled")
        self.left_background = self.left.text.cget("background")
        self.tooltips["lock"] = Tooltip(self.left.toggle, "Unlock to edit the transcript")
        self.tooltips["paint"] = Tooltip(self.right.toggle, "Paint mode: click words to link them")

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
            self.retry_callback()

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
            self.tooltips["lock"].text = "Unlock to edit the transcript"
        else:
            self.left.text.configure(state="normal", background=UNLOCKED_TINT)
            self.left.toggle.configure(text=ICONS["unlocked"])
            self.tooltips["lock"].text = "Lock the transcript"
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
        self.settings_dialog = SettingsDialog(self.root, config.values(self.config), self.keylog, self.apply_settings)

    def apply_settings(self, values: dict[Field, str]) -> None:
        """Write the dialog's values into the .cfg and log what changed, secrets by name only.

        Step 4 makes the running app pick them up: hotkeys, providers, and so on.
        """
        try:
            new_config, warnings = config.update_file(self.config_path, values)
        except (OSError, UnicodeDecodeError, configparser.Error) as e:
            message = f"Settings not saved: {config.describe_error(e)}"
            log.warning("%s", message)
            self.set_status(message, "error")
            return
        changes = describe_changes(config.values(self.config), values)
        for line in changes:
            log.info("setting %s", line)
        self.config = new_config
        self.set_status("Settings saved" if changes else "Settings saved, nothing changed")
        for warning in warnings:
            self.set_status(warning, "warning")

    # --- banner stubs, wired up in step 6 --------------------------------

    def open_file(self) -> None:
        log.info("stub: open_file")

    def save(self) -> None:
        log.info("stub: save")

    def save_as(self) -> None:
        log.info("stub: save_as")

    def record(self) -> None:
        log.info("stub: record")

    def discard(self) -> None:
        log.info("stub: discard")

    def choose_vault(self) -> None:
        log.info("stub: choose_vault")

    def open_log(self) -> None:
        log.info("stub: open_log")


def main() -> None:
    # Console logging until logsetup.py lands in step 13.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_dpi_awareness()
    root = tk.Tk()
    app = App(root, config.CONFIG_PATH)
    logging.getLogger().setLevel(app.config["logging"]["level"].upper())
    root.mainloop()
