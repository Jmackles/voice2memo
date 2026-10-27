"""The Frame+Text+Scrollbar factory, lock toggle, paint mode, and key log.

Remember: ``edit_modified(False)`` must be called inside the Modified handler
every time, or tkinter fires it exactly once and never again.
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import font as tkfont
from tkinter import ttk

keylog = logging.getLogger("voice2memos.keys")

# Keys that type no printable character, as they read in the key log.
KEY_SYMBOLS = {
    "BackSpace": "⌫",
    "Delete": "⌦",
    "Return": "⏎",
    "KP_Enter": "⏎",
    "Tab": "⇥",
    "Left": "←",
    "Right": "→",
    "Up": "↑",
    "Down": "↓",
    "Home": "⇱",
    "End": "⇲",
    "Prior": "⇞",
    "Next": "⇟",
    "Escape": "⎋",
}
MODIFIER_KEYS = {
    "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Meta_L", "Meta_R",
    "Super_L", "Super_R", "Win_L", "Win_R", "Caps_Lock", "Num_Lock", "ISO_Level3_Shift", "App",
}
CONTROL_MASK = 0x4


@dataclass
class Pane:
    frame: ttk.Frame
    text: tk.Text
    scrollbar: ttk.Scrollbar
    toggle: ttk.Button


def make_pane(parent: tk.Misc, font: tkfont.Font, toggle_text: str, toggle_command: Callable[[], None]) -> Pane:
    """A word-wrapped Text with its own scrollbar, and a toggle button under the scroll arrow."""
    frame = ttk.Frame(parent)
    # A 1x1 requested size lets the PanedWindow weights decide the split
    # instead of Text's 80x24 default.
    text = tk.Text(frame, wrap="word", width=1, height=1, undo=True, font=font, padx=8, pady=6)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scrollbar.set)
    toggle = ttk.Button(frame, text=toggle_text, command=toggle_command, style="Toggle.TButton")

    text.grid(row=0, column=0, rowspan=2, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    toggle.grid(row=1, column=1)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return Pane(frame, text, scrollbar, toggle)


def replace_text(text: tk.Text, content: str) -> None:
    """Swap a Text's contents even while it is disabled, then put its state back."""
    state = text.cget("state")
    text.configure(state="normal")
    text.delete("1.0", "end")
    text.insert("1.0", content)
    text.edit_reset()  # so Ctrl+Z can't undo the program's own write
    text.configure(state=state)


def describe_key(keysym: str, char: str, state: int) -> str:
    """How one keypress reads in the key log; empty for a bare modifier like Shift."""
    if keysym in MODIFIER_KEYS:
        return ""
    ctrl = state & CONTROL_MASK
    if keysym in KEY_SYMBOLS:
        return f"[Ctrl+{KEY_SYMBOLS[keysym]}]" if ctrl else KEY_SYMBOLS[keysym]
    if char and char.isprintable():
        return char  # also covers AltGr characters, which arrive with Control held on Windows
    if ctrl:
        return f"[Ctrl+{keysym.upper() if len(keysym) == 1 else keysym}]"
    return f"[{keysym}]"


@dataclass
class _Burst:
    first: float  # monotonic time of the first key
    last: float  # and of the latest one
    deadline: str  # after id: max_seconds from the first key
    idle: str  # after id: idle_seconds from the latest key
    keys: str = ""


class KeyLog:
    """Logs what is typed into watched fields as one line per burst, not one line per key.

    A burst ends on Enter, after ``idle_seconds`` with no key, at ``max_chars``,
    ``max_seconds`` after its first key, or when focus leaves the field. A click
    in a field ends every burst and is logged as its own line. Keys pressed while
    a field is disabled change nothing and aren't logged. Never watch a field
    that holds a secret, like an API key: everything here ends up in the log file.
    """

    def __init__(
        self, root: tk.Misc, max_chars: int = 80, max_seconds: float = 30, idle_seconds: float = 3
    ) -> None:
        self.root = root
        self.max_chars = max_chars
        self.max_seconds = max_seconds
        self.idle_seconds = idle_seconds
        self.bursts: dict[str, _Burst] = {}

    def watch(self, widget: tk.Text, name: str) -> None:
        def on_key(event: tk.Event) -> None:
            if str(widget.cget("state")) != "disabled":
                self.record(name, describe_key(event.keysym, event.char, event.state))

        def on_click(event: tk.Event) -> None:
            self.flush_all()  # first, so the log stays in the order things happened
            line, char = widget.index(f"@{event.x},{event.y}").split(".")
            keylog.info("%s: clicked at line %s, char %d", name, line, int(char) + 1)

        widget.bind("<KeyPress>", on_key, add="+")
        widget.bind("<Button-1>", on_click, add="+")
        widget.bind("<FocusOut>", lambda _e: self.flush(name), add="+")

    def record(self, name: str, key: str) -> None:
        if not key:
            return
        now = time.monotonic()
        burst = self.bursts.get(name)
        if burst is None:
            deadline = self.root.after(int(self.max_seconds * 1000), self.flush, name)
            burst = self.bursts[name] = _Burst(now, now, deadline, idle="")
        else:
            self.root.after_cancel(burst.idle)
        burst.idle = self.root.after(int(self.idle_seconds * 1000), self.flush, name)
        burst.last = now
        burst.keys += key
        if key == KEY_SYMBOLS["Return"] or len(burst.keys) >= self.max_chars:
            self.flush(name)

    def flush(self, name: str) -> None:
        burst = self.bursts.pop(name, None)
        if burst is None:
            return
        self.root.after_cancel(burst.deadline)
        self.root.after_cancel(burst.idle)
        # The seconds are how long the typing took, not how long the line waited.
        keylog.info('%s, %.1fs: "%s"', name, burst.last - burst.first, burst.keys)

    def flush_all(self) -> None:
        for name in list(self.bursts):
            self.flush(name)
