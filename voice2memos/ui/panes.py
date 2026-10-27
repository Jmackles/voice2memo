"""The Frame+Text+Scrollbar factory, lock toggle, and paint mode.

Remember: ``edit_modified(False)`` must be called inside the Modified handler
every time, or tkinter fires it exactly once and never again.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import font as tkfont
from tkinter import ttk


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
