"""The footer under the panes: the note's name, the tag chips, and a preview of the note as it would be saved.

The Name field holds the name the note is saved under, the model's until the
user types one; the hint beside it says what file that makes.

The preview is the serializer's output, raw markdown, read-only; it is the
one place the frontmatter, the links and the footer of the note can be seen
before Save. It folds away behind its title, and the app remembers whether
it was open as ``[window] preview``.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont
from tkinter import ttk

from voice2memos.ui.chips import HINT_COLOR, TITLE_WIDTH, ChipRow
from voice2memos.ui.panes import replace_text

PREVIEW_LINES = 9
OPEN, FOLDED = "▾", "▸"
NAME_WIDTH = 44  # characters


class Footer:
    def __init__(
        self,
        parent: tk.Misc,
        chip_font: tkfont.Font,
        preview_font: tkfont.Font,
        on_tag_toggle: Callable[[str, bool], None],
        on_preview_toggle: Callable[[], None],
        on_name_change: Callable[[str], None],
    ) -> None:
        self.frame = ttk.Frame(parent)
        name_row = ttk.Frame(self.frame)
        name_row.pack(fill="x", pady=(0, 6))
        ttk.Label(name_row, text="Name", style="PaneTitle.TLabel", width=TITLE_WIDTH).pack(side="left")
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_row, textvariable=self.name_var, width=NAME_WIDTH, state="disabled")
        self.name_entry.pack(side="left", padx=(8, 0))
        self.name_hint = ttk.Label(name_row, foreground=HINT_COLOR)
        self.name_hint.pack(side="left", padx=(8, 0))
        self.setting_name = False  # the program is filling the field, not the user
        self.name_var.trace_add("write", lambda *_: self.setting_name or on_name_change(self.name_var.get()))
        self.chips = ChipRow(self.frame, chip_font, on_tag_toggle)
        self.chips.frame.pack(fill="x")
        header = ttk.Frame(self.frame)
        header.pack(fill="x", pady=(6, 0))
        self.fold_button = ttk.Button(header, text=OPEN, style="Toggle.TButton", command=on_preview_toggle)
        self.fold_button.pack(side="left")
        self.preview_title = ttk.Label(header, text="Note preview", style="PaneTitle.TLabel", cursor="hand2")
        self.preview_title.pack(side="left", padx=(6, 0))
        self.preview_title.bind("<Button-1>", lambda _e: on_preview_toggle())
        self.body = ttk.Frame(self.frame)
        self.preview = tk.Text(
            self.body, wrap="word", width=1, height=PREVIEW_LINES, font=preview_font, padx=8, pady=6,
            state="disabled", background="#f7f7f7", relief="flat",
        )
        scrollbar = ttk.Scrollbar(self.body, orient="vertical", command=self.preview.yview)
        self.preview.configure(yscrollcommand=scrollbar.set)
        self.preview.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.shown = False
        self.set_shown(True)

    def set_shown(self, shown: bool) -> None:
        if shown == self.shown:
            return
        self.shown = shown
        self.fold_button.configure(text=OPEN if shown else FOLDED)
        if shown:
            self.body.pack(fill="x", pady=(4, 0))
        else:
            self.body.pack_forget()

    def show_preview(self, text: str) -> None:
        if text == self.preview_text():
            return
        top = self.preview.yview()[0]
        replace_text(self.preview, text)
        self.preview.yview_moveto(top)

    def show_name(self, name: str | None) -> None:
        """The program's name for the note; None empties the field and locks it (no memo)."""
        self.setting_name = True
        try:
            self.name_entry.configure(state="normal")
            if self.name_var.get() != (name or ""):
                self.name_var.set(name or "")
            if name is None:
                self.name_entry.configure(state="disabled")
        finally:
            self.setting_name = False

    def show_name_hint(self, text: str) -> None:
        if self.name_hint.cget("text") != text:
            self.name_hint.configure(text=text)

    def preview_text(self) -> str:
        return self.preview.get("1.0", "end-1c")
