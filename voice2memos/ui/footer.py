"""The footer under the panes: the tag chips, and a preview of the note as it would be saved.

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

from voice2memos.ui.chips import ChipRow
from voice2memos.ui.panes import replace_text

PREVIEW_LINES = 9
OPEN, FOLDED = "▾", "▸"


class Footer:
    def __init__(
        self,
        parent: tk.Misc,
        chip_font: tkfont.Font,
        preview_font: tkfont.Font,
        on_tag_toggle: Callable[[str, bool], None],
        on_preview_toggle: Callable[[], None],
    ) -> None:
        self.frame = ttk.Frame(parent)
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

    def preview_text(self) -> str:
        return self.preview.get("1.0", "end-1c")
