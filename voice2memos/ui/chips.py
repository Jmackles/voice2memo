"""Tag chip row: the model's proposed tags, each a chip a click strikes through.

A struck chip stays out of the note's frontmatter; clicking it again puts it
back. Silence accepts. The row has no idea where tags come from or go: it
shows what it's given and reports each click to ``on_toggle(tag, struck)``.
"""

from __future__ import annotations

from collections.abc import Callable
from tkinter import font as tkfont
from tkinter import ttk

HINT_COLOR = "#666666"
STRUCK_COLOR = "#999999"
KEPT_COLOR = "#1a1a1a"
NO_TAGS = "none yet"
CLICK_HINT = "click one to leave it out"


class ChipRow:
    def __init__(self, parent: ttk.Frame, font: tkfont.Font, on_toggle: Callable[[str, bool], None]) -> None:
        self.on_toggle = on_toggle
        self.font = font
        self.struck_font = font.copy()
        self.struck_font.configure(overstrike=True)
        self.frame = ttk.Frame(parent)
        self.title = ttk.Label(self.frame, text="Tags", style="PaneTitle.TLabel")
        self.title.pack(side="left")
        self.chips: dict[str, ttk.Label] = {}
        self.rejected: set[str] = set()
        self.hint = ttk.Label(self.frame, text=NO_TAGS, foreground=HINT_COLOR)
        self.hint.pack(side="left", padx=(8, 0))

    def show(self, tags: list[str], rejected: list[str]) -> None:
        """Replace the chips with these tags; the rejected ones start struck."""
        self.clear()
        self.rejected = {t for t in rejected if t in tags}
        for tag in tags:
            chip = ttk.Label(self.frame, text=tag, padding=(8, 1), relief="solid", borderwidth=1, cursor="hand2")
            chip.bind("<Button-1>", lambda _e, t=tag: self.toggle(t))
            chip.pack(side="left", padx=(8, 0))
            self.chips[tag] = chip
            self._paint(tag)
        self.hint.configure(text=CLICK_HINT if tags else NO_TAGS)
        self.hint.pack_forget()
        self.hint.pack(side="left", padx=(8, 0))  # after the chips

    def clear(self) -> None:
        for chip in self.chips.values():
            chip.destroy()
        self.chips.clear()
        self.rejected.clear()
        self.hint.configure(text=NO_TAGS)

    def toggle(self, tag: str) -> None:
        if tag not in self.chips:
            return
        struck = tag not in self.rejected
        if struck:
            self.rejected.add(tag)
        else:
            self.rejected.discard(tag)
        self._paint(tag)
        self.on_toggle(tag, struck)

    def is_struck(self, tag: str) -> bool:
        return tag in self.rejected

    def _paint(self, tag: str) -> None:
        struck = tag in self.rejected
        self.chips[tag].configure(
            font=self.struck_font if struck else self.font, foreground=STRUCK_COLOR if struck else KEPT_COLOR
        )
