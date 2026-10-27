"""Root window, banner, panes, worker wiring.

Scaffold only: opens an empty titled window so ``python -m voice2memos`` runs.
The window shell with every control is build-order step 2.
"""

from __future__ import annotations

import tkinter as tk

from voice2memos import __version__


def main() -> None:
    root = tk.Tk()
    root.title(f"voice2memos {__version__}")
    root.geometry("960x600")
    root.minsize(640, 400)
    root.mainloop()
