"""Platform glue: Windows DPI awareness, macOS permission popups, Wayland hotkey setup.

On Windows call ``SetProcessDpiAwareness(1)`` through ctypes before creating
the root window, or everything renders blurry on scaled displays.
"""

from __future__ import annotations

import ctypes
import sys


def set_dpi_awareness() -> None:
    """Ask Windows for real pixels instead of a stretched bitmap. A no-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass  # older than Windows 8.1, or already set; the window is just blurrier
