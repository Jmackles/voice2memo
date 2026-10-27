"""Platform glue: Windows DPI awareness, macOS permission popups, Wayland hotkey setup.

On Windows call ``SetProcessDpiAwareness(1)`` through ctypes before creating
the root window, or everything renders blurry on scaled displays.
"""
