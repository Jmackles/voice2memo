"""Capture: global hotkey, mic stream, WAV writer, socket trigger.

Rule above all: the WAV is on disk before anything downstream runs.

The sounddevice callback runs on PortAudio's thread. It may append to the
buffer and feed the level meter's RMS; it never touches a tk widget.
"""
