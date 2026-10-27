"""Config: ``.cfg`` load, defaults dict, apply.

Sections mirror ``voice2memos.example.cfg``: capture, whisper, cleanup,
entities, tags, vault, keys, logging. The defaults dict is the single source of
truth for shipped values; the installer writes the initial .cfg from it.

Notes from the design doc:
- construct ``configparser.ConfigParser(interpolation=None)``, or any ``%`` in
  the rewrite prompt raises; it also lowercases keys, so never rely on key case
- missing file gives defaults, partial file gets missing keys filled, the
  multi-line prompt must round-trip
"""
