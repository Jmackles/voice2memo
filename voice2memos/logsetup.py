"""Logging config and excepthooks.

Rotating file handler at ``logs/voice2memos.log`` beside the config, level from
the .cfg (INFO by default), plus a console handler when run from a terminal so
bench.py and the app log the same way. ``sys.excepthook`` and
``threading.excepthook`` both route to the logger and the status line.

Every stage logs start, end, elapsed, and the memo stem; the stem is the
correlation id. DEBUG captures prompts and responses verbatim. WARNING is for
every fallback the user saw in the status line.
"""
