"""Workbench: run any single stage against a saved fixture and diff the result.

    python bench.py transcribe tests/fixtures/<name>/memo.wav
    python bench.py clean tests/fixtures/<name>/segments.json
    python bench.py clean --all
    python bench.py entities tests/fixtures/<name>/cleaned.json
    python bench.py note tests/fixtures/<name>

Every stage prints elapsed seconds. Skeleton only; stages are wired in as they
are built.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("transcribe", help="WAV in, segments JSON out")
    p.add_argument("wav")

    p = sub.add_parser("clean", help="segments JSON in, cleaned JSON out")
    p.add_argument("segments", nargs="?")
    p.add_argument("--all", action="store_true", help="run every fixture and show diffs")

    p = sub.add_parser("entities", help="cleaned JSON in, entities + tags out")
    p.add_argument("cleaned")

    p = sub.add_parser("note", help="fixture folder in, rendered note out")
    p.add_argument("fixture")

    args = parser.parse_args(argv)
    print(f"bench: stage '{args.stage}' is not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
