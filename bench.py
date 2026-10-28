"""Workbench: run any single stage against a saved fixture and diff the result.

    python bench.py transcribe tests/fixtures/<name>/memo.wav
    python bench.py clean tests/fixtures/<name>/segments.json
    python bench.py clean --all
    python bench.py entities tests/fixtures/<name>/cleaned.json
    python bench.py note tests/fixtures/<name>

Backends come from the config, so with the defaults a stage reports that its
real backend isn't built yet; ``--fake`` runs the stage through the fake
backend reading the input's own folder, which exercises everything but the
model. Every stage prints elapsed seconds and, when the expected file sits
beside the input, a diff against it. Logging goes to the console at
``--log-level``, DEBUG to see prompts and answers verbatim.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from voice2memos import cleanup, config, transcribe
from voice2memos.errors import VoiceMemoError
from voice2memos.logsetup import FORMAT

REPO = config.APP_DIR
FIXTURES = REPO / "tests" / "fixtures"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.py", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=config.CONFIG_PATH, help="a .cfg other than the real one")
    parser.add_argument("--fake", action="store_true", help="fake backends reading the input's folder")
    parser.add_argument("--log-level", default="WARNING", help="DEBUG shows prompts and answers")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("transcribe", help="WAV in, segments JSON out")
    p.add_argument("wav", type=Path)

    p = sub.add_parser("clean", help="segments JSON in, cleaned JSON out")
    p.add_argument("segments", nargs="?", type=Path)
    p.add_argument("--all", action="store_true", help="run every fixture and show diffs")

    p = sub.add_parser("entities", help="cleaned JSON in, entities + tags out")
    p.add_argument("cleaned", type=Path)

    p = sub.add_parser("note", help="fixture folder in, rendered note out")
    p.add_argument("fixture", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format=FORMAT)
    cfg, warnings = config.load(args.config)
    for warning in warnings:
        print(f"bench: {warning}", file=sys.stderr)

    if args.stage == "transcribe":
        return run_transcribe(cfg, args.wav, args.fake)
    if args.stage == "clean":
        if args.all:
            inputs = sorted(FIXTURES.glob(f"*/{transcribe.SEGMENTS_FILE}"))
            if not inputs:
                print(f"bench: no fixtures under {FIXTURES}", file=sys.stderr)
                return 2
            return max(run_clean(cfg, path, args.fake) for path in inputs)
        if args.segments is None:
            parser.error("clean needs a segments.json or --all")
        return run_clean(cfg, args.segments, args.fake)
    if args.stage == "entities":
        return run_entities(cfg, args.cleaned, args.fake)
    print(f"bench: stage '{args.stage}' is not implemented yet", file=sys.stderr)
    return 2


def run_transcribe(cfg, wav: Path, fake: bool) -> int:
    print(f"== transcribe {wav}")
    transcriber = transcribe.FakeWhisper() if fake else transcribe.make_transcriber(cfg, REPO)
    with elapsed():
        try:
            segments = transcriber.transcribe(wav)
        except VoiceMemoError as e:
            return failed(e)
    text = dump([asdict(s) for s in segments])
    print(text, end="")
    return diff(text, wav.with_name(transcribe.SEGMENTS_FILE))


def run_clean(cfg, segments_path: Path, fake: bool) -> int:
    print(f"== clean {segments_path}")
    provider = cleanup.FakeProvider(segments_path.parent) if fake else cleanup.make_provider(cfg, "cleanup", REPO)
    prompt = cfg["cleanup"]["prompt"].strip() or cleanup.load_prompt("rewrite")
    try:
        segments = transcribe.load_segments(segments_path)
    except VoiceMemoError as e:
        return failed(e)
    with elapsed():
        try:
            warnings = cleanup.rewrite(provider, segments, prompt, float(cfg["cleanup"]["timeout"]), segments_path.parent.name)
        except VoiceMemoError as e:
            return failed(e)
    for warning in warnings:
        print(f"bench: {warning}", file=sys.stderr)
    for segment in segments:
        print(f"[{segment.index}] {segment.text.strip()}")
        print(f"  > {segment.shown.strip()}")
    text = dump([asdict(s) for s in segments])
    return diff(text, segments_path.with_name(cleanup.CLEANED_FILE))


def run_entities(cfg, cleaned_path: Path, fake: bool) -> int:
    print(f"== entities {cleaned_path}")
    provider = cleanup.FakeProvider(cleaned_path.parent) if fake else cleanup.make_provider(cfg, "entities", REPO)
    try:
        segments = transcribe.load_segments(cleaned_path)
    except VoiceMemoError as e:
        return failed(e)
    text = transcribe.joined_text(segments, cleaned=True)
    timeout = float(cfg["entities"]["timeout"])
    stem = cleaned_path.parent.name
    pool = cfg["tags"]["pool"].split()
    status = 0
    with elapsed():
        try:
            entities = cleanup.extract_entities(provider, text, timeout, stem)
        except VoiceMemoError as e:
            status, entities = failed(e), []
        try:
            tags = cleanup.choose_tags(provider, text, pool, timeout, stem)
        except VoiceMemoError as e:
            status, tags = failed(e), []
    entities_text = dump(entities)
    tags_text = dump({"proposed": tags, "rejected": []})
    print(entities_text + tags_text, end="")
    return max(status, diff(entities_text, cleaned_path.with_name(cleanup.ENTITIES_FILE)), diff(tags_text, cleaned_path.with_name(cleanup.TAGS_FILE)))


# --- helpers ----------------------------------------------------------------


class elapsed:
    def __enter__(self):
        self.started = time.perf_counter()

    def __exit__(self, *_):
        print(f"-- {time.perf_counter() - self.started:.2f}s")


def failed(error: VoiceMemoError) -> int:
    print(f"bench: {error.message}: {error.detail}", file=sys.stderr)
    return 1


def dump(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def diff(actual: str, expected_path: Path) -> int:
    """Print a unified diff against the expected file when it exists. Nonzero when they differ."""
    if not expected_path.exists():
        print(f"-- no {expected_path.name} to compare against")
        return 0
    expected = expected_path.read_text(encoding="utf-8")
    if json.loads(actual) == json.loads(expected):
        print(f"-- matches {expected_path.name}")
        return 0
    lines = difflib.unified_diff(expected.splitlines(), actual.splitlines(), str(expected_path.name), "actual", lineterm="")
    print("\n".join(lines))
    return 1


if __name__ == "__main__":
    sys.exit(main())
