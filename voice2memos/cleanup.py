"""Cleanup: provider base class, Ollama/OpenAI/Anthropic/Fake, rewrite + entity + tag calls.

Each provider exposes one ``complete`` method. The Fake provider answers from a
fixture by matching on which prompt it was handed, and has modes that raise or
sleep past the timeout for error-state tests. ``provider = fake`` in the .cfg
runs the whole app end to end with no models loaded. The real providers are
the grey box step 11 fills in.

The three stage calls, ``rewrite``, ``extract_entities`` and ``choose_tags``,
hold everything that isn't talking to a model: prompt loading, the per-segment
loop, output rejection, think-tag stripping, JSON parsing, term sanitizing. A
stage raises its own error class on a hard failure and the caller degrades.

Ollama ``format: json`` guarantees valid JSON, not an array: prompts ask for an
object like ``{"items": [...]}`` and the caller reads ``items``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from voice2memos.errors import CleanupError, EntityError
from voice2memos.logsetup import timed
from voice2memos.transcribe import Segment, load_segments

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
CLEANED_FILE = "cleaned.json"
ENTITIES_FILE = "entities.json"
TAGS_FILE = "tags.json"

MAX_GROWTH = 1.5  # cleaned output longer than this times the input is rejected
MAX_TAGS = 2
THINK_TAGS = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
LINK_BREAKERS = re.compile(r"[/\\|#^\[\]]")


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


# --- providers --------------------------------------------------------------


class Provider:
    """One call: system prompt, user text, answer. Subclasses do the talking."""

    name = "provider"

    def complete(self, system: str, user: str, *, json_mode: bool = False, timeout: float = 60) -> str:
        raise NotImplementedError


class FakeProvider(Provider):
    """Answers from a fixture folder, picking the file by which prompt it was handed.

    Rewrite prompts get the matching segment's ``cleaned`` text from
    cleaned.json, or the input back unchanged when nothing matches or there is
    no fixture. The entity and tag prompts get entities.json and tags.json as
    the model would return them. ``fail`` names stages that raise; ``delay``
    sleeps, and past ``timeout`` it raises the way a real client would.
    """

    name = "fake"

    def __init__(self, fixture: Path | None = None, fail: set[str] | None = None, delay: float = 0) -> None:
        self.fixture = fixture
        self.fail = fail or set()
        self.delay = delay

    @staticmethod
    def kind(system: str) -> str:
        if "tags" in system:
            return "tags"
        if "subjects" in system:
            return "entities"
        return "cleanup"

    def complete(self, system: str, user: str, *, json_mode: bool = False, timeout: float = 60) -> str:
        kind = self.kind(system)
        if self.delay:
            time.sleep(min(self.delay, timeout))
            if self.delay > timeout:
                raise TimeoutError(f"fake {kind} call took longer than {timeout:g}s")
        if kind in self.fail:
            raise RuntimeError(f"fake provider set to fail {kind}")
        if kind == "cleanup":
            for segment in self._segments():
                if segment.text.strip() == user.strip():
                    return segment.cleaned
            return user
        if kind == "entities":
            return json.dumps({"items": self._json(ENTITIES_FILE, [])})
        return json.dumps({"items": self._json(TAGS_FILE, {}).get("proposed", [])})

    def _segments(self) -> list[Segment]:
        if self.fixture is None or not (self.fixture / CLEANED_FILE).exists():
            return []
        return load_segments(self.fixture / CLEANED_FILE)

    def _json(self, name: str, default):
        if self.fixture is None or not (self.fixture / name).exists():
            return default
        return json.loads((self.fixture / name).read_text(encoding="utf-8"))


class _NotBuiltYet(Provider):
    """Placeholder for a real provider. Holds its config; step 11 makes it talk."""

    def __init__(self, model: str, temperature: float, **options) -> None:
        self.model = model
        self.temperature = temperature
        self.options = options

    def complete(self, system: str, user: str, *, json_mode: bool = False, timeout: float = 60) -> str:
        raise RuntimeError(f"{self.name} provider arrives in build step 11")


class OllamaProvider(_NotBuiltYet):
    name = "ollama"


class OpenAIProvider(_NotBuiltYet):
    name = "openai"


class AnthropicProvider(_NotBuiltYet):
    name = "anthropic"


def make_provider(cfg, section: str, repo_dir: Path) -> Provider:
    """The provider ``[section]`` asks for. ``repo_dir`` is where a relative fake fixture lives.

    API keys are not read here: a provider that needs one reads the config or
    the environment when it connects, so no key sits on an object longer than it must.
    """
    settings = cfg[section]
    name = settings["provider"].strip().lower()
    if name == "fake":
        fake = cfg["fake"]
        fixture = fake["fixture"].strip()
        return FakeProvider(
            fixture=repo_dir / fixture if fixture else None,
            fail=set(fake["fail"].split()),
            delay=float(fake["delay"] or 0),
        )
    classes = {"ollama": OllamaProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider}
    if name not in classes:
        raise CleanupError("Unknown provider", f"[{section}] provider = {name}")
    options = {"keep_alive": settings["keep_alive"]} if section == "cleanup" else {}
    return classes[name](settings["model"], float(settings["temperature"]), **options)


# --- stages -----------------------------------------------------------------


def rewrite(provider: Provider, segments: list[Segment], prompt: str, timeout: float, stem: str) -> list[str]:
    """Fill each segment's ``cleaned`` text, one call per segment so timestamps survive.

    Returns status-line warnings for segments whose output was rejected; those
    keep ``cleaned`` empty, so the note shows the raw text there. A call that
    fails raises CleanupError, leaving the rest raw too.
    """
    warnings = []
    with timed(log, f"cleanup ({provider.name})", stem):
        for segment in segments:
            log.debug("%s: cleanup segment %d prompt:\n%s\n---\n%s", stem, segment.index, prompt, segment.text)
            try:
                answer = provider.complete(prompt, segment.text, timeout=timeout)
            except Exception as e:
                raise CleanupError("Cleanup failed, showing original", f"segment {segment.index}: {type(e).__name__}: {e}") from e
            log.debug("%s: cleanup segment %d answer:\n%s", stem, segment.index, answer)
            cleaned = strip_think_tags(answer, stem)
            if not acceptable(cleaned, segment.text):
                log.warning("%s: cleanup output rejected for segment %d, %d chars from %d", stem, segment.index, len(cleaned), len(segment.text))
                log.debug("%s: rejected output was:\n%s\n--- for input:\n%s", stem, cleaned, segment.text)
                warnings.append("Cleanup output rejected")
                continue
            segment.cleaned = cleaned
    return sorted(set(warnings))


def strip_think_tags(text: str, stem: str = "") -> str:
    stripped, count = THINK_TAGS.subn("", text)
    if count:
        log.warning("%s: think tags leaked into cleanup output, stripped", stem)
    return stripped.strip()


def acceptable(cleaned: str, original: str) -> bool:
    return bool(cleaned) and len(cleaned) <= MAX_GROWTH * max(len(original), 20)


def extract_entities(provider: Provider, text: str, timeout: float, stem: str) -> list[str]:
    """The terms the note is about, sanitized and deduped. Bad JSON gives an empty list."""
    with timed(log, f"entities ({provider.name})", stem):
        answer = _ask_json(provider, load_prompt("entities"), text, timeout, stem, EntityError, "Entities failed")
    return sanitize_terms(parse_items(answer, stem))


def choose_tags(provider: Provider, text: str, pool: list[str], timeout: float, stem: str) -> list[str]:
    """Up to MAX_TAGS tags, only ever from the pool, in the pool's own spelling."""
    prompt = load_prompt("tags").replace("{pool}", ", ".join(pool))
    with timed(log, f"tags ({provider.name})", stem):
        answer = _ask_json(provider, prompt, text, timeout, stem, EntityError, "Tags failed")
    by_fold = {tag.casefold(): tag for tag in pool}
    chosen = []
    for item in parse_items(answer, stem):
        tag = by_fold.get(item.strip().casefold())
        if tag is None:
            log.debug("%s: tag outside the pool dropped: %r", stem, item)
        elif tag not in chosen:
            chosen.append(tag)
    return chosen[:MAX_TAGS]


def _ask_json(provider, system, user, timeout, stem, error, message) -> str:
    log.debug("%s: prompt:\n%s\n---\n%s", stem, system, user)
    try:
        answer = provider.complete(system, user, json_mode=True, timeout=timeout)
    except Exception as e:
        raise error(message, f"{type(e).__name__}: {e}") from e
    log.debug("%s: answer:\n%s", stem, answer)
    return answer


def parse_items(answer: str, stem: str = "") -> list[str]:
    """The strings in ``{"items": [...]}``, or in a bare array. Anything else is an empty list."""
    try:
        data = json.loads(answer)
    except ValueError:
        log.warning("%s: model answer was not JSON", stem)
        log.debug("%s: answer was:\n%s", stem, answer)
        return []
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        log.warning("%s: model answer held no list", stem)
        return []
    return [item for item in data if isinstance(item, str)]


def sanitize_terms(items: list[str]) -> list[str]:
    """Strip what breaks a wikilink, drop empties, dedupe case-insensitively keeping the first spelling."""
    seen = set()
    terms = []
    for item in items:
        term = " ".join(LINK_BREAKERS.sub("", item).split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms
