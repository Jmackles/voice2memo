"""The stage calls against the fake provider: rewrite per segment, rejection, entities, tags, failures."""

from __future__ import annotations

import logging

import pytest
from hypothesis import given
from hypothesis import strategies as st

from voice2memos import cleanup, config
from voice2memos.cleanup import FakeProvider, Provider, make_provider
from voice2memos.errors import CleanupError, EntityError
from voice2memos.transcribe import load_segments

POOL = config.DEFAULTS["tags"]["pool"].split()


class Canned(Provider):
    """Answers with whatever the test says, and remembers what it was asked."""

    name = "canned"

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, system, user, *, json_mode=False, timeout=60):
        self.calls.append((system, user, json_mode))
        return self.answers.pop(0)


@pytest.fixture
def sample(fixtures_dir):
    return fixtures_dir / "sample"


@pytest.fixture
def segments(sample):
    return load_segments(sample / "segments.json")


def test_rewrite_fills_each_segment_from_the_fixture(sample, segments, caplog):
    with caplog.at_level(logging.DEBUG):
        warnings = cleanup.rewrite(FakeProvider(sample), segments, "rewrite it", 60, "sample")
    assert warnings == []
    expected = load_segments(sample / "cleaned.json")
    assert [s.cleaned for s in segments] == [s.cleaned for s in expected]
    assert "sample: cleanup (fake) start" in caplog.messages
    assert any(m.startswith("sample: cleanup segment 0 prompt:\nrewrite it") for m in caplog.messages)


def test_rewrite_calls_once_per_segment_with_the_prompt_as_system(segments):
    provider = Canned("One.", "Two.")
    cleanup.rewrite(provider, segments, "the prompt", 60, "x")
    assert [(system, json_mode) for system, _, json_mode in provider.calls] == [("the prompt", False)] * 2
    assert [user for _, user, _ in provider.calls] == [s.text for s in segments]
    assert [s.cleaned for s in segments] == ["One.", "Two."]


def test_rewrite_rejects_empty_and_bloated_output_but_keeps_going(segments, caplog):
    long = "x" * 200
    with caplog.at_level(logging.DEBUG):
        warnings = cleanup.rewrite(Canned("", long), segments, "p", 60, "x")
    assert warnings == ["Cleanup output rejected"]
    assert [s.cleaned for s in segments] == ["", ""]
    assert [s.shown for s in segments] == [s.text for s in segments]
    rejected = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(rejected) == 2
    assert long not in rejected[1].getMessage()  # the texts themselves only at DEBUG
    assert long in caplog.text


def test_rewrite_strips_leaked_think_tags(segments, caplog):
    with caplog.at_level(logging.WARNING):
        cleanup.rewrite(Canned("<think>hmm\nhmm</think>\nOne.", "Two."), segments, "p", 60, "x")
    assert segments[0].cleaned == "One."
    assert "think tags leaked" in caplog.text


def test_rewrite_failure_names_the_segment_in_the_detail_only(sample, segments):
    with pytest.raises(CleanupError) as info:
        cleanup.rewrite(FakeProvider(sample, fail={"cleanup"}), segments, "p", 60, "x")
    assert info.value.message == "Cleanup failed, showing original"
    assert info.value.detail.startswith("segment 0: RuntimeError")
    assert [s.cleaned for s in segments] == ["", ""]


def test_a_delay_past_the_timeout_times_out(segments):
    with pytest.raises(CleanupError) as info:
        cleanup.rewrite(FakeProvider(delay=0.05), segments[:1], "p", 0.01, "x")
    assert "TimeoutError" in info.value.detail


def test_the_fake_echoes_without_a_fixture(segments):
    cleanup.rewrite(FakeProvider(), segments, "p", 60, "x")
    assert [s.cleaned for s in segments] == [s.text.strip() for s in segments]


def test_entities_come_sanitized_from_the_fixture(sample):
    assert cleanup.extract_entities(FakeProvider(sample), "text", 30, "x") == ["database", "Bookinnit", "fields", "tables"]


def test_entities_use_the_prompt_file_in_json_mode():
    provider = Canned('{"items": ["a"]}')
    cleanup.extract_entities(provider, "the memo", 30, "x")
    [(system, user, json_mode)] = provider.calls
    assert system == cleanup.load_prompt("entities") and user == "the memo" and json_mode


@pytest.mark.parametrize("answer", ["not json", "{}", '{"items": "no"}', "[]", "[1, 2]"])
def test_bad_json_gives_no_entities(answer, caplog):
    with caplog.at_level(logging.WARNING):
        assert cleanup.extract_entities(Canned(answer), "t", 30, "x") == []


def test_a_bare_array_still_counts():
    assert cleanup.extract_entities(Canned('["one", "two"]'), "t", 30, "x") == ["one", "two"]


def test_entity_failure_is_an_entity_error(sample):
    with pytest.raises(EntityError, match="Entities failed"):
        cleanup.extract_entities(FakeProvider(sample, fail={"entities"}), "t", 30, "x")


def test_sanitize_terms_strips_link_breakers_and_dedupes_by_case():
    assert cleanup.sanitize_terms(["Python", "python", "a/b", "[[x]]", " ", "#", "c|d^e", "  two   words "]) == [
        "Python",
        "ab",
        "x",
        "cde",
        "two words",
    ]


@given(st.lists(st.text()))
def test_sanitize_terms_never_raises_or_leaves_a_link_breaker(items):
    for term in cleanup.sanitize_terms(items):
        assert term and not cleanup.LINK_BREAKERS.search(term)
        assert term == term.strip()


def test_tags_come_from_the_pool_in_its_spelling(sample):
    assert cleanup.choose_tags(FakeProvider(sample), "t", POOL, 30, "x") == ["idea", "bookinnit"]


def test_tags_outside_the_pool_are_dropped_and_two_at_most(caplog):
    provider = Canned('{"items": ["IDEA", "database", "todo", "idea", "rant"]}')
    with caplog.at_level(logging.DEBUG):
        tags = cleanup.choose_tags(provider, "t", POOL, 30, "x")
    assert tags == ["idea", "todo"]
    assert "tag outside the pool dropped: 'database'" in caplog.text
    assert "idea, question, rant, homework, bookinnit, todo" in provider.calls[0][0]


def test_tag_failure_is_an_entity_error(sample):
    with pytest.raises(EntityError, match="Tags failed"):
        cleanup.choose_tags(FakeProvider(sample, fail={"tags"}), "t", POOL, 30, "x")


def test_make_provider_follows_the_config(fixtures_dir):
    cfg = config.new_parser()
    assert isinstance(make_provider(cfg, "cleanup", config.APP_DIR), cleanup.OllamaProvider)
    cfg["entities"]["provider"] = "anthropic"
    assert isinstance(make_provider(cfg, "entities", config.APP_DIR), cleanup.AnthropicProvider)
    cfg["cleanup"]["provider"] = "fake"
    cfg["fake"]["fixture"] = "tests/fixtures/sample"
    cfg["fake"]["fail"] = "tags"
    cfg["fake"]["delay"] = "0.5"
    fake = make_provider(cfg, "cleanup", config.APP_DIR)
    assert isinstance(fake, FakeProvider)
    assert (fake.fixture, fake.fail, fake.delay) == (fixtures_dir / "sample", {"tags"}, 0.5)


# --- the real Ollama provider, over a fake client ------------------------------------

from voice2memos.cleanup import OllamaProvider  # noqa: E402
from voice2memos.errors import ProviderError  # noqa: E402

from .conftest import FakeOllamaClient  # noqa: E402


def ollama_error(status: int, text: str = "nope"):
    import ollama

    return ollama.ResponseError(text, status)


def test_ollama_chats_with_thinking_off_and_the_settings_from_config(segments):
    provider = make_provider(config.new_parser(), "cleanup", config.APP_DIR)
    assert isinstance(provider, OllamaProvider)
    FakeOllamaClient.answers = ["One.", "Two."]
    cleanup.rewrite(provider, segments, "the prompt", 60, "x")
    assert [s.cleaned for s in segments] == ["One.", "Two."]
    first = FakeOllamaClient.calls[0]
    assert first["model"] == "gemma4:e2b"
    assert first["messages"] == [{"role": "system", "content": "the prompt"}, {"role": "user", "content": segments[0].text}]
    assert first["options"] == {"temperature": 0.0} and first["format"] is None
    assert first["keep_alive"] == "30m" and first["think"] is False
    assert FakeOllamaClient.made[0] == ("fake", 60)


def test_ollama_leaves_think_alone_for_a_model_without_it():
    FakeOllamaClient.capabilities = ["completion"]
    provider = OllamaProvider("plain:latest", 0.0)
    provider.complete("s", "u")
    provider.complete("s", "u")
    assert all(call["think"] is None for call in FakeOllamaClient.calls)
    assert provider.thinks is False  # asked once, remembered


def test_ollama_json_mode_asks_for_json():
    OllamaProvider("m", 0.0).complete("s", "u", json_mode=True)
    assert FakeOllamaClient.calls[0]["format"] == "json"


def test_a_missing_model_names_it(segments):
    FakeOllamaClient.raise_on_chat = ollama_error(404, "model not found")
    with pytest.raises(CleanupError) as info:
        cleanup.rewrite(OllamaProvider("gone:latest", 0.0), segments, "p", 60, "x")
    assert info.value.message == "Model gone:latest not found in Ollama"
    assert "HTTP 404" in info.value.detail


def test_ollama_down_is_said_plainly(segments):
    FakeOllamaClient.raise_on_show = ConnectionError("refused")
    with pytest.raises(CleanupError) as info:
        cleanup.rewrite(OllamaProvider("m", 0.0), segments, "p", 60, "x")
    assert info.value.message == "Ollama not running"


def test_a_timeout_is_said_as_one(segments):
    class ReadTimeout(Exception):
        pass

    FakeOllamaClient.raise_on_chat = ReadTimeout("slow")
    with pytest.raises(CleanupError) as info:
        cleanup.rewrite(OllamaProvider("m", 0.0), segments, "p", 7, "x")
    assert info.value.message == "Cleanup timed out, showing original" and "after 7s" in info.value.detail


def test_entities_keep_their_own_message_and_the_providers_detail():
    FakeOllamaClient.raise_on_show = ConnectionError("refused")
    with pytest.raises(EntityError) as info:
        cleanup.extract_entities(OllamaProvider("m", 0.0), "t", 30, "x")
    assert info.value.message == "Entities failed" and "ConnectionError" in info.value.detail


def test_the_api_providers_say_they_are_not_built(segments):
    provider = make_provider(config.new_parser(), "entities", config.APP_DIR)
    assert isinstance(provider, cleanup.OllamaProvider)
    cfg = config.new_parser()
    cfg["cleanup"]["provider"] = "anthropic"
    with pytest.raises(CleanupError) as info:
        cleanup.rewrite(make_provider(cfg, "cleanup", config.APP_DIR), segments, "p", 60, "x")
    assert info.value.message == "Anthropic provider isn't built yet"
