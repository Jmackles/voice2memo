"""Model names come from what exists: the providers' own lists, faster-whisper's, and the startup check."""

from __future__ import annotations

import io
import logging
import time
import urllib.error

import pytest

from voice2memos import cleanup

from .conftest import FAKE_CFG

SECRET = "sk-test-0123456789abcdefSECRET"


# --- asking the providers, against a fake HTTP layer ---------------------------------


@pytest.fixture
def http(monkeypatch, no_model_servers):
    """The real available_models over a fake get_json; records each request."""
    monkeypatch.setattr(cleanup, "available_models", REAL_AVAILABLE)
    calls = []
    answers = {}

    def get_json(url, headers, timeout, local=False):
        calls.append((url, headers, local))
        answer = answers[url.split("?")[0]]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(cleanup, "get_json", get_json)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return calls, answers


REAL_AVAILABLE = cleanup.available_models
OLLAMA_TAGS = "http://127.0.0.1:11434/api/tags"


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid", code, "nope", {}, io.BytesIO(b"server said " + SECRET.encode()))


def test_ollama_lists_what_is_pulled_without_a_proxy(http, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    calls, answers = http
    answers[OLLAMA_TAGS] = {"models": [{"name": "zeta:latest"}, {"name": "gemma4:e2b"}]}
    assert cleanup.available_models("ollama") == (["gemma4:e2b", "zeta:latest"], "")
    assert calls == [(OLLAMA_TAGS, {}, True)]


def test_ollama_with_nothing_pulled_or_not_running_says_which(http, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    _, answers = http
    answers[OLLAMA_TAGS] = {"models": []}
    assert cleanup.available_models("ollama")[1].startswith("Ollama has no models pulled yet")
    answers[OLLAMA_TAGS] = urllib.error.URLError("refused")
    assert cleanup.available_models("ollama") == ([], "Ollama isn't running. Start it, then reopen Settings.")


@pytest.mark.parametrize(
    "host, url",
    [("", "http://127.0.0.1:11434"), ("0.0.0.0", "http://127.0.0.1:11434"), ("box:9999", "http://box:9999"),
     ("https://ollama.example", "https://ollama.example:443")],
)
def test_ollama_url_follows_ollama_host(monkeypatch, host, url):
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert cleanup.ollama_url() == url


def test_openai_lists_newest_first_with_the_key_as_a_bearer(http):
    calls, answers = http
    answers[cleanup.OPENAI_MODELS_URL] = {"data": [{"id": "older", "created": 1}, {"id": "newer", "created": 2}]}
    assert cleanup.available_models("openai", SECRET) == (["newer", "older"], "")
    assert calls[0][1] == {"Authorization": f"Bearer {SECRET}"} and not calls[0][2]


def test_anthropic_lists_in_its_own_order_with_the_version_header(http):
    calls, answers = http
    answers[cleanup.ANTHROPIC_MODELS_URL.split("?")[0]] = {"data": [{"id": "first"}, {"id": "second"}]}
    assert cleanup.available_models("anthropic", SECRET) == (["first", "second"], "")
    url, headers, _ = calls[0]
    assert url.endswith("?limit=1000")
    assert headers == {"x-api-key": SECRET, "anthropic-version": cleanup.ANTHROPIC_VERSION}


def test_a_blank_key_falls_back_to_the_environment(http, monkeypatch):
    calls, answers = http
    answers[cleanup.OPENAI_MODELS_URL] = {"data": []}
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    cleanup.available_models("openai", "  ")
    assert calls[0][1] == {"Authorization": "Bearer from-env"}


def test_no_key_at_all_asks_nobody(http):
    calls, _ = http
    names, note = cleanup.available_models("anthropic", "")
    assert names == [] and "ANTHROPIC_API_KEY" in note and calls == []


@pytest.mark.parametrize("code, note", [(401, "OpenAI didn't accept the API key."), (500, "OpenAI answered HTTP 500")])
def test_errors_name_the_problem_never_the_key_or_the_reply(http, caplog, code, note):
    _, answers = http
    answers[cleanup.OPENAI_MODELS_URL] = http_error(code)
    with caplog.at_level(logging.DEBUG):
        names, said = cleanup.available_models("openai", SECRET)
    assert names == [] and said.startswith(note)
    assert "SECRET" not in said and "SECRET" not in caplog.text


def test_an_unreachable_api_says_so(http):
    _, answers = http
    answers[cleanup.OPENAI_MODELS_URL] = OSError("no network")
    assert cleanup.available_models("openai", SECRET) == ([], "Couldn't reach OpenAI.")


@pytest.mark.parametrize("a, b, same", [("gemma4", "gemma4:latest", True), ("gemma4:e2b", "gemma4:e2b", True),
                                        ("gemma4:e2b", "gemma4", False)])
def test_same_model_reads_a_bare_name_as_latest(a, b, same):
    assert cleanup.same_model(a, b) is same


# --- Settings offers what exists ---------------------------------------------------


def wait_for_lists(dialog, seconds: float = 3) -> None:
    deadline = time.monotonic() + seconds
    while (dialog.listing or not dialog.lists_ready.empty()) and time.monotonic() < deadline:
        dialog.window.update()
        time.sleep(0.01)
    dialog.window.update()
    assert not dialog.listing


@pytest.fixture
def dialog(app):
    app.open_settings()
    wait_for_lists(app.settings_dialog)
    return app.settings_dialog


def values(dialog, field) -> list[str]:
    return list(dialog.widgets[field].cget("values"))


def test_the_model_lists_are_what_ollama_and_faster_whisper_have(dialog):
    assert values(dialog, ("cleanup", "model")) == ["gemma4:e2b", "llama3.2:latest"]
    assert values(dialog, ("entities", "model")) == ["gemma4:e2b", "llama3.2:latest"]
    assert values(dialog, ("whisper", "model_size")) == ["tiny", "small", "large-v3", "fake"]
    assert dialog.model_notes["cleanup"].cget("text") == "2 models pulled in Ollama."
    assert dialog.model_notes["whisper"].cget("text").startswith("Downloaded: small.")


def test_each_source_is_asked_once(dialog, no_model_servers):
    assert sorted(source for source, _ in no_model_servers.asked) == ["ollama", "whisper"]


def test_a_model_that_is_not_pulled_is_flagged(app_factory, no_model_servers):
    app = app_factory("[cleanup]\nmodel = gemma9:huge\n")
    app.open_settings()
    wait_for_lists(app.settings_dialog)
    note = app.settings_dialog.model_notes["cleanup"]
    assert note.cget("text") == "gemma9:huge isn't pulled in Ollama; pick one from the list."
    app.settings_dialog.vars[("cleanup", "model")].set("gemma4:e2b")
    app.settings_dialog.widgets[("cleanup", "model")].event_generate("<<ComboboxSelected>>")
    assert note.cget("text") == "2 models pulled in Ollama."


def test_switching_provider_lists_that_providers_models_with_the_typed_key(dialog, no_model_servers):
    no_model_servers.lists["anthropic"] = (["model-b", "model-a"], "")
    dialog.vars[("keys", "anthropic")].set(SECRET)
    dialog.vars[("cleanup", "provider")].set("anthropic")
    dialog.widgets[("cleanup", "provider")].event_generate("<<ComboboxSelected>>")
    wait_for_lists(dialog)
    assert values(dialog, ("cleanup", "model")) == ["model-b", "model-a"]
    assert ("anthropic", SECRET) in no_model_servers.asked
    assert dialog.model_notes["cleanup"].cget("text") == "gemma4:e2b isn't offered by Anthropic; pick one from the list."
    assert values(dialog, ("entities", "model")) == ["gemma4:e2b", "llama3.2:latest"]  # still on Ollama


def test_without_a_key_the_note_says_where_to_put_one(dialog):
    dialog.vars[("cleanup", "provider")].set("openai")
    dialog.widgets[("cleanup", "provider")].event_generate("<<ComboboxSelected>>")
    wait_for_lists(dialog)
    assert values(dialog, ("cleanup", "model")) == []
    assert "OPENAI_API_KEY" in dialog.model_notes["cleanup"].cget("text")


def test_a_key_typed_afterwards_asks_again(dialog, no_model_servers):
    dialog.vars[("cleanup", "provider")].set("openai")
    dialog.widgets[("cleanup", "provider")].event_generate("<<ComboboxSelected>>")
    wait_for_lists(dialog)
    no_model_servers.lists["openai"] = (["model-x"], "")
    dialog.vars[("keys", "openai")].set(SECRET)
    dialog._key_changed("openai")
    wait_for_lists(dialog)
    assert values(dialog, ("cleanup", "model")) == ["model-x"]


def test_a_list_that_blows_up_says_so_instead_of_hanging(app, no_model_servers, monkeypatch, caplog):
    def broken():
        raise RuntimeError("import exploded")

    monkeypatch.setattr("voice2memos.transcribe.whisper_models", broken)
    with caplog.at_level(logging.ERROR):
        app.open_settings()
        wait_for_lists(app.settings_dialog)
    assert app.settings_dialog.model_notes["whisper"].cget("text") == "Couldn't list the models, see log"


def test_closing_before_the_lists_arrive_is_fine(app):
    app.open_settings()
    app.settings_dialog.cancel()
    app.root.update()


# --- startup ---------------------------------------------------------------------------


def test_startup_is_quiet_when_ollama_has_the_models(app):
    app.check_models()
    assert app.status_label.cget("text") == "Ready"


def test_startup_says_when_ollama_lacks_a_configured_model(app_factory):
    app = app_factory("[entities]\nmodel = gemma9:huge\n")
    app.check_models()
    assert app.status_label.cget("text") == "Ollama has no gemma9:huge; pick an installed model in Settings > Models"


def test_startup_says_when_ollama_is_not_running(app, no_model_servers):
    no_model_servers.lists["ollama"] = ([], "Ollama isn't running. Start it, then reopen Settings.")
    app.check_models()
    assert app.status_label.cget("text") == "Ollama isn't running; cleanup will show the raw transcript"
    assert app.status_label.cget("foreground") is not None


def test_startup_asks_ollama_nothing_when_no_task_uses_it(app_factory, no_model_servers):
    app = app_factory(FAKE_CFG)
    app.check_models()
    assert no_model_servers.asked == []
