from pathlib import Path

import pytest

import install
from install import Check, InstallError, run_stage


@pytest.fixture(autouse=True)
def plain_output(monkeypatch):
    monkeypatch.setattr(install, "_color", False)


# --- the check, install, check-again flow ---


def test_everything_found_is_good_to_go(capsys):
    assert run_stage("Thing", lambda: [Check("a", True, "1.0")], install=True)
    out = capsys.readouterr().out
    assert "[ ok ] a" in out
    assert "good to go :)" in out


def test_missing_is_installed_then_checked_again(capsys):
    state = {"installed": False}
    calls = []

    def fix():
        calls.append("fix")
        state["installed"] = True

    def check():
        return [Check("a", state["installed"], fix=fix), Check("b", state["installed"], fix=fix)]

    assert run_stage("Thing", check, install=True)
    assert calls == ["fix"], "a fix shared by two checks runs once"
    out = capsys.readouterr().out
    assert out.index("[miss] a") < out.index("Checking again") < out.index("[ ok ] a")
    assert out.rstrip().endswith("good to go :)")


def test_check_only_never_installs(capsys):
    calls = []
    assert not run_stage("Thing", lambda: [Check("a", False, fix=lambda: calls.append(1))], install=False)
    assert calls == []
    assert "Run python install.py" in capsys.readouterr().out


def test_manual_fix_stops_before_installing_anything(capsys):
    calls = []
    checks = [Check("Python", False, hint="install a newer Python"), Check("b", False, fix=lambda: calls.append(1))]
    assert not run_stage("Thing", lambda: checks, install=True)
    assert calls == []
    assert "install a newer Python" in capsys.readouterr().out


def test_failed_install_is_reported(capsys):
    def fix():
        raise InstallError("the download broke")

    assert not run_stage("Thing", lambda: [Check("a", False, fix=fix)], install=True)
    assert "the download broke" in capsys.readouterr().out


def test_install_that_does_not_help_is_reported(capsys):
    assert not run_stage("Thing", lambda: [Check("a", False, fix=lambda: None)], install=True)
    assert "Still missing after installing: a" in capsys.readouterr().out


def test_info_and_warn_lines_never_fail(capsys):
    checks = [install.note("GPU", "none"), install.note("folder", "synced", kind="warn"), Check("a", True)]
    assert run_stage("Thing", lambda: checks, install=True)
    out = capsys.readouterr().out
    assert "[info] GPU" in out
    assert "[warn] folder" in out


# --- colors ---


def test_paint_is_plain_without_color():
    assert install.paint("hi", install.CYAN) == "hi"


def test_good_to_go_is_cyan_with_a_yellow_smiley(monkeypatch, capsys):
    monkeypatch.setattr(install, "_color", True)
    install.good_to_go()
    out = capsys.readouterr().out
    assert "\033[36mgood to go\033[0m" in out
    assert "\033[33;1m:)\033[0m" in out


@pytest.mark.parametrize("env, expected", [({"NO_COLOR": "1", "FORCE_COLOR": "1"}, False), ({"FORCE_COLOR": "1"}, True)])
def test_color_env_overrides(monkeypatch, env, expected):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert install.enable_color() is expected


# --- parsing ---


def test_read_requirements(tmp_path):
    base = tmp_path / "a.txt"
    base.write_text(
        "# comment\n-r other.txt\nfaster-whisper\nsounddevice>=0.4  # why\n\nPyYAML[extra]==6.0\n", encoding="utf-8"
    )
    cuda = tmp_path / "b.txt"
    cuda.write_text("nvidia-cudnn-cu12>=9,<10\nfaster_whisper\n", encoding="utf-8")
    assert install.read_requirements([base, cuda]) == ["faster-whisper", "sounddevice", "PyYAML", "nvidia-cudnn-cu12"]


def test_real_requirement_files_parse():
    names = install.read_requirements([install.REQUIREMENTS, install.REQUIREMENTS_CUDA])
    assert "faster-whisper" in names
    assert "nvidia-cudnn-cu12" in names


def test_normalize():
    assert install.normalize("Faster_Whisper") == "faster-whisper"
    assert install.normalize("typing.extensions") == "typing-extensions"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("| NVIDIA-SMI 535.98     Driver Version: 535.98     CUDA Version: 12.2     |", "12.2"),
        ("| NVIDIA-SMI 617.14     KMD Version: 617.14     CUDA UMD Version: 13.4     |", "13.4"),
        ("no GPU in here", ""),
    ],
)
def test_parse_cuda_version(text, expected):
    assert install.parse_cuda_version(text) == expected


@pytest.mark.parametrize("name, cuda, ok", [("RTX", "11.8", False), ("RTX", "12.0", True), ("RTX", "13.4", True), ("RTX", "", False), ("", "12.4", False)])
def test_gpu_needs_cuda_12(name, cuda, ok):
    assert install.Gpu(name, cuda).cuda_ok is ok


@pytest.mark.parametrize(
    "host, expected",
    [
        (None, "http://127.0.0.1:11434"),
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("0.0.0.0:5000", "http://127.0.0.1:5000"),
        ("http://box.lan", "http://box.lan:80"),
        ("https://box.lan:8443", "https://box.lan:8443"),
    ],
)
def test_ollama_url_follows_ollama_host(monkeypatch, host, expected):
    if host is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", host)
    assert install.ollama_url() == expected


def test_has_model():
    assert install.has_model(["gemma4:e2b"], "gemma4:e2b")
    assert install.has_model(["llama3:latest"], "llama3")
    assert not install.has_model(["gemma4:e4b"], "gemma4:e2b")


@pytest.mark.parametrize(
    "path, service",
    [
        ("C:/Users/j/OneDrive - Some College/Documents/repo", "OneDrive"),
        ("/home/j/Dropbox/repo", "Dropbox"),
        ("/home/j/code/repo", ""),
    ],
)
def test_sync_service(path, service):
    assert install.sync_service(Path(path)) == service
